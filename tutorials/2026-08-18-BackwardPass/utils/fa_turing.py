"""flash-attention-turing for a Colab GPU box: fetch a prebuilt wheel, or build one.

FlashAttention-2 for sm75 (Turing), from https://github.com/ssiu/flash-attention-turing.
fp16 only, head_dim 64/128, causal, varlen and GQA native. No KV-cache path.

The problem this file exists to solve
-------------------------------------
The extension is a PyTorch C++/CUDA extension, so it is welded to the Python ABI
tag, the torch version, torch's C++ ABI flag and the CUDA toolchain of the box it
was built on. Building it takes ~12 minutes on Colab's 2 vCPUs, which is a long
time to spend before a tutorial reaches its first line -- hence the prebuilt
wheels in the model repo.

But Colab's runtime image moves, and moving it changes the ABI tag. It went
python 3.12 -> 3.13 on 2026-08-20 with nothing else changing, which was enough
to invalidate every prebuilt wheel. So the contract is a MAP keyed by ABI tag,
one entry per wheel, and this module picks the entry matching the box it is on:

    fa_turing/flash_attn_turing.json
        { "builds": { "cp312": {...}, "cp313": {...} }, ... }

`ensure()` downloads the matching wheel and puts its .so on sys.path (seconds).
If nothing matches -- Colab has moved again and we have not caught up -- it
BUILDS one on the spot rather than leaving the reader stuck at cell 3. That
costs ~12 minutes and says so loudly.

Usage
-----
From a notebook (the normal path):

    from fa_turing import ensure
    ensure()                      # -> import flash_attn_turing now works

From a shell, to catch the repo up after a runtime move (needs a write token):

    python fa_turing.py publish   # build + upload wheel and sidecar entry
    python fa_turing.py build     # build only, leave the wheel in --out-dir
    python fa_turing.py ensure    # what the notebook does, for testing

Everything is ASCII so the file survives the Colab CLI.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

REPO = "ChrisMcCormick/qwen-arithmetic-t4"
PREFIX = "fa_turing"
SIDECAR = "flash_attn_turing.json"
SOURCE = "https://github.com/ssiu/flash-attention-turing"
ARCH = "sm75"
TORCH_CUDA_ARCH_LIST = "7.5"
DEFAULT_CACHE = Path.home() / ".cache" / "qwen-arithmetic" / "data" / PREFIX

NOTES = ("fp16 only; head_dim 64/128; causal, varlen and GQA native; NO KV-cache "
         "path, so decode keeps the CUTLASS FMHA. One wheel per Colab runtime ABI: "
         "read builds[cp<major><minor>] for the box you are on. When Colab's image "
         "moves and no entry matches, utils/fa_turing.py builds one on the box and "
         "`python fa_turing.py publish` ADDS an entry -- older entries are kept, so "
         "a reader who selected an older runtime in Colab's 'Change runtime type' "
         "still gets a prebuilt wheel.")


# ----------------------------------------------------------------------------
# What box are we on
# ----------------------------------------------------------------------------
def abi_tag():
    """cp313 on python 3.13. The one field Colab's 2026-08-20 image moved."""
    return f"cp{sys.version_info[0]}{sys.version_info[1]}"


def so_name():
    return (f"flash_attn_turing.cpython-{sys.version_info[0]}{sys.version_info[1]}"
            f"-x86_64-linux-gnu.so")


def runtime():
    """The compatibility identity of this box. Imports torch, so call it late."""
    import torch
    return {
        "python": f"{sys.version_info[0]}.{sys.version_info[1]}",
        "python_full": sys.version.split()[0],
        "abi_tag": abi_tag(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cxx11abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "arch": ARCH,
        "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG", ""),
    }


def _compatible(built_for, have):
    """A wheel loads iff these three agree. CUDA is deliberately not in the
    tuple: the extension links against torch's CUDA runtime, and a minor
    toolkit bump does not break the ABI."""
    return (built_for["abi_tag"], built_for["torch"], built_for["cxx11abi"]) == \
           (have["abi_tag"], have["torch"], have["cxx11abi"])


# ----------------------------------------------------------------------------
# The sidecar (the compatibility contract)
# ----------------------------------------------------------------------------
def load_sidecar(repo=REPO):
    """The published contract, or None if the repo has no sidecar yet."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    try:
        p = hf_hub_download(repo, f"{PREFIX}/{SIDECAR}")
    except EntryNotFoundError:
        return None
    side = json.loads(Path(p).read_text())
    if "builds" not in side:                       # pre-2026-08-21 single-slot form
        side = {"builds": {side["built_for"]["abi_tag"]: side}, "source": SOURCE}
    return side


# ----------------------------------------------------------------------------
# ensure -- the notebook's entry point
# ----------------------------------------------------------------------------
def ensure(repo=REPO, cache_dir=None, allow_build=True, verbose=True):
    """Make `import flash_attn_turing` work on this box.

    Returns a dict: {"how": "cached"|"prebuilt"|"built", "abi_tag": ..., ...}.
    Puts `cache_dir` on sys.path either way -- the .so is extracted from the
    wheel rather than pip-installed, so nothing in site-packages is touched.
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    have = runtime()
    log = print if verbose else (lambda *a, **k: None)

    def _use():
        if str(cache_dir) not in sys.path:
            sys.path.insert(0, str(cache_dir))

    if (cache_dir / so_name()).exists():
        _use()
        log(f"flash-attention-turing: already in {cache_dir} ({have['abi_tag']})")
        return {"how": "cached", **have}

    side = load_sidecar(repo)
    entry = (side or {}).get("builds", {}).get(have["abi_tag"])

    if entry is not None and _compatible(entry["built_for"], have):
        from huggingface_hub import hf_hub_download
        log(f"flash-attention-turing: fetching the prebuilt {have['abi_tag']} wheel ...")
        whl = hf_hub_download(repo, f"{PREFIX}/{entry['wheel']}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(whl) as z:
            z.extract(so_name(), str(cache_dir))
        _use()
        log(f"  ok: {entry['wheel']}")
        return {"how": "prebuilt", "wheel": entry["wheel"], **have}

    # Nothing matches. Say precisely why, then build rather than block.
    available = sorted((side or {}).get("builds", {}))
    if entry is None:
        why = (f"no prebuilt wheel for {have['abi_tag']} (the repo has {available}); "
               f"Colab's image has moved")
    else:
        why = (f"the prebuilt {have['abi_tag']} wheel was built against torch "
               f"{entry['built_for']['torch']} / cxx11abi "
               f"{entry['built_for']['cxx11abi']}, this box has {have['torch']} / "
               f"{have['cxx11abi']}")
    if not allow_build:
        raise RuntimeError(why + ". Pass allow_build=True to build one here.")

    log("=" * 78)
    log(f"flash-attention-turing: {why}.")
    log("Building it on this box instead -- about 12 minutes on Colab's 2 vCPUs.")
    log(f"This box is: {have['colab_release_tag'] or 'not a Colab runtime'}")
    log(f"python {have['python_full']} | torch {have['torch']} | cuda {have['cuda']}")
    log("(To spare the next reader this wait: `python fa_turing.py publish`.)")
    log("=" * 78)

    whl, _meta = build(out_dir=cache_dir.parent / "wheels", verbose=verbose)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(whl) as z:
        z.extract(so_name(), str(cache_dir))
    _use()
    log(f"flash-attention-turing: built and installed ({whl.name})")
    return {"how": "built", "wheel": whl.name, "wheel_path": str(whl), **have}


# ----------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------
def _heartbeat(stop, t0, log_path):
    """The compile prints nothing for many minutes. Distinguish alive from dead."""
    while not stop.wait(30):
        n = log_path.stat().st_size if log_path.exists() else 0
        last = ""
        if n:
            with open(log_path, "rb") as f:
                f.seek(max(0, n - 4000))
                lines = [l for l in f.read().decode("utf-8", "replace").splitlines()
                         if l.strip()]
                last = lines[-1][:100] if lines else ""
        print(f"  [{time.time() - t0:5.0f}s] building ... {n / 1024:.0f} KB | {last}",
              flush=True)


def build(out_dir=None, src_dir=None, verbose=True):
    """Clone upstream and build the wheel for THIS runtime. -> (wheel_path, meta)."""
    import torch
    # setup.py hardcodes -arch=sm_75, so the wheel this produces runs on Turing
    # and nothing else. Catching that here saves 12 minutes and a kernel that
    # would fail at the first launch with a far less obvious error.
    assert torch.cuda.is_available(), "building flash-attention-turing needs a GPU"
    cc = torch.cuda.get_device_capability()
    assert cc == (7, 5), (
        f"flash-attention-turing is sm75 (Turing) only; this box is sm_{cc[0]}{cc[1]} "
        f"({torch.cuda.get_device_name(0)}). Use a T4 -- on newer cards the stock "
        f"flash-attn wheels apply instead.")
    out_dir = Path(out_dir) if out_dir else Path.home() / ".cache" / "fa_turing_build"
    src = Path(src_dir) if src_dir else out_dir / "src"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "build.log"
    log = print if verbose else (lambda *a, **k: None)

    env = dict(os.environ)
    env.update(MAX_JOBS="2",             # 2 vCPUs; upstream notes more OOMs on Colab
               CC="gcc", CXX="g++",
               TORCH_CUDA_ARCH_LIST=TORCH_CUDA_ARCH_LIST,
               PIP_PROGRESS_BAR="off")

    def run(cmd, **kw):
        log(f"$ {' '.join(str(c) for c in cmd)}")
        return subprocess.run([str(c) for c in cmd], env=env, check=True, **kw)

    if not (src / "setup.py").exists():
        run(["git", "clone", "--recursive", "--depth", "1", SOURCE, src])
    commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    cutlass = subprocess.run(["git", "-C", str(src / "csrc" / "cutlass"), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    assert (src / "csrc" / "cutlass" / "include").is_dir(), "cutlass submodule missing"
    log(f"flash-attention-turing {commit}\ncutlass {cutlass or '(none)'}")

    run([sys.executable, "-m", "pip", "install", "-q", "ninja", "setuptools", "wheel"])

    t0 = time.time()
    stop = threading.Event()
    if verbose:
        threading.Thread(target=_heartbeat, args=(stop, t0, log_path), daemon=True).start()
    try:
        with open(log_path, "w") as lf:
            # --no-build-isolation: link against THIS runtime's torch, not a fresh
            # one pip would download into an isolated env.
            run([sys.executable, "-m", "pip", "wheel", ".", "--no-build-isolation",
                 "--no-deps", "-w", out_dir], cwd=str(src), stdout=lf,
                stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        stop.set()
        tail = log_path.read_text(errors="replace").splitlines()[-40:]
        print("---- last 40 build log lines ----")
        print("\n".join("  " + l[:200] for l in tail))
        raise
    finally:
        stop.set()
    build_s = time.time() - t0

    wheels = sorted(out_dir.glob("flash_attn_turing-*.whl"))
    assert wheels, f"no wheel produced in {out_dir}"
    whl = wheels[-1]
    # The filename must carry THIS interpreter's ABI tag; if it does not, the
    # build linked against something other than the running python and the
    # wheel would be mislabelled the moment it was published.
    assert f"-{abi_tag()}-" in whl.name, f"{whl.name} is not a {abi_tag()} wheel"

    meta = dict(wheel=whl.name,
                sha256=hashlib.sha256(whl.read_bytes()).hexdigest(),
                bytes=whl.stat().st_size,
                build_minutes=round(build_s / 60, 1),
                commit=commit, cutlass=cutlass, gpu=torch.cuda.get_device_name(0),
                **runtime())
    (out_dir / "wheel_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"built {whl.name} ({whl.stat().st_size / 2**20:.1f} MB) in "
        f"{build_s / 60:.1f} min -> {out_dir}")
    return whl, meta


# ----------------------------------------------------------------------------
# smoke -- the smallest real call, plus an fp32 spot-check
# ----------------------------------------------------------------------------
def smoke(verbose=True):
    """Import the extension and check one causal GQA segment against fp32."""
    import torch
    import flash_attn_turing
    log = print if verbose else (lambda *a, **k: None)
    log(f"imported {flash_attn_turing.__file__}")

    T, HQ, HKV, DH = 128, 14, 2, 64
    scale = DH ** -0.5
    dev = torch.device("cuda", 0)
    torch.manual_seed(0)
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    q = (torch.randn(T, HQ, DH, device=dev) * 0.5).half()
    k = (torch.randn(T, HKV, DH, device=dev) * 0.5).half()
    v = (torch.randn(T, HKV, DH, device=dev) * 0.5).half()
    out, lse = flash_attn_turing.varlen_fwd(q, k, v, cu, cu, T, T, scale, True)
    dq, dk, dv = flash_attn_turing.varlen_bwd(q, k, v, out, lse, torch.randn_like(out),
                                              cu, cu, T, T, scale, True)
    assert dk.shape[1] == HKV and dv.shape[1] == HKV, "dK/dV are not GQA-native"
    assert torch.isfinite(out).all() and torch.isfinite(dq).all(), "non-finite output"

    # fp32 reference for the forward: query head h reads kv head h // group.
    group = HQ // HKV
    qs, ks, vs = (t.float().transpose(0, 1) for t in
                  (q, k[:, torch.arange(HQ, device=dev) // group],
                   v[:, torch.arange(HQ, device=dev) // group]))
    s = (qs @ ks.transpose(-1, -2)) * scale
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), 1),
                      float("-inf"))
    ref = (torch.softmax(s, -1) @ vs).transpose(0, 1)
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(),
                                                ref.flatten(), dim=0).item()
    assert cos >= 0.999, f"forward disagrees with the fp32 reference (cos {cos:.6f})"
    log(f"smoke ok: fwd cos {cos:.6f} vs fp32, dK/dV GQA-native at {HKV} heads")
    return cos


# ----------------------------------------------------------------------------
# publish -- add this box's wheel to the repo, keeping every existing entry
# ----------------------------------------------------------------------------
def publish(whl, meta, repo=REPO, colab_runtime=None, dry_run=False,
            release_tag=None):
    """Upload the wheel and ADD its sidecar entry. Existing entries survive."""
    from huggingface_hub import HfApi
    whl = Path(whl)

    m = re.search(r"-(cp\d+)-\1-", whl.name)
    assert m, f"cannot read an ABI tag out of {whl.name}"
    tag = m.group(1)
    # meta["python"] may be "3.13" or a full "3.13.15", depending on which
    # version of the build script produced it. Compare on major.minor only --
    # the patch level is not part of the ABI.
    py_mm = ".".join(str(meta["python"]).split(".")[:2])
    assert tag == f"cp{py_mm.replace('.', '')}", \
        f"{whl.name} disagrees with wheel_meta.json python {meta['python']}"
    assert whl.stat().st_size == meta["bytes"], "wheel size != wheel_meta.json bytes"

    rt = (release_tag or os.environ.get("COLAB_RELEASE_TAG")
          or meta.get("colab_release_tag", ""))
    if colab_runtime is None:
        # release-colab-external-images_20260820-060050_RC00 -> 2026.08
        d = re.search(r"_(\d{4})(\d{2})\d{2}-", rt)
        colab_runtime = f"{d.group(1)}.{d.group(2)}" if d else "unknown"

    entry = {
        "wheel": whl.name, "sha256": meta["sha256"], "bytes": meta["bytes"],
        "commit": meta["commit"], "cutlass": meta["cutlass"],
        "built_for": {"python": py_mm, "abi_tag": tag, "torch": meta["torch"],
                      "cuda": meta["cuda"], "cxx11abi": meta["cxx11abi"], "arch": ARCH,
                      "colab_runtime": colab_runtime, "colab_release_tag": rt},
        "built_on": "Google Colab, in the runtime it targets",
    }

    side = load_sidecar(repo) or {"builds": {}}
    builds = dict(side.get("builds", {}))
    if tag in builds:
        print(f"replacing the existing builds[{tag!r}] entry")
    builds[tag] = entry
    sidecar = {"source": SOURCE, "notes": NOTES, "builds": builds,
               # Mirror of the newest entry, so copies of the pre-2026-08-21
               # single-slot guard still resolve on the current runtime.
               "wheel": entry["wheel"], "built_for": entry["built_for"]}

    side_path = whl.with_name(SIDECAR)
    side_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps(sidecar, indent=2))
    print(f"\nABI tags in the sidecar: {sorted(builds)}")
    if dry_run:
        print(f"--dry-run: wrote {side_path}, uploaded nothing")
        return sidecar

    api = HfApi()
    msg = f"fa_turing: sm75 FlashAttention-2 for Colab {colab_runtime} ({tag})"
    uploads = [(whl, f"{PREFIX}/{whl.name}"), (side_path, f"{PREFIX}/{SIDECAR}")]
    # Mirror this file next to the wheels so the notebook can fetch its own
    # fallback builder from the same repo it already downloads from.
    me = Path(__file__).resolve()
    if me.is_file():
        uploads.append((me, f"{PREFIX}/{me.name}"))
    for src_path, dst in uploads:
        api.upload_file(path_or_fileobj=str(src_path), path_in_repo=dst,
                        repo_id=repo, repo_type="model", commit_message=msg)
        print(f"  -> {repo}/{dst}")
    print("PUBLISH OK")
    return sidecar


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("ensure", "build", "publish", "status"))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--out-dir", default=None, help="where to put the built wheel")
    ap.add_argument("--cache-dir", default=None, help="where to extract the .so")
    ap.add_argument("--colab-runtime", default=None, help="label, e.g. 2026.08")
    ap.add_argument("--no-build", action="store_true",
                    help="ensure: fail instead of building when nothing matches")
    ap.add_argument("--dry-run", action="store_true", help="publish: upload nothing")
    ap.add_argument("--wheel", default=None,
                    help="publish: upload this already-built wheel instead of "
                         "building one here (it must already have been smoke-tested "
                         "on a box with its ABI)")
    ap.add_argument("--meta", default=None,
                    help="publish: the wheel_meta.json for --wheel "
                         "(default: wheel_meta.json beside it)")
    ap.add_argument("--release-tag", default=None,
                    help="publish: COLAB_RELEASE_TAG the wheel was built on")
    a = ap.parse_args(argv)

    if a.cmd == "status":
        have = runtime()
        print(json.dumps(have, indent=2))
        side = load_sidecar(a.repo)
        builds = (side or {}).get("builds", {})
        print(f"\nsidecar ABI tags: {sorted(builds)}")
        e = builds.get(have["abi_tag"])
        if e is None:
            print(f"NO WHEEL for {have['abi_tag']} -- `fa_turing.py publish` to add one")
        elif _compatible(e["built_for"], have):
            print(f"MATCH: {e['wheel']}")
        else:
            print(f"STALE: {e['wheel']} was built for {e['built_for']}")
        return 0

    if a.cmd == "ensure":
        info = ensure(a.repo, a.cache_dir, allow_build=not a.no_build)
        smoke()
        print(json.dumps(info, indent=2))
        return 0

    if a.wheel:
        # Publishing a wheel built somewhere else -- typically one built on a
        # Colab box and fetched back, so this host need not even be Linux.
        assert a.cmd == "publish", "--wheel only applies to publish"
        whl = Path(a.wheel)
        meta = json.loads(Path(a.meta or whl.with_name("wheel_meta.json")).read_text())
        print(f"publishing a pre-built wheel: {whl.name}")
        print("  NOT smoke-tested here -- it must have been validated on a box "
              "with its ABI.")
    else:
        whl, meta = build(out_dir=a.out_dir)
        # Never publish a wheel that has not been imported and checked.
        tmp = Path(whl).parent / "_smoke"
        tmp.mkdir(exist_ok=True)
        with zipfile.ZipFile(whl) as z:
            z.extract(so_name(), str(tmp))
        sys.path.insert(0, str(tmp))
        smoke()

    if a.cmd == "publish":
        publish(whl, meta, a.repo, a.colab_runtime, a.dry_run, a.release_tag)
    else:
        print(f"wheel: {whl}\nmeta:  {Path(whl).parent / 'wheel_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
