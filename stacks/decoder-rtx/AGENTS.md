A submitted run should read like the baseline it's competing with: one self-contained
script that someone can open and follow start to finish. The code is attached to the wandb
run automatically, so it *is* the submission -- settle these before launching a candidate.

Have a subagent compare your code to the repo's baseline and report on each item below.

**What the submission should be**

(1) **Self-contained.** Everything the run needs is hardcoded in the script -- no command
    line arguments, no environment variables -- so the attached code reproduces the number.
    `WANDB_API_KEY` is the one exception.
(2) **Fully contained in the logged scripts**, `train_stack.py` and `utils.py`. If it ran,
    it's in there.
(3) **One code path.** Any feature you made configurable during development gets rewritten
    down to the single path that runs in the final submission -- the reader should see what
    happened, not what could have.

**How the code should read**

(4) **Direct.** Inline the helper functions you added at their call sites, and convert any
    classes you added with methods into functions or inline code (plain data structures are
    fine). Read values straight from the global `cfg` rather than passing them in as
    function arguments. Every hop a reader has to make costs them the thread.
(5) **Self-documenting through names.** Give the variables you introduced descriptive names
    -- prefer clarity over brevity, and use your judgment about how long is useful.
(6) **Comment-light.** Keep the original comments: restore any your edits removed. Of your
    own, keep terse block markers and terse function descriptions; let the code carry the
    rest.
(7) **Baked-in hyperparameters.** Weight initialization, scheduling and optimizer values you
    added belong at the point of use, not in `StackConfig`.
(8) **Aligned.** Left-align the values in multi-line object definitions (the code has
    examples), and restore any alignment your edits disturbed.

**The tools this speedrun sets aside**

(9) Plain PyTorch is the interesting constraint here, so submissions leave out **Triton** and
    **FP8 / quantization**. Both are very cool and both make a big difference -- they just
    lead to some pretty nasty code, and a readable single file is what this speedrun is for.
