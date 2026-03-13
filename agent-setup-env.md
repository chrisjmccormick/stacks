Grab micromamba:

```bash
# 1. For x64 - Grab micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
# 1. For ARM - Grab micromamba for ARM64 (aarch64)
#curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xvj bin/micromamba

# 2. Initialize it
./bin/micromamba shell init --shell bash --root-prefix ~/micromamba

source ~/.bashrc
```

Create the speedrun environment and activate it:

```bash
micromamba create -n stacks python=3.12 -y

micromamba activate stacks
```

Add my API keys to the environment

```bash
source ~/env.sh
```

Configure GitHub and clone my repo:

```bash
git config --global user.name $GH_YOUR_NAME
git config --global user.email $GH_EMAIL

cd ~/
git clone https://${GITHUB_TOKEN}@github.com/chrisjmccormick/stacks.git && cd stacks
git remote set-url origin https://${GITHUB_TOKEN}@github.com/chrisjmccormick/stacks.git
```

Install requirements:

```bash
pip install -r requirements-x86.txt
```

Finally, remind me to activate the environment in my terminal:

```bash
source ~/.bashrc
micromamba activate stacks
source ~/env.sh
```

and change directories:

```bash
cd ~/stacks/
```