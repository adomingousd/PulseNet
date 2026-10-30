# Python Virtual Environment (venv)

## What it is

A virtual environment is a private folder of Python packages for this project only. It lives in `.venv/` at the repo root and is never committed to git.

Why we use one:

- Everyone runs the same package versions (see `docs/environment.md`).
- Installing something for PulseNet can't break other Python projects on your computer, and the reverse.
- It matches the Pis, which run Python 3.13.

## First-time setup (macOS)

Run these from the repo root (the `PulseNet` folder).

```bash
brew install python@3.13
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `requirements.txt` isn't in the repo yet, install the pinned packages directly:

```bash
pip install torch==2.14.1 "flwr[simulation]==1.42.0" numpy pandas mlflow pytest
```

## Every time you open a new terminal

```bash
cd path/to/PulseNet
source .venv/bin/activate
```

You'll see `(.venv)` at the start of your prompt. If it's not there, the venv isn't active, and `python` or `flwr` won't be found.

To leave the venv: `deactivate`

## Check that it's working

```bash
which python          # should end in PulseNet/.venv/bin/python
python --version      # should say Python 3.13.x
python -c "import flwr, torch; print(flwr.__version__, torch.__version__)"   # 1.42.0 2.14.1
```

## Restart the venv from scratch

Do this if packages are broken, versions are wrong, or you moved or renamed the project folder (a venv stops working when its folder moves).

```bash
deactivate                       # ignore the error if no venv was active
cd path/to/PulseNet
rm -rf .venv
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the checks above. Deleting `.venv` only removes installed packages; your code and data are untouched.

## Common problems

| Problem | Cause | Fix |
|---|---|---|
| `python: command not found` | The venv isn't active | `source .venv/bin/activate` from the repo root |
| `python3 --version` shows 3.15 | You're using your Mac's system Python | Activate the venv |
| `flwr: command not found` | Venv not active, or packages not installed in this venv | Activate it, then `pip install -r requirements.txt` |
| `Refusing to create a venv … PATH separator :` | A folder in the path has a colon in its name | Rename the folder, then restart the venv |
| Flower says `Internal server error` (log shows `no such table: fab`) | A space somewhere in the project path | Rename the folder to remove spaces, restart the venv, then `rm -rf ~/.flwr/local-superlink` |
| Things broke after moving or renaming the project folder | The venv still points to the old location | Restart the venv |

## Rules of thumb

- Never commit `.venv/`; it's in `.gitignore`.
- Keep the project path free of spaces and colons.
- If you install a new package the project needs, add it to `requirements.txt` so everyone gets it.
