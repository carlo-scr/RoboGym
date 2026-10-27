# Setup cell, inlined into every notebook by notebooks/build.py.
# Locally: finds the repo. On a Colab runtime (also VS Code connected to Colab), where the code is not
# on the VM: clones ROBOGYM_REPO, pip-installs it, sets MUJOCO_GL=egl and keeps data/ + runs/ on Drive
# when it is mounted. The repo is private: add a Colab secret named GITHUB_TOKEN (a GitHub fine-grained
# token with read access to the repo; key icon in the left bar, "notebook access" on) or set
# ROBOGYM_REPO to a token URL. ROBOGYM_BRANCH picks a branch (default: the repo's default branch).
import os, pathlib, subprocess, sys

DEFAULT_REPO = "https://github.com/carlo-scr/RoboGym.git"
ROBOGYM_REPO = os.environ.get("ROBOGYM_REPO") or DEFAULT_REPO
ROBOGYM_BRANCH = os.environ.get("ROBOGYM_BRANCH")
USE_DRIVE = os.environ.get("ROBOGYM_DRIVE", "1") == "1"
IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def _github_token():
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("ROBOGYM_GH_TOKEN")
    if not tok and IN_COLAB:
        try:
            from google.colab import userdata

            tok = userdata.get("GITHUB_TOKEN")
        except Exception:  # noqa: BLE001  (no secret, or access not granted to this notebook)
            tok = None
    return tok


def _clone(repo, dest, branch=None):
    """Clone (with a token for a private GitHub repo); the stored remote URL never contains the token."""
    url, tok = repo, _github_token()
    if tok and "@" not in repo and repo.startswith("https://github.com/"):
        url = repo.replace("https://github.com/", f"https://x-access-token:{tok}@github.com/")
    cmd = ["git", "clone", "-q"] + (["-b", branch] if branch else []) + [url, str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if r.returncode:
        err = r.stderr.replace(tok, "***") if tok else r.stderr
        hint = ("the repo is private and no GitHub token was found: add a Colab secret GITHUB_TOKEN (key icon in the left bar, "
                "grant notebook access), or set ROBOGYM_REPO='https://<token>@github.com/carlo-scr/RoboGym.git'")
        raise RuntimeError(f"git clone of {repo} failed ({err.strip()[:300]}). {hint if not tok else 'Check the token has read access to the repo.'}")
    if url != repo:
        subprocess.run(["git", "-C", str(dest), "remote", "set-url", "origin", repo], check=True)


if IN_COLAB:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    _icd = pathlib.Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    if subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0 and not _icd.exists():
        _icd.parent.mkdir(parents=True, exist_ok=True)
        _icd.write_text('{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}\n')
    CODE = pathlib.Path("/content/RoboGym")
    DRIVE = pathlib.Path("/content/drive/MyDrive")
    if (CODE / ".git").exists():
        tok = _github_token()
        pull_url = ROBOGYM_REPO.replace("https://github.com/", f"https://x-access-token:{tok}@github.com/") if tok and "@" not in ROBOGYM_REPO else ROBOGYM_REPO
        subprocess.run(["git", "-C", str(CODE), "pull", "-q", pull_url] + ([ROBOGYM_BRANCH] if ROBOGYM_BRANCH else []), env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if not (CODE / "robogym").exists():
        if CODE.exists():
            subprocess.run(["rm", "-rf", str(CODE)])  # a clone that failed half-way
        _clone(ROBOGYM_REPO, CODE, ROBOGYM_BRANCH)
    if USE_DRIVE and not DRIVE.exists():
        try:
            from google.colab import drive

            drive.mount("/content/drive")
        except Exception as e:  # noqa: BLE001
            print("Drive not mounted:", e)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{CODE}[gym]"], check=True)
    if DRIVE.exists():
        for d in ("data", "runs"):
            tgt = DRIVE / "robogym_runs" / d
            tgt.mkdir(parents=True, exist_ok=True)
            link = CODE / d
            if not link.is_symlink():
                subprocess.run(["rm", "-rf", str(link)])
                link.symlink_to(tgt)
    os.chdir(CODE)
ROOT = pathlib.Path.cwd()
while not (ROOT / "robogym").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
print("ROOT:", ROOT, "| colab:", IN_COLAB)
