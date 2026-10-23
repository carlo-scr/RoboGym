# Setup cell, inlined into every notebook by notebooks/build.py.
# Locally: finds the repo. On a Colab runtime (also VS Code connected to Colab), where the code is not
# on the VM: clones ROBOGYM_REPO (default: the public repo), pip-installs it, sets MUJOCO_GL=egl and
# keeps data/ + runs/ on Drive when it is mounted.
import os, pathlib, subprocess, sys

DEFAULT_REPO = "https://github.com/carlo-scr/RoboGym.git"
ROBOGYM_REPO = os.environ.get("ROBOGYM_REPO") or DEFAULT_REPO
USE_DRIVE = os.environ.get("ROBOGYM_DRIVE", "1") == "1"
IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
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
        subprocess.run(["git", "-C", str(CODE), "pull", "-q"])
    if not (CODE / "robogym").exists():
        subprocess.run(["git", "clone", "-q", ROBOGYM_REPO, str(CODE)], check=True)
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
