"""Hook de pre-commit: corre pytest con el Python del .venv (activado o no)."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    git_dir = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"], text=True, cwd=ROOT
    ).strip()
    main_root = Path(git_dir).resolve().parent
except Exception:
    main_root = ROOT

CANDIDATES = [
    ROOT / ".venv" / "Scripts" / "python.exe",
    ROOT / ".venv" / "bin" / "python",
    main_root / ".venv" / "Scripts" / "python.exe",
    main_root / ".venv" / "bin" / "python",
]
PY = next((str(p) for p in CANDIDATES if p.exists()), sys.executable)

env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
sys.exit(subprocess.call([PY, "-m", "pytest", "-m", "not slow and not gpu"], cwd=ROOT, env=env))
