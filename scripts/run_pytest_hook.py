"""Hook de pre-commit: corre pytest con el Python del .venv (activado o no)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"]
PY = next((str(p) for p in CANDIDATES if p.exists()), sys.executable)

sys.exit(subprocess.call([PY, "-m", "pytest", "-m", "not slow and not gpu"], cwd=ROOT))
