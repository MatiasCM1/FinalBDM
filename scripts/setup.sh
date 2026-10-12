#!/usr/bin/env bash
# Crea .venv e instala el proyecto en Linux/macOS.
#
# Uso:
#   bash scripts/setup.sh                    # PyTorch con CUDA (índice por defecto: cu128)
#   TORCH_INDEX=cu126 bash scripts/setup.sh  # otra versión de CUDA
#   TORCH_INDEX=cpu   bash scripts/setup.sh  # sin GPU
#   TORCH_INDEX=pypi  bash scripts/setup.sh  # torch tal cual viene en PyPI
#   PYTHON=python3.11 bash scripts/setup.sh  # forzar intérprete
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
TORCH_INDEX="${TORCH_INDEX:-cu128}"
VENV=".venv"

# 1) Intérprete: preferir 3.11 (soportado: 3.10 a 3.13).
if [[ -z "${PYTHON:-}" ]]; then
  for c in python3.11 python3.12 python3.10 python3.13 python3; do
    if command -v "$c" >/dev/null 2>&1; then PYTHON="$c"; break; fi
  done
fi
[[ -n "${PYTHON:-}" ]] || { echo "ERROR: no se encontró Python 3.10-3.13." >&2; exit 1; }
"$PYTHON" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' \
  || { echo "ERROR: se requiere Python 3.10-3.13 (encontrado: $("$PYTHON" -V))." >&2; exit 1; }
echo ">> Python: $("$PYTHON" -V)"

# 2) Entorno virtual.
if [[ ! -d "$VENV" ]]; then
  echo ">> Creando $VENV"
  "$PYTHON" -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip wheel

# 3) PyTorch primero (para controlar CUDA/CPU), después el resto.
if [[ "$TORCH_INDEX" == "pypi" ]]; then
  echo ">> Instalando torch desde PyPI"
  "$PY" -m pip install torch
else
  echo ">> Instalando torch desde https://download.pytorch.org/whl/$TORCH_INDEX"
  "$PY" -m pip install torch --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
fi

echo ">> Instalando dependencias"
"$PY" -m pip install -r requirements-dev.txt
"$PY" -m pip install -e . --no-deps

# 4) Congelar el entorno exacto (auditoría / reproducibilidad).
"$PY" -m pip freeze --exclude-editable > requirements.lock

# 5) Extras de comodidad.
[[ -f .env ]] || { cp .env.example .env; echo ">> Creado .env (completar TABPFN_TOKEN)"; }
if [[ -d .git ]]; then "$VENV/bin/pre-commit" install || true; fi

# 6) Verificación.
"$PY" scripts/check_env.py || true

echo
echo ">> Listo. Activar con:  source $VENV/bin/activate"
