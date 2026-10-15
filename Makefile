ifeq ($(OS),Windows_NT)
    SHELL := powershell.exe
    .SHELLFLAGS := -NoProfile -Command
    SETUP_CMD := powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
    CLEAN_CMD := Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .pytest_cache, .ruff_cache, build, dist, src/*.egg-info
    VENV_PY := .venv/Scripts/python.exe
else
    SETUP_CMD := bash scripts/setup.sh
    CLEAN_CMD := rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
    VENV_PY := .venv/bin/python
endif

# Usa el Python del .venv si existe (no hace falta activarlo); si no, el del sistema.
PY ?= $(if $(wildcard $(VENV_PY)),$(VENV_PY),python)

.PHONY: setup check test lint fmt clean

setup:       ## Crea .venv e instala todo
	$(SETUP_CMD)

check:       ## Verifica entorno, GPU y token
	$(PY) scripts/check_env.py

test:        ## Tests rápidos (sin GPU ni descarga de pesos)
	$(PY) -m pytest -m "not slow and not gpu"

lint:
	$(PY) -m ruff check src tests scripts

fmt:
	$(PY) -m ruff format src tests scripts

clean:
	$(CLEAN_CMD)