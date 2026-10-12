# Atajos para Linux/macOS. En Windows (PowerShell) usar los comandos equivalentes del README.
PY ?= python

.PHONY: setup check test lint fmt clean

setup:       ## Crea .venv e instala todo
	bash scripts/setup.sh

check:       ## Verifica entorno, GPU y token
	$(PY) scripts/check_env.py

test:        ## Tests rápidos (sin GPU ni descarga de pesos)
	$(PY) -m pytest -m "not slow and not gpu"

lint:
	$(PY) -m ruff check src tests scripts

fmt:
	$(PY) -m ruff format src tests scripts

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
