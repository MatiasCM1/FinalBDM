"""Rutas del proyecto, independientes del directorio desde el que se ejecute."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA_RAW = ROOT / "data" / "raw"
DATA_SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
