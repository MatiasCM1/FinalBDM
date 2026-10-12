"""Verifica que el paquete se instala bien y que la config del protocolo es coherente."""

import yaml

import tpf
from tpf.paths import CONFIGS, ROOT


def test_package_imports():
    assert tpf.__version__
    assert (ROOT / "pyproject.toml").exists()


def test_config_is_consistent():
    cfg = yaml.safe_load((CONFIGS / "default.yaml").read_text(encoding="utf-8"))
    assert len(cfg["seeds"]) == 8
    assert abs(sum(cfg["split"][k] for k in ("train", "val", "test")) - 1.0) < 1e-9
    assert cfg["tabpfn"]["balance_probabilities"] is False
