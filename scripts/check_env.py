"""Diagnóstico del entorno: Python, PyTorch/CUDA, TabPFN, token.

Uso:
    python scripts/check_env.py            # chequeos rápidos (sin descargar pesos)
    python scripts/check_env.py --tabpfn   # + smoke test con TabPFN-2.5 (necesita token y red)
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from importlib import metadata

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"
PACKAGES = [
    "tabpfn",
    "torch",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "imbalanced-learn",
    "xgboost",
    "scipy",
    "statsmodels",
]


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def check_python() -> bool:
    ok = (3, 10) <= sys.version_info[:2] <= (3, 13)
    print(f"{OK if ok else FAIL} Python {platform.python_version()} ({platform.system()})")
    return ok


def check_packages() -> bool:
    ok = True
    for name in PACKAGES:
        try:
            print(f"{OK} {name:<17} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            print(f"{FAIL} {name:<17} no instalado")
            ok = False
    return ok


def check_torch() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if torch.version.cuda is None:
        print(
            f"{WARN} torch {torch.__version__} es una build CPU-only "
            "(reinstalar con scripts/setup.* y TORCH_INDEX=cu128 / -TorchIndex cu128)"
        )
        return True
    if not torch.cuda.is_available():
        print(
            f"{WARN} torch con CUDA {torch.version.cuda}, pero no ve la GPU "
            "(revisar driver con `nvidia-smi`)"
        )
        return True
    p = torch.cuda.get_device_properties(0)
    print(
        f"{OK} GPU: {p.name} | {p.total_memory / 2**30:.1f} GiB | "
        f"sm_{p.major}{p.minor} | CUDA {torch.version.cuda}"
    )
    return True


def check_token() -> None:
    if os.environ.get("TABPFN_TOKEN"):
        print(f"{OK} TABPFN_TOKEN definido")
    else:
        print(f"{WARN} TABPFN_TOKEN no definido (completar .env o login por navegador)")


def smoke_tabpfn() -> bool:
    import numpy as np
    import torch
    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_auc_score
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    X, y = make_classification(n_samples=600, n_features=12, weights=[0.9, 0.1], random_state=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2_5, device=device, n_estimators=4, balance_probabilities=False
    )
    clf.fit(X[:400], y[:400])
    proba = clf.predict_proba(X[400:])[:, 1]
    auc = roc_auc_score(y[400:], proba)
    print(f"{OK if np.isfinite(auc) else FAIL} TabPFN-2.5 en {device}: AUC toy = {auc:.3f}")
    return bool(np.isfinite(auc))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabpfn", action="store_true", help="ejecuta un fit/predict de prueba")
    args = ap.parse_args()

    load_dotenv_if_available()
    ok = check_python()
    ok &= check_packages()
    ok &= check_torch()
    check_token()
    if args.tabpfn:
        try:
            ok &= smoke_tabpfn()
        except Exception as exc:  # noqa: BLE001 - diagnóstico: mostrar cualquier causa
            print(f"{FAIL} smoke test TabPFN: {type(exc).__name__}: {exc}")
            ok = False
    print("\nEntorno listo." if ok else "\nHay problemas: revisar los [FAIL] de arriba.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
