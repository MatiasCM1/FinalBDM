"""Piloto D08: TabPFN-2.5, contextos m=1000 y lotes de predicción.

Uso:
    python scripts/03_pilot_tabpfn.py
    python scripts/03_pilot_tabpfn.py --n-estimators 4
"""

from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from tpf.contexts.builders import build_context
from tpf.data.loaders import load_dataset
from tpf.data.preprocessing import cat_indices, tabpfn_view
from tpf.data.splits import load_split
from tpf.paths import CONFIGS, RESULTS

SEED = 0
M = 1_000
N_PRED = 5_000
BATCH_SIZES = (1_000, 2_000, 5_000)

COLUMNS = (
    "dataset",
    "strategy",
    "seed",
    "m",
    "n_pos",
    "n_neg",
    "pi_ctx",
    "capped",
    "n_estimators",
    "n_pred",
    "batch_size",
    "n_batches",
    "preparation_s",
    "model_creation_s",
    "fit_s",
    "predict_s",
    "predict_s_per_1k",
    "fit_peak_allocated_mib",
    "fit_peak_reserved_mib",
    "predict_peak_allocated_mib",
    "predict_peak_reserved_mib",
    "status",
    "error",
)

KEY = ("dataset", "strategy", "seed", "m", "n_estimators", "n_pred", "batch_size")


def measure(fn):
    """Tiempo y pico de memoria PyTorch durante una operación CUDA."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    allocated = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    reserved = round(torch.cuda.max_memory_reserved() / 2**20, 1)

    return result, round(seconds, 3), allocated, reserved


def predict_in_batches(clf, X_pred, batch_size):
    """Predice siempre las mismas filas, cambiando solo el tamaño del lote."""
    for start in range(0, len(X_pred), batch_size):
        block = X_pred[start : start + batch_size]
        probabilities = clf.predict_proba(block)

        if probabilities.shape != (len(block), 2) or not np.isfinite(probabilities).all():
            raise ValueError("TabPFN devolvió probabilidades inválidas")


def save_log(path: Path, new_rows: list[dict]) -> None:
    """Actualiza estas mediciones sin borrar otras configuraciones del CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    previous = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            previous = list(csv.DictReader(f))

    keys = {tuple(str(row[field]) for field in KEY) for row in new_rows}
    previous = [row for row in previous if tuple(str(row[field]) for field in KEY) not in keys]

    temp = path.with_suffix(".tmp")

    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(previous + new_rows)

    temp.replace(path)


def run_case(dataset: str, strategy: str, n_estimators: int, cfg: dict) -> list[dict]:
    """Prepara datos desde train, ajusta el modelo y mide tres tamaños de lote."""
    start = time.perf_counter()

    X, y = load_dataset(dataset)
    parts = load_split(dataset, SEED)
    tr, va = parts["train"], parts["val"]
    y_tr = np.asarray(y)[tr]

    view = tabpfn_view().fit(X.iloc[tr])
    idx, meta = build_context(y_tr, M, strategy, SEED)
    X_ctx = view.transform(X.iloc[tr].iloc[idx]).to_numpy(dtype=np.float32)

    rng = np.random.default_rng(SEED)
    pred_idx = np.sort(rng.choice(va, size=min(N_PRED, len(va)), replace=False))
    X_pred = view.transform(X.iloc[pred_idx]).to_numpy(dtype=np.float32)

    prep_s = round(time.perf_counter() - start, 3)

    base = {
        "dataset": dataset,
        "strategy": strategy,
        "seed": SEED,
        "m": M,
        "n_pos": meta.n_pos,
        "n_neg": meta.n_neg,
        "pi_ctx": round(meta.pi_ctx, 6),
        "capped": meta.capped,
        "n_estimators": n_estimators,
        "n_pred": len(X_pred),
        "preparation_s": prep_s,
    }

    print(f"\n{dataset} / {strategy}: {meta.n_pos} positivos, {meta.n_neg} negativos")

    if meta.n_pos == 0 or meta.n_neg == 0:
        print("  [SKIP] Contexto de una sola clase: TabPFN no puede clasificar")
        return [
            {
                **base,
                "batch_size": batch,
                "n_batches": 0,
                "status": "skipped_one_class",
                "error": "El contexto tiene una sola clase",
            }
            for batch in BATCH_SIZES
        ]

    clf = None
    rows = []

    try:
        start_model = time.perf_counter()

        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5,
            device="cuda",
            random_state=SEED,
            categorical_features_indices=cat_indices(view),
            n_estimators=n_estimators,
            fit_mode=cfg["fit_mode"],
            memory_saving_mode=cfg["memory_saving_mode"],
            balance_probabilities=cfg["balance_probabilities"],
        )

        create_s = round(time.perf_counter() - start_model, 3)

        _, fit_s, fit_alloc, fit_res = measure(lambda model=clf: model.fit(X_ctx, y_tr[idx]))

        print(f"  Fit: {fit_s:.2f} s | VRAM reservada pico: {fit_res:.1f} MiB")

        for batch in BATCH_SIZES:
            row = {
                **base,
                "batch_size": batch,
                "n_batches": (len(X_pred) + batch - 1) // batch,
                "model_creation_s": create_s,
                "fit_s": fit_s,
                "fit_peak_allocated_mib": fit_alloc,
                "fit_peak_reserved_mib": fit_res,
            }

            try:
                _, sec, alloc, reserved = measure(
                    lambda model=clf, batch_size=batch: predict_in_batches(
                        model, X_pred, batch_size
                    )
                )

                row.update(
                    {
                        "predict_s": sec,
                        "predict_s_per_1k": round(sec / len(X_pred) * 1000, 3),
                        "predict_peak_allocated_mib": alloc,
                        "predict_peak_reserved_mib": reserved,
                        "status": "complete",
                    }
                )

                print(
                    f"  Lote {batch}: {sec:.2f} s | "
                    f"{row['predict_s_per_1k']} s/1k | "
                    f"VRAM pico {reserved:.1f} MiB"
                )

            except Exception as exc:  # noqa: BLE001 - registrar OOM
                row.update(
                    {
                        "status": "error_predict",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"  [ERROR] Lote {batch}: {exc}")
                torch.cuda.empty_cache()

            rows.append(row)

    except Exception as exc:  # noqa: BLE001 - registrar fallas de creación/fit
        print(f"  [ERROR] Creación/fit: {type(exc).__name__}: {exc}")
        rows = [
            {
                **base,
                "batch_size": batch,
                "n_batches": 0,
                "status": "error_fit",
                "error": f"{type(exc).__name__}: {exc}",
            }
            for batch in BATCH_SIZES
        ]

    finally:
        del clf
        gc.collect()
        torch.cuda.empty_cache()

    return rows


def main() -> None:
    load_dotenv()

    cfg = yaml.safe_load((CONFIGS / "default.yaml").read_text(encoding="utf-8"))["tabpfn"]

    if cfg["model_version"] != "v2.5":
        raise ValueError("El piloto exige ModelVersion.V2_5")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets",
        nargs="+",
        choices=("ulb", "home_credit"),
        default=("ulb", "home_credit"),
    )
    ap.add_argument(
        "--strategies",
        nargs="+",
        choices=("uniform", "balanced"),
        default=("uniform", "balanced"),
    )
    ap.add_argument(
        "--n-estimators",
        type=int,
        choices=(4, 8),
        default=cfg["n_estimators"],
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("No se detectó CUDA. Ejecutar primero python scripts/check_env.py")

    path = RESULTS / "pilot_log.csv"

    for dataset in args.datasets:
        for strategy in args.strategies:
            rows = run_case(dataset, strategy, args.n_estimators, cfg)
            save_log(path, rows)

    print(f"\nLog del piloto: {path}")


if __name__ == "__main__":
    main()
