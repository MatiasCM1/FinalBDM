"""TabPFN-2.5 en D0/D1, semilla 0, Uniform/Balanced y m=1k/5k.

Las predicciones se guardan una vez; R0/R1/R2 se evalúan luego sin GPU.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from tpf.data.loaders import load_dataset
from tpf.evaluation.metrics import save_reliability_plot
from tpf.evaluation.runner import run_once
from tpf.paths import CONFIGS, REPORTS, RESULTS

SEED = 0
DATASETS = ("home_credit", "ulb")
SIZES = (1_000, 5_000)
STRATEGIES = ("uniform", "balanced")
RULES = ("R0", "R1", "R2")
METRICS = (
    "auc_roc",
    "mcc",
    "f1_fraud",
    "balanced_accuracy",
    "brier",
    "ece",
)


def make_predict_fn(cfg, n_estimators, batch_size):
    """Devuelve la función que usa el runner para una única corrida."""

    def predict_fn(X_ctx, y_ctx, X_val, X_test, cat_cols):
        clf = None
        try:
            clf = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2_5,
                device="cuda",
                random_state=SEED,
                categorical_features_indices=cat_cols,
                n_estimators=n_estimators,
                fit_mode=cfg["fit_mode"],
                memory_saving_mode=cfg["memory_saving_mode"],
                balance_probabilities=cfg["balance_probabilities"],
            )
            clf.fit(X_ctx, y_ctx)

            def batches(X, classifier):
                pieces = []
                for start in range(0, len(X), batch_size):
                    p = classifier.predict_proba(X[start : start + batch_size])[:, 1]
                    pieces.append(p)
                return np.concatenate(pieces)

            return batches(X_val, clf), batches(X_test, clf)
        finally:
            del clf
            gc.collect()
            torch.cuda.empty_cache()

    return predict_fn


def export_summary(model_id):
    """Genera 24 filas: 8 configuraciones x 3 reglas."""

    rows = []

    for dataset in DATASETS:
        for m in SIZES:
            for strategy in STRATEGIES:
                run_dir = RESULTS / dataset / f"seed{SEED}" / f"{strategy}_m{m}_{model_id}"

                info_path = run_dir / "run.json"
                info = (
                    json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
                )

                metrics_path = run_dir / "metrics.json"
                scores = (
                    json.loads(metrics_path.read_text(encoding="utf-8"))["rules"]
                    if metrics_path.exists()
                    else {}
                )

                ctx = info.get("context", {})

                for rule in RULES:
                    result = scores.get(rule, {})
                    row = {
                        "dataset": dataset,
                        "seed": SEED,
                        "strategy": strategy,
                        "m": m,
                        "rule": rule,
                        "threshold": result.get("threshold", ""),
                        "n_pos": ctx.get("n_pos", ""),
                        "n_neg": ctx.get("n_neg", ""),
                        "pi_ctx": ctx.get("pi_ctx", ""),
                        "capped": ctx.get("capped", ""),
                        "status": info.get("status", "pending"),
                    }

                    row.update(
                        {metric: result.get("test", {}).get(metric, "") for metric in METRICS}
                    )
                    rows.append(row)

    path = REPORTS / "tables" / "d10_1_tabpfn_seed0.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = (
        "dataset",
        "seed",
        "strategy",
        "m",
        "rule",
        "threshold",
        "n_pos",
        "n_neg",
        "pi_ctx",
        "capped",
        *METRICS,
        "status",
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Tabla: {path}")

    for m in (5_000, 1_000):
        run_dir = RESULTS / "ulb" / f"seed{SEED}" / f"balanced_m{m}_{model_id}"

        if not (run_dir / "predictions.npz").exists():
            continue

        _, y = load_dataset("ulb")

        with np.load(run_dir / "predictions.npz") as saved:
            labels = np.asarray(y)[saved["idx_test"]]
            probs = saved["p_test"]

        figure = REPORTS / "figures" / "d10_1_calibracion_ulb.png"
        save_reliability_plot(labels, probs, figure)

        print(f"[OK] Curva de calibración ULB Balanced m={m}: {figure}")
        break


def main():
    load_dotenv()

    cfg = yaml.safe_load((CONFIGS / "default.yaml").read_text(encoding="utf-8"))["tabpfn"]

    if cfg["model_version"] != "v2.5" or cfg["balance_probabilities"]:
        raise ValueError("Se exige TabPFN-2.5 y balance_probabilities=False")

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    ap.add_argument("--sizes", nargs="+", type=int, choices=SIZES, default=SIZES)
    ap.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES)
    ap.add_argument("--n-estimators", type=int, choices=(4, 8), default=cfg["n_estimators"])
    ap.add_argument("--batch-size", type=int, default=cfg["predict_batch_size"])
    ap.add_argument("--summary-only", action="store_true")

    args = ap.parse_args()

    if args.batch_size <= 0:
        ap.error("--batch-size debe ser positivo")

    model_id = f"tabpfn25_e{args.n_estimators}_bs{args.batch_size}"

    if not args.summary_only:
        if not torch.cuda.is_available():
            raise RuntimeError("No se detectó GPU CUDA; ver scripts/check_env.py")

        predict_fn = make_predict_fn(cfg, args.n_estimators, args.batch_size)

        for dataset in args.datasets:
            for m in args.sizes:
                for strategy in args.strategies:
                    run = run_once(dataset, SEED, strategy, m, model_id, predict_fn)
                    print(f"{dataset} / {strategy} / m={m}: {run['status']}")

    export_summary(model_id)


if __name__ == "__main__":
    main()
