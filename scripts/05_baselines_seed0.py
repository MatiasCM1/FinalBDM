"""Random Forest y XGBoost entrenados sobre todo train (semilla 0).

No se les aplica remuestreo ni pesos de clase. R2 se elige solo en val.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from tpf.data.loaders import load_dataset
from tpf.data.preprocessing import aux_view
from tpf.data.splits import load_split
from tpf.evaluation.runner import score_saved_run
from tpf.paths import REPORTS, RESULTS

SEED = 0
DATASETS = ("home_credit", "ulb")
MODELS = ("rf100", "xgb100_hist")
RULES = ("R0", "R2")
METRICS = (
    "auc_roc",
    "mcc",
    "f1_fraud",
    "balanced_accuracy",
    "brier",
    "ece",
)


def make_model(name):
    if name == "rf100":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=SEED,
            n_jobs=-1,
        )

    if name == "xgb100_hist":
        return XGBClassifier(
            n_estimators=100,
            tree_method="hist",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        )

    raise ValueError(f"Modelo desconocido: {name}")


def run_dir(dataset, model_id):
    return RESULTS / dataset / f"seed{SEED}" / f"full_train_{model_id}"


def is_cached(dataset, model_id):
    path = run_dir(dataset, model_id)
    info_path = path / "run.json"

    if not info_path.exists() or not (path / "predictions.npz").exists():
        return False

    info = json.loads(info_path.read_text(encoding="utf-8"))

    if info.get("status") != "complete":
        return False

    if not (path / "metrics.json").exists():
        score_saved_run(path)

    return True


def save_prediction(dataset, model_id, model, X_train, y_train, X_val, X_test, va, te):
    path = run_dir(dataset, model_id)
    path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - start

    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    pred_s = time.perf_counter() - start - fit_s

    tmp = path / "predictions.tmp.npz"

    np.savez_compressed(
        tmp,
        p_val=p_val,
        p_test=p_test,
        idx_val=va,
        idx_test=te,
    )

    os.replace(tmp, path / "predictions.npz")

    info = {
        "status": "complete",
        "dataset": dataset,
        "seed": SEED,
        "strategy": "full_train",
        "m": int(len(y_train)),
        "model_id": model_id,
        "context": {"pi_ctx": float(np.mean(y_train))},
        "timings_seconds": {
            "fit": round(fit_s, 3),
            "predict": round(pred_s, 3),
        },
    }

    (path / "run.json").write_text(
        json.dumps(info, indent=2),
        encoding="utf-8",
    )

    score_saved_run(path)


def export_summary():
    rows = []

    for dataset in DATASETS:
        for model_id in MODELS:
            path = run_dir(dataset, model_id)
            metrics_path = path / "metrics.json"

            scores = (
                json.loads(metrics_path.read_text(encoding="utf-8"))["rules"]
                if metrics_path.exists()
                else {}
            )

            for rule in RULES:
                result = scores.get(rule, {})

                row = {
                    "dataset": dataset,
                    "seed": SEED,
                    "model": model_id,
                    "rule": rule,
                    "threshold": result.get("threshold", ""),
                    "status": "complete" if result else "pending",
                }

                row.update({key: result.get("test", {}).get(key, "") for key in METRICS})
                rows.append(row)

    path = REPORTS / "tables" / "d10_2_clasicos_seed0.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = (
        "dataset",
        "seed",
        "model",
        "rule",
        "threshold",
        *METRICS,
        "status",
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Tabla: {path}")


def main():
    for dataset in DATASETS:
        missing = [name for name in MODELS if not is_cached(dataset, name)]

        if not missing:
            print(f"{dataset}: modelos ya calculados")
            continue

        X, y = load_dataset(dataset)
        parts = load_split(dataset, SEED)

        tr, va, te = (parts[part] for part in ("train", "val", "test"))

        # El preprocesamiento se ajusta SOLO con train.
        preprocessor = aux_view().fit(X.iloc[tr])

        X_train = preprocessor.transform(X.iloc[tr]).to_numpy(dtype=np.float32)

        X_val = preprocessor.transform(X.iloc[va]).to_numpy(dtype=np.float32)

        X_test = preprocessor.transform(X.iloc[te]).to_numpy(dtype=np.float32)

        y_train = np.asarray(y)[tr]

        for model_id in missing:
            model = make_model(model_id)

            save_prediction(dataset, model_id, model, X_train, y_train, X_val, X_test, va, te)

            del model
            gc.collect()

            print(f"{dataset} / {model_id}: complete")

    export_summary()


if __name__ == "__main__":
    main()
