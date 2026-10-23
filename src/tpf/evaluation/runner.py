from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tpf.contexts.builders import build_context
from tpf.data.loaders import load_dataset
from tpf.data.preprocessing import aux_view, cat_indices, tabpfn_view
from tpf.data.splits import load_split
from tpf.evaluation.metrics import check_inputs, evaluate
from tpf.evaluation.thresholds import choose_thresholds
from tpf.paths import RESULTS


def _write_json(path, obj):
    """Guarda primero un temporal y después lo reemplaza."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def score_saved_run(run_dir, y_val=None, y_test=None):
    """Recalcula R0/R1/R2 usando solo archivos guardados (sin GPU ni TabPFN)."""
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if info["status"] != "complete":
        raise ValueError("La corrida no tiene probabilidades para evaluar")

    if y_val is None and y_test is None:
        _, y = load_dataset(info["dataset"])
        parts = load_split(info["dataset"], info["seed"])
        y_val = np.asarray(y)[parts["val"]]
        y_test = np.asarray(y)[parts["test"]]
    elif y_val is None or y_test is None:
        raise ValueError("Pasar ambas etiquetas (val y test), o ninguna")

    with np.load(run_dir / "predictions.npz") as data:
        p_val, p_test = data["p_val"], data["p_test"]
        w_val = data["w_val"] if "w_val" in data else None
        w_test = data["w_test"] if "w_test" in data else None
        if len(p_val) != len(y_val) or len(p_test) != len(y_test):
            raise ValueError("Las probabilidades no coinciden con las filas de val/test")
        rules = choose_thresholds(y_val, p_val, info["context"]["pi_ctx"], w_val)
        scores = {
            rule: {
                "threshold": threshold,
                "val": evaluate(y_val, p_val, threshold, w_val),
                "test": evaluate(y_test, p_test, threshold, w_test),
            }
            for rule, threshold in rules.items()
        }

    def safe(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [safe(v) for v in value]
        return value

    result = safe({"rules": scores})
    _write_json(run_dir / "metrics.json", result)
    return result


def run_once(
    dataset,
    seed,
    strategy,
    m,
    model_id,
    predict_fn,
    results_dir=None,
    sample_weight_val=None,
    sample_weight_test=None,
):
    """Carga split, arma contexto, infiere una sola vez y guarda todo.

    model_id debe cambiar si cambia versión, estimadores u otra opción del modelo.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model_id):
        raise ValueError("model_id solo admite letras, números, _ y -")
    base = Path(results_dir) if results_dir is not None else RESULTS
    run_dir = base / dataset / f"seed{seed}" / f"{strategy}_m{m}_{model_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    info_path = run_dir / "run.json"
    probabilities_path = run_dir / "predictions.npz"
    if info_path.exists():
        prev = json.loads(info_path.read_text(encoding="utf-8"))
        if prev["status"] == "skipped":
            return {"status": "cached_skip", "path": str(run_dir)}
        if prev["status"] == "complete" and probabilities_path.exists():
            if not (run_dir / "metrics.json").exists():
                score_saved_run(run_dir)
            return {"status": "cached", "path": str(run_dir)}

    start = time.perf_counter()
    X, y = load_dataset(dataset)
    y = np.asarray(y)
    parts = load_split(dataset, seed)
    tr, va, te = (parts[p] for p in ("train", "val", "test"))
    X_tr, y_tr = X.iloc[tr], y[tr]

    # Ambos preprocesadores se ajustan SOLO con train y ANTES de armar el contexto.
    view = tabpfn_view().fit(X_tr)
    X_aux = None
    if strategy in {"nearmiss", "iht"}:
        X_aux = aux_view().fit_transform(X_tr).to_numpy(dtype=np.float32)
    ctx_idx, meta = build_context(y_tr, m, strategy, seed, X_tr=X_aux)

    info = {
        "status": "complete",
        "dataset": dataset,
        "seed": int(seed),
        "strategy": strategy,
        "m": int(m),
        "model_id": model_id,
        "context": asdict(meta),
    }
    if meta.n_pos == 0 or meta.n_neg == 0:
        info["status"] = "skipped"
        info["reason"] = "El contexto tiene una sola clase; no se ejecuta TabPFN"
        _write_json(info_path, info)
        return {"status": "skipped", "path": str(run_dir)}

    X_ctx = view.transform(X_tr.iloc[ctx_idx]).to_numpy(dtype=np.float32)
    X_val = view.transform(X.iloc[va]).to_numpy(dtype=np.float32)
    X_test = view.transform(X.iloc[te]).to_numpy(dtype=np.float32)
    prepared = time.perf_counter()

    import torch

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    p_val, p_test = predict_fn(X_ctx, y_tr[ctx_idx], X_val, X_test, cat_indices(view))
    if use_cuda:
        torch.cuda.synchronize()
    ended = time.perf_counter()

    p_val = check_inputs(y[va], p_val)[1]
    p_test = check_inputs(y[te], p_test)[1]
    payload = {"p_val": p_val, "p_test": p_test, "idx_val": va, "idx_test": te}
    if sample_weight_val is not None:
        payload["w_val"] = check_inputs(y[va], p_val, sample_weight_val)[2]
    if sample_weight_test is not None:
        payload["w_test"] = check_inputs(y[te], p_test, sample_weight_test)[2]
    tmp = run_dir / "predictions.tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, probabilities_path)
    info["timings_seconds"] = {
        "preparation": round(prepared - start, 3),
        "inference": round(ended - prepared, 3),
        "total": round(ended - start, 3),
    }
    info["vram_peak_mib"] = (
        round(torch.cuda.max_memory_allocated() / 2**20, 1) if use_cuda else None
    )
    _write_json(info_path, info)
    score_saved_run(run_dir, y[va], y[te])
    return {"status": "complete", "path": str(run_dir)}
