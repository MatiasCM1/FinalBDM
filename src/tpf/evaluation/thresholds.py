"""Tres reglas de decisión: R0 fija, R1 por contexto, R2 por validación."""

from __future__ import annotations

import numpy as np

from tpf.evaluation.metrics import check_inputs


def choose_thresholds(y_val, p_val, pi_ctx, sample_weight=None):
    """Devuelve los umbrales R0, R1 y R2. Nunca utiliza etiquetas de test."""
    if not np.isfinite(pi_ctx) or not 0 <= pi_ctx <= 1:
        raise ValueError("pi_ctx debe estar entre 0 y 1")
    y, p, w = check_inputs(y_val, p_val, sample_weight)

    order = np.argsort(-p, kind="stable")
    ps, ys, ws = p[order], y[order], w[order]
    end = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]

    tp = np.cumsum(ws * ys)[end]
    fp = np.cumsum(ws * (1 - ys))[end]
    positives = float((w * y).sum())
    negatives = float((w * (1 - y)).sum())
    fn = np.maximum(positives - tp, 0.0)
    tn = np.maximum(negatives - fp, 0.0)

    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denominator, out=np.zeros(len(end)), where=denominator > 0)

    # Si hay empate, preferimos el umbral más alto (menos falsas alarmas).
    scores = np.r_[0.0, mcc]
    thresholds = np.r_[np.nextafter(ps[0], np.inf), ps[end]]
    best = int(np.argmax(scores))
    return {"R0": 0.5, "R1": float(pi_ctx), "R2": float(thresholds[best])}
