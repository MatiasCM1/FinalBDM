from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def check_inputs(y_true, p_fraud, sample_weight=None):
    """Valida etiquetas, probabilidades y pesos; devuelve arrays 1D."""
    y = np.asarray(y_true).ravel()
    p = np.asarray(p_fraud, dtype=float).ravel()
    if not len(y) or len(y) != len(p):
        raise ValueError("y_true y p_fraud deben tener la misma longitud no vacía")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true debe contener solo 0 y 1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p_fraud debe contener probabilidades entre 0 y 1")
    w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float).ravel()
    if len(w) != len(y) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("sample_weight debe tener pesos no negativos y suma positiva")
    return y.astype(int), p, w


def reliability_curve(y_true, p_fraud, sample_weight=None, n_bins=10):
    """Puntos de calibración por intervalos iguales (incluye p=1 en el último)."""
    y, p, w = check_inputs(y_true, p_fraud, sample_weight)
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins debe ser un entero positivo")

    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total_weight = w.sum()
    points = []
    for b in range(n_bins):
        mask = bins == b
        weight = float(w[mask].sum())
        if weight == 0:
            continue
        points.append(
            {
                "bin_left": b / n_bins,
                "bin_right": (b + 1) / n_bins,
                "n": int(mask.sum()),
                "mean_probability": float(np.average(p[mask], weights=w[mask])),
                "fraction_positive": float(np.average(y[mask], weights=w[mask])),
                "weight_fraction": float(weight / total_weight),
            }
        )
    return points


def save_reliability_plot(y_true, p_fraud, path, sample_weight=None, n_bins=10):
    """Guarda la curva de confiabilidad como una imagen PNG."""
    from pathlib import Path

    import matplotlib.pyplot as plt

    points = reliability_curve(y_true, p_fraud, sample_weight, n_bins)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", label="Ideal")
    ax.plot(
        [point["mean_probability"] for point in points],
        [point["fraction_positive"] for point in points],
        "o-",
        label="Modelo",
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Probabilidad predicha de fraude",
        ylabel="Frecuencia real de fraude",
        title="Curva de confiabilidad",
    )
    ax.legend()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def evaluate(y_true, p_fraud, threshold=0.5, sample_weight=None, n_bins=10):
    """Métricas; si solo existe una clase real, AUC-ROC no está definida (NaN)."""
    y, p, w = check_inputs(y_true, p_fraud, sample_weight)
    if not np.isfinite(threshold):
        raise ValueError("threshold debe ser finito")
    pred = (p >= threshold).astype(int)
    curve = reliability_curve(y, p, sample_weight=w, n_bins=n_bins)
    ece = sum(
        point["weight_fraction"] * abs(point["fraction_positive"] - point["mean_probability"])
        for point in curve
    )
    both_classes = len(np.unique(y[w > 0])) == 2
    return {
        "auc_roc": float(roc_auc_score(y, p, sample_weight=w)) if both_classes else float("nan"),
        "mcc": float(matthews_corrcoef(y, pred, sample_weight=w)),
        "f1_fraud": float(f1_score(y, pred, pos_label=1, sample_weight=w, zero_division=0)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y, pred, sample_weight=w))
            if both_classes
            else float("nan")
        ),
        "brier": float(np.average((p - y) ** 2, weights=w)),
        "ece": float(ece),
        "reliability_curve": curve,
    }
