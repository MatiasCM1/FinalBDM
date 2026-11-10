from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
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


def _weighted_quantiles(
    values: np.ndarray, quantiles: tuple[float, ...], weights: np.ndarray
) -> list[float]:
    if len(values) == 0 or weights.sum() <= 0:
        return [float("nan")] * len(quantiles)
    order = np.argsort(values, kind="mergesort")
    v_sorted = values[order]
    w_sorted = weights[order]
    cdf = np.cumsum(w_sorted) / w_sorted.sum()
    return [float(np.interp(q, cdf, v_sorted)) for q in quantiles]


def _ks_and_top_recall(
    y: np.ndarray, p: np.ndarray, w: np.ndarray, top_frac: float = 0.01
) -> tuple[float, float]:
    pos_mask = y == 1
    neg_mask = y == 0
    w_pos = float(w[pos_mask].sum())
    w_neg = float(w[neg_mask].sum())
    if w_pos <= 0 or w_neg <= 0:
        return float("nan"), float("nan")

    order = np.argsort(-p, kind="mergesort")
    p_ord = p[order]
    y_ord = y[order]
    w_ord = w[order]

    cum_pos = np.cumsum(w_ord * (y_ord == 1)) / w_pos
    cum_neg = np.cumsum(w_ord * (y_ord == 0)) / w_neg
    distinct_ends = np.flatnonzero(np.r_[p_ord[:-1] != p_ord[1:], True])

    ks = float(np.max(np.abs(cum_pos[distinct_ends] - cum_neg[distinct_ends])))
    cum_weight_frac = np.cumsum(w_ord) / (w_pos + w_neg)
    recall_top = float(
        np.interp(
            top_frac,
            np.r_[0.0, cum_weight_frac[distinct_ends]],
            np.r_[0.0, cum_pos[distinct_ends]],
        )
    )
    return ks, recall_top


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

    pos_mask = y == 1
    neg_mask = y == 0
    tp = float(w[pos_mask & (pred == 1)].sum())
    fn = float(w[pos_mask & (pred == 0)].sum())
    fp = float(w[neg_mask & (pred == 1)].sum())
    tn = float(w[neg_mask & (pred == 0)].sum())

    recall_pos = float(recall_score(y, pred, pos_label=1, sample_weight=w, zero_division=0))
    precision_pos = float(precision_score(y, pred, pos_label=1, sample_weight=w, zero_division=0))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    fpr = fp / (tn + fp) if (tn + fp) > 0 else float("nan")
    g_mean = (
        float(np.sqrt(recall_pos * specificity))
        if np.isfinite(specificity) and (pos_mask & (w > 0)).any()
        else float("nan")
    )

    ks_stat, recall_top1pct = _ks_and_top_recall(y, p, w, top_frac=0.01)
    neg_q10, neg_q50, neg_q90, neg_q99 = _weighted_quantiles(
        p[neg_mask], (0.10, 0.50, 0.90, 0.99), w[neg_mask]
    )
    pos_q10, pos_q50, pos_q90, pos_q99 = _weighted_quantiles(
        p[pos_mask], (0.10, 0.50, 0.90, 0.99), w[pos_mask]
    )

    return {
        "auc_roc": float(roc_auc_score(y, p, sample_weight=w)) if both_classes else float("nan"),
        "auc_pr": (
            float(average_precision_score(y, p, sample_weight=w)) if both_classes else float("nan")
        ),
        "ks_stat": ks_stat,
        "recall_top1pct": recall_top1pct,
        "mcc": float(matthews_corrcoef(y, pred, sample_weight=w)),
        "f1_fraud": float(f1_score(y, pred, pos_label=1, sample_weight=w, zero_division=0)),
        "precision_fraud": precision_pos,
        "recall_fraud": recall_pos,
        "specificity": float(specificity),
        "fpr": float(fpr),
        "g_mean": g_mean,
        "balanced_accuracy": (
            float(balanced_accuracy_score(y, pred, sample_weight=w))
            if both_classes
            else float("nan")
        ),
        "brier": float(np.average((p - y) ** 2, weights=w)),
        "log_loss": (
            float(log_loss(y, np.clip(p, 1e-15, 1.0 - 1e-15), sample_weight=w, labels=[0, 1]))
            if both_classes
            else float("nan")
        ),
        "ece": float(ece),
        "prevalence": float(np.average(y, weights=w)),
        "mean_p_fraud": float(np.average(p, weights=w)),
        "predicted_positive_rate": float(np.average(pred, weights=w)),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "tp_raw": int((pos_mask & (pred == 1)).sum()),
            "fp_raw": int((neg_mask & (pred == 1)).sum()),
            "tn_raw": int((neg_mask & (pred == 0)).sum()),
            "fn_raw": int((pos_mask & (pred == 0)).sum()),
        },
        "prob_distribution": {
            "neg_mean": float(np.average(p[neg_mask], weights=w[neg_mask]))
            if w[neg_mask].sum() > 0
            else float("nan"),
            "neg_q10": neg_q10,
            "neg_q50": neg_q50,
            "neg_q90": neg_q90,
            "neg_q99": neg_q99,
            "pos_mean": float(np.average(p[pos_mask], weights=w[pos_mask]))
            if w[pos_mask].sum() > 0
            else float("nan"),
            "pos_q10": pos_q10,
            "pos_q50": pos_q50,
            "pos_q90": pos_q90,
            "pos_q99": pos_q99,
        },
        "reliability_curve": curve,
    }


def _histogram_overlap(
    v0: np.ndarray, w0: np.ndarray, v1: np.ndarray, w1: np.ndarray, n_bins: int = 20
) -> float:
    """Coeficiente de solapamiento OVL en [0, 1] entre dos distribuciones 1D ponderadas."""
    if len(v0) == 0 or len(v1) == 0 or w0.sum() <= 0 or w1.sum() <= 0:
        return float("nan")
    lo = float(min(v0.min(), v1.min()))
    hi = float(max(v0.max(), v1.max()))
    if lo == hi:
        return 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    h0, _ = np.histogram(v0, bins=edges, weights=w0)
    h1, _ = np.histogram(v1, bins=edges, weights=w1)
    p0 = h0 / h0.sum()
    p1 = h1 / h1.sum()
    return float(np.minimum(p0, p1).sum())


def analyze_feature_distributions(
    X,
    y_true,
    p_fraud,
    threshold: float = 0.5,
    sample_weight=None,
    top_k: int = 15,
    n_bins: int = 20,
) -> list[dict]:
    """Analiza qué variables influyeron en la predicción y el solapamiento de distribuciones.

    Devuelve una lista ordenada por `abs(corr_with_pred)` con las `top_k` variables numéricas:
    - corr_with_pred: correlación de rangos (Spearman) entre la variable y la probabilidad predicha.
    - corr_with_true: correlación de rangos (Spearman) entre la variable y la etiqueta real.
    - overlap_true: solapamiento [0..1] de la variable entre y=0 e y=1 reales.
    - overlap_pred: solapamiento [0..1] de la variable entre pred=0 y pred=1 según `threshold`.
    - ks_true / ks_pred: separación Kolmogorov-Smirnov entre las dos distribuciones.
    - cuantiles (q10, q50, q90) en negativos vs positivos (reales y predichos).
    """
    import pandas as pd
    from scipy.stats import spearmanr

    y, p, w = check_inputs(y_true, p_fraud, sample_weight)
    pred = (p >= threshold).astype(int)
    df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    candidates = []
    for col in num_cols:
        vals = df[col].to_numpy(dtype=float)
        valid = np.isfinite(vals)
        if valid.sum() < 10:
            continue
        v_ok = vals[valid]
        p_ok = p[valid]
        y_ok = y[valid]
        if np.all(v_ok == v_ok[0]) or np.all(p_ok == p_ok[0]):
            continue
        r_pred = float(spearmanr(v_ok, p_ok).statistic)
        r_true = (
            float(spearmanr(v_ok, y_ok).statistic) if len(np.unique(y_ok)) == 2 else float("nan")
        )
        if not np.isfinite(r_pred):
            continue
        candidates.append((abs(r_pred), col, r_pred, r_true))

    candidates.sort(key=lambda item: item[0], reverse=True)
    results = []

    for _, col, r_pred, r_true in candidates[:top_k]:
        vals = df[col].to_numpy(dtype=float)
        valid = np.isfinite(vals)
        v = vals[valid]
        y_v = y[valid]
        pr_v = pred[valid]
        w_v = w[valid]

        neg_true = y_v == 0
        pos_true = y_v == 1
        neg_pred = pr_v == 0
        pos_pred = pr_v == 1

        ks_true, _ = _ks_and_top_recall(y_v, v, w_v)
        ks_pred, _ = _ks_and_top_recall(pr_v, v, w_v)

        t_neg_q10, t_neg_q50, t_neg_q90 = _weighted_quantiles(
            v[neg_true], (0.10, 0.50, 0.90), w[valid][neg_true]
        )
        t_pos_q10, t_pos_q50, t_pos_q90 = _weighted_quantiles(
            v[pos_true], (0.10, 0.50, 0.90), w[valid][pos_true]
        )
        p_neg_q10, p_neg_q50, p_neg_q90 = _weighted_quantiles(
            v[neg_pred], (0.10, 0.50, 0.90), w[valid][neg_pred]
        )
        p_pos_q10, p_pos_q50, p_pos_q90 = _weighted_quantiles(
            v[pos_pred], (0.10, 0.50, 0.90), w[valid][pos_pred]
        )

        ovl_true = _histogram_overlap(
            v[neg_true], w_v[neg_true], v[pos_true], w_v[pos_true], n_bins
        )
        ovl_pred = _histogram_overlap(
            v[neg_pred], w_v[neg_pred], v[pos_pred], w_v[pos_pred], n_bins
        )

        results.append(
            {
                "feature": str(col),
                "missing_rate": float(1.0 - valid.mean()),
                "corr_with_pred": r_pred,
                "corr_with_true": r_true,
                "overlap_true": ovl_true,
                "ks_true": ks_true,
                "overlap_pred": ovl_pred,
                "ks_pred": ks_pred,
                "true_neg_quantiles": {"q10": t_neg_q10, "q50": t_neg_q50, "q90": t_neg_q90},
                "true_pos_quantiles": {"q10": t_pos_q10, "q50": t_pos_q50, "q90": t_pos_q90},
                "pred_neg_quantiles": {"q10": p_neg_q10, "q50": p_neg_q50, "q90": p_neg_q90},
                "pred_pos_quantiles": {"q10": p_pos_q10, "q50": p_pos_q50, "q90": p_pos_q90},
            }
        )

    return results
