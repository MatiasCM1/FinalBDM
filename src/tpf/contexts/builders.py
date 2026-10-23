"""Construcción del contexto para TabPFN (Uniform, Stratified, Balanced, NearMiss).

Contrato:
    build_context(y_tr, m, strategy, seed, X_tr=None) -> (idx, meta)

- idx: posiciones dentro de train, sin repetidos y ordenadas.
  Devuelve índices, no datos.
- meta: ContextMeta con los conteos reales del contexto (n_pos, n_neg, pi_ctx, capped).
- La semilla gobierna el muestreo (protocolo: misma semilla para split y contexto).
- X_tr: solo lo usa NearMiss. Debe ser la vista `aux_view` ajustada solo con train
  (numérica, sin NaN, una fila por elemento de y_tr).

NearMiss-2: se eligen los positivos igual que en Balanced y se
completan con los negativos de menor distancia media a los NEARMISS_K positivos más
lejanos. Se calcula por bloques para no armar la matriz negativos x positivos completa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from tpf.contexts.expected import expected_context_positives

STRATEGIES = ("uniform", "stratified", "balanced", "nearmiss", "iht")
NEARMISS_K = 3  # n_neighbors de Mqadi et al.
CHUNK = 2_000  # negativos por bloque de distancias


@dataclass(frozen=True)
class ContextMeta:
    strategy: str
    m: int
    seed: int
    n_pos: int
    n_neg: int
    pi_ctx: float  # proporción real de positivos en el contexto (insumo de R1)
    capped: bool  # True si no hubo positivos suficientes para llegar a 50/50


def _nearmiss2_negatives(X, neg, pos_sel, n_sel, k=NEARMISS_K, chunk=CHUNK) -> np.ndarray:
    """Los n_sel negativos con menor distancia media a los k positivos más lejanos."""
    X_pos = X[pos_sel]
    k = min(k, len(pos_sel))
    score = np.empty(len(neg))
    for i in range(0, len(neg), chunk):
        d = euclidean_distances(X[neg[i : i + chunk]], X_pos)
        score[i : i + chunk] = np.partition(d, -k, axis=1)[:, -k:].mean(axis=1)
    return neg[np.argsort(score, kind="stable")[:n_sel]]  # estable => determinista


def build_context(
    y_tr, m: int, strategy: str, seed: int, X_tr=None
) -> tuple[np.ndarray, ContextMeta]:
    """Devuelve (idx, meta) para un contexto de tamaño m armado desde train."""
    if strategy not in STRATEGIES:
        raise ValueError(f"estrategia desconocida: {strategy!r}. Opciones: {STRATEGIES}")
    if strategy == "iht":
        raise NotImplementedError("iht: pendiente.")

    y = np.asarray(y_tr)
    n = len(y)
    if not 0 < m <= n:
        raise ValueError(f"m={m} debe estar entre 1 y el tamaño de train ({n}).")
    if strategy == "nearmiss" and (X_tr is None or len(X_tr) != n):
        raise ValueError("nearmiss necesita X_tr con una fila por cada elemento de y_tr.")

    rng = np.random.default_rng(seed)
    capped = False

    if strategy == "uniform":
        idx = rng.choice(n, size=m, replace=False)
    else:  # stratified, balanced y nearmiss eligen los positivos igual; difieren en los negativos
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        if len(pos) == 0:
            raise ValueError("train no tiene positivos.")
        exp = expected_context_positives(n, len(pos), m, strategy)
        k = int(exp.n_pos)
        pos_sel = rng.choice(pos, size=k, replace=False)
        if strategy == "nearmiss":
            neg_sel = _nearmiss2_negatives(np.asarray(X_tr, dtype=np.float32), neg, pos_sel, m - k)
        else:
            neg_sel = rng.choice(neg, size=m - k, replace=False)
        idx = np.concatenate([pos_sel, neg_sel])
        capped = exp.capped

    idx = np.sort(idx)
    n_pos = int(y[idx].sum())
    meta = ContextMeta(strategy, m, seed, n_pos, m - n_pos, n_pos / m, capped)
    return idx, meta
