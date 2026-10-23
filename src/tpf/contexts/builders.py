"""Construcción del contexto para TabPFN (Uniform, Stratified, Balanced).

Contrato:
    build_context(y_tr, m, strategy, seed, X_tr=None) -> (idx, meta)

- idx: posiciones dentro de train, sin repetidos y ordenadas. Devuelve índices.
- meta: ContextMeta con los conteos reales del contexto (n_pos, n_neg, pi_ctx, capped).
- La semilla gobierna el muestreo (protocolo: misma semilla para split y contexto).
- X_tr solo lo usan NearMiss e IHT, acá se ignora.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tpf.contexts.expected import expected_context_positives

SIMPLE = ("uniform", "stratified", "balanced")
INFORMED = ("nearmiss", "iht")


@dataclass(frozen=True)
class ContextMeta:
    strategy: str
    m: int
    seed: int
    n_pos: int
    n_neg: int
    pi_ctx: float  # proporción real de positivos en el contexto
    capped: bool  # True si Balanced no pudo llegar a 50/50 por falta de positivos


def build_context(
    y_tr, m: int, strategy: str, seed: int, X_tr=None
) -> tuple[np.ndarray, ContextMeta]:
    """Devuelve (idx, meta) para un contexto de tamaño m armado desde train."""
    if strategy in INFORMED:
        raise NotImplementedError(f"{strategy!r}: pendiente (necesita X_tr).")
    if strategy not in SIMPLE:
        raise ValueError(f"estrategia desconocida: {strategy!r}. Opciones: {SIMPLE + INFORMED}")

    y = np.asarray(y_tr)
    n = len(y)
    if not 0 < m <= n:
        raise ValueError(f"m={m} debe estar entre 1 y el tamaño de train ({n}).")

    rng = np.random.default_rng(seed)
    capped = False

    if strategy == "uniform":
        idx = rng.choice(n, size=m, replace=False)
    else:  # stratified y balanced: solo difieren en cuántos positivos piden
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        if len(pos) == 0:
            raise ValueError("train no tiene positivos.")
        exp = expected_context_positives(n, len(pos), m, strategy)
        k = int(exp.n_pos)
        idx = np.concatenate(
            [rng.choice(pos, size=k, replace=False), rng.choice(neg, size=m - k, replace=False)]
        )
        capped = exp.capped

    idx = np.sort(idx)
    n_pos = int(y[idx].sum())
    meta = ContextMeta(strategy, m, seed, n_pos, m - n_pos, n_pos / m, capped)
    return idx, meta
