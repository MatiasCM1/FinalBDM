from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import hypergeom


@dataclass(frozen=True)
class ExpectedContext:
    n_pos: float  # positivos esperados en el contexto
    pi_ctx: float  # proporción de positivos en el contexto
    p_zero_pos: float  # prob. de que el contexto no tenga ningún positivo
    capped: bool  # True si no hay positivos suficientes para llegar a 50/50


def expected_context_positives(
    n_train: int, pos_train: int, m: int, strategy: str
) -> ExpectedContext:
    """Positivos esperados en un contexto de tamaño m armado desde train."""
    pi = pos_train / n_train
    p_zero, capped = 0.0, False

    if strategy == "uniform":
        k = m * pi
        p_zero = float(hypergeom.pmf(0, n_train, pos_train, m))
    elif strategy == "stratified":
        k = max(round(m * pi), 1)
    elif strategy in {"balanced", "nearmiss", "iht"}:
        # se toman primero los positivos y se completa con negativos.
        k = min(pos_train, m // 2)
        capped = pos_train < m // 2
    else:
        raise ValueError(f"estrategia desconocida: {strategy!r}")

    return ExpectedContext(float(k), float(k) / m, p_zero, capped)
