import numpy as np
import pytest

from tpf.data.splits import PARTS, index_hash, make_split

N, N_POS = 50_000, 85


@pytest.fixture(scope="module")
def y():
    rng = np.random.default_rng(123)
    y = np.zeros(N, dtype="int8")
    y[rng.choice(N, N_POS, replace=False)] = 1
    return y


# Comprueba que todos los registros estén repartidos entre las particiones
# y que ninguno aparezca más de una vez.
def test_disjoint_and_complete(y):
    parts = make_split(y, seed=0)
    all_idx = np.concatenate([parts[p] for p in PARTS])
    assert len(all_idx) == len(np.unique(all_idx)) == N


# Comprueba que las particiones tengan los tamaños esperados (60%, 20% y 20%)
# y que los positivos se distribuyan de forma similar entre ellas.
def test_proportions_and_positives(y):
    parts = make_split(y, seed=0)
    for p, frac in zip(PARTS, (0.6, 0.2, 0.2), strict=True):
        assert abs(len(parts[p]) / N - frac) < 0.001
        assert abs(y[parts[p]].sum() - frac * N_POS) <= 1
        assert y[parts[p]].sum() >= 10


# Comprueba que usar la misma semilla produzca la misma partición y que
# cambiarla genere una partición diferente.
def test_reproducible_by_hash(y):
    h = [index_hash(make_split(y, seed=s)["test"]) for s in (0, 0, 1)]
    assert h[0] == h[1]
    assert h[0] != h[2]
