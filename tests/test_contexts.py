import numpy as np
import pytest

from tpf.contexts.builders import build_context
from tpf.data.splits import make_split

N_ULB, POS_ULB = 170_884, 295
SIMPLE = ["uniform", "stratified", "balanced"]
SIZES = [1_000, 5_000, 10_000]


def make_y(n, n_pos, seed=123):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype="int8")
    y[rng.choice(n, n_pos, replace=False)] = 1
    return y


@pytest.fixture(scope="module")
def y_ulb():
    return make_y(N_ULB, POS_ULB)


@pytest.mark.parametrize("strategy", SIMPLE)
@pytest.mark.parametrize("m", SIZES)
def test_size_uniqueness_and_meta(y_ulb, strategy, m):
    idx, meta = build_context(y_ulb, m, strategy, seed=0)
    assert len(idx) == m
    assert np.all(np.diff(idx) > 0)  # ordenado y sin repetidos
    assert idx.min() >= 0 and idx.max() < N_ULB
    assert meta.n_pos == int(y_ulb[idx].sum())
    assert meta.n_pos + meta.n_neg == m
    assert meta.pi_ctx == pytest.approx(meta.n_pos / m)


@pytest.mark.parametrize("strategy", SIMPLE)
def test_deterministic_by_seed(y_ulb, strategy):
    a, _ = build_context(y_ulb, 1_000, strategy, seed=3)
    b, _ = build_context(y_ulb, 1_000, strategy, seed=3)
    c, _ = build_context(y_ulb, 1_000, strategy, seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize(
    ("strategy", "m", "n_pos", "capped"),
    [
        ("stratified", 1_000, 2, False),
        ("stratified", 5_000, 9, False),
        ("stratified", 10_000, 17, False),
        ("balanced", 1_000, 295, True),
        ("balanced", 5_000, 295, True),
        ("balanced", 10_000, 295, True),
    ],
)
def test_counts_match_table2(y_ulb, strategy, m, n_pos, capped):
    _, meta = build_context(y_ulb, m, strategy, seed=0)
    assert meta.n_pos == n_pos
    assert meta.capped is capped


def test_balanced_not_capped_is_50_50():
    y = make_y(10_000, 800)  # tipo Home Credit (8 %)
    _, meta = build_context(y, 1_000, "balanced", seed=0)
    assert meta.n_pos == 500 and meta.pi_ctx == 0.5 and not meta.capped


def test_stratified_keeps_at_least_one_positive():
    y = make_y(20_000, 3)  # m * pi = 0.15 -> se fuerza 1
    _, meta = build_context(y, 1_000, "stratified", seed=0)
    assert meta.n_pos == 1


def test_uniform_ignores_class(y_ulb):
    # Promedio de positivos sobre muchas semillas ≈ m * pi = 1.726
    mean = np.mean([build_context(y_ulb, 1_000, "uniform", s)[1].n_pos for s in range(300)])
    assert abs(mean - 1_000 * POS_ULB / N_ULB) < 0.5


@pytest.mark.parametrize("strategy", SIMPLE)
def test_context_is_subset_of_train(strategy):
    y = make_y(20_000, 35)
    parts = make_split(y, seed=0)
    idx, _ = build_context(y[parts["train"]], 1_000, strategy, seed=0)
    global_idx = parts["train"][idx]  # así lo hará el runner
    assert np.isin(global_idx, parts["train"]).all()
    assert not np.isin(global_idx, parts["val"]).any()
    assert not np.isin(global_idx, parts["test"]).any()


def test_invalid_inputs(y_ulb):
    with pytest.raises(ValueError, match="m="):
        build_context(y_ulb, N_ULB + 1, "uniform", seed=0)
    with pytest.raises(ValueError, match="desconocida"):
        build_context(y_ulb, 1_000, "foo", seed=0)
    with pytest.raises(NotImplementedError):
        build_context(y_ulb, 1_000, "nearmiss", seed=0)
