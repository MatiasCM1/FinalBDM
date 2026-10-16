"""Tests anti-fuga: los preprocesadores solo pueden aprender de train."""

import numpy as np
import pandas as pd
import pytest

from tpf.data.preprocessing import aux_view, cat_indices, tabpfn_view
from tpf.data.splits import make_split

N = 2000
VIEWS = [tabpfn_view, aux_view]


# Genera 2000 filas (2 numéricas + 1 categórica, con NaNs en "b") y target 5% positivos.
@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=N),
            "b": rng.normal(5, 2, size=N),
            "c": rng.choice(["x", "y", "z"], N),
        }
    )
    X.loc[rng.choice(N, 100, replace=False), "b"] = np.nan
    y = (rng.random(N) < 0.05).astype("int8")
    return X, y


# Devuelve una copia de X con val+test deformados.
def alter_eval(X, parts):
    """Copia de X con val/test deformados: valores extremos y una categoría nueva."""
    X2 = X.copy()
    ev = np.concatenate([parts["val"], parts["test"]])
    X2.loc[ev, "a"] = 1e6
    X2.loc[ev[:50], "c"] = "nueva"
    return X2, ev


# Ajustar con el mismo train da la misma transformación: fit no mira val/test.
@pytest.mark.parametrize("make_view", VIEWS)
def test_fit_ignores_val_test(make_view, data):
    X, y = data
    p = make_split(y, seed=0)
    X2, _ = alter_eval(X, p)
    tr = p["train"]
    v1 = make_view().fit(X.iloc[tr])
    v2 = make_view().fit(X2.iloc[tr])
    pd.testing.assert_frame_equal(v1.transform(X.iloc[tr]), v2.transform(X2.iloc[tr]))


# Transformar val/test no altera el estado: re-transformar train devuelve lo mismo.
@pytest.mark.parametrize("make_view", VIEWS)
def test_transform_does_not_refit(make_view, data):
    X, y = data
    p = make_split(y, seed=0)
    X2, ev = alter_eval(X, p)
    v = make_view().fit(X.iloc[p["train"]])
    before = v.transform(X.iloc[p["train"]])
    v.transform(X2.iloc[ev])
    pd.testing.assert_frame_equal(before, v.transform(X.iloc[p["train"]]))


# En tabpfn_view una categoría no vista se vuelve NaN (ordinal) y queda marcada en cat_indices.
def test_unknown_category_tabpfn(data):
    X, y = data
    p = make_split(y, seed=0)
    X2, ev = alter_eval(X, p)
    v = tabpfn_view().fit(X.iloc[p["train"]])
    out = v.transform(X2.iloc[ev[:50]])  # estas 50 filas tienen c = "nueva"
    assert out["c"].isna().all()
    assert cat_indices(v) == [2]


# En aux_view una categoría no vista se ignora: todas sus columnas one-hot (c_) quedan en 0.
def test_unknown_category_aux(data):
    X, y = data
    p = make_split(y, seed=0)
    X2, ev = alter_eval(X, p)
    v = aux_view().fit(X.iloc[p["train"]])
    out = v.transform(X2.iloc[ev[:50]])
    assert (out.filter(like="c_").sum(axis=1) == 0).all()


# Con el mismo seed, split y transformación son idénticos (determinista).
@pytest.mark.parametrize("make_view", VIEWS)
def test_deterministic_by_seed(make_view, data):
    X, y = data
    outs = []
    for _ in range(2):
        p = make_split(y, seed=3)
        v = make_view().fit(X.iloc[p["train"]])
        outs.append(v.transform(X.iloc[p["test"]]))
    pd.testing.assert_frame_equal(*outs)
