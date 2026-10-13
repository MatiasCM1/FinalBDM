"""Tests de loaders.py.

- Los tests sintéticos validan la lógica de validación sin necesitar los CSV reales.
- Los tests con datos reales se saltean solos si data/raw/ no tiene los archivos.
"""

from __future__ import annotations

import pytest

from tpf.data.loaders import (
    SPECS,
    DatasetSpec,
    load_dataset,
)
from tpf.paths import DATA_RAW

TOY = DatasetSpec(
    name="toy",
    filename="toy.csv",
    target="y",
    n_rows=100,
    n_cols=4,  # id, a, b, y
    n_pos=10,
    id_cols=("id",),
    allow_missing_features=False,
)


def test_specs_match_protocol():
    assert SPECS["ulb"].n_pos == 492 and SPECS["ulb"].n_rows == 284_807
    assert SPECS["home_credit"].n_pos == 24_825 and SPECS["home_credit"].n_rows == 307_511


# ---------------- datos reales (se saltean si no están descargados) ----------------


@pytest.mark.slow
@pytest.mark.parametrize("name", ["ulb", "home_credit"])
def test_real_dataset(name):
    spec = SPECS[name]
    if not (DATA_RAW / spec.filename).exists():
        pytest.skip(f"{spec.filename} no está en data/raw/")
    X, y = load_dataset(name)
    assert len(X) == len(y) == spec.n_rows
    assert int(y.sum()) == spec.n_pos
    assert spec.target not in X.columns
    assert not set(spec.id_cols) & set(X.columns)
    assert X.shape[1] == spec.n_cols - 1 - len(spec.id_cols)
