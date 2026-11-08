"""Tests de loaders.py.

- Los tests sintéticos validan la carga sin necesitar los CSV reales.
- Los tests con datos reales se saltean si faltan los archivos.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tpf.data import loaders
from tpf.data.loaders import (
    SPECS,
    DataIntegrityError,
    DatasetSpec,
    load_dataset,
)
from tpf.paths import DATA_RAW

TOY = DatasetSpec(
    name="toy",
    filename="toy.csv",
    target="y",
    n_rows=100,
    n_cols=4,
    n_pos=10,
    id_cols=("id",),
    allow_missing_features=False,
)


def test_specs_match_protocol():
    assert SPECS["ulb"].n_pos == 492 and SPECS["ulb"].n_rows == 284_807
    assert SPECS["home_credit"].n_pos == 24_825 and SPECS["home_credit"].n_rows == 307_511
    assert SPECS["ieee_cis"].n_pos == 20_663 and SPECS["ieee_cis"].n_rows == 590_540


# ---------------- datos sintéticos ----------------


def test_ieee_cis_join_sintetico(tmp_path, monkeypatch):
    """Comprueba que el join conserve todas las transacciones y su orden."""

    transaction = pd.DataFrame(
        {
            "TransactionID": [10, 11, 12],
            "isFraud": [0, 1, 0],
            "TransactionAmt": [100.0, 200.0, 300.0],
            "ProductCD": ["W", "H", "W"],
        }
    )

    identity = pd.DataFrame(
        {
            "TransactionID": [12, 10],
            "DeviceType": ["mobile", "desktop"],
            "id_01": [-5.0, -10.0],
        }
    )

    transaction.to_csv(tmp_path / "train_transaction.csv", index=False)
    identity.to_csv(tmp_path / "train_identity.csv", index=False)

    # Adaptamos los tamaños esperados solamente para este test pequeño.
    spec = DatasetSpec(
        name="ieee_cis",
        filename="train_transaction.csv",
        target="isFraud",
        n_rows=3,
        n_cols=4,
        n_pos=1,
        id_cols=("TransactionID",),
    )

    monkeypatch.setitem(SPECS, "ieee_cis", spec)
    monkeypatch.setattr(loaders, "IEEE_IDENTITY_SHAPE", (2, 3))

    X, y = load_dataset("ieee_cis", tmp_path)

    assert X.shape == (3, 4)
    assert y.tolist() == [0, 1, 0]
    assert "TransactionID" not in X.columns
    assert "isFraud" not in X.columns

    assert X.loc[0, "DeviceType"] == "desktop"
    assert pd.isna(X.loc[1, "DeviceType"])
    assert X.loc[2, "DeviceType"] == "mobile"

    # Un ID duplicado en identity debe ser rechazado.
    identity.loc[1, "TransactionID"] = 12
    identity.to_csv(tmp_path / "train_identity.csv", index=False)

    with pytest.raises(DataIntegrityError, match="repetidos"):
        load_dataset("ieee_cis", tmp_path)


# ---------------- datos reales ----------------


@pytest.mark.slow
@pytest.mark.parametrize("name", ["ulb", "home_credit", "ieee_cis"])
def test_real_dataset(name):
    spec = SPECS[name]

    paths = [DATA_RAW / spec.filename]

    if name == "ieee_cis":
        paths.append(DATA_RAW / "train_identity.csv")

    if any(not path.exists() for path in paths):
        pytest.skip(f"Faltan archivos de {name} en data/raw/")

    X, y = load_dataset(name)

    assert len(X) == len(y) == spec.n_rows
    assert int(y.sum()) == spec.n_pos
    assert spec.target not in X.columns
    assert not set(spec.id_cols) & set(X.columns)

    expected_features = spec.n_cols - 1 - len(spec.id_cols)

    if name == "ieee_cis":
        expected_features += loaders.IEEE_IDENTITY_SHAPE[1] - 1

    assert X.shape[1] == expected_features
