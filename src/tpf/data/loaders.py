"""Carga de los datasets crudos como (X, y), sin transformar.

Contrato (protocolo, paso 0):
- No se limpia, imputa, codifica ni escala nada acá. Eso va en `preprocessing.py`
  y SOLO se ajusta con train (anti-fuga).
- Lo único que se hace: leer el CSV, validar que es el archivo esperado
  (forma y positivos) y separar la target.
- Se descartan las columnas identificadoras (`id_cols`): no son features.
- Los valores esperados viven en SPECS; el test
  los reutiliza.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tpf.paths import DATA_RAW


class DataIntegrityError(ValueError):
    """El archivo leído no coincide con lo esperado (versión, descarga corrupta, etc.)."""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    target: str
    n_rows: int
    n_cols: int  # columnas del CSV, incluida la target
    n_pos: int
    id_cols: tuple[str, ...] = ()
    allow_missing_features: bool = True


SPECS: dict[str, DatasetSpec] = {
    # D1: ULB Credit Card Fraud (0,17 %). Time, V1-V28, Amount, Class.
    "ulb": DatasetSpec(
        name="ulb",
        filename="creditcard.csv",
        target="Class",
        n_rows=284_807,
        n_cols=31,
        n_pos=492,
        allow_missing_features=False,
    ),
    # D0: Home Credit, solo application_train.csv (limitación declarada, decisión #10).
    "home_credit": DatasetSpec(
        name="home_credit",
        filename="application_train.csv",
        target="TARGET",
        n_rows=307_511,
        n_cols=122,
        n_pos=24_825,
        id_cols=("SK_ID_CURR",),
    ),
    # D2: IEEE-CIS Fraud Detection.
    "ieee_cis": DatasetSpec(
        name="ieee_cis",
        filename="train_transaction.csv",
        target="isFraud",
        n_rows=590_540,
        n_cols=394,
        n_pos=20_663,
        id_cols=("TransactionID",),
    ),
}


def _validate(df: pd.DataFrame, spec: DatasetSpec, path: Path) -> None:
    def fail(msg: str) -> None:
        raise DataIntegrityError(f"{spec.name} ({path.name}): {msg}")

    if df.shape != (spec.n_rows, spec.n_cols):
        fail(f"forma {df.shape}, se esperaba {(spec.n_rows, spec.n_cols)}")

    for col in (spec.target, *spec.id_cols):
        if col not in df.columns:
            fail(f"falta la columna '{col}'")

    target = df[spec.target]
    if target.isna().any():
        fail("la target tiene NaN")
    if not set(target.unique()) <= {0, 1}:
        fail(f"la target no es binaria 0/1: {sorted(target.unique())[:5]}")
    n_pos = int(target.sum())
    if n_pos != spec.n_pos:
        fail(f"{n_pos} positivos, se esperaban {spec.n_pos}")

    for col in spec.id_cols:
        if df[col].duplicated().any():
            fail(f"la columna identificadora '{col}' tiene valores repetidos")

    feats = df.drop(columns=[spec.target, *spec.id_cols])
    if not spec.allow_missing_features and feats.isna().any().any():
        fail("hay NaN en las features y este dataset no debería tenerlos")


def load_from_spec(
    spec: DatasetSpec, data_dir: str | Path | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Lee `spec.filename`, valida y devuelve (X, y) con índice 0..n-1."""
    path = Path(data_dir if data_dir is not None else DATA_RAW) / spec.filename
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Descargarlo y registrarlo en data/MANIFEST.md (ver Día 2)."
        )
    df = pd.read_csv(path)
    _validate(df, spec, path)

    y = df[spec.target].astype("int8")
    X = df.drop(columns=[spec.target, *spec.id_cols])
    return X, y


def load_ulb(data_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """D1. X: 30 columnas (Time, V1-V28, Amount) float64; y: Class int8."""
    return load_from_spec(SPECS["ulb"], data_dir)


def load_home_credit(data_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """D0. X: 120 columnas (sin SK_ID_CURR), con NaN y categóricas como texto; y: TARGET int8."""
    return load_from_spec(SPECS["home_credit"], data_dir)


# Forma esperada del archivo train_identity.csv.
IEEE_IDENTITY_SHAPE = (144_233, 41)


def load_ieee_cis(data_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """D2. Une transaction e identity por TransactionID y devuelve (X, y).

    Se conserva cada transacción, incluso cuando no tiene datos de identidad.
    No se imputan, escalan ni codifican variables en este paso.
    """

    root = Path(data_dir if data_dir is not None else DATA_RAW)
    transaction_path = root / "train_transaction.csv"
    identity_path = root / "train_identity.csv"

    for path in (transaction_path, identity_path):
        if not path.exists():
            raise FileNotFoundError(f"No existe {path}. Descargarlo de IEEE-CIS en Kaggle.")

    # El archivo principal tiene las transacciones y la etiqueta isFraud.
    transaction = pd.read_csv(transaction_path, low_memory=False)
    spec = SPECS["ieee_cis"]
    _validate(transaction, spec, transaction_path)

    # Identity aporta información adicional para parte de las transacciones.
    identity = pd.read_csv(identity_path, low_memory=False)

    if identity.shape != IEEE_IDENTITY_SHAPE:
        raise DataIntegrityError(
            f"ieee_cis ({identity_path.name}): forma {identity.shape}, "
            f"se esperaba {IEEE_IDENTITY_SHAPE}"
        )

    if "TransactionID" not in identity.columns:
        raise DataIntegrityError("ieee_cis: falta TransactionID en identity")

    if identity["TransactionID"].isna().any():
        raise DataIntegrityError("ieee_cis: hay TransactionID nulos en identity")

    if identity["TransactionID"].duplicated().any():
        raise DataIntegrityError("ieee_cis: hay TransactionID repetidos en identity")

    if not identity["TransactionID"].isin(transaction["TransactionID"]).all():
        raise DataIntegrityError(
            "ieee_cis: identity contiene transacciones que no están en transaction"
        )

    # Left join: no se pierden transacciones sin información de identity.
    df = transaction.merge(
        identity,
        on="TransactionID",
        how="left",
        sort=False,
        validate="one_to_one",
    )

    expected_cols = spec.n_cols + IEEE_IDENTITY_SHAPE[1] - 1

    if df.shape != (spec.n_rows, expected_cols):
        raise DataIntegrityError(
            f"ieee_cis: forma después del join {df.shape}, "
            f"se esperaba {(spec.n_rows, expected_cols)}"
        )

    y = df[spec.target].astype("int8")
    X = df.drop(columns=[spec.target, *spec.id_cols])

    return X, y


def load_dataset(name: str, data_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Punto de entrada por nombre (los de configs/default.yaml)."""
    if name == "ieee_cis":
        return load_ieee_cis(data_dir)
    if name not in SPECS:
        raise KeyError(f"Dataset desconocido: {name!r}. Opciones: {sorted(SPECS)}")
    return load_from_spec(SPECS[name], data_dir)
