"""Particiones train/val/test estratificadas.

- Dos etapas: train vs (val+test), y luego val vs test. Ambas estratificadas por la target.
- La semilla gobierna la partición. Se guardan SOLO índices (parquet largo: idx, part).
- Los hashes de índices van al manifiesto, para verificar reproducibilidad.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tpf.paths import DATA_SPLITS

PARTS = ("train", "val", "test")


def make_split(
    y, seed: int, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
) -> dict[str, np.ndarray]:
    """Devuelve {'train','val','test'} -> índices posicionales ordenados."""
    y = np.asarray(y)
    f_tr, f_va, f_te = fractions
    if abs(f_tr + f_va + f_te - 1.0) > 0.001:
        raise ValueError("las fracciones deben sumar 1")
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=f_va + f_te, stratify=y, random_state=seed)
    va, te = train_test_split(
        rest, test_size=f_te / (f_va + f_te), stratify=y[rest], random_state=seed
    )
    return {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}


def split_path(dataset: str, seed: int, root: str | Path | None = None) -> Path:
    return Path(root if root is not None else DATA_SPLITS) / f"{dataset}_seed{seed}.parquet"


def save_split(parts: dict[str, np.ndarray], dataset: str, seed: int, root=None) -> Path:
    df = pd.DataFrame(
        {
            "idx": np.concatenate([parts[p] for p in PARTS]).astype("int32"),
            "part": np.repeat(PARTS, [len(parts[p]) for p in PARTS]),
        }
    )
    path = split_path(dataset, seed, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_split(dataset: str, seed: int, root=None) -> dict[str, np.ndarray]:
    df = pd.read_parquet(split_path(dataset, seed, root))
    return {p: df.loc[df["part"] == p, "idx"].to_numpy(dtype="int64") for p in PARTS}


def index_hash(idx: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(idx, dtype="int64").tobytes()).hexdigest()


def summarize(parts: dict[str, np.ndarray], y) -> dict[str, dict]:
    """n, positivos y hash de índices por partición"""
    y = np.asarray(y)
    return {
        p: {
            "n": int(len(parts[p])),
            "n_pos": int(y[parts[p]].sum()),
            "sha256": index_hash(parts[p]),
        }
        for p in PARTS
    }


def row_hashes(X: pd.DataFrame) -> np.ndarray:
    """Hash por fila sobre las features (para detectar duplicados exactos)."""
    return pd.util.hash_pandas_object(X, index=False).to_numpy()


def duplicates_global(h: np.ndarray, y) -> dict[str, int]:
    """Filas que repiten otra fila anterior"""
    dup = pd.Series(h).duplicated(keep="first").to_numpy()
    return {"n_dup_rows": int(dup.sum()), "n_dup_pos": int(np.asarray(y)[dup].sum())}


def duplicates_cross(h: np.ndarray, parts: dict[str, np.ndarray]) -> dict[str, int]:
    """Filas de val/test con una copia exacta en train (fuga por duplicados)."""
    h_train = h[parts["train"]]
    return {p: int(np.isin(h[parts[p]], h_train).sum()) for p in ("val", "test")}
