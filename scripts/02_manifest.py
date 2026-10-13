"""Regenera data/MANIFEST.md con tamaño y SHA-256 de cada archivo en data/raw/."""

import datetime as dt
import hashlib

from tpf.paths import DATA_RAW, ROOT

SOURCES = {
    "creditcard.csv": "Kaggle: mlg-ulb/creditcardfraud (D1)",
    "application_train.csv": "Kaggle: home-credit-default-risk (D0)",
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


DESCARGA = {
    "creditcard.csv": "2026-10-13",
    "application_train.csv": "2026-10-13",
}

lines = [
    "# Manifiesto de datos",
    "",
    "SHA-256, tamaño y fecha de descarga de cada archivo en `data/raw/`.",
    "",
    "| Archivo | Fuente | Tamaño (bytes) | SHA-256 | Descargado | Modif. del archivo |",
    "|---|---|---|---|---|---|",
]
for p in sorted(DATA_RAW.glob("*.csv")):
    mtime = dt.datetime.fromtimestamp(p.stat().st_mtime).date().isoformat()
    lines.append(
        f"| {p.name} | {SOURCES.get(p.name, '?')} | {p.stat().st_size} "
        f"| `{sha256(p)}` | {DESCARGA.get(p.name, '?')} | {mtime} |"
    )
(ROOT / "data" / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
