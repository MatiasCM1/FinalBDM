"""Genera las particiones 60/20/20 estratificadas (solo índices)"""

from __future__ import annotations

import argparse
import json

import yaml

from tpf.data.loaders import SPECS, load_dataset
from tpf.data.splits import (
    duplicates_cross,
    duplicates_global,
    make_split,
    row_hashes,
    save_split,
    summarize,
)
from tpf.paths import CONFIGS, DATA_SPLITS


def main() -> None:

    cfg = yaml.safe_load((CONFIGS / "default.yaml").read_text(encoding="utf-8"))
    available = [d for d in cfg["datasets"] if d in SPECS]

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=available, choices=available)
    args = ap.parse_args()

    fr = (cfg["split"]["train"], cfg["split"]["val"], cfg["split"]["test"])
    manifest: dict = {"split": cfg["split"], "seeds": cfg["seeds"], "datasets": {}}

    for name in args.datasets:
        X, y = load_dataset(name)
        h = row_hashes(X)
        entry = {
            "n_rows": int(len(y)),
            "n_pos": int(y.sum()),
            "dup_global": duplicates_global(h, y),
            "by_seed": {},
        }
        print(
            f"\n== {name}: {entry['n_rows']} filas, {entry['n_pos']} positivos, "
            f"duplicados exactos (2.ª copia en adelante): {entry['dup_global']}"
        )

        for seed in cfg["seeds"]:
            parts = make_split(y, seed, fr)
            save_split(parts, name, seed)
            info = summarize(parts, y)
            info["dup_cross"] = duplicates_cross(h, parts)  # val/test con copia exacta en train
            entry["by_seed"][str(seed)] = info
            print(
                f"seed {seed}: "
                + " | ".join(
                    f"{p} {info[p]['n']}/{info[p]['n_pos']}" for p in ("train", "val", "test")
                )
                + f" | dup val/test en train: {info['dup_cross']}"
            )
        manifest["datasets"][name] = entry

    out = DATA_SPLITS / "MANIFEST_splits.json"
    # Guardar manifiesto, actualizando datasets previos si existían
    if out.exists():
        prev = json.loads(out.read_text(encoding="utf-8"))
        prev["datasets"].update(manifest["datasets"])
        manifest["datasets"] = prev["datasets"]
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifiesto: {out}")


if __name__ == "__main__":
    main()
