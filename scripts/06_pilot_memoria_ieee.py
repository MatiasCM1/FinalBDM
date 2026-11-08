"""Piloto de memoria para IEEE-CIS

Usa el runner real: split, preprocesamiento de train, contexto, inferencia en
lotes, submuestreo ponderado de val/test y evaluación R0/R1/R2.

Ejemplos de ejecucion:
    python scripts/06_pilot_memoria_ieee.py
    python scripts/06_pilot_memoria_ieee.py --sizes 10000 --strategies balanced
    python scripts/06_pilot_memoria_ieee.py --sizes 10000 --strategies nearmiss iht
"""

from __future__ import annotations

import argparse
import csv
import gc
import tempfile
import threading
import time
from datetime import datetime

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from tpf.data.splits import split_path
from tpf.evaluation.runner import run_once
from tpf.paths import CONFIGS, RESULTS

try:
    import psutil
except ImportError:
    psutil = None

DATASET = "ieee_cis"
SEED = 0
BATCH_SIZE = 5_000
MAX_EVAL_NEG = 10_000


class RamMonitor:
    """Mide RAM del proceso y memoria libre del sistema mientras corre el piloto."""

    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None
        self.peak_rss = 0
        self.min_available = float("inf")

    def sample(self):
        if psutil is not None:
            self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
            self.min_available = min(self.min_available, psutil.virtual_memory().available)

    def start(self):
        if psutil is not None:
            self.sample()
            self.thread = threading.Thread(target=self.watch, daemon=True)
            self.thread.start()

    def watch(self):
        while not self.stop_event.wait(0.1):
            self.sample()

    def finish(self):
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()
            self.sample()


def gpu_measure(name, fn, stats):
    """Ejecuta una operación y registra sus picos de VRAM PyTorch."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    try:
        return fn()
    finally:
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass

        stats[f"{name}_s"] = round(time.perf_counter() - start, 2)
        stats[f"{name}_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        stats[f"{name}_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)


def make_predict_fn(cfg, n_estimators, batch_size, seed, stats):
    """Predicción igual a la corrida real, con mediciones durante fit e inferencia."""

    def predict_fn(X_ctx, y_ctx, X_val, X_test, cat_cols):
        clf = None

        try:

            def create_and_fit():
                classifier = TabPFNClassifier.create_default_for_version(
                    ModelVersion.V2_5,
                    device="cuda",
                    random_state=seed,
                    categorical_features_indices=cat_cols,
                    n_estimators=n_estimators,
                    fit_mode=cfg["fit_mode"],
                    memory_saving_mode=cfg["memory_saving_mode"],
                    balance_probabilities=cfg["balance_probabilities"],
                )
                classifier.fit(X_ctx, y_ctx)
                return classifier

            clf = gpu_measure("fit", create_and_fit, stats)

            def predict_batches(X, classifier):
                pieces = []

                for start in range(0, len(X), batch_size):
                    block = X[start : start + batch_size]
                    probs = classifier.predict_proba(block)[:, 1]

                    if len(probs) != len(block) or not np.isfinite(probs).all():
                        raise ValueError("TabPFN devolvió probabilidades inválidas")

                    pieces.append(probs)

                return np.concatenate(pieces)

            p_val = gpu_measure("val", lambda: predict_batches(X_val, clf), stats)
            p_test = gpu_measure("test", lambda: predict_batches(X_test, clf), stats)

            return p_val, p_test

        finally:
            clf = None
            gc.collect()
            torch.cuda.empty_cache()

    return predict_fn


def append_log(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))

        if f.tell() == 0:
            writer.writeheader()

        writer.writerow(row)


def main():
    load_dotenv()

    config = yaml.safe_load((CONFIGS / "default.yaml").read_text(encoding="utf-8"))
    cfg = config["tabpfn"]

    if cfg["model_version"] != "v2.5" or cfg["balance_probabilities"]:
        raise ValueError("El piloto requiere TabPFN-2.5 y balance_probabilities=False")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        choices=config["context"]["sizes"],
        default=config["context"]["sizes"],
    )
    ap.add_argument(
        "--strategies",
        nargs="+",
        choices=config["context"]["strategies"],
        default=["uniform", "balanced"],
    )
    ap.add_argument(
        "--n-estimators",
        type=int,
        choices=(4, 8),
        default=cfg["n_estimators"],
    )
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-eval-neg", type=int, default=MAX_EVAL_NEG)

    args = ap.parse_args()

    if args.batch_size <= 0 or args.max_eval_neg <= 0:
        ap.error("--batch-size y --max-eval-neg deben ser positivos")

    if not torch.cuda.is_available():
        raise RuntimeError("No se detectó CUDA. Revisar python scripts/check_env.py")

    if not split_path(DATASET, SEED).exists():
        raise FileNotFoundError("Falta el split: ejecutar 01_make_splits.py --datasets ieee_cis")

    RESULTS.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "pilot_ieee_memoria.csv"

    total_vram = round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1)

    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM total: {total_vram:.0f} MiB")
    print(f"TabPFN-2.5, n_estimators={args.n_estimators}, batch={args.batch_size}")
    print(f"Evaluación: todos los positivos + máximo {args.max_eval_neg} negativos por partición")

    if psutil is None:
        print("[AVISO] Para medir también la RAM, instalar: python -m pip install psutil")

    # Carpeta temporal distinta en cada ejecución: nunca usa una corrida cacheada.
    with tempfile.TemporaryDirectory(prefix="ieee_pilot_", dir=RESULTS) as temp_dir:
        for m in args.sizes:
            for strategy in args.strategies:
                print(f"\n== {DATASET} / {strategy} / m={m} ==", flush=True)

                gc.collect()
                torch.cuda.empty_cache()

                free_before = round(torch.cuda.mem_get_info()[0] / 2**20, 1)
                stats = {}
                monitor = RamMonitor()
                start = time.perf_counter()
                status, error = "complete", ""
                n_val, n_test = 0, 0

                try:
                    monitor.start()

                    predict_fn = make_predict_fn(
                        cfg, args.n_estimators, args.batch_size, SEED, stats
                    )

                    result = run_once(
                        DATASET,
                        SEED,
                        strategy,
                        m,
                        f"mem_e{args.n_estimators}_b{args.batch_size}_neg{args.max_eval_neg}",
                        predict_fn,
                        results_dir=temp_dir,
                        max_eval_neg=args.max_eval_neg,
                    )

                    status = result["status"]

                    if status != "complete":
                        raise RuntimeError(f"El runner devolvió: {status}")

                    with np.load(result["path"] + "/predictions.npz") as pred:
                        n_val = len(pred["idx_val"])
                        n_test = len(pred["idx_test"])

                        if "w_val" not in pred or "w_test" not in pred:
                            raise ValueError("Faltan los pesos de evaluación")

                except Exception as exc:  # noqa: BLE001 - registrar errores del piloto
                    status, error = "error", f"{type(exc).__name__}: {exc}"

                finally:
                    monitor.finish()

                row = {
                    "fecha": datetime.now().isoformat(timespec="seconds"),
                    "dataset": DATASET,
                    "seed": SEED,
                    "strategy": strategy,
                    "m": m,
                    "n_estimators": args.n_estimators,
                    "batch_size": args.batch_size,
                    "max_eval_neg": args.max_eval_neg,
                    "n_val": n_val,
                    "n_test": n_test,
                    "vram_total_mib": total_vram,
                    "vram_free_before_mib": free_before,
                    "fit_allocated_mib": stats.get("fit_allocated_mib", ""),
                    "fit_reserved_mib": stats.get("fit_reserved_mib", ""),
                    "val_reserved_mib": stats.get("val_reserved_mib", ""),
                    "test_reserved_mib": stats.get("test_reserved_mib", ""),
                    "ram_rss_peak_gib": (
                        round(monitor.peak_rss / 2**30, 2) if psutil is not None else ""
                    ),
                    "ram_available_min_gib": (
                        round(monitor.min_available / 2**30, 2) if psutil is not None else ""
                    ),
                    "total_seconds": round(time.perf_counter() - start, 1),
                    "status": status,
                    "error": error,
                }

                append_log(log_path, row)

                peak = max(
                    (float(v) for k, v in row.items() if k.endswith("reserved_mib") and v != ""),
                    default=0.0,
                )

                print(f"Estado: {status} | VRAM reservada pico: {peak:.0f}/{total_vram:.0f} MiB")
                print(
                    f"RAM pico del proceso: {row['ram_rss_peak_gib']} GiB "
                    f"| RAM libre mínima: {row['ram_available_min_gib']} GiB"
                )
                print(
                    f"Filas inferidas: val={n_val}, test={n_test} | tiempo={row['total_seconds']} s"
                )

                if error:
                    print(f"[ERROR] {error}")
                    print("Se detiene el piloto para no continuar después de un error.")
                    print(f"Registro parcial: {log_path}")
                    return

    print(f"\n[OK] Piloto terminado. Registro: {log_path}")
    print("Éxito en el piloto NO garantiza que la evaluación completa use la misma RAM.")


if __name__ == "__main__":
    main()
