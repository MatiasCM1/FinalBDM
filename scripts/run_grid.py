"""Runner parametrizable y resiliente para la grilla de experimentos de TabPFN.

Permite:
- Filtrar por datasets, semillas, estrategias y tamaños de contexto m.
- Pausar en cualquier momento con Ctrl+C sin perder corridas ya finalizadas.
- Omitir automáticamente corridas ya completadas (modo idempotente / cached).
- Auto-desconectar en Google Colab (--colab-auto-unassign) para no quemar créditos ociosos.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from tpf.data.splits import split_path
from tpf.evaluation.runner import run_once
from tpf.paths import CONFIGS, RESULTS


def make_predict_fn(cfg, n_estimators, batch_size, seed):
    """Crea la función de predicción con TabPFN-2.5 para una corrida específica."""

    def predict_fn(X_ctx, y_ctx, X_val, X_test, cat_cols):
        clf = None
        try:
            clf = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2_5,
                device="cuda",
                random_state=seed,
                categorical_features_indices=cat_cols,
                n_estimators=n_estimators,
                fit_mode=cfg.get("fit_mode", "low_memory"),
                memory_saving_mode=cfg.get("memory_saving_mode", "auto"),
                balance_probabilities=cfg.get("balance_probabilities", False),
            )
            clf.fit(X_ctx, y_ctx)

            def batches(X, classifier):
                pieces = []
                for start in range(0, len(X), batch_size):
                    block = X[start : start + batch_size]
                    p = classifier.predict_proba(block)[:, 1]
                    if len(p) != len(block) or not np.isfinite(p).all():
                        raise ValueError("TabPFN devolvió probabilidades inválidas o NaN")
                    pieces.append(p)
                return np.concatenate(pieces)

            p_val = batches(X_val, clf)
            p_test = batches(X_test, clf)
            return p_val, p_test

        finally:
            del clf
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return predict_fn


def run_experiments(args) -> dict[str, int]:
    load_dotenv()

    config_path = CONFIGS / "default.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg_tabpfn = config.get("tabpfn", {})

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS
    results_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_neg{args.max_eval_neg}" if args.max_eval_neg else ""
    model_id = f"tabpfn25_e{args.n_estimators}{suffix}"

    # Construir grilla completa a ejecutar
    plan: list[tuple[str, int, str, int]] = []
    for d in args.datasets:
        for s in args.seeds:
            sp = split_path(d, s)
            if not sp.exists():
                print(f"[ALERTA] No existe split para {d} seed {s} en {sp}. Se omitirá.")
                continue
            for m in args.sizes:
                for strat in args.strategies:
                    plan.append((d, s, strat, m))

    total_tasks = len(plan)
    if total_tasks == 0:
        print("[INFO] No hay tareas en la grilla seleccionada o faltan archivos de split.")
        return {"completed": 0, "cached": 0, "skipped": 0, "interrupted": 0}

    print("=" * 70)
    print("TPF - RUNNER DE EXPERIMENTOS TABPFN")
    print(f"Modelo: TabPFN-2.5 | n_estimators={args.n_estimators} | batch={args.batch_size}")
    print(f"Submuestreo: máx {args.max_eval_neg} negativos ponderados por partición")
    print(f"Salida: {results_dir.resolve()}")
    print(f"Total combinaciones seleccionadas: {total_tasks}")
    print("Tip: Podés pausar en cualquier momento con [Ctrl + C]. No se pierde nada.")
    print("=" * 70)

    if args.status_only:
        cached_count = 0
        for d, s, strat, m in plan:
            run_dir = results_dir / d / f"seed{s}" / f"{strat}_m{m}_{model_id}"
            if (run_dir / "run.json").exists() and (run_dir / "predictions.npz").exists():
                cached_count += 1
        pending = total_tasks - cached_count
        pct = cached_count / total_tasks * 100
        print(f"\n[ESTADO] Completadas: {cached_count} / {total_tasks} ({pct:.1f}%)")
        print(f"[ESTADO] Pendientes:   {pending} combinaciones por correr.")
        return {"completed": 0, "cached": cached_count, "skipped": 0, "interrupted": 0}

    if not torch.cuda.is_available():
        raise RuntimeError("No se detectó GPU CUDA disponible. Este modelo requiere GPU NVIDIA.")

    device_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 2**20
    print(f"GPU activa: {device_name} ({vram_total:.0f} MiB VRAM)\n")

    completed = 0
    cached = 0
    skipped = 0

    try:
        for idx, (dataset, seed, strategy, m) in enumerate(plan, 1):
            run_dir = results_dir / dataset / f"seed{seed}" / f"{strategy}_m{m}_{model_id}"
            info_file = run_dir / "run.json"
            pred_file = run_dir / "predictions.npz"

            if info_file.exists() and pred_file.exists():
                cached += 1
                print(
                    f"[{idx}/{total_tasks}] (YA LISTO) {dataset} | seed {seed} | {strategy} | m={m}"
                )
                continue

            print(
                f"\n[{idx}/{total_tasks} - {idx / total_tasks * 100:.1f}%] "
                f"CORRIENDO: {dataset} | seed {seed} | {strategy} | m={m} ...",
                flush=True,
            )

            predict_fn = make_predict_fn(cfg_tabpfn, args.n_estimators, args.batch_size, seed)

            t0 = time.perf_counter()
            result = run_once(
                dataset=dataset,
                seed=seed,
                strategy=strategy,
                m=m,
                model_id=model_id,
                predict_fn=predict_fn,
                results_dir=results_dir,
                max_eval_neg=args.max_eval_neg,
            )
            dt = time.perf_counter() - t0

            status = result.get("status", "unknown")
            vram_peak = torch.cuda.max_memory_reserved() / 2**20 if torch.cuda.is_available() else 0

            if status == "complete":
                completed += 1
                print(
                    f"   -> [OK] Completado en {dt:.1f} s ({dt / 60:.1f} min) "
                    f"| VRAM pico: {vram_peak:.0f} MiB"
                )
            elif status == "skipped":
                skipped += 1
                print(f"   -> [SALTEADO] {result.get('reason', 'Sin positivos suficientes')}")
            else:
                print(f"   -> [ESTADO] {status} en {dt:.1f} s")

    except KeyboardInterrupt:
        print("\n\n" + "!" * 70)
        print("[PAUSADO POR EL USUARIO]")
        print("Se presionó Ctrl+C. La corrida actual se canceló y la GPU se liberó.")
        print("Todas las corridas anteriores quedaron guardadas en disco sin pérdida.")
        print("Para continuar cuando quieras, volvé a ejecutar el script.")
        print("!" * 70 + "\n")
        return {"completed": completed, "cached": cached, "skipped": skipped, "interrupted": 1}

    print("\n" + "=" * 70)
    print("RESUMEN DE ESTA SESIÓN:")
    print(f"  - Nuevas corridas completadas: {completed}")
    print(f"  - Ya estaban completas (caché): {cached}")
    print(f"  - Salteadas (por regla):        {skipped}")
    print(f"  - Pendientes restantes:         {total_tasks - (completed + cached + skipped)}")
    print("=" * 70)

    if args.colab_auto_unassign:
        try:
            from google.colab import runtime

            print("\n[COLAB] Apagando el entorno automáticamente para no consumir más unidades...")
            runtime.unassign()
        except ImportError:
            pass

    return {"completed": completed, "cached": cached, "skipped": skipped, "interrupted": 0}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Runner modular y reanudable de la grilla experimental TabPFN."
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        choices=["ulb", "home_credit", "ieee_cis"],
        default=["ulb", "home_credit", "ieee_cis"],
        help="Datasets a correr.",
    )
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=list(range(8)),
        default=list(range(8)),
        help="Semillas (0 a 7).",
    )
    ap.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        choices=[1000, 5000, 10000],
        default=[1000, 5000, 10000],
        help="Tamaños de contexto m.",
    )
    ap.add_argument(
        "--strategies",
        nargs="+",
        choices=["uniform", "stratified", "balanced", "nearmiss", "iht"],
        default=["uniform", "stratified", "balanced", "nearmiss", "iht"],
        help="Estrategias de construcción de contexto.",
    )
    ap.add_argument(
        "--n-estimators",
        type=int,
        choices=[4, 8],
        default=4,
        help="Cantidad de estimadores TabPFN (4 recomendado para acelerar 2x).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Tamaño de lote de inferencia (5000 óptimo en hardware probado).",
    )
    ap.add_argument(
        "--max-eval-neg",
        type=int,
        default=10000,
        help="Tope de negativos en val/test para submuestreo ponderado (10000).",
    )
    ap.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directorio de resultados (útil para guardar directo en Google Drive).",
    )
    ap.add_argument(
        "--status-only",
        action="store_true",
        help="No corre nada; solo audita cuántas corridas faltan y cuántas están hechas.",
    )
    ap.add_argument(
        "--colab-auto-unassign",
        action="store_true",
        help="Desconecta el entorno de Google Colab al terminar para frenar el gasto de unidades.",
    )
    return ap


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_experiments(args)
