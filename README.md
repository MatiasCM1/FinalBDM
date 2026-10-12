# TPF — Remuestreo o umbral: construcción de contexto para TFMs bajo desbalance extremo

Trabajo Final Integrador, *Bases de Datos Masivas* (UNLu). Autor: Sebastián Juárez.

**Pregunta de investigación.** ¿Cómo interactúan la estrategia de construcción del contexto de TabPFN
(Uniform, Stratified, Balanced, NearMiss, IHT; m = 1k/5k/10k) y la regla de umbral
(R0 = 0,5, R1 = π_ctx, R2 = óptimo en validación por MCC) bajo desbalance extremo
(Home Credit, ULB 0,17 %, IEEE-CIS)?

## Puesta en marcha

Requisitos: Python 3.10–3.13 (recomendado 3.11), git, driver NVIDIA actualizado si hay GPU.

| | Linux / macOS | Windows (PowerShell) |
|---|---|---|
| Instalar | `bash scripts/setup.sh` | `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1` |
| Activar | `source .venv/bin/activate` | `.\.venv\Scripts\Activate.ps1` |
| Sin GPU | `TORCH_INDEX=cpu bash scripts/setup.sh` | `... setup.ps1 -TorchIndex cpu` |
| Otra CUDA | `TORCH_INDEX=cu126 bash scripts/setup.sh` | `... setup.ps1 -TorchIndex cu126` |
| Verificar | `python scripts/check_env.py` | `python scripts\check_env.py` |
| Smoke TabPFN-2.5 | `python scripts/check_env.py --tabpfn` | igual |
| Tests | `make test` | `python -m pytest -m "not slow and not gpu"` |
| Lint | `make lint` | `python -m ruff check src tests scripts` |

Después: copiar `.env.example` a `.env` (el setup lo hace) y completar `TABPFN_TOKEN`
(aceptar la licencia de TabPFN-2.5 en <https://ux.priorlabs.ai>).

## Estructura

```
configs/      parámetros del protocolo (semillas, splits, m, reglas, TabPFN)
data/         raw/ (no versionado) · interim/ · splits/ (índices, manifiestos) · MANIFEST.md
src/tpf/      código reutilizable: data/ contexts/ models/ evaluation/ paths.py
scripts/      puntos de entrada numerados (01_make_splits.py, ...) + setup + check_env
tests/        pytest (anti-fuga, splits, contextos, métricas)
results/      probabilidades y logs de corridas (no versionado)
reports/      figuras y tablas del informe
docs/         protocolo, decisiones, notas de hardware
notebooks/    exploración (no es parte del pipeline)
```

## Reglas del protocolo (resumen)

Orden inmutable: split → ajuste de preprocesadores solo con train → contexto desde train →
inferencia en val/test → reglas de umbral → métricas. Semillas 0–7. `ModelVersion.V2_5` fijo.
Detalle en `docs/protocolo.md`.
