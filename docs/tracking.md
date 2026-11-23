# Experiment tracking: wandb plus Weave

All stages log through `src/utils/tracking.py`. Tracking never breaks a run:
without credentials, network, or the packages, every call warns once and
continues. Set `WANDB_MODE=offline` to queue locally and `wandb sync` later
(useful in sandboxes, where wandb-core cannot bind its local socket).

## What goes where

| Content | Where | How |
|---|---|---|
| Run config, scalar metrics | wandb run | `init_run` + `log_metrics` |
| Per-sample scores | wandb Table | `log_table` (capped at 10k rows) |
| Figures (analysis PNGs) | wandb Images | `log_figure` |
| Datasets, models, reports | wandb Artifacts | `log_artifact` |
| Stage lineage (translate, gate, finetune, eval) | Weave calls | `log_call` with summarized IO |

One wandb run exists per stage run (datagen dataset, finetune cell, eval cell,
covariate script); Weave calls link them into a sweep trace. Naming follows
AGENTS.md (`{stage}-{benchmark}-...`, tags for benchmark, language, condition,
base role, series).

## Modal jobs

Modal functions read the `wandb` cloud secret (`modal.Secret.from_name("wandb")`;
create it with `WANDB_API_KEY` inside). GPU serving jobs log nothing; training
and translation jobs log through this module with the resolved config.

## Boards to build in the UI

- Sweep table: group eval runs by benchmark and language, compare accuracy.
- Samples table per cell: filter failures by `leak_id`, cap hits, language label.
- Figures: forest plots and exposure scatters logged from `src/analysis/report.py`.
- Weave calls view: trace any score back to its dataset, model, and config.
