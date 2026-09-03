@AGENTS.md

## Claude Code specifics

### Skill routing

Invoke the matching project skill before starting work on a stage:

| Task | Skill |
|---|---|
| Start a new experiment (new language, benchmark, condition or series) | `new-experiment` |
| Create, translate, or enhance a dataset; build preference pairs; publish datasets | `datagen` |
| Launch or monitor AutoScientist finetuning; publish models | `finetune` |
| Run the transparent LoRA backend on Modal (condition `sft_check`, `pipeline/training/sft_finetune.py`) | `finetune` for the recipe constants and naming, then `compute` for the Modal job |
| Write or run Inspect AI evals; judge setup; eval sweeps | `evaluate` |
| Decide a tier, park or promote a benchmark or language, amend a pre-registered constant | `new-experiment` (tiers and amendments) |
| Modal GPU jobs or Lightning AI compute/inference | `compute` |

The `metr-*` skills are productivity-study tooling, unrelated to the research workflow.

### Agent-readable docs

When skill content isn't enough, these endpoints serve markdown directly:

- Adaption: `https://docs.adaptionlabs.ai/<path>/index.md` (e.g. `/autoscientist/overview/index.md`; root index at `/index.md`)
- Inspect AI: `https://inspect.aisi.org.uk/<page>.html.md` (index: `/llms.txt`)
- Modal: `https://modal.com/llms.txt`
- wandb: `https://docs.wandb.ai/llms.txt`

Verify Adaption SDK behavior against the installed package (`.venv/lib/python3.10/site-packages/adaption/`) since the SDK evolves quickly.
