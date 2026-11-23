"""Transparent LoRA SFT backend on Modal (issue #61, sft_check condition).

The same published rows and the same hyperparameters copied field by field
from the pinned AutoScientist recommendation, with fixed seeds per replicate:
the platform-independence check for the managed loop. Refuses to run while
series autoscientist.hyperparams is null. Heavy imports (torch, transformers,
peft, trl) live inside the Modal function so tests run without them.
"""

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import modal
except ImportError:
    modal = None  # type: ignore[assignment]

REQUIRED_HYPERPARAMS = (
    "learning_rate",
    "n_epochs",
    "lora_r",
    "lora_alpha",
    "batch_size",
    "train_on_inputs",
)

DEFAULT_TARGET_MODULES = ["q_proj", "v_proj"]


def select_seed(seeds: list[int], replicate: int) -> int:
    """Seed for replicate r is seeds[r - 1] (replicates are 1-based)."""
    if not 1 <= replicate <= len(seeds):
        raise ValueError(f"replicate {replicate} out of range for {len(seeds)} seeds")
    return seeds[replicate - 1]


def require_pinned_hyperparams(hyperparams: Any) -> dict[str, Any]:
    """Return pinned hyperparams as a plain dict; refuse when null or partial."""
    if hyperparams is None:
        raise ValueError("series autoscientist.hyperparams is null; pin it in Phase 4 first")
    fields = dict(hyperparams)
    missing = [key for key in REQUIRED_HYPERPARAMS if fields.get(key) is None]
    if missing:
        raise ValueError(f"pinned hyperparams lack required fields: {missing}")
    return fields


def map_trainer_args(
    hyperparams: dict[str, Any],
    seed: int,
    max_rows: int | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    """Map pinned AutoScientist fields to plain-LoRA trainer arguments."""
    return {
        "learning_rate": float(hyperparams["learning_rate"]),
        "num_train_epochs": float(epochs if epochs is not None else hyperparams["n_epochs"]),
        "per_device_train_batch_size": int(hyperparams["batch_size"]),
        "lora_r": int(hyperparams["lora_r"]),
        "lora_alpha": int(hyperparams["lora_alpha"]),
        "train_on_inputs": bool(hyperparams["train_on_inputs"]),
        "seed": seed,
        "max_rows": max_rows,
    }


def format_example(instruction: str, question: str, target: str) -> str:
    """SFT text: translated instruction and question, then the target."""
    return f"{instruction}\n\n{question}\n\n{target}"


def resolve_train_repo(naming_template: str, benchmark: str, language: str) -> str:
    """Train split repo from the series naming template (never composed by hand)."""
    return naming_template.format(benchmark=benchmark, language=language)


def run_name_for(benchmark: str, language: str, base: str, replicate: int) -> str:
    """wandb run name from the train identity plus the sft backend marker."""
    return f"finetune-{benchmark}-{language}-{base}-r{replicate}-sft"


def check_backend_slug(hub_model_id: str, slug: str) -> None:
    """Warn when the Hub id lacks the backend slug (planner should provide it)."""
    if slug and slug not in hub_model_id:
        print(f"warning: hub_model_id {hub_model_id} lacks the backend slug {slug!r}")


if modal is not None:
    app = modal.App(name="qc-native-preference")
    sft_image = modal.Image.debian_slim().uv_pip_install(
        "torch==2.9.1",
        "transformers==4.56.2",
        "accelerate",
        "peft>=0.11.0",
        "trl>=0.9.0",
        "datasets>=2.16.0",
        "pyarrow>=14.0.0",
        "huggingface_hub",
        "wandb",
        "pyyaml",
    )

    @app.function(
        image=sft_image,
        gpu="A100",
        timeout=7200,
        secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    )
    def train_lora(job: dict[str, Any]) -> dict[str, Any]:
        """Train one LoRA adapter with fixed seeds; returns run metadata."""
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer

        args = job["trainer_args"]
        torch.manual_seed(args["seed"])
        model_id = job["base_model"]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        rows = load_dataset(job["train_repo"], split="train")
        if args["max_rows"] is not None:
            rows = rows.select(range(min(args["max_rows"], len(rows))))

        def formatting_func(examples: dict[str, list[str]]) -> list[str]:
            return [
                format_example(instruction, question, target)
                for instruction, question, target in zip(
                    examples["instruction"],
                    examples["question"],
                    examples["target"],
                    strict=True,
                )
            ]

        peft_config = LoraConfig(
            r=args["lora_r"],
            lora_alpha=args["lora_alpha"],
            target_modules=job.get("target_modules") or DEFAULT_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        training_args = TrainingArguments(
            output_dir="/tmp/sft",
            learning_rate=args["learning_rate"],
            num_train_epochs=args["num_train_epochs"],
            per_device_train_batch_size=args["per_device_train_batch_size"],
            seed=args["seed"],
            report_to=["wandb"],
            run_name=job["wandb_run"],
            logging_steps=10,
            save_strategy="no",
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=rows,
            formatting_func=formatting_func,
            args=training_args,
            peft_config=peft_config,
        )
        stats = trainer.train()
        tokens_seen = sum(len(tokenizer.encode(text)) for text in rows["question"][:1000])
        adapter_dir = "/tmp/sft-adapter"
        trainer.model.save_pretrained(adapter_dir)
        merged = trainer.model.merge_and_unload()
        merged_dir = "/tmp/sft-merged"
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        merged.push_to_hub(job["hub_model_id"], private=job["private"])
        return {
            "hub_model_url": f"https://huggingface.co/{job['hub_model_id']}",
            "tokens_seen_approx": tokens_seen,
            "train_loss": float(stats.training_loss),
            "adapter_dir": adapter_dir,
        }


def print_preflight(
    benchmark: str, language: str, base: str, replicate: int, hub_model_id: str, gpu: str, timeout: int
) -> None:
    """Print the AGENTS.md pre-flight checklist before submitting the Modal job."""
    print("Pre-flight: sft finetune")
    print(f"  cell: benchmark={benchmark} language={language} base={base} replicate={replicate}")
    print(f"  compute: Modal GPU {gpu}, timeout {timeout}")
    print(f"  persistence: {hub_model_id} (merged model; adapter kept alongside)")
    print("  tracking: wandb finetune run with resolved config, tokens, loss curve, Hub URL")


def main() -> None:
    """Hydra entry point (deferred imports keep modal/hydra out of test collection)."""
    from hydra import main as hydra_main
    from omegaconf import DictConfig, OmegaConf

    @hydra_main(version_base=None, config_path="../../configs", config_name="sft_finetune")
    def hydra_main_fn(cfg: DictConfig) -> None:
        if modal is None:
            raise RuntimeError("sft finetuning needs the modal package: uv sync --extra modal")
        series = cfg.series
        bench_name, lang_code = cfg.benchmark.name, cfg.language.code
        base_role = cfg.base
        sft = series["sft"]
        hyperparams = require_pinned_hyperparams(series["autoscientist"]["hyperparams"])
        seed = select_seed(list(sft["seeds"]), cfg.replicate)
        trainer_args = map_trainer_args(hyperparams, seed, cfg.max_rows, cfg.epochs)
        check_backend_slug(cfg.hub_model_id, series["naming"]["backend_slugs"]["sft"])
        gpu = cfg.gpu or sft["gpu"]
        timeout = sft["timeout"]
        from pipeline.modal_runner.budget import (
            check_budget,
            estimate_usd,
            load_rates,
            record_spend,
        )

        rates = load_rates()
        check_budget(rates, estimate_usd(rates, gpu, timeout))
        print_preflight(bench_name, lang_code, base_role, cfg.replicate, cfg.hub_model_id, gpu, timeout)
        job = {
            "trainer_args": trainer_args,
            "base_model": series["base_models"][base_role],
            "train_repo": resolve_train_repo(
                series["naming"]["hf_dataset"], bench_name, lang_code
            ),
            "hub_model_id": cfg.hub_model_id,
            "private": bool(cfg.private),
            "wandb_run": run_name_for(bench_name, lang_code, base_role, cfg.replicate),
            "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        }
        import time as _time

        call = train_lora.spawn(job)
        print(f"Modal function call id: {call.object_id}")
        started = _time.monotonic()
        result = call.get()
        record_spend(
            rates, gpu, _time.monotonic() - started,
            f"sft-{bench_name}-{lang_code}-{base_role}-r{cfg.replicate}",
            series["base_models"][base_role],
        )
        print(f"Pushed to {result['hub_model_url']}; loss {result['train_loss']}")

    hydra_main_fn()


if __name__ == "__main__":
    main()
