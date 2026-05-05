"""Training pipeline for preference tuning (DPO / SFT / ORPO)."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def run_training_pipeline(
    config: dict,
    dataset_path: Optional[str] = None,
) -> None:
    """Run the full training pipeline using TRL's DPOTrainer (or SFT/ORPO).

    Args:
        config: Full project configuration dictionary.
        dataset_path: Optional override for the dataset path / HF dataset name.
    """
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    from src.models import load_base_model_and_tokenizer
    from src.utils import set_seed, ensure_dir

    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    lora_cfg = config.get("lora", {})

    set_seed(data_cfg.get("seed", 42))

    # Load model and tokenizer
    model, tokenizer = load_base_model_and_tokenizer(
        model_name=model_cfg.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct"),
        torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
        model_max_length=model_cfg.get("model_max_length", 2048),
        use_lora=lora_cfg.get("use_lora", True),
        lora_config=lora_cfg,
    )

    # Load dataset
    ds_name = dataset_path or data_cfg.get("dataset_name")
    logger.info("Loading dataset: %s", ds_name)
    dataset = load_dataset(ds_name)

    output_dir = ensure_dir(train_cfg.get("output_dir", "outputs/model"))
    method = train_cfg.get("method", "dpo").lower()

    if method == "dpo":
        _run_dpo(model, tokenizer, dataset, train_cfg, data_cfg, output_dir)
    elif method == "sft":
        _run_sft(model, tokenizer, dataset, train_cfg, data_cfg, output_dir)
    else:
        raise ValueError(f"Unsupported training method: {method}. Choose from: dpo, sft.")


def _run_dpo(model, tokenizer, dataset, train_cfg: dict, data_cfg: dict, output_dir: Path) -> None:
    """Train using Direct Preference Optimization."""
    from trl import DPOConfig, DPOTrainer

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 5e-6),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_steps=train_cfg.get("eval_steps", 100),
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        beta=train_cfg.get("beta", 0.1),
        report_to=train_cfg.get("report_to", "wandb"),
        run_name=train_cfg.get("run_name", None),
        max_prompt_length=data_cfg.get("max_prompt_length", 512),
        max_length=data_cfg.get("max_length", 1024),
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset[data_cfg.get("train_split", "train")],
        eval_dataset=dataset.get(data_cfg.get("eval_split", "test")),
        tokenizer=tokenizer,
    )
    logger.info("Starting DPO training …")
    trainer.train()
    trainer.save_model(str(output_dir))
    logger.info("DPO training complete. Model saved to %s", output_dir)


def _run_sft(model, tokenizer, dataset, train_cfg: dict, data_cfg: dict, output_dir: Path) -> None:
    """Train using Supervised Fine-Tuning on chosen responses."""
    from trl import SFTConfig, SFTTrainer

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 100),
        report_to=train_cfg.get("report_to", "wandb"),
        max_seq_length=data_cfg.get("max_length", 1024),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset[data_cfg.get("train_split", "train")],
        tokenizer=tokenizer,
    )
    logger.info("Starting SFT training …")
    trainer.train()
    trainer.save_model(str(output_dir))
    logger.info("SFT training complete. Model saved to %s", output_dir)


__all__ = ["run_training_pipeline"]
