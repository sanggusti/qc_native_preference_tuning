"""Training configuration utilities."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""

    use_lora: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    @classmethod
    def from_dict(cls, cfg: dict) -> "LoRAConfig":
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""

    method: str = "dpo"
    output_dir: str = "outputs/model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5.0e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    bf16: bool = True
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    beta: float = 0.1
    report_to: str = "wandb"
    run_name: Optional[str] = None

    @classmethod
    def from_dict(cls, cfg: dict) -> "TrainingConfig":
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


__all__ = ["TrainingConfig", "LoRAConfig"]
