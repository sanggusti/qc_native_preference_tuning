"""Model loading and management utilities."""

import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def load_base_model_and_tokenizer(
    model_name: str,
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_2",
    model_max_length: int = 2048,
    device_map: str = "auto",
    use_lora: bool = False,
    lora_config: Optional[dict] = None,
):
    """Load a base causal language model and its tokenizer.

    Args:
        model_name: HuggingFace model identifier or local path.
        torch_dtype: Dtype string for the model weights ('bfloat16', 'float16', 'float32').
        attn_implementation: Attention backend ('flash_attention_2', 'sdpa', 'eager').
        model_max_length: Maximum sequence length for the tokenizer.
        device_map: Device placement strategy (e.g., 'auto', 'cuda').
        use_lora: Whether to wrap the model with LoRA adapters.
        lora_config: Dictionary of LoRA configuration parameters.

    Returns:
        Tuple of (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    logger.info("Loading tokenizer from %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=model_max_length,
        padding_side="right",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model from %s (dtype=%s)", model_name, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
        trust_remote_code=True,
    )

    if use_lora and lora_config:
        model = _apply_lora(model, lora_config)

    return model, tokenizer


def _apply_lora(model, lora_config: dict):
    """Apply LoRA adapters to a model.

    Args:
        model: The base model to wrap.
        lora_config: Dictionary with LoRA hyperparameters.

    Returns:
        Model wrapped with LoRA.
    """
    from peft import LoraConfig, get_peft_model, TaskType

    config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=lora_config.get("target_modules", None),
        bias=lora_config.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def save_model(
    model,
    tokenizer,
    output_dir: str | Path,
    merge_lora: bool = False,
) -> None:
    """Save model and tokenizer to disk.

    Args:
        model: Model to save.
        tokenizer: Tokenizer to save.
        output_dir: Directory to save to.
        merge_lora: If True and model has LoRA adapters, merge and unload before saving.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if merge_lora:
        try:
            model = model.merge_and_unload()
            logger.info("Merged LoRA weights into base model.")
        except AttributeError:
            logger.warning("merge_and_unload not available — saving adapter weights only.")

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Model and tokenizer saved to %s", output_dir)


__all__ = ["load_base_model_and_tokenizer", "save_model"]
