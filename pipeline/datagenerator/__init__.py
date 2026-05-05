"""Data generation pipeline for preference tuning datasets."""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = "You are a helpful assistant."

_TOPIC_INSTRUCTION_TEMPLATES: Dict[str, str] = {
    "general_knowledge": "Answer the following general knowledge question:\n{question}",
    "coding": "Write code to solve the following problem:\n{question}",
    "math": "Solve the following math problem step by step:\n{question}",
    "writing": "Write a short text on the following topic:\n{question}",
    "reasoning": "Think through the following problem carefully:\n{question}",
}


def generate_preference_sample(
    prompt: str,
    client,
    generator_model: str,
    temperature: float = 0.8,
    max_tokens: int = 1024,
) -> Optional[Dict[str, str]]:
    """Generate a chosen/rejected pair for a given prompt.

    A higher-temperature sample is used as 'rejected' and a lower-temperature
    sample is used as 'chosen'.

    Args:
        prompt: The user prompt.
        client: OpenAI-compatible API client.
        generator_model: Model name for generation.
        temperature: Base temperature for chosen response.
        max_tokens: Maximum tokens per response.

    Returns:
        Dictionary with 'prompt', 'chosen', 'rejected' keys or None on error.
    """
    try:
        chosen_resp = client.chat.completions.create(
            model=generator_model,
            temperature=max(0.0, temperature - 0.4),
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        rejected_resp = client.chat.completions.create(
            model=generator_model,
            temperature=min(1.5, temperature + 0.4),
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return {
            "prompt": prompt,
            "chosen": chosen_resp.choices[0].message.content.strip(),
            "rejected": rejected_resp.choices[0].message.content.strip(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to generate sample for prompt '%s': %s", prompt[:80], exc)
        return None


def run_data_generation_pipeline(
    config: dict,
    prompts: List[str],
    output_path: str | Path,
    client=None,
) -> List[Dict[str, Any]]:
    """Run the full data generation pipeline.

    Args:
        config: Data generator configuration dictionary.
        prompts: List of user prompts to generate responses for.
        output_path: Path to save the generated JSONL dataset.
        client: Optional pre-constructed API client.

    Returns:
        List of generated preference samples.
    """
    if client is None:
        from openai import OpenAI
        client = OpenAI()

    gen_cfg = config.get("datagenerator", config)
    generator_model = gen_cfg.get("generator_model", "gpt-4o-mini")
    temperature = gen_cfg.get("temperature", 0.8)
    max_tokens = gen_cfg.get("max_tokens", 1024)

    samples: List[Dict[str, Any]] = []
    for i, prompt in enumerate(prompts):
        logger.info("Generating sample %d / %d", i + 1, len(prompts))
        sample = generate_preference_sample(
            prompt=prompt,
            client=client,
            generator_model=generator_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if sample is not None:
            samples.append(sample)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info("Saved %d samples to %s", len(samples), output_path)
    return samples


__all__ = ["generate_preference_sample", "run_data_generation_pipeline"]
