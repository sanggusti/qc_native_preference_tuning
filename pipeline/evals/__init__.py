"""Evaluation pipeline for preference-tuned models."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evals import EvalSample, compute_win_rate
from src.evals.tasks import LLMJudgeEvaluator
from src.utils import batch_iterable

logger = logging.getLogger(__name__)


def run_eval_pipeline(
    config: dict,
    model_path: str,
    baseline_model_path: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    client=None,
) -> Dict[str, Any]:
    """Run the full evaluation pipeline for a preference-tuned model.

    Generates responses from the tuned model and an optional baseline model,
    then uses an LLM judge to compare them.

    Args:
        config: Full project configuration dictionary.
        model_path: Path or HF identifier of the tuned model to evaluate.
        baseline_model_path: Optional path/identifier of a baseline model to compare against.
            Defaults to the base model in config if not provided.
        output_dir: Directory to write evaluation results. Defaults to outputs/evals.
        client: Optional pre-constructed OpenAI client.

    Returns:
        Dictionary with evaluation metrics.
    """
    from datasets import load_dataset
    from src.models import load_base_model_and_tokenizer
    from src.utils import set_seed

    eval_cfg = config.get("evals", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    set_seed(42)

    if client is None:
        from openai import OpenAI
        client = OpenAI()

    output_dir = Path(output_dir or "outputs/evals")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluation dataset
    ds_name = data_cfg.get("dataset_name")
    eval_split = data_cfg.get("eval_split", "test")
    logger.info("Loading eval dataset %s[%s]", ds_name, eval_split)
    dataset = load_dataset(ds_name, split=eval_split)

    num_samples = eval_cfg.get("num_eval_samples", 200)
    if num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    # Generate responses from tuned model
    tuned_responses = _generate_responses(
        prompts=[ex[data_cfg.get("prompt_column", "prompt")] for ex in dataset],
        model_name=model_path,
        model_cfg=model_cfg,
        batch_size=eval_cfg.get("batch_size", 8),
    )

    # Use existing chosen responses (or generate from baseline) as comparison
    baseline_responses: List[str]
    if baseline_model_path:
        baseline_responses = _generate_responses(
            prompts=[ex[data_cfg.get("prompt_column", "prompt")] for ex in dataset],
            model_name=baseline_model_path,
            model_cfg=model_cfg,
            batch_size=eval_cfg.get("batch_size", 8),
        )
    else:
        baseline_responses = [
            ex[data_cfg.get("chosen_column", "chosen")] for ex in dataset
        ]

    # Build eval samples and run judge
    prompts = [ex[data_cfg.get("prompt_column", "prompt")] for ex in dataset]
    eval_samples = [
        EvalSample(
            prompt=p,
            response_a=tuned,
            response_b=base,
            language=ex.get("language", "en"),
        )
        for p, tuned, base, ex in zip(prompts, tuned_responses, baseline_responses, dataset)
    ]

    evaluator = LLMJudgeEvaluator(config=eval_cfg, client=client)
    results = evaluator.evaluate_batch(eval_samples)

    win_rate = compute_win_rate(results, model_is_a=True)
    metrics = {"win_rate": win_rate, "num_samples": len(results)}
    logger.info("Evaluation complete. Win rate: %.3f (%d samples)", win_rate, len(results))

    # Save results
    results_path = output_dir / "results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "prompt": r.prompt,
                        "response_a": r.response_a,
                        "response_b": r.response_b,
                        "winner": r.winner,
                        "reasoning": r.reasoning,
                        "language": r.language,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Results saved to %s", output_dir)
    return metrics


def _generate_responses(
    prompts: List[str],
    model_name: str,
    model_cfg: dict,
    batch_size: int = 8,
) -> List[str]:
    """Generate responses from a model for a list of prompts.

    Args:
        prompts: Input prompts.
        model_name: Model path or HF identifier.
        model_cfg: Model configuration dictionary.
        batch_size: Number of prompts per batch.

    Returns:
        List of generated response strings.
    """
    import torch
    from src.models import load_base_model_and_tokenizer

    model, tokenizer = load_base_model_and_tokenizer(
        model_name=model_name,
        torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
        model_max_length=model_cfg.get("model_max_length", 2048),
    )
    model.eval()

    responses: List[str] = []
    for batch in batch_iterable(prompts, batch_size):
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(
            model.device
        )
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        responses.extend(decoded)

    return responses


__all__ = ["run_eval_pipeline"]
