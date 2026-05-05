"""Modal cloud runner for training and evaluation pipelines."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modal app definition
# ---------------------------------------------------------------------------
# This module defines Modal stubs/apps for running GPU-heavy workloads on
# Modal cloud infrastructure.  Import modal lazily so the rest of the
# project can be imported in environments where Modal is not installed.
# ---------------------------------------------------------------------------


def get_modal_image(config: dict):
    """Build a Modal Image for the project.

    Args:
        config: Full project configuration dictionary.

    Returns:
        A modal.Image instance.
    """
    import modal

    modal_cfg = config.get("modal", {})
    base_image = modal_cfg.get("image", "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime")

    return (
        modal.Image.from_registry(base_image)
        .pip_install(
            "transformers>=4.40.0",
            "datasets>=2.19.0",
            "trl>=0.9.0",
            "peft>=0.11.0",
            "accelerate>=0.30.0",
            "bitsandbytes>=0.43.0",
            "openai>=1.0.0",
            "wandb",
            "pyyaml",
            "numpy",
        )
        .copy_local_dir(".", "/app")
        .workdir("/app")
    )


def build_modal_app(config: dict):
    """Construct and return a configured Modal App with training and eval functions.

    Args:
        config: Full project configuration dictionary.

    Returns:
        modal.App instance with registered functions.
    """
    import modal

    modal_cfg = config.get("modal", {})
    app_name = modal_cfg.get("app_name", "qc_preference_tuning")
    gpu = modal_cfg.get("gpu", "A100")
    gpu_count = modal_cfg.get("gpu_count", 1)
    timeout = modal_cfg.get("timeout", 86400)
    secret_names = modal_cfg.get("secret_names", [])

    image = get_modal_image(config)
    secrets = [modal.Secret.from_name(name) for name in secret_names]

    app = modal.App(app_name)

    @app.function(
        image=image,
        gpu=modal.gpu.A100(count=gpu_count) if gpu == "A100" else modal.gpu.T4(count=gpu_count),
        timeout=timeout,
        secrets=secrets,
    )
    def train(cfg: dict, dataset_path: Optional[str] = None) -> None:
        """Remote training function executed on Modal."""
        from pipeline.training import run_training_pipeline
        run_training_pipeline(cfg, dataset_path=dataset_path)

    @app.function(
        image=image,
        gpu=modal.gpu.A100(count=gpu_count) if gpu == "A100" else modal.gpu.T4(count=gpu_count),
        timeout=timeout,
        secrets=secrets,
    )
    def evaluate(cfg: dict, model_path: str, baseline_model_path: Optional[str] = None):
        """Remote evaluation function executed on Modal."""
        from pipeline.evals import run_eval_pipeline
        return run_eval_pipeline(cfg, model_path, baseline_model_path)

    return app


__all__ = ["get_modal_image", "build_modal_app"]
