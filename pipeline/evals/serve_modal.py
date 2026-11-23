"""Serve one checkpoint per vLLM server on Modal (issue #55).

Pure builders (command, endpoint resolution) are tested with no Modal
dependency; the app definition below them follows the standard Modal
vLLM pattern and is exercised at deploy time, not in unit tests.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import modal
except ImportError:
    modal = None  # type: ignore[assignment]

GPU = "A10G"  # fits 3-4B instruct models with KV cache; size up per model
TIMEOUT_SECONDS = 7200  # one server lives for all of its checkpoint's cells
SERVER_PORT = 8000


def build_vllm_command(
    model_id: str, port: int = SERVER_PORT, extra_args: tuple[str, ...] = ()
) -> list[str]:
    """vLLM OpenAI-compatible server command for one checkpoint."""
    return [
        "vllm", "serve", model_id,
        "--host", "0.0.0.0", "--port", str(port),
        "--trust-remote-code",
        *extra_args,
    ]


def resolve_endpoint(
    model_ref: str, endpoints: dict[str, dict[str, str]], provider: str = "openai-api"
) -> dict[str, str]:
    """Resolve {base_url, model_arg} for a model ref from the endpoints map.

    A single-entry map serves as the default for every model (base and
    finetuned share one harness); otherwise the ref must match exactly.
    """
    if model_ref in endpoints:
        entry = endpoints[model_ref]
    elif len(endpoints) == 1:
        entry = next(iter(endpoints.values()))
    else:
        raise ValueError(f"no endpoint for model {model_ref!r} in {sorted(endpoints)}")
    return {
        "base_url": entry["base_url"],
        "model_arg": f"{provider}/{entry.get('served_name', model_ref)}",
    }


if modal is not None:
    app = modal.App(name="qc-native-preference-eval")
    vllm_image = modal.Image.debian_slim().uv_pip_install(
        "vllm==0.8.5",
        "huggingface_hub",
    )

    @app.cls(
        image=vllm_image,
        gpu=GPU,
        timeout=TIMEOUT_SECONDS,
        secrets=[modal.Secret.from_name("huggingface")],
    )
    class VLLMServer:
        """One OpenAI-compatible endpoint per checkpoint (deploy per model)."""

        model_id: str = modal.parameter()

        @modal.web_server(port=SERVER_PORT, startup_timeout=1800)
        def serve(self) -> None:
            import subprocess

            subprocess.Popen(build_vllm_command(self.model_id, SERVER_PORT))

    def tunnel_url(model_id: str, app_name: str = "qc-native-preference-eval") -> str:
        """Look up the deployed tunnel URL for a checkpoint's server."""
        cls = modal.Cls.lookup(app_name, "VLLMServer")
        return str(cls(model_id=model_id).serve.get_web_url())
