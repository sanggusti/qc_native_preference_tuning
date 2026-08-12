import modal

app = modal.App(name="qc-native-preference")

image = modal.Image.debian_slim().env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}).uv_pip_install(
        # Core ML
        "torch==2.9.1",
        "torchvision",
        "transformers==4.56.2",
        "accelerate",
        "huggingface_hub",
        "tokenizers",
        "sentencepiece",
        "protobuf",
        # Dataset loading
        "datasets>=2.16.0",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
        "orjson>=3.9.0",
        # Config
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "pyyaml",
        # Training
        "peft",
        "wandb",
        "einops",
        "tqdm",
        # Utilities
        "Pillow",
        "numpy",
    )

@app.function(
    image=image,
    gpu="A100",
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")]
)
def run_inference():
    pass
