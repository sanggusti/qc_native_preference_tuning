"""Configuration package for qc_native_preference_tuning."""

from pathlib import Path
import yaml

CONFIG_DIR = Path(__file__).parent
ROOT_DIR = CONFIG_DIR.parent


def load_config(config_path: str | Path | None = None) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the config file. Defaults to configs/config.yaml.

    Returns:
        Dictionary containing the configuration.
    """
    if config_path is None:
        config_path = CONFIG_DIR / "config.yaml"
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


__all__ = ["load_config", "CONFIG_DIR", "ROOT_DIR"]
