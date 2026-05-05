"""Utility functions for qc_native_preference_tuning."""

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging for the project.

    Args:
        level: Logging level string (e.g., 'INFO', 'DEBUG').
        log_file: Optional path to write logs to a file.
    """
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
    )


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten.
        parent_key: Prefix for nested keys.
        sep: Separator between key levels.

    Returns:
        Flattened dictionary.
    """
    items: List = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        Resolved Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def batch_iterable(iterable: List[Any], batch_size: int):
    """Yield successive batches from an iterable.

    Args:
        iterable: List to batch.
        batch_size: Size of each batch.

    Yields:
        Batches of the input list.
    """
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


__all__ = [
    "set_seed",
    "setup_logging",
    "flatten_dict",
    "ensure_dir",
    "batch_iterable",
]
