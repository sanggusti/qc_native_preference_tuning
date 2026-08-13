"""Shared test fixtures for qc_native_preference_tuning."""

import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Ensure project root is on sys.path so `src.*` imports resolve.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load repository .env so env-dependent tests can run without manual export.
load_dotenv(dotenv_path=Path(PROJECT_ROOT) / ".env", override=False)
