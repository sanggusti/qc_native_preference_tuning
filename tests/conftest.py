"""Shared test fixtures for qc_native_preference_tuning."""

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so `src.*` imports resolve.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
