"""Tests for Adaption SDK integration."""

import os

import pytest
from adaption import Adaption


@pytest.fixture
def adaption_client():
    """Create an Adaption client (requires ADAPTION_API_KEY env var)."""
    api_key = os.getenv("ADAPTION_API_KEY")
    if not api_key:
        pytest.skip("ADAPTION_API_KEY not set")
    return Adaption(api_key=api_key)


@pytest.mark.skipif(
    not os.getenv("ADAPTION_API_KEY"),
    reason="ADAPTION_API_KEY not set",
)
def test_client_initializes(adaption_client):
    """Adaption client can be instantiated with an API key."""
    assert adaption_client is not None
