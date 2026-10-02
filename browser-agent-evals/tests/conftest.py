"""Global test configuration and fixtures."""

import pytest
from agent.llm import reset_provider_exhaustion


@pytest.fixture(autouse=True)
def clean_exhausted_providers():
    """Ensure every test starts and ends with a clean exhaustion state."""
    reset_provider_exhaustion()
    yield
    reset_provider_exhaustion()
