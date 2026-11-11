"""The faithfulness tests ask a real model (RAGAS judge): they need API keys, take minutes and spend free-tier quota,
and they fail whenever the provider rate-limits. They are therefore opt-in:

    RUN_LIVE_LLM=1 pytest tests/domains

Everything else runs offline."""
import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_LIVE_LLM") == "1":
        return
    skip = pytest.mark.skip(reason="live LLM test; set RUN_LIVE_LLM=1 to run it")
    for item in items:
        if "test_faithfulness" in item.nodeid:
            item.add_marker(skip)
