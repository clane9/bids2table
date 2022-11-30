import logging

import pytest

from bids2table.env import collect_env_info


def test_collect_env_info():
    env_report = collect_env_info()
    logging.info("Environment:\n%s", env_report)
    assert "bids2table" in env_report


if __name__ == "__main__":
    pytest.main([__file__])
