from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from pytest import FixtureRequest

from bids2table.config import Config
from bids2table.engine import launch

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(params=["true", "false"])
def config(request: FixtureRequest, tmp_path: Path) -> Config:
    overrides = [
        f"db_dir={tmp_path / 'db'}",
        f"log_dir={tmp_path / 'log'}",
        "run_id=test_run",
        f"paths.list=[\"{DATA_DIR / 'ds000102-mriqc' / 'sub-*'}\"]",
        f"dry_run={request.param}",
    ]
    with initialize_config_module(
        "bids2table.config", version_base="1.1", job_name="bids2table"
    ):
        cfg = compose(config_name="mriqc", overrides=overrides)
    return cfg


def test_launch(config: Config):
    launch(config)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
