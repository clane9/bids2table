from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from pytest import FixtureRequest

from bids2table.config import Config
from bids2table.engine import launch
from bids2table.logging import ProcessedLog

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(params=["true", "false"])
def config(request: FixtureRequest, tmp_path: Path) -> Config:
    overrides = [
        f"db_dir={tmp_path / 'db'}",
        f"log_dir={tmp_path / 'log'}",
        "collection_id=test_run",
        f"paths.list=[\"{DATA_DIR / 'ds000102-mriqc' / 'sub-*'}\"]",
        f"dry_run={request.param}",
    ]
    with initialize_config_module(
        "bids2table.config", version_base="1.1", job_name="bids2table"
    ):
        cfg = compose(config_name="mriqc", overrides=overrides)
    # DictConfig duck-typed as a Config, ignore type error
    return cfg  # type: ignore


def test_launch(config: Config):
    launch(config)

    if not config.dry_run:
        proc_log = ProcessedLog(config.db_dir)
        assert proc_log.df.shape == (4, 8)
        assert proc_log.df["error_rate"].max() == 0

        # Try running again, with the same paths. Should process no paths
        config.collection_id = "test_run2"
        launch(config)

        proc_log2 = ProcessedLog(config.db_dir)
        assert not (proc_log2.df["collection_id"] == "test_run2").any()


if __name__ == "__main__":
    pytest.main(["-s", __file__])
