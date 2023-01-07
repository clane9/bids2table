import logging
from pathlib import Path
from typing import List

import pytest
from hydra import compose, errors, initialize, initialize_config_module
from omegaconf import OmegaConf

from bids2table.config import Config

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def overrides(tmp_path: Path) -> List[str]:
    return [
        f"db_dir={tmp_path / 'db'}",
        f"log_dir={tmp_path / 'log'}",
        "collection_id=test_run",
        f"paths.list=[\"{DATA_DIR / 'ds000102-mriqc' / 'sub-*'}\"]",
        "dry_run=true",
    ]


@pytest.fixture
def missing_overrides(tmp_path: Path) -> List[str]:
    return [
        f"db_dir={tmp_path / 'db'}",
        f"log_dir={tmp_path / 'log'}",
        f"paths.list=[\"{DATA_DIR / 'ds000102-mriqc' / 'sub-*'}\"]",
    ]


@pytest.fixture
def wrong_type_overrides(tmp_path: Path) -> List[str]:
    return [
        f"db_dir={tmp_path / 'db'}",
        f"log_dir={tmp_path / 'log'}",
        "collection_id=test_run",
        f"paths.list=[\"{DATA_DIR / 'ds000102-mriqc' / 'sub-*'}\"]",
        "dry_run=1.0",
    ]


def test_config(overrides: List[str]):
    cfg = _load_config(overrides)
    table_cfg = cfg.tables["mriqc_anat"]
    assert table_cfg.name == "mriqc_anat"
    handler_cfg = table_cfg.handlers["mriqc_anat_T1w"]
    assert handler_cfg.name == "wrap_handler"


def test_local_config(overrides: List[str]):
    cfg = _load_local_config(overrides)
    assert "mriqc_func_local" in cfg.tables


def test_config_missing(missing_overrides: List[str]):
    cfg = _load_config(missing_overrides)
    missing_keys = OmegaConf.missing_keys(cfg)
    assert missing_keys == {"collection_id"}


def test_config_wrong_type(wrong_type_overrides: List[str]):
    with pytest.raises(errors.ConfigCompositionException):
        _load_config(wrong_type_overrides)


def _load_config(_overrides: List[str]) -> Config:
    logging.info(f"overrides:\n{_overrides}")
    with initialize_config_module(
        "bids2table.config", version_base="1.2", job_name="bids2table"
    ):
        cfg = compose(config_name="mriqc", overrides=_overrides)
    logging.info(f"config:\n{OmegaConf.to_yaml(cfg)}")
    # DictConfig duck-typed as a Config, ignore type error
    return cfg  # type: ignore


def _load_local_config(_overrides: List[str]) -> Config:
    logging.info(f"overrides:\n{_overrides}")
    with initialize("config_local", version_base="1.2", job_name="bids2table"):
        cfg = compose(config_name="mriqc_local", overrides=_overrides)
    logging.info(f"config:\n{OmegaConf.to_yaml(cfg)}")
    return cfg  # type: ignore


if __name__ == "__main__":
    pytest.main([__file__])
