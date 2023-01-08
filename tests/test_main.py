import argparse
import logging
from pathlib import Path

import pytest
import yaml

from bids2table.__main__ import _main

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def args(tmp_path: Path) -> argparse.Namespace:
    yaml_overrides = [
        {"db_dir": str(tmp_path / "db")},  # `- key: value` override format
        f"log_dir={tmp_path / 'log'}",  # `- key=value` override format
        {"paths.list": [str(DATA_DIR / "ds000102-mriqc" / "sub-*")]},
        "collection_id=dummy",  # to be overridden below
    ]
    yaml_overrides_path = tmp_path / "overrides.yaml"
    with open(yaml_overrides_path, "w") as f:
        yaml.safe_dump(yaml_overrides, f)

    cli_overrides = ["collection_id=test_run", "dry_run=true"]

    args_ = argparse.Namespace(
        config="mriqc",
        print_only=False,
        overrides_yaml=str(yaml_overrides_path),
        overrides=cli_overrides,
    )
    return args_


def test_main(args: argparse.Namespace):
    logging.info("args:\n%s", args.__dict__)
    with open(args.overrides_yaml) as f:
        logging.info("yaml overrides:\n%s", f.read())
    _main(args)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
