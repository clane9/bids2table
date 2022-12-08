import argparse
import json
from pathlib import Path
from typing import List

import yaml
from hydra import compose, initialize_config_dir, initialize_config_module
from omegaconf import OmegaConf

from bids2table.engine import launch


def main():
    """
    Main command-line program
    """
    parser = argparse.ArgumentParser("bids2table")
    parser.add_argument(
        "--config",
        "-c",
        metavar="NAME",
        required=True,
        type=str,
        help="name of or path to config file",
    )
    parser.add_argument(
        "--overrides-yaml",
        "-y",
        metavar="PATH",
        type=str,
        help="path to YAML file containing a list of overrides",
    )
    parser.add_argument(
        "--print-only",
        "-p",
        action="store_true",
        help="only print composed config and exit",
    )
    parser.add_argument(
        "overrides",
        metavar="KEY=VALUE",
        type=str,
        nargs="*",
        help="list of config overrides",
    )

    args = parser.parse_args()

    # load overrides
    if args.overrides_yaml:
        overrides = _load_overrides_yaml(args.overrides_yaml)
    else:
        overrides = []
    overrides = _merge_overrides(overrides, args.overrides)

    # compose config
    config_path = Path(args.config)
    config = config_path.stem
    # config name relative to internal configs
    if str(config_path) == config_path.stem:
        with initialize_config_module(
            "bids2table.config", version_base="1.1", job_name="bids2table"
        ):
            cfg = compose(config, overrides=overrides)
    # local config path
    else:
        config_dir = str(config_path.parent.absolute())
        with initialize_config_dir(
            config_dir, version_base="1.1", job_name="bids2table"
        ):
            cfg = compose(config, overrides=overrides)

    if args.print_only:
        print(OmegaConf.to_yaml(cfg))
    else:
        launch(cfg)


def _load_overrides_yaml(path: str) -> List[str]:
    """
    Load list of overrides from a yaml file
    """
    with open(path) as f:
        overrides_yaml = yaml.safe_load(f)

    if not isinstance(overrides_yaml, list):
        raise ValueError(
            "Invalid YAML overrides; expected a list of `key: value` pairs; "
            f"got:\n{overrides_yaml}"
        )

    overrides = []
    for arg in overrides_yaml:
        if isinstance(arg, dict) and len(arg) == 1:
            k, v = list(arg.items())[0]
            overrides.append(f"{k}={json.dumps(v)}")
        else:
            raise ValueError(
                f"Invalid YAML overrides; expected a `key: value` pair; got `{arg}`"
            )
    return overrides


def _merge_overrides(overrides: List[str], other: List[str]) -> List[str]:
    """
    Merge two lists of overrides, keeping only last value for repeated keys.
    """
    merged = {}
    for arg in overrides + other:
        key, val = arg.strip().split("=")
        merged[key] = val
    return [f"{k}={v}" for k, v in merged.items()]


if __name__ == "__main__":
    main()
