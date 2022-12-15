import logging
from typing import Any, Dict, Iterator, Optional, Tuple

import pytest

from bids2table.indexers import bids, get_indexer


def _generate_entity_cases() -> Iterator[
    Tuple[bids.BIDSEntityConfig, str, Optional[str]]
]:
    example: str
    expected: Any
    name: str
    key: Optional[str]
    dtype: str

    # cases with various examples for subject key
    cfg = bids.BIDSEntityConfig(name="subject", key="sub")
    # tuples of (example, expected)
    cases = [
        # basic
        ("sub-blah", "blah"),
        # capitalized
        ("sub-Blah", "Blah"),
        # key follows /
        ("/sub-blah_ses-01.nii.gz", "blah"),
        # key follows _
        ("ds-01_sub-blah_ses-01.nii.gz", "blah"),
        # repeated key
        ("ds/sub-blah/ses-01/sub-blah_ses-01_bold.nii.gz", "blah"),
        # value includes -
        ("sub-blah-01", "blah-01"),
        # key typo
        ("subb-blah", None),
        # key capitalized
        ("Sub-blah", None),
        # wrong key value separator
        ("sub_blah", None),
    ]
    for example, expected in cases:
        yield cfg, example, expected

    # cases for a single example file and multiple keys
    example = "ds/sub-blah/ses-01/sub-blah_ses-01_task-abc_run-01_bold.nii.gz"
    for name, key, dtype, expected in [
        ("subject", "sub", "str", "blah"),
        ("session", "ses", "str", "01"),
        ("task", None, "str", "abc"),
        ("run", None, "int", 1),
    ]:
        cfg = bids.BIDSEntityConfig(name=name, key=key, dtype=dtype)
        yield cfg, example, expected

    # special suffix cases
    cfg = bids.BIDSEntityConfig(name="suffix")
    cases = [
        # basic
        ("abc_bold.nii", "bold"),
        # compound extension
        ("abc_bold.nii.gz", "bold"),
        # suffixes must be preceded by _
        ("abc.nii.gz", None),
    ]
    for example, expected in cases:
        yield cfg, example, expected

    # special extension cases
    cfg = bids.BIDSEntityConfig(name="extension")
    cases = [
        # basic
        ("abc_bold.nii", ".nii"),
        # compound extension with suffix
        ("abc_bold.nii.gz", ".nii.gz"),
        # compound extension w/o suffix
        ("abc.nii.gz", ".nii.gz"),
        # ignore directory extension
        ("data.bids/abc.nii.gz", ".nii.gz"),
        # no extension
        ("abc", None),
    ]
    for example, expected in cases:
        yield cfg, example, expected


@pytest.mark.parametrize("case", _generate_entity_cases())
def test_bids_entity(case: Tuple[bids.BIDSEntityConfig, str, Optional[str]]):
    cfg, example, expected = case
    entity = bids.BIDSEntity.from_config(cfg)
    result = entity.search(example)
    logging.debug(f"\nentity={entity}\nexample={example}\nresult={result}")
    assert result == expected


@pytest.fixture
def indexer() -> bids.BIDSIndexer:
    entcfgs = [
        bids.BIDSEntityConfig(name="subject", key="sub"),
        bids.BIDSEntityConfig(name="session", key="ses"),
        bids.BIDSEntityConfig(name="task"),
        bids.BIDSEntityConfig(name="run", dtype="int"),
    ]
    cfg = bids.BIDSIndexerConfig(columns=entcfgs)
    indexer = bids.BIDSIndexer.from_config(cfg)
    indexer.set_root("/dummy")
    return indexer


_indexer_cases = [
    (
        # basic
        "ds/sub-blah/ses-01/sub-blah_ses-01_task-abc_run-01_bold.nii.gz",
        {"subject": "blah", "session": "01", "task": "abc", "run": 1},
    ),
    (
        # missing ses and run
        "ds/sub-blah/sub-blah_task-abc.json",
        {"subject": "blah", "session": None, "task": "abc", "run": None},
    ),
]


@pytest.mark.parametrize("case", _indexer_cases)
def test_bids_indexer(indexer: bids.BIDSIndexer, case: Tuple[str, Dict[str, Any]]):
    example, expected = case
    key = indexer(example)
    assert key == expected


def test_get_bids_indexer():
    assert get_indexer("bids_indexer") == bids.BIDSIndexer


if __name__ == "__main__":
    pytest.main([__file__])
