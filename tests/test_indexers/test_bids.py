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

    cfg = bids.BIDSEntityConfig(name="subject", key="sub")
    cases = [
        ("sub-blah", "blah"),
        ("sub-Blah", "Blah"),
        ("/sub-blah_ses-01.nii.gz", "blah"),
        ("ds-01_sub-blah_ses-01.nii.gz", "blah"),
        ("ds/sub-blah/ses-01/sub-blah_ses-01_bold.nii.gz", "blah"),
        ("sub-blah-01", "blah-01"),
        ("subb-blah", None),
        ("Sub-blah", None),
        ("sub_blah", None),
    ]
    for example, expected in cases:
        yield cfg, example, expected

    example = "ds/sub-blah/ses-01/sub-blah_ses-01_task-abc_run-01_bold.nii.gz"
    for name, key, dtype, expected in [
        ("subject", "sub", "str", "blah"),
        ("session", "ses", "str", "01"),
        ("task", None, "str", "abc"),
        ("run", None, "int", 1),
    ]:
        cfg = bids.BIDSEntityConfig(name=name, key=key, dtype=dtype)
        yield cfg, example, expected

    cfg = bids.BIDSEntityConfig(
        name="suffix",
        pattern=bids.BIDS_PATTERNS.suffix,
    )
    cases = [
        ("abc_bold.nii", "bold"),
        ("abc_bold.nii.gz", "bold"),
        ("abc.nii.gz", None),
    ]
    for example, expected in cases:
        yield cfg, example, expected

    cfg = bids.BIDSEntityConfig(
        name="extension",
        pattern=bids.BIDS_PATTERNS.extension,
    )
    cases = [
        ("abc_bold.nii", ".nii"),
        ("abc_bold.nii.gz", ".nii.gz"),
        ("abc.nii.gz", ".nii.gz"),
        ("data.bids/abc.nii.gz", ".nii.gz"),
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
        bids.BIDSEntityConfig(name="subject", key="sub", required=True),
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
        "ds/sub-blah/ses-01/sub-blah_ses-01_task-abc_run-01_bold.nii.gz",
        {"subject": "blah", "session": "01", "task": "abc", "run": 1},
    ),
    (
        "ds/sub-blah/ses-01/sub-blah_ses-01_task-abc.json",
        {"subject": "blah", "session": "01", "task": "abc", "run": None},
    ),
]


@pytest.mark.parametrize("case", _indexer_cases)
def test_bids_indexer(indexer: bids.BIDSIndexer, case: Tuple[str, Dict[str, Any]]):
    example, expected = case
    key = indexer(example)
    assert key == expected


def test_required_entity(indexer: bids.BIDSIndexer):
    with pytest.raises(RuntimeError):
        indexer("bold.nii.gz")


def test_get_bids_indexer():
    assert get_indexer("bids_indexer") == bids.BIDSIndexer


if __name__ == "__main__":
    pytest.main([__file__])
