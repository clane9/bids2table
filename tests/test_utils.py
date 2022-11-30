import logging
from typing import List

import pytest

from bids2table import utils as ut


@pytest.fixture(scope="module")
def pattern_lut() -> ut.PatternLUT:
    patterns = [
        "*.txt",
        "*_blah.txt",
        "blah*",
        "foo/*.gz",
    ]
    items = [(p, ii) for ii, p in enumerate(patterns)]
    lut = ut.PatternLUT(items)
    return lut


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("a.txt", [0]),
        ("blah/a.txt", [0]),
        ("blah.txt", [0]),
        ("a_blah.txt", [0, 1]),
        ("blah_blah.txt", [0, 1]),
        ("blah", [2]),
        ("blah_blah", [2]),
        ("foo/a.gz", [3]),
        ("foo/a/b.gz", [3]),  # TODO: might not want this
        ("blah/foo/a.gz", []),  # TODO: might not want this
        ("bar", []),
    ],
)
def test_pattern_lut(pattern_lut: ut.PatternLUT, path: str, expected: List[int]):
    matches = list(pattern_lut.lookup(path))
    logging.debug(f"path={path}\tmatches={matches}")
    value_matches = [v[1] for v in matches]
    assert value_matches == expected


if __name__ == "__main__":
    pytest.main([__file__])
