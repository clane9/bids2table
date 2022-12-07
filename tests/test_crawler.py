import logging
from pathlib import Path
from typing import Tuple

import pytest

from bids2table.crawler import Crawler
from bids2table.handlers import HandlerTuple, WrapHandler, WrapHandlerConfig
from bids2table.indexers import bids
from bids2table.loaders import LoaderConfig

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def crawler():
    anat_indexer = bids.BIDSIndexer.from_config(
        bids.BIDSIndexerConfig(
            columns=[bids.BIDSEntityConfig(name="subject", key="sub", required=True)]
        )
    )
    anat_handler = WrapHandler.from_config(
        WrapHandlerConfig(
            loader=LoaderConfig(name="load_json_dict", kwargs={"nested": False}),
            example=Path("mriqc_anat_T1w.json"),
        )
    )

    func_indexer = bids.BIDSIndexer.from_config(
        bids.BIDSIndexerConfig(
            columns=[
                bids.BIDSEntityConfig(name="subject", key="sub", required=True),
                bids.BIDSEntityConfig(name="task"),
                bids.BIDSEntityConfig(name="run", dtype="int"),
            ]
        )
    )
    func_handler = WrapHandler.from_config(
        WrapHandlerConfig(
            loader=LoaderConfig(name="load_json_dict", kwargs={"nested": False}),
            example=Path("mriqc_func_bold.json"),
        )
    )

    indexers_map = {"anat": anat_indexer, "func": func_indexer}
    handlers_map = {
        "anat": [HandlerTuple("anat", "*_T1w.json", "mriqc_anat_T1w", anat_handler)],
        "func": [HandlerTuple("func", "*_bold.json", "mriqc_func_bold", func_handler)],
    }
    crawler = Crawler(indexers_map=indexers_map, handlers_map=handlers_map)
    yield crawler
    crawler.close()


@pytest.fixture(scope="module")
def sloppy_crawler():
    """
    This crawler uses a file pattern that is too broad resulting in bad matches
    """
    func_indexer = bids.BIDSIndexer.from_config(
        bids.BIDSIndexerConfig(
            columns=[
                bids.BIDSEntityConfig(name="subject", key="sub", required=True),
                bids.BIDSEntityConfig(name="task"),
                bids.BIDSEntityConfig(name="run", dtype="int"),
            ]
        )
    )
    func_handler = WrapHandler.from_config(
        WrapHandlerConfig(
            loader=LoaderConfig(name="load_json_dict", kwargs={"nested": False}),
            example=Path("mriqc_func_bold.json"),
        )
    )

    indexers_map = {"func": func_indexer}
    handlers_map = {
        "func": [HandlerTuple("func", "*.json", "mriqc_func_bold", func_handler)],
    }
    sloppy_crawler = Crawler(indexers_map=indexers_map, handlers_map=handlers_map)
    yield sloppy_crawler
    sloppy_crawler.close()


@pytest.mark.parametrize("dirpath", ["sub-01", "sub-02"])
def test_crawler(crawler: Crawler, dirpath: str):
    tables, errs, counts = crawler.crawl(DATA_DIR / "ds000102-mriqc" / dirpath)
    assert counts.total == counts.process == 3
    assert counts.error == len(errs) == 0

    anat_df = tables["anat"].to_pandas()
    func_df = tables["func"].to_pandas()
    logging.info("anat:\n%s", anat_df)
    logging.info("func:\n%s", func_df)
    assert anat_df.shape == (1, 70)
    assert func_df.shape == (2, 48)


@pytest.mark.parametrize(
    (
        "dirpath",
        "expected_counts",
    ),
    [("sub-01", (3, 2, 0)), ("sub_bad", (5, 2, 1))],
)
def test_crawler_skips_errors(
    sloppy_crawler: Crawler,
    dirpath: str,
    expected_counts: Tuple[int, int, int],
):
    _, errs, counts = sloppy_crawler.crawl(DATA_DIR / "ds000102-mriqc" / dirpath)
    exp_total, exp_processed, exp_error = expected_counts
    assert counts.total == exp_total
    assert counts.process == exp_processed
    assert counts.error == len(errs) == exp_error

    if len(errs) > 0:
        err_dict = errs[0].to_dict()
        assert list(err_dict.keys()) == ["path", "pattern", "handler", "exception"]


if __name__ == "__main__":
    pytest.main([__file__])
