import logging
from pathlib import Path

import pytest

from bids2table.crawler import Crawler
from bids2table.handlers import HandlerTuple, WrapHandler, WrapHandlerConfig
from bids2table.indexers import bids
from bids2table.loaders import LoaderConfig

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def crawler() -> Crawler:
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
                bids.BIDSEntityConfig(name="run"),
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
    return crawler


@pytest.fixture(scope="module")
def sloppy_crawler() -> Crawler:
    func_indexer = bids.BIDSIndexer.from_config(
        bids.BIDSIndexerConfig(
            columns=[
                bids.BIDSEntityConfig(name="subject", key="sub", required=True),
                bids.BIDSEntityConfig(name="task"),
                bids.BIDSEntityConfig(name="run"),
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
        # note the pattern is too broad
        "func": [HandlerTuple("func", "*.json", "mriqc_func_bold", func_handler)],
    }
    sloppy_crawler = Crawler(indexers_map=indexers_map, handlers_map=handlers_map)
    return sloppy_crawler


@pytest.mark.parametrize("dirpath", ["sub-01", "sub-02"])
def test_crawler(crawler: Crawler, dirpath: str):
    tables, errs, counts = crawler.crawl(DATA_DIR / "ds000102-mriqc" / dirpath)
    assert counts.error_count == len(errs) == 0

    anat_df = tables["anat"].to_pandas()
    func_df = tables["func"].to_pandas()
    logging.info("anat:\n%s", anat_df)
    logging.info("func:\n%s", func_df)
    assert anat_df.shape == (1, 70)
    assert func_df.shape == (2, 48)


@pytest.mark.parametrize(
    ("dirpath", "expected_error_count"), [("sub-01", 1), ("sub_bad", 2)]
)
def test_crawler_errors(
    sloppy_crawler: Crawler, dirpath: str, expected_error_count: int
):
    _, errs, counts = sloppy_crawler.crawl(DATA_DIR / "ds000102-mriqc" / dirpath)
    assert counts.error_count == len(errs) == expected_error_count


if __name__ == "__main__":
    pytest.main([__file__])
