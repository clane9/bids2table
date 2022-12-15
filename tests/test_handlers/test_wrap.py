import logging
from importlib import resources

import pytest

from bids2table.handlers import WrapHandler, WrapHandlerConfig, get_handler
from bids2table.loaders import LoaderConfig


@pytest.mark.parametrize(
    "example", ["mriqc_anat_T1w.json", "bids2table/examples/mriqc_anat_T1w.json"]
)
def test_wrap_handler(example: str):
    cfg = WrapHandlerConfig(
        loader=LoaderConfig(name="load_json_dict", kwargs={"nested": False}),
        example=example,
        # overwrite a datatype, add an extra field
        fields={"EFC": "float32", "_dummy": "str"},
        # set metadata for the extra field
        metadata={"_dummy": "some missing data"},
        # rename one field, delete another
        rename_map={"efc": "EFC", "fber": WrapHandler.DELETE},
        overlap_threshold=0.5,
    )
    cls = get_handler(cfg.name)
    handler = cls.from_config(cfg)
    logging.info("Handler: %s", handler)

    with resources.path(WrapHandler.EXAMPLES_PKG, "mriqc_anat_T1w.json") as p:
        record = handler(p)
    record_keys = list(record.keys())

    assert record is not None
    # check that _dummy is in schema but not in record
    assert record_keys + ["_dummy"] == handler.schema.names
    # check that efc has been properly renamed without changing position
    assert record_keys[2] == "EFC" and "efc" not in record
    # check that fber has been removed
    assert "fber" not in record
    # check that the efc type is updated
    assert handler.schema.field("EFC").type == "float32"

    with resources.path(WrapHandler.EXAMPLES_PKG, "mriqc_func_bold.json") as p:
        record = handler(p)
        assert record is None


if __name__ == "__main__":
    pytest.main([__file__])
