import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Dict, Optional

import pyarrow as pa
from omegaconf import MISSING

from bids2table import RecordDict, StrOrPath, find_file
from bids2table.loaders import Loader, LoaderConfig, loader_from_config
from bids2table.schema import Fields, get_fields

from .handler import Handler, HandlerConfig
from .registry import register_handler

__all__ = ["WrapHandlerConfig", "WrapHandler"]


@dataclass
class WrapHandlerConfig(HandlerConfig):
    name: str = "wrap_handler"
    loader: LoaderConfig = MISSING
    fields: Optional[Dict[str, str]] = None
    example: Optional[Path] = None
    rename_map: Optional[Dict[str, str]] = None


@register_handler(name="wrap_handler")
class WrapHandler(Handler):
    """
    A Handler wrapping a ``loader`` function.
    """

    EXAMPLES_PKG = "bids2table.examples"
    DELETE = "__delete__"

    def __init__(
        self,
        loader: Loader,
        fields: Optional[Fields] = None,
        example: Optional[StrOrPath] = None,
        metadata: Optional[Dict[str, str]] = None,
        rename_map: Optional[Dict[str, str]] = None,
    ):
        fields_ = {}
        if example is not None:
            record = self.load_example(loader, example)
            record = self.apply_renaming(rename_map, record)
            if record is None:
                raise ValueError(f"Example {example} could not be loaded by {loader}")
            batch = pa.RecordBatch.from_pylist([record])
            fields_ = get_fields(batch.schema)
        if fields is not None:
            fields_.update(fields)
        if len(fields_) == 0:
            raise ValueError("A non-empty 'fields' or 'example' is required")

        super().__init__(fields_, metadata)
        self.loader = loader
        self.example = example
        self.rename_map = rename_map

    def _load(self, path: StrOrPath) -> Optional[RecordDict]:
        record = self.loader(path)
        record = self.apply_renaming(self.rename_map, record)
        return record

    @classmethod
    def load_example(cls, loader: Loader, path: StrOrPath) -> Optional[RecordDict]:
        """
        Load an example that may be located as a package resource, absolute path, or a
        relative path under one of the directories in the internal ``PATH``.
        """
        path = Path(path)

        p = find_file(path)
        if p is not None:
            return loader(p)

        # TODO: Is this how we want to publish examples?
        if resources.is_resource(cls.EXAMPLES_PKG, path.name):
            with resources.path(cls.EXAMPLES_PKG, path.name) as p:
                return loader(p)

        raise FileNotFoundError(f"Example {path} not found")

    @classmethod
    def apply_renaming(
        cls, rename_map: Optional[Dict[str, str]], record: Optional[RecordDict]
    ) -> Optional[RecordDict]:
        """
        Apply a renaming map to the keys in ``record``.
        """
        if rename_map is None or record is None:
            return record

        # This may be a bit inefficient, as the rename_map is typically much smaller.
        # However doing it this way to guarantee the same order.
        return {
            rename_map.get(k, k): v
            for k, v in record.items()
            if rename_map.get(k) != cls.DELETE
        }

    @classmethod
    def from_config(cls, cfg: WrapHandlerConfig) -> "Handler":  # type: ignore[override]
        """
        Initialze a Handler from a config.
        """
        return cls(
            loader=loader_from_config(cfg.loader),
            fields=cfg.fields,
            example=cfg.example,
            metadata=cfg.metadata,
            rename_map=cfg.rename_map,
        )

    def __repr__(self) -> str:
        return (
            f"{super().__repr__()}\n"
            f"\tloader: {self.loader}\n"
            f"\texample: {self.example}\n"
            f"\trename_map: {json.dumps(self.rename_map)}"
        )
