from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import pyarrow as pa
from omegaconf import MISSING

from bids2table.loaders import Loader, LoaderConfig, PartialLoader
from bids2table.path import locate_file
from bids2table.schema import DataType, get_fields
from bids2table.types import RecordDict, StrOrPath

from .handler import Handler, HandlerConfig
from .registry import register_handler

__all__ = ["WrapHandlerConfig", "WrapHandler"]


@dataclass
class WrapHandlerConfig(HandlerConfig):
    name: str = "wrap_handler"
    loader: LoaderConfig = MISSING
    example: Optional[str] = None
    fields: Optional[Dict[str, str]] = None


@register_handler(name="wrap_handler")
class WrapHandler(Handler):
    """
    A Handler wrapping a ``loader`` function.
    """

    EXAMPLES_PKG = "bids2table.config.examples"

    def __init__(
        self,
        loader: Loader,
        example: Optional[StrOrPath] = None,
        fields: Optional[Mapping[str, DataType]] = None,
        metadata: Optional[Dict[str, str]] = None,
        rename_map: Optional[Dict[str, str]] = None,
        overlap_threshold: float = 0.25,
    ):
        fields_ = {}
        if example is not None:
            with locate_file(example, pkg=self.EXAMPLES_PKG) as path:
                if path is None:
                    raise FileNotFoundError(f"Example {example} not found")
                record = loader(path)
            record = self.apply_renaming(rename_map, record)
            if not record:
                raise ValueError(
                    f"Example {example} is empty or could not be loaded by {loader}"
                )
            # Replace any items that will be overridden by the fields arg with a
            # placeholder. This way, if there are items whose type can't be inferred
            # (e.g. extension types), the user can correct the error by overriding with
            # fields. Note that we replace with placeholder rather than remove to
            # preserve column order.
            if fields is not None:
                record = {k: (0 if k in fields else v) for k, v in record.items()}

            batch = pa.RecordBatch.from_pylist([record])
            fields_ = get_fields(batch.schema)
        if fields is not None:
            fields_.update(fields)
        if len(fields_) == 0:
            raise ValueError("A non-empty fields or example is required")

        super().__init__(
            fields=fields_,
            metadata=metadata,
            rename_map=rename_map,
            overlap_threshold=overlap_threshold,
        )
        self.loader = loader
        self.example = example

    def _load(self, path: StrOrPath) -> Optional[RecordDict]:
        return self.loader(path)

    @classmethod
    def from_config(cls, cfg: WrapHandlerConfig) -> "Handler":  # type: ignore[override]
        """
        Initialze a Handler from a config.
        """
        return cls(
            loader=PartialLoader.from_config(cfg.loader),
            example=cfg.example,
            fields=cfg.fields,
            metadata=cfg.metadata,
            rename_map=cfg.rename_map,
            overlap_threshold=cfg.overlap_threshold,
        )

    def __str__(self) -> str:
        return (
            f"{super().__str__()}\n"
            f"\tloader: {self.loader}\n"
            f"\texample: {self.example}"
        )
