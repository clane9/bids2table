import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional

from omegaconf import MISSING

from bids2table import RecordDict, StrOrPath
from bids2table.schema import Fields, cast_to_schema, create_schema, format_schema
from bids2table.utils import set_overlap

__all__ = [
    "HandlerConfig",
    "Handler",
    "HandlerTuple",
]


@dataclass
class HandlerConfig:
    name: str = MISSING
    pattern: str = MISSING
    label: str = MISSING
    fields: Optional[Dict[str, str]] = MISSING
    metadata: Optional[Dict[str, str]] = None


class Handler(ABC):
    """
    Abstract handler for extracting records with an expected schema from files.

    Sub-classes must implement:

    - ``_load()``: generate the record dict from a given file path. The record need not
        match the schema, as this is handled afterward.

    Sub-classes may want to specialize other methods as needed.
    """

    _CAST_WITH_NULL = False
    OVERLAP_WARN_THRESHOLD = 1.0

    def __init__(
        self,
        fields: Fields,
        metadata: Optional[Dict[str, str]] = None,
    ):
        self.fields = fields
        self.metadata = metadata
        self.schema = create_schema(fields, metadata=metadata)

    @abstractmethod
    def _load(self, path: StrOrPath) -> Optional[RecordDict]:
        """
        Generate a raw record from a file path, or ``None`` if a valid record can't be
        generated. Note the record need not match the ``schema``.
        """
        raise NotImplementedError

    def __call__(self, path: StrOrPath) -> Optional[RecordDict]:
        """
        Generate a record matching the schema from a file path, or ``None`` if a valid
        record can't be generated.
        """
        record = self._load(path)
        if record is not None:
            overlap = set_overlap(self.schema.names, record.keys())
            if overlap < self.OVERLAP_WARN_THRESHOLD:
                logging.warning(
                    f"Field overlap between record and schema is only {overlap:.2f}\n"
                    f"\tpath: {path}\n"
                    f"\thandler: {self}"
                )
            # We leave out null values so that the record for this handler can be
            # accumulated over several calls and paths.
            record = cast_to_schema(
                record,
                schema=self.schema,
                safe=True,
                with_null=self._CAST_WITH_NULL,
            )
        return record

    @classmethod
    def from_config(cls, cfg: HandlerConfig) -> "Handler":
        """
        Initialze a Handler from a config.
        """
        if cfg.fields is None:
            raise ValueError("cfg.fields is required")
        return cls(fields=cfg.fields, metadata=cfg.metadata)

    def __repr__(self) -> str:
        schema_fmt = "\n\t".join(format_schema(self.schema).split("\n"))
        return f"{self.__class__.__name__}:\n\t{schema_fmt}"


class HandlerTuple(NamedTuple):
    group: Optional[str]
    pattern: str
    label: str
    handler: Handler
