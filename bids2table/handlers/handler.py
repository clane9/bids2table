import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional

from omegaconf import MISSING

from bids2table import RecordDict, StrOrPath
from bids2table.schema import Fields, cast_to_schema, create_schema, format_schema

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
    overlap_threshold: float = 0.25


class Handler(ABC):
    """
    Abstract handler for extracting records with an expected schema from files.

    Sub-classes must implement:

    - ``_load()``: generate the record dict from a given file path. The record need not
        match the schema, as this is handled afterward.

    Sub-classes may want to specialize other methods as needed.
    """

    _CAST_WITH_NULL = False

    def __init__(
        self,
        fields: Fields,
        metadata: Optional[Dict[str, str]] = None,
        overlap_threshold: float = 0.5,
    ):
        self.fields = fields
        self.metadata = metadata
        self.overlap_threshold = overlap_threshold
        self.schema = create_schema(fields, metadata=metadata)

        self._schema_names = set(self.schema.names)

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
            record_names = set(record.keys())
            schema_diff = self._schema_names - record_names
            if schema_diff:
                logging.debug(
                    f"Record is missing {len(schema_diff)}/{len(self._schema_names)} "
                    "fields from schema\n"
                    f"\tpath: {path}\n"
                    f"\tmissing fields: {schema_diff}"
                )
            record_diff = record_names - self._schema_names
            if record_diff:
                logging.debug(
                    f"Record contains {len(record_diff)}/{len(record_names)} "
                    "extra fields not in schema\n"
                    f"\tpath: {path}\n"
                    f"\textra fields: {record_diff}"
                )
            overlap = 1 - len(record_diff) / len(record_names)
            if overlap < self.overlap_threshold:
                raise ValueError(
                    f"Record doesn't match schema; overlap is only {100*overlap:.0f}% "
                    f"< {100*self.overlap_threshold:.0f}%; "
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
    @abstractmethod
    def from_config(cls, cfg: HandlerConfig) -> "Handler":
        """
        Initialze a Handler from a config.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        schema_fmt = "\n\t".join(format_schema(self.schema).split("\n"))
        return f"{self.__class__.__name__}:\n\t{schema_fmt}"


class HandlerTuple(NamedTuple):
    group: Optional[str]
    pattern: str
    label: str
    handler: Handler
