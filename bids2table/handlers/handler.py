import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional

from omegaconf import MISSING

from bids2table.schema import DataType, cast_to_schema, create_schema, format_schema
from bids2table.types import RecordDict, StrOrPath

__all__ = [
    "HandlerConfig",
    "Handler",
    "HandlerTuple",
]


@dataclass
class HandlerConfig:
    name: str = MISSING
    pattern: List[str] = MISSING
    label: str = MISSING
    fields: Optional[Dict[str, str]] = MISSING
    metadata: Optional[Dict[str, str]] = None
    rename_map: Optional[Dict[str, str]] = None
    overlap_threshold: Optional[float] = 0.5


class Handler(ABC):
    """
    Abstract handler for extracting records with an expected schema from files.

    Sub-classes must implement:

    - ``_load()``: generate the record dict from a given file path. The record need not
        match the schema, as this is handled afterward.

    Sub-classes may want to specialize other methods as needed.
    """

    # Include missing fields with null values in output records.
    # By leaving out null values, the record for this handler can be accumulated over
    # several calls and paths.
    CAST_WITH_NULL = False
    # Special delete symbol for rename map.
    # In general, the purpose of the rename map is to support concise configs leveraging
    # config inheritance.
    DELETE = "__delete__"

    def __init__(
        self,
        fields: Dict[str, DataType],
        metadata: Optional[Dict[str, str]] = None,
        rename_map: Optional[Dict[str, str]] = None,
        overlap_threshold: Optional[float] = 0.5,
    ):
        self.fields = fields
        self.metadata = metadata
        self.rename_map = rename_map
        self.overlap_threshold = overlap_threshold

        # apply renaming to fields
        fields_ = self.apply_renaming(rename_map, fields)
        self.schema = create_schema(fields_, metadata=metadata)

        self._schema_names = set(self.schema.names)
        self._schema_fmt = "\n\t".join(format_schema(self.schema).split("\n"))
        self._id = hex(id(self))[2:]

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
        record = self.apply_renaming(self.rename_map, record)
        if not record:
            return record

        record_names = set(record.keys())
        schema_diff = self._schema_names - record_names
        record_diff = record_names - self._schema_names
        # |record & schema| / |schema|
        overlap = (len(record_names) - len(record_diff)) / len(self._schema_names)
        if overlap < 1.0:
            logging.info(
                f"Record overlap with schema is only {overlap:.2f}\n"
                f"\tpath: {path}\n"
                f"\tmissing fields: {schema_diff}\n"
                f"\textra fields: {record_diff}\n"
                f"\thandler: {repr(self)}"
            )
            if self.overlap_threshold and overlap < self.overlap_threshold:
                logging.warning(
                    f"Record overlap {overlap:.2f} < {self.overlap_threshold}; "
                    "discarding"
                )
                return None

        record = cast_to_schema(
            record,
            schema=self.schema,
            safe=True,
            with_null=self.CAST_WITH_NULL,
        )
        return record

    @classmethod
    def apply_renaming(
        cls, rename_map: Optional[Dict[str, str]], record: Optional[RecordDict]
    ) -> Optional[RecordDict]:
        """
        Apply a renaming map to the keys in ``record``.
        """
        if not (rename_map and record):
            return record

        # This may be a bit inefficient, as the rename_map is typically much smaller.
        # However doing it this way to guarantee the same order.
        return {
            rename_map.get(k, k): v
            for k, v in record.items()
            if rename_map.get(k) != cls.DELETE
        }

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: HandlerConfig) -> "Handler":
        """
        Initialze a Handler from a config.
        """
        raise NotImplementedError

    def id(self) -> str:
        """
        Return the unique handler ID (generated from the memory address)
        """
        return self._id

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.id()}>"

    def __str__(self) -> str:
        return (
            f"{repr(self)}:\n"
            f"\tschema: {self._schema_fmt}\n"
            f"\trename_map: {json.dumps(self.rename_map)}\n"
            f"\toverlap_threshold: {self.overlap_threshold}"
        )


class HandlerTuple(NamedTuple):
    group: Optional[str]
    pattern: List[str]
    label: str
    handler: Handler
