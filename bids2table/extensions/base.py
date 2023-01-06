from typing import Any, Iterable

import pyarrow as pa

__all__ = ["PaPyExtensionType", "PaExtensionArray"]


# TODO: some registry to so that pack can be called easily within handlers


class PaPyExtensionType(pa.PyExtensionType):
    """
    A shallow sub-class of ``pyarrow.PyExtensionType`` that adds methods to be
    implemented:

        - pack()
    """

    def pack(self, value: Any) -> Any:
        """
        Pack an object so it can be directly consumed by pyarrow as this type
        """
        raise NotImplementedError


class PaExtensionArray(pa.ExtensionArray):
    """
    A shallow sub-class of ``pyarrow.ExtensionArray`` that adds methods to be
    implemented:

        - from_sequence()
    """

    @classmethod
    def from_sequence(cls, values: Iterable[Any]) -> "PaExtensionArray":
        """
        Construct an array from a python sequence, first packing each of the ``values``
        according to the matching extension type.
        """
        raise NotImplementedError
