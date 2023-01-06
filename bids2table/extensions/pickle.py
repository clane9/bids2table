import pickle
from typing import Any, Iterable, Optional, Union

import numpy as np
import pyarrow as pa
from pandas.api.extensions import ExtensionDtype, register_extension_dtype
from pandas.core.arrays import PandasArray

from .base import PaExtensionArray, PaPyExtensionType

__all__ = [
    "PaPickleType",
    "PaPickleScalar",
    "PaPickleArray",
    "PdPickleDtype",
    "PdPickleArray",
]


class PaPickleType(PaPyExtensionType):
    """
    PyArrow binary extension type for holding arbitrary pickled objects.

    See `here <https://arrow.apache.org/docs/python/extending_types.html>`_ for more
    details on extension types.
    """

    def __init__(self):
        super().__init__(pa.binary())

    def __reduce__(self):
        return PaPickleType, ()

    def __arrow_ext_scalar_class__(self):
        return PaPickleScalar

    def __arrow_ext_class__(self):
        return PaPickleArray

    def to_pandas_dtype(self):
        return PdPickleDtype()

    def pack(self, value: Any) -> bytes:
        """
        Pack an object by serializing with pickle
        """
        return pickle.dumps(value)

    def __str__(self) -> str:
        return "pickle"


class PaPickleScalar(pa.ExtensionScalar):
    """
    PyArrow object scalar that deserializes with pickle in python conversion
    (``scalar.as_py()``).
    """

    def as_py(self) -> Any:
        val = self.value
        if val is not None:
            val = pickle.loads(val.as_py())
        return val


class PaPickleArray(PaExtensionArray):
    """
    PyArrow pickle array that deserializes with pickle in python conversion
    (``array.to_pylist()``) and numpy conversion (``array.to_numpy()``).
    """

    def to_numpy(self, **kwargs):
        return np.array(self.to_pylist(), dtype=object)

    @classmethod
    def from_sequence(cls, values: Iterable[Any]) -> "PaPickleArray":
        """
        Construct an array from a python sequence, first serializing the ``values``
        using pickle.
        """
        typ = PaPickleType()
        storage = [typ.pack(val) for val in values]
        storage = pa.array(storage, type=typ.storage_type)
        array = cls.from_storage(typ, storage)
        return array


@register_extension_dtype
class PdPickleDtype(ExtensionDtype):
    """
    Pandas extension dtype for arbitrary objects supporting conversion to a PyArrow
    binary extension type (``PaPickleType``) via pickle serialization.

    See `here
    <https://pandas.pydata.org/docs/development/extending.html#extension-types>`_ for
    more details on extension types.
    """

    name = "pickle"
    type = object
    kind = "O"
    base = np.dtype("O")
    na_value = None

    @classmethod
    def construct_array_type(cls):
        """
        Return the array type associated with this dtype.
        """
        return PdPickleArray

    def __from_arrow__(
        self, array: Union[pa.Array, pa.ChunkedArray]
    ) -> "PdPickleArray":
        data = np.array(array.to_pylist(), dtype=object)
        return PdPickleArray(data)

    def __repr__(self) -> str:
        return "PdPickleDtype()"


class PdPickleArray(PandasArray):
    """
    Pandas extension array for arbitrary objects supporting conversion to PyArrow (via
    pickle serialization).
    """

    # In pandas.core.internals.api.make_block, PandasArray instances are automatically
    # unboxed to plain NumPy arrays. But the result is one-dimensional. When
    # `len(placement) == 1` (as is the case inside `Table.to_pandas()`), a 2d row
    # vector seems to be expected. As a result, `check_ndim` raises errors like:
    #
    #   ValueError: Wrong number of items passed 2, placement implies 1
    #
    # To bypass this shape mismatch issue, we make sure this PandasArray sub-class is
    # *not* automatically unboxed, by changing this _typ attribute. Searching for
    # ABCPandasArray in the pandas codebase gives a bit more background why this works.
    #
    # TLDR: we need to reset _typ for `Table.to_pandas()` to work properly.
    _typ = "extension"
    _dtype = PdPickleDtype()
    _internal_fill_value = None
    _str_na_value = None

    def __init__(self, values: np.ndarray, copy: bool = False):
        values = np.asarray(values, dtype=object)
        if values.ndim != 1:
            raise ValueError("Only one-dimensional arrays supported")
        if copy:
            values = values.copy()
        super(PandasArray, self).__init__(values, PdPickleDtype())

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Any = None, copy: bool = False
    ) -> "PdPickleArray":
        return PdPickleArray(scalars)

    def __arrow_array__(self, type: Optional[pa.DataType] = None) -> PaPickleArray:
        return PaPickleArray.from_sequence(self._ndarray)
