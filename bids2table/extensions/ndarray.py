from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pyarrow as pa
from pandas.api.extensions import ExtensionDtype, register_extension_dtype
from pandas.core.arrays import PandasArray

from .base import PaExtensionArray, PaPyExtensionType

__all__ = [
    "PaNDArrayType",
    "PaNDArrayScalar",
    "PaNDArrayArray",
    "PdNDArrayDtype",
    "PdNDArrayArray",
]


class PaNDArrayType(PaPyExtensionType):
    """
    PyArrow ndarray extension type.

    See `here <https://arrow.apache.org/docs/python/extending_types.html>`_ for more
    details on extension types.
    """

    def __init__(self, item_type: pa.DataType):
        fields = {
            "data": pa.list_(item_type),
            "shape": pa.list_(pa.int64()),
        }
        self.item_type = item_type
        super().__init__(pa.struct(fields))

    def __reduce__(self):
        return PaNDArrayType, (self.item_type,)

    def __arrow_ext_scalar_class__(self):
        return PaNDArrayScalar

    def __arrow_ext_class__(self):
        return PaNDArrayArray

    def to_pandas_dtype(self):
        return PdNDArrayDtype()

    def pack(self, value: np.ndarray) -> Dict[str, Any]:
        """
        Pack an object so it can be consumed by pyarrow with this type
        """
        dtype = self.item_type.to_pandas_dtype()
        data = value.flatten().astype(dtype)
        return {"data": data, "shape": value.shape}


class PaNDArrayScalar(pa.ExtensionScalar):
    """
    PyArrow ndarray scalar that unflattens in python conversion (``scalar.as_py()``).
    """

    def as_py(self) -> Any:
        val = self.value
        if val is not None:
            data = val["data"].values.to_numpy()
            shape = val["shape"].as_py()
            val = data.reshape(shape)
        return val


class PaNDArrayArray(PaExtensionArray):
    """
    PyArrow ndarray array that unpacks and unflattens in python conversion
    (``array.to_pylist()``) and numpy conversion (``array.to_numpy()``).
    """

    def to_numpy(self, **kwargs):
        pa_type = self.type
        if not isinstance(pa_type, PaNDArrayType):
            raise TypeError(f"Expected array of type PaNDArrayType, got {pa_type}")
        item_dtype = pa_type.item_type.to_pandas_dtype()
        return _array_of_arrays(self.to_pylist(), item_dtype=item_dtype)

    @classmethod
    def from_sequence(
        cls,
        values: Iterable[np.ndarray],
        *,
        item_dtype: Any = None,
    ) -> "PaNDArrayArray":
        """
        Construct an array from a python iterable, first flattening and packing the
        ``values`` as structs.
        """
        if item_dtype is None:
            item_dtype = _infer_dtype(values)

        typ = PaNDArrayType(pa.from_numpy_dtype(item_dtype))
        storage = [typ.pack(v) for v in values]
        storage = pa.array(storage, type=typ.storage_type)
        array = cls.from_storage(typ, storage)
        return array


@register_extension_dtype
class PdNDArrayDtype(ExtensionDtype):
    """
    Pandas extension dtype for ndarrays supporting conversion to a PyArrow struct-backed
    extension type (``PaNDArrayType``).

    See `here
    <https://pandas.pydata.org/docs/development/extending.html#extension-types>`_ for
    more details on extension types.
    """

    name = "ndarray"
    type = object
    kind = "O"
    base = np.dtype("O")
    na_value = None

    @classmethod
    def construct_array_type(cls):
        """Return the array type associated with this dtype."""
        return PdNDArrayArray

    def __from_arrow__(
        self, array: Union[pa.Array, pa.ChunkedArray]
    ) -> "PdNDArrayArray":
        pa_type = array.type
        if not isinstance(pa_type, PaNDArrayType):
            raise TypeError(f"Expected array of type PaNDArrayType, got {pa_type}")
        item_dtype = pa_type.item_type.to_pandas_dtype()
        return PdNDArrayArray(array.to_pylist(), item_dtype=item_dtype)

    def __repr__(self) -> str:
        return "PdNDArrayDtype()"


class PdNDArrayArray(PandasArray):
    """
    Pandas extension array for ndarrays supporting conversion to a PyArrow struct-backed
    extension type (``PaNDArrayType``).
    """

    _typ = "extension"
    _dtype = PdNDArrayDtype()
    _internal_fill_value = None
    _str_na_value = None

    def __init__(
        self,
        values: Iterable[np.ndarray],
        *,
        copy: bool = False,
        item_dtype: Any = None,
    ):
        values_ = _array_of_arrays(values, item_dtype=item_dtype)
        if copy:
            values_ = values_.copy()

        super(PandasArray, self).__init__(values_, PdNDArrayDtype())
        self.item_dtype = item_dtype

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Any = None, copy: bool = False
    ) -> "PdNDArrayArray":
        return PdNDArrayArray(scalars, copy=copy)

    def __arrow_array__(self, type: Optional[pa.DataType] = None) -> PaNDArrayArray:
        """
        Convert myself into a PyArrow array
        """
        return PaNDArrayArray.from_sequence(self._ndarray, item_dtype=self.item_dtype)


def _array_of_arrays(
    values: Iterable[np.ndarray], item_dtype: Any = None
) -> np.ndarray:
    """
    Construct a 1d array of numpy arrays.
    """
    if item_dtype is None:
        item_dtype = _infer_dtype(values)

    # Our goal is to construct a one-dimensional array of arrays. But numpy will
    # auto-stack the arrays into a single ndarray if the constituents happen to be all
    # the same shape. As a hack workaround, we initialize with an empty array, which we
    # remove below.
    values_ = [np.array([], dtype=item_dtype)]
    for v in values:
        v_ = np.asarray(v, dtype=item_dtype)
        values_.append(v_)
    array = np.asarray(values_, dtype=object)[1:]
    return array


def _infer_dtype(values: Iterable[np.ndarray]) -> np.dtype:
    for v in values:
        v_ = np.asarray(v)
        if v_.dtype != object:
            return v_.dtype
        if v_.size > 0:
            return np.dtype(type(v_.flatten()[0]))
    raise ValueError("Can't infer dtype from empty arrays of type object")
