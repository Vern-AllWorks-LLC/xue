"""
Tensor type with runtime shape validation and type-checker plugin support.

Patent-safe approach: uses runtime checking and standard Python generics
(not liquid types). Shape validation happens at construction and operation
time, catching shape mismatches with clear error messages.

Usage:
    from xue.tensor import Tensor, float32, float64, int32

    # Create typed, shape-checked tensors
    a = Tensor[[3, 4], float32]([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    b = Tensor[[4, 2], float32]([[1,2], [3,4], [5,6], [7,8]])
    c = a @ b  # Tensor[[3, 2], float32] — shape checked!

    # Shape mismatch caught at runtime
    d = Tensor[[3, 3], float32]([[1,2,3]])  # ShapeError!

    # Dynamic shape
    e = Tensor(data=[[1, 2], [3, 4]])  # infers shape [2, 2], dtype float64
"""

from __future__ import annotations
import typing as _t
import math
import array as _array


# --- Dtype definitions ---

class DType:
    """Represents a tensor element data type."""

    __slots__ = ("name", "size", "python_type", "typecode")

    def __init__(self, name: str, size: int, python_type: type,
                 typecode: str) -> None:
        self.name = name
        self.size = size
        self.python_type = python_type
        self.typecode = typecode

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        if isinstance(other, DType):
            return self.name == other.name
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)


float16 = DType("float16", 2, float, "e")
float32 = DType("float32", 4, float, "f")
float64 = DType("float64", 8, float, "d")
int8 = DType("int8", 1, int, "b")
int16 = DType("int16", 2, int, "h")
int32 = DType("int32", 4, int, "i")
int64 = DType("int64", 8, int, "q")
uint8 = DType("uint8", 1, int, "B")
bool_ = DType("bool", 1, bool, "b")


class ShapeError(ValueError):
    """Raised when tensor shapes are incompatible."""
    pass


class Tensor:
    """N-dimensional tensor with runtime shape validation.

    Supports basic arithmetic, matrix multiplication, reshaping, and slicing.
    Data is stored as a flat array with stride-based indexing.
    """

    __slots__ = ("_data", "_shape", "_strides", "_dtype", "_offset")

    def __init__(self, data=None, shape: list[int] | tuple[int, ...] | None = None,
                 dtype: DType = float64) -> None:
        if data is not None:
            flat, inferred_shape = _flatten(data)
            if shape is not None:
                shape = tuple(shape)
                expected_size = 1
                for s in shape:
                    expected_size *= s
                if expected_size != len(flat):
                    raise ShapeError(
                        f"Data has {len(flat)} elements but shape {list(shape)} "
                        f"requires {expected_size}"
                    )
            else:
                shape = inferred_shape

            self._data = _array.array(dtype.typecode, [dtype.python_type(x) for x in flat])
        else:
            if shape is None:
                raise ValueError("Either data or shape must be provided")
            shape = tuple(shape)
            size = 1
            for s in shape:
                size *= s
            self._data = _array.array(dtype.typecode, [dtype.python_type(0)] * size)

        self._shape = shape
        self._dtype = dtype
        self._strides = _compute_strides(shape)
        self._offset = 0

    # --- Class-level subscript for type annotation syntax ---
    # Tensor[[3, 4], float32] returns a TensorFactory
    def __class_getitem__(cls, params):
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("Tensor requires [shape, dtype], e.g. Tensor[[3, 4], float32]")
        shape, dtype = params
        if isinstance(shape, list):
            shape = tuple(shape)
        return _TensorFactory(shape, dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        s = 1
        for d in self._shape:
            s *= d
        return s

    def _flat_index(self, indices: tuple[int, ...]) -> int:
        idx = self._offset
        for i, s in zip(indices, self._strides):
            idx += i * s
        return idx

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            key = (key,)
        if isinstance(key, tuple):
            # Check for slicing
            if any(isinstance(k, slice) for k in key):
                return self._slice(key)
            # Direct element access
            if len(key) != self.ndim:
                raise IndexError(
                    f"Expected {self.ndim} indices, got {len(key)}"
                )
            return self._data[self._flat_index(key)]
        raise TypeError(f"Invalid index type: {type(key)}")

    def __setitem__(self, key, value):
        if isinstance(key, (int, slice)):
            key = (key,)
        if isinstance(key, tuple) and len(key) == self.ndim:
            self._data[self._flat_index(key)] = self._dtype.python_type(value)
        else:
            raise TypeError(f"Invalid index for assignment")

    def _slice(self, key):
        """Basic slicing support — returns a new Tensor."""
        ranges = []
        new_shape = []
        for i, k in enumerate(key):
            if isinstance(k, int):
                ranges.append(range(k, k + 1))
            elif isinstance(k, slice):
                r = range(*k.indices(self._shape[i]))
                ranges.append(r)
                new_shape.append(len(r))
            else:
                raise TypeError(f"Invalid slice component: {type(k)}")

        # Pad remaining dimensions
        for i in range(len(key), self.ndim):
            ranges.append(range(self._shape[i]))
            new_shape.append(self._shape[i])

        # Collect data
        flat = []
        self._slice_recursive(ranges, 0, (), flat)
        return Tensor(data=flat, shape=new_shape, dtype=self._dtype)

    def _slice_recursive(self, ranges, dim, prefix, flat):
        if dim == len(ranges):
            flat.append(self._data[self._flat_index(prefix)])
            return
        for i in ranges[dim]:
            self._slice_recursive(ranges, dim + 1, prefix + (i,), flat)

    # --- Arithmetic ---

    def _make_result(self):
        """Create a zero tensor with same shape/dtype, returning (tensor, data_array)."""
        r = Tensor(shape=self._shape, dtype=self._dtype)
        return r, r._data

    def __add__(self, other):
        sd = self._data
        if isinstance(other, (int, float)):
            r, rd = self._make_result()
            for i in range(len(sd)):
                rd[i] = sd[i] + other
            return r
        if isinstance(other, Tensor):
            if self._shape != other._shape:
                raise ShapeError(f"Shape mismatch: {list(self._shape)} vs {list(other._shape)}")
            od = other._data
            r, rd = self._make_result()
            for i in range(len(sd)):
                rd[i] = sd[i] + od[i]
            return r
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        sd = self._data
        if isinstance(other, (int, float)):
            r, rd = self._make_result()
            for i in range(len(sd)):
                rd[i] = sd[i] - other
            return r
        if isinstance(other, Tensor):
            if self._shape != other._shape:
                raise ShapeError(f"Shape mismatch: {list(self._shape)} vs {list(other._shape)}")
            od = other._data
            r, rd = self._make_result()
            for i in range(len(sd)):
                rd[i] = sd[i] - od[i]
            return r
        return NotImplemented

    def __rsub__(self, other):
        sd = self._data
        if isinstance(other, (int, float)):
            r, rd = self._make_result()
            for i in range(len(sd)):
                rd[i] = other - sd[i]
            return r
        return NotImplemented

    def __mul__(self, other):
        sd = self._data
        if isinstance(other, (int, float)):
            r, rd = self._make_result()
            for i in range(len(sd)):
                rd[i] = sd[i] * other
            return r
        if isinstance(other, Tensor):
            if self._shape != other._shape:
                raise ShapeError(f"Shape mismatch: {list(self._shape)} vs {list(other._shape)}")
            od = other._data
            r, rd = self._make_result()
            for i in range(len(sd)):
                rd[i] = sd[i] * od[i]
            return r
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        sd = self._data
        if isinstance(other, (int, float)):
            r, rd = self._make_result()
            for i in range(len(sd)):
                rd[i] = sd[i] / other
            return r
        if isinstance(other, Tensor):
            if self._shape != other._shape:
                raise ShapeError(f"Shape mismatch: {list(self._shape)} vs {list(other._shape)}")
            od = other._data
            r, rd = self._make_result()
            for i in range(len(sd)):
                rd[i] = sd[i] / od[i]
            return r
        return NotImplemented

    def __neg__(self):
        r, rd = self._make_result()
        sd = self._data
        for i in range(len(sd)):
            rd[i] = -sd[i]
        return r

    def __matmul__(self, other: Tensor) -> Tensor:
        """Matrix multiplication with shape checking."""
        if not isinstance(other, Tensor):
            return NotImplemented
        if self.ndim < 2 or other.ndim < 2:
            raise ShapeError("Matrix multiply requires at least 2D tensors")
        if self._shape[-1] != other._shape[-2]:
            raise ShapeError(
                f"Matrix multiply shape mismatch: "
                f"{list(self._shape)} @ {list(other._shape)} — "
                f"inner dimensions {self._shape[-1]} != {other._shape[-2]}"
            )
        m, k = self._shape[-2], self._shape[-1]
        n = other._shape[-1]
        result = Tensor(shape=(m, n), dtype=self._dtype)
        # Direct data access — avoid __getitem__ overhead in inner loop
        a_data = self._data
        b_data = other._data
        r_data = result._data
        a_off = self._offset
        b_off = other._offset
        a_stride0 = self._strides[-2]
        b_stride0 = other._strides[-2]
        for i in range(m):
            a_row = a_off + i * a_stride0
            r_row = i * n
            for j in range(n):
                s = 0.0
                for p in range(k):
                    s += a_data[a_row + p] * b_data[b_off + p * b_stride0 + j]
                r_data[r_row + j] = s
        return result

    # --- Shape operations ---

    def reshape(self, *new_shape: int) -> Tensor:
        """Reshape tensor to new shape (total elements must match)."""
        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            new_shape = tuple(new_shape[0])
        new_size = 1
        for s in new_shape:
            new_size *= s
        if new_size != self.size:
            raise ShapeError(
                f"Cannot reshape {list(self._shape)} ({self.size} elements) "
                f"to {list(new_shape)} ({new_size} elements)"
            )
        result = Tensor(shape=new_shape, dtype=self._dtype)
        result._data = _array.array(self._dtype.typecode, self._data)
        return result

    def transpose(self) -> Tensor:
        """Transpose a 2D tensor."""
        if self.ndim != 2:
            raise ShapeError(f"Transpose requires 2D tensor, got {self.ndim}D")
        m, n = self._shape
        result = Tensor(shape=(n, m), dtype=self._dtype)
        for i in range(m):
            for j in range(n):
                result[j, i] = self[i, j]
        return result

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def sum(self, axis: int | None = None) -> float | Tensor:
        """Sum elements, optionally along an axis."""
        if axis is None:
            return sum(self._data)
        # TODO: axis-specific reduction
        raise NotImplementedError("Axis-specific sum not yet implemented")

    def mean(self, axis: int | None = None) -> float | Tensor:
        if axis is None:
            return sum(self._data) / self.size
        raise NotImplementedError("Axis-specific mean not yet implemented")

    def max(self, axis: int | None = None):
        if axis is None:
            return max(self._data)
        raise NotImplementedError("Axis-specific max not yet implemented")

    def min(self, axis: int | None = None):
        if axis is None:
            return min(self._data)
        raise NotImplementedError("Axis-specific min not yet implemented")

    def tolist(self) -> list:
        """Convert tensor to nested Python list."""
        if self.ndim == 1:
            return list(self._data[self._offset:self._offset + self._shape[0]])
        result = []
        stride = self._strides[0]
        for i in range(self._shape[0]):
            sub = Tensor(shape=self._shape[1:], dtype=self._dtype)
            start = self._offset + i * stride
            end = start + stride
            sub._data = _array.array(self._dtype.typecode, self._data[start:end])
            result.append(sub.tolist())
        return result

    def __repr__(self) -> str:
        data_str = self.tolist()
        return f"Tensor({data_str}, shape={list(self._shape)}, dtype={self._dtype})"

    def __len__(self) -> int:
        return self._shape[0] if self._shape else 0

    def __eq__(self, other) -> bool:
        if isinstance(other, Tensor):
            return self._shape == other._shape and list(self._data) == list(other._data)
        return NotImplemented


class _TensorFactory:
    """Created by Tensor[[shape], dtype] syntax to construct shape-checked tensors."""

    __slots__ = ("_shape", "_dtype")

    def __init__(self, shape: tuple[int, ...], dtype: DType) -> None:
        self._shape = shape
        self._dtype = dtype

    def __call__(self, data) -> Tensor:
        return Tensor(data=data, shape=list(self._shape), dtype=self._dtype)

    def __repr__(self) -> str:
        return f"Tensor[{list(self._shape)}, {self._dtype}]"

    def zeros(self) -> Tensor:
        return Tensor(shape=list(self._shape), dtype=self._dtype)

    def ones(self) -> Tensor:
        t = Tensor(shape=list(self._shape), dtype=self._dtype)
        for i in range(t.size):
            t._data[i] = self._dtype.python_type(1)
        return t


# --- Utility functions ---

def _flatten(data) -> tuple[list, tuple[int, ...]]:
    """Recursively flatten nested lists/tuples and infer shape."""
    if isinstance(data, (int, float)):
        return [data], ()
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Cannot create tensor from {type(data)}")

    if len(data) == 0:
        return [], (0,)

    sub_flats = []
    sub_shape = None
    for item in data:
        flat, shape = _flatten(item)
        if sub_shape is not None and shape != sub_shape:
            raise ShapeError(f"Inconsistent sub-array shapes: {sub_shape} vs {shape}")
        sub_shape = shape
        sub_flats.extend(flat)

    return sub_flats, (len(data),) + (sub_shape or ())


def _compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Compute row-major strides for a shape."""
    strides = []
    stride = 1
    for s in reversed(shape):
        strides.append(stride)
        stride *= s
    return tuple(reversed(strides))


# --- Factory functions ---

def zeros(shape: list[int] | tuple[int, ...], dtype: DType = float64) -> Tensor:
    return Tensor(shape=shape, dtype=dtype)


def ones(shape: list[int] | tuple[int, ...], dtype: DType = float64) -> Tensor:
    t = Tensor(shape=shape, dtype=dtype)
    for i in range(t.size):
        t._data[i] = dtype.python_type(1)
    return t


def eye(n: int, dtype: DType = float64) -> Tensor:
    """Create an n x n identity matrix."""
    t = Tensor(shape=(n, n), dtype=dtype)
    for i in range(n):
        t[i, i] = 1
    return t


def arange(start: float, stop: float | None = None, step: float = 1.0,
           dtype: DType = float64) -> Tensor:
    """Create a 1D tensor with evenly spaced values."""
    if stop is None:
        start, stop = 0, start
    data = []
    v = start
    while v < stop:
        data.append(v)
        v += step
    return Tensor(data=data, dtype=dtype)

# ── C-accelerated override ────────────────────────────────────────
# Replace Python classes with C implementations.
# SIMD (AVX2) elementwise ops and tiled matmul. Optional CUDA GPU.
try:
    from ._tensor_accel import (
        DType as DType,
        Tensor as Tensor,
        ShapeError as ShapeError,
        float16 as float16,
        float32 as float32,
        float64 as float64,
        int8 as int8,
        int16 as int16,
        int32 as int32,
        int64 as int64,
        uint8 as uint8,
        bool_ as bool_,
        zeros as zeros,
        ones as ones,
        eye as eye,
        arange as arange,
    )
except ImportError:
    pass
