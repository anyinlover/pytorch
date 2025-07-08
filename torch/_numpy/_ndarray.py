# mypy: ignore-errors

from __future__ import annotations

import builtins
import math
import operator
from collections.abc import Sequence

import torch

from . import _dtypes, _dtypes_impl, _funcs, _ufuncs, _util
from ._normalizations import (
    ArrayLike,
    normalize_array_like,
    normalizer,
    NotImplementedType,
)


newaxis = None

FLAGS = [
    "C_CONTIGUOUS",
    "F_CONTIGUOUS",
    "OWNDATA",
    "WRITEABLE",
    "ALIGNED",
    "WRITEBACKIFCOPY",
    "FNC",
    "FORC",
    "BEHAVED",
    "CARRAY",
    "FARRAY",
]

SHORTHAND_TO_FLAGS = {
    "C": "C_CONTIGUOUS",
    "F": "F_CONTIGUOUS",
    "O": "OWNDATA",
    "W": "WRITEABLE",
    "A": "ALIGNED",
    "X": "WRITEBACKIFCOPY",
    "B": "BEHAVED",
    "CA": "CARRAY",
    "FA": "FARRAY",
}


class Flags:
    def __init__(self, flag_to_value: dict):
        assert all(k in FLAGS for k in flag_to_value.keys())  # sanity check
        self._flag_to_value = flag_to_value

    def __getattr__(self, attr: str):
        if attr.islower() and attr.upper() in FLAGS:
            return self[attr.upper()]
        else:
            raise AttributeError(f"No flag attribute '{attr}'")

    def __getitem__(self, key):
        if key in SHORTHAND_TO_FLAGS.keys():
            key = SHORTHAND_TO_FLAGS[key]
        if key in FLAGS:
            try:
                return self._flag_to_value[key]
            except KeyError as e:
                raise NotImplementedError(f"{key=}") from e
        else:
            raise KeyError(f"No flag key '{key}'")

    def __setattr__(self, attr, value):
        if attr.islower() and attr.upper() in FLAGS:
            self[attr.upper()] = value
        else:
            super().__setattr__(attr, value)

    def __setitem__(self, key, value):
        if key in FLAGS or key in SHORTHAND_TO_FLAGS.keys():
            raise NotImplementedError("Modifying flags is not implemented")
        else:
            raise KeyError(f"No flag key '{key}'")


def create_method(fn, name=None):
    name = name or fn.__name__

    def f(*args, **kwargs):
        return fn(*args, **kwargs)

    f.__name__ = name
    f.__qualname__ = f"ndarray.{name}"
    return f


# Map ndarray.name_method -> np.name_func
# If name_func == None, it means that name_method == name_func
methods = {
    "clip": None,
    "nonzero": None,
    "repeat": None,
    "round": None,
    "squeeze": None,
    "swapaxes": None,
    "ravel": None,
    # linalg
    "diagonal": None,
    "dot": None,
    "trace": None,
    # sorting
    "argsort": None,
    "searchsorted": None,
    # reductions
    "argmax": None,
    "argmin": None,
    "any": None,
    "all": None,
    "max": None,
    "min": None,
    "ptp": None,
    "sum": None,
    "prod": None,
    "mean": None,
    "var": None,
    "std": None,
    # scans
    "cumsum": None,
    "cumprod": None,
    # advanced indexing
    "take": None,
    "choose": None,
}

dunder = {
    "abs": "absolute",
    "invert": None,
    "pos": "positive",
    "neg": "negative",
    "gt": "greater",
    "lt": "less",
    "ge": "greater_equal",
    "le": "less_equal",
}

# dunder methods with right-looking and in-place variants
ri_dunder = {
    "add": None,
    "sub": "subtract",
    "mul": "multiply",
    "truediv": "divide",
    "floordiv": "floor_divide",
    "pow": "power",
    "mod": "remainder",
    "and": "bitwise_and",
    "or": "bitwise_or",
    "xor": "bitwise_xor",
    "lshift": "left_shift",
    "rshift": "right_shift",
    "matmul": None,
}


def _upcast_int_indices(index):
    if isinstance(index, torch.Tensor):
        if index.dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            return index.to(torch.int64)
    elif isinstance(index, tuple):
        return tuple(_upcast_int_indices(i) for i in index)
    return index


# =============================================================================
# NumPy-Compatible Advanced Indexing
# =============================================================================
#
# PyTorch and NumPy handle advanced indexing differently when indices are
# "separated" by slices, ellipsis, or scalars. NumPy moves the broadcast
# dimensions to the front, while PyTorch keeps them in place.
#
# This section implements the necessary detection and transpose logic to make
# torch._numpy match NumPy's behavior exactly.
#
# Key NumPy rule: When advanced indices are separated by non-advanced indices,
# the broadcast dimensions move to the front of the result array.
# =============================================================================


def _is_advanced_index(idx):
    """Check if an index is an advanced index (list or tensor, but not slice)."""
    return isinstance(idx, (list, torch.Tensor)) and not isinstance(idx, slice)


def _analyze_numpy_advanced_indexing(index):
    """
    Analyze indexing pattern for NumPy compatibility.

    Per NumPy spec: When advanced indices are separated by slices/ellipsis/newaxis,
    the dimensions from advanced indexing come FIRST in the result array.
    When advanced indices are adjacent, they stay in place.

    Args:
        index: The indexing tuple

    Returns:
        tuple: (index, transpose_info or None)
    """
    if not isinstance(index, tuple):
        return index, None

    # Find positions of advanced indices (lists or tensors, but not slices)
    advanced_positions = []
    for i, idx in enumerate(index):
        if isinstance(idx, (list, torch.Tensor)) and not isinstance(idx, slice):
            advanced_positions.append(i)

    if not advanced_positions:
        return index, None

    # Check if advanced indices are separated by slices/other indices
    def are_separated_by_slices(positions, index_tuple):
        """Check if advanced indices are separated by slices, ellipsis, or newaxis."""
        if len(positions) == 0:
            return False

        if len(positions) == 1:
            # Single advanced index: Based on empirical testing with NumPy,
            # vectorized indexing (broadcast dims to front) is triggered when:
            # 1. There are scalars BEFORE the advanced index, OR
            # 2. There's a slice between the advanced index and subsequent scalars

            pos = positions[0]

            # Check for scalars before the advanced index
            has_scalar_before = any(
                not isinstance(index_tuple[i], (list, torch.Tensor, slice))
                for i in range(pos)
            )

            if has_scalar_before:
                return True  # Scalar before advanced index -> vectorized indexing

            # Check for pattern: advanced index, slice, scalar(s) after
            # Example: [:, [1,2], :, 3, :] -> broadcast dims move to front
            if pos + 1 < len(index_tuple) and isinstance(index_tuple[pos + 1], slice):
                # There's a slice right after the advanced index
                if pos + 2 < len(index_tuple):
                    # Check if there are non-slice items after that slice
                    has_non_slice_after = any(
                        not isinstance(index_tuple[i], slice)
                        for i in range(pos + 2, len(index_tuple))
                    )
                    if has_non_slice_after:
                        return True  # This pattern should be separated

            return False  # All other single advanced index cases: not separated

        # Multiple advanced indices: check if they're separated by non-advanced indices
        for i in range(len(positions) - 1):
            start_pos = positions[i]
            end_pos = positions[i + 1]

            # If positions are not consecutive, check what's between them
            if end_pos - start_pos > 1:
                for j in range(start_pos + 1, end_pos):
                    idx = index_tuple[j]
                    # If there's a slice, ellipsis, or newaxis between advanced indices
                    if isinstance(idx, slice) or idx is ... or idx is None:
                        return True
                    # Also separated if there are scalars between them
                    if not isinstance(idx, (list, torch.Tensor)):
                        return True
        return False

    # NumPy rule: separated advanced indices move to front
    if are_separated_by_slices(advanced_positions, index):
        return index, {
            "advanced_positions": advanced_positions,
            "separated": True,
            "move_to_front": True,
        }

    # Adjacent indices stay in place - no transpose needed
    return index, None


def _numpy_style_advanced_indexing(tensor, index):
    """Convert NumPy-style advanced indexing to PyTorch-style for compatibility."""
    index, transpose_info = _analyze_numpy_advanced_indexing(index)
    result = tensor[index]

    if transpose_info is None or not transpose_info.get("move_to_front", False):
        return result

    # Calculate transpose axes for separated advanced indices
    advanced_positions = transpose_info["advanced_positions"]
    transpose_axes = _calculate_transpose_axes_for_separated_indices(
        result, index, advanced_positions
    )

    if transpose_axes:
        return result.permute(transpose_axes)
    else:
        return result


def _numpy_style_advanced_setitem(tensor, index, value):
    """
    Handle NumPy-style advanced indexing for setitem operations.

    For separated advanced indices, we need to transpose the value to match
    PyTorch's expected layout before assignment.
    """
    index, transpose_info = _analyze_numpy_advanced_indexing(index)

    if (
        transpose_info is not None
        and transpose_info.get("move_to_front", False)
        and hasattr(value, "ndim")
        and value.ndim > 1
    ):
        # For separated advanced indices, the value comes in NumPy layout but PyTorch
        # expects it in PyTorch's native layout. We need to undo the getitem transpose.

        # Get what PyTorch's raw indexing would produce
        raw_result = tensor[index]

        # Calculate the same transpose that getitem would apply
        advanced_positions = transpose_info["advanced_positions"]
        transpose_axes = _calculate_transpose_axes_for_separated_indices(
            raw_result, index, advanced_positions
        )

        if transpose_axes:
            # Apply inverse transpose to convert from NumPy layout back to PyTorch layout
            inverse_axes = [0] * len(transpose_axes)
            for i, axis in enumerate(transpose_axes):
                inverse_axes[axis] = i

            if hasattr(value, "permute"):
                value = value.permute(inverse_axes)
            elif hasattr(value, "transpose") and len(transpose_axes) == 2:
                # For simple 2D case, use transpose
                value = value.transpose(0, 1)
            else:
                value = torch.tensor(value).permute(inverse_axes)

    return tensor.__setitem__(index, value)


def _calculate_transpose_axes_for_separated_indices(
    result_tensor, index_tuple, advanced_positions
):
    """
    Calculate transpose axes when advanced indices are separated.

    Per NumPy spec: separated advanced indices move their broadcast dimensions
    to specific positions based on the context.

    Args:
        result_tensor: The tensor resulting from PyTorch's indexing
        index_tuple: The original indexing tuple
        advanced_positions: Positions of advanced indices

    Returns:
        list or None: Transpose axes, or None if no transpose needed
    """
    result_shape = result_tensor.shape
    result_ndim = len(result_shape)

    if result_ndim < 2:
        return None

    # For separated advanced indices, we need to determine which dimension
    # corresponds to the advanced indexing and move it to the front.

    # Strategy: Count how many dimensions come from slices before the first advanced index.
    # The advanced index dimension will be at that position + number of previous advanced dims.

    first_advanced_pos = min(advanced_positions)

    # Count slice dimensions before the first advanced index
    slice_dims_before = 0
    for i in range(first_advanced_pos):
        if isinstance(index_tuple[i], slice):
            slice_dims_before += 1

    # Count how many advanced indices come before this position (for multiple advanced indices)
    advanced_dims_before = 0
    for pos in advanced_positions:
        if pos < first_advanced_pos:
            advanced_dims_before += 1

    # The position of the first advanced index dimension in the result
    advanced_dim_pos = slice_dims_before + advanced_dims_before

    # If there's only one advanced index, it should be moved to front
    if len(advanced_positions) == 1 and advanced_dim_pos < result_ndim:
        # Move the advanced index dimension to position 0
        if advanced_dim_pos != 0:
            axes = list(range(result_ndim))
            # Remove the advanced dimension from its current position
            axes.pop(advanced_dim_pos)
            # Insert it at the front
            axes.insert(0, advanced_dim_pos)
            return axes

    # For multiple advanced indices, the broadcast shape moves to front
    # This is more complex and might need refinement based on actual cases
    elif len(advanced_positions) > 1:
        # For multiple separated advanced indices, PyTorch already produces the correct layout
        # in most cases, so we may not need a transpose at all.
        # Only apply transpose if empirically needed based on the specific pattern.

        # Check if this is a pattern where PyTorch and NumPy differ
        # For now, assume PyTorch is correct for multiple advanced indices
        return None

    return None


# Used to indicate that a parameter is unspecified (as opposed to explicitly
# `None`)
class _Unspecified:
    pass


_Unspecified.unspecified = _Unspecified()

###############################################################
#                      ndarray class                          #
###############################################################


class ndarray:
    def __init__(self, t=None):
        if t is None:
            self.tensor = torch.Tensor()
        elif isinstance(t, torch.Tensor):
            self.tensor = t
        else:
            raise ValueError(
                "ndarray constructor is not recommended; prefer"
                "either array(...) or zeros/empty(...)"
            )

    # Register NumPy functions as methods
    for method, name in methods.items():
        fn = getattr(_funcs, name or method)
        vars()[method] = create_method(fn, method)

    # Regular methods but coming from ufuncs
    conj = create_method(_ufuncs.conjugate, "conj")
    conjugate = create_method(_ufuncs.conjugate)

    for method, name in dunder.items():
        fn = getattr(_ufuncs, name or method)
        method = f"__{method}__"
        vars()[method] = create_method(fn, method)

    for method, name in ri_dunder.items():
        fn = getattr(_ufuncs, name or method)
        plain = f"__{method}__"
        vars()[plain] = create_method(fn, plain)
        rvar = f"__r{method}__"
        vars()[rvar] = create_method(lambda self, other, fn=fn: fn(other, self), rvar)
        ivar = f"__i{method}__"
        vars()[ivar] = create_method(
            lambda self, other, fn=fn: fn(self, other, out=self), ivar
        )

    # There's no __idivmod__
    __divmod__ = create_method(_ufuncs.divmod, "__divmod__")
    __rdivmod__ = create_method(
        lambda self, other: _ufuncs.divmod(other, self), "__rdivmod__"
    )

    # prevent loop variables leaking into the ndarray class namespace
    del ivar, rvar, name, plain, fn, method

    @property
    def shape(self):
        return tuple(self.tensor.shape)

    @property
    def size(self):
        return self.tensor.numel()

    @property
    def ndim(self):
        return self.tensor.ndim

    @property
    def dtype(self):
        return _dtypes.dtype(self.tensor.dtype)

    @property
    def strides(self):
        elsize = self.tensor.element_size()
        return tuple(stride * elsize for stride in self.tensor.stride())

    @property
    def itemsize(self):
        return self.tensor.element_size()

    @property
    def flags(self):
        # Note contiguous in torch is assumed C-style
        return Flags(
            {
                "C_CONTIGUOUS": self.tensor.is_contiguous(),
                "F_CONTIGUOUS": self.T.tensor.is_contiguous(),
                "OWNDATA": self.tensor._base is None,
                "WRITEABLE": True,  # pytorch does not have readonly tensors
            }
        )

    @property
    def data(self):
        return self.tensor.data_ptr()

    @property
    def nbytes(self):
        return self.tensor.storage().nbytes()

    @property
    def T(self):
        return self.transpose()

    @property
    def real(self):
        return _funcs.real(self)

    @real.setter
    def real(self, value):
        self.tensor.real = asarray(value).tensor

    @property
    def imag(self):
        return _funcs.imag(self)

    @imag.setter
    def imag(self, value):
        self.tensor.imag = asarray(value).tensor

    # ctors
    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        if order != "K":
            raise NotImplementedError(f"astype(..., order={order} is not implemented.")
        if casting != "unsafe":
            raise NotImplementedError(
                f"astype(..., casting={casting} is not implemented."
            )
        if not subok:
            raise NotImplementedError(f"astype(..., subok={subok} is not implemented.")
        if not copy:
            raise NotImplementedError(f"astype(..., copy={copy} is not implemented.")
        torch_dtype = _dtypes.dtype(dtype).torch_dtype
        t = self.tensor.to(torch_dtype)
        return ndarray(t)

    @normalizer
    def copy(self: ArrayLike, order: NotImplementedType = "C"):
        return self.clone()

    @normalizer
    def flatten(self: ArrayLike, order: NotImplementedType = "C"):
        return torch.flatten(self)

    def resize(self, *new_shape, refcheck=False):
        # NB: differs from np.resize: fills with zeros instead of making repeated copies of input.
        if refcheck:
            raise NotImplementedError(
                f"resize(..., refcheck={refcheck} is not implemented."
            )
        if new_shape in [(), (None,)]:
            return

        # support both x.resize((2, 2)) and x.resize(2, 2)
        if len(new_shape) == 1:
            new_shape = new_shape[0]
        if isinstance(new_shape, int):
            new_shape = (new_shape,)

        if builtins.any(x < 0 for x in new_shape):
            raise ValueError("all elements of `new_shape` must be non-negative")

        new_numel, old_numel = math.prod(new_shape), self.tensor.numel()

        self.tensor.resize_(new_shape)

        if new_numel >= old_numel:
            # zero-fill new elements
            assert self.tensor.is_contiguous()
            b = self.tensor.flatten()  # does not copy
            b[old_numel:].zero_()

    def view(self, dtype=_Unspecified.unspecified, type=_Unspecified.unspecified):
        if dtype is _Unspecified.unspecified:
            dtype = self.dtype
        if type is not _Unspecified.unspecified:
            raise NotImplementedError(f"view(..., type={type} is not implemented.")
        torch_dtype = _dtypes.dtype(dtype).torch_dtype
        tview = self.tensor.view(torch_dtype)
        return ndarray(tview)

    @normalizer
    def fill(self, value: ArrayLike):
        # Both Pytorch and NumPy accept 0D arrays/tensors and scalars, and
        # error out on D > 0 arrays
        self.tensor.fill_(value)

    def tolist(self):
        return self.tensor.tolist()

    def __iter__(self):
        return (ndarray(x) for x in self.tensor.__iter__())

    def __str__(self):
        return (
            str(self.tensor)
            .replace("tensor", "torch.ndarray")
            .replace("dtype=torch.", "dtype=")
        )

    __repr__ = create_method(__str__)

    def __eq__(self, other):
        try:
            return _ufuncs.equal(self, other)
        except (RuntimeError, TypeError):
            # Failed to convert other to array: definitely not equal.
            falsy = torch.full(self.shape, fill_value=False, dtype=bool)
            return asarray(falsy)

    def __ne__(self, other):
        return ~(self == other)

    def __index__(self):
        try:
            return operator.index(self.tensor.item())
        except Exception as exc:
            raise TypeError(
                "only integer scalar arrays can be converted to a scalar index"
            ) from exc

    def __bool__(self):
        return bool(self.tensor)

    def __int__(self):
        return int(self.tensor)

    def __float__(self):
        return float(self.tensor)

    def __complex__(self):
        return complex(self.tensor)

    def is_integer(self):
        try:
            v = self.tensor.item()
            result = int(v) == v
        except Exception:
            result = False
        return result

    def __len__(self):
        return self.tensor.shape[0]

    def __contains__(self, x):
        return self.tensor.__contains__(x)

    def transpose(self, *axes):
        # np.transpose(arr, axis=None) but arr.transpose(*axes)
        return _funcs.transpose(self, axes)

    def reshape(self, *shape, order="C"):
        # arr.reshape(shape) and arr.reshape(*shape)
        return _funcs.reshape(self, shape, order=order)

    def sort(self, axis=-1, kind=None, order=None):
        # ndarray.sort works in-place
        _funcs.copyto(self, _funcs.sort(self, axis, kind, order))

    def item(self, *args):
        # Mimic NumPy's implementation with three special cases (no arguments,
        # a flat index and a multi-index):
        # https://github.com/numpy/numpy/blob/main/numpy/_core/src/multiarray/methods.c#L702
        if args == ():
            return self.tensor.item()
        elif len(args) == 1:
            # int argument
            return self.ravel()[args[0]]
        else:
            return self.__getitem__(args)

    def __getitem__(self, index):
        tensor = self.tensor

        def neg_step(i, s):
            if not (isinstance(s, slice) and s.step is not None and s.step < 0):
                return s

            nonlocal tensor
            tensor = torch.flip(tensor, (i,))

            # Account for the fact that a slice includes the start but not the end
            assert isinstance(s.start, int) or s.start is None
            assert isinstance(s.stop, int) or s.stop is None
            start = s.stop + 1 if s.stop else None
            stop = s.start + 1 if s.start else None

            return slice(start, stop, -s.step)

        if isinstance(index, Sequence):
            index = type(index)(neg_step(i, s) for i, s in enumerate(index))
        else:
            index = neg_step(0, index)
        index = _util.ndarrays_to_tensors(index)
        index = _upcast_int_indices(index)

        # Use NumPy-style advanced indexing for compatibility
        result = _numpy_style_advanced_indexing(tensor, index)
        return ndarray(result)

    def __setitem__(self, index, value):
        index = _util.ndarrays_to_tensors(index)
        index = _upcast_int_indices(index)

        if not _dtypes_impl.is_scalar(value):
            value = normalize_array_like(value)
            value = _util.cast_if_needed(value, self.tensor.dtype)

        # Use NumPy-style advanced indexing for setitem compatibility
        return _numpy_style_advanced_setitem(self.tensor, index, value)

    take = _funcs.take
    put = _funcs.put

    def __dlpack__(self, *, stream=None):
        return self.tensor.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


def _tolist(obj):
    """Recursively convert tensors into lists."""
    a1 = []
    for elem in obj:
        if isinstance(elem, (list, tuple)):
            elem = _tolist(elem)
        if isinstance(elem, ndarray):
            a1.append(elem.tensor.tolist())
        else:
            a1.append(elem)
    return a1


# This is the ideally the only place which talks to ndarray directly.
# The rest goes through asarray (preferred) or array.


def array(obj, dtype=None, *, copy=True, order="K", subok=False, ndmin=0, like=None):
    if subok is not False:
        raise NotImplementedError("'subok' parameter is not supported.")
    if like is not None:
        raise NotImplementedError("'like' parameter is not supported.")
    if order != "K":
        raise NotImplementedError

    # a happy path
    if (
        isinstance(obj, ndarray)
        and copy is False
        and dtype is None
        and ndmin <= obj.ndim
    ):
        return obj

    if isinstance(obj, (list, tuple)):
        # FIXME and they have the same dtype, device, etc
        if obj and all(isinstance(x, torch.Tensor) for x in obj):
            # list of arrays: *under torch.Dynamo* these are FakeTensors
            obj = torch.stack(obj)
        else:
            # XXX: remove tolist
            # lists of ndarrays: [1, [2, 3], ndarray(4)] convert to lists of lists
            obj = _tolist(obj)

    # is obj an ndarray already?
    if isinstance(obj, ndarray):
        obj = obj.tensor

    # is a specific dtype requested?
    torch_dtype = None
    if dtype is not None:
        torch_dtype = _dtypes.dtype(dtype).torch_dtype

    tensor = _util._coerce_to_tensor(obj, torch_dtype, copy, ndmin)
    return ndarray(tensor)


def asarray(a, dtype=None, order="K", *, like=None):
    return array(a, dtype=dtype, order=order, like=like, copy=False, ndmin=0)


def ascontiguousarray(a, dtype=None, *, like=None):
    arr = asarray(a, dtype=dtype, like=like)
    if not arr.tensor.is_contiguous():
        arr.tensor = arr.tensor.contiguous()
    return arr


def from_dlpack(x, /):
    t = torch.from_dlpack(x)
    return ndarray(t)


def _extract_dtype(entry):
    try:
        dty = _dtypes.dtype(entry)
    except Exception:
        dty = asarray(entry).dtype
    return dty


def can_cast(from_, to, casting="safe"):
    from_ = _extract_dtype(from_)
    to_ = _extract_dtype(to)

    return _dtypes_impl.can_cast_impl(from_.torch_dtype, to_.torch_dtype, casting)


def result_type(*arrays_and_dtypes):
    tensors = []
    for entry in arrays_and_dtypes:
        try:
            t = asarray(entry).tensor
        except (RuntimeError, ValueError, TypeError):
            dty = _dtypes.dtype(entry)
            t = torch.empty(1, dtype=dty.torch_dtype)
        tensors.append(t)

    torch_dtype = _dtypes_impl.result_type_impl(*tensors)
    return _dtypes.dtype(torch_dtype)
