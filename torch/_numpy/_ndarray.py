# mypy: ignore-errors

from __future__ import annotations

import builtins
import math
import operator
from collections.abc import Sequence
from enum import IntFlag

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
# This implementation closely follows NumPy's proven state machine from
# numpy/_core/src/multiarray/mapping.c (mapiter_fill_info function).
#
# Key NumPy rule: When advanced indices are separated by non-advanced indices,
# the broadcast dimensions move to the front of the result array.
# =============================================================================

# Index type constants matching NumPy's mapping.c
# Pythonic index type classification using IntFlag


class IndexType(IntFlag):
    INTEGER = 1
    NEWAXIS = 2
    SLICE = 4
    ELLIPSIS = 8
    FANCY = 16
    BOOL = 32
    SCALAR_ARRAY = 64
    BOOL_0D = FANCY | 128


def _classify_index(idx):
    """Classify an index element according to NumPy's indexing rules."""
    # Direct type mapping for common cases
    type_map = {
        type(None): IndexType.NEWAXIS,
        type(...): IndexType.ELLIPSIS,
        slice: IndexType.SLICE,
        int: IndexType.INTEGER,
        list: IndexType.FANCY,
    }

    if type(idx) in type_map:
        return type_map[type(idx)]

    # Handle tensors with dtype/ndim checks
    if isinstance(idx, torch.Tensor):
        if idx.dtype == torch.bool:
            return IndexType.BOOL_0D if idx.ndim == 0 else IndexType.BOOL
        return IndexType.INTEGER if idx.ndim == 0 else IndexType.FANCY

    # Default for numpy scalars, etc.
    return IndexType.INTEGER


def _analyze_numpy_advanced_indexing(index):
    """
    Analyze indexing pattern using NumPy's consecutive/separated detection logic.

    Returns (index, transpose_info) where transpose_info indicates if broadcast
    dimensions need to be moved to the front for NumPy compatibility.
    """
    index = index if isinstance(index, tuple) else (index,)
    index_types = [_classify_index(idx) for idx in index]

    # NumPy's state machine: -1=init, 0=found_advanced, 1=gap_after_advanced, 2=separated
    state = -1
    consec_position = 0
    result_dim = 0
    advanced_positions = []

    for i, idx_type in enumerate(index_types):
        is_advanced = idx_type & (IndexType.FANCY | IndexType.INTEGER)

        if is_advanced:
            advanced_positions.append(i)
            if state == -1:  # First advanced index
                consec_position = result_dim
                state = 0
            elif state == 1:  # Advanced after gap -> separated
                state = 2
                consec_position = 0
        else:
            if state == 0:  # Gap after advanced index
                state = 1

        # Count dimensions added by slices/newaxis
        if idx_type in (IndexType.SLICE, IndexType.NEWAXIS):
            result_dim += 1

    if not advanced_positions:
        return index, None

    # Separated if explicitly detected or starting at position 0 with advanced indices
    is_separated = state == 2 or (consec_position == 0 and state != -1)

    return (
        (
            index,
            {
                "advanced_positions": advanced_positions,
                "separated": is_separated,
                "move_to_front": is_separated,
                "consec_position": consec_position,
                "consec_status": state,
            },
        )
        if is_separated
        else (index, None)
    )


def _numpy_style_advanced_indexing(tensor, index):
    """Convert NumPy-style advanced indexing to PyTorch-style for compatibility."""
    index, transpose_info = _analyze_numpy_advanced_indexing(index)
    result = tensor[index]

    if transpose_info is None or not transpose_info.get("move_to_front", False):
        return result

    # Calculate transpose axes for separated advanced indices using NumPy's logic
    advanced_positions = transpose_info["advanced_positions"]
    transpose_axes = _calculate_transpose_axes_for_separated_indices(
        result, index, advanced_positions, transpose_info
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

        # Calculate the same transpose that getitem would apply using NumPy's logic
        advanced_positions = transpose_info["advanced_positions"]
        transpose_axes = _calculate_transpose_axes_for_separated_indices(
            raw_result, index, advanced_positions, transpose_info
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


def _numpy_get_transpose(fancy_ndim, consec, ndim, getmap):
    """Calculate transpose axes using NumPy's algorithm for consecutive indices."""
    if consec == 0 or fancy_ndim == 0:
        return None

    n1, n2, n3 = fancy_ndim, consec, ndim
    bnd = n1 if getmap else n2

    # Build transpose axes in three parts following NumPy's pattern
    return (
        list(range(bnd, n1 + n2))  # First part
        + list(range(bnd))  # Second part
        + list(range(n1 + n2, n3))
    )  # Third part


def _calculate_transpose_axes_for_separated_indices(
    result_tensor, index_tuple, advanced_positions, transpose_info
):
    """
    Calculate transpose axes when advanced indices are separated.

    For separated indices, PyTorch keeps dims in place but NumPy moves broadcast
    dims to front. We only transpose when needed.
    """
    result_ndim = result_tensor.ndim
    if result_ndim < 2:
        return None

    consec_position = transpose_info.get("consec_position", 0)
    consec_status = transpose_info.get("consec_status", -1)

    if consec_status == 2:  # Separated indices
        # Count actual fancy indices (not just scalars)
        fancy_indices = [
            pos
            for pos in advanced_positions
            if isinstance(index_tuple[pos], (list, torch.Tensor))
            and not isinstance(index_tuple[pos], slice)
        ]

        if len(fancy_indices) == 1:
            # Single separated fancy index: always needs transpose for NumPy compatibility
            fancy_pos = fancy_indices[0]
            slice_dims_before = sum(
                1 for i in range(fancy_pos) if isinstance(index_tuple[i], slice)
            )

            if 0 < slice_dims_before < result_ndim:
                # Move broadcast dimension to front
                axes = list(range(result_ndim))
                return (
                    [slice_dims_before]
                    + axes[:slice_dims_before]
                    + axes[slice_dims_before + 1 :]
                )

        # Multiple separated indices: PyTorch usually already correct
        return None

    elif consec_position > 0:  # Consecutive indices
        # Use NumPy's transpose for consecutive case
        return _numpy_get_transpose(
            fancy_ndim=1, consec=consec_position, ndim=result_ndim, getmap=True
        )

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
