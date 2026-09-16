"""Missing values as a NumPy dtype.

Two layouts, one meaning:

    Nullable(np.float64)     itemsize 8 — one value of T is reserved to mean NA
    FlagLayout(np.float64)   itemsize 9 — the value, then a validity byte

`Nullable` picks whichever the wrapped type allows; the two are the same to
use, and `astype` converts between them.
"""
import numpy as np

from _nulldtype import NA, LongDoubleWarning
from _nulldtype import Nullable as FlagLayout, NullableDType as FlagDType
from _nulldtype import Sentinel as BitpatternLayout, SentinelDType as BitpatternDType

__all__ = ["NA", "LongDoubleWarning", "Nullable", "is_nullable",
           "FlagLayout", "FlagDType", "BitpatternLayout", "BitpatternDType",
           "isna", "notna", "filled", "to_numpy", "all", "any",
           "sum", "prod", "min", "max", "mean", "std", "var",
           "median", "quantile", "percentile", "unique"]

_LAYOUT_DTYPES = (BitpatternDType, FlagDType)


def Nullable(t):
    """`T` with missing values.  The layout is an implementation detail.

    Numbers -- ints, unsigned ints, floats, complex, bool, datetime64 --
    fixed-width `S`, `U` and `V`, and records of any of these reserve one of
    their own bit patterns for NA, so an element costs exactly what `T` costs.
    A record is missing when every field holds its own type's NA.  See
    LAYOUTS.md for every pattern.  `longdouble` and
    `clongdouble` are stored as `float64` and `complex128`, with a
    `LongDoubleWarning` wherever that loses precision.

    Both mean exactly the same thing; the test suite checks that on every
    operation.  Ask for `BitpatternLayout` or `FlagLayout` by name only to
    benchmark or to test one against the other.
    """
    try:
        return BitpatternLayout(t)
    except TypeError:
        return FlagLayout(t)


def is_nullable(dtype):
    """True for a dtype produced by `Nullable`, whichever layout it picked."""
    return isinstance(dtype, _LAYOUT_DTYPES)


def _check(a):
    a = np.asarray(a)
    if not is_nullable(a.dtype):
        raise TypeError(f"expected a Nullable array, got {a.dtype!r}")
    return np.ascontiguousarray(a)


def _parts(a):
    """Plain arrays of the values and of the validity flags."""
    a = _check(a)
    wrapped = a.dtype.wrapped

    if isinstance(a.dtype, FlagDType):
        raw = a.view(np.uint8).reshape(a.shape + (a.dtype.itemsize,))
        # the flag follows the value; anything after it is alignment padding
        size = wrapped.itemsize
        values = raw[..., :size].copy().view(wrapped).reshape(a.shape)
        return values, raw[..., size].astype(bool)

    values = a.view(wrapped)
    return values, ~_na_mask(values)


def _na_mask(values):
    """True where a plain array of a stored dtype holds that dtype's NA."""
    dt = values.dtype
    if dt.names is not None:
        # a record is missing when every field is: one field with a value makes
        # it a value, and padding belongs to no field
        na = np.ones(values.shape, dtype=bool)
        for name in dt.names:
            per_record = int(np.prod(dt.fields[name][0].shape, dtype=np.intp))
            field_na = _na_mask(values[name])
            na &= field_na.reshape(values.shape + (per_record,)).all(axis=-1)
        return na

    kind, size = dt.kind, dt.itemsize
    if kind in "cSUV":
        # the views below change the item size, which needs contiguous data,
        # and a record's fields are strided views
        values = np.ascontiguousarray(values)
    if kind == "f" and size == 8:
        return _na_float(values.view(np.uint64), 0x7FFFFFFFFFFFFFFF)
    if kind == "f" and size == 4:
        return _na_float(values.view(np.uint32), 0x7FFFFFFF)
    if kind == "f" and size == 2:
        return _na_float(values.view(np.uint16), 0x7FFF)
    if kind == "c":
        # the pattern lives in the real half, which is the first field
        half = np.float64 if size == 16 else np.float32
        return _parts_na_real(values.view(half).reshape(values.shape + (2,))[..., 0])
    if kind == "b":
        return values.view(np.uint8) == 2
    if kind in "Mm":
        # numpy already spells this NaT, so ask numpy
        return np.isnat(values)
    if kind in "SUV":
        # the whole cell filled with 0xFF; b"\xffab" is an ordinary value
        raw = values.view(np.uint8).reshape(values.shape + (size,))
        return (raw == 0xFF).all(axis=-1)
    if kind == "u":
        # UINT_MAX, the unsigned mirror of INT_MIN; `min` would be 0 here
        return values == np.iinfo(dt).max
    return values == np.iinfo(dt).min


def _na_float(bits, pattern):
    """NA is every bit but the sign; the sign itself is ignored."""
    p = bits.dtype.type(pattern)
    return (bits & p) == p


def _parts_na_real(real):
    if real.dtype == np.float64:
        return _na_float(real.view(np.uint64), 0x7FFFFFFFFFFFFFFF)
    return _na_float(real.view(np.uint32), 0x7FFFFFFF)


def _as_fill(values, value):
    """A tuple standing in for a record becomes a record `np.where` can use."""
    if values.dtype.names is not None and not isinstance(value, np.ndarray):
        return np.array(value, dtype=values.dtype)
    return value


def notna(a):
    """A plain bool array, True where the element has a value."""
    return _parts(a)[1]


def isna(a):
    """A plain bool array, True where the element is missing."""
    return ~notna(a)


def filled(a, value):
    """A plain array of the wrapped dtype, with `value` in place of NA."""
    values, valid = _parts(a)
    return np.where(valid, values, _as_fill(values, value))


def to_numpy(a, na_value=None):
    """Leave the missing-data world.  Refuses to guess if anything is missing."""
    values, valid = _parts(a)
    if na_value is None:
        if not valid.all():
            raise ValueError(
                "array contains missing values; pass na_value= to say what "
                "they should become")
        return values.copy()
    return np.where(valid, values, _as_fill(values, na_value))


def all(a):
    """Kleene `all`.  `np.all` cannot be used: it forces a plain bool result."""
    return np.logical_and.reduce(_check(a))


def any(a):
    """Kleene `any`.  See `all` for why `np.any` does not work."""
    return np.logical_or.reduce(_check(a))


# --------------------------------------------------------------- statistics
#
# `np.median`, `np.percentile` and `np.quantile` are written in Python on top
# of `sort`/`partition`.  NA sorts last, so they take it for the largest value
# and hand back a plausible-looking number:
#
#     >>> np.median(np.array([3.0, 1.0, NA, 5.0], dtype=Sentinel(np.float64)))
#     4.0          # the median of 1, 3, 5 is 3.0
#
# Nothing in a dtype can intercept that -- removing `compare` would only break
# `sort` as well.  These take its place, and follow R: propagate by default,
# `skipna=True` to drop.  A reduction over nothing left is NA, not zero.
#
# Reductions that go through ufuncs (`a.sum()`, `a.mean()`, `a.std()`) already
# propagate correctly; the versions here exist so that `skipna=` reads the same
# way across the whole surface instead of turning into `where=nd.notna(a)`.


# Reducing over nothing.  `sum` and `prod` have identity elements, so 0 and 1
# are the answers, not inventions -- R agrees, and so does numpy.  The rest
# have no identity: the mean, median or maximum of no numbers is not a number,
# it is unknown, so it is NA.  Returning NaN there would be worse than useless
# here, because in this dtype NaN is an ordinary value that some column might
# legitimately hold.
#
# Nothing distinguishes an input that was empty to begin with from one that
# `skipna` emptied, so both take this path.
_IDENTITY = {"sum": 0, "prod": 1}


def _stat_values(a, skipna):
    """The plain values to compute on, or None when the answer is NA."""
    values, valid = _parts(a)
    if not skipna:
        return None if not valid.all() else values
    return values[valid]


def _all_missing_like(a, shape):
    """An all-NA result in `a`'s own dtype, for a call that asked for an array."""
    out = np.empty(shape, dtype=np.asarray(a).dtype)
    out[...] = NA
    return out


def _reduce_empty(name, dtype):
    if name not in _IDENTITY:
        return NA
    return dtype.type(_IDENTITY[name])


def _simple(name):
    def stat(a, skipna=False):
        values = _stat_values(a, skipna)
        if values is None:
            return NA
        if values.size == 0:
            return _reduce_empty(name, values.dtype)
        return getattr(np, name)(values)
    stat.__name__ = name
    stat.__doc__ = (f"`np.{name}`, propagating NA unless `skipna=True`. "
                    f"Returns a plain scalar, or NA.")
    return stat


sum = _simple("sum")
prod = _simple("prod")
min = _simple("min")
max = _simple("max")
mean = _simple("mean")
std = _simple("std")
var = _simple("var")
median = _simple("median")


def quantile(a, q, skipna=False, **kwargs):
    """`np.quantile`, propagating NA unless `skipna=True`.

    A scalar `q` gives a scalar or NA; an array `q` gives an array, all NA if
    the input was missing anything.
    """
    values = _stat_values(a, skipna)
    if values is None or values.size == 0:
        return NA if np.ndim(q) == 0 else _all_missing_like(a, np.shape(q))
    return np.quantile(values, q, **kwargs)


def percentile(a, q, skipna=False, **kwargs):
    """`np.percentile`, propagating NA unless `skipna=True`.  See `quantile`."""
    values = _stat_values(a, skipna)
    if values is None or values.size == 0:
        return NA if np.ndim(q) == 0 else _all_missing_like(a, np.shape(q))
    return np.percentile(values, q, **kwargs)


def unique(a):
    """R's `unique`: NA is one distinct value of its own, and it sorts last.

    `np.unique` cannot be used -- it compares neighbours in the sorted array
    and assigns the result into a plain bool mask, which NA refuses to become.
    """
    arr = _check(a)
    values, valid = _parts(arr)
    found = np.unique(values[valid]).astype(arr.dtype)
    if valid.all():
        return found
    out = np.empty(found.size + 1, dtype=arr.dtype)
    out[:found.size] = found
    out[found.size] = NA
    return out
