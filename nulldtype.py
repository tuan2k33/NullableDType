"""Missing values as a NumPy dtype.

NA is a reserved value of the wrapped type itself, so an element costs exactly
what the wrapped type costs:

    Nullable(np.float64)   itemsize 8  — every bit but the sign
    Nullable("U10")        itemsize 40 — the whole cell filled with 0xFF
    Nullable(record)       itemsize T  — every field its own type's NA

LAYOUTS.md has the pattern for every type.
"""
import math

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

from _nulldtype import NA, LongDoubleWarning, NullableDType

__all__ = ["NA", "LongDoubleWarning", "Nullable", "is_nullable",
           "NullableDType",
           "isna", "notna", "filled", "to_numpy", "count", "dropna",
           "all", "any", "array_equal", "isin", "isclose", "allclose", "dot",
           "argmax", "argmin", "cumsum", "cumprod",
           "sum", "prod", "min", "max", "mean", "std", "var",
           "median", "quantile", "percentile", "unique"]


def Nullable(t):
    """`T` with missing values.

    Numbers -- ints, unsigned ints, floats, complex, bool, datetime64 --
    fixed-width `S`, `U` and `V`, and records of any of these reserve one of
    their own bit patterns for NA, so an element costs exactly what `T` costs.
    A record is missing when every field holds its own type's NA.  See
    LAYOUTS.md for every pattern.  `longdouble` and `clongdouble` are stored as
    `float64` and `complex128`, with a `LongDoubleWarning` wherever that loses
    precision.
    """
    return NullableDType(t)


def is_nullable(dtype):
    """True for a dtype produced by `Nullable`."""
    return isinstance(dtype, NullableDType)


def _check(a):
    a = np.asarray(a)
    if not is_nullable(a.dtype):
        raise TypeError(f"expected a Nullable array, got {a.dtype!r}")
    # `ascontiguousarray` hands back a 1-d array for a 0-d one, and a scalar
    # must stay a scalar all the way through `isna` and `filled`
    return np.ascontiguousarray(a).reshape(a.shape)


def _parts(a):
    """Plain arrays of the values and of the validity flags."""
    a = _check(a)
    values = a.view(a.dtype.wrapped)
    return values, ~_na_mask(values)


def _na_mask(values):
    """True where a plain array of a stored dtype holds that dtype's NA."""
    if values.ndim == 0:
        return _na_mask(values.reshape(1)).reshape(())
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
    if kind == "U":
        # every character U+FFFF, the noncharacter kept for internal use
        chars = values.view(np.uint32).reshape(values.shape + (size // 4,))
        return (chars == 0xFFFF).all(axis=-1)
    if kind in "SV":
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


def count(a):
    """How many elements have a value.  `numpy.ma` spells this `a.count()`."""
    return int(notna(a).sum())


def dropna(a):
    """The same array with the missing elements taken out."""
    a = _check(a)
    return a[notna(a)]


def dot(a, b):
    """`np.dot` cannot be implemented -- its legacy slot is handed a NULL array
    -- but for 1-D and 2-D operands `@` is the same operation."""
    x, y = np.asarray(a), np.asarray(b)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("nd.dot covers 1-D and 2-D; for more, use `@` directly")
    return x @ y


def array_equal(a1, a2):
    """True, False, or NA when a missing element leaves the answer undecided.

    `np.array_equal` calls `np.all`, which pins its accumulator to a plain bool.
    """
    x, y = np.asarray(a1), np.asarray(a2)
    if x.shape != y.shape:
        return False
    if x.size == 0:
        return True
    return all(x == y)


def isin(a, test_elements):
    """Like `np.isin`, but an element that is missing answers NA instead of
    raising: whether an unknown value is in a set is itself unknown."""
    values, valid = _parts(a)
    test = np.asarray(test_elements)
    gap_in_test = False
    if is_nullable(test.dtype):
        gap_in_test = bool(isna(test).any())
        test = to_numpy(dropna(test))
    found = np.isin(values, test)
    out = np.zeros(values.shape, dtype=Nullable(np.bool_))
    out[...] = found
    if gap_in_test:
        # a gap in the set could be any value, so not finding x among the known
        # ones does not rule x out -- SQL's `3 IN (1, NULL)` is NULL too
        out[~found] = NA
    out[~valid] = NA
    return out


def _arg_extreme(a, skipna, find):
    a = _check(a)
    if not skipna:
        return find(a)
    values, valid = _parts(a)
    valid = valid.ravel()
    if not valid.any():
        return NA
    positions = np.flatnonzero(valid)
    return positions[find(values.ravel()[valid])]


def argmax(a, skipna=False):
    """Flat position of the largest value.

    A gap could be anything, so by default the answer is the position of the
    first gap -- numpy's rule for NaN, and what keeps `a[nd.argmax(a)]` equal to
    `nd.max(a)`.  `skipna=True` skips the gaps, like `np.nanargmax`, and is NA
    when nothing is left.
    """
    return _arg_extreme(a, skipna, np.argmax)


def argmin(a, skipna=False):
    """Flat position of the smallest value; see `argmax`."""
    return _arg_extreme(a, skipna, np.argmin)


def isclose(a, b, rtol=1e-05, atol=1e-08):
    """`np.isclose` asks `np.result_type(dtype, 1.0)` before anything else and
    gives up there; this one is the same formula over nullable arithmetic."""
    x, y = np.asarray(a), np.asarray(b)
    return abs(x - y) <= atol + rtol * abs(y)


def allclose(a, b, rtol=1e-05, atol=1e-08):
    """Kleene `allclose`: NA when a missing element leaves it undecided."""
    return all(isclose(a, b, rtol, atol))


def all(a, axis=None, skipna=False, keepdims=False):
    """Kleene `all`: False wins, otherwise NA if anything is missing.
    `skipna=True` asks about the known elements only, and is a plain bool.

    `np.all` cannot be used: it forces a plain bool result.
    """
    if skipna:
        return np.all(filled(a, True), axis=axis, keepdims=keepdims)
    return np.logical_and.reduce(_check(a), axis=axis, keepdims=keepdims)


def any(a, axis=None, skipna=False, keepdims=False):
    """Kleene `any`: True wins, otherwise NA if anything is missing.  See `all`."""
    if skipna:
        return np.any(filled(a, False), axis=axis, keepdims=keepdims)
    return np.logical_or.reduce(_check(a), axis=axis, keepdims=keepdims)


def _accumulate(name, a, axis, skipna):
    arr = _check(a)
    if axis is None:
        arr, axis = arr.ravel(), 0
    # numpy widens bool and small ints to the platform int before summing
    widened = getattr(np, name)(np.zeros(0, arr.dtype.wrapped)).dtype
    if widened != arr.dtype.wrapped:
        arr = arr.astype(Nullable(widened))
    if not skipna:
        return getattr(np, name)(arr, axis=axis)
    gaps = isna(arr)
    out = np.empty(arr.shape, dtype=arr.dtype)
    out[...] = getattr(np, name)(filled(arr, _IDENTITY[name[3:]]), axis=axis)
    out[gaps] = NA
    return out


def cumsum(a, axis=None, skipna=False):
    """Running sum.  By default everything from the first gap on is NA.
    `skipna=True` carries the total past a gap and leaves NA at the gap
    itself -- pandas' `cumsum`.  `axis=None` flattens, like `np.cumsum`."""
    return _accumulate("cumsum", a, axis, skipna)


def cumprod(a, axis=None, skipna=False):
    """Running product; see `cumsum`."""
    return _accumulate("cumprod", a, axis, skipna)


# --------------------------------------------------------------- statistics
#
# `np.median`, `np.percentile` and `np.quantile` are written in Python on top
# of `sort`/`partition`.  NA sorts last, so they take it for the largest value
# and hand back a plausible-looking number:
#
#     >>> np.median(np.array([3.0, 1.0, NA, 5.0], dtype=Nullable(np.float64)))
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


def _whole(name, a, skipna, kwargs):
    """The whole array down to one plain scalar, or NA."""
    values = _stat_values(a, skipna)
    if values is None:
        return NA
    if values.size == 0:
        return _reduce_empty(name, values.dtype)
    return getattr(np, name)(values, **kwargs)


def _result_dtype(name, wrapped, kwargs):
    """What `np.<name>` returns for `wrapped`, asked of a throwaway array."""
    with np.errstate(all="ignore"):
        return np.asarray(getattr(np, name)(np.ones(3, wrapped), **kwargs)).dtype


def _lanes(name, a, axis, keepdims, one, kwargs):
    """Apply `one` -- a whole-array reduction -- to every lane along `axis`.

    A Python loop over the lanes, each one handed to numpy's own function, so
    the numbers are numpy's.  `numpy.ma` would be faster and is wrong for this:
    it masks every non-finite result, so the mean of `[inf, 1]` comes back
    masked and its std as 0.
    """
    arr = _check(a)
    axes = normalize_axis_tuple(range(arr.ndim) if axis is None else axis, arr.ndim)
    rest = [i for i in range(arr.ndim) if i not in axes]
    moved = arr.transpose(rest + list(axes))
    outer = moved.shape[:len(rest)]
    lanes = moved.reshape(math.prod(outer), math.prod(moved.shape[len(rest):]))

    results = [one(lane) for lane in lanes]
    q_shape = ()
    for r in results:
        if r is not NA:
            q_shape = np.shape(r)
            break
    else:
        q_shape = np.shape(kwargs["q"]) if "q" in kwargs else ()
    out = np.empty((len(results),) + q_shape,
                   dtype=Nullable(_result_dtype(name, arr.dtype.wrapped, kwargs)))
    for i, r in enumerate(results):
        if r is NA or (isinstance(r, np.ndarray) and is_nullable(r.dtype)):
            out[i] = NA          # a Nullable array here is `_all_missing_like`
        else:
            out[i] = r
    out = out.reshape(outer + q_shape)
    # numpy puts the dimensions of an array `q` in front
    out = np.moveaxis(out, range(len(outer), out.ndim), range(len(q_shape)))
    if keepdims:
        out = np.expand_dims(out, tuple(len(q_shape) + i for i in axes))
    return out


def _simple(name):
    def stat(a, axis=None, skipna=False, keepdims=False, **kwargs):
        if axis is None and not keepdims:
            return _whole(name, a, skipna, kwargs)
        if name in _IDENTITY:
            # vectorised: fill the gaps with the identity, then put NA back on
            # every lane that had one
            values, valid = _parts(a)
            plain = np.where(valid, values, values.dtype.type(_IDENTITY[name]))
            got = np.asarray(getattr(np, name)(plain, axis=axis, keepdims=keepdims,
                                               **kwargs))
            out = np.empty(got.shape, dtype=Nullable(got.dtype))
            out[...] = got
            if not skipna:
                out[~np.logical_and.reduce(valid, axis=axis, keepdims=keepdims)] = NA
            return out
        return _lanes(name, a, axis, keepdims,
                      lambda lane: _whole(name, lane, skipna, kwargs), kwargs)
    stat.__name__ = name
    stat.__doc__ = (f"`np.{name}`, propagating NA unless `skipna=True`.  With "
                    f"no `axis`, a plain scalar or NA; with one, a Nullable "
                    f"array with NA on every lane that is missing something.")
    return stat


sum = _simple("sum")
prod = _simple("prod")
min = _simple("min")
max = _simple("max")
mean = _simple("mean")
std = _simple("std")
var = _simple("var")
median = _simple("median")


def _quantile_like(name, a, q, axis, skipna, keepdims, kwargs):
    def one(lane):
        values = _stat_values(lane, skipna)
        if values is None or values.size == 0:
            return NA if np.ndim(q) == 0 else _all_missing_like(lane, np.shape(q))
        return getattr(np, name)(values, q, **kwargs)
    if axis is None and not keepdims:
        return one(a)
    return _lanes(name, a, axis, keepdims, one, dict(kwargs, q=q))


def quantile(a, q, axis=None, skipna=False, keepdims=False, **kwargs):
    """`np.quantile`, propagating NA unless `skipna=True`.

    A scalar `q` gives a scalar or NA; an array `q` gives an array, all NA if
    the input was missing anything.  With `axis`, NA per lane.
    """
    return _quantile_like("quantile", a, q, axis, skipna, keepdims, kwargs)


def percentile(a, q, axis=None, skipna=False, keepdims=False, **kwargs):
    """`np.percentile`, propagating NA unless `skipna=True`.  See `quantile`."""
    return _quantile_like("percentile", a, q, axis, skipna, keepdims, kwargs)


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


# --------------------------------------------------------------- the rest
#
# Everything numpy already gets right on a Nullable array -- `sort`, `clip`,
# `concatenate`, every ufunc -- is forwarded untouched, so `nd.sort` *is*
# `np.sort`.  Only two kinds of name are not: the ones defined above, which
# numpy gets wrong or refuses, and the ones below, which would look like they
# work.  `VS-NUMPY-MA.md` has the measured list.
_REFUSED = {
    name: "NA is not NaN: these skip NaN, which a gap only happens to be for "
          "floats.  Use nd.{}(a, skipna=True)".format(name[3:])
    for name in ("nansum", "nanprod", "nanmean", "nanstd", "nanvar",
                 "nanmedian", "nanmax", "nanmin", "nanquantile", "nanpercentile")
}
_REFUSED.update({
    "nanargmax": "use nd.argmax(a, skipna=True); NA is not NaN",
    "nanargmin": "use nd.argmin(a, skipna=True); NA is not NaN",
    "nancumsum": "use nd.cumsum(a, skipna=True); NA is not NaN",
    "nancumprod": "use nd.cumprod(a, skipna=True); NA is not NaN",
})


def __getattr__(name):
    message = _REFUSED.get(name)
    if message is not None:
        raise AttributeError(f"nd.{name}: {message}")
    try:
        forwarded = getattr(np, name)
    except AttributeError:
        raise AttributeError(
            f"module 'nulldtype' has no attribute {name!r}") from None
    globals()[name] = forwarded       # forward once, then it is just numpy
    return forwarded


def __dir__():
    return sorted(set(__all__) | set(n for n in dir(np) if not n.startswith("_")))
