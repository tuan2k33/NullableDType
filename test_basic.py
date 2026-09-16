"""Run with: ./run_tests.sh"""
import itertools
import warnings

import numpy as np
import pytest

import nulldtype as nd


def test_dtype_is_parametric():
    assert nd.Nullable(np.int32) != nd.Nullable(np.float64)


def test_na_costs_no_byte():
    """NA is a reserved bit pattern of the wrapped type itself, whatever it is."""
    for np_dtype in (np.float64, np.int16, np.uint8, np.uint64,
                     np.complex128, np.dtype("M8[D]"),
                     np.dtype("S3"), np.dtype("U3"), np.dtype("V4"),
                     np.dtype([("a", "i4"), ("b", "f8")]),
                     np.dtype([("a", "u1"), ("b", "i4")], align=True)):
        dt = nd.Nullable(np_dtype)
        assert isinstance(dt, nd.NullableDType)
        assert dt.itemsize == np.dtype(np_dtype).itemsize


def test_roundtrip():
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 2.0, 3.0], dtype=dt)
    assert a[0] == 1.0
    assert a.itemsize == 8        # NA costs no byte


def test_na_assignment():
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 2.0, 3.0], dtype=dt)
    a[1] = nd.NA
    assert a[1] is nd.NA
    assert a[0] == 1.0


def test_masked_value_is_not_observable():
    """Assigning NA overwrites the old value with the NA pattern, leaving nothing of it."""
    dt = nd.Nullable(np.float64)
    a = np.array([12345.0], dtype=dt)
    a[0] = nd.NA
    assert a.tobytes() == np.array([0x7FFFFFFFFFFFFFFF], np.uint64).tobytes()


@pytest.mark.parametrize("np_dtype", [np.float64, np.float32, np.int32, np.int64])
def test_add_propagates_mask(np_dtype):
    dt = nd.Nullable(np_dtype)
    a = np.array([1, 2, 3, 4], dtype=dt)
    b = np.array([10, 20, 30, 40], dtype=dt)
    a[1] = nd.NA
    b[3] = nd.NA

    res = a + b

    assert res[0] == 11
    assert res[1] is nd.NA
    assert res[2] == 33
    assert res[3] is nd.NA


def test_add_borrows_the_wrapped_loop():
    """Values must match the wrapped dtype's own addition, overflow included."""
    dt = nd.Nullable(np.int8)
    a = np.array([100], dtype=dt)
    b = np.array([100], dtype=dt)
    plain = np.int8(100) + np.int8(100)      # overflows, wraps around
    assert (a + b)[0] == plain


# ------------------------------------------------------------------ casts

@pytest.mark.parametrize("np_dtype", [np.float64, np.float32, np.int32, np.bool_])
def test_cast_into_nullable_is_same_kind(np_dtype):
    dt = nd.Nullable(np_dtype)
    plain = np.array([1, 0, 1], dtype=np_dtype)

    # one value is given up for NA, so the cast is automatic but not "safe"
    assert np.can_cast(np_dtype, dt, "same_kind")
    assert not np.can_cast(np_dtype, dt)
    got = plain.astype(dt)

    assert got.dtype == dt
    assert [got[i] for i in range(3)] == list(plain)


def test_cast_out_of_masked_is_not_implicit():
    """Dropping NA loses information, so it must never happen implicitly."""
    dt = nd.Nullable(np.float64)
    assert not np.can_cast(dt, np.float64)


def test_cast_out_works_without_na():
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 2.0, 3.0]).astype(dt)
    np.testing.assert_array_equal(a.astype(np.float64), [1.0, 2.0, 3.0])


def test_cast_out_raises_on_na():
    """There is no implicit path by which NA turns into a number."""
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 2.0, 3.0]).astype(dt)
    a[1] = nd.NA

    with pytest.raises(ValueError, match="missing value"):
        a.astype(np.float64)


def test_roundtrip_through_cast_keeps_values():
    dt = nd.Nullable(np.int64)
    plain = np.arange(1000)
    np.testing.assert_array_equal(plain.astype(dt).astype(np.int64), plain)


# ------------------------------------------------------- mixed operations

def test_add_plain_array_either_order():
    dt = nd.Nullable(np.float64)
    a = np.arange(4.0).astype(dt)
    a[2] = nd.NA
    plain = np.array([10.0, 20.0, 30.0, 40.0])

    for res in (a + plain, plain + a):
        assert res.dtype == dt
        assert res[0] == 10.0
        assert res[1] == 21.0
        assert res[2] is nd.NA
        assert res[3] == 43.0


@pytest.mark.parametrize("scalar", [1.0, 1])
def test_add_python_scalar_either_order(scalar):
    dt = nd.Nullable(np.float64)
    a = np.arange(4.0).astype(dt)
    a[2] = nd.NA

    for res in (a + scalar, scalar + a):
        assert res[0] == 1.0
        assert res[2] is nd.NA


def test_nan_is_not_na():
    """NA is missing data, NaN is an undefined result — two different things."""
    dt = nd.Nullable(np.float64)
    a = np.arange(4.0).astype(dt)
    a[2] = nd.NA

    res = a + np.array([np.nan] * 4)

    assert np.isnan(res[0])          # NaN propagates as usual
    assert res[2] is nd.NA           # NA stays NA, it does not become NaN


def test_mixed_value_dtypes_promote():
    """Nullable[i8] + Nullable[f8] promotes exactly like i8 + f8."""
    a = np.arange(3.0).astype(nd.Nullable(np.float64))
    b = np.arange(3).astype(nd.Nullable(np.int64))
    a[1] = nd.NA
    b[2] = nd.NA

    res = a + b

    assert res.dtype == nd.Nullable(np.float64)
    assert res[0] == 0.0
    assert res[1] is nd.NA
    assert res[2] is nd.NA


def test_cast_between_wrapped_dtypes_keeps_na():
    b = np.arange(3).astype(nd.Nullable(np.int64))
    b[2] = nd.NA

    res = b.astype(nd.Nullable(np.float64))

    assert res[0] == 0.0
    assert res[2] is nd.NA


def test_cast_into_a_different_wrapped_dtype():
    got = np.array([1, 2, 3]).astype(nd.Nullable(np.int16))
    assert got.dtype == nd.Nullable(np.int16)
    assert got[0] == 1


# ------------------------------------------------------------ binary ufuncs

BINOPS = ["add", "subtract", "multiply", "true_divide",
          "floor_divide", "power", "maximum", "minimum"]


@pytest.mark.parametrize("op", BINOPS)
def test_binop_matches_the_wrapped_dtype(op):
    """Values must match the wrapped dtype's own operation, and gaps propagate."""
    dt = nd.Nullable(np.float64)
    plain_a = np.array([1.0, 2.0, 3.0, 4.0])
    plain_b = np.array([10.0, 20.0, 30.0, 40.0])
    a, b = plain_a.astype(dt), plain_b.astype(dt)
    a[2] = nd.NA
    b[3] = nd.NA

    with np.errstate(all="ignore"):
        got = getattr(np, op)(a, b)
        want = getattr(np, op)(plain_a, plain_b)

    assert got[0] == want[0]
    assert got[1] == want[1]
    assert got[2] is nd.NA
    assert got[3] is nd.NA


def test_true_divide_promotes_like_numpy():
    """i8 / i8 must give f8, as in numpy — resolve asks the wrapped ufunc itself."""
    i = np.array([7, 8, 9]).astype(nd.Nullable(np.int64))
    i[1] = nd.NA

    res = i / i

    assert res.dtype == nd.Nullable(np.float64)
    assert res[0] == 1.0
    assert res[1] is nd.NA


def test_floor_divide_keeps_int():
    i = np.array([7, 8, 9]).astype(nd.Nullable(np.int64))
    res = i // 2
    assert res.dtype == nd.Nullable(np.int64)
    assert res[0] == 3


def test_chained_expression():
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 2.0, 3.0, 4.0]).astype(dt)
    a[2] = nd.NA

    res = (a * 2 - 1) / 3

    assert res[0] == pytest.approx(1 / 3)
    assert res[2] is nd.NA


def test_no_spurious_warning_from_missing_values():
    """A gap must not raise a floating-point warning."""
    dt = nd.Nullable(np.float64)
    a = np.array([20.0, 20.0, 20.0]).astype(dt)
    b = np.array([0.0, 5.0, 0.0]).astype(dt)
    b[0] = nd.NA
    b[2] = nd.NA

    with warnings.catch_warnings():
        warnings.simplefilter("error")      # moi canh bao thanh loi
        res = a / b

    assert res[0] is nd.NA
    assert res[1] == 4.0
    assert res[2] is nd.NA


def test_real_warning_still_reaches_the_user():
    """But dividing by a real 0 must still warn."""
    dt = nd.Nullable(np.float64)
    a = np.array([20.0, 20.0]).astype(dt)
    b = np.array([0.0, 5.0]).astype(dt)      # a real 0, not NA

    with pytest.warns(RuntimeWarning, match="divide by zero"):
        res = a / b

    assert np.isinf(res[0])


def test_missing_values_stay_unobservable_after_an_op():
    dt = nd.Nullable(np.float64)
    a = np.array([12345.0, 1.0]).astype(dt)
    b = np.array([12345.0, 1.0]).astype(dt)
    b[0] = nd.NA

    res = a * b
    assert res.tobytes()[:8] == np.array([0x7FFFFFFFFFFFFFFF], np.uint64).tobytes()


# ------------------------------------------------------------- comparisons

COMPARISONS = ["equal", "not_equal", "less", "less_equal",
               "greater", "greater_equal"]


@pytest.mark.parametrize("op", COMPARISONS)
def test_comparison_returns_nullable_bool(op):
    dt = nd.Nullable(np.float64)
    plain = np.array([1.0, 2.0, 3.0])
    a = plain.astype(dt)
    a[1] = nd.NA

    got = getattr(np, op)(a, 2.0)
    want = getattr(np, op)(plain, 2.0)

    assert got.dtype == nd.Nullable(np.bool_)
    assert got[0] == want[0]
    assert got[1] is nd.NA
    assert got[2] == want[2]


def test_na_is_not_equal_to_itself():
    """NA == NA is NA, not True — as in R and SQL."""
    a = np.array([1.0, 2.0]).astype(nd.Nullable(np.float64))
    a[1] = nd.NA

    assert (a == a)[0] == True
    assert (a == a)[1] is nd.NA
    assert (a != a)[1] is nd.NA


def test_filtering_with_a_missing_value_is_refused():
    """No implicit path: numpy.ma quietly treats masked as False; this does not."""
    a = np.array([1.0, 2.0, 3.0]).astype(nd.Nullable(np.float64))
    a[1] = nd.NA
    mask = a > 1.5

    with pytest.raises(IndexError):
        a[mask]


# ----------------------------------------------------- Kleene three-valued

T, F, X = True, False, None          # X = NA


def _scalar(value):
    a = np.array([True]).astype(nd.Nullable(np.bool_))
    if value is None:
        a[0] = nd.NA
    else:
        a[0] = value
    return a


def _result(got):
    return None if got[0] is nd.NA else bool(got[0])


@pytest.mark.parametrize("a, b, want", [
    (T, T, T), (T, F, F), (T, X, X),
    (F, T, F), (F, F, F), (F, X, F),     # False quyet dinh, du ben kia la NA
    (X, T, X), (X, F, F), (X, X, X),
])
def test_kleene_and(a, b, want):
    assert _result(np.logical_and(_scalar(a), _scalar(b))) is want


@pytest.mark.parametrize("a, b, want", [
    (T, T, T), (T, F, T), (T, X, T),     # True quyet dinh
    (F, T, T), (F, F, F), (F, X, X),
    (X, T, T), (X, F, X), (X, X, X),
])
def test_kleene_or(a, b, want):
    assert _result(np.logical_or(_scalar(a), _scalar(b))) is want


@pytest.mark.parametrize("a, b, want", [
    (T, T, F), (T, F, T), (T, X, X),
    (F, T, T), (F, F, F), (F, X, X),
    (X, T, X), (X, F, X), (X, X, X),     # xor is never certain
])
def test_kleene_xor(a, b, want):
    assert _result(np.logical_xor(_scalar(a), _scalar(b))) is want


def test_kleene_on_a_whole_array():
    db = nd.Nullable(np.bool_)
    a = np.array([True, False, True]).astype(db)
    b = np.array([True, True, True]).astype(db)
    a[2] = nd.NA
    b[0] = nd.NA

    got = np.logical_and(a, b)
    assert got[0] is nd.NA        # NA & True
    assert got[1] == False        # False & True
    assert got[2] is nd.NA        # NA & True

    got = np.logical_or(a, b)
    assert got[0] == True         # NA | True -> True
    assert got[2] == True


# ------------------------------------------------------- helpers, reduction

def test_isna_notna():
    a = np.array([1.0, 2.0, 3.0]).astype(nd.Nullable(np.float64))
    a[1] = nd.NA

    np.testing.assert_array_equal(nd.isna(a), [False, True, False])
    np.testing.assert_array_equal(nd.notna(a), [True, False, True])


def test_filled():
    a = np.array([1.0, 2.0, 3.0]).astype(nd.Nullable(np.float64))
    a[1] = nd.NA

    np.testing.assert_array_equal(nd.filled(a, 0.0), [1.0, 0.0, 3.0])
    assert np.isnan(nd.filled(a, np.nan)[1])


def test_to_numpy_refuses_to_guess():
    a = np.array([1.0, 2.0]).astype(nd.Nullable(np.float64))
    a[1] = nd.NA

    with pytest.raises(ValueError, match="missing values"):
        nd.to_numpy(a)

    np.testing.assert_array_equal(nd.to_numpy(a, na_value=-1.0), [1.0, -1.0])


@pytest.mark.parametrize("op, want", [
    ("add", 10.0), ("multiply", 24.0), ("maximum", 4.0), ("minimum", 1.0),
])
def test_reduction_without_missing_values(op, want):
    a = np.array([1.0, 2.0, 3.0, 4.0]).astype(nd.Nullable(np.float64))
    assert getattr(np, op).reduce(a) == want


def test_reduction_over_a_large_array():
    """Long enough for numpy to split it into several chunks."""
    a = np.arange(10000.0).astype(nd.Nullable(np.float64))
    assert a.sum() == np.arange(10000.0).sum()


def test_reduction_propagates_by_default():
    """Propagates by default, like R's `na.rm = FALSE`."""
    a = np.array([1.0, 2.0, 3.0, 4.0]).astype(nd.Nullable(np.float64))
    a[1] = nd.NA
    assert a.sum() is nd.NA


def test_reduction_can_skip_missing_values():
    """Skipping has to be asked for, like `na.rm = TRUE`."""
    a = np.array([1.0, 2.0, 3.0, 4.0]).astype(nd.Nullable(np.float64))
    a[1] = nd.NA
    assert np.add.reduce(a, where=nd.notna(a), initial=0.0) == 8.0


def test_reduction_along_an_axis():
    a = np.arange(6.0).reshape(2, 3).astype(nd.Nullable(np.float64))
    res = a.sum(axis=1)
    assert res[0] == 3.0
    assert res[1] == 12.0


@pytest.mark.parametrize("op, want", [
    ("add", [1.0, 3.0, 6.0, 10.0]),
    ("multiply", [1.0, 2.0, 6.0, 24.0]),
    ("maximum", [1.0, 2.0, 3.0, 4.0]),
])
def test_accumulate(op, want):
    a = np.array([1.0, 2.0, 3.0, 4.0]).astype(nd.Nullable(np.float64))
    got = getattr(np, op).accumulate(a)
    assert [got[i] for i in range(4)] == want


def test_accumulate_is_missing_from_the_first_gap_on():
    a = np.array([1.0, 2.0, 3.0, 4.0]).astype(nd.Nullable(np.float64))
    a[2] = nd.NA

    got = np.add.accumulate(a)

    assert got[0] == 1.0
    assert got[1] == 3.0
    assert got[2] is nd.NA
    assert got[3] is nd.NA


def test_accumulate_over_a_large_array():
    a = np.arange(10000.0).astype(nd.Nullable(np.float64))
    assert np.add.accumulate(a)[-1] == np.arange(10000.0).cumsum()[-1]


def test_skipping_needs_no_initial_value():
    """Because the identity is declared, `where=` works on its own."""
    a = np.array([1.0, 2.0, 3.0, 4.0]).astype(nd.Nullable(np.float64))
    a[1] = nd.NA

    assert np.add.reduce(a, where=nd.notna(a)) == 8.0
    assert np.multiply.reduce(a, where=nd.notna(a)) == 12.0


def test_sum_of_an_empty_array_is_the_identity():
    empty = np.array([]).astype(nd.Nullable(np.float64))
    assert empty.sum() == 0.0


def test_inplace_still_works():
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 2.0]).astype(dt)
    a += np.array([10.0, 10.0]).astype(dt)
    assert a[0] == 11.0


# ------------------------------------------------------------ unary ufuncs

UNOPS = ["negative", "absolute", "square", "exp", "sign", "floor", "ceil"]


@pytest.mark.parametrize("op", UNOPS)
def test_unary_matches_the_wrapped_dtype(op):
    dt = nd.Nullable(np.float64)
    plain = np.array([1.0, 4.0, 9.0, -2.0])
    a = plain.astype(dt)
    a[2] = nd.NA

    got = getattr(np, op)(a)
    want = getattr(np, op)(plain)

    assert got[0] == want[0]
    assert got[1] == want[1]
    assert got[2] is nd.NA
    assert got[3] == want[3]


def test_unary_operators():
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, -2.0]).astype(dt)
    a[1] = nd.NA

    assert (-a)[0] == -1.0
    assert (-a)[1] is nd.NA
    assert abs(a)[0] == 1.0


def test_unary_does_not_touch_missing_values():
    """log of a gap (underlying value 0) would warn 'divide by zero' if it were computed."""
    a = np.array([1.0, 2.0, 3.0]).astype(nd.Nullable(np.float64))
    a[1] = nd.NA

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = np.log(a)

    assert res[1] is nd.NA


def test_unary_real_warning_still_reaches_the_user():
    a = np.array([1.0, 0.0, 3.0]).astype(nd.Nullable(np.float64))
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        np.log(a)


# ------------------------------------------------------ everyday array work

def test_construction_from_a_list_with_na():
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, nd.NA, 3.0], dtype=dt)
    assert a[0] == 1.0
    assert a[1] is nd.NA


def test_zeros_ones_full():
    """np.zeros means real zeros, not gaps."""
    dt = nd.Nullable(np.float64)
    assert np.zeros(3, dt)[0] == 0.0
    assert np.ones(3, dt)[0] == 1.0
    assert np.full(3, 7.0, dtype=dt)[0] == 7.0


def test_shape_operations_keep_the_gaps():
    dt = nd.Nullable(np.float64)
    a = np.arange(6.0).astype(dt)
    a[2] = nd.NA

    assert a.copy()[2] is nd.NA
    assert a.reshape(2, 3)[0, 2] is nd.NA
    assert np.concatenate([a, a])[8] is nd.NA
    assert a[[0, 2]][1] is nd.NA


def test_sort_puts_missing_values_last():
    dt = nd.Nullable(np.float64)
    a = np.array([3.0, 1.0, 2.0, 5.0]).astype(dt)
    a[2] = nd.NA

    got = np.sort(a)
    assert [got[i] for i in range(3)] == [1.0, 3.0, 5.0]
    assert got[3] is nd.NA
    np.testing.assert_array_equal(np.argsort(a), [1, 0, 3, 2])


@pytest.mark.parametrize("values, want_all, want_any", [
    ([T, T], True, True),
    ([T, F], False, True),
    ([T, X], None, True),      # any duoc quyet dinh boi True
    ([F, X], False, None),     # all duoc quyet dinh boi False
    ([X, X], None, None),
    ([F, F], False, False),
])
def test_kleene_all_any(values, want_all, want_any):
    a = np.array([True] * len(values)).astype(nd.Nullable(np.bool_))
    for i, v in enumerate(values):
        a[i] = nd.NA if v is None else v

    assert _result([nd.all(a)]) is want_all
    assert _result([nd.any(a)]) is want_any


def test_isnan_of_a_missing_value_is_missing():
    """Without the value there is no answer to whether it is NaN."""
    a = np.array([1.0, 2.0]).astype(nd.Nullable(np.float64))
    a[1] = nd.NA

    assert np.isnan(a)[0] == False
    assert np.isnan(a)[1] is nd.NA


def test_pickle_roundtrip():
    import pickle

    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 2.0, 3.0]).astype(dt)
    a[1] = nd.NA

    assert pickle.loads(pickle.dumps(dt)) == dt
    back = pickle.loads(pickle.dumps(a))
    assert back[0] == 1.0
    assert back[1] is nd.NA


def test_nonzero_does_not_crash_and_refuses_gaps():
    """Without a `nonzero` slot numpy calls a NULL pointer and segfaults."""
    dt = nd.Nullable(np.float64)
    a = np.array([3.0, 0.0, 2.0]).astype(dt)

    np.testing.assert_array_equal(np.nonzero(a)[0], [0, 2])
    assert np.count_nonzero(a) == 2

    a[1] = nd.NA
    with pytest.raises(ValueError, match="truth value of a missing"):
        np.nonzero(a)


def test_argmax_argmin_point_at_the_first_gap():
    """MISSING, not IGNORED: the hidden value might be the largest, so the
    answer is the gap -- numpy's NaN rule, and `a[a.argmax()]` is NA exactly
    when `a.max()` is."""
    dt = nd.Nullable(np.float64)
    a = np.array([3.0, 1.0, 9.0, 5.0, 0.0]).astype(dt)
    a[2] = a[4] = nd.NA

    assert a.argmax() == 2 and a.argmin() == 2
    assert a[a.argmax()] is nd.NA and a.max() is nd.NA
    assert a[a.argmin()] is nd.NA and a.min() is nd.NA

    m = a.reshape(1, 5).repeat(2, axis=0)
    assert m.argmax(axis=1).tolist() == [2, 2]


def test_nd_argmax_skipna_is_the_explicit_opt_in():
    dt = nd.Nullable(np.float64)
    a = np.array([3.0, 1.0, 9.0, 5.0, 0.0]).astype(dt)
    a[2] = a[4] = nd.NA

    assert nd.argmax(a) == 2
    assert nd.argmax(a, skipna=True) == 3     # 5.0, index into the original
    assert nd.argmin(a, skipna=True) == 1

    a[:] = nd.NA
    assert nd.argmax(a, skipna=True) is nd.NA
    assert nd.argmin(a, skipna=True) is nd.NA


def test_std_and_var_propagate():
    dt = nd.Nullable(np.float64)
    clean = np.array([1.0, 2.0, 3.0, 4.0]).astype(dt)
    assert clean.std() == pytest.approx(np.array([1.0, 2.0, 3.0, 4.0]).std())

    a = clean.copy()
    a[1] = nd.NA
    assert a.std() is nd.NA


# ======================================================================
# Bit pattern — NA is a reserved value of T, itemsize unchanged
# ======================================================================

import _nulldtype as _c

@pytest.mark.parametrize("np_dtype", [np.float64, np.float32, np.int64,
                                      np.int32, np.int16, np.int8, np.bool_])
def test_sentinel_costs_no_extra_byte(np_dtype):
    assert nd.Nullable(np_dtype).itemsize == np.dtype(np_dtype).itemsize


@pytest.mark.parametrize("np_dtype", [np.float64, np.float32, np.int64,
                                      np.int32, np.int8, np.bool_])
def test_sentinel_roundtrip(np_dtype):
    dt = nd.Nullable(np_dtype)
    a = np.array([1, 0, 1], dtype=dt)
    a[1] = nd.NA

    assert a[0] == 1
    assert a[1] is nd.NA
    assert a[2] == 1


@pytest.mark.parametrize("op", ["add", "subtract", "multiply", "true_divide"])
def test_sentinel_propagates(op):
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 2.0, 3.0], dtype=dt)
    b = np.array([10.0, 20.0, 30.0], dtype=dt)
    a[1] = nd.NA

    res = getattr(np, op)(a, b)
    assert res[0] == getattr(np, op)(1.0, 10.0)
    assert res[1] is nd.NA
    assert res[2] == getattr(np, op)(3.0, 30.0)


def test_sentinel_int_propagates_without_hardware_help():
    """INT_MIN does not propagate on its own like NaN; the fixup pass has to do it."""
    dt = nd.Nullable(np.int64)
    a = np.array([5, 6], dtype=dt)
    a[0] = nd.NA

    assert (a + 1)[0] is nd.NA
    assert (a + 1)[1] == 7
    assert (1 + a)[0] is nd.NA


def test_sentinel_beats_hardware_operand_order():
    """`nan + NA` and `NA + nan` must both give NA, even though x86 favours the left operand."""
    dt = nd.Nullable(np.float64)
    na = np.array([1.0, 1.0], dtype=dt)
    na[0] = nd.NA
    nan = np.array([np.nan, np.nan], dtype=dt)

    assert (na + nan)[0] is nd.NA
    assert (nan + na)[0] is nd.NA


def test_sentinel_keeps_nan_distinct_from_na():
    dt = nd.Nullable(np.float64)
    a = np.array([np.nan, 1.0], dtype=dt)
    a[1] = nd.NA

    assert np.isnan(a[0]) and a[0] is not nd.NA
    assert a[1] is nd.NA


def test_sentinel_refuses_the_reserved_value():
    dt = nd.Nullable(np.int64)
    a = np.zeros(1, dtype=dt)
    with pytest.raises(ValueError, match="reserved"):
        a[0] = np.iinfo(np.int64).min


@pytest.mark.parametrize("np_dtype, message", [
    (np.dtype([("a", "i4"), ("b", "O")]), "holding references"),
    (np.dtype({"names": [], "formats": [], "itemsize": 4}), "no spare value"),
    (np.dtype("S"), "no spare value"),
    (np.dtype(("i4", (2,))), "no spare value"),
    (np.dtype(object), "holding references"),
    (np.dtypes.StringDType(), "brings its own missing value"),
], ids=["object-field", "only-padding", "S-without-length", "bare-subarray",
        "object", "StringDType"])
def test_sentinel_needs_a_spare_value(np_dtype, message):
    """A record holds NA in its fields.  A field with no pattern of its own --
    an object -- or no field at all leaves nowhere to put it, and neither does
    `S` with no length or a subarray dtype on its own.  `StringDType` is turned
    away for the opposite reason: it already has a missing value of its own."""
    with pytest.raises(TypeError, match=message):
        nd.Nullable(np_dtype)


# ----------------------------------------------------------------------
# S, U, V: NA is the whole cell filled with 0xFF, at any width

_FLEX = pytest.mark.parametrize(
    "np_dtype",
    [np.dtype("S3"), np.dtype("U3"), np.dtype("V4"), np.dtype("U10"),
     np.dtype(">U3")],
    ids=["S3", "U3", "V4", "U10-wider-than-16-bytes", "U3-big-endian"])


def _flex_values(np_dtype):
    """Three distinct values of `np_dtype`, smallest first."""
    if np_dtype.kind == "V":
        return [np.void(b"\x00\x01\x02\x03"), np.void(b"ab\x00\x00"),
                np.void(b"zz\xfe\xff")]
    if np_dtype.kind == "S":
        return [b"", b"ab", b"\xfeq"]
    return ["", "ab", "￿q"]


def _flex_na_bytes(np_dtype):
    """The reserved cell of `np_dtype`, built here rather than read off the
    dtype under test: every byte 0xFF for S and V, every character U+FFFF for
    U -- a noncharacter, so the cell stays a valid string."""
    if np_dtype.kind == "U":
        # stored native, whatever byte order was asked for
        return np.full(np_dtype.itemsize // 4, 0xFFFF, np.uint32).tobytes()
    return b"\xff" * np_dtype.itemsize


def _flex_na_value(np_dtype):
    """A 1-element plain array holding the reserved value, in `np_dtype`'s own
    byte order -- for U that is the string of U+FFFF, not raw stored bytes."""
    if np_dtype.kind == "U":
        return np.array(["\uffff" * (np_dtype.itemsize // 4)], dtype=np_dtype)
    return np.frombuffer(b"\xff" * np_dtype.itemsize, dtype=np_dtype)


def _filled(dt, items):
    a = np.zeros(len(items), dtype=dt)
    for i, v in enumerate(items):
        a[i] = v
    return a


@_FLEX
def test_flexible_na_fills_the_whole_cell(np_dtype):
    dt = nd.Nullable(np_dtype)
    assert isinstance(dt, nd.NullableDType)
    assert dt.itemsize == np_dtype.itemsize

    vals = _flex_values(np_dtype)
    a = _filled(dt, vals + [nd.NA])
    assert a[3] is nd.NA
    assert a.tobytes()[3 * dt.itemsize:] == _flex_na_bytes(np_dtype)
    assert nd.isna(a).tolist() == [False, False, False, True]
    assert [a[i] for i in range(3)] == vals


@_FLEX
def test_flexible_refuses_the_reserved_value(np_dtype):
    a = np.zeros(2, dtype=nd.Nullable(np_dtype))
    full = _flex_na_value(np_dtype)
    with pytest.raises(ValueError, match="reserved"):
        a[:1] = full
    if np_dtype.kind == "S":
        with pytest.raises(ValueError, match="reserved"):
            a[0] = b"\xff" * np_dtype.itemsize
        # only the whole cell counts
        a[1] = b"\xff" + b"a" * (np_dtype.itemsize - 1)
        assert a[1] is not nd.NA
    if np_dtype.kind == "U":
        n = np_dtype.itemsize // 4
        with pytest.raises(ValueError, match="reserved"):
            a[0] = "\uffff" * n
        a[1] = "\uffff" * (n - 1) + "q"       # one real character is enough
        assert a[1] is not nd.NA


@_FLEX
def test_flexible_sort_argmax_and_nonzero(np_dtype):
    """
    These reached numpy's STRING/UNICODE/VOID compare with no array, and those
    read the item size off it: a segfault.
    """
    lo, mid, hi = _flex_values(np_dtype)
    a = _filled(nd.Nullable(np_dtype), [hi, nd.NA, lo, mid, nd.NA])

    s = np.sort(a)
    assert [s[i] for i in range(3)] == [lo, mid, hi]
    assert s[3] is nd.NA and s[4] is nd.NA
    assert np.argsort(a, kind="stable").tolist() == [2, 3, 0, 1, 4]
    assert a.argmax() == 1 and a.argmin() == 1

    with pytest.raises(ValueError, match="truth value"):
        np.nonzero(a)
    plain = np.array([hi, lo, mid], dtype=np_dtype)
    assert (np.nonzero(a[[0, 2, 3]])[0].tolist()
            == np.nonzero(plain)[0].tolist())


@_FLEX
def test_flexible_na_cell_is_a_usable_value(np_dtype):
    """Whatever the reserved cell is, reading the values raw must hand back
    well-formed objects.  U+FFFFFFFF would give up nothing, but numpy then
    builds a `str` whose maximum character is out of range, and `len()` on it
    raises SystemError -- a gap must not be a broken object."""
    a = _filled(nd.Nullable(np_dtype), [_flex_values(np_dtype)[1], nd.NA])
    raw = a.view(a.dtype.wrapped)        # deliberately behind the dtype's back
    gap = raw[1]

    assert len(raw.tolist()) == 2
    if np_dtype.kind == "U":
        assert len(gap) == np_dtype.itemsize // 4
        assert ord(gap[0]) == 0xFFFF and gap.encode("utf-8")
        assert np.strings.upper(raw)[0] == "AB"
    assert np.sort(raw).shape == (2,)


@_FLEX
def test_flexible_casts_in_and_out(np_dtype):
    plain = np.array(_flex_values(np_dtype), dtype=np_dtype)
    a = plain.astype(nd.Nullable(np_dtype))
    # a fresh `np.dtype("S3")` is a new object; it must still be the same dtype
    assert a.dtype == nd.Nullable(np.dtype(np_dtype.str))
    assert a.astype(np_dtype).tobytes() == plain.tobytes()

    a[1] = nd.NA
    with pytest.raises(ValueError, match="missing"):
        a.astype(np_dtype)


def test_a_cast_that_would_produce_na_raises():
    """The cast itself can land on the reserved value; that must not quietly
    turn a real value into a gap."""
    a = _filled(nd.Nullable("S5"), [b"\xff\xff\xffab"])
    with pytest.raises(ValueError, match="reserved"):
        a.astype(nd.Nullable("S3"))

    with pytest.raises(ValueError, match="reserved"):
        np.array([2**31], dtype=np.int64).astype(nd.Nullable(np.int32))


@pytest.mark.parametrize("target", [np.float64, np.float32, np.int16, np.int64])
def test_casts_to_another_plain_width(target):
    """`astype` to a plain dtype of another width copied the source bytes as
    they were: garbage in the top half of a float64, and a write past the end
    of the buffer for int16."""
    plain = np.array([1, -2, 300], dtype=np.int32)
    got = plain.astype(nd.Nullable(np.int32)).astype(target)
    assert got.dtype == target
    assert got.tolist() == plain.astype(target).tolist()


def test_string_comparison_and_concatenation_propagate():
    dt = nd.Nullable("S3")
    a = _filled(dt, [b"ab", nd.NA, b"x"])
    b = _filled(dt, [b"ab", b"q", nd.NA])

    eq = a == b
    assert eq[0] == True and eq[1] is nd.NA and eq[2] is nd.NA   # noqa: E712
    s = a + b
    assert s[0] == b"abab" and s[1] is nd.NA and s[2] is nd.NA


@pytest.mark.parametrize("rec", [
    np.dtype([("a", "i4"), ("b", "f8")]),
    np.dtype([("a", "u1"), ("b", "i4")], align=True),
], ids=["packed", "aligned"])
def test_structured_sort_argmax_and_nonzero(rec):
    """A record's compare got a NULL array too, and VOID_compare reads the
    fields off it -- and needs the GIL, which `sort` released because the
    Nullable dtype did not carry NPY_NEEDS_PYAPI."""
    a = _filled(nd.Nullable(rec), [(2, 1), nd.NA, (1, 5), (2, 0)])

    s = np.sort(a)
    assert [tuple(s[i]) for i in range(3)] == [(1, 5), (2, 0), (2, 1)]
    assert s[3] is nd.NA
    assert a.argmax() == 1 and a.argmin() == 1

    with pytest.raises(ValueError, match="truth value"):
        np.nonzero(a)
    assert np.nonzero(a[[0, 2]])[0].tolist() == [0, 1]


# ----------------------------------------------------------------------
# Records: missing when every field holds its own type's NA

_REC = np.dtype([("i", "i4"), ("f", "f8"), ("s", "S3"), ("u", "u1"),
                 ("t", "M8[s]"), ("b", "?")], align=True)


def _rec_na():
    """What a missing `_REC` must hold, built from the plain dtypes field by
    field, so it does not lean on the code under test.  Padding stays zero."""
    na = np.zeros(1, dtype=_REC)
    na["i"] = np.iinfo(np.int32).min
    na["f"].view(np.uint64)[...] = 0x7FFFFFFFFFFFFFFF
    na["s"] = b"\xff\xff\xff"
    na["u"] = 255
    na["t"] = np.datetime64("NaT", "s")
    na["b"].view(np.uint8)[...] = 2
    return na


def test_record_na_is_every_field_s_own_na():
    """Not a cell of 0xFF: that would read back as (-1, ..., True) -- values."""
    a = np.zeros(3, dtype=nd.Nullable(_REC))
    assert isinstance(a.dtype, nd.NullableDType)
    assert a.itemsize == _REC.itemsize and _REC.itemsize > 4 + 8 + 3 + 1 + 8 + 1

    a[1] = nd.NA
    assert a[1] is nd.NA
    assert a[1:2].tobytes() == _rec_na().tobytes()
    assert nd.isna(a).tolist() == [False, True, False]
    # a field taken out of the gap is a gap of its own type
    fields = np.frombuffer(a[1:2].tobytes(), dtype=_REC)
    assert np.isnat(fields["t"][0])
    assert fields["i"][0] == np.iinfo(np.int32).min


def test_record_with_some_reserved_fields_is_a_value():
    """Only the record whose every field is reserved is given up."""
    a = np.zeros(2, dtype=nd.Nullable(_REC))
    a[0] = (np.iinfo(np.int32).min, 1.0, b"\xff\xff\xff", 255, "NaT", True)
    assert a[0] is not nd.NA and not nd.isna(a)[0]
    assert a[0]["i"] == np.iinfo(np.int32).min

    with pytest.raises(ValueError, match="reserved"):
        a[1:] = _rec_na()


def test_record_padding_bytes_decide_nothing():
    """A gap is written with its padding zeroed, but nothing may depend on that:
    numpy copies records field by field, so padding at the destination is
    whatever the buffer held.  Reading looks at the fields only."""
    rec = np.dtype([("a", "u1"), ("b", "i4")], align=True)   # 3 bytes of padding
    a = np.zeros(2, dtype=nd.Nullable(rec))
    a[0] = nd.NA
    a[1] = (7, 9)
    assert a[0:1].tobytes() == b"\xff\x00\x00\x00\x00\x00\x00\x80"

    raw = a.view(np.uint8).reshape(2, rec.itemsize)
    raw[0, 1:4] = [0xAA, 0xBB, 0xCC]          # garbage, exactly in the padding
    assert a[0] is nd.NA and nd.isna(a).tolist() == [True, False]
    assert tuple(a[1]) == (7, 9)


def test_nested_records_and_subarray_fields():
    inner = np.dtype([("x", "i2"), ("y", "f4")])
    rec = np.dtype([("p", inner), ("q", "f8", (3,)), ("r", inner, (2,))])
    a = np.zeros(3, dtype=nd.Nullable(rec))
    a[1] = nd.NA
    assert nd.isna(a).tolist() == [False, True, False]

    gap = np.frombuffer(a[1:2].tobytes(), dtype=rec).copy()
    with pytest.raises(ValueError, match="reserved"):
        gap.astype(nd.Nullable(rec))

    # one element of one subarray holding a value makes the record a value
    for field, index, value in (("q", (0, 2), 5.0), ("r", (0, 1), None)):
        almost = gap.copy()
        if value is None:
            almost["r"]["x"][0, 1] = 7
        else:
            almost[field][index] = value
        b = almost.astype(nd.Nullable(rec))
        assert nd.isna(b).tolist() == [False]
        assert b.astype(rec).tobytes() == almost.tobytes()


def test_record_casts_scalars_and_fill():
    rec = np.dtype([("a", "i4"), ("b", "f8")])
    plain = np.array([(1, 2.0), (3, 4.0), (5, 6.0)], rec)
    a = plain.astype(nd.Nullable(rec))
    assert a.astype(rec).tobytes() == plain.tobytes()

    a[0] = np.array((7, 8.0), rec)[()]
    a[1] = nd.NA
    with pytest.raises(ValueError, match="missing"):
        a.astype(rec)
    assert nd.filled(a, (0, 0.0)).tolist() == [(7, 8.0), (0, 0.0), (5, 6.0)]
    assert nd.to_numpy(a, na_value=(9, 9.0)).tolist() == [
        (7, 8.0), (9, 9.0), (5, 6.0)]

    a[...] = nd.NA
    assert nd.isna(a).all()


def test_record_fields_are_stored_native():
    big = np.dtype([("a", ">i4"), ("b", ">f8"), ("c", ">U2")])
    dt = nd.Nullable(big)
    assert dt == nd.Nullable(big.newbyteorder("="))

    plain = np.array([(3, 1.5, "x"), (1, 2.5, "y")], big)
    a = plain.astype(dt)
    assert a.astype(big).tobytes() == plain.tobytes()
    assert [tuple(np.sort(a)[i]) for i in range(2)] == [(1, 2.5, "y"),
                                                         (3, 1.5, "x")]
    a[0] = nd.NA
    assert nd.isna(a).tolist() == [True, False]


def test_record_equality_propagates_na():
    """numpy has no ufunc loop for `==` on records, and when `==` finds no
    loop it returns an all-False array instead of raising: `a == a` came out
    [False, False, False], without a warning."""
    rec = np.dtype([("a", "i4"), ("b", "f8")])
    plain_a = np.array([(1, 2.0), (3, 4.0), (5, 6.0), (7, np.nan)], rec)
    plain_b = np.array([(1, 2.0), (0, 4.0), (5, 6.0), (7, np.nan)], rec)
    a = plain_a.astype(nd.Nullable(rec))
    b = plain_b.astype(nd.Nullable(rec))
    b[2] = nd.NA

    eq, ne = a == b, a != b
    assert isinstance(eq.dtype, nd.NullableDType)
    assert [eq[i] for i in (0, 1, 3)] == (plain_a == plain_b)[[0, 1, 3]].tolist()
    assert [ne[i] for i in (0, 1, 3)] == (plain_a != plain_b)[[0, 1, 3]].tolist()
    assert eq[2] is nd.NA and ne[2] is nd.NA
    assert (a == a)[3] == False, "a NaN field compares unequal, as in numpy"  # noqa: E712
    assert [x for x in (a == plain_a)] == [True, True, True, False]

    other = np.dtype([("x", "i4"), ("y", "i4")])
    with pytest.raises(TypeError, match="different dtypes"):
        a == np.zeros(4, dtype=nd.Nullable(other))


@pytest.mark.parametrize("np_dtype", [">i4", ">f8", ">U3"])
def test_non_native_byte_order_is_stored_native(np_dtype):
    """Kept as given, `>i4` reserved the bytes 00 00 00 80 -- 128 read
    big-endian, not INT_MIN -- and `np.sort` crashed: numpy
    asks for a byte-swapped copy of the dtype, which a new-style DType cannot
    give, and does not check for NULL."""
    big = np.dtype(np_dtype)
    plain = np.array(["b", "a", "c"] if big.kind == "U" else [3, 1, 2], big)
    dt = nd.Nullable(big)
    assert dt == nd.Nullable(big.newbyteorder("="))

    a = plain.astype(dt)
    assert a.astype(big).tobytes() == plain.tobytes()
    s = np.sort(a)
    assert [s[i] for i in range(3)] == sorted(plain.tolist())

    if big.kind == "i":
        assert np.array([128], big).astype(dt)[0] == 128
        with pytest.raises(ValueError, match="reserved"):
            np.array([np.iinfo(np.int32).min], big).astype(dt)


def test_nd_forwards_to_numpy_and_refuses_misleading_names():
    """Whatever numpy already gets right is forwarded untouched, so `nd.sort`
    *is* `np.sort`.  The names that would look like they work are refused."""
    assert nd.sort is np.sort
    assert nd.add is np.add and nd.concatenate is np.concatenate

    with pytest.raises(AttributeError, match="skipna=True"):
        nd.nansum          # NA is not NaN
    with pytest.raises(AttributeError, match=r"nd.argmax\(a, skipna=True\)"):
        nd.nanargmax
    with pytest.raises(AttributeError, match="no attribute"):
        nd.no_such_function


def test_nd_helpers_for_what_numpy_refuses():
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 2.0, 0.0, 4.0], dtype=dt)
    a[2] = nd.NA
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=dt)

    assert nd.count(a) == 3
    assert [x for x in nd.dropna(a)] == [1.0, 2.0, 4.0]

    # np.dot cannot be implemented at all; 1-D and 2-D go through `@`
    assert nd.dot(b, b) == 30.0 and nd.dot(a, b) is nd.NA

    # np.array_equal calls np.all, which pins its accumulator to plain bool
    assert nd.array_equal(b, b) == True                       # noqa: E712
    assert nd.array_equal(b, b[:2]) == False                  # noqa: E712
    assert nd.array_equal(a, a) is nd.NA, "a gap leaves it undecided"

    # np.isin refuses a gap; this one answers NA for it
    got = nd.isin(a, [2.0, 4.0])
    assert [got[i] for i in (0, 1, 3)] == [False, True, True]
    assert got[2] is nd.NA

    # np.isclose asks result_type(dtype, 1.0) and gives up there
    close = nd.isclose(b, b)
    assert close[0] == True and nd.allclose(b, b) == True      # noqa: E712
    assert nd.allclose(a, a) is nd.NA


def test_kleene_ops_take_any_wrapped_dtype():
    """`logical_and` resolved to (f8, f8, bool), and `reduce` refuses three
    different dtypes -- that is what kept `nd.all` to bool arrays.  Everything
    resolves to Nullable[bool] now, the way numpy casts operands for plain
    arrays: `np.all([1.5, 0.0])` is a bool reduction too."""
    f8 = nd.Nullable(np.float64)
    a = np.array([1.5, 0.0], dtype=f8)
    b = np.array([2.0, 2.0], dtype=f8)
    got = np.logical_and(a, b)
    assert got.dtype == nd.Nullable(np.bool_)
    assert got[0] == True and got[1] == False           # noqa: E712

    gap = np.array([1.0, 0.0], dtype=f8)
    gap[0] = nd.NA
    assert nd.all(gap) == False                         # noqa: E712  settled
    assert nd.any(gap) is nd.NA
    assert nd.all(np.array([1.0, 2.0], dtype=f8)) == True   # noqa: E712


def test_isin_works_and_still_refuses_a_gap():
    """`np.isin` writes `|=` into a plain bool array.  The cast out of Nullable
    was `unsafe`, so numpy would not even try; at `same_kind` it does, and the
    loop is still the thing that refuses a missing value."""
    di = nd.Nullable(np.int64)
    a = np.array([1, 2], dtype=di)
    assert np.isin(a, np.array([2, 3], dtype=di)).tolist() == [False, True]

    a[0] = nd.NA
    with pytest.raises(ValueError, match="missing value"):
        np.isin(a, np.array([2, 3], dtype=di))

    # dropping NA still never happens by accident
    f8 = nd.Nullable(np.float64)
    assert not np.can_cast(f8, np.float64), "not a safe cast, and never will be"
    with pytest.raises(ValueError, match="missing value"):
        np.array([nd.NA], dtype=f8).astype(np.float64)
    assert np.array([1.0], dtype=f8).astype(np.float64).tolist() == [1.0]


def test_nd_isin_treats_a_gap_in_the_test_set_as_unknown():
    """`3 in [1, NA]` cannot be False: the gap might be 3.  SQL's `IN` says
    the same.  A hit stays True whatever else the set holds."""
    di = nd.Nullable(np.int64)
    test = np.array([1, 0], dtype=di)
    test[1] = nd.NA
    a = np.array([1, 3, 0], dtype=di)
    a[2] = nd.NA

    got = nd.isin(a, test)
    assert got[0] == True                     # noqa: E712
    assert got[1] is nd.NA and got[2] is nd.NA
    assert nd.isin(np.array([1, 3], dtype=di), np.array([1, 3], dtype=di)).tolist() == [True, True]


def test_bitwise_is_kleene_on_bool_and_propagates_on_int():
    """`&`, `|` and `^` carry two truth tables.  They were left unregistered
    because one loop cannot hold both, but the resolver sees the wrapped dtype
    and can pick: Kleene for bool, ordinary arithmetic for ints."""
    db = nd.Nullable(np.bool_)
    kleene = [("bitwise_and", nd.NA, False, False), ("bitwise_and", nd.NA, True, nd.NA),
              ("bitwise_or", nd.NA, True, True), ("bitwise_or", nd.NA, False, nd.NA),
              ("bitwise_xor", nd.NA, True, nd.NA), ("bitwise_and", True, True, True)]
    for name, x, y, want in kleene:
        got = getattr(np, name)(np.array([x], dtype=db), np.array([y], dtype=db))[0]
        if want is nd.NA:
            assert got is nd.NA, (name, x, y)
        else:
            assert got == want, (name, x, y)

    di = nd.Nullable(np.int64)
    a = np.array([6, 6], dtype=di)
    a[1] = nd.NA
    got = a & np.array([3, 3], dtype=di)
    assert got[0] == 2 and got[1] is nd.NA

    # reductions keep each table too
    flags = np.array([True, True], dtype=db)
    flags[0] = nd.NA
    assert np.bitwise_or.reduce(flags) == True                      # noqa: E712
    assert np.bitwise_or.reduce(np.array([1, 2, 4], dtype=di)) == 7
    # bitwise_and's identity is -1, which no uint8 cell can hold: no identity,
    # so numpy folds from the first element instead of turning it into a gap
    assert np.bitwise_and.reduce(np.array([7, 3], dtype=nd.Nullable(np.uint8))) == 3


def test_invert_and_logical_not():
    di = nd.Nullable(np.int64)
    a = np.array([1, 2], dtype=di)
    a[1] = nd.NA
    assert (~a)[0] == -2 and (~a)[1] is nd.NA

    db = nd.Nullable(np.bool_)
    b = np.array([True, True], dtype=db)
    b[1] = nd.NA
    assert np.logical_not(b)[0] == False and np.logical_not(b)[1] is nd.NA  # noqa: E712

    with pytest.raises(ValueError, match="reserved value"):
        ~np.array([2**31 - 1], dtype=nd.Nullable(np.int32))


def test_arithmetic_refuses_to_compute_the_reserved_value():
    """The last silent path: `2**31 - 1 + 1` wraps exactly onto INT_MIN, and
    that cell would read back as a gap nobody ever wrote."""
    i4 = nd.Nullable(np.int32)
    u1 = nd.Nullable(np.uint8)
    with pytest.raises(ValueError, match="reserved value"):
        np.array([2**31 - 1], dtype=i4) + np.array([1], dtype=i4)
    with pytest.raises(ValueError, match="reserved value"):
        np.array([0], dtype=u1) - np.array([1], dtype=u1)
    with pytest.raises(ValueError, match="reserved value"):
        np.array([2**31 - 1, 1], dtype=i4).sum()
    with pytest.raises(ValueError, match="reserved value"):
        np.array([[2**30]], dtype=i4) @ np.array([[2]], dtype=i4)

    # wrapping onto any other value still wraps, exactly as numpy does
    assert (np.array([2**31 - 1], dtype=i4) + np.array([2], dtype=i4))[0] == -(2**31) + 1
    # floats are exempt: the pattern is a NaN the hardware never synthesises
    f8 = np.array([1e308], dtype=nd.Nullable(np.float64))
    with np.errstate(over="ignore"):
        assert np.isinf((f8 * np.array([10.0], dtype=nd.Nullable(np.float64)))[0])


def test_arange():
    """`np.arange` refuses a dtype with no `fill` slot, so it used to refuse
    this one outright."""
    dt = nd.Nullable(np.float64)
    assert np.arange(5, dtype=dt).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert np.arange(2, 11, 3, dtype=nd.Nullable(np.int32)).tolist() == [2, 5, 8]

    lo = np.iinfo(np.int32).min
    with pytest.raises(ValueError, match="reserved"):
        np.arange(lo + 2, lo - 1, -1, dtype=nd.Nullable(np.int32))
    with pytest.raises(TypeError, match="count out a range"):
        np.arange(3, dtype=nd.Nullable("S3"))


def test_python_scalars_keep_the_arrays_dtype():
    """NEP 50: a Python scalar is weak and adopts the array's dtype.  Declaring
    a common dtype for Python scalars would make `np.result_type(dtype, 1.0)`
    work, and `np.select` with it, but it takes that rule away from every ufunc
    -- measured: `Nullable[f4] + 1.0` came out float64 -- so it stays undeclared
    and `result_type` with a Python scalar refuses instead."""
    f8 = np.array([1.5], dtype=nd.Nullable(np.float64))
    i8 = np.array([7, 8], dtype=nd.Nullable(np.int64))
    assert (f8 + 1.0).dtype == nd.Nullable(np.float64)
    assert (i8 // 2).dtype == nd.Nullable(np.int64)
    assert (i8 + 2).dtype == nd.Nullable(np.int64)
    with pytest.raises(np.exceptions.DTypePromotionError):
        np.result_type(nd.Nullable(np.float64), 1.0)


@pytest.mark.xfail(strict=True, reason="the promoter hands the Python scalar "
                                       "its own default dtype, so a narrow "
                                       "wrapped dtype widens; NEP 50 keeps it")
def test_a_python_scalar_does_not_widen_a_narrow_dtype():
    i4 = np.array([7], dtype=nd.Nullable(np.int32))
    assert (i4 + 2).dtype == nd.Nullable(np.int32)


def test_zero_dimensional_arrays_keep_their_shape():
    """A scalar must stay a scalar: `np.ascontiguousarray` hands back a 1-d
    array for a 0-d one, so the helpers answered with shape (1,)."""
    dt = nd.Nullable(np.float64)
    z = np.array(1.5, dtype=dt)
    gap = np.array(nd.NA, dtype=dt)
    assert z.shape == () and z[()] == 1.5 and gap[()] is nd.NA

    assert nd.isna(z).shape == () and not nd.isna(z)
    assert nd.isna(gap).shape == () and nd.isna(gap)
    assert nd.filled(gap, 7.0).shape == () and nd.filled(gap, 7.0) == 7.0
    assert nd.to_numpy(z).shape == () and nd.to_numpy(z) == 1.5

    # the byte-scan branch of the mask, and a record's field-by-field one
    s = np.array(b"ab", dtype=nd.Nullable("S3"))
    # a tuple is a sequence to numpy, so a 0-d record is built and then filled
    rec = np.zeros((), dtype=nd.Nullable(np.dtype([("a", "i4"), ("b", "f8")])))
    rec[()] = (1, 2.0)
    assert nd.isna(s).shape == () and not nd.isna(s)
    assert nd.isna(rec).shape == () and not nd.isna(rec)
    rec[()] = nd.NA
    assert nd.isna(rec).shape == () and nd.isna(rec)


def test_empty_arrays():
    dt = nd.Nullable(np.float64)
    e = np.array([], dtype=dt)
    assert e.itemsize == 8 and nd.isna(e).tolist() == []
    assert (e + e).shape == (0,) and np.sort(e).shape == (0,)
    assert np.nonzero(e)[0].tolist() == [] and nd.unique(e).shape == (0,)
    assert np.zeros(0, nd.Nullable("S3")).shape == (0,)
    with pytest.raises(ValueError, match="empty sequence"):
        e.argmax()


def test_takes_numpy_scalars_and_arrays():
    """Assigning a numpy scalar is a cast from the scalar's dtype.  Without one
    for S, U and V, records and `np.bytes_` were refused and an unstructured
    `np.void` segfaulted."""
    rec = np.dtype([("a", "i4"), ("b", "f8")])
    a = np.zeros(3, dtype=nd.Nullable(rec))
    a[0] = np.array((3, 4.0), rec)[()]
    a[1:] = np.array([(1, 2.0), (5, 6.0)], rec)
    assert a.astype(rec).tolist() == [(3, 4.0), (1, 2.0), (5, 6.0)]

    v = np.zeros(2, dtype=nd.Nullable("V4"))
    v[0] = np.void(b"zz\xfe\xff")
    assert v[0] == np.void(b"zz\xfe\xff")

    s = np.zeros(2, dtype=nd.Nullable("S3"))
    s[0] = np.bytes_(b"ab")
    u = np.zeros(2, dtype=nd.Nullable("U3"))
    u[0] = np.str_("ab")
    assert s[0] == b"ab" and u[0] == "ab"


@pytest.mark.parametrize("np_dtype", [np.float64, np.dtype("S3")])
def test_nonzero_on_a_gap_raises_on_a_large_array(np_dtype):
    """Above 500 elements numpy runs `nonzero` with the GIL released unless
    the dtype says it needs the Python API; raising there was a segfault."""
    a = np.zeros(2000, dtype=nd.Nullable(np_dtype))
    a[1500] = nd.NA
    with pytest.raises(ValueError, match="truth value"):
        np.nonzero(a)
    a[1500] = 1 if np_dtype == np.float64 else b"x"
    assert np.nonzero(a)[0].tolist() == [1500]


# ======================================================================
# Shared semantics: propagation, comparison, reduction, sorting, statistics
# ======================================================================


def test_conformance_roundtrip():
    a = np.array([1.0, 2.0, 3.0], dtype=nd.Nullable(np.float64))
    a[1] = nd.NA
    assert a[0] == 1.0 and a[1] is nd.NA and a[2] == 3.0


@pytest.mark.parametrize("op", ["add", "subtract", "multiply", "true_divide"])
def test_conformance_propagation(op):
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 2.0, 3.0], dtype=dt)
    b = np.array([10.0, 20.0, 30.0], dtype=dt)
    a[1] = nd.NA
    b[2] = nd.NA

    res = getattr(np, op)(a, b)
    assert res[0] == getattr(np, op)(1.0, 10.0)
    assert res[1] is nd.NA
    assert res[2] is nd.NA


def test_conformance_mixed_with_plain():
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 2.0], dtype=dt)
    a[1] = nd.NA
    for res in (a + 1.0, 1.0 + a, a + np.array([1.0, 1.0])):
        assert res[0] == 2.0
        assert res[1] is nd.NA


def test_conformance_comparison_yields_na():
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 2.0, 3.0], dtype=dt)
    a[1] = nd.NA

    assert (a > 2)[0] == False
    assert (a > 2)[1] is nd.NA
    assert (a == a)[1] is nd.NA


def test_conformance_reduction_propagates():
    dt = nd.Nullable(np.float64)
    clean = np.array([1.0, 2.0, 3.0, 4.0], dtype=dt)
    assert clean.sum() == 10.0
    assert np.multiply.reduce(clean) == 24.0

    a = clean.copy()
    a[1] = nd.NA
    assert a.sum() is nd.NA
    assert np.add.reduce(a, where=nd.notna(a)) == 8.0


def test_conformance_accumulate():
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=dt)
    got = np.add.accumulate(a)
    assert [got[i] for i in range(4)] == [1.0, 3.0, 6.0, 10.0]

    a[2] = nd.NA
    got = np.add.accumulate(a)
    assert got[1] == 3.0 and got[2] is nd.NA and got[3] is nd.NA


def test_conformance_unary():
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 4.0], dtype=dt)
    a[1] = nd.NA
    assert np.sqrt(a)[0] == 1.0
    assert np.sqrt(a)[1] is nd.NA
    assert (-a)[1] is nd.NA


def test_conformance_no_silent_way_out():
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 2.0], dtype=dt)
    a[1] = nd.NA

    with pytest.raises(ValueError, match="missing"):
        a.astype(np.float64)
    with pytest.raises(ValueError, match="missing"):
        nd.to_numpy(a)
    with pytest.raises(ValueError, match="truth value"):
        np.nonzero(a)


def test_conformance_helpers():
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 2.0, 3.0], dtype=dt)
    a[1] = nd.NA

    np.testing.assert_array_equal(nd.isna(a), [False, True, False])
    np.testing.assert_array_equal(nd.filled(a, 0.0), [1.0, 0.0, 3.0])
    np.testing.assert_array_equal(nd.to_numpy(a, na_value=-1.0), [1.0, -1.0, 3.0])


def test_conformance_sorting():
    dt = nd.Nullable(np.float64)
    a = np.array([3.0, 1.0, 2.0, 5.0], dtype=dt)
    a[2] = nd.NA

    got = np.sort(a)
    assert [got[i] for i in range(3)] == [1.0, 3.0, 5.0]
    assert got[3] is nd.NA
    assert a.argmax() == 2


def test_conformance_nan_is_not_na():
    dt = nd.Nullable(np.float64)
    a = np.array([np.nan, 1.0], dtype=dt)
    a[1] = nd.NA
    assert np.isnan(a[0]) and a[0] is not nd.NA
    assert a[1] is nd.NA


@pytest.mark.parametrize("np_dtype", [np.float64, np.float32, np.int64,
                                      np.int32, np.int8])
def test_conformance_across_dtypes(np_dtype):
    dt = nd.Nullable(np_dtype)
    a = np.array([1, 2, 3], dtype=dt)
    a[1] = nd.NA
    res = a + a
    assert res[0] == 2 and res[1] is nd.NA and res[2] == 6


@pytest.mark.parametrize("op, a, b, want", [
    ("logical_and", X, F, False),   # False quyet dinh
    ("logical_and", X, T, None),
    ("logical_or", X, T, True),     # True quyet dinh
    ("logical_or", X, F, None),
    ("logical_xor", X, T, None),    # xor is never certain
    ("logical_and", T, T, True),
    ("logical_or", F, F, False),
])
def test_conformance_kleene(op, a, b, want):
    db = nd.Nullable(np.bool_)

    def mk(v):
        arr = np.array([True], dtype=db)
        arr[0] = nd.NA if v is None else v
        return arr

    got = getattr(np, op)(mk(a), mk(b))[0]
    assert (None if got is nd.NA else bool(got)) is want


@pytest.mark.parametrize("op", ["maximum", "minimum"])
@pytest.mark.parametrize("view", ["contiguous", "step2", "step3", "reversed"])
def test_conformance_minmax_any_stride(op, view):
    """
    `maximum`/`minimum` borrow a numpy loop whose SIMD path converts the byte
    stride to an element stride by dividing, without checking that it divides
    evenly (see the note in nulldtype.c).  Anything that is not a whole number
    of elements has to stay off the fast path, so exercise every stride shape.
    """
    dt = nd.Nullable(np.float64)
    a = np.array([5.0, 1.0, 0.0, 9.0, 2.0, 0.0, 7.0, 3.0], dtype=dt)
    b = np.array([2.0, 8.0, 4.0, 1.0, 0.0, 0.0, 7.0, 3.0], dtype=dt)
    a[2] = a[5] = nd.NA
    b[4] = b[5] = nd.NA

    sl = {"contiguous": slice(None), "step2": slice(None, None, 2),
          "step3": slice(None, None, 3), "reversed": slice(None, None, -1)}[view]
    av, bv = a[sl], b[sl]

    got = getattr(np, op)(av, bv)
    pick = max if op == "maximum" else min
    for i in range(len(got)):
        if av[i] is nd.NA or bv[i] is nd.NA:
            assert got[i] is nd.NA, f"{view}[{i}] should propagate"
        else:
            assert got[i] == pick(av[i], bv[i]), f"{view}[{i}]"


@pytest.mark.parametrize("op", ["add", "multiply", "subtract",
                                "maximum", "minimum"])
def test_conformance_na_raises_no_fp_warning(op):
    """
    Touching a gap is not an arithmetic error.  Nullable stores NA as a NaN,
    so the pattern must have the quiet bit set -- R's own NA_real_ is a
    signaling NaN, which would make numpy report "invalid value encountered"
    on every operation that merely passes over missing data.
    """
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 0.0, 3.0], dtype=dt)
    a[1] = nd.NA

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = getattr(np, op)(a, a)
    assert res[1] is nd.NA


# ---------------------------------------------------------------- op coverage
# The names come from the C module, so these sweeps always cover exactly what
# is registered.  Adding an op to `binop_names` puts it under test for free.

BINOPS = _c.binop_names
UNOPS = _c.unop_names

SWEEP_A = [1.5, 2.0, 3.0, 4.0]
SWEEP_B = [2.0, 2.0, 2.0, 0.5]

# No single input is legal for every op: arcsin and friends need |x| <= 1,
# arccosh needs x >= 1.  A domain error is a real warning, not a leak from a
# gap, so the sweeps have to stay inside each op's domain to mean anything.
UNIT_DOMAIN = ["arcsin", "arccos", "arctanh"]


def _unop_input(name):
    return [0.1, 0.25, 0.5, 0.75] if name in UNIT_DOMAIN else SWEEP_A


# `&`, `|`, `^` and `~` are integer ops; sweeping them in float64 says nothing
INT_ONLY = ("bitwise_and", "bitwise_or", "bitwise_xor", "invert")


def _sweep_kind(name):
    return np.int64 if name in INT_ONLY else np.float64


def _sweep_arrays(plain_a, plain_b=None, kind=np.float64):
    dt = nd.Nullable(kind)
    a = np.array(np.asarray(plain_a).astype(kind), dtype=dt)
    a[2] = nd.NA
    if plain_b is None:
        return a
    return a, np.array(np.asarray(plain_b).astype(kind), dtype=dt)


@pytest.mark.parametrize("name", BINOPS)
def test_every_binop_matches_numpy_and_propagates(name):
    kind = _sweep_kind(name)
    a, b = _sweep_arrays(SWEEP_A, SWEEP_B, kind)
    want = getattr(np, name)(np.array(SWEEP_A).astype(kind),
                             np.array(SWEEP_B).astype(kind))
    got = getattr(np, name)(a, b)

    assert got[2] is nd.NA, "a gap on either side must reach the result"
    for i in (0, 1, 3):
        assert got[i] == want[i], f"{name}[{i}]"


@pytest.mark.parametrize("name", UNOPS)
def test_every_unop_matches_numpy_and_propagates(name):
    kind = _sweep_kind(name)
    plain = np.array(_unop_input(name)).astype(kind)
    a = _sweep_arrays(plain, kind=kind)
    want = getattr(np, name)(plain)
    got = getattr(np, name)(a)

    assert got[2] is nd.NA
    for i in (0, 1, 3):
        assert got[i] == want[i], f"{name}[{i}]"


@pytest.mark.parametrize("name", BINOPS + UNOPS)
def test_no_spurious_fp_warning_over_gaps(name):
    """
    A value that is not there must not produce an arithmetic complaint.

    Nullable keeps a NaN in every gap and normally lets the wrapped loop run
    right over it.  That is silent for arithmetic, but `<` and `>` are
    *signaling* predicates -- IEEE 754 has them raise FE_INVALID even on a
    quiet NaN -- so any op that compares its operands has to take the masked
    path instead (`binop_signals_on_gap` in nulldtype.c).  This sweep is what
    keeps that list honest.
    """
    kind = _sweep_kind(name)
    if name in UNOPS:
        args = (_sweep_arrays(_unop_input(name), kind=kind),)
    else:
        args = _sweep_arrays(SWEEP_A, SWEEP_B, kind)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        getattr(np, name)(*args)


@pytest.mark.parametrize("name", ["fmax", "fmin"])
def test_nan_swallowing_ops_still_propagate_na(name):
    """
    `fmax`/`fmin` return the non-NaN side by design, so for Nullable a gap
    leaves no NaN behind in the output.  The fixup's "is there any NaN to fix?"
    pre-scan would then skip the row it was supposed to repair.  NA is not NaN:
    it propagates whatever the op does with NaN.
    """
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 2.0, 3.0], dtype=dt)
    b = np.array([9.0, 9.0, 9.0], dtype=dt)
    a[1] = nd.NA

    got = getattr(np, name)(a, b)
    assert got[1] is nd.NA
    assert got[0] == (9.0 if name == "fmax" else 1.0)


# --------------------------------------------------------------- statistics
# `np.median`/`percentile`/`quantile` are the one place where this dtype cannot
# stop numpy from returning a wrong number: they sort, NA lands last, and the
# middle shifts.  `nd.median` and friends are the supported way.

CLEAN = [3.0, 1.0, 5.0]


def _with_gap(values=(3.0, 1.0, 0.0, 5.0), gap=2):
    a = np.array(list(values), dtype=nd.Nullable(np.float64))
    a[gap] = nd.NA
    return a


def test_numpy_median_is_wrong_and_nd_median_is_not():
    a = _with_gap()

    # documented trap: a plausible number, quietly wrong
    assert np.median(a) == 4.0
    # the median of the values that exist is 3.0
    assert nd.median(a, skipna=True) == np.median(CLEAN) == 3.0
    # and by default a missing input means a missing answer
    assert nd.median(a) is nd.NA


@pytest.mark.parametrize("name", ["sum", "prod", "min", "max", "mean",
                                  "std", "var", "median"])
def test_stat_propagates_by_default_and_skips_on_request(name):
    a = _with_gap()
    fn = getattr(nd, name)

    assert fn(a) is nd.NA
    assert fn(a, skipna=True) == getattr(np, name)(np.array(CLEAN))


@pytest.mark.parametrize("source", ["empty", "all-missing"])
def test_reducing_over_nothing(source):
    """
    An empty input and one that `skipna` emptied are the same situation, so
    they must answer the same way -- they did not, once.

    `sum` and `prod` have identity elements, so 0 and 1 are the answers rather
    than inventions; R and numpy agree.  Everything else has no identity: the
    mean or maximum of no numbers is unknown, so NA.  NaN would be actively
    wrong here, because in this dtype NaN is an ordinary value a column may
    legitimately hold -- `nd.median` of an empty array used to return it.
    """
    dt = nd.Nullable(np.float64)
    if source == "empty":
        a, kw = np.array([], dtype=dt), {}
    else:
        a = np.array([0.0, 0.0], dtype=dt)
        a[0] = a[1] = nd.NA
        kw = {"skipna": True}

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert nd.sum(a, **kw) == 0
        assert nd.prod(a, **kw) == 1
        for name in ("mean", "median", "min", "max", "std", "var"):
            assert getattr(nd, name)(a, **kw) is nd.NA, name
        assert nd.quantile(a, 0.5, **kw) is nd.NA


def test_argmax_of_all_missing_matches_numpys_all_nat():
    """
    0, the same answer numpy gives for an all-NaT or all-nan array.  Raising
    would be better but the slot cannot: numpy ignores its return value.  Nothing is invented -- the index leads to
    a gap, not to a number.
    """
    a = np.array([1.0, 2.0], dtype=nd.Nullable(np.float64))
    a[0] = a[1] = nd.NA

    assert a.argmax() == 0
    assert a[a.argmax()] is nd.NA


@pytest.mark.parametrize("fn, q", [("quantile", 0.5), ("percentile", 50)])
def test_quantile_scalar_and_array_q(fn, q):
    a = _with_gap()
    f = getattr(nd, fn)
    ref = getattr(np, fn)

    assert f(a, q) is nd.NA
    assert f(a, q, skipna=True) == ref(np.array(CLEAN), q)

    many = f(a, [q, q])
    assert isinstance(many.dtype, type(a.dtype))
    assert many[0] is nd.NA and many[1] is nd.NA
    assert list(f(a, [q, q], skipna=True)) == list(ref(np.array(CLEAN), [q, q]))


def test_unique_keeps_na_once_and_last():
    a = np.array([3.0, 1.0, 3.0, 0.0, 1.0, 0.0], dtype=nd.Nullable(np.float64))
    a[3] = a[5] = nd.NA

    got = nd.unique(a)
    assert isinstance(got.dtype, type(a.dtype))
    assert [got[i] for i in range(len(got) - 1)] == [1.0, 3.0]
    assert got[-1] is nd.NA, "two gaps are one distinct value, and it sorts last"


def test_unique_without_gaps_has_no_na():
    a = np.array([3.0, 1.0, 3.0], dtype=nd.Nullable(np.float64))
    got = nd.unique(a)
    assert [got[i] for i in range(len(got))] == [1.0, 3.0]


# ------------------------------------------------------------- the NA scalar
# Indexing hands back a plain Python value for a present element and the NA
# singleton for a missing one.  Whatever a caller does with `a[0]` they will
# eventually do with `a[1]`, so the singleton has to behave, not explode.


@pytest.mark.parametrize("expr", [
    lambda NA: NA + 1, lambda NA: 1 + NA,
    lambda NA: NA - 1, lambda NA: 1 - NA,
    lambda NA: NA * 2, lambda NA: 2 * NA,
    lambda NA: NA / 2, lambda NA: 2 / NA,
    lambda NA: NA // 2, lambda NA: NA % 2, lambda NA: NA ** 2,
    lambda NA: -NA, lambda NA: +NA, lambda NA: abs(NA),
    lambda NA: NA + NA, lambda NA: NA * NA,
])
def test_na_scalar_arithmetic_is_na(expr):
    assert expr(nd.NA) is nd.NA


@pytest.mark.parametrize("expr", [
    lambda NA: NA > 1, lambda NA: NA < 1, lambda NA: NA >= 1,
    lambda NA: NA <= 1, lambda NA: NA == 1, lambda NA: NA != 1,
    lambda NA: NA == NA,
])
def test_na_scalar_comparison_is_na(expr):
    """Two things you do not know are not thereby known to be equal."""
    assert expr(nd.NA) is nd.NA


def test_na_scalar_identity_still_answers():
    """`is` is how code asks "is this missing"; comparison must not break it."""
    a = np.array([1.0, 2.0], dtype=nd.Nullable(np.float64))
    a[1] = nd.NA
    assert a[1] is nd.NA
    assert a[0] is not nd.NA
    # and it stays usable as a key
    assert {nd.NA: "gap"}[nd.NA] == "gap"


def test_bool_of_na_refuses():
    """
    This used to answer True -- Python's default for an object with no
    `__bool__` -- which quietly made `if a[i]:` read a hole in the data as a
    yes.  There is no right answer, so it must not invent one.
    """
    with pytest.raises(TypeError, match="truth value of NA is undefined"):
        bool(nd.NA)

    a = np.array([0.0, 1.0], dtype=nd.Nullable(np.float64))
    a[0] = nd.NA
    with pytest.raises(TypeError, match="truth value of NA is undefined"):
        if a[0]:
            pass


def test_na_scalar_matches_the_array_loops():
    """The scalar and the elementwise path must not disagree."""
    dt = nd.Nullable(np.float64)
    a = np.array([1.0, 2.0], dtype=dt)
    a[1] = nd.NA
    ones = np.array([1.0, 1.0], dtype=dt)

    assert (a + ones)[1] is nd.NA and (a[1] + 1.0) is nd.NA
    assert (a > ones)[1] is nd.NA and (a[1] > 1.0) is nd.NA
    assert np.negative(a)[1] is nd.NA and (-a[1]) is nd.NA


# ------------------------------------------------------- Kleene reductions
# Elementwise Kleene was already right; folding it was not.  A settling value
# -- False for `all`, True for `any` -- fixes the answer for good, and no later
# gap can unfix it: `any([True, NA, False])` is True whatever the NA turns out
# to be.  The reduce used to clear the decision when it met the gap, so the
# answer depended on where the gap sat.

def _kleene_ref(op, values):
    settling = False if op == "all" else True
    if settling in values:
        return settling
    if None in values:
        return None
    return not settling


@pytest.mark.parametrize("op", ["all", "any"])
@pytest.mark.parametrize("values", list(itertools.product(
    [True, False, None], repeat=3)))
def test_kleene_reduce_exhaustive(op, values):
    a = np.array([True] * len(values), dtype=nd.Nullable(np.bool_))
    for i, v in enumerate(values):
        a[i] = nd.NA if v is None else v

    got = getattr(nd, op)(a)
    got = None if got is nd.NA else bool(got)
    assert got == _kleene_ref(op, values)


def test_kleene_reduce_does_not_depend_on_gap_position():
    """The regression itself: same multiset, different order, same answer."""
    def fold(values):
        a = np.array([True] * len(values), dtype=nd.Nullable(np.bool_))
        for i, v in enumerate(values):
            a[i] = nd.NA if v is None else v
        got = nd.any(a)
        return None if got is nd.NA else bool(got)

    assert fold([True, None, False]) is True
    assert fold([False, None, True]) is True
    assert fold([None, False, True]) is True
    assert fold([True, False, None]) is True


# ------------------------------------------------------------ clip, matmul
PLAIN_A = np.array([[1.0, 2.0], [3.0, 4.0]])
PLAIN_B = np.array([[5.0, 6.0], [7.0, 8.0]])


@pytest.mark.parametrize("bounds", ["scalars", "arrays", "mixed", "plain"])
def test_clip_matches_numpy_and_propagates(bounds):
    dt = nd.Nullable(np.float64)
    plain = np.array([0.5, 2.0, 3.0, 9.0])
    a = plain.astype(dt)
    a[2] = nd.NA

    lo, hi = {
        "scalars": (1.0, 5.0),
        "arrays": (np.array([1.0] * 4, dtype=dt), np.array([5.0] * 4, dtype=dt)),
        "mixed": (np.array([1.0] * 4, dtype=dt), 5.0),
        "plain": (np.array([1.0] * 4), 5.0),
    }[bounds]

    got = np.clip(a, lo, hi)
    want = np.clip(plain, 1.0, 5.0)
    assert got[2] is nd.NA
    for i in (0, 1, 3):
        assert got[i] == want[i]


def test_clip_propagates_a_missing_bound():
    """"Keep this between 1 and something I do not know" has no answer."""
    dt = nd.Nullable(np.float64)
    a = np.array([0.5, 2.0], dtype=dt)
    lo = np.array([1.0, 1.0], dtype=dt)
    lo[0] = nd.NA

    got = np.clip(a, lo, 5.0)
    assert got[0] is nd.NA
    assert got[1] == 2.0


def test_matmul_matches_numpy():
    dt = nd.Nullable(np.float64)
    A, B = PLAIN_A.astype(dt), PLAIN_B.astype(dt)
    want = PLAIN_A @ PLAIN_B

    got = A @ B
    for i in range(2):
        for j in range(2):
            assert got[i, j] == want[i, j]


def test_matmul_gap_takes_out_a_whole_row_and_column():
    """
    `C[i,j]` sums over k, so it is missing exactly when row i of A or column j
    of B has a gap anywhere -- not just the one cell that touched it.
    """
    dt = nd.Nullable(np.float64)
    A, B = PLAIN_A.astype(dt), PLAIN_B.astype(dt)

    An = A.copy()
    An[0, 1] = nd.NA
    out = An @ B
    assert out[0, 0] is nd.NA and out[0, 1] is nd.NA, "row 0 of A is tainted"
    assert out[1, 0] == (PLAIN_A @ PLAIN_B)[1, 0], "row 1 is untouched"

    Bn = B.copy()
    Bn[1, 0] = nd.NA
    out = A @ Bn
    assert out[0, 0] is nd.NA and out[1, 0] is nd.NA, "column 0 of B is tainted"
    assert out[0, 1] == (PLAIN_A @ PLAIN_B)[0, 1]


@pytest.mark.parametrize("shape", ["vec@mat", "mat@vec", "vec@vec", "batched"])
def test_matmul_handles_the_signature_shortcuts(shape):
    """`(n?,k),(k,m?)->(n?,m?)`: the `?` dimensions have their own code path."""
    dt = nd.Nullable(np.float64)
    A = PLAIN_A.astype(dt)
    v = np.array([1.0, 2.0], dtype=dt)
    p = np.array([1.0, 2.0])

    got, want = {
        "vec@mat": (v @ A, p @ PLAIN_A),
        "mat@vec": (A @ v, PLAIN_A @ p),
        "vec@vec": (v @ v, p @ p),
        "batched": (np.stack([A, A]) @ A, np.stack([PLAIN_A, PLAIN_A]) @ PLAIN_A),
    }[shape]

    assert np.shape(got) == np.shape(want)
    if np.shape(want) == ():
        assert got == want
    else:
        for idx in np.ndindex(np.shape(want)):
            assert got[idx] == want[idx], idx


def test_np_dot_is_out_of_reach_and_says_so():
    """
    `np.dot` goes through the legacy `dotfunc` slot, which numpy calls with a
    NULL array pointer (PyArray_MatrixProduct2).  A parametric dtype cannot
    find its own element layout from that, so the slot stays unset and numpy
    reports it plainly.  `@` is the supported spelling.
    """
    a = np.array([1.0, 2.0], dtype=nd.Nullable(np.float64))
    with pytest.raises(ValueError, match="dot not available"):
        np.dot(a, a)


# ------------------------------------------------------- NA bit patterns
# The exact bytes a gap is stored as, for every bitpattern type.  complex128
# used to get only its real half: the pattern buffer was 8 bytes, the copy
# read 16, and the imaginary half was whatever sat past the end of the
# descriptor (AddressSanitizer: heap-buffer-overflow).  Pinning the bytes
# catches that kind of over-read even without a sanitizer.

_F8_NA = 0x7FFFFFFFFFFFFFFF
_F4_NA = 0x7FFFFFFF


@pytest.mark.parametrize("np_dtype, view, want", [
    (np.float64,    np.uint64, [_F8_NA]),
    (np.float32,    np.uint32, [_F4_NA]),
    (np.float16,    np.uint16, [0x7FFF]),
    (np.complex128, np.uint64, [_F8_NA, _F8_NA]),
    (np.complex64,  np.uint32, [_F4_NA, _F4_NA]),
    (np.int64,      np.uint64, [0x8000000000000000]),
    (np.int32,      np.uint32, [0x80000000]),
    (np.int16,      np.uint16, [0x8000]),
    (np.int8,       np.uint8,  [0x80]),
    (np.bool_,      np.uint8,  [2]),
    (np.uint64,     np.uint64, [0xFFFFFFFFFFFFFFFF]),
    (np.uint32,     np.uint32, [0xFFFFFFFF]),
    (np.uint16,     np.uint16, [0xFFFF]),
    (np.uint8,      np.uint8,  [0xFF]),
    (np.dtype("M8[D]"), np.uint64, [0x8000000000000000]),
    (np.dtype("m8[s]"), np.uint64, [0x8000000000000000]),
])
def test_bitpattern_na_bytes_are_exact(np_dtype, view, want):
    dt = nd.Nullable(np_dtype)
    # start from a filler that differs from the pattern, so a byte that never
    # gets written cannot pass for one that did
    fill = b"\x00" if np.dtype(np_dtype).kind == "u" else b"\xff"
    a = np.frombuffer(fill * dt.itemsize * 3, dtype=dt).copy()
    a[1] = nd.NA

    raw = a[1:2].view(np.uint8).view(view)
    assert [int(x) for x in raw] == want
    assert a[1] is nd.NA



# ---------------------------------------------------------- unsigned ints
# UINT_MAX is the reserved value.  Giving up a real number is only acceptable
# because it is refused loudly on every way in: a silent path would turn every
# white pixel of a uint8 image into a gap.

UINTS = [np.uint8, np.uint16, np.uint32, np.uint64]


@pytest.mark.parametrize("np_dtype", UINTS)
def test_uint_max_is_refused_on_construction(np_dtype):
    top = int(np.iinfo(np_dtype).max)
    with pytest.raises(ValueError, match="reserved"):
        np.array([1, top], dtype=nd.Nullable(np_dtype))

    a = np.array([0, top - 1], dtype=nd.Nullable(np_dtype))
    assert int(a[0]) == 0 and int(a[1]) == top - 1, "0 must not be a gap"
    assert nd.isna(a).tolist() == [False, False]


@pytest.mark.parametrize("np_dtype, reserved", [
    (np.uint8, 2**8 - 1), (np.uint16, 2**16 - 1), (np.uint64, 2**64 - 1),
    (np.int8, -2**7), (np.int64, -2**63),
])
def test_casting_the_reserved_value_in_refuses(np_dtype, reserved):
    """`astype` is the other way in, and the one a plain image array takes."""
    plain = np.array([1, reserved], dtype=np_dtype)
    with pytest.raises(ValueError, match="reserved"):
        plain.astype(nd.Nullable(np_dtype))


def test_uint_arithmetic_wraps_like_numpy():
    """Cells hold the value itself, so numpy's uint loops apply unchanged."""
    dt = nd.Nullable(np.uint8)
    a = np.array([200, 5, 7], dtype=dt)
    b = np.array([100, 10, 7], dtype=dt)
    a[2] = nd.NA
    with np.errstate(over="ignore"):
        s, d = a + b, a - b
    assert int(s[0]) == 44 and int(d[1]) == 251
    assert s[2] is nd.NA and d[2] is nd.NA


@pytest.mark.parametrize("left, right", [
    (np.uint8, np.int8), (np.uint8, np.int16), (np.uint16, np.int32),
    (np.uint8, np.uint16), (np.int8, np.int16),
])
def test_na_survives_mixed_width_promotion(left, right):
    """
    Mixed widths cast before the loop runs: uint8 + int8 computes in int16.
    The gap has to be translated, not converted -- uint8's NA byte is 255, and
    255 is an ordinary int16.
    """
    a = np.array([1, 0], dtype=nd.Nullable(left))
    a[1] = nd.NA
    b = np.array([1, 1], dtype=nd.Nullable(right))
    for r in (a + b, b + a):
        assert int(r[0]) == 2
        assert r[1] is nd.NA



# ------------------------------------------------------- float NA convention
# Every float width stores NA as all bits but the sign: 0x7FFF, 0x7FFFFFFF,
# 0x7FFF...F.  A quiet NaN with the largest payload, which ordinary arithmetic
# never makes.

_FLOATS = [np.float16, np.float32, np.float64, np.complex64, np.complex128]


@pytest.mark.parametrize("np_dtype", _FLOATS)
def test_ordinary_nans_are_values_not_gaps(np_dtype):
    """`np.nan` is 0x7FF8..., -nan flips the sign, and x86 makes inf - inf
    0xFFF8...; none of them may read as NA."""
    dt = nd.Nullable(np_dtype)
    with np.errstate(invalid="ignore"):
        inf = np.array([np.inf], dtype=np_dtype)
        made = (inf - inf)[0]
    a = np.array([np.nan, -np.nan, made], dtype=dt)
    assert nd.isna(a).tolist() == [False, False, False]
    assert all(a[i] is not nd.NA for i in range(3))


@pytest.mark.parametrize("np_dtype, view, flipped", [
    (np.float64, np.uint64, 0xFFFFFFFFFFFFFFFF),
    (np.float32, np.uint32, 0xFFFFFFFF),
    (np.float16, np.uint16, 0xFFFF),
])
def test_float_na_ignores_the_sign(np_dtype, view, flipped):
    """A gap whose sign got flipped is still a gap."""
    dt = nd.Nullable(np_dtype)
    a = np.array([0.0, 0.0], dtype=dt)
    a.view(np.uint8).view(view)[0] = flipped
    assert a[0] is nd.NA
    assert nd.isna(a).tolist() == [True, False]



# ---------------------------------------------------- long double -> double
# Nullable has no long double.  longdouble and clongdouble are stored as
# float64 and complex128, and say so with a warning wherever that loses
# anything -- on Windows long double already is a double, so nothing is lost.

_LD_IS_WIDER = np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant


@pytest.mark.parametrize("np_dtype, stored", [(np.longdouble, np.float64),
                                              (np.clongdouble, np.complex128)])
def test_long_double_is_stored_as_double(np_dtype, stored):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dt = nd.Nullable(np_dtype)
    assert dt == nd.Nullable(stored)
    warned = [w for w in caught if issubclass(w.category, nd.LongDoubleWarning)]
    assert bool(warned) == _LD_IS_WIDER, "warn exactly when precision is lost"


@pytest.mark.skipif(not _LD_IS_WIDER, reason="long double is a double here")
def test_long_double_values_are_rounded_to_double():
    third = np.longdouble(1) / 3
    with pytest.warns(nd.LongDoubleWarning, match="float64"):
        dt = nd.Nullable(np.longdouble)
    a = np.array([third, 1, 2], dtype=dt)
    a[1] = nd.NA
    assert a.dtype == nd.Nullable(np.float64)
    assert a[0] == np.float64(third)
    assert a[1] is nd.NA
    assert nd.isna(a).tolist() == [False, True, False]


def test_record_long_double_fields_are_stored_as_double():
    """The same substitution inside a record: the record is rebuilt with
    doubles, and numpy's field-by-field cast carries the values over."""
    rec = np.dtype([("a", "i2"), ("g", "g"), ("z", "G", (2,))], align=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dt = nd.Nullable(rec)
    warned = [w for w in caught if issubclass(w.category, nd.LongDoubleWarning)]
    assert bool(warned) == _LD_IS_WIDER, "warn exactly when precision is lost"

    stored = dt.wrapped
    assert stored.names == ("a", "g", "z")
    assert stored["g"] == np.float64
    assert stored["z"].base == np.complex128 and stored["z"].shape == (2,)

    plain = np.zeros(2, dtype=rec)
    plain["a"] = [1, 2]
    plain["g"] = [0.5, 1.5]
    plain["z"] = [[1 + 2j, 3], [4, 5j]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nd.LongDoubleWarning)
        a = plain.astype(dt)
    back = a.astype(stored)
    assert back["a"].tolist() == [1, 2]
    assert back["g"].tolist() == [0.5, 1.5]
    assert back["z"].tolist() == [[1 + 2j, 3], [4, 5j]]


def test_long_double_results_stay_float64():
    """The stored dtype is float64, so every result is too.  A long double
    operand is rounded on the way in, not promoted back on the way out."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nd.LongDoubleWarning)
        dt = nd.Nullable(np.longdouble)
        a = np.array([1.5, 2.5], dtype=np.longdouble).astype(dt)
        mixed = a * np.array([2.0, 2.0], dtype=np.longdouble)

    f8 = nd.Nullable(np.float64)
    assert a.dtype == f8 and (a + a).dtype == f8 and np.sqrt(a).dtype == f8
    assert mixed.dtype == f8 and mixed[0] == 3.0

    # getting one back is a plain widening cast; the precision is long gone
    back = a.astype(np.longdouble)
    assert back.dtype == np.longdouble and back[0] == np.longdouble(1.5)


@pytest.mark.skipif(not _LD_IS_WIDER, reason="long double is a double here")
def test_a_long_double_operand_warns_on_every_op():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nd.LongDoubleWarning)
        a = np.array([1.5], dtype=np.longdouble).astype(nd.Nullable(np.longdouble))
    plain = np.array([2.0], dtype=np.longdouble)
    with pytest.warns(nd.LongDoubleWarning, match="float64"):
        a * plain


def test_plain_long_double_array_casts_in():
    plain = np.array([1.5, 2.5], dtype=np.longdouble)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nd.LongDoubleWarning)
        dt = nd.Nullable(np.longdouble)
    got = plain.astype(dt)
    assert got.dtype == nd.Nullable(np.float64)
    assert got.tolist() == [1.5, 2.5]


def test_mixing_with_long_double_follows_the_substitution():
    """numpy promotes float64 + longdouble to longdouble; a Nullable long double
    is a float64, so that is what comes out, and the gap still propagates."""
    a = np.array([1.0, 0.0], dtype=nd.Nullable(np.float64))
    a[1] = nd.NA
    b = np.array([1.0, 1.0], dtype=np.longdouble)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nd.LongDoubleWarning)
        r = a + b
    assert r.dtype == nd.Nullable(np.float64)
    assert r[0] == 2.0 and r[1] is nd.NA


# ------------------------------------------------ skipna across the surface

_STATS = ["sum", "prod", "min", "max", "mean", "std", "var", "median"]


@pytest.mark.parametrize("name", _STATS)
@pytest.mark.parametrize("axis", [0, 1, -1, (0, 1)])
def test_reduction_axis_matches_numpy_lane_by_lane(name, axis):
    """With an axis, a lane with a gap is NA unless `skipna=True`, and every
    other lane is numpy's own answer -- shape included, keepdims too."""
    rng = np.random.default_rng(0)
    plain = rng.normal(size=(4, 5))
    gaps = np.zeros((4, 5), dtype=bool)
    gaps[1, 2] = True
    gaps[3, :] = True
    a = plain.astype(nd.Nullable(np.float64))
    a[gaps] = nd.NA

    fn = getattr(nd, name)
    for keepdims in (False, True):
        got = fn(a, axis=axis, keepdims=keepdims)
        want = getattr(np, name)(plain, axis=axis, keepdims=keepdims)
        assert got.shape == want.shape
        hit = np.logical_or.reduce(gaps, axis=axis, keepdims=keepdims)
        np.testing.assert_array_equal(nd.isna(got), hit)
        np.testing.assert_allclose(nd.filled(got, 0.0)[~hit], want[~hit])

    got = fn(a, axis=axis, skipna=True)
    for idx in np.ndindex(got.shape):
        lanes = plain if axis == (0, 1) else np.moveaxis(plain, axis, -1)
        keep = gaps if axis == (0, 1) else np.moveaxis(gaps, axis, -1)
        lane, off = lanes[idx], keep[idx]
        if off.all():
            if name in ("sum", "prod"):
                assert got[idx] == {"sum": 0, "prod": 1}[name]
            else:
                assert got[idx] is nd.NA
        else:
            np.testing.assert_allclose(got[idx], getattr(np, name)(lane[~off]))


def test_reduction_axis_keeps_nonfinite_results():
    """What `numpy.ma` gets wrong: it masks the mean of `[inf, 1]`."""
    a = np.array([[np.inf, 1.0]], dtype=nd.Nullable(np.float64))
    assert nd.mean(a, axis=1)[0] == np.inf
    assert nd.max(a, axis=1, skipna=True)[0] == np.inf


def test_reduction_axis_over_empty_lanes():
    e = np.zeros((0, 3), dtype=nd.Nullable(np.float64))
    assert all(x is nd.NA for x in nd.mean(e, axis=0))
    assert nd.sum(e, axis=0).tolist() == [0.0, 0.0, 0.0]
    assert nd.mean(e, axis=1).shape == (0,)


def test_reduction_axis_on_ints_follows_numpys_result_dtype():
    i = np.array([[1, 2], [3, 0]], dtype=nd.Nullable(np.int8))
    i[1, 1] = nd.NA
    got = nd.mean(i, axis=1)
    assert got.dtype == nd.Nullable(np.float64)
    assert got[0] == 1.5 and got[1] is nd.NA
    assert nd.sum(i, axis=0, skipna=True).tolist() == [4, 2]
    assert nd.std(i, axis=0, skipna=True, ddof=0)[1] == 0.0


@pytest.mark.parametrize("fn", ["quantile", "percentile"])
def test_quantile_axis_puts_q_first_like_numpy(fn):
    plain = np.arange(24.0).reshape(2, 3, 4)
    a = plain.astype(nd.Nullable(np.float64))
    a[0, 1, 2] = nd.NA
    q = [10, 50, 90] if fn == "percentile" else [0.1, 0.5, 0.9]
    for axis in (0, 2, (0, 2)):
        got = getattr(nd, fn)(a, q, axis=axis, keepdims=True)
        want = getattr(np, fn)(plain, q, axis=axis, keepdims=True)
        assert got.shape == want.shape
    got = getattr(nd, fn)(a, q, axis=2)
    assert all(got[k, 0, 1] is nd.NA for k in range(3))
    np.testing.assert_allclose(nd.filled(got, 0.0)[:, 1],
                               getattr(np, fn)(plain, q, axis=2)[:, 1])
    lane = getattr(nd, fn)(a, q, axis=2, skipna=True)[:, 0, 1]
    np.testing.assert_allclose(lane.astype(np.float64),
                               getattr(np, fn)(np.array([4.0, 5.0, 7.0]), q))


def test_all_any_skipna_and_axis():
    db = nd.Nullable(np.bool_)
    a = np.array([[True, True], [False, True], [True, False]], dtype=db)
    a[0, 1] = a[2, 1] = nd.NA
    got = nd.all(a, axis=1)
    assert got[0] is nd.NA and got[1] == False and got[2] is nd.NA   # noqa: E712
    got = nd.any(a, axis=1)
    assert got[0] == True and got[1] == True and got[2] == True      # noqa: E712
    assert nd.all(a, axis=1, skipna=True).tolist() == [True, False, True]
    assert nd.any(a, axis=0, skipna=True).tolist() == [True, True]
    assert nd.all(a, axis=1, keepdims=True).shape == (3, 1)
    assert nd.all(a, skipna=True) == False                           # noqa: E712
    assert nd.all(a[:1, :]) is nd.NA
    assert nd.all(a[:1, :], skipna=True) == True                     # noqa: E712


def test_cumsum_cumprod_propagate_or_skip():
    c = np.array([1, 0, 2, 3], dtype=nd.Nullable(np.int8))
    c[1] = nd.NA
    got = nd.cumsum(c)
    assert got[0] == 1 and all(got[k] is nd.NA for k in (1, 2, 3))
    assert got.dtype == nd.Nullable(np.cumsum(np.int8([1])).dtype)

    got = nd.cumsum(c, skipna=True)                 # pandas: NA stays at the gap
    assert got[1] is nd.NA and [got[k] for k in (0, 2, 3)] == [1, 3, 6]
    got = nd.cumprod(c, skipna=True)
    assert got[1] is nd.NA and [got[k] for k in (0, 2, 3)] == [1, 2, 6]

    m = np.array([[1.0, 2.0], [0.0, 4.0]], dtype=nd.Nullable(np.float64))
    m[1, 0] = nd.NA
    got = nd.cumsum(m, axis=1, skipna=True)
    assert got[0].tolist() == [1.0, 3.0] and got[1, 0] is nd.NA and got[1, 1] == 4.0
    assert nd.cumsum(m).shape == (4,)                # axis=None flattens

    with pytest.raises(AttributeError, match=r"nd.cumsum\(a, skipna=True\)"):
        nd.nancumsum


def test_reductions_over_several_axes():
    """The loops did not say they were reorderable, so numpy refused even
    `a.sum()` on a 2-d array.  They now follow numpy's own rule: an identity,
    or `maximum`-style "reorderable none", means reorderable."""
    plain = np.arange(24.0).reshape(2, 3, 4)
    a = plain.astype(nd.Nullable(np.float64))
    for u in (np.add, np.multiply, np.maximum, np.minimum):
        np.testing.assert_array_equal(u.reduce(a, axis=(0, 2)).astype(np.float64),
                                      u.reduce(plain, axis=(0, 2)))
        assert u.reduce(a, axis=None) == u.reduce(plain, axis=None)
    assert a.sum() == plain.sum() and a.mean() == plain.mean()

    a[1, 2, 3] = nd.NA
    assert a.sum() is nd.NA and a.max() is nd.NA
    got = a.sum(axis=(0, 1))
    assert got[3] is nd.NA and got[:3].tolist() == plain.sum(axis=(0, 1))[:3].tolist()

    with pytest.raises(ValueError, match="not reorderable"):
        np.subtract.reduce(a, axis=(0, 1))       # numpy refuses this one too

    b = np.array([[True, False], [True, True]], dtype=nd.Nullable(np.bool_))
    b[0, 1] = nd.NA
    assert np.logical_and.reduce(b, axis=None) is nd.NA
    assert np.logical_or.reduce(b, axis=None) == True       # noqa: E712
    i = np.arange(6).reshape(2, 3).astype(nd.Nullable(np.int32))
    assert np.bitwise_or.reduce(i, axis=None) == 7


@pytest.mark.xfail(strict=True, reason="numpy picks a float mean only for its "
                   "own integer types; see NUMPY-PATCHES.md, section 2")
def test_numpy_mean_of_nullable_int_is_not_truncated():
    i = np.array([1, 2], dtype=nd.Nullable(np.int64))
    assert nd.mean(i) == 1.5          # the nd version is right
    assert np.mean(i) == 1.5          # numpy's gives 1


@pytest.mark.parametrize("t", [np.float64, np.float32, np.int64, np.int32,
                               np.int16, np.uint8])
def test_block_scan_edges(t):
    """The fixup scans in blocks of 2048 and skips blocks with no gap; a gap on
    either side of a block edge, in the last short block, or in only one
    operand must still come out as NA, and nothing else may."""
    n = 2048 * 3 + 5
    dt = nd.Nullable(t)
    a = np.ones(n, dtype=dt)
    b = np.ones(n, dtype=dt)
    gaps_a = [0, 2047, 2048, 4095, n - 1]
    gaps_b = [100, 4096, n - 3]
    for i in gaps_a:
        a[i] = nd.NA
    for i in gaps_b:
        b[i] = nd.NA
    got = nd.isna(a + b)
    want = np.zeros(n, dtype=bool)
    want[gaps_a + gaps_b] = True
    np.testing.assert_array_equal(got, want)
    assert (nd.filled(a + b, 0)[~want] == 2).all()


def test_block_scan_still_catches_a_computed_na_after_a_clean_block():
    """A block without gaps still checks whether the result landed on the
    reserved value."""
    n = 2048 * 2 + 1
    a = np.zeros(n, dtype=nd.Nullable(np.int32))
    a[-1] = np.iinfo(np.int32).max
    with pytest.raises(ValueError, match="reserved value"):
        a + np.ones(n, dtype=nd.Nullable(np.int32))


_SPECIALS = [0.0, -0.0, 1.0, -1.0, 2.0, 0.5, np.inf, -np.inf, np.nan]
# ops whose answer depends on the missing value although IEEE gives a number
_DEPENDS_ON_GAP = {"fmax", "fmin", "copysign"}


@pytest.mark.parametrize("t", [np.float64, np.float32, np.float16,
                               np.complex128, np.complex64])
@pytest.mark.parametrize("n", [1, 5000])
def test_every_binop_keeps_a_gap(t, n):
    """A gap is a value nobody knows.  Where the result would differ for some
    value of it, the result is NA; where IEEE 754 already says it cannot --
    `1 ** x == x ** 0 == 1`, `hypot(inf, x) == inf`, `heaviside(y != 0, x)` --
    the number stands, as in R and pandas.  So the expected answer is IEEE's
    with a NaN in place of the gap, NA wherever that is NaN, and NA always for
    `fmax`, `fmin` and `copysign`, which do look at the missing value.
    Complex operands conservatively always give NA.

    Every element involves a gap, so any floating-point warning is a spurious
    one (`binop_signals_on_gap`); complex division, power and ordering used to
    raise "invalid value" this way."""
    dt = nd.Nullable(t)
    is_complex = np.dtype(t).kind == "c"
    gap = np.empty(n, dtype=dt)
    gap[...] = nd.NA
    nan = np.full(n, np.nan, dtype=t)
    wrong = []
    for name in _c.binop_names:
        f = getattr(np, name)
        for v in _SPECIALS:
            plain = np.full(n, v, dtype=t)
            other = plain.astype(dt)
            for x, y, px, py, side in ((other, gap, plain, nan, "right"),
                                       (gap, other, nan, plain, "left")):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)
                        r = f(x, y)
                except RuntimeWarning as w:
                    wrong.append((name, v, side, str(w)))
                    continue
                except (TypeError, ValueError):
                    continue            # op not defined for this dtype
                if not nd.is_nullable(r.dtype):
                    continue
                with np.errstate(all="ignore"):
                    ref = np.asarray(f(px, py))
                if ref.dtype.kind in "fc" and not is_complex \
                        and name not in _DEPENDS_ON_GAP:
                    expect_na = np.isnan(ref)
                else:
                    expect_na = np.ones(n, dtype=bool)
                got_na = nd.isna(r)
                if not (got_na == expect_na).all():
                    wrong.append((name, v, side, "NA where", got_na[0], "expected", expect_na[0]))
                    continue
                if (~expect_na).any():
                    kept = nd.to_numpy(r[~expect_na])
                    if not (kept == ref[~expect_na]).all():
                        wrong.append((name, v, side, "value", kept[0], ref[~expect_na][0]))
    assert not wrong


@pytest.mark.parametrize("t", [np.int64, np.int32, np.int8, np.uint8, np.uint64])
@pytest.mark.parametrize("n", [1, 5000])
def test_integer_power_over_a_gap(t, n):
    """An integer gap is INT_MIN (or UINT_MAX); integer `power` used to read it
    as a negative exponent and raise.  Now it never sees the gap, and the two
    determined cases give 1 as for floats.  A real negative exponent still
    raises, as for a plain array."""
    dt = nd.Nullable(t)
    base = np.array([1, 2, 0, 0, 3], dtype=t).astype(dt)
    expo = np.array([0, 0, 0, 2, 0], dtype=t).astype(dt)
    base[2] = base[3] = nd.NA
    expo[0] = expo[1] = nd.NA
    base, expo = np.tile(base, n), np.tile(expo, n)
    got = np.power(base, expo)
    assert nd.isna(got).tolist() == [False, True, False, True, False] * n
    assert got[0] == 1 and got[2] == 1 and got[4] == 1

    if np.dtype(t).kind == "i":
        with pytest.raises(ValueError, match="negative integer powers"):
            np.power(np.array([2], dtype=t).astype(dt),
                     np.array([-1], dtype=t).astype(dt))
