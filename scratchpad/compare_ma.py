"""Measure `numpy.ma` and `NullableDType` side by side; print the tables of
VS-NUMPY-MA.md.

Every cell is a real call and its real result, so the document cannot drift
away from the code.  Run it again after any change.
"""
import io
import pickle
import timeit
import warnings

import numpy as np
import nulldtype as nd

REC = np.dtype([("a", "i4"), ("b", "f8")])


def ma(values=(3.0, 1.0, 12345.0, 5.0), mask=(False, False, True, False)):
    return np.ma.masked_array(list(values), mask=list(mask))


def nu(values=(3.0, 1.0, 12345.0, 5.0), mask=(False, False, True, False), t=np.float64):
    a = np.zeros(len(values), dtype=nd.Nullable(t))
    for i, (v, m) in enumerate(zip(values, mask)):
        a[i] = nd.NA if m else v
    return a


def show(fn):
    """Run it, and report the result or the exception, warnings included."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            r = fn()
        except Exception as e:
            return f"raise {type(e).__name__}"
    s = " ".join(repr(r).replace("\n", " ").split())
    if len(s) > 30:
        s = s[:27] + "..."
    warned = [w for w in caught if issubclass(w.category, (RuntimeWarning, UserWarning))]
    return s + (f" +{warned[0].category.__name__}" if warned else "")


COVERED = []


def table(title, rows):
    COVERED.extend(r[0] for r in rows)
    print(f"\n### {title}\n")
    print("| Operation | `np.ma` call → result | `nd` call → result | Note |")
    print("|---|---|---|---|")
    for label, ma_call, ma_fn, nd_call, nd_fn, note in rows:
        left = f"`{ma_call}` → `{show(ma_fn)}`" if ma_fn else f"`{ma_call}`"
        right = f"`{nd_call}` → `{show(nd_fn)}`" if nd_fn else (f"`{nd_call}`" if nd_call else "—")
        print(f"| `{label}` | {left} | {right} | {note} |")


# the arrays every call below is written against
A_MA, A_ND = ma(), nu()
B_MA, B_ND = ma([1.0, 2.0, 2.0, 2.0], [False] * 4), nu([1.0, 2.0, 2.0, 2.0], [False] * 4)
I_MA, I_ND = ma([1, 2, 6, 4], [False, False, True, False]), nu([1, 2, 6, 4], [False, False, True, False], np.int64)
T_MA, T_ND = ma([True, False], [True, False]), nu([True, False], [True, False], np.bool_)
M_MA = np.ma.masked_array([[1.0, 2.0], [3.0, 4.0]], mask=[[False, True], [False, False]])
M_ND = nu([1.0, 2.0, 3.0, 4.0], [False, True, False, False]).reshape(2, 2)

UNIT_MA = ma([0.3, 0.1, 0.5, 0.9], [False, False, True, False])
UNIT_ND = nu([0.3, 0.1, 0.5, 0.9], [False, False, True, False])
UP_MA = ma([3.0, 1.5, 2.0, 5.0], [False, False, True, False])
UP_ND = nu([3.0, 1.5, 2.0, 5.0], [False, False, True, False])
UNIT = {"arcsin", "arccos", "arctanh"}
ABOVE_ONE = {"arccosh"}

NOTE = {}
UNARY = ["abs", "absolute", "angle", "arccos", "arccosh", "arcsin", "arcsinh",
         "arctan", "arctanh", "around", "ceil", "conjugate", "cos", "cosh",
         "exp", "fabs", "floor", "log", "log2", "log10", "logical_not",
         "negative", "round", "sin", "sinh", "sqrt", "tan", "tanh"]
rows = []
for n in UNARY:
    ma_in, nd_in = (UNIT_MA, UNIT_ND) if n in UNIT else (
        (UP_MA, UP_ND) if n in ABOVE_ONE else (A_MA, A_ND))
    note = NOTE.get(n, "domain: \\|x\\| <= 1" if n in UNIT else ("domain: x >= 1" if n in ABOVE_ONE else ""))
    rows.append((n, f"np.ma.{n}(a)[2]", (lambda n=n, x=ma_in: getattr(np.ma, n)(x)[2]),
                 f"nd.{n}(a)[2]", (lambda n=n, x=nd_in: getattr(nd, n)(x)[2]), note))
table("Unary ufuncs", rows)

BINARY = ["add", "subtract", "multiply", "divide", "true_divide", "floor_divide",
          "mod", "remainder", "power", "fmod", "hypot", "arctan2", "maximum",
          "minimum", "equal", "not_equal", "less", "less_equal", "greater",
          "greater_equal", "logical_and", "logical_or", "logical_xor"]
INT_BINARY = ["bitwise_and", "bitwise_or", "bitwise_xor", "left_shift", "right_shift"]
rows = []
for n in BINARY:
    rows.append((n, f"np.ma.{n}(a, b)[2]", (lambda n=n: getattr(np.ma, n)(A_MA, B_MA)[2]),
                 f"nd.{n}(a, b)[2]", (lambda n=n: getattr(nd, n)(A_ND, B_ND)[2]), ""))
for n in INT_BINARY:
    note = "Kleene on bool, propagates on int" if n.startswith("bitwise") else "integers only"
    rows.append((n, f"np.ma.{n}(i, i)[2]", (lambda n=n: getattr(np.ma, n)(I_MA, I_MA)[2]),
                 f"nd.{n}(i, i)[2]", (lambda n=n: getattr(nd, n)(I_ND, I_ND)[2]), note))
table("Binary ufuncs", rows)


# ---------------------------------------------------------------- reductions
def rt(label, call, ma_fn, nd_call, nd_fn, note=""):
    return (label, call, ma_fn, nd_call, nd_fn, note)

table("Reductions and statistics", [
 rt("sum", "a.sum()", lambda: A_MA.sum(), "a.sum()", lambda: A_ND.sum(),
    "`np.ma` skips a gap, this propagates: `nd.sum(a, skipna=True)` to skip"),
 rt("prod", "np.ma.prod(a)", lambda: np.ma.prod(A_MA), "nd.prod(a)", lambda: nd.prod(A_ND), ""),
 rt("sum, skipping", "a.sum()", lambda: A_MA.sum(), "nd.sum(a, skipna=True)", lambda: nd.sum(A_ND, skipna=True),
    "every `nd` reduction takes `skipna=`, default False"),
 rt("mean", "a.mean()", lambda: A_MA.mean(), "a.mean()", lambda: A_ND.mean(), ""),
 rt("mean along an axis", "np.ma.mean(m, axis=0)", lambda: np.ma.mean(M_MA, axis=0),
    "nd.mean(m, axis=0)", lambda: nd.mean(M_ND, axis=0), "NA on each lane with a gap"),
 rt("mean along an axis, skipping", "np.ma.mean(m, axis=0)", lambda: np.ma.mean(M_MA, axis=0),
    "nd.mean(m, axis=0, skipna=True)", lambda: nd.mean(M_ND, axis=0, skipna=True), ""),
 rt("std", "a.std()", lambda: A_MA.std(), "a.std()", lambda: A_ND.std(), ""),
 rt("var", "a.var()", lambda: A_MA.var(), "a.var()", lambda: A_ND.var(), ""),
 rt("min / amin / minmax", "np.ma.min(a)", lambda: np.ma.min(A_MA), "nd.min(a)", lambda: nd.min(A_ND), ""),
 rt("max / amax", "np.ma.max(a)", lambda: np.ma.max(A_MA), "nd.max(a)", lambda: nd.max(A_ND), ""),
 rt("ptp", "np.ma.ptp(a)", lambda: np.ma.ptp(A_MA), "nd.ptp(a)", lambda: nd.ptp(A_ND), ""),
 rt("median", "np.ma.median(a)", lambda: np.ma.median(A_MA), "nd.median(a)", lambda: nd.median(A_ND),
    "plain `np.median(a)` answers **4.0** here, silently: NA sorts last"),
 rt("average", "np.ma.average(a)", lambda: np.ma.average(A_MA), "nd.average(a)", lambda: nd.average(A_ND), ""),
 rt("all", "np.ma.all(t)", lambda: np.ma.all(T_MA), "nd.all(t)", lambda: nd.all(T_ND), "Kleene"),
 rt("any", "np.ma.any(t)", lambda: np.ma.any(T_MA), "nd.any(t)", lambda: nd.any(T_ND), "Kleene"),
 rt("all, skipping", "np.ma.all(t)", lambda: np.ma.all(T_MA), "nd.all(t, skipna=True)", lambda: nd.all(T_ND, skipna=True),
    "what `np.ma` does, asked for by name"),
 rt("count", "a.count()", lambda: A_MA.count(), "nd.count(a)", lambda: nd.count(A_ND), ""),
 rt("count_masked", "np.ma.count_masked(a)", lambda: np.ma.count_masked(A_MA),
    "nd.isna(a).sum()", lambda: int(nd.isna(A_ND).sum()), ""),
 rt("argmax", "a.argmax()", lambda: A_MA.argmax(), "a.argmax()", lambda: A_ND.argmax(),
    "the first gap, numpy's NaN rule: `a[a.argmax()]` is NA exactly when `a.max()` is"),
 rt("argmax, skipping", "a.argmax()", lambda: A_MA.argmax(), "nd.argmax(a, skipna=True)", lambda: nd.argmax(A_ND, skipna=True),
    "position in the original array"),
 rt("argmin", "a.argmin()", lambda: A_MA.argmin(), "a.argmin()", lambda: A_ND.argmin(), "as `argmax`"),
 rt("cumsum", "np.ma.cumsum(a)[3]", lambda: np.ma.cumsum(A_MA)[3], "nd.cumsum(a)[3]", lambda: nd.cumsum(A_ND)[3],
    "`np.ma` treats a gap as 0 and carries on; here a gap poisons the rest"),
 rt("cumsum, skipping", "np.ma.cumsum(a)[3]", lambda: np.ma.cumsum(A_MA)[3], "nd.cumsum(a, skipna=True)[3]",
    lambda: nd.cumsum(A_ND, skipna=True)[3], "carries past the gap; the gap itself stays NA (`np.ma`: masked)"),
 rt("cumprod", "np.ma.cumprod(a)[3]", lambda: np.ma.cumprod(A_MA)[3], "nd.cumprod(a)[3]", lambda: nd.cumprod(A_ND)[3], ""),
 rt("anom / anomalies", "np.ma.anom(a)[0]", lambda: np.ma.anom(A_MA)[0], "(a - nd.mean(a))[0]", lambda: (A_ND - A_ND.mean())[0],
    "no `nd.anom`; the mean propagates so every element is NA"),
 rt("corrcoef", "np.ma.corrcoef(a, b)[0, 1]", lambda: np.ma.corrcoef(A_MA, B_MA)[0, 1],
    "nd.corrcoef(a, b)", lambda: nd.corrcoef(A_ND, B_ND), "not supported yet"),
 rt("cov", "np.ma.cov(a, b)[0, 1]", lambda: np.ma.cov(A_MA, B_MA)[0, 1], "nd.cov(a, b)", lambda: nd.cov(A_ND, B_ND),
    "not supported yet"),
 rt("trace", "np.ma.trace(m)", lambda: np.ma.trace(M_MA), "nd.trace(m)", lambda: nd.trace(M_ND), ""),
 rt("allclose", "np.ma.allclose(a, a)", lambda: np.ma.allclose(A_MA, A_MA), "nd.allclose(a, a)", lambda: nd.allclose(A_ND, A_ND),
    "NA when a gap leaves it undecided"),
 rt("allequal", "np.ma.allequal(a, a)", lambda: np.ma.allequal(A_MA, A_MA), "nd.array_equal(a, a)", lambda: nd.array_equal(A_ND, A_ND), ""),
])

S_MA = ma([1.0, 1.0, 2.0], [False, True, False])
S_ND = nu([1.0, 1.0, 2.0], [False, True, False])

table("Sorting, searching, sets, products", [
 rt("sort", "np.ma.sort(a)[-1]", lambda: np.ma.sort(A_MA)[-1], "nd.sort(a)[-1]", lambda: nd.sort(A_ND)[-1], "gaps last on both sides"),
 rt("argsort", "np.ma.argsort(a)[-1]", lambda: np.ma.argsort(A_MA)[-1], "nd.argsort(a)[-1]", lambda: nd.argsort(A_ND)[-1], ""),
 rt("unique", "len(np.ma.unique(s))", lambda: len(np.ma.unique(S_MA)), "len(nd.unique(s))", lambda: len(nd.unique(S_ND)),
    "plain `np.unique` refuses; `nd.unique` keeps NA once, last"),
 rt("isin", "np.ma.isin(i, [2, 4])[2]", lambda: np.ma.isin(I_MA, [2, 4])[2],
    "nd.isin(i, [2, 4])[2]", lambda: nd.isin(I_ND, [2, 4])[2], "`nd.isin` answers NA for a gap"),
 rt("isin, gap in the set", "np.ma.isin(b, i)[1]", lambda: np.ma.isin(B_MA, I_MA)[1],
    "nd.isin(b, i)[1]", lambda: nd.isin(B_ND, I_ND)[1],
    "is 2.0 in `[1, 2, NA, 4]`: yes on both sides"),
 rt("isin, gap in the set, no hit", "np.ma.isin([5.0], i)[0]", lambda: np.ma.isin(ma([5.0], [False]), I_MA)[0],
    "nd.isin([5.0], i)[0]", lambda: nd.isin(nu([5.0], [False]), I_ND)[0],
    "is 5 in `[1, 2, NA, 4]`: the gap might be 5, so NA -- SQL's `IN` agrees"),
 rt("intersect1d", "np.ma.intersect1d(i, i)[0]", lambda: np.ma.intersect1d(I_MA, I_MA)[0],
    "nd.intersect1d(i, i)", lambda: nd.intersect1d(I_ND, I_ND), "needs a bool cast of a gap"),
 rt("union1d", "len(np.ma.union1d(i, i))", lambda: len(np.ma.union1d(I_MA, I_MA)),
    "nd.union1d(i, i)", lambda: nd.union1d(I_ND, I_ND), ""),
 rt("setdiff1d", "len(np.ma.setdiff1d(i, i))", lambda: len(np.ma.setdiff1d(I_MA, I_MA)),
    "nd.setdiff1d(i, i)", lambda: nd.setdiff1d(I_ND, I_ND), ""),
 rt("setxor1d", "len(np.ma.setxor1d(i, i))", lambda: len(np.ma.setxor1d(I_MA, I_MA)),
    "nd.setxor1d(i, i)", lambda: nd.setxor1d(I_ND, I_ND), ""),
 rt("nonzero", "np.ma.nonzero(a)[0]", lambda: np.ma.nonzero(A_MA)[0], "nd.nonzero(a)", lambda: nd.nonzero(A_ND),
    "refusing is the point: a gap has no truth value"),
 rt("where", "np.ma.where(a > 2, a, 0)[2]", lambda: np.ma.where(A_MA > 2, A_MA, 0)[2],
    "nd.where(nd.notna(a), a, zero)[2]", lambda: nd.where(nd.notna(A_ND), A_ND, nu([0.0]*4, [False]*4))[2],
    "the condition must be a plain bool array, the other arm a Nullable one"),
 rt("choose", "np.ma.choose([0, 1, 0, 1], [a, b])[2]", lambda: np.ma.choose([0, 1, 0, 1], [A_MA, B_MA])[2],
    "nd.choose([0, 1, 0, 1], [a, b])[2]", lambda: nd.choose([0, 1, 0, 1], [A_ND, B_ND])[2], ""),
 rt("compress / compressed", "np.ma.compressed(a)", lambda: len(np.ma.compressed(A_MA)),
    "nd.dropna(a)", lambda: len(nd.dropna(A_ND)), "`compressed` drops the gaps"),
 rt("take", "np.ma.take(a, [0, 2])[1]", lambda: np.ma.take(A_MA, [0, 2])[1],
    "nd.take(a, [0, 2])[1]", lambda: nd.take(A_ND, [0, 2])[1], ""),
 rt("put", "np.ma.put(a.copy(), [0], [9.0])", lambda: np.ma.put(A_MA.copy(), [0], [9.0]),
    "nd.put(a.copy(), [0], [9.0])", lambda: nd.put(A_ND.copy(), [0], [9.0]), "in place, returns None"),
 rt("putmask", "np.ma.putmask(a.copy(), [1,0,0,0], 9.0)", lambda: np.ma.putmask(A_MA.copy(), [1,0,0,0], 9.0),
    "nd.putmask(a.copy(), [1,0,0,0], 9.0)", lambda: nd.putmask(A_ND.copy(), [1,0,0,0], 9.0), ""),
 rt("diff / ediff1d", "np.ma.diff(a)[1]", lambda: np.ma.diff(A_MA)[1], "nd.diff(a)[1]", lambda: nd.diff(A_ND)[1], ""),
 rt("dot", "np.ma.dot(a, b)", lambda: np.ma.dot(A_MA, B_MA), "nd.dot(a, b)", lambda: nd.dot(A_ND, B_ND),
    "`np.dot` itself cannot work: its legacy slot gets a NULL array"),
 rt("inner", "np.ma.inner(a, b)", lambda: np.ma.inner(A_MA, B_MA), "nd.inner(a, b)", lambda: nd.inner(A_ND, B_ND), "same slot"),
 rt("outer", "np.ma.outer(a, b)[2, 0]", lambda: np.ma.outer(A_MA, B_MA)[2, 0],
    "nd.outer(a, b)[2, 0]", lambda: nd.outer(A_ND, B_ND)[2, 0], ""),
 rt("matmul", "a @ b", lambda: A_MA @ B_MA, "a @ b", lambda: A_ND @ B_ND, "`@` is a gufunc, so this one works"),
 rt("convolve", "np.ma.convolve(a, b)[0]", lambda: np.ma.convolve(A_MA, B_MA)[0],
    "nd.convolve(a, b)", lambda: nd.convolve(A_ND, B_ND), "goes through `dot`"),
 rt("correlate", "np.ma.correlate(a, b)[0]", lambda: np.ma.correlate(A_MA, B_MA)[0],
    "nd.correlate(a, b)", lambda: nd.correlate(A_ND, B_ND), "goes through `dot`"),
 rt("polyfit", "np.ma.polyfit(b, b, 1)[0]", lambda: np.ma.polyfit(B_MA, B_MA, 1)[0],
    "nd.polyfit(b, b, 1)", lambda: nd.polyfit(B_ND, B_ND, 1), "linear algebra wants plain floats"),
 rt("vander", "np.ma.vander(b)[0, 0]", lambda: np.ma.vander(B_MA)[0, 0], "nd.vander(b)[0, 0]", lambda: nd.vander(B_ND)[0, 0], ""),
 rt("clip", "np.ma.clip(a, 2, 4)[2]", lambda: np.ma.clip(A_MA, 2.0, 4.0)[2],
    "np.clip(a, lo, hi)[2]", lambda: np.clip(A_ND, nu([2.0]*4, [False]*4), nu([4.0]*4, [False]*4))[2],
    "bounds must be Nullable too"),
 rt("apply_along_axis", "np.ma.apply_along_axis(np.sum, 0, a)", lambda: np.ma.apply_along_axis(np.sum, 0, A_MA),
    "nd.apply_along_axis(np.sum, 0, a)", lambda: nd.apply_along_axis(np.sum, 0, A_ND), ""),
 rt("apply_over_axes", "np.ma.apply_over_axes(np.sum, m, [0])[0,0]", lambda: np.ma.apply_over_axes(np.sum, M_MA, [0])[0, 0],
    "nd.apply_over_axes(np.sum, m, [0])[0,0]", lambda: nd.apply_over_axes(np.sum, M_ND, [0])[0, 0], ""),
 rt("unwrap", "np.ma.unwrap(a)[2]", lambda: np.ma.unwrap(A_MA)[2], "nd.unwrap(a)", lambda: nd.unwrap(A_ND),
    "phase unwrapping goes through `np.diff` plus in-place masking"),
 rt("ndenumerate", "list(np.ma.ndenumerate(a))[2]", lambda: list(np.ma.ndenumerate(A_MA))[2],
    "list(nd.ndenumerate(a))[2]", lambda: list(nd.ndenumerate(A_ND))[2], "`np.ma` skips gaps, numpy's does not"),
])

table("Creation, shape, joining", [
 rt("array", "np.ma.array([1.0, 2.0])[0]", lambda: np.ma.array([1.0, 2.0])[0],
    "np.array([1.0, 2.0], nd.Nullable('f8'))[0]", lambda: np.array([1.0, 2.0], nd.Nullable(np.float64))[0], ""),
 rt("masked_array", "np.ma.masked_array(v, mask=m)[2]", lambda: ma()[2],
    "a[2] = nd.NA", lambda: nu()[2], "the mask is an argument there, a value here"),
 rt("asarray / asanyarray", "np.ma.asarray(a)[2]", lambda: np.ma.asarray(A_MA)[2], "nd.asarray(a)[2]", lambda: nd.asarray(A_ND)[2], ""),
 rt("copy", "np.ma.copy(a)[2]", lambda: np.ma.copy(A_MA)[2], "nd.copy(a)[2]", lambda: nd.copy(A_ND)[2], ""),
 rt("zeros / ones / empty", "np.ma.zeros(2)[0]", lambda: np.ma.zeros(2)[0],
    "np.zeros(2, nd.Nullable('f8'))[0]", lambda: np.zeros(2, nd.Nullable(np.float64))[0], "a dtype argument, not a function"),
 rt("zeros_like / ones_like", "np.ma.zeros_like(a)[2]", lambda: np.ma.zeros_like(A_MA)[2],
    "nd.zeros_like(a)[2]", lambda: nd.zeros_like(A_ND)[2], "`np.ma` keeps the mask, here a zero is a zero"),
 rt("empty_like", "np.ma.empty_like(a)[0]", lambda: np.ma.empty_like(A_MA)[0], "nd.empty_like(a)[0]", lambda: nd.empty_like(A_ND)[0], ""),
 rt("masked_all", "np.ma.masked_all(2)[0]", lambda: np.ma.masked_all(2)[0],
    "x[...] = nd.NA", lambda: (lambda x: (x.__setitem__(Ellipsis, nd.NA), x[0])[1])(np.zeros(2, nd.Nullable(np.float64))), ""),
 rt("masked_all_like", "np.ma.masked_all_like(a)[0]", lambda: np.ma.masked_all_like(A_MA)[0],
    "y = nd.empty_like(a); y[...] = nd.NA", lambda: (lambda y: (y.__setitem__(Ellipsis, nd.NA), y[0])[1])(nd.empty_like(A_ND)), ""),
 rt("arange", "np.ma.arange(3)[1]", lambda: np.ma.arange(3)[1],
    "np.arange(3, dtype=nd.Nullable('i8'))[1]", lambda: np.arange(3, dtype=nd.Nullable(np.int64))[1], "needed the `fill` slot"),
 rt("identity", "np.ma.identity(2)[0, 0]", lambda: np.ma.identity(2)[0, 0],
    "np.eye(2, dtype=nd.Nullable('f8'))[0,0]", lambda: np.eye(2, dtype=nd.Nullable(np.float64))[0, 0], ""),
 rt("indices / fromfunction", "np.ma.indices((2,))[0][1]", lambda: np.ma.indices((2,))[0][1],
    "nd.indices((2,))[0][1]", lambda: nd.indices((2,))[0][1], "plain integer arrays, no gaps involved"),
 rt("frombuffer", "np.ma.frombuffer(bs, 'f8')[0]", lambda: np.ma.frombuffer(np.zeros(1).tobytes(), dtype=np.float64)[0],
    "np.frombuffer(bs, nd.Nullable('f8'))[0]", lambda: np.frombuffer(np.zeros(1).tobytes(), dtype=nd.Nullable(np.float64))[0], ""),
 rt("reshape", "np.ma.reshape(a, (2, 2))[1, 0]", lambda: np.ma.reshape(A_MA, (2, 2))[1, 0],
    "nd.reshape(a, (2, 2))[1, 0]", lambda: nd.reshape(A_ND, (2, 2))[1, 0], ""),
 rt("resize", "np.ma.resize(a, 6)[2]", lambda: np.ma.resize(A_MA, 6)[2], "nd.resize(a, 6)[2]", lambda: nd.resize(A_ND, 6)[2], ""),
 rt("ravel / flatten", "np.ma.ravel(m)[1]", lambda: np.ma.ravel(M_MA)[1], "nd.ravel(m)[1]", lambda: nd.ravel(M_ND)[1], ""),
 rt("squeeze", "np.ma.squeeze(a)[2]", lambda: np.ma.squeeze(A_MA)[2], "nd.squeeze(a)[2]", lambda: nd.squeeze(A_ND)[2], ""),
 rt("transpose", "np.ma.transpose(m)[1, 0]", lambda: np.ma.transpose(M_MA)[1, 0], "nd.transpose(m)[1, 0]", lambda: nd.transpose(M_ND)[1, 0], ""),
 rt("swapaxes", "np.ma.swapaxes(m, 0, 1)[1, 0]", lambda: np.ma.swapaxes(M_MA, 0, 1)[1, 0],
    "nd.swapaxes(m, 0, 1)[1, 0]", lambda: nd.swapaxes(M_ND, 0, 1)[1, 0], ""),
 rt("expand_dims", "np.ma.expand_dims(a, 0)[0, 2]", lambda: np.ma.expand_dims(A_MA, 0)[0, 2],
    "nd.expand_dims(a, 0)[0, 2]", lambda: nd.expand_dims(A_ND, 0)[0, 2], ""),
 rt("atleast_1d / atleast_2d / atleast_3d", "np.ma.atleast_2d(a)[0, 2]", lambda: np.ma.atleast_2d(A_MA)[0, 2],
    "nd.atleast_2d(a)[0, 2]", lambda: nd.atleast_2d(A_ND)[0, 2], ""),
 rt("concatenate", "np.ma.concatenate([a, a])[2]", lambda: np.ma.concatenate([A_MA, A_MA])[2],
    "nd.concatenate([a, a])[2]", lambda: nd.concatenate([A_ND, A_ND])[2],
    "plain `np.concatenate` drops the mask; here there is nothing to drop"),
 rt("stack / hstack / vstack", "np.ma.stack([a, a])[0, 2]", lambda: np.ma.stack([A_MA, A_MA])[0, 2],
    "nd.stack([a, a])[0, 2]", lambda: nd.stack([A_ND, A_ND])[0, 2], ""),
 rt("dstack / column_stack", "np.ma.column_stack([a, a])[2, 0]", lambda: np.ma.column_stack([A_MA, A_MA])[2, 0],
    "nd.column_stack([a, a])[2, 0]", lambda: nd.column_stack([A_ND, A_ND])[2, 0], ""),
 rt("append", "np.ma.append(a, a)[2]", lambda: np.ma.append(A_MA, A_MA)[2], "nd.append(a, a)[2]", lambda: nd.append(A_ND, A_ND)[2], ""),
 rt("repeat", "np.ma.repeat(a, 2)[4]", lambda: np.ma.repeat(A_MA, 2)[4], "nd.repeat(a, 2)[4]", lambda: nd.repeat(A_ND, 2)[4], ""),
 rt("hsplit", "np.ma.hsplit(a, 2)[1][0]", lambda: np.ma.hsplit(A_MA, 2)[1][0], "nd.hsplit(a, 2)[1][0]", lambda: nd.hsplit(A_ND, 2)[1][0], ""),
 rt("diag / diagflat / diagonal", "np.ma.diagonal(m)[1]", lambda: np.ma.diagonal(M_MA)[1], "nd.diagonal(m)[1]", lambda: nd.diagonal(M_ND)[1], ""),
 rt("shape / size / ndim", "np.ma.shape(a)", lambda: np.ma.shape(A_MA), "nd.shape(a)", lambda: nd.shape(A_ND), ""),
 rt("mr_", "np.ma.mr_[a, a][2]", lambda: np.ma.mr_[A_MA, A_MA][2], "np.r_[a, a][2]", lambda: np.r_[A_ND, A_ND][2], ""),
])

def gap_at(idx=2):
    """A fresh Nullable array with a gap punched at `idx` by a boolean mask."""
    x = nu([3.0, 1.0, 12345.0, 5.0], [False] * 4)
    x[np.asarray(idx)] = nd.NA
    return x


table("The mask-only API", [
 rt("masked / masked_singleton", "a[2] is np.ma.masked", lambda: A_MA[2] is np.ma.masked,
    "a[2] is nd.NA", lambda: A_ND[2] is nd.NA, "one is a sentinel object, the other a value of the dtype"),
 rt("nomask", "np.ma.nomask", lambda: np.ma.nomask, "", None, "no mask exists, so nothing to be empty"),
 rt("getmask / getmaskarray", "np.ma.getmaskarray(a)[2]", lambda: np.ma.getmaskarray(A_MA)[2],
    "nd.isna(a)[2]", lambda: nd.isna(A_ND)[2], ""),
 rt("getdata", "np.ma.getdata(a)[2]", lambda: np.ma.getdata(A_MA)[2],
    "nd.to_numpy(a, na_value=0.0)[2]", lambda: nd.to_numpy(A_ND, na_value=0.0)[2],
    "**the leak**: `getdata` hands back the value under the mask"),
 rt("filled", "np.ma.filled(a, 0.0)[2]", lambda: np.ma.filled(A_MA, 0.0)[2],
    "nd.filled(a, 0.0)[2]", lambda: nd.filled(A_ND, 0.0)[2], ""),
 rt("compressed", "len(np.ma.compressed(a))", lambda: len(np.ma.compressed(A_MA)),
    "len(nd.dropna(a))", lambda: len(nd.dropna(A_ND)), ""),
 rt("count / count_masked", "a.count()", lambda: A_MA.count(), "nd.count(a)", lambda: nd.count(A_ND), ""),
 rt("masked_where", "np.ma.masked_where(a > 4, a)[3]", lambda: np.ma.masked_where(A_MA > 4, A_MA)[3],
    "x[nd.filled(x > 4, False)] = nd.NA", lambda: (lambda x: (x.__setitem__(nd.filled(x > 4, False), nd.NA), x[3])[1])(gap_at()),
    "a comparison answers `Nullable[bool]`, so fill it before indexing"),
 rt("masked_equal", "np.ma.masked_equal(a, 5.0)[3]", lambda: np.ma.masked_equal(A_MA, 5.0)[3],
    "x[nd.filled(x == 5.0, False)] = nd.NA", lambda: (lambda x: (x.__setitem__(nd.filled(x == 5.0, False), nd.NA), x[3])[1])(gap_at()), ""),
 rt("masked_greater / masked_greater_equal / masked_less / masked_less_equal / masked_not_equal / masked_inside / masked_outside / masked_values",
    "np.ma.masked_greater(a, 4.0)[3]", lambda: np.ma.masked_greater(A_MA, 4.0)[3],
    "same shape: compare, fill, assign NA", lambda: (lambda x: (x.__setitem__(nd.filled(x > 4.0, False), nd.NA), x[3])[1])(gap_at()),
    "one pattern covers all of them"),
 rt("masked_invalid / fix_invalid", "np.ma.masked_invalid(nanarr)[1]", lambda: np.ma.masked_invalid(np.array([1.0, np.nan]))[1],
    "y[np.isnan(y)] = nd.NA", lambda: (lambda y: (y.__setitem__(np.isnan(np.array([1.0, np.nan])), nd.NA), y[1])[1])(nu([1.0, 0.0], [False, False])),
    "a NaN is an ordinary value here until you say otherwise"),
 rt("masked_object", "np.ma.masked_object(objarr, None)", lambda: np.ma.masked_object(np.array([1, None], object), None)[1],
    "", None, "object arrays are refused outright"),
 rt("is_mask / is_masked", "np.ma.is_masked(a)", lambda: np.ma.is_masked(A_MA),
    "nd.isna(a).any()", lambda: bool(nd.isna(A_ND).any()), ""),
 rt("isMaskedArray", "np.ma.isMaskedArray(a)", lambda: np.ma.isMaskedArray(A_MA),
    "nd.is_nullable(a.dtype)", lambda: nd.is_nullable(A_ND.dtype), "one asks about the array, one about the dtype"),
 rt("make_mask / make_mask_none / make_mask_descr / mask_or / flatten_mask",
    "np.ma.make_mask([1, 0])", lambda: np.ma.make_mask([1, 0]), "", None,
    "these build and combine mask arrays; there is no mask object here"),
 rt("harden_mask / soften_mask", "a.harden_mask()", lambda: A_MA.copy().harden_mask() is None,
    "", None, "a gap is a value: assigning over it always works"),
 rt("default_fill_value / common_fill_value", "np.ma.default_fill_value(a)", lambda: np.ma.default_fill_value(A_MA),
    "", None, "no fill value is carried around; `nd.filled(a, x)` says it at the call"),
 rt("maximum_fill_value / minimum_fill_value", "np.ma.maximum_fill_value(a)", lambda: np.ma.maximum_fill_value(A_MA),
    "", None, "same: nothing to configure"),
 rt("set_fill_value / a.fill_value", "a.fill_value", lambda: A_MA.fill_value, "", None, "same"),
 rt("masked_print_option", "str(np.ma.masked_print_option)", lambda: str(np.ma.masked_print_option),
    "repr always prints NA", None, "not configurable, on purpose"),
 rt("notmasked_edges / flatnotmasked_edges", "np.ma.notmasked_edges(a)", lambda: np.ma.notmasked_edges(A_MA),
    "np.flatnonzero(nd.notna(a))[[0, -1]]", lambda: np.flatnonzero(nd.notna(A_ND))[[0, -1]], "plain numpy over `nd.notna`"),
 rt("notmasked_contiguous / flatnotmasked_contiguous", "len(np.ma.flatnotmasked_contiguous(a))",
    lambda: len(np.ma.flatnotmasked_contiguous(A_MA)), "same idea over nd.notna(a)", None, ""),
 rt("clump_masked / clump_unmasked", "len(np.ma.clump_masked(a))", lambda: len(np.ma.clump_masked(A_MA)),
    "same idea over nd.isna(a)", None, ""),
 rt("compress_rows / compress_cols / compress_rowcols / compress_nd", "np.ma.compress_rows(m).shape", lambda: np.ma.compress_rows(M_MA).shape,
    "m[~nd.isna(m).any(axis=1)].shape", lambda: M_ND[~nd.isna(M_ND).any(axis=1)].shape, ""),
 rt("mask_rows / mask_cols / mask_rowcols", "np.ma.mask_rows(m)[0, 0]", lambda: np.ma.mask_rows(M_MA)[0, 0],
    "m[nd.isna(m).any(axis=1)] = nd.NA", lambda: (lambda x: (x.__setitem__(nd.isna(x).any(axis=1), nd.NA), x[0, 0])[1])(M_ND.copy()), ""),
 rt("MaskedArray / MaskType / mvoid / bool_", "np.ma.MaskedArray", lambda: np.ma.MaskedArray.__name__,
    "nd.NullableDType", lambda: nd.NullableDType.__name__, "a subclass of ndarray vs a dtype"),
 rt("MAError / MaskError", "np.ma.MaskError", lambda: np.ma.MaskError.__name__, "", None,
    "errors come out as plain `ValueError` / `TypeError`"),
 rt("fromflex / flatten_structured_array / ids", "np.ma.fromflex", lambda: np.ma.fromflex.__name__, "", None,
    "internals of the mask representation"),
 rt("core / extras", "np.ma.core", lambda: np.ma.core.__name__, "", None, "submodules"),
])


# ------------------------------------------------------ did anything escape?
import re
import sys

seen = set()
for label in COVERED:
    seen.update(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", label))
# numpy 1 spellings numpy.ma still exports but numpy 2 removed or deprecated
# at the top level; they are left out of the tables on purpose
DROPPED = {"round_", "alltrue", "sometrue", "product", "row_stack", "in1d",
           "innerproduct", "outerproduct", "isarray", "isMA"}
names = sorted(n for n in np.ma.__all__ if n not in DROPPED)
missing = [n for n in names if n not in seen]
print(f"\n<!-- covered {len(names) - len(missing)}/{len(names)} of numpy.ma.__all__, "
      f"less {len(DROPPED)} numpy-1 spellings -->")
if missing:
    print("NOT COVERED:", ", ".join(missing), file=sys.stderr)
