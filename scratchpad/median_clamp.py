"""Does clamping by the last partitioned element reproduce numpy's NaN check
on plain arrays, and give NA on Nullable ones?"""
import warnings
import numpy as np
import nulldtype as nd
from numpy.lib._function_base_impl import partition, mean


def median_clamp(a, axis=None):
    a = np.asanyarray(a)
    sz = a.size if axis is None else a.shape[axis]
    kth = [sz // 2 - 1, sz // 2] if sz % 2 == 0 else [(sz - 1) // 2]
    kth.append(-1)
    part = partition(a, kth, axis=axis)
    if axis is None:
        axis = 0
    idx = [slice(None)] * part.ndim
    h = part.shape[axis] // 2
    idx[axis] = slice(h, h + 1) if part.shape[axis] % 2 else slice(h - 1, h + 1)
    rout = mean(part[tuple(idx)], axis=axis)
    if sz > 0:
        # the median never exceeds the maximum, and `minimum` propagates
        # whatever the dtype propagates: NaN, NaT, NA
        rout = np.minimum(rout, part.take(-1, axis=axis))
    return rout


rng = np.random.default_rng(1)
bad = 0
for trial in range(20000):
    dt = rng.choice([np.float64, np.float32, np.float16, np.int64, np.complex128, "M8[s]", "m8[s]"])
    shape = tuple(rng.integers(1, 5, size=rng.integers(1, 3)))
    x = rng.normal(size=shape) * 10
    if np.dtype(dt).kind in "fc":
        x = x.astype(dt)
        for special in (np.nan, np.inf, -np.inf, 0.0, -0.0):
            if rng.random() < 0.15:
                x.flat[rng.integers(x.size)] = special
    elif str(dt)[0] in "Mm":
        x = x.astype(np.int64).astype(dt)
        if rng.random() < 0.3:
            x.flat[rng.integers(x.size)] = np.datetime64("NaT", "s")
    else:
        x = x.astype(dt)
    axis = None if rng.random() < 0.4 else int(rng.integers(x.ndim))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            want = np.median(x, axis=axis)
        except Exception as e:
            want = type(e)
        try:
            got = median_clamp(x, axis=axis)
        except Exception as e:
            got = type(e)
    same = (want is got) or np.array_equal(np.asarray(want), np.asarray(got), equal_nan=True)
    if same and np.asarray(want).dtype.kind == "f":
        # the sign of zero too
        same = np.array_equal(np.signbit(np.asarray(want)), np.signbit(np.asarray(got)))
    if not same:
        bad += 1
        if bad <= 5:
            print("DIFF", dt, x.ravel()[:6], axis, want, got)
print("plain mismatches:", bad)

f8 = nd.Nullable(np.float64)
a = np.array([3.0, 1.0, 0.0, 5.0], f8)
a[2] = nd.NA
print("Nullable:", median_clamp(a), "| np.median:", np.median(a))
m = np.array([[3.0, 1.0, 5.0], [2.0, 0.0, 4.0]], f8)
m[1, 1] = nd.NA
print("Nullable axis=1:", median_clamp(m, axis=1))
i = np.array([3, 1, 0, 5], nd.Nullable(np.int64))
print("Nullable int, no gap:", median_clamp(i))
