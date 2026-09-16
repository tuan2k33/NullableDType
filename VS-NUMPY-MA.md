# `numpy.ma` and `NullableDType`, side by side

Every cell below is a real call and its real result, produced by
`scratchpad/compare_ma.py` against numpy `2.6.0.dev0`. Run it again after any
change and paste the output back; the document cannot drift away from the code.

It covers **every name in `numpy.ma.__all__`**, 218 of them. Ten more are left
out on purpose: numpy 1 spellings that numpy 2 removed or deprecated at the top
level (`round_`, `alltrue`, `sometrue`, `product`, `row_stack`, `in1d`,
`innerproduct`, `outerproduct`, `isarray`, `isMA`); use their numpy 2 names.

## The arrays these calls are written against

```python
a = [3.0, 1.0, 12345.0, 5.0]    third element missing   # 12345.0: does it leak?
b = [1.0, 2.0, 2.0, 2.0]        nothing missing
i = [1, 2, 6, 4]                int64, third missing
t = [True, False]               first missing
s = [1.0, 1.0, 2.0]             middle missing
m = [[1.0, 2.0], [3.0, 4.0]]    m[0, 1] missing
x, y                            fresh copies, for the rows that punch their own gap
```

In the two ufunc tables only the gap element is shown, because that is the only
place the two differ: every other element matches plain numpy exactly, which
the test suite checks op by op against `numpy` itself.

### Unary ufuncs

| Operation | `np.ma` call → result | `nd` call → result | Note |
|---|---|---|---|
| `abs` | `np.ma.abs(a)[2]` → `masked` | `nd.abs(a)[2]` → `NA` |  |
| `absolute` | `np.ma.absolute(a)[2]` → `masked` | `nd.absolute(a)[2]` → `NA` |  |
| `angle` | `np.ma.angle(a)[2]` → `masked` | `nd.angle(a)[2]` → `NA` |  |
| `arccos` | `np.ma.arccos(a)[2]` → `masked` | `nd.arccos(a)[2]` → `NA` | domain: \|x\| <= 1 |
| `arccosh` | `np.ma.arccosh(a)[2]` → `masked` | `nd.arccosh(a)[2]` → `NA` | domain: x >= 1 |
| `arcsin` | `np.ma.arcsin(a)[2]` → `masked` | `nd.arcsin(a)[2]` → `NA` | domain: \|x\| <= 1 |
| `arcsinh` | `np.ma.arcsinh(a)[2]` → `masked` | `nd.arcsinh(a)[2]` → `NA` |  |
| `arctan` | `np.ma.arctan(a)[2]` → `masked` | `nd.arctan(a)[2]` → `NA` |  |
| `arctanh` | `np.ma.arctanh(a)[2]` → `masked` | `nd.arctanh(a)[2]` → `NA` | domain: \|x\| <= 1 |
| `around` | `np.ma.around(a)[2]` → `masked` | `nd.around(a)[2]` → `NA` |  |
| `ceil` | `np.ma.ceil(a)[2]` → `masked` | `nd.ceil(a)[2]` → `NA` |  |
| `conjugate` | `np.ma.conjugate(a)[2]` → `masked` | `nd.conjugate(a)[2]` → `NA` |  |
| `cos` | `np.ma.cos(a)[2]` → `masked` | `nd.cos(a)[2]` → `NA` |  |
| `cosh` | `np.ma.cosh(a)[2]` → `masked +RuntimeWarning` | `nd.cosh(a)[2]` → `NA` |  |
| `exp` | `np.ma.exp(a)[2]` → `masked +RuntimeWarning` | `nd.exp(a)[2]` → `NA` |  |
| `fabs` | `np.ma.fabs(a)[2]` → `masked` | `nd.fabs(a)[2]` → `NA` |  |
| `floor` | `np.ma.floor(a)[2]` → `masked` | `nd.floor(a)[2]` → `NA` |  |
| `log` | `np.ma.log(a)[2]` → `masked` | `nd.log(a)[2]` → `NA` |  |
| `log2` | `np.ma.log2(a)[2]` → `masked` | `nd.log2(a)[2]` → `NA` |  |
| `log10` | `np.ma.log10(a)[2]` → `masked` | `nd.log10(a)[2]` → `NA` |  |
| `logical_not` | `np.ma.logical_not(a)[2]` → `masked` | `nd.logical_not(a)[2]` → `NA` |  |
| `negative` | `np.ma.negative(a)[2]` → `masked` | `nd.negative(a)[2]` → `NA` |  |
| `round` | `np.ma.round(a)[2]` → `masked` | `nd.round(a)[2]` → `NA` |  |
| `sin` | `np.ma.sin(a)[2]` → `masked` | `nd.sin(a)[2]` → `NA` |  |
| `sinh` | `np.ma.sinh(a)[2]` → `masked +RuntimeWarning` | `nd.sinh(a)[2]` → `NA` |  |
| `sqrt` | `np.ma.sqrt(a)[2]` → `masked` | `nd.sqrt(a)[2]` → `NA` |  |
| `tan` | `np.ma.tan(a)[2]` → `masked` | `nd.tan(a)[2]` → `NA` |  |
| `tanh` | `np.ma.tanh(a)[2]` → `masked` | `nd.tanh(a)[2]` → `NA` |  |

### Binary ufuncs

| Operation | `np.ma` call → result | `nd` call → result | Note |
|---|---|---|---|
| `add` | `np.ma.add(a, b)[2]` → `masked` | `nd.add(a, b)[2]` → `NA` |  |
| `subtract` | `np.ma.subtract(a, b)[2]` → `masked` | `nd.subtract(a, b)[2]` → `NA` |  |
| `multiply` | `np.ma.multiply(a, b)[2]` → `masked` | `nd.multiply(a, b)[2]` → `NA` |  |
| `divide` | `np.ma.divide(a, b)[2]` → `masked` | `nd.divide(a, b)[2]` → `NA` |  |
| `true_divide` | `np.ma.true_divide(a, b)[2]` → `masked` | `nd.true_divide(a, b)[2]` → `NA` |  |
| `floor_divide` | `np.ma.floor_divide(a, b)[2]` → `masked` | `nd.floor_divide(a, b)[2]` → `NA` |  |
| `mod` | `np.ma.mod(a, b)[2]` → `masked` | `nd.mod(a, b)[2]` → `NA` |  |
| `remainder` | `np.ma.remainder(a, b)[2]` → `masked` | `nd.remainder(a, b)[2]` → `NA` |  |
| `power` | `np.ma.power(a, b)[2]` → `masked` | `nd.power(a, b)[2]` → `NA` |  |
| `fmod` | `np.ma.fmod(a, b)[2]` → `masked` | `nd.fmod(a, b)[2]` → `NA` |  |
| `hypot` | `np.ma.hypot(a, b)[2]` → `masked` | `nd.hypot(a, b)[2]` → `NA` |  |
| `arctan2` | `np.ma.arctan2(a, b)[2]` → `masked` | `nd.arctan2(a, b)[2]` → `NA` |  |
| `maximum` | `np.ma.maximum(a, b)[2]` → `masked` | `nd.maximum(a, b)[2]` → `NA` |  |
| `minimum` | `np.ma.minimum(a, b)[2]` → `masked` | `nd.minimum(a, b)[2]` → `NA` |  |
| `equal` | `np.ma.equal(a, b)[2]` → `masked` | `nd.equal(a, b)[2]` → `NA` |  |
| `not_equal` | `np.ma.not_equal(a, b)[2]` → `masked` | `nd.not_equal(a, b)[2]` → `NA` |  |
| `less` | `np.ma.less(a, b)[2]` → `masked` | `nd.less(a, b)[2]` → `NA` |  |
| `less_equal` | `np.ma.less_equal(a, b)[2]` → `masked` | `nd.less_equal(a, b)[2]` → `NA` |  |
| `greater` | `np.ma.greater(a, b)[2]` → `masked` | `nd.greater(a, b)[2]` → `NA` |  |
| `greater_equal` | `np.ma.greater_equal(a, b)[2]` → `masked` | `nd.greater_equal(a, b)[2]` → `NA` |  |
| `logical_and` | `np.ma.logical_and(a, b)[2]` → `masked` | `nd.logical_and(a, b)[2]` → `NA` |  |
| `logical_or` | `np.ma.logical_or(a, b)[2]` → `masked` | `nd.logical_or(a, b)[2]` → `np.True_` |  |
| `logical_xor` | `np.ma.logical_xor(a, b)[2]` → `masked` | `nd.logical_xor(a, b)[2]` → `NA` |  |
| `bitwise_and` | `np.ma.bitwise_and(i, i)[2]` → `masked` | `nd.bitwise_and(i, i)[2]` → `NA` | Kleene on bool, propagates on int |
| `bitwise_or` | `np.ma.bitwise_or(i, i)[2]` → `masked` | `nd.bitwise_or(i, i)[2]` → `NA` | Kleene on bool, propagates on int |
| `bitwise_xor` | `np.ma.bitwise_xor(i, i)[2]` → `masked` | `nd.bitwise_xor(i, i)[2]` → `NA` | Kleene on bool, propagates on int |
| `left_shift` | `np.ma.left_shift(i, i)[2]` → `masked` | `nd.left_shift(i, i)[2]` → `raise UFuncTypeError` | integers only |
| `right_shift` | `np.ma.right_shift(i, i)[2]` → `masked` | `nd.right_shift(i, i)[2]` → `raise UFuncTypeError` | integers only |

### Reductions and statistics

| Operation | `np.ma` call → result | `nd` call → result | Note |
|---|---|---|---|
| `sum` | `a.sum()` → `np.float64(9.0)` | `a.sum()` → `NA` | `np.ma` skips a gap, this propagates: `nd.sum(a, skipna=True)` to skip |
| `prod` | `np.ma.prod(a)` → `np.float64(15.0)` | `nd.prod(a)` → `NA` |  |
| `sum, skipping` | `a.sum()` → `np.float64(9.0)` | `nd.sum(a, skipna=True)` → `np.float64(9.0)` | every `nd` reduction takes `skipna=`, default False |
| `mean` | `a.mean()` → `np.float64(3.0)` | `a.mean()` → `NA` |  |
| `mean along an axis` | `np.ma.mean(m, axis=0)` → `masked_array(data=[2.0, 4.0...` | `nd.mean(m, axis=0)` → `array([2.0, NA], dtype=Null...` | NA on each lane with a gap |
| `mean along an axis, skipping` | `np.ma.mean(m, axis=0)` → `masked_array(data=[2.0, 4.0...` | `nd.mean(m, axis=0, skipna=True)` → `array([2.0, 4.0], dtype=Nul...` |  |
| `std` | `a.std()` → `np.float64(1.632993161855452)` | `a.std()` → `NA` |  |
| `var` | `a.var()` → `np.float64(2.6666666666666665)` | `a.var()` → `NA` |  |
| `min / amin / minmax` | `np.ma.min(a)` → `np.float64(1.0)` | `nd.min(a)` → `NA` |  |
| `max / amax` | `np.ma.max(a)` → `np.float64(5.0)` | `nd.max(a)` → `NA` |  |
| `ptp` | `np.ma.ptp(a)` → `np.float64(4.0)` | `nd.ptp(a)` → `NA` |  |
| `median` | `np.ma.median(a)` → `np.float64(3.0)` | `nd.median(a)` → `NA` | plain `np.median(a)` answers **4.0** here, silently: NA sorts last |
| `average` | `np.ma.average(a)` → `np.float64(3.0)` | `nd.average(a)` → `NA` |  |
| `all` | `np.ma.all(t)` → `np.False_` | `nd.all(t)` → `np.False_` | Kleene |
| `any` | `np.ma.any(t)` → `np.False_` | `nd.any(t)` → `NA` | Kleene |
| `all, skipping` | `np.ma.all(t)` → `np.False_` | `nd.all(t, skipna=True)` → `np.False_` | what `np.ma` does, asked for by name |
| `count` | `a.count()` → `np.int64(3)` | `nd.count(a)` → `3` |  |
| `count_masked` | `np.ma.count_masked(a)` → `np.int64(1)` | `nd.isna(a).sum()` → `1` |  |
| `argmax` | `a.argmax()` → `np.int64(3)` | `a.argmax()` → `np.int64(2)` | the first gap, numpy's NaN rule: `a[a.argmax()]` is NA exactly when `a.max()` is |
| `argmax, skipping` | `a.argmax()` → `np.int64(3)` | `nd.argmax(a, skipna=True)` → `np.int64(3)` | position in the original array |
| `argmin` | `a.argmin()` → `np.int64(1)` | `a.argmin()` → `np.int64(2)` | as `argmax` |
| `cumsum` | `np.ma.cumsum(a)[3]` → `np.float64(9.0)` | `nd.cumsum(a)[3]` → `NA` | `np.ma` treats a gap as 0 and carries on; here a gap poisons the rest |
| `cumsum, skipping` | `np.ma.cumsum(a)[3]` → `np.float64(9.0)` | `nd.cumsum(a, skipna=True)[3]` → `np.float64(9.0)` | carries past the gap; the gap itself stays NA (`np.ma`: masked) |
| `cumprod` | `np.ma.cumprod(a)[3]` → `np.float64(15.0)` | `nd.cumprod(a)[3]` → `NA` |  |
| `anom / anomalies` | `np.ma.anom(a)[0]` → `np.float64(0.0)` | `(a - nd.mean(a))[0]` → `NA` | no `nd.anom`; the mean propagates so every element is NA |
| `corrcoef` | `np.ma.corrcoef(a, b)[0, 1]` → `np.float64(0.0)` | `nd.corrcoef(a, b)` → `raise AttributeError` | not supported yet |
| `cov` | `np.ma.cov(a, b)[0, 1]` → `np.float64(0.0)` | `nd.cov(a, b)` → `raise AttributeError` | not supported yet |
| `trace` | `np.ma.trace(m)` → `np.float64(5.0)` | `nd.trace(m)` → `np.float64(5.0)` |  |
| `allclose` | `np.ma.allclose(a, a)` → `np.True_` | `nd.allclose(a, a)` → `NA` | NA when a gap leaves it undecided |
| `allequal` | `np.ma.allequal(a, a)` → `np.True_` | `nd.array_equal(a, a)` → `NA` |  |

### Sorting, searching, sets, products

| Operation | `np.ma` call → result | `nd` call → result | Note |
|---|---|---|---|
| `sort` | `np.ma.sort(a)[-1]` → `masked` | `nd.sort(a)[-1]` → `NA` | gaps last on both sides |
| `argsort` | `np.ma.argsort(a)[-1]` → `np.int64(2)` | `nd.argsort(a)[-1]` → `np.int64(2)` |  |
| `unique` | `len(np.ma.unique(s))` → `3` | `len(nd.unique(s))` → `3` | plain `np.unique` refuses; `nd.unique` keeps NA once, last |
| `isin` | `np.ma.isin(i, [2, 4])[2]` → `np.False_` | `nd.isin(i, [2, 4])[2]` → `NA` | `nd.isin` answers NA for a gap |
| `isin, gap in the set` | `np.ma.isin(b, i)[1]` → `np.True_` | `nd.isin(b, i)[1]` → `np.True_` | is 2.0 in `[1, 2, NA, 4]`: yes on both sides |
| `isin, gap in the set, no hit` | `np.ma.isin([5.0], i)[0]` → `masked` | `nd.isin([5.0], i)[0]` → `NA` | is 5 in `[1, 2, NA, 4]`: the gap might be 5, so NA -- SQL's `IN` agrees |
| `intersect1d` | `np.ma.intersect1d(i, i)[0]` → `np.int64(1)` | `nd.intersect1d(i, i)` → `raise ValueError` | needs a bool cast of a gap |
| `union1d` | `len(np.ma.union1d(i, i))` → `4` | `nd.union1d(i, i)` → `array([1, 2, 4, NA], dtype=...` |  |
| `setdiff1d` | `len(np.ma.setdiff1d(i, i))` → `0` | `nd.setdiff1d(i, i)` → `raise ValueError` |  |
| `setxor1d` | `len(np.ma.setxor1d(i, i))` → `0` | `nd.setxor1d(i, i)` → `raise ValueError` |  |
| `nonzero` | `np.ma.nonzero(a)[0]` → `array([0, 1, 3])` | `nd.nonzero(a)` → `raise ValueError` | refusing is the point: a gap has no truth value |
| `where` | `np.ma.where(a > 2, a, 0)[2]` → `masked` | `nd.where(nd.notna(a), a, zero)[2]` → `np.float64(0.0)` | the condition must be a plain bool array, the other arm a Nullable one |
| `choose` | `np.ma.choose([0, 1, 0, 1], [a, b])[2]` → `masked` | `nd.choose([0, 1, 0, 1], [a, b])[2]` → `NA` |  |
| `compress / compressed` | `np.ma.compressed(a)` → `3` | `nd.dropna(a)` → `3` | `compressed` drops the gaps |
| `take` | `np.ma.take(a, [0, 2])[1]` → `masked` | `nd.take(a, [0, 2])[1]` → `NA` |  |
| `put` | `np.ma.put(a.copy(), [0], [9.0])` → `None` | `nd.put(a.copy(), [0], [9.0])` → `None` | in place, returns None |
| `putmask` | `np.ma.putmask(a.copy(), [1,0,0,0], 9.0)` → `None` | `nd.putmask(a.copy(), [1,0,0,0], 9.0)` → `None` |  |
| `diff / ediff1d` | `np.ma.diff(a)[1]` → `masked` | `nd.diff(a)[1]` → `NA` |  |
| `dot` | `np.ma.dot(a, b)` → `masked_array(data=15., mask...` | `nd.dot(a, b)` → `NA` | `np.dot` itself cannot work: its legacy slot gets a NULL array |
| `inner` | `np.ma.inner(a, b)` → `np.float64(15.0)` | `nd.inner(a, b)` → `raise ValueError` | same slot |
| `outer` | `np.ma.outer(a, b)[2, 0]` → `masked` | `nd.outer(a, b)[2, 0]` → `NA` |  |
| `matmul` | `a @ b` → `raise ValueError` | `a @ b` → `NA` | `@` is a gufunc, so this one works |
| `convolve` | `np.ma.convolve(a, b)[0]` → `np.float64(3.0)` | `nd.convolve(a, b)` → `raise ValueError` | goes through `dot` |
| `correlate` | `np.ma.correlate(a, b)[0]` → `masked` | `nd.correlate(a, b)` → `raise ValueError` | goes through `dot` |
| `polyfit` | `np.ma.polyfit(b, b, 1)[0]` → `np.float64(0.9999999999999997)` | `nd.polyfit(b, b, 1)` → `raise ValueError` | linear algebra wants plain floats |
| `vander` | `np.ma.vander(b)[0, 0]` → `np.float64(1.0)` | `nd.vander(b)[0, 0]` → `np.float64(1.0)` |  |
| `clip` | `np.ma.clip(a, 2, 4)[2]` → `masked` | `np.clip(a, lo, hi)[2]` → `NA` | bounds must be Nullable too |
| `apply_along_axis` | `np.ma.apply_along_axis(np.sum, 0, a)` → `masked_array(data=9., mask=...` | `nd.apply_along_axis(np.sum, 0, a)` → `array(NA, dtype=Nullable(dt...` |  |
| `apply_over_axes` | `np.ma.apply_over_axes(np.sum, m, [0])[0,0]` → `np.float64(4.0)` | `nd.apply_over_axes(np.sum, m, [0])[0,0]` → `np.float64(4.0)` |  |
| `unwrap` | `np.ma.unwrap(a)[2]` → `masked` | `nd.unwrap(a)` → `raise DTypePromotionError` | phase unwrapping goes through `np.diff` plus in-place masking |
| `ndenumerate` | `list(np.ma.ndenumerate(a))[2]` → `((3,), np.float64(5.0))` | `list(nd.ndenumerate(a))[2]` → `((2,), NA)` | `np.ma` skips gaps, numpy's does not |

### Creation, shape, joining

| Operation | `np.ma` call → result | `nd` call → result | Note |
|---|---|---|---|
| `array` | `np.ma.array([1.0, 2.0])[0]` → `np.float64(1.0)` | `np.array([1.0, 2.0], nd.Nullable('f8'))[0]` → `np.float64(1.0)` |  |
| `masked_array` | `np.ma.masked_array(v, mask=m)[2]` → `masked` | `a[2] = nd.NA` → `NA` | the mask is an argument there, a value here |
| `asarray / asanyarray` | `np.ma.asarray(a)[2]` → `masked` | `nd.asarray(a)[2]` → `NA` |  |
| `copy` | `np.ma.copy(a)[2]` → `masked` | `nd.copy(a)[2]` → `NA` |  |
| `zeros / ones / empty` | `np.ma.zeros(2)[0]` → `np.float64(0.0)` | `np.zeros(2, nd.Nullable('f8'))[0]` → `np.float64(0.0)` | a dtype argument, not a function |
| `zeros_like / ones_like` | `np.ma.zeros_like(a)[2]` → `masked` | `nd.zeros_like(a)[2]` → `np.float64(0.0)` | `np.ma` keeps the mask, here a zero is a zero |
| `empty_like` | `np.ma.empty_like(a)[0]` → `np.float64(0.0)` | `nd.empty_like(a)[0]` → `np.float64(0.0)` |  |
| `masked_all` | `np.ma.masked_all(2)[0]` → `masked` | `x[...] = nd.NA` → `NA` |  |
| `masked_all_like` | `np.ma.masked_all_like(a)[0]` → `masked` | `y = nd.empty_like(a); y[...] = nd.NA` → `NA` |  |
| `arange` | `np.ma.arange(3)[1]` → `np.int64(1)` | `np.arange(3, dtype=nd.Nullable('i8'))[1]` → `np.int64(1)` | needed the `fill` slot |
| `identity` | `np.ma.identity(2)[0, 0]` → `np.float64(1.0)` | `np.eye(2, dtype=nd.Nullable('f8'))[0,0]` → `np.float64(1.0)` |  |
| `indices / fromfunction` | `np.ma.indices((2,))[0][1]` → `np.int64(1)` | `nd.indices((2,))[0][1]` → `np.int64(1)` | plain integer arrays, no gaps involved |
| `frombuffer` | `np.ma.frombuffer(bs, 'f8')[0]` → `np.float64(0.0)` | `np.frombuffer(bs, nd.Nullable('f8'))[0]` → `np.float64(0.0)` |  |
| `reshape` | `np.ma.reshape(a, (2, 2))[1, 0]` → `masked` | `nd.reshape(a, (2, 2))[1, 0]` → `NA` |  |
| `resize` | `np.ma.resize(a, 6)[2]` → `masked` | `nd.resize(a, 6)[2]` → `NA` |  |
| `ravel / flatten` | `np.ma.ravel(m)[1]` → `masked` | `nd.ravel(m)[1]` → `NA` |  |
| `squeeze` | `np.ma.squeeze(a)[2]` → `masked` | `nd.squeeze(a)[2]` → `NA` |  |
| `transpose` | `np.ma.transpose(m)[1, 0]` → `masked` | `nd.transpose(m)[1, 0]` → `NA` |  |
| `swapaxes` | `np.ma.swapaxes(m, 0, 1)[1, 0]` → `masked` | `nd.swapaxes(m, 0, 1)[1, 0]` → `NA` |  |
| `expand_dims` | `np.ma.expand_dims(a, 0)[0, 2]` → `masked` | `nd.expand_dims(a, 0)[0, 2]` → `NA` |  |
| `atleast_1d / atleast_2d / atleast_3d` | `np.ma.atleast_2d(a)[0, 2]` → `masked` | `nd.atleast_2d(a)[0, 2]` → `NA` |  |
| `concatenate` | `np.ma.concatenate([a, a])[2]` → `masked` | `nd.concatenate([a, a])[2]` → `NA` | plain `np.concatenate` drops the mask; here there is nothing to drop |
| `stack / hstack / vstack` | `np.ma.stack([a, a])[0, 2]` → `masked` | `nd.stack([a, a])[0, 2]` → `NA` |  |
| `dstack / column_stack` | `np.ma.column_stack([a, a])[2, 0]` → `masked` | `nd.column_stack([a, a])[2, 0]` → `NA` |  |
| `append` | `np.ma.append(a, a)[2]` → `masked` | `nd.append(a, a)[2]` → `NA` |  |
| `repeat` | `np.ma.repeat(a, 2)[4]` → `masked` | `nd.repeat(a, 2)[4]` → `NA` |  |
| `hsplit` | `np.ma.hsplit(a, 2)[1][0]` → `masked` | `nd.hsplit(a, 2)[1][0]` → `NA` |  |
| `diag / diagflat / diagonal` | `np.ma.diagonal(m)[1]` → `np.float64(4.0)` | `nd.diagonal(m)[1]` → `np.float64(4.0)` |  |
| `shape / size / ndim` | `np.ma.shape(a)` → `(4,)` | `nd.shape(a)` → `(4,)` |  |
| `mr_` | `np.ma.mr_[a, a][2]` → `masked` | `np.r_[a, a][2]` → `NA` |  |

### The mask-only API

| Operation | `np.ma` call → result | `nd` call → result | Note |
|---|---|---|---|
| `masked / masked_singleton` | `a[2] is np.ma.masked` → `True` | `a[2] is nd.NA` → `True` | one is a sentinel object, the other a value of the dtype |
| `nomask` | `np.ma.nomask` → `np.False_` | — | no mask exists, so nothing to be empty |
| `getmask / getmaskarray` | `np.ma.getmaskarray(a)[2]` → `np.True_` | `nd.isna(a)[2]` → `np.True_` |  |
| `getdata` | `np.ma.getdata(a)[2]` → `np.float64(12345.0)` | `nd.to_numpy(a, na_value=0.0)[2]` → `np.float64(0.0)` | **the leak**: `getdata` hands back the value under the mask |
| `filled` | `np.ma.filled(a, 0.0)[2]` → `np.float64(0.0)` | `nd.filled(a, 0.0)[2]` → `np.float64(0.0)` |  |
| `compressed` | `len(np.ma.compressed(a))` → `3` | `len(nd.dropna(a))` → `3` |  |
| `count / count_masked` | `a.count()` → `np.int64(3)` | `nd.count(a)` → `3` |  |
| `masked_where` | `np.ma.masked_where(a > 4, a)[3]` → `masked` | `x[nd.filled(x > 4, False)] = nd.NA` → `NA` | a comparison answers `Nullable[bool]`, so fill it before indexing |
| `masked_equal` | `np.ma.masked_equal(a, 5.0)[3]` → `masked` | `x[nd.filled(x == 5.0, False)] = nd.NA` → `NA` |  |
| `masked_greater / masked_greater_equal / masked_less / masked_less_equal / masked_not_equal / masked_inside / masked_outside / masked_values` | `np.ma.masked_greater(a, 4.0)[3]` → `masked` | `same shape: compare, fill, assign NA` → `NA` | one pattern covers all of them |
| `masked_invalid / fix_invalid` | `np.ma.masked_invalid(nanarr)[1]` → `masked` | `y[np.isnan(y)] = nd.NA` → `NA` | a NaN is an ordinary value here until you say otherwise |
| `masked_object` | `np.ma.masked_object(objarr, None)` → `masked` | — | object arrays are refused outright |
| `is_mask / is_masked` | `np.ma.is_masked(a)` → `True` | `nd.isna(a).any()` → `True` |  |
| `isMaskedArray` | `np.ma.isMaskedArray(a)` → `True` | `nd.is_nullable(a.dtype)` → `True` | one asks about the array, one about the dtype |
| `make_mask / make_mask_none / make_mask_descr / mask_or / flatten_mask` | `np.ma.make_mask([1, 0])` → `array([ True, False])` | — | these build and combine mask arrays; there is no mask object here |
| `harden_mask / soften_mask` | `a.harden_mask()` → `False` | — | a gap is a value: assigning over it always works |
| `default_fill_value / common_fill_value` | `np.ma.default_fill_value(a)` → `1e+20` | — | no fill value is carried around; `nd.filled(a, x)` says it at the call |
| `maximum_fill_value / minimum_fill_value` | `np.ma.maximum_fill_value(a)` → `-inf` | — | same: nothing to configure |
| `set_fill_value / a.fill_value` | `a.fill_value` → `np.float64(1e+20)` | — | same |
| `masked_print_option` | `str(np.ma.masked_print_option)` → `'--'` | `repr always prints NA` | not configurable, on purpose |
| `notmasked_edges / flatnotmasked_edges` | `np.ma.notmasked_edges(a)` → `array([0, 3])` | `np.flatnonzero(nd.notna(a))[[0, -1]]` → `array([0, 3])` | plain numpy over `nd.notna` |
| `notmasked_contiguous / flatnotmasked_contiguous` | `len(np.ma.flatnotmasked_contiguous(a))` → `2` | `same idea over nd.notna(a)` |  |
| `clump_masked / clump_unmasked` | `len(np.ma.clump_masked(a))` → `1` | `same idea over nd.isna(a)` |  |
| `compress_rows / compress_cols / compress_rowcols / compress_nd` | `np.ma.compress_rows(m).shape` → `(1, 2)` | `m[~nd.isna(m).any(axis=1)].shape` → `(1, 2)` |  |
| `mask_rows / mask_cols / mask_rowcols` | `np.ma.mask_rows(m)[0, 0]` → `masked` | `m[nd.isna(m).any(axis=1)] = nd.NA` → `NA` |  |
| `MaskedArray / MaskType / mvoid / bool_` | `np.ma.MaskedArray` → `'MaskedArray'` | `nd.NullableDType` → `'NullableDType'` | a subclass of ndarray vs a dtype |
| `MAError / MaskError` | `np.ma.MaskError` → `'MaskError'` | — | errors come out as plain `ValueError` / `TypeError` |
| `fromflex / flatten_structured_array / ids` | `np.ma.fromflex` → `'fromflex'` | — | internals of the mask representation |
| `core / extras` | `np.ma.core` → `'numpy.ma.core'` | — | submodules |

<!-- covered 218/218 of numpy.ma.__all__, less 10 numpy-1 spellings -->

## How to read this

### The first rows are why the prototype exists

With `np.ma` the value under the mask is still there, and it comes back out
the moment a function does not know about `MaskedArray`: `np.asarray`,
`astype`, `np.save`, `np.concatenate`, `np.ma.getdata`. It returns **looking
like real data**. Here a gap *is* the NA pattern, so there is nothing to leak,
and every way out has to say what a gap should become.

### These are two different concepts, not two implementations

Nathaniel Smith separated them back in 2012, in the
[NA discussion summary](https://github.com/njsmith/numpy/wiki/NA-discussion-status):

> **MISSING** — "MISSINGness acts like a property of a datum — assigning
> MISSING to a location is like assigning any other value."
>
> **IGNORED** — "IGNOREDness acts like a property of the array — toggling a
> location to be IGNORED is kind of vaguely similar to changing an array's
> shape."

`NullableDType` is the **MISSING** branch, and not by choice: NA lives in the
element's own bytes, so it is a property of the datum. All three consequences
in that definition hold here — assigning NA goes through `setitem` like any
other value, the old value cannot be recovered, and there is no
"non-destructive mask" to leak through.

`np.ma` is the **IGNORED** branch. It keeps the value so unmasking can bring it
back, and by default it skips masked elements in reductions.

So most of the differences in the reduction table are **not better or worse,
they are a different concept**. `a.sum()` giving `9.0` there and `NA` here are
both right by their own definition. For the other behaviour, say
`nd.sum(a, skipna=True)`.

### The two layers stack

The question left open in 2012 was whether these should be one interface with
compromises or two features that compose. They compose:

```python
m = np.ma.masked_array(a, mask=[False, True, False, False])
m        # [3.0, --, NA, 5.0]   two kinds of hole, printed differently
m[1]     # masked   <- IGNORED: the user set it aside, unmasking brings it back
m[2]     # NA       <- MISSING: the data is not there, nothing to bring back
```

`np.ma` keeps its mask outside, the dtype keeps NA inside, and they do not tread
on each other. What does not line up yet is `m.sum()`, which answers `NA`: the
two layers do not share a notion of `skipna`.

### The arithmetic-error rows point in opposite directions

`np.ma` hides a real error — `1/0` becomes a masked element — while it warns
about data that does not exist, because it still evaluates `sqrt(-1)` on the
value under the mask. This one does the opposite, which is the right way round:
a real error must be loud, and something that is not there must be silent.

### MISSING all the way through

A gap is a value nobody knows, so every answer that depends on it is unknown
too. A review against that rule found two places that had slipped into
IGNORED behaviour, and both are fixed:

- **`argmax` / `argmin`** used to skip gaps, so `a.max()` said NA while
  `a[a.argmax()]` said 5.0. They now point at the first gap — numpy's rule for
  NaN — and `nd.argmax(a, skipna=True)` skips on request.
- **`nd.isin`** dropped the gaps from the test set, so `5 in [1, 2, NA, 4]`
  was False. The gap might be 5: it is NA now, as SQL's `IN` has it. A hit is
  still True.

Three conventions look like skipping but are not, and stay:

- `sort` puts NA last. That is where it goes, not a claim about its size, and
  NaN is treated the same way.
- `nd.unique` keeps NA once, at the end, as R and SQL's `DISTINCT` do.
- `nd.count` counts the values that are there; that is its definition.

### Skipping is always asked for by name

Every reduction in `nd` — `sum`, `prod`, `min`, `max`, `mean`, `std`, `var`,
`median`, `quantile`, `percentile`, `all`, `any`, `argmax`, `argmin`,
`cumsum`, `cumprod` — takes `skipna=False` by default, and the reductions take
`axis=` and `keepdims=` as numpy does. Along an axis, a lane with a gap is NA
and every other lane is numpy's own answer. `skipna=True` gives the IGNORED
answer, except over nothing at all: then `sum` and `prod` give their
identity, and everything else gives NA, not NaN. `np.sum(a, skipna=True)`
itself cannot exist without changing numpy, so the keyword lives on `nd.*`.

### When numpy cannot, there is an `nd.*`

`nd` forwards everything to numpy — `nd.sort` **is** `np.sort` — and only
implements what numpy gets wrong or refuses, or what needs `skipna=`: the
reductions above, `nd.unique`, `nd.dot`, `nd.array_equal`, `nd.isin`,
`nd.isclose`, `nd.allclose`, `nd.count`, `nd.dropna`. Names that would
mislead, like `np.nansum`, are refused with the call to use instead, because
NA is not NaN.

### Where this one loses

`np.median` returns a wrong number without a sound, while `np.ma` at least
warns — both should be avoided, but silence is worse. `np.all`,
`np.array_equal` and `np.dot` do not work at all. The "Limitation" section
of `README.md` says what each of them is stuck on.
