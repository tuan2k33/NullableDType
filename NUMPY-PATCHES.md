# Making numpy's array functions work on Nullable

This document looks at the numpy functions that summarise a whole array, such as
`np.sum`, `np.median`, `np.all`, `np.dot` and `np.argmax`. For each one it
says:

- where numpy implements it;
- why it fails or is wrong on a `Nullable` array;
- what a numpy patch could look like, if it must leave **plain arrays
  unchanged** and let `nd.*` hand work back to numpy.

This is an analysis for possible upstream proposals. The project itself still
builds on stock numpy, so nothing here is applied.

Links point to numpy at
[`3deeb20adb`](https://github.com/numpy/numpy/tree/3deeb20adb47f5da91e490ef8feda81114823710)
(main, 2026-09-14).

[base]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710

## The rule every patch follows

A patch may only touch paths that **raise or give a wrong answer today** for a
DType that is not built into numpy. Two gates keep plain arrays byte-for-byte
unchanged:

- **Fall back on today's error.** Try exactly what numpy does now, and take the
  new path only if that raises. Anything that works today, `StringDType`
  included, stays on the old path.
- **Branch on `type(dtype)._legacy`.** Every builtin dtype is legacy.
  `StringDType` is not, so each such branch also has to leave it alone. In every
  case below `StringDType` either takes the fallback gate or already fails for
  another reason.

What the patches deliver is **MISSING** semantics: numpy's functions then
propagate NA the way the dtype's ufuncs already do. `skipna=` stays in `nd.*`.
Adding it to `np.sum` itself would be a change to numpy's API, not a patch.
numpy already has an explicit way to skip, `where=`, and it works on Nullable
today (see [sum, prod, mean, std, var](#1-sum-prod-min-max-cumsum)).

## Summary

| Function | Today on `Nullable` | Where | Patch size | Plain arrays |
|---|---|---|---|---|
| `sum`, `prod`, `cumsum`, `min`, `max` | correct, propagates | ufunc reductions | none | — |
| `mean`, `var`, `std` on int/bool | **silently wrong**: `mean([1, 2])` is `1` | `_methods.py` | small, Python | unchanged (gated) |
| `all`, `any`, `array_equal` | raise `UFuncTypeError` | `_methods.py` | small, Python | unchanged (fallback) |
| `median`, `quantile`, `percentile` | **silently wrong**: NA treated as the maximum | `_function_base_impl.py` | small, Python | unchanged (gated) |
| `argmax`, `argmin` | an error inside the slot is ignored | `calculation.c` | a few lines, C | unchanged |
| `dot`, `inner`, `vdot` | refused: the slot gets `arr = NULL` | `multiarraymodule.c` | one argument, C | unchanged |
| `isclose`, `allclose` | raise `DTypePromotionError` | `numeric.py` | a few lines, or a fix here | unchanged (fallback) |
| `unique` | raises cleanly | `_arraysetops_impl.py` | needs a new DType hook | — |
| `nan*` | refused by `nd` | `_nanfunctions_impl.py` | none: they skip NaN, not NA | — |
| `count_nonzero`, `histogram`, `nonzero` | raise cleanly | — | none: refusing is right | — |

## 1. `sum`, `prod`, `min`, `max`, `cumsum`

**Path.**
[`fromnumeric.sum`][base-sum] → [`_wrapreduction`][base-wrap] → `ndarray.sum` →
[`_methods._sum`][base-_sum] → `np.add.reduce`. `cumsum` is
`np.add.accumulate`.

[base-sum]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/fromnumeric.py#L2466
[base-wrap]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/fromnumeric.py#L43
[base-_sum]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/_methods.py#L47

**Today.** These already work. The loops are this dtype's own ufunc loops, so
NA propagates, and reductions over several axes work since the loops declare
`NPY_METH_IS_REORDERABLE`.

**Skipping without a patch.** `where=` already expresses IGNORED on request:

```python
np.sum(m, axis=1, where=nd.notna(m))            # [6.0 10.0]
np.mean(m, axis=1, where=nd.notna(m))           # [2.0 5.0]
np.var(m, axis=1, where=nd.notna(m), ddof=1)    # [1.0 2.0]
np.max(m, axis=1, where=nd.notna(m))            # ValueError: needs initial=
np.max(m, axis=1, where=nd.notna(m), initial=-np.inf)   # [3.0 6.0]
```

`nd.sum`, `nd.mean`, `nd.std` and `nd.var` could therefore drop their
lane-by-lane loop along an axis and call numpy with `where=`, after two fixes:

- a lane with nothing left gives `nan` plus a warning, which `nd` would turn
  into NA;
- `min`/`max` need an `initial=` from the wrapped dtype, and an empty lane
  still has to become NA.

**No patch needed.**

## 2. `mean`, `var`, `std`: silently wrong on integer and bool

**Path.** [`_methods._mean`][base-_mean] and [`_methods._var`][base-_var].

[base-_mean]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/_methods.py#L115
[base-_var]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/_methods.py#L148

**Today.** A wrong number, with no warning:

```python
i = np.array([1, 2], dtype=nd.Nullable(np.int64))
np.mean(i)                          # 1        should be 1.5
np.var(np.array([1, 2, 4], dtype=nd.Nullable(np.int64)))   # 1   should be 1.555...
np.mean(np.array([True, False], dtype=nd.Nullable(np.bool_)))  # True   should be 0.5
```

**Cause.** A float result is chosen [only for builtin types][base-f8]:

```python
if issubclass(arr.dtype.type, (nt.integer, nt.bool)):
    dtype = mu.dtype('f8')
```

`Nullable(int64).type` is the NA scalar type, so no float result is chosen.
The sum stays an integer, and then:

- with `axis=None`, `ret.dtype.type(ret / rcount)` truncates 1.5 back to
  `int64`;
- along an axis, [`true_divide(..., out=ret, casting='unsafe')`][base-div]
  truncates it the same way.

Passing `dtype=` does not help: a ufunc accepts only the DType *class* of a
parametric user dtype, and then this dtype's resolver picks `Nullable(int64)`
again.

[base-f8]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/_methods.py#L127
[base-div]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/_methods.py#L134

**Patch.** For a DType that is not legacy, ask the DType what dividing by a
count gives. That is exactly what a mean is, and it covers timedelta as well:

```diff
     if dtype is None:
         if issubclass(arr.dtype.type, (nt.integer, nt.bool)):
             dtype = mu.dtype('f8')
         elif issubclass(arr.dtype.type, nt.float16):
             dtype = mu.dtype('f4')
             is_float16_result = True
+        elif not type(arr.dtype)._legacy:
+            try:
+                res = um.true_divide.resolve_dtypes((arr.dtype, int, None))[2]
+            except TypeError:
+                res = arr.dtype
+            if res != arr.dtype:
+                # a user DType instance cannot be passed as `dtype=`
+                arr = arr.astype(res)
```

The same change goes into `_var`. Measured without patching numpy, by casting
first:

| input | result type from dividing by a count | `mean` / `var` along `axis=1` |
|---|---|---|
| `Nullable(int64)` | `Nullable(float64)` | `[2.333 NA]` / `[1.556 NA]` |
| `Nullable(bool)` | `Nullable(float64)` | `0.5` |
| `Nullable(m8[s])` | `Nullable(m8[s])` | unchanged |

**Plain arrays.** The branch is only reached by DTypes that are not legacy.
`StringDType` reaches it and fails in `resolve_dtypes`, which keeps `res`
equal to its own dtype, so it behaves exactly as today (the mean of strings
already raises).

**Caveat on this side.** `Nullable(float32)` divided by a weak `int` resolves
to `Nullable(float64)`, not `Nullable(float32)`. That is the known NEP 50 gap
in this dtype's resolver, and it should be fixed here, not in numpy.

**Until then.** `nd.mean`, `nd.var` and `nd.std` compute on the plain values
and are correct. `a.mean()` and `np.mean(a)` on an integer or bool Nullable
array are not.

## 3. `all`, `any`, and `array_equal` through them

**Path.** [`fromnumeric.all`][base-all] → `ndarray.all`, which is
[forwarded][base-fwd] to [`_methods._all`][base-_all]:

```python
def _all(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    # By default, return a boolean for any and all
    if dtype is None:
        dtype = bool_dt
    ...
    return umr_all(a, axis, dtype, out, keepdims)
```

The C API [`PyArray_All`][base-PyAll] does the same with `NPY_BOOL`.

[base-all]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/fromnumeric.py#L2715
[base-fwd]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/src/multiarray/methods.c#L2550
[base-_all]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/_methods.py#L64
[base-PyAll]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/src/multiarray/calculation.c#L789

**Today.** `dtype=bool` pins the reduction to `(bool, bool) -> bool`, and there
is no such loop for a Nullable input, so it raises `UFuncTypeError`.
`dtype=None` would find this dtype's Kleene loop:
`np.logical_and.reduce(a)` is already `NA` or `False` as it should be.

**Why `dtype=bool` is there at all.** Without it, `logical_and.reduce` on an
object array returns the last object (`'x'`), not a bool. It has to stay for
everything that works today.

**Patch.** Fall back only when today's call raises:

```diff
 def _all(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
-    if dtype is None:
-        dtype = bool_dt
+    if dtype is None:
+        try:
+            return _all(a, axis, bool_dt, out, keepdims, where=where)
+        except _UFuncNoLoopError:
+            if type(a.dtype)._legacy:
+                raise
+            # a user DType with its own logical_and, e.g. three-valued
+            return umr_all(a, axis, None, out, keepdims, where=where)
```

The same goes into `_any`.

**Plain arrays.** Every call that works today returns on the first line.
`StringDType` works today through the bool cast, so it never reaches the
fallback.

**What follows.** `np.array_equal` calls `.all()`
([L2561][base-ae1], [L2564][base-ae2]) and then `builtins.bool(...)`, so on a gap
it would raise `TypeError` ("truth value of NA") instead of `UFuncTypeError`.
That is the right refusal for a function that promises a Python bool.
`nd.array_equal` stays for the answer NA.

[base-ae1]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/numeric.py#L2561
[base-ae2]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/numeric.py#L2564

## 4. `median`, `quantile`, `percentile`: silently wrong

**Path.** [`_median`][base-med] and [`_quantile`][base-q], both built on
`partition`.

[base-med]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/lib/_function_base_impl.py#L4032
[base-q]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/lib/_function_base_impl.py#L4753

**Today.** `np.median([3, 1, NA, 5])` is `4.0`. NA sorts last, so it is taken
for the largest value.

**Cause.** numpy handles NaN, which also sorts last, with a special case that
only builtin types reach:

```python
supports_nans = np.issubdtype(a.dtype, np.inexact) or a.dtype.kind in 'Mm'
if supports_nans:
    kth.append(-1)            # also partition the last element into place
...
if supports_nans and sz > 0:
    rout = np.lib._utils_impl._median_nancheck(part, rout, axis)
```

([L4049][base-sn], [`_median_nancheck`][base-nc]; `_quantile` has the same
check at [L4781][base-qsn].) The check calls `np.isnan` on the last element and
copies it into the result where that is true. For a Nullable array
`np.isnan(NA)` is `NA`, not `True`, so reusing that check as it is does not
work.

[base-sn]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/lib/_function_base_impl.py#L4049
[base-nc]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/lib/_utils_impl.py#L407
[base-qsn]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/lib/_function_base_impl.py#L4781

**Patch.** A median never exceeds the largest element, and `np.minimum`
propagates whatever the dtype propagates: NaN, NaT, NA. So capping the result
by the last partitioned element does the NaN check's job without asking what
the poison value is:

```diff
-    supports_nans = np.issubdtype(a.dtype, np.inexact) or a.dtype.kind in 'Mm'
+    supports_nans = np.issubdtype(a.dtype, np.inexact) or a.dtype.kind in 'Mm'
+    # a user DType may have a value that sorts last and must propagate
+    cap_by_last = not supports_nans and not type(a.dtype)._legacy
-    if supports_nans:
+    if supports_nans or cap_by_last:
         kth.append(-1)
     ...
     if supports_nans and sz > 0:
         rout = np.lib._utils_impl._median_nancheck(part, rout, axis)
+    elif cap_by_last and sz > 0:
+        rout = np.minimum(rout, part.take(-1, axis=axis), out=out)
```

`_quantile` gets the same treatment in its three `supports_nans` branches: the
interpolated value lies between two order statistics, so it never exceeds the
last one either.

**Measured** with `scratchpad/median_clamp.py`: the capped median against
`np.median` on 20,000 random arrays of `f8`, `f4`, `f2`, `i8`, `c16`, `M8`,
`m8`, with NaN, ±inf, NaT and signed zeros, over random axes.

- Every value and every NaN matches.
- The only differences are the **sign of zero**: `np.median([-0.0])` is
  `0.0`, while the capped version gives `-0.0`. That is why the new path is
  gated to DTypes that are not legacy, and not used to replace the NaN check.
- On Nullable: `[3, 1, NA, 5]` gives `NA`, and along `axis=1`,
  `[[3, 1, 5], [2, NA, 4]]` gives `[3.0 NA]`.

**Plain arrays.** Never reach `cap_by_last`. `StringDType` does, but already
fails earlier, in `mean`.

**What follows.** `np.median(a)` gives NA instead of a wrong number, and
`nd.median` only has to do `skipna`.

## 5. `argmax`, `argmin`: the error channel is ignored

**Path.** [`_PyArray_ArgMinMaxCommon`][base-amm]:

```c
NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap));
for (ip = PyArray_DATA(ap), i = 0; i < n; i++, ip += elsize*m) {
    arg_func(ip, m, rptr, ap);         /* return value dropped */
    rptr += 1;
}
NPY_END_THREADS_DESCR(PyArray_DESCR(ap));
```

[base-amm]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/src/multiarray/calculation.c#L179

**Today.** This dtype's slot cannot report anything. That is why an all-NA
array answers `0`, like numpy's all-NaT case, instead of raising or answering
NA. Two builtin slots have the same gap:

- the flexible ones return `0` when `PyMem_RawMalloc` fails, with no error set;
- `OBJECT_argmax` returns `0` after a failed comparison, with the error set.

**Patch.**

```diff
+    int res = 0;
     NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(ap));
     for (...) {
-        arg_func(ip, m, rptr, ap);
+        res = arg_func(ip, m, rptr, ap);
+        if (res < 0) {
+            break;
+        }
         rptr += 1;
     }
     NPY_END_THREADS_DESCR(PyArray_DESCR(ap));
+    if (res < 0 || (PyDataType_FLAGCHK(PyArray_DESCR(ap), NPY_NEEDS_PYAPI)
+                     && PyErr_Occurred())) {
+        goto fail;
+    }
```

The `PyErr_Occurred` check only makes sense with the GIL held, that is for
dtypes flagged `NPY_NEEDS_PYAPI`.

**Plain arrays.** Every builtin slot returns 0 on success, so nothing changes.
The one changed case is an error that is already set, which then surfaces as
itself rather than as whatever the caller makes of a result with an exception
set.

**What follows.** Not much for Nullable: numpy's rule that an argmax lands on
the first NaN is already mirrored here. A slot could raise on an all-NA lane,
but numpy returns 0 for all-NaT, so matching that is defensible.

## 6. `dot`, `inner`, `vdot`: the slot gets a NULL array

**Path.** [`PyArray_MatrixProduct2`][base-dot] and [`array_vdot`][base-vdot]:

```c
dot(it1->dataptr, is1, it2->dataptr, is2, op, l, NULL);
vdot(ip1, stride1, ip2, stride2, op, n, NULL);
```

[base-dot]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/src/multiarray/multiarraymodule.c#L1115
[base-vdot]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/src/multiarray/multiarraymodule.c#L2717

**Today.** A parametric dtype cannot learn its own descriptor from the slot, so
this dtype leaves `dotfunc` empty and numpy says
`dot not available for this type`. `np.correlate` uses the same slot and
[already passes the result array][base-corr]. `dot` and `vdot` are the odd ones
out.

[base-corr]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/src/multiarray/multiarraymodule.c#L1242

**Patch.** Pass the array, as `correlate` does:

```diff
-            dot(it1->dataptr, is1, it2->dataptr, is2, op, l, NULL);
+            dot(it1->dataptr, is1, it2->dataptr, is2, op, l, out_buf);
 ...
-        vdot(ip1, stride1, ip2, stride2, op, n, NULL);
+        vdot(ip1, stride1, ip2, stride2, op, n, (PyArrayObject *)ap1);
```

Both paths already check `PyErr_Occurred()` after the loop, and
`NPY_BEGIN_THREADS_DESCR` keeps the GIL for `NPY_NEEDS_PYAPI` dtypes.

**Plain arrays.** Every builtin `dotfunc` declares the argument
[`NPY_UNUSED`][base-dotsig], and [the BLAS path][base-cblas] only handles
builtin float and complex types. Old-style user types registered with
`PyArray_RegisterDataType` get a valid array instead of NULL, which is strictly
more than they get now.

[base-dotsig]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/src/multiarray/arraytypes.c.src#L3558
[base-cblas]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/src/common/cblasfuncs.c#L548

**What follows.** This dtype can register a `dotfunc` like its `matmul` loop,
and `nd.dot` goes away.

## 7. `isclose`, `allclose`

**Path.** [`isclose`][base-isc]:

```python
dt = multiarray.result_type(y, 1.)
```

[base-isc]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/numeric.py#L2439

**Today.** `DTypePromotionError`. This dtype declares no common DType with
Python's `float`: declaring one made every scalar operation `float64`, which
breaks NEP 50. The rest of `isclose` works as is; run by hand it gives
`[True True NA True]`.

**Two ways out.**

- **Here (preferred).** Answer `result_type(Nullable(T), 1.)` without breaking
  weak promotion. This is the same NEP 50 gap as in section 2.
- **In numpy:**

  ```diff
  -        dt = multiarray.result_type(y, 1.)
  +        try:
  +            dt = multiarray.result_type(y, 1.)
  +        except exceptions.DTypePromotionError:
  +            if type(dtype)._legacy:
  +                raise
  +            dt = dtype
  ```

  Plain arrays never raise there. `allclose` then reaches `all()`, which needs
  section 3, and `builtins.bool`, which refuses NA. That is correct, and
  `nd.allclose` stays for the answer NA.

## 8. `unique`: needs a hook, not a patch

**Path.** [`_unique1d`][base-uq]. The hash path does not support user dtypes,
so it sorts and then [compares neighbours][base-uq2] into a plain bool mask:

```python
mask[1:] = aux[1:] != aux[:-1]     # NA != NA is NA -> cannot be a bool
```

[base-uq]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/lib/_arraysetops_impl.py#L373
[base-uq2]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/lib/_arraysetops_impl.py#L401

`equal_nan` has special code for `kind in "cfmM"`, keyed on `isnan`.

Handling NA the same way needs numpy to ask a dtype "which of these are
missing", and no such question exists today. The nearest precedent is
[`NPY_DT_get_constant`][base-const] (`NPY_CONSTANT_nan` and friends). A missing
value hook would be a DType API addition, not a patch. `nd.unique` stays.

[base-const]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/_core/include/numpy/dtype_api.h#L433

## 9. What stays as it is

- **`nan*`.** [`_replace_nan`][base-rn] acts only on `np.inexact` types. On a
  Nullable array `np.nansum` is just `np.sum` and propagates, while `np.nanmax`
  fails on `NA.any`. They skip NaN, which is IGNORED semantics for a value this
  dtype keeps distinct from NA, so `nd` refuses the names and points to
  `skipna=True`. Making `nanmax` fail with a clearer message would be a cleanup,
  not a feature.
- **`count_nonzero`, `nonzero`, `histogram`, masks holding NA.** They need a
  yes/no from a value nobody knows, and refusing is the MISSING answer.
- **`np.sum(a, skipna=True)`.** An API change for every ufunc reduction, not a
  patch. `where=nd.notna(a)` already says it.

[base-rn]: https://github.com/numpy/numpy/blob/3deeb20adb47f5da91e490ef8feda81114823710/numpy/lib/_nanfunctions_impl.py#L103

## If these land: what `nd` gives back

| `nd` today | after |
|---|---|
| `nd.mean/var/std` | still needed for `skipna`; `a.mean()` correct |
| `nd.all`, `nd.any` | `np.all`/`np.any` propagate; `nd` keeps `skipna` |
| `nd.median/quantile/percentile` | numpy propagates; `nd` keeps `skipna` |
| `nd.dot` | removed; `np.dot`, `np.inner`, `np.vdot` work |
| `nd.isclose` | removed once the promotion is fixed here |
| `nd.array_equal`, `nd.allclose`, `nd.unique` | stay: numpy promises a Python bool, or needs a hook |

## Suggested order upstream

1. **`dot`/`vdot` pass the array.** One argument each, with a precedent in
   `correlate`.
2. **`argmax`/`argmin` check the result.** A bug fix that also covers the
   builtin failure paths.
3. **`mean`/`var` for DTypes that are not legacy.** A silent wrong result
   today; small and gated.
4. **`all`/`any` fallback.**
5. **`median`/`quantile` cap by the last element.** Needs the most discussion,
   since it relies on "the maximum sorts last, and `minimum` propagates".

`type(dtype)._legacy` is private; an upstream patch would use the internal
check it wraps (`NPY_DT_is_legacy`) or expose a public one.
