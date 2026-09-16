# Nullable[T] — prototype

`Nullable[T]` is a **parametric DType**: it wraps any other dtype and gives it a
notion of missing data. The goal is to replace `numpy.ma` with something where
the missingness lives **inside the dtype**, not in a parallel array class.

NA is a reserved value of the wrapped type itself, so an element costs exactly
what that type costs:

```python
nd.Nullable(np.float64)   # itemsize 8   — NA is every bit but the sign
nd.Nullable("U10")        # itemsize 40  — NA is the whole cell of U+FFFF
nd.Nullable(record)       # itemsize T   — every field holds its own type's NA
```

582 tests check the semantics operation by operation.

| Document | What is in it |
|---|---|
| `LAYOUTS.md` | the NA pattern of every type, and what each gives up |
| `VS-NUMPY-MA.md` | all 218 names in `numpy.ma.__all__` that numpy 2 still spells that way, side by side with `nd`, with real calls and results |
| `DISCUSSION.md` | how it works and why: the borrowed loops, performance, the bugs found on the way, leak checking |
| `NUMPY-PATCHES.md` | where numpy's array functions fail on Nullable, and patches that leave plain arrays unchanged |
| `AGENTS.md` | notes for coding agents working on the repo |
| `archive/flag-layout/` | the earlier two-layout version, frozen |

## Running it

```bash
./build.sh       # compile
./run_tests.sh   # build + pytest
```

The build uses the numpy dev tree at `/mnt/c/Dev/projects/numpy/build-install`.

## What works

```python
import numpy as np, nulldtype as nd

dt = nd.Nullable(np.float64)      # itemsize 8, NA is a bit pattern
a = np.arange(5.0).astype(dt)
a[2] = nd.NA

a                      # [0.0 1.0 NA 3.0 4.0]
a + a                  # [0.0 2.0 NA 6.0 8.0]

np.can_cast(np.float64, dt)               # False — a value is given up, so not "safe"
np.can_cast(np.float64, dt, "same_kind")  # True  — astype and assignment still automatic
np.can_cast(dt, np.float64)               # False — dropping NA has to be said out loud

a.astype(np.float64)   # ValueError: cannot convert a missing value to dtype('float64')

a + 1.0                # [1.0 2.0 NA 4.0 5.0]   a Python scalar works
a + np.arange(5.0)     # so does a plain array, either way round

np.sqrt(a) * 2         # unary ufuncs too
a.sum()                # NA          propagates, R's default
np.add.reduce(a, where=nd.notna(a))                # 8.0, skipping, said out loud
np.add.accumulate(a)   # [0.0 1.0 NA NA NA]

nd.isna(a)             # [False False True False False]
nd.filled(a, 0.0)      # [0. 1. 0. 3. 4.]
nd.to_numpy(a)         # ValueError: pass na_value= to say what they should become
```

Strings, raw bytes and records:

```python
s = np.zeros(3, nd.Nullable("S3"))    # itemsize 3
s[0] = b"ab"; s[1] = nd.NA
s                      # [b'ab' NA b'']
s.tobytes()            # b'ab\x00\xff\xff\xff\x00\x00\x00'   the gap is all 0xFF
s[2] = b"\xff\xff\xff" # ValueError: ... is the value reserved to mean NA

u = np.array(["b", "a", "c"]).astype(nd.Nullable("U3"))
u[1] = nd.NA
np.sort(u)             # [b c NA]    gaps sort last

r = np.zeros(2, nd.Nullable(np.dtype([("a", "i4"), ("b", "f8")])))  # itemsize 12
r[0] = (1, 2.0); r[1] = nd.NA   # the gap: (0x80000000, 0x7FFFFFFFFFFFFFFF)
r == r                          # [True NA]
nd.Nullable(np.longdouble)   # Nullable(float64), with a LongDoubleWarning
nd.Nullable(">i4")           # Nullable(int32) — stored in native byte order
```

| Working | Notes |
|---|---|
| Parametric DType | `Nullable(f8)`, `Nullable(u1)`, `Nullable("U10")`, `Nullable(record)`... |
| Records | missing when every field holds its own type's NA; nested records, subarray fields, `longdouble` fields; `==`/`!=` answer `Nullable(bool)` |
| `S`, `V` | NA is the whole cell of `0xFF`, at any width |
| `U` | NA is every character `U+FFFF`, the noncharacter Unicode reserves for internal use, so a gap is still a well-formed string |
| `longdouble`, `clongdouble` | stored as `float64` / `complex128`, with a `LongDoubleWarning` where precision is actually lost |
| Byte order | always stored native: `Nullable(">i4") == Nullable("<i4")`, and big-endian data still casts in and out correctly |
| `setitem` / `getitem` | both `nd.NA` and `None` mark an element missing |
| A gap is not observable | its bytes are the NA pattern; nothing of the old value survives |
| Cast `Nullable[T] -> Nullable[T]` | mandatory: numpy rejects a DType without it |
| 29 binary ufuncs | arithmetic, `maximum`/`minimum`, the 6 comparisons, `remainder`, `fmod`, `arctan2`, `hypot`, `logaddexp`, `copysign`, `nextafter`, `fmax`/`fmin`, `heaviside`, `float_power` — **borrowing T's own loop**, merging validity is all this code does |
| `&`, `\|`, `^` | two truth tables under one name: Kleene for `Nullable[bool]`, propagation for `Nullable[int]`; the resolver picks by wrapped dtype |
| Results follow numpy's rules | `resolve` asks the wrapped ufunc, so `Nullable[i8] / Nullable[i8]` is `Nullable[f8]` |
| Comparisons answer `Nullable[bool]` | `NA == NA` is `NA`, not `True` |
| `clip`, `matmul` | `clip` is a 3-input ufunc; `matmul` is a gufunc, including `v@A`, `A@v` and the batched forms |
| 41 unary ufuncs | all of trig/hyperbolic/log/exp, rounding, `isnan`/`isinf`/`isfinite`, `signbit`, `conjugate`, `~` and `logical_not` |
| Reductions | `sum`/`prod`/`max`/`min`, **propagating by default**, like R's `na.rm = FALSE`; over several axes at once too, following numpy's own reorderable rule |
| The `nd.*` statistics layer | `sum`/`prod`/`min`/`max`/`mean`/`std`/`var`/`median`/`quantile`/`percentile`, each with `axis=`, `keepdims=` and `skipna=False`, plus `unique` — replacing the numpy functions built on `sort`, which return **wrong numbers** when a gap is present |
| `nd` forwards to numpy | `nd.sort` **is** `np.sort`; only what numpy gets wrong or refuses has its own implementation, and `nansum` and friends are refused with the call to use instead |
| `nd.count`, `nd.dropna`, `nd.dot`, `nd.array_equal`, `nd.isin`, `nd.isclose`, `nd.allclose` | stand-ins for what numpy cannot do; they answer `NA` when the answer is genuinely unknown — `nd.isin(5, [1, NA])` is NA, since the gap might be 5 |
| Skipping NA | `skipna=True` on any `nd` reduction, `nd.all`/`nd.any`, `nd.argmax`/`nd.argmin` and `nd.cumsum`/`nd.cumprod` — always asked for by name; `np.add.reduce(a, where=nd.notna(a))` works too |
| `accumulate` | `cumsum`/`cumprod`/`cummax`, NA from the first gap onwards; `nd.cumsum(a, skipna=True)` carries past a gap and leaves NA at it, as pandas does |
| `isna`, `notna`, `filled`, `to_numpy` | explicit ways out; `to_numpy` refuses to guess |
| `sort`, `argsort` | gaps last, two gaps compare equal — as in R |
| `nd.all`, `nd.any` | Kleene reductions, with `axis=` and `skipna=`; `np.all` cannot be used because it pins `dtype=bool` |
| `isnan`, `isinf`, `isfinite` | of NA answer NA — you cannot answer a question about a value you do not have |
| pickle | both the dtype and the array |
| `nonzero`, `count_nonzero` | work, and **refuse** on a gap rather than guess, at any array size |
| `argmax`, `argmin` | the position of the first gap, numpy's rule for NaN, so `a[a.argmax()]` is NA exactly when `a.max()` is; `nd.argmax(a, skipna=True)` skips |
| `std`, `var`, `mean` | propagate NA |
| `np.zeros` | gives real zeros, not gaps |
| Kleene three-valued logic | `logical_and/or/xor`, `NA & False = False`, `NA \| True = True` |
| Nothing is computed on a gap | `where=` is handed to the wrapped ufunc when needed, so a gap raises no spurious warning while real errors still do |
| Cast `T -> Nullable[T]` | automatic (`same_kind`, because a value is given up); hitting the reserved value raises |
| Cast `Nullable[T] -> T` | `same_kind`, and **raises on a gap** — the guard is in the loop, not in the casting level, which is why `np.isin` works while NA still cannot slip out |
| Cast to another plain dtype | `Nullable[i4].astype(np.float64)`, to `int16`, to big-endian — numpy casts the values |
| Cast between wrapped dtypes | `Nullable[i8] -> Nullable[f8]`, gaps preserved |
| Nothing invents an NA | casting `int64 2**31 -> Nullable[i4]`, truncating `S5 -> S3` down to all `0xFF`, and arithmetic landing exactly on `INT_MIN` or `UINT_MAX` all raise |
| `np.arange` | the `fill` slot; a range that reaches the reserved value raises |
| Assigning numpy scalars | `np.void`, `np.bytes_`, `np.str_`, records |
| `Nullable[T] op T` | a promoter, both ways round and with Python scalars |
| Value promotion | `Nullable[i8] + Nullable[f8] -> Nullable[f8]` |

Tests: 582 passed, plus 2 expected failures that pin known gaps, and the whole
suite also passes on an ASAN + UBSAN build.
Among them, `test_add_borrows_the_wrapped_loop` compares against
`np.int8(100) + np.int8(100)` to prove the addition really is the wrapped
dtype's, overflow behaviour included.

## Statistics and `skipna`

`np.median`, `np.percentile` and `np.quantile` return a wrong number on a
Nullable array (NA sorts last, so they take it for the maximum), and
`np.mean`/`np.var`/`np.std` truncate on integer and bool ones. Use `nd`:

```python
nd.median(a)                # NA    propagates, the default
nd.median(a, skipna=True)   # 3.0
nd.quantile(a, [.25, .75])  # [NA NA]
nd.unique(a)                # NA is a value of its own, sorted last

m = [[1.0, 2.0, 3.0],
     [4.0, NA,  6.0]]
nd.mean(m, axis=1)               # [2.0 NA]    a lane with a gap is NA
nd.mean(m, axis=1, skipna=True)  # [2.0 5.0]
```

`nd.sum/prod/min/max/mean/std/var/median/quantile/percentile`, `nd.all`/`nd.any`,
`nd.argmax`/`nd.argmin` and `nd.cumsum`/`nd.cumprod` all take `skipna=False`;
the reductions take `axis=` and `keepdims=` as numpy does. Every other name in
`nd` is numpy's own (`nd.sort is np.sort`). Why the numpy versions fail, and
how numpy could be patched without changing plain arrays, is in
`NUMPY-PATCHES.md`.

## Known limitations

- **`np.mean`, `np.var`, `np.std` (and `a.mean()`…) are silently wrong on
  integer and bool Nullable arrays**: `np.mean([1, 2])` gives `1`. numpy picks a
  float result only for its own integer types, so the sum is divided and
  truncated back. Use `nd.mean`/`nd.var`/`nd.std`. Not fixable from the dtype
  side; `NUMPY-PATCHES.md` has the patch.
- **`np.median`, `np.quantile`, `np.percentile` are silently wrong** when a gap
  is present; use the `nd` versions (see above).
- **A Python scalar widens a narrow dtype**: `Nullable[i4] + 2` gives
  `Nullable[i8]`, where numpy keeps `int32` under NEP 50's weak-scalar rule. The
  promoter hands the scalar its own default dtype instead of borrowing the
  array's. Fixable with `NPY_METH_resolve_descriptors_with_scalars`, not done
  yet; an `xfail` test holds the place.
- `np.result_type(dtype, 1.0)` raises `DTypePromotionError`, which takes
  `np.isclose`, `np.allclose` and `np.select` down with it. Declaring
  `common_dtype` for Python scalars makes those three work, but **every** ufunc
  then loses the weak-scalar rule — measured: `Nullable[f4] + 1.0` came out
  float64, and so did `Nullable[i8] // 2`. Left as is.
- **`repr` of a `U` array containing an empty string raises `ValueError`.**
  `arrayprint` formats a non-numpy dtype with `str()`, `str('')` is empty, and
  `_extendLine_pretty` then calls `max()` on an empty list of lines. A numpy bug.
- A gap in `S` and `V` is `0xFF` bytes, which is not valid text, so `astype("U")`
  on a raw view of the values fails to decode. Through the dtype it never comes
  up: a gap always reads as `NA`. For `U` the gap is a valid string, so even the
  raw view works.
- `sort`, `argmax` and `nonzero` hold the GIL for the whole loop, because the
  dtype must declare `NPY_NEEDS_PYAPI` (see the segfaults in `DISCUSSION.md`).
  Large arrays get no threading benefit.
- `maximum` / `minimum` only take the fast path when every stride is a multiple
  of the item size; the item size equals `T`'s, so that always holds (see the
  numpy bug in `DISCUSSION.md`).
- The loops still declare `NPY_METH_REQUIRES_PYAPI`, because the fallback path
  needs the GIL.
- `np.dot` **cannot be implemented**, see `DISCUSSION.md`. Use `@` (`matmul`) or `nd.dot`.
- `a + 1.0` is about 2.6x slower than `a + a` — the scalar operand has stride 0
  and appears to fall off the borrowed-loop path. Not chased down; the results
  are correct.
- `np.unique` fails cleanly; use `nd.unique`.
- `np.array([(1, 2.0), nd.NA], dtype=nd.Nullable(rec))` raises a shape error:
  numpy only treats a tuple as one element when the requested dtype is a plain
  record, and for this dtype it is a sequence. Build with `np.zeros` and assign,
  or cast from a plain record array.
- Records support only `==` and `!=`; `+`, `sum` and `<` fail, as they do for
  plain records.
- `nd.Nullable("S")` (no length) and a bare subarray dtype have no cell to fill,
  so they raise `TypeError`. A subarray as a record field works.
- `np.all` / `np.any` cannot be used, and **cannot be fixed from the dtype
  side**: `ndarray.all` pins its accumulator to plain `bool`, so numpy looks for
  the signature `(BoolDType, NullableDType) -> BoolDType`. A plain `bool` has no
  room for NA. Registering that signature would unlock `np.all`, but it would
  also hijack `np.logical_and(plain_bool_array, nullable_array)`, which today
  correctly answers `Nullable[bool]`. Use `nd.all` / `nd.any` — they now take any
  wrapped dtype, not just bool. `np.array_equal` breaks for the same reason,
  since it calls `np.all`.
- `np.histogram` and `np.einsum` fail cleanly — both need a boolean decision
  about possibly-missing data, and `bool(NA)` refuses to answer.
- Indexing with a mask that contains NA raises `IndexError` from numpy. That is
  **deliberate** — you have to say what a gap should count as — but
  `.filled(False)` is the short way to say it.

## What is next, in order

1. **Consider reporting the numpy bugs upstream.** Four candidates:
   - the SIMD stride bug — patch and reproducer ready;
   - the `can_cast_pyscalar_scalar_to` assertion — patch and reproducer ready;
   - `sort` not checking for `NULL` after `PyArray_DescrNewByteorder`;
   - `repr` of a non-numpy dtype blowing up when `str()` of an element is empty.
2. **Give NEP 50's weak-scalar rule back** with
   `NPY_METH_resolve_descriptors_with_scalars`, so `Nullable[i4] + 2` stops
   widening to `int64`.
3. **A real `longdouble`** through `numpy-quaddtype` instead of substituting
   `float64`.
4. **Remove the fixed cost of `get_loop`** — two Python calls per operation, most
   visible on small arrays.

## Settled design notes

- NA **has a type**: `Nullable[i2]` is not `Nullable[f8]`, and there is no shared
  singleton.
- **One layout only**: NA is a bit pattern of `T` itself.
- A record is **missing when every field is**, each field holding its own type's
  NA.
- The wrapped dtype is **always stored in native byte order**; `long double` is
  stored as `double`.
- Reductions **propagate by default**, `skipna=True` to skip. Nothing skips a
  gap unless asked: `argmax` points at it, and `isin` against a set with a gap
  is NA unless it finds a hit.
- Coercion to a Python scalar **raises**; use `.filled(x)` to get out.
- **A cast never quietly creates an NA**, and never quietly drops one.
- The data under an NA is **not observable**.

## License

BSD 3-Clause, the same terms as NumPy; see `LICENSE.txt`.
