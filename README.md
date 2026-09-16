# Nullable[T] — prototype

[![CI](https://github.com/tuan2k33/NullableDType/actions/workflows/ci.yml/badge.svg)](https://github.com/tuan2k33/NullableDType/actions/workflows/ci.yml)

`Nullable[T]` is a **parametric DType**: it wraps any other dtype and gives it a
notion of missing data that lives **inside the dtype**, not in a parallel mask.
NA is a reserved value of the wrapped type itself, so an element costs exactly
what that type costs:

```python
nd.Nullable(np.float64)   # itemsize 8   — NA is every bit but the sign
nd.Nullable("U10")        # itemsize 40  — NA is the whole cell of U+FFFF
nd.Nullable(record)       # itemsize T   — every field holds its own type's NA
```

Nullable treats NA as a datum, not as a number that is hidden, so there is
nothing like `a.unmask()` or `a[2].value`. Assigning NA overwrites the old value
the way assigning any other value would, and the old value cannot leak out
later. To set values aside and bring them back, keep the original array, or put
a mask on top (`numpy.ma`, [marray](https://github.com/mdhaber/marray)). The
two layers compose. In the terms of the 2012 NumPy discussion, this is
**MISSING**, and masks are **IGNORED**.

| Document | What is in it |
|---|---|
| `LAYOUTS.md` | the NA pattern of every type, and what each gives up |
| `VS-NUMPY-MA.md` | all 218 names in `numpy.ma.__all__` that numpy 2 still spells that way, side by side with `nd`, with real calls and results |
| `DISCUSSION.md` | how it works and why: the borrowed loops, performance, the bugs found on the way, leak checking |
| `NUMPY-PATCHES.md` | where numpy's array functions fail on Nullable, and patches that leave plain arrays unchanged |
| `AGENTS.md` | notes for coding agents working on the repo |
| `archive/flag-layout/` | the earlier two-layout version, frozen |

## Running it

Needs Python 3.12+, numpy 2.5+ and a C compiler. Tested on Linux with numpy
2.5 and the nightly wheels. It will not load on numpy 2.3 or older, which lack
the DType API slots it uses.

```bash
pip install .                    # build and install
pip install pytest && pytest     # run the suite
```

For working on the C code, `./build.sh` builds in place and `./run_tests.sh`
builds and runs pytest (extra arguments go to pytest). They use `python3`;
set `PYTHON`, and `NUMPY_SITE` for a numpy dev tree that is not installed, or
put both in an uncommitted `local.env`.

The suite has 582 tests plus 2 expected failures that pin known gaps. It also
passes on an ASAN + UBSAN build, which CI runs on every push.

## What works

```python
import numpy as np, nulldtype as nd

dt = nd.Nullable(np.float64)      # itemsize 8, NA is a bit pattern
a = np.arange(5.0).astype(dt)
a[2] = nd.NA

a                      # [0.0 1.0 NA 3.0 4.0]
a + a                  # [0.0 2.0 NA 6.0 8.0]
a + 1.0                # [1.0 2.0 NA 4.0 5.0]   a Python scalar works
a + np.arange(5.0)     # so does a plain array, either way round
np.sqrt(a) * 2         # unary ufuncs too
a.sum()                # NA          propagates, R's default
np.add.accumulate(a)   # [0.0 1.0 NA NA NA]

np.can_cast(np.float64, dt)               # False — a value is given up, so not "safe"
np.can_cast(np.float64, dt, "same_kind")  # True  — astype and assignment still automatic
a.astype(np.float64)   # ValueError: cannot convert a missing value to dtype('float64')

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

| Area | What it does |
|---|---|
| Types | every int, uint, float, complex, bool, `datetime64`, `timedelta64`; `S`, `V` (NA is the whole cell of `0xFF`); `U` (every character `U+FFFF`, a noncharacter, so a gap is still a well-formed string); records (missing when every field holds its own NA; nested records, subarray fields) |
| `longdouble`, `clongdouble` | stored as `float64` / `complex128`, with a `LongDoubleWarning` where precision is actually lost |
| Byte order | always stored native: `Nullable(">i4") == Nullable("<i4")`, and big-endian data still casts in and out correctly |
| Assignment | `nd.NA` and `None` mark an element missing; numpy scalars (`np.void`, `np.bytes_`, `np.str_`, records) assign as values |
| 29 binary ufuncs | arithmetic, `maximum`/`minimum`, the 6 comparisons, `remainder`, `fmod`, `arctan2`, `hypot`, `logaddexp`, `copysign`, `nextafter`, `fmax`/`fmin`, `heaviside`, `float_power`, `&`/`\|`/`^` — **borrowing T's own loop**; merging validity is all this code does |
| 41 unary ufuncs | all of trig/hyperbolic/log/exp, rounding, `signbit`, `conjugate`, `~`, `logical_not`; `isnan`/`isinf`/`isfinite` of NA answer NA |
| `clip`, `matmul` | `clip` is a 3-input ufunc; `matmul` is a gufunc, including `v@A`, `A@v` and the batched forms |
| Results follow numpy's rules | `resolve` asks the wrapped ufunc, so `Nullable[i8] / Nullable[i8]` is `Nullable[f8]` and `Nullable[i8] + Nullable[f8]` is `Nullable[f8]` |
| Mixed operands | `Nullable[T] op T` and Python scalars, both ways round, through promoters |
| Comparisons | answer `Nullable[bool]`; `NA == NA` is `NA`, not `True` |
| Three-valued logic | `logical_and/or/xor` and `&`, `\|`, `^` on `Nullable[bool]` are Kleene: `NA & False = False`, `NA \| True = True`; on ints `&`, `\|`, `^` propagate |
| Reductions and `accumulate` | propagate by default, like R's `na.rm = FALSE`, over several axes at once too; `cumsum` is NA from the first gap on |
| `argmax`, `argmin` | the position of the first gap, numpy's rule for NaN, so `a[a.argmax()]` is NA exactly when `a.max()` is |
| `sort`, `argsort` | gaps last, two gaps compare equal — as in R |
| `nonzero`, `count_nonzero` | work, and **refuse** on a gap rather than guess, at any array size |
| Nothing is computed on a gap | `where=` is handed to the wrapped ufunc when needed, so a gap raises no spurious warning while real errors still do |
| Casts in | `T -> Nullable[T]` is automatic (`same_kind`); between wrapped dtypes (`Nullable[i8] -> Nullable[f8]`) gaps are kept |
| Casts out | `Nullable[T] -> T` or any plain dtype (`int16`, big-endian…) **raises on a gap**; the guard is in the loop, which is why `np.isin` works while NA still cannot slip out |
| Nothing invents an NA | casting `int64 2**31 -> Nullable[i4]`, truncating `S5 -> S3` down to all `0xFF`, arithmetic landing exactly on `INT_MIN` or `UINT_MAX`, and an `np.arange` reaching the reserved value all raise |
| Explicit ways out | `nd.isna`, `nd.notna`, `nd.filled`, `nd.to_numpy` (which refuses to guess) |
| pickle | both the dtype and the array |
| `np.zeros` | gives real zeros, not gaps |

## The `nd` namespace and `skipna`

Every name in `nd` that it does not define is numpy's own (`nd.sort is
np.sort`). It defines only what numpy gets wrong or refuses on a Nullable
array, and refuses the `nan*` names with the call to use instead, because NA is
not NaN:

- `np.median`, `np.percentile` and `np.quantile` return a wrong number (NA
  sorts last, so they take it for the maximum);
- `np.mean`, `np.var` and `np.std` truncate on integer and bool arrays;
- `np.all`, `np.any`, `np.dot`, `np.unique`, `np.isclose` do not work at all.

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

- **Reductions:** `nd.sum/prod/min/max/mean/std/var/median/quantile/percentile`
  take `axis=`, `keepdims=` and `skipna=False`.
- **`skipna` elsewhere:** so do `nd.all`/`nd.any` (Kleene),
  `nd.argmax`/`nd.argmin` and `nd.cumsum`/`nd.cumprod`. `nd.cumsum(a,
  skipna=True)` carries past a gap and leaves NA at it, as pandas does.
- **Other stand-ins:** `nd.count`, `nd.dropna`, `nd.dot`, `nd.array_equal`,
  `nd.isin`, `nd.isclose` and `nd.allclose` answer NA when the answer is
  genuinely unknown. Against a set `[1, NA]`, `nd.isin` answers NA for 5: the
  gap might be 5.
- **Skipping without `nd`:** `np.add.reduce(a, where=nd.notna(a))` works too.

Why the numpy versions fail, and how numpy could be patched without changing
plain arrays, is in `NUMPY-PATCHES.md`.

## Known limitations

- **`np.mean`, `np.var`, `np.std` (and `a.mean()`…) are silently wrong on
  integer and bool arrays**: `np.mean([1, 2])` gives `1`. numpy picks a float
  result only for its own integer types, so the sum is divided and truncated
  back. **`np.median`, `np.quantile`, `np.percentile` are silently wrong** when
  a gap is present. Neither is fixable from the dtype side; use `nd`.
- `np.all` / `np.any` (and `np.array_equal`, which calls them) cannot be used:
  `ndarray.all` pins its accumulator to plain `bool`, so numpy looks for
  `(BoolDType, NullableDType) -> BoolDType`. Registering that would also hijack
  `np.logical_and(plain_bool_array, nullable_array)`, which today correctly
  answers `Nullable[bool]`.
- `np.dot` **cannot be implemented**: numpy hands its slot a NULL array (see
  `DISCUSSION.md`). Use `@` or `nd.dot`.
- **Python scalars and NEP 50.**
  - **Widening:** `Nullable[i4] + 2` gives `Nullable[i8]`, where numpy keeps
    `int32` under the weak-scalar rule. Fixable with
    `NPY_METH_resolve_descriptors_with_scalars`, not done yet; an `xfail` test
    holds the place.
  - **Promotion error:** `np.result_type(dtype, 1.0)` raises
    `DTypePromotionError`, which takes `np.isclose`, `np.allclose` and
    `np.select` down with it. Declaring `common_dtype` for Python scalars fixes
    those three, but breaks the weak-scalar rule for every ufunc (measured:
    `Nullable[f4] + 1.0` came out float64).
- **`repr` of a `U` array containing an empty string raises `ValueError`.**
  `arrayprint` formats a non-numpy dtype with `str()`, `str('')` is empty, and
  `_extendLine_pretty` then calls `max()` on an empty list of lines. A numpy bug.
- A gap in `S` and `V` is `0xFF` bytes, which is not valid text, so `astype("U")`
  on a raw view of the values fails to decode. Through the dtype it never comes
  up: a gap always reads as `NA`.
- `sort`, `argmax` and `nonzero` hold the GIL for the whole loop, because the
  dtype must declare `NPY_NEEDS_PYAPI` (see the segfaults in `DISCUSSION.md`),
  and the ufunc loops declare `NPY_METH_REQUIRES_PYAPI` for their fallback path.
  Large arrays get no threading benefit.
- `a + 1.0` is about 2.6x slower than `a + a` — the scalar operand has stride 0
  and appears to fall off the borrowed-loop path. Not chased down; the results
  are correct.
- `np.array([(1, 2.0), nd.NA], dtype=nd.Nullable(rec))` raises a shape error:
  numpy only treats a tuple as one element when the requested dtype is a plain
  record. Build with `np.zeros` and assign, or cast from a plain record array.
- Records support only `==` and `!=`; `+`, `sum` and `<` fail, as they do for
  plain records.
- `nd.Nullable("S")` (no length) and a bare subarray dtype have no cell to fill,
  so they raise `TypeError`. A subarray as a record field works.
- `np.histogram` and `np.einsum` fail cleanly — both need a boolean decision
  about possibly-missing data, and `bool(NA)` refuses to answer. Indexing with a
  mask that contains NA raises `IndexError` for the same reason;
  `nd.filled(mask, False)` says what a gap should count as.

## What is next, in order

1. **Report the numpy bugs upstream, and propose the hooks in
   `NUMPY-PATCHES.md`.** Bug candidates:
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

- NA is **a datum**: assigning it overwrites the old value, which is **not
  observable** afterwards.
- NA **has a type**: `Nullable[i2]` is not `Nullable[f8]`, and there is no shared
  singleton.
- **One layout only**: NA is a bit pattern of `T` itself.
- A record is **missing when every field is**, each field holding its own type's
  NA.
- The wrapped dtype is **always stored in native byte order**; `long double` is
  stored as `double`.
- **Nothing skips a gap unless asked**: reductions propagate, `argmax` points at
  the gap, `isin` against a set with a gap is NA unless it finds a hit;
  `skipna=True` says otherwise.
- Coercion to a Python scalar **raises**; use `nd.filled(x, ...)` to get out.
- **A cast never quietly creates an NA**, and never quietly drops one.

## AI Disclosure

AI was used in writing this project's code, tests and documentation.

## License

BSD 3-Clause, the same terms as NumPy; see `LICENSE.txt`.
