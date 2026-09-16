# Nullable[T] — prototype

`Nullable[T]` is a **parametric DType**: it wraps any dtype and gives it a notion
of "missing data". The goal is to replace `numpy.ma` with something where the
missing value lives **inside the dtype**, not in a parallel array.

There are two storage layouts underneath, but **the user sees one name**:

```python
nd.Nullable(np.float64)   # itemsize 8   — NA is a bit pattern of f8 itself
nd.Nullable("U10")        # itemsize 40  — NA is the whole cell of 0xFF
nd.Nullable(record)       # itemsize T   — every field holds its own type's NA
nd.FlagLayout(np.float64) # itemsize 9   — the value plus a flag byte, by name only
```

`Nullable` chooses, and every type with a spare value uses a bit pattern — NA
**costs no extra bytes**. The two layouts behave the same, and 783 tests check
that operation by operation. The full table of types, sizes and NA patterns is
in `LAYOUTS.md`.

R-style semantics — see `../numpy-pr-work/maskeddtype-spec.md`.
Background on why `numpy.ma` is broken — see `../numpy-pr-work/gh-9750-analysis.md`.

## Running

```bash
./build.sh       # compile
./run_tests.sh   # build + pytest
```

The build uses the numpy dev build in `/mnt/c/Dev/projects/numpy/build-install`.

## What works

```python
import numpy as np, nulldtype as nd

dt = nd.Nullable(np.float64)      # itemsize 8, NA is a bit pattern
a = np.arange(5.0).astype(dt)
a[2] = nd.NA

a                      # [0.0 1.0 NA 3.0 4.0]
a + a                  # [0.0 2.0 NA 6.0 8.0]

np.can_cast(np.float64, dt)               # False — one value is given up, so not "safe"
np.can_cast(np.float64, dt, "same_kind")  # True  — astype and assignment still work
np.can_cast(dt, np.float64)               # False — dropping NA has to be said

a.astype(np.float64)   # ValueError: cannot convert a missing value to dtype('float64')

a + 1.0                # [1.0 2.0 NA 4.0 5.0]   Python scalars work too
a + np.arange(5.0)     # plain arrays work too, both ways round

np.sqrt(a) * 2         # unary ufuncs work too
a.sum()                # NA          propagates, R's default
np.add.reduce(a, where=nd.notna(a))                # 8.0, skips NA, has to be said
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
s.tobytes()            # b'ab\x00\xff\xff\xff\x00\x00\x00'   the NA cell is all 0xFF
s[2] = b"\xff\xff\xff" # ValueError: ... is the value reserved to mean NA

u = np.array(["b", "a", "c"]).astype(nd.Nullable("U3"))
u[1] = nd.NA
np.sort(u)             # [b c NA]    NA sorts last

r = np.zeros(2, nd.Nullable(np.dtype([("a", "i4"), ("b", "f8")])))  # itemsize 12
r[0] = (1, 2.0); r[1] = nd.NA   # the NA cell: (0x80000000, 0x7FFFFFFFFFFFFFFF)
r == r                          # [True NA]
nd.Nullable(np.longdouble)   # Nullable(float64), with a LongDoubleWarning
nd.Nullable(">i4")           # Nullable(int32) — stored in native byte order
```

| Done | Notes |
|---|---|
| Parametric DType | `Nullable(f8)`, `Nullable(u1)`, `Nullable("U10")`, `Nullable(record)`... |
| Two layouts, one name | bit pattern for every numeric type, datetime, `S`/`U`/`V` and records; a flag byte when `FlagLayout(T)` is asked for — see `LAYOUTS.md` |
| Records | NA when every field holds its type's NA; nested records, subarray fields, `longdouble` fields; `==`/`!=` give `Nullable(bool)` |
| `S`, `U`, `V` | NA is the whole cell of `0xFF`, at any width; `sort`, `argmax`, `==`, `+` work |
| `longdouble`, `clongdouble` | stored as `float64` / `complex128`, with a `LongDoubleWarning` when precision is really lost |
| Byte order | always stored native: `Nullable(">i4") == Nullable("<i4")`, and big-endian data still casts in and out correctly |
| Flag layout: interleaved `{value, valid}` | one buffer, as numpy requires |
| `setitem` / `getitem` | both `nd.NA` and `None` mark a value missing |
| Flag layout: data under NA is zeroed | not observable, no garbage leaks |
| Cast `Nullable[T] -> Nullable[T]` | mandatory; numpy refuses a DType without it |
| 26 binary ufuncs | arithmetic, `maximum`/`minimum`, the 6 comparisons, `remainder`, `fmod`, `arctan2`, `hypot`, `logaddexp`, `copysign`, `nextafter`, `fmax`/`fmin`, `heaviside`, `float_power` — **borrowing T's loop**, only combining validity itself |
| Results follow numpy's rules | `resolve` asks the wrapped ufunc, so `Nullable[i8] / Nullable[i8]` is `Nullable[f8]` |
| Comparisons answer `Nullable[bool]` | `NA == NA` is `NA`, not `True` |
| `clip`, `matmul` | `clip` is a 3-input ufunc; `matmul` is a gufunc, including `v@A`, `A@v` and the batched forms |
| 39 unary ufuncs | all of trig/hyperbolic/log/exp, rounding, `isnan`/`isinf`/`isfinite`, `signbit`, `conjugate` |
| Reductions | `sum`/`prod`/`max`/`min`, **propagating by default**, like R's `na.rm=FALSE` |
| The `nd.*` statistics layer | `median`/`quantile`/`percentile`/`unique` — replacing the numpy functions built on `sort`, which return **wrong numbers** when a gap is present |
| Skipping NA in a reduction | `np.add.reduce(a, where=nd.notna(a))` — has to be said, but needs no `initial=` |
| `accumulate` | `cumsum`/`cumprod`/`cummax`, NA from the first gap onwards |
| `isna`, `notna`, `filled`, `to_numpy` | explicit ways out; `to_numpy` refuses to guess |
| `sort`, `argsort` | gaps last, two gaps compare equal — as in R |
| `nd.all`, `nd.any` | Kleene reductions; `np.all` cannot be used because it pins `dtype=bool` |
| `isnan`, `isinf`, `isfinite` | of NA answer NA — you cannot answer a question about a value you do not have |
| pickle | both the dtype and the array |
| `nonzero`, `count_nonzero` | work, and **refuse** on a gap rather than guess, at any array size |
| `argmax`, `argmin` | skip gaps, never return the position of one |
| `std`, `var`, `mean` | propagate NA |
| `np.zeros` / `np.empty` | zeros gives real zeros; empty gives NA in the flag layout, since uninitialised means no data |
| Kleene three-valued logic | `logical_and/or/xor`, `NA & False = False`, `NA \| True = True` |
| Nothing is computed on a gap | `where=` is handed to the wrapped ufunc when needed, so a gap raises no spurious warning while real errors still do |
| Cast `T -> Nullable[T]` | automatic: `same_kind` for a bit pattern (a value is given up), `safe` for a flag byte; the reserved value raises |
| Cast `Nullable[T] -> T` | UNSAFE, and **raises on a gap** — there is no implicit way out |
| Cast to another plain dtype | `Nullable[i4].astype(np.float64)`, to `int16`, to big-endian — numpy casts the values |
| Cast between wrapped dtypes | `Nullable[i8] -> Nullable[f8]`, gaps preserved |
| A cast never invents an NA | `int64 2**31 -> Nullable[i4]` or `S5 -> S3` truncating down to all `0xFF` raise, rather than quietly becoming a gap |
| Assigning numpy scalars | `np.void`, `np.bytes_`, `np.str_`, records — both layouts |
| `Nullable[T] op T` | a promoter, both ways round and with Python scalars |
| Value promotion | `Nullable[i8] + Nullable[f8] -> Nullable[f8]` |

Tests: 783 passed, and the whole suite also passes on an ASAN + UBSAN build.
Among them, `test_add_borrows_the_wrapped_loop` compares against
`np.int8(100) + np.int8(100)` to prove that the addition really is done by the
wrapped dtype, overflow behaviour included.

## Borrowing the loop, the core idea

Because storage is interleaved, the value field is a **strided** array of `T`.
numpy's loops already take strides, so wrapping each operand in a **zero-copy**
1-D view and handing it to `np.add` is all it takes:

```c
PyObject *a = value_view(wrapped, data[0], N, strides[0]);
/* ... call np.add(a, b, out=o) ... */
for (i = 0; i < N; i++)
    out_valid[i] = in0_valid[i] & in1_valid[i];
```

This one function serves every binary ufunc — in place of `numpy.ma`'s 50
hand-written wrappers.

And because `where=` is passed as the union of the two validity arrays, the
wrapped loop **never touches a gap**. So nothing needs to know any operation's
domain: `a / b` with `b` missing does not warn "divide by zero", but a real zero
in `b` still does.

## Two storage layouts, one name

`Nullable(T)` chooses for you, based on exactly one question: **does `T` have a
value worth taking over as NA?**

| | bit pattern | flag byte |
|---|---|---|
| itemsize | **equal to `T`** | `T` + 1 |
| Extra memory | **0** | +12.5% for 8 bytes, +100% for `int8` |
| Alignment | **intact** | broken |
| Faster | **2-3.5x** | — |
| Values given up | one | none |
| Usable types | every type with a spare value | every type |

```
bit pattern:      bool, int8..int64, uint8..uint64, float16/32/64,
                  complex64/128, datetime64, timedelta64, S<n>, U<n>, V<n>,
                  records of the above
flag byte:        FlagLayout(T); chosen automatically only when there is no
                  cell to fill (S0, a bare subarray)
stored as double: longdouble, clongdouble
refused:          object, StringDType
```

To force a specific layout for benchmarking or comparison, call
`nd.BitpatternLayout(T)` / `nd.FlagLayout(T)`; `astype` converts between them.
Beyond those two uses there is no reason to care.

### Which value each type gives up

**`int`.** `INT_MIN` — a signed range is already lopsided by one, as in R.

**`uint`.** `UINT_MAX`, the mirror of `INT_MIN`. An unsigned type uses all 2ⁿ
patterns, so a real number has to go; with `uint8` that is 255, often a white
pixel. In return the itemsize stays the same and numpy's SIMD loops run
unchanged. A bias shift (store `x - 2ⁿ⁻¹`, reserve `INT_MIN`) is worse twice
over: it gives up 0 instead of `UINT_MAX`, and the cells live in a shifted
frame, so a borrowed loop adds the right number in the wrong frame —
`200 + 100` lands on the cell that means 172.

**`float`, `complex`.** Every bit except the sign; see the *quiet* NaN section
below.

**`S`, `U`, `V`.** The whole cell of `0xFF`, at any width. For `U` that is the
code unit `0xFFFFFFFF`, which is not a Unicode code point — `chr()` refuses
it — so **no string is given up**. For `S` and `V` it is a legal byte string,
given up the way `UINT_MAX` is; but `0xFF` never occurs in ASCII or UTF-8. The
**whole cell** must be checked: `b"\xffab"` is an ordinary value, so looking at
the first byte alone is wrong. A cell holding a value almost always stops at its
first byte that is not `0xFF`, so this is still fast.

**Records.** Each field holds its own type's NA: a missing `(i4, f8, S3)` holds
`(INT32_MIN, 0x7FFF…, 0xFFFFFF)`, with padding bytes `0x00`. A record is missing
only when **every** field holds NA; one field with a value is enough for the
record to have a value, so the only thing given up is the combination in which
every field was already a reserved value.

Filling the whole cell with `0xFF` does not work. It gives up `(-1, -1)` for two
`int32` fields, a very common old-style "missing" value. Worse, the fields of a
missing cell would read back as `-1`, `True` and `1969-12-31` for `int32`,
`bool` and `datetime64` — **looking like real data**, exactly the kind of leak
`numpy.ma` has. Field by field, a field taken out of a missing cell is still its
own type's NA.

When the dtype is created, a record is flattened into leaf fields `(offset,
repeat count, NA pattern)`: nested records and subarray fields are unrolled,
and padding belongs to no leaf, so it never decides anything. NA is per record,
not per field: per-field NA would need a scalar type of its own, because
`np.void` cannot hold `NA`. The byte layout would be the same either way, so
adding it later could still read old data. A bare subarray dtype, outside a
record, still uses the flag layout.

**`longdouble`, `clongdouble`.** No counterpart of their own: stored as
`float64` and `complex128`, inside records too — the record is rebuilt, the
later fields move up, and numpy's cast moves the values in field order. A real
`long double` differs from place to place — x87 80-bit on x86-64, IEEE quad on
aarch64, `double` under MSVC — so keeping it would mean three float formats for
one dtype. About 3 digits are lost and the range becomes ±1.8e308;
`LongDoubleWarning` fires exactly when something is really lost, and stays
silent where `long double` already is a `double` (Windows).

**`object`, `StringDType`.** Refused: they hold references, and Nullable does not
do reference counting yet.

### Padding the flag layout was tried, and dropped

A `stride` of 9 is the worst case for vector units: nothing aligns, and SIMD
loops fall back to scalar code. So padding to 16 sounds very reasonable.
Measured on `uint64 add`:

| n | 9 bytes | 16 bytes | padding faster by |
|---|---|---|---|
| 50,000 | 135 µs | 111 µs | 1.21x |
| 500,000 | 1.98 ms | 3.75 ms | **0.53x** |
| 5,000,000 | 22.5 ms | 34.9 ms | **0.64x** |
| 20,000,000 | 86.6 ms | 137 ms | 0.63x |

In cache it is 4-21% faster; out of cache it is **1.6-1.9x slower**, plus 78%
more RAM. The slowdown tracks the 16/9 = 1.78 ratio closely, so it is purely
memory bandwidth — no tuning can rescue it. Padding buys a little where things
are already fast and costs dearly where speed really matters, so the tight
layout stays.

A warning about measuring: the first microbenchmark used `np.add` on
separately-strided arrays and gave the **opposite** result. It was wrong
because `np.add` goes through nditer, and nditer **buffers the data into a
contiguous block before calling the loop** — it measured the buffering, not the
inner loop. Only measuring through the dtype itself gives the real number.

**The semantics of the two layouts are identical**, and a test suite running on
both in parallel checks that: NA propagation, comparisons giving NA, Kleene
three-valued logic, propagating reductions, `skipna` having to be said,
`accumulate`, sorting NA last, no implicit way out, `nan` distinct from `NA`.
The same claims, two storage layouts. There are two visible differences:
`np.can_cast(T, Nullable(T))` is only `same_kind` for the bit pattern because a
value is given up, and only the bit pattern does `==` on records — the flag
layout raises `TypeError`.

### Which value is the reserved NA

| dtype | NA |
|---|---|
| `float16`, `float32`, `float64` | every bit except the sign: `0x7FFF`, `0x7FFFFFFF`, `0x7FFFFFFFFFFFFFFF` |
| `complex64/128` | the float pattern in **both halves**, read from the real half |
| `int8` … `int64` | `INT_MIN` — as in R |
| `uint8` … `uint64` | `UINT_MAX` |
| `bool` | the byte `2`, a value numpy never produces |
| `datetime64`, `timedelta64` | `INT64_MIN` — **for free** |
| `S<n>`, `U<n>`, `V<n>` | the whole cell of `0xFF` |
| record | each field its own type's NA, padding `0x00` |

Datetime is the strongest case: numpy **already** reserved exactly `INT64_MIN`
for `NaT` long ago. No legal value is taken away, and `filled()` gives back a
real `NaT` rather than an invented date. In other words numpy already uses
bit-pattern NA, just for exactly two types — the rest of this is generalising
it.

Every pattern is written in native byte order, and the dtype is always stored
native too: see the segfault section below on `>i4`.

### Not relying on the hardware

x86 hardware does propagate NaN payloads, but when both operands are NaN **the
left operand wins** — so `NA + nan` keeps NA while `nan + NA` loses it. That is
an x86 convention, not IEEE 754, and some architectures canonicalise NaN and
wipe the payload entirely.

So the loops **do not rely on propagation**. They read the inputs and write NA
into the output themselves:

```
NA + nan  ->  [NA nan]
nan + NA  ->  [NA nan]     both ways round, and a plain nan stays nan
```

Floats have a shortcut: after the operation NaN has already propagated, so
**one pass over the output**, while it is still hot in cache, is enough — an
array with no NA stops right there. With `int`, `INT_MIN` does not propagate at
all, so the inputs have to be scanned.

## Performance

`a + a` with `f8`:

| n | numpy | flag byte | bit pattern | bit pattern vs numpy |
|---|---|---|---|---|
| 1,000 | 0.6 µs | 6.4 µs | **3.3 µs** | 5.9x |
| 8,192 | 2.1 µs | 29.6 µs | **8.4 µs** | 4.1x |
| 100,000 | 20.6 µs | 271 µs | **90 µs** | 4.4x |
| 1,000,000 | 860 µs | 4.0 ms | **1.8 ms** | 2.0x |
| 5,000,000 | 5.1 ms | 21.2 ms | **14.1 ms** | 2.7x |

The bit pattern is **2-3.5 times** faster than the flag byte and only 2-4.4x
behind plain numpy, while the flag byte is 4-13x behind. Almost all of the bit
pattern's remaining gap is the NA scan — one extra read over data just written.
On small arrays the fixed cost of `get_loop` (two Python calls per operation)
dominates, and a cache could remove it.

### Two changes that took the flag layout from 48x to 12x

**Borrowing `T`'s C loop instead of calling a Python ufunc per chunk.**
`NPY_METH_get_loop` runs once per operation, and there we ask the wrapped ufunc
for its C function pointer through the `numpy_1.24_ufunc_call_info` capsule
(`_resolve_dtypes_and_context`, then `_get_strided_loop`). The per-chunk loop
then never touches Python: it scans for consecutive runs of values and hands
each run straight to `T`'s loop. A 15-30% gain.

**Block copies instead of per-element `memcpy`.** The `Nullable → Nullable` cast
used to `memcpy` 9 bytes at a time, N times. When both sides are contiguous, a
single `memcpy` of the whole block is enough. This change alone made `copy()`
2.6x faster — and, surprisingly, made `a + b` twice as fast, because ufuncs
**buffer the operands** of a user dtype, so every buffer fill goes through that
very cast.

Lesson: the time is not spent where you guess. Each layer — copy, cast, loop —
has to be measured on its own; the total does not show it.

### Tried: several cells sharing one flag byte, and padding for alignment

**8 values + 1 shared bitmap byte: numpy does not allow it.** A dtype says
element `i` lives at `data + i*itemsize` and must carry everything about
itself. A shared flag byte breaks as soon as `a[3]`, `a[1::2]` or `a[5:]`
happens — a pointer to one element does not say where the shared byte is. That
is exactly why Arrow and pandas use **two separate buffers**, which a numpy
dtype cannot describe.

**Padding the itemsize to 16 so values align: possible, but a loss.** It turns
out the penalty does not come from "not contiguous" but from **misalignment**:

| n | contiguous 8 | stride 9 | stride 12 | stride 16 |
|---|---|---|---|---|
| 8,192 | 3.4 µs | 12.7 | 12.4 | **3.8** |
| 100,000 | 19.8 µs | 135 | 135 | **45.6** |
| 1,000,000 | 910 µs | 2376 | 2864 | 1929 |

Stride 12 is misaligned too, so it is slow too; stride 16 keeps `float64` on an
8-byte boundary, so SIMD can load it. But measured through the dtype itself:

| n | stride 9 | stride 16 |
|---|---|---|
| 1,000 | 5.9 µs | **4.7 µs** |
| 8,192 | 26.8 µs | **20.1 µs** |
| 100,000 | 262 µs | **231 µs** |
| 1,000,000 | **4.1 ms** | 7.2 ms |
| 5,000,000 | **22.6 ms** | 40.1 ms |

12-25% faster while the array fits in cache, **78% slower for large arrays**,
and 78% more memory. Large arrays are bound by memory bandwidth, and padding
moves 16 bytes for every 8 useful ones. So the `value + 1` layout stays.

### `maximum` and `minimum` in the flag layout skip the fast path

The loop obtained through the capsule for these two used to return wrong
results at stride 9. The root cause is a numpy bug; see "A numpy bug found along
the way". The bit-pattern layout now takes the fast path; the flag layout still
goes through Python — slower, but correct.

### Tried: can hand-written SIMD save it

No. Measured on 100k elements:

| Approach | Time |
|---|---|
| Adding directly at stride 9 | 172 µs |
| Only copying stride → contiguous (one array) | 41 µs |
| Adding contiguous | 34 µs |
| Gather to contiguous, compute, scatter back | **149 µs** — only 13% better |

The three copies take 123 µs, more than the addition itself. The problem is not
the instructions but **how memory is touched**: whether deinterleaving with
AVX-512 `vpermb` or with gather, you still touch 9 bytes to get 8 useful ones,
you are still misaligned, and a cache line still does not load as one vector.
The cleverest hand-written code might reach 1.8x; it cannot win back 6-10x.

Conclusion at the time: the fixable part was the 4-7x from Python calls; the
flag layout's 6-10x is inherent, and removing it means switching to a bit
pattern and accepting every trade-off that comes with it. **That is why the bit
pattern is now the default**, and no type picks the flag byte automatically any
more, except a few dtypes with no cell to fill.

All of this follows from the constraint that **a numpy array has one buffer**:
the dtype has to squeeze the validity flag into the same cell as the value.
Arrow and pandas are not affected because they keep two separate buffers — a
contiguous value array and a separate validity bitmap.

Within numpy's dtype framework only one way out remains: **bit-pattern NA**,
borrowing one of `T`'s own values as the marker. The itemsize stays the same,
the array stays contiguous, SIMD stays intact, and no extra bandwidth is spent.
The price is one legal value given up, and for floats NA is hard to tell apart
from NaN — exactly what the two sides argued about in NEP 12 and NEP 24. The
measurements above are empirical evidence for the bit-pattern side.

## Known limitations

- **Arithmetic can still produce the reserved value.** Casts are guarded, but
  `int32` `2**31-1 + 1` wraps exactly onto `INT_MIN` and quietly becomes NA;
  with strings, `np.strings.add` of two all-`0xFF` cells also gives an all-`0xFF`
  cell.
- **`repr` of a `U` array holding an empty string raises `ValueError`**, in both
  layouts. `arrayprint` prints non-numpy dtypes with `str()`, `str('')` is
  empty, and `_extendLine_pretty` calls `max()` on an empty list of lines. A
  numpy bug.
- `sort`, `argmax` and `nonzero` hold the GIL for the whole loop, because the
  dtype has to declare `NPY_NEEDS_PYAPI` (see the segfault section). Large
  arrays get no multithreading benefit.
- `maximum` / `minimum` take the fast path only when every stride is a multiple
  of the element size — that is, **the bit pattern does, the flag byte does
  not** (see the numpy bug below).
- The loops still declare `NPY_METH_REQUIRES_PYAPI` because the fallback path
  needs the GIL.
- `np.dot` **cannot be implemented**; see below. Use `@` (`matmul`), which works.
- `a + 1.0` is about 2.6 times slower than `a + a` in both layouts — the scalar
  operand has stride 0 and apparently drops off the borrowed-loop path. Not yet
  traced; results are still correct.
- `np.unique` fails cleanly; use `nd.unique`.
- `np.array([(1, 2.0), nd.NA], dtype=nd.Nullable(rec))` raises a shape error:
  numpy treats a tuple as one element only when the requested dtype is a plain
  record, and with this dtype it is a sequence. Create with `np.zeros` and
  assign, or cast from a plain record array.
- Records support only `==` and `!=`; `+`, `sum`, `<` raise as they do on plain
  records.
- `nd.Nullable("S")` (no length yet) has no cell to fill, so it falls back to
  the flag layout, itemsize 1.
- `bitwise_and` / `or` / `xor` are **deliberately not registered**: on
  `Nullable[bool]` they must follow Kleene, on `Nullable[int]` they must
  propagate. One loop cannot do both without branching on the wrapped type, and
  silently shipping the wrong truth table is exactly what this prototype exists
  to avoid.
- `np.all` / `np.any` do not work: `fromnumeric` forces `dtype=bool`, so the
  ufunc finds no loop. Use `nd.all` / `nd.any`, or `np.logical_and.reduce`. The
  same reason breaks `np.array_equal`.
- `np.histogram` and `np.einsum` fail cleanly — both need a boolean decision
  from data that may be missing, and `bool(NA)` refuses to give one.
- Indexing with a mask that holds NA raises `IndexError` in numpy. This is
  **intended** — you have to say what NA should count as — but it takes
  `.filled(False)` to do that tidily.

### Pitfall: functions built on `sort` return silently wrong results

`np.median`, `np.percentile` and `np.quantile` are written in Python on top of
`sort`. NA sorts last, so they **take NA for the largest value** and return a
plausible-looking number:

```python
a = [3.0, 1.0, 5.0, NA]
np.median(a)    # 4.0   -- wrong, that is the mean of 3 and 5
```

Nothing in the dtype can stop it: removing `compare` would kill `sort` too. This
is why pandas rewrote its own `median` instead of calling numpy — and why there
is a statistics layer here too:

```python
nd.median(a)                # NA    propagates, the default
nd.median(a, skipna=True)   # 3.0   correct
nd.quantile(a, [.25, .75])  # [NA NA]
nd.unique(a)                # NA is a value of its own, sorted last
```

`nd.sum/prod/min/max/mean/std/var/median/quantile/percentile` all take the same
`skipna=`, instead of spelling out `where=nd.notna(a)`. A reduction over nothing
gives **NA, not 0** — a zero there would be a number invented out of thin air.

`np.unique` is also unusable, but for a different reason and it **fails
cleanly**: it compares neighbours in the sorted array and assigns into a plain
bool mask, which NA refuses to become.

## Segfaults found while moving `S`/`U`/`V` to bit patterns

The type change itself took a few dozen lines; most of the effort went into
**five existing C bugs** that no test had reached before. Each now has its own
test.

**`compare` and `nonzero` pass a NULL array.** These two slots call the wrapped
dtype's function with `ap = NULL`. The numeric functions ignore `ap`, so no
harm, but `STRING_compare` and `UNICODE_compare` read the item size from `ap`,
and `VOID_compare` reads the field list — `sort`, `argmax` and `nonzero` on
`S`/`U`/`V`/records all segfaulted, **in both layouts**. `S`/`U`/`V` are now
compared directly (`memcmp`, code unit by code unit), and records use a fake
array on the stack — the same trick `VOID_compare` uses for each field.

**`nonzero` on a large array with NA.** The `nonzero` slot raises on NA. But for
arrays over 500 elements numpy releases the GIL before the loop unless the dtype
declares `NPY_NEEDS_PYAPI`, so setting an exception there was a segfault — for
**every** type, `float64` included. `VOID_compare` also needs the GIL (it asks
the memory handler for a buffer), and numpy's own records carry that flag. Both
layouts now declare `NPY_NEEDS_PYAPI`.

**`astype` to a different width.** The cast from a bit pattern to a plain dtype
copied the source's byte count as is: `Nullable[i4].astype(np.float64)` left the
upper half of every `double` as garbage, and `.astype(np.int16)` **wrote past
the end of the buffer**. The flag layout refused outright. Both now check for NA
first and let numpy cast the values.

**Non-native byte order.** `Nullable(">i4")` kept the `>`, but the NA pattern
was written native, so NA was the bytes `00 00 00 80` — read big-endian that is
**128**: the number 128 was refused while `INT_MIN` slipped in as an ordinary
value. Worse, `np.sort` calls `PyArray_DescrNewByteorder` to get a byte-swapped
copy of the dtype; for a new-style DType that returns `NULL`, and numpy uses it
**without checking** — a segfault in both layouts. Every dtype is now stored
native; big-endian data goes in and out through numpy's casts.

**Assigning `np.void` into the flag layout.** Assigning a numpy scalar is a cast
from the scalar's dtype. The flag layout had no casts for `S`/`U`/`V`, so
`np.bytes_` and records were refused, and an unstructured `np.void` segfaulted
inside numpy. Both layouts now register casts in and out for those three types.

In the same round: `np.dtype("S3")` is not a singleton — each call is a new
object — so every dtype comparison by identity (`wrapped == value`) missed for
strings and pushed data down the slow cast path. They now compare with
`PyArray_EquivTypes`.

## `==` on records used to be silently wrong

numpy has no ufunc loop for comparing records; `==` on plain record arrays takes
a separate branch in `array_richcompare`, keyed on `type_num == NPY_VOID`. When
`==` or `!=` finds no loop, numpy **swallows** `UFuncNoLoopError` and returns an
array of all `False` (or all `True`), with no warning. So in both layouts:

```python
a == a    # [False False False]   -- wrong, and nothing says so
```

The bit pattern now has its own resolver for `equal`/`not_equal` on records: the
loop calls numpy's own `==` on two value views and then stamps NA. Records of
different structure raise `TypeError`. The flag layout is about to be removed,
so it only raises a clear `TypeError` — numpy swallows only
`UFuncNoLoopError`, other errors still come out. Every other ufunc on records
still raises, because only `==` and `!=` are swallowed.

## A numpy bug found along the way

`maximum`/`minimum` used to return wrong results when called through the
borrowed capsule loop. At first it looked like "the last element is skipped";
traced to the end, the cause was entirely different and lives in numpy, not in
this dtype.

`loops_minmax.dispatch.c.src` converts byte strides to element strides by
dividing `is1 / sizeof(STYPE)`, after asking `npyv_loadable_stride`. But in
`numpy/_core/src/common/simd/simd.h`, the divisibility check is wrapped in an
extra condition:

```c
if (alignof(npyv_lanetype_##SFX) != sizeof(npyv_lanetype_##SFX) &&
        stride % sizeof(npyv_lanetype_##SFX) != 0) {
    return 0;
}
stride = stride / sizeof(npyv_lanetype_##SFX);
```

For `double` on x86-64, `alignof == sizeof == 8`, so **the left side is false
and the divisibility check never runs**. The division then truncates: stride 9
becomes 1, stride 12 also becomes 1, and the SIMD loop scans the array as if it
were contiguous — reading both inputs wrong and writing the output wrong.

The very commit that added this (`94854dbc77`, "BUG: Fix simd loadable stride
logic") states the opposite intent: *"Strides are now guaranteed to be a
multiple and compatible"*. The `alignof != sizeof` condition disables exactly
what it just added.

numpy itself is not affected because it buffers the data into a contiguous
block before calling the loop. Only a third party calling
`ufunc._get_strided_loop(..., fixed_strides=...)` directly — a public API since
1.24, and exactly what this prototype uses — hits it.

A minimal ctypes reproducer, with no custom dtype involved:
`scratchpad/minmax_raw.py`. The proposed patch (dropping the `alignof` clause)
is in `numpy-simd-stride.patch`. **Not yet reported upstream.**

On the prototype side it was enough to check divisibility ourselves instead of
excluding those two ops outright: the bit pattern (itemsize 8) can now use the
fast path, the flag byte (itemsize 9) still cannot.

| n | `add` | `maximum` |
|---|---|---|
| 100,000 | flag 12.9x · bit pattern 3.4x | flag 26.6x · **bit pattern 3.1x** |
| 5,000,000 | flag 2.7x · bit pattern 1.6x | flag 4.9x · **bit pattern 1.4x** |

## Why the float NA is `0x7FFF…` and a *quiet* NaN

R uses `0x7FF00000000007A2` for `NA_real_` (0x7A2 = 1954, the year Ross Ihaka
was born). That pattern is a **signaling NaN** — the top mantissa bit is 0. R
gets away with it because it never reads the IEEE flags; numpy does, and turns
it into `RuntimeWarning: invalid value encountered in add` every time an
operation merely *passes over* a gap. Missing data is not an arithmetic error,
so it must not be reported as one.

The first version kept R's payload and set the quiet bit. The current version
drops the payload and uses **every bit except the sign**: `0x7FFF`,
`0x7FFFFFFF`, `0x7FFFFFFFFFFFFFFF`.

- **One rule for every width**, the same "all ones" idea as `UINT_MAX` and the
  all-`0xFF` string cell. `float16` no longer has to squeeze a payload into 10
  bits.
- **Still a quiet NaN** (exponent and top mantissa bit both set), so passing
  over a gap raises no `FE_INVALID`.
- **Arithmetic never produces it**: `np.nan` is `0x7FF8000000000000`, and the
  NaN x86 produces for `inf - inf` is `0xFFF8000000000000`.
- **The sign bit is ignored when reading**, so a gap whose sign a borrowed loop
  flipped is still NA.

## Two traps in borrowing someone else's loop

Going from 14 to 26 binary ops exposed two hidden assumptions the first 14 never
touched. Both affect only the bit-pattern layout, and both are **silently
wrong**.

**`fmax`/`fmin` swallow NaN.** The bit-pattern fixup has an optimisation: scan
the output for NaN, and if there is none, there is nothing to fix — relying on
the hardware having put NaN in every cell that was NA. `fmax` deliberately
returns the *non*-NaN side, so the gap vanishes from the output and the scan
skips exactly the row that needs fixing: `fmax(NA, 2.0)` gave `2.0`. NA is not
NaN — it propagates whatever the op does with NaN.

**`<` and `>` are *signaling* comparisons.** IEEE 754 requires them to raise
`FE_INVALID` even for a quiet NaN, while `==` and arithmetic stay quiet —
verified in `scratchpad/qnan.c`. The bit-pattern layout lets the loop run
straight over NA cells, so any op that compares its operands internally
(`logaddexp`, `logaddexp2`) warned `invalid value encountered` about data that
**does not exist**. They have to take the `where=` path.

Both lists are empirical, not derivable, so the test suite sweeps *every* op,
taking the names straight from the C table: an op missing from a list turns a
test red instead of showing the user a strange warning. Checked by mutation —
disabling each guard in turn turned 2, 4 and 10 tests red.

## A second numpy bug: an assertion too strict for non-legacy DTypes

`Nullable[f8]` **in the flag layout** plus a Python literal, on an array larger
than `NPY_BUFSIZE` (8192), makes the numpy dev build **abort**:

```
can_cast_pyscalar_scalar_to: Assertion `NPY_DT_is_legacy(NPY_DTYPE(to))' failed.
```

All three are needed: a non-legacy DType, the other operand a **Python literal**
(writing `np.float64(1.0)` does not trigger it), and an array split into several
buffered chunks — raising `np.setbufsize` above the array size makes it go
away. The bit-pattern layout is not affected because itemsize 8 is aligned, so
no buffering is needed.

But reading further into that function, **the assertion is what is wrong, not
the code below it**. The three shortcuts at the top of the function
(`ISCOMPLEX`/`ISFLOAT`/`ISINTEGER`) all key on `type_num` and all return `false`
for this descriptor (`num = -1`); the final branch is generic — it builds a
descriptor for the scalar and calls the general `PyArray_CanCastTypeTo`, which
works with any DType.

Verified: removing just the `assert` line and rebuilding numpy, every array size
works and the results match plain numpy. The patch is in
`numpy-pyscalar-assert.patch`.

It only affects builds with assertions enabled; in released numpy `assert` is
compiled out and the generic path was already correct. **Not yet reported
upstream.**

## Next steps, in order

1. **Consider reporting the numpy bugs upstream.** Four candidates:
   - the SIMD stride — has a patch and a reproducer;
   - the `can_cast_pyscalar_scalar_to` assertion — has a patch and a reproducer;
   - `sort` does not check for `NULL` after `PyArray_DescrNewByteorder`;
   - `repr` of a non-numpy dtype array crashes when an element's `str()` is
     empty.
2. **Guard against the reserved value produced by arithmetic**, not only by
   casts: `int`/`uint` overflow, string ops producing an all-`0xFF` cell.
3. **Real `longdouble`** through `numpy-quaddtype` instead of substituting
   `float64`.
4. **Remove the flag layout.** No common type chooses it any more; every C bug
   in the last session had to be fixed twice, and the test suite runs twice as
   long. Keep it one more round to compare records across the two layouts, then
   drop it.

## Three-valued logic

`logical_and` and `logical_or` do not just propagate NA — sometimes the result
is certain even with one side missing:

```
   AND |  T    F   NA          OR  |  T    F   NA          XOR |  T    F   NA
     T |  T    F   NA            T |  T    T    T            T |  F    T   NA
     F |  F    F    F            F |  T    F   NA            F |  T    F   NA
    NA | NA    F   NA           NA |  T   NA   NA           NA | NA   NA   NA
```

The same as SQL, R and pandas. These are the only three ufuncs that need a loop
of their own, because the validity of the result depends on the *values*, not
only on the validity of the inputs.

## Flag layout: why the bytes under NA are always 0

NA means *no value*, so those bytes are filler, not data. We overwrite them with
0 and never read them back. This is what `numpy.ma` does not do, and it is the
root of nearly all its bugs:

- Every `numpy.ma` leak bug ends with the value under the mask reaching the user
  **looking like real data**. A canonical zero fools nobody.
- Deterministic results: keeping whatever was computed would make `tobytes()`,
  hashes and pickles depend on evaluation order and compiler optimisations.
- A value the user has masked does not travel with the array into files or over
  the network.
- Because the loop passes `where=`, a gap is **never computed**; without the
  memset the cell would keep the garbage of an uninitialised output buffer.

The bit-pattern layout reaches the same goals another way: the bytes of a
missing cell **are** the NA pattern, so it is just as deterministic and keeps no
old value either; the padding of a missing record is written as `0x00` too. R
chose exactly that for floats; here it is the default for every type.

## Why there is `matmul` but no `np.dot`

`A @ B` works: `matmul` is a gufunc, going through the new DType API, so the
loop gets both the descriptors and the core dimensions. `C[i,j]` is a sum over
`k`, so it is missing exactly when row `i` of A **or** column `j` of B has a gap
anywhere — decided per row and per column rather than per cell, i.e. two scans
of size `n` and `m` instead of `n*m`.

`np.dot` does not, and not out of laziness. It goes through the old `dotfunc`
slot:

```c
void dotfunc(void *ip1, npy_intp is1, void *ip2, npy_intp is2,
             void *op, npy_intp n, void *arr);
```

`arr` is the only way to reach the descriptor, and only the descriptor gives
`wrapped->elsize` — that is, where the flag byte is. numpy passes **NULL**
there:

```c
dot(it1->dataptr, is1, it2->dataptr, is2, op, l, NULL);
/* multiarraymodule.c, PyArray_MatrixProduct2 */
```

I did implement it as a test and it segfaulted right on the line reading `arr`.
On top of that, numpy releases the GIL around the loop, so calling back into
Python to look it up is not possible either. This ABI predates parametric
DTypes, and a parametric dtype cannot use it. The slot stays empty, and numpy
says `dot not available for this type` — a clean, correct error.

## The `NA` scalar

Indexing returns a plain Python value if the cell has data, and the `NA`
singleton if not. Anything someone does with `a[0]` sooner or later someone will
do with `a[1]`, so the singleton has to behave rather than blow up.

```python
NA + 1      # NA        every operation touching NA gives NA
NA > 1      # NA
NA == NA    # NA        two unknowns are not equal just because both are unknown
a[1] is NA  # True      identity still answers, and NA is still hashable
bool(NA)    # TypeError
float(NA)   # TypeError
```

`bool(NA)` **used to return `True`** — Python's default for an object with no
`__bool__` — so `if a[i]:` quietly treated a hole in the data as "yes". There is
no right answer to "is the value you do not have true or false", so it has to
refuse rather than invent one. R says *"missing value where TRUE/FALSE needed"*,
and `bool(pd.NA)` raises too.

Fixing this also turned `np.histogram`'s confusing `TypeError` about
`NAType.__sub__` into a message that names the actual problem.

## Reducing over nothing

An empty array and an array that `skipna` emptied are **the same situation**, so
they have to give the same answer — at one point they did not, and that was a
bug.

| | |
|---|---|
| `sum` → `0`, `prod` → `1` | there is an identity element, so this is the right answer rather than an invention; R and numpy agree |
| `mean`, `median`, `min`, `max`, `std`, `var`, `quantile` → **NA** | no identity element: the mean of no numbers is *unknown* |

`NaN` must **not** be returned here. In this dtype NaN is an ordinary value that
a column may legitimately hold, so returning NaN would say "the answer is the
number not-a-number" rather than "unknown". `nd.median` of an empty array used
to return exactly that, with two `RuntimeWarning`s attached.

A note in an older test of mine claimed *"R returns NA for
`sum(c(NA,NA), na.rm=TRUE)`"* — **wrong**, R returns `0`. Fixed.

## `argmax` on an all-NA array

Returns `0`, exactly what numpy returns for an array of all `NaT` or all `nan`.
Raising would be better, but this slot cannot: `_PyArray_ArgMinMaxCommon`
**ignores the return value** and never looks at the error state. Originally the
loop also ran with the GIL released, so touching the error state there was a
segfault — tried, and it was. The dtype now declares `NPY_NEEDS_PYAPI` so the
GIL is held, but an exception set there would still only surface as a
`SystemError` instead of a message. Nothing is invented either way:
`a[a.argmax()]` is NA.

Same family of limitation as `dotfunc`: an old ABI with no channel for errors.

## Checking for memory leaks

`scratchpad/leakcheck.py`. Measured with `sys.getallocatedblocks()` — the
number of blocks currently allocated by CPython's allocator, so a leaked
`PyObject` shows up immediately as `+1`. Far more precise than RSS; my first
attempt used RSS and it gave two false alarms.

At **N = 40,000 iterations**, a leak of 1 block per call would show `+40,000`.
The largest number measured was `+62`, i.e. arena noise. Alongside, we track
`sys.getrefcount` of the **shared** objects — descriptors, `wrapped`, `NA`,
each ufunc, the DType class itself — because leaking a reference to them
allocates no new block, so block counting cannot see it. None of them grew.

This covers the fast path, the slow path with `where=`, the reduction path, the
gufunc path, and 8 **error** paths (where refcount bugs like to hide in
`goto fail` branches).

**The checker is validated by mutation.** Deliberately removing
`PyMem_Free(rowna/colna)` in `matmul` and removing `Py_DECREF(res)` in the
bit-pattern slow path:

```
FlagDType    M @ M              blocks  +40000    (2 blocks per call, exactly 2 PyMem_Free)
BitpatternD  logaddexp (slow)   blocks  +20001    (1 block per call, exactly 1 Py_DECREF)
```

Both are caught immediately. Without this step, "no leaks" would just be a
claim.

**The `S`/`U`/`V` round**: `scratchpad/leak_suv.py` measures 18 new paths —
`sort`/`argmax` on `S`/`U`/`V`/records, five raising paths (reserved value, a
cast producing NA, `nonzero` on a large array), width-changing casts,
big-endian, assigning `np.void` and records. At 3,000 iterations the largest is
`+17`; the two biggest outliers, measured again at 12,000 and 48,000
iterations, stay flat, i.e. they do not grow with the number of calls.

**The records round**: `scratchpad/leak_records.py` measures 13 paths —
creating and destroying descriptors (allocating and freeing the leaf-field
list), two refusal paths (an object field, a record of nothing but padding),
rebuilding a record containing `longdouble`, assignment, casting in and out,
`sort`/`argmax`, `nonzero` raising, `isna`/`filled`. Plus three comparison
paths (`==`/`!=`, records of different structure, the flag layout raising). The
largest is `+33` at 3,000 iterations, flat again at 12,000 and 48,000; the
refcounts of the dtype and of `NA` did not move.

**ASAN + UBSAN.** A separate build with `-fsanitize=address,undefined`, running
all 783 tests and `leak_records.py`: not a single report. This is how the
`na_bytes[8]` 8-byte over-read with `complex128` was caught, before it was
widened to 16.

## Settled design notes

- NA **has a type**: `Nullable[i2]` is not `Nullable[f8]`, and there is no shared
  singleton.
- **The bit pattern is the default**; a flag byte only when `FlagLayout(T)` is
  asked for by name.
- A record is **missing when every field is**, each field holding its own
  type's NA.
- The wrapped dtype is **always stored in native byte order**; `long double` is
  stored as `double`.
- Reductions **propagate by default**, `skipna=True` to skip.
- Coercion to a Python scalar **raises**; use `.filled(x)` to get out.
- **A cast never quietly creates an NA**, and never quietly drops one.
- The data under an NA is **not observable**.
