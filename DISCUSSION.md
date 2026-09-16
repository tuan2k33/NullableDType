# Nullable[T] — discussion

How the prototype works and why it is built this way: the borrowed loops, the
layout decision, performance, the bugs found along the way and how memory
safety is checked. `README.md` covers using it.

## Borrowing the loop, the core of it

The values of a `Nullable[T]` array **are** a contiguous array of `T`, so each
operand is wrapped in a **zero-copy** 1-D view and handed to numpy's own loop;
all this code does is stamp NA onto the output:

```c
PyObject *a = value_view(wrapped, data[0], N, strides[0]);
/* ... call np.add's C loop on a, b, o ... */
for (i = 0; i < N; i++)
    if (is_na(in0[i]) || is_na(in1[i]))
        put_na(out + i);
```

One function covers every binary ufunc — instead of the 50 hand-written
wrappers in `numpy.ma`.

For floats there is a shortcut: after the operation the hardware has already
propagated a NaN, so **one pass over the output** while it is still hot in cache
says whether there is anything to fix at all, and an array without gaps stops
there. For `int`, `INT_MIN` propagates nothing, so the inputs have to be scanned.

Both scans run in blocks of 2048 elements and are written so the compiler
vectorises them at `-O2`: integer bit tests instead of float compares, and no
branches. For floats, a block whose output holds no NaN or inf is skipped
outright; for integers, a block costs one pass that ORs "is NA" over both
inputs and "landed on the reserved value" over the output. Only a block that
holds a gap gets a second, blending pass, so one gap costs one block, not a
second pass over the whole array. The earlier version branched per element and
scanned every input as soon as the output held a single NaN, and
`-fopt-info-vec` showed that none of these loops vectorised.

A few operations cannot run straight over a gap, and they take a `where=` path
so the wrapped loop never touches it: see "Two traps in borrowing someone
else's loop".

## One layout

There used to be two. The surviving one is the **bit pattern**: NA is a reserved
value of `T` itself, so the item size is unchanged, the array is contiguous,
alignment is intact and numpy's SIMD runs as usual. The price is one legal value.

The other one was the **flag byte**: the value followed by a validity byte,
`T + 1` bytes. It gives up no value, but it breaks alignment and runs 2-3.5x
slower. It existed for the types with no value worth giving up. Once `S`, `U`,
`V` and records all got bit patterns, nothing common needed it any more, while
every C bug had to be fixed twice and the test suite ran twice. It lives in
`archive/flag-layout/`, together with all its measurements and layout
experiments.

This follows from the constraint that **a numpy array has exactly one buffer**:
a dtype that wants a validity flag has to put it in the same cell as the value.
Arrow and pandas avoid this by keeping two separate buffers — a contiguous value
array and a separate validity bitmap. Within numpy's dtype model, the bit
pattern is the only way to keep the performance, and its price — one value gone,
and for floats NA being hard to tell from NaN — is exactly what the two camps
argued about in NEP 12 and NEP 24.

### What each kind gives up

**`int`.** `INT_MIN` — a signed range is lopsided by one anyway, as in R.

**`uint`.** `UINT_MAX`, the mirror of `INT_MIN`. Unsigned ints use all 2ⁿ
patterns, so a real number has to go; for `uint8` that is 255, often a white
pixel. In exchange the item size is unchanged and numpy's SIMD loops run
untouched. The bias-shift alternative (store `x - 2ⁿ⁻¹`, reserve `INT_MIN`) is
worse twice over: it gives up 0 instead of `UINT_MAX`, and the cells then live in
a shifted frame, so a borrowed loop adds the right number in the wrong frame —
`200 + 100` lands on the cell that means 172.

**`float`, `complex`.** Every bit but the sign; see the *quiet* NaN section below.

**`S`, `V`.** The whole cell of `0xFF`, at any width. That is a legal byte
string, given up the way `UINT_MAX` is, but `0xFF` never occurs in ASCII or
UTF-8.

**`U`.** Every character is `U+FFFF`, a **noncharacter**: Unicode reserves that
group for a program's internal use and says it is never to be interchanged,
which is exactly this job. The first version used `0xFFFFFFFF`, which `chr()`
refuses and which therefore **gives up nothing at all** — but when someone reads
the values raw, numpy builds a `str` whose maximum character is outside Unicode,
and even `len()` on that string raises `SystemError`. A gap must not be a broken
object, so it changed. `U+FFFD` is out of the question: it is the replacement
character, produced by every decode with `errors="replace"`, so it is very often
real data.

The **whole cell** has to be checked: `b"\xffab"` and `"￿q"` are ordinary
values, so looking at the first byte alone is wrong. A cell with a value almost
always stops at the first character, so it stays fast.

**Records.** Every field holds the NA of its own type: a missing `(i4, f8, S3)`
holds `(INT32_MIN, 0x7FFF…, 0xFFFFFF)`, with padding bytes `0x00`. A record is
missing only when **every** field is; one field with a value makes the whole
record a value, so the only thing given up is the combination where every field
was already reserved.

Filling the cell with `0xFF` does not work. It gives up `(-1, -1)` for two
`int32` fields, a very common old-style "missing" value. Worse, the fields of
such a cell would read back as `-1`, `True` and `1969-12-31` for `int32`, `bool`
and `datetime64` — **looking like real data**, exactly `numpy.ma`'s kind of leak.
Field by field, a field taken out of a gap is still a gap of its own type.

When the dtype is created the record is flattened into leaf fields
`(offset, repeat count, NA pattern)`: nested records and subarray fields are
unrolled, and padding belongs to no leaf so it never decides anything. NA is per
record, not per field: per-field NA would need its own scalar type, because
`np.void` cannot hold `NA`. The byte layout is the same either way, so adding it
later would still read old data. A subarray dtype on its own, outside a record,
is rejected.

**`longdouble`, `clongdouble`.** No counterpart of their own: stored as
`float64` and `complex128`. Every result stays `float64`/`complex128` too; a
`longdouble` operand mixed into an operation is rounded on the way in, with a
warning, and never promoted back on the way out. `astype` gets a `longdouble`
array back, but the precision went at the door. The substitution also applies
inside a record — the record is rebuilt, the fields after it move up, and
numpy's field-by-field cast carries the values over. A real `long double`
differs everywhere — x87 80-bit on x86-64, IEEE quad on aarch64, plain `double`
under MSVC — so keeping it would mean three float formats for one dtype. About 3
digits are lost and the range becomes ±1.8e308; `LongDoubleWarning` fires
exactly when something is actually lost, and stays quiet where `long double`
already is a `double` (Windows).

**`object`.** Refused: it holds references, and this dtype does not do reference
counting.

**`StringDType`.** Refused, and not for lack of work. It **already has an NA of
its own** from NEP 55: `np.dtypes.StringDType(na_object=np.nan)`. Its elements
also own heap allocations that only it knows how to free, and the public API
gives a wrapping dtype no way to ask it to. Wrapping it would be both redundant
and unsafe.

### Which value is reserved

| dtype | NA |
|---|---|
| `float16`, `float32`, `float64` | every bit but the sign: `0x7FFF`, `0x7FFFFFFF`, `0x7FFFFFFFFFFFFFFF` |
| `complex64/128` | the float pattern in **both halves**, read from the real one |
| `int8` … `int64` | `INT_MIN` — as in R |
| `uint8` … `uint64` | `UINT_MAX` |
| `bool` | the byte `2`, which numpy never produces |
| `datetime64`, `timedelta64` | `INT64_MIN` — **free** |
| `S<n>`, `V<n>` | the whole cell of `0xFF` |
| `U<n>` | every character `U+FFFF` |
| records | every field holds its own type's NA, padding `0x00` |

Datetime is the strongest case: numpy **already** reserves exactly `INT64_MIN`
for `NaT`. Nothing legal is taken away, and `filled()` hands back a real `NaT`
rather than an invented date. In other words numpy already ships a bit-pattern
NA — for exactly two types — and the rest of this is generalising it.

Every pattern is written in native byte order, and the dtype itself is always
stored native: see the `>i4` segfault below.

### Not relying on the hardware

x86 propagates NaN payloads, but when both operands are NaN the **left one
wins** — so `NA + nan` keeps NA while `nan + NA` loses it. That is an x86
convention, not IEEE 754, and some architectures canonicalise NaNs and wipe the
payload entirely.

So the loops **do not rely on propagation**. They read the inputs and write NA
into the output themselves:

```
NA + nan  ->  [NA nan]
nan + NA  ->  [NA nan]     both ways round, and an ordinary nan stays a nan
```

## Performance

`a + a` on `f8`:

| n | numpy | Nullable | slower by |
|---|---|---|---|
| 1,000 | 0.6 µs | 3.3 µs | 5.9x |
| 8,192 | 2.1 µs | 8.4 µs | 4.1x |
| 100,000 | 20.6 µs | 90 µs | 4.4x |
| 1,000,000 | 860 µs | 1.8 ms | 2.0x |
| 5,000,000 | 5.1 ms | 14.1 ms | 2.7x |

Almost all of the difference is the NA scan — one extra read over data that was
just written. On small arrays the fixed cost dominates: about 2.5 µs per
operation, of which the loop lookups are about 1.4 µs (`resolve_dtypes` 0.54 µs,
`_resolve_dtypes_and_context` plus `_get_strided_loop` 0.89 µs). A cache could
remove most of that.

Making the scan branch-free and blocked (above) changed `a + a`, measured with
`scratchpad/bench_fixup.py` (numpy 2.5.3, gcc 15, `-O2`, best of 9; WSL timings
are noisy, so read these as ranges):

| case | before | after | plain numpy |
|---|---|---|---|
| `f8`, n = 100,000 | 62–66 µs | 53–55 µs | 20 µs |
| `f8`, n = 100,000, one gap | 125–137 µs | 51–54 µs | — |
| `i4`, n = 100,000 | 71–73 µs | 40–54 µs | 10 µs |
| `i4`, n = 1,000,000 | 730–820 µs | 380–500 µs | 90–140 µs |
| `i8`, `f8` at n ≥ 1,000,000 | — | about the same | — |
| n = 1,000 | ~3 µs | ~3 µs | 0.6 µs |

At ten million elements everything is bound by memory bandwidth, and Nullable
stays around 1.6–2x plain numpy: it has to read the inputs as well as the
output.

The two changes that mattered most, measured on the old layout at the time but
still in use: **borrowing T's C loop through the `numpy_1.24_ufunc_call_info`
capsule** instead of calling a Python-level ufunc per chunk, and **copying in
bulk instead of `memcpy` per element** in the `Nullable → Nullable` cast. The
second made `copy()` 2.6x faster and, unexpectedly, `a + b` twice as fast,
because ufuncs buffer the operands of a custom dtype so every buffer refill went
through exactly that cast. The lesson: measure one layer at a time — copy, cast,
loop — because the total hides it.

The flag layout was 2-3.5x slower than the bit pattern and 4-13x slower than
plain numpy. All of its measurements, plus the experiments with padding the item
size to 16, sharing one flag byte across cells, and hand-written SIMD, are in
`archive/flag-layout/README.md`.

## The trap: functions built on `sort` return wrong numbers silently

`np.median`, `np.percentile` and `np.quantile` are implemented in Python on top
of `sort`. Because gaps sort last, they **treat NA as the largest value** and
return a plausible-looking number:

```python
a = [3.0, 1.0, 5.0, NA]
np.median(a)    # 4.0   -- wrong, that is the mean of 3 and 5
```

Nothing in the dtype can stop it: removing `compare` would kill `sort` too. This
is why pandas rewrote its own `median` instead of calling numpy — and why there
is a statistics layer here:

```python
nd.median(a)                # NA    propagates, the default
nd.median(a, skipna=True)   # 3.0   correct
nd.quantile(a, [.25, .75])  # [NA NA]
nd.unique(a)                # NA is a value of its own, sorted last
```

`nd.sum/prod/min/max/mean/std/var/median/quantile/percentile` all take the same
`skipna=False`, instead of spelling out `where=nd.notna(a)`, and `axis=` and
`keepdims=` as numpy does:

```python
m = [[1.0, 2.0, 3.0],
     [4.0, NA,  6.0]]
nd.mean(m, axis=1)               # [2.0 NA]    a lane with a gap is NA
nd.mean(m, axis=1, skipna=True)  # [2.0 5.0]
```

`sum` and `prod` along an axis are vectorised; the others run numpy's own
function lane by lane, which is slower but keeps the numbers numpy's. `numpy.ma`
would be faster and is wrong for this: it masks every non-finite result, so the
mean of `[inf, 1]` comes back masked and its std as 0. A reduction over nothing
is covered below. `np.sum(a, skipna=True)` itself cannot exist without changing
numpy, which is why the keyword lives on `nd.*`.

`np.unique` is also unusable, but for a different reason and it **fails
cleanly**: it compares neighbours in the sorted array and assigns into a plain
bool mask, which NA refuses to become.

## The segfaults found while moving `S`/`U`/`V` to bit patterns

Moving the types took a few dozen lines; most of the work was **five existing C
bugs** that no test had touched before. Each now has a test of its own.

**`compare` and `nonzero` passed a NULL array.** Both slots called the wrapped
dtype's function with `ap = NULL`. The numeric ones ignore `ap`, so they were
fine, but `STRING_compare` and `UNICODE_compare` read the item size off it and
`VOID_compare` reads the field list — so `sort`, `argmax` and `nonzero` on
`S`/`U`/`V`/records all segfaulted. Now `S`/`U`/`V` are compared directly
(`memcmp`, code unit by code unit), and a record gets a stack array standing in
for the real one — the same trick `VOID_compare` uses for each field.

**`nonzero` on a large array with a gap.** The `nonzero` slot raises on a gap.
But above 500 elements numpy releases the GIL before the loop unless the dtype
declares `NPY_NEEDS_PYAPI`, so setting an exception there was a segfault — for
**every** wrapped type, `float64` included. `VOID_compare` needs the GIL too (it
asks the memory handler for scratch buffers), and numpy's own record dtypes
carry that flag. The dtype now declares `NPY_NEEDS_PYAPI`.

**`astype` to a different width.** The cast to a plain dtype copied the source's
bytes as they were: `Nullable[i4].astype(np.float64)` left the top half of every
`double` as garbage, and `.astype(np.int16)` **wrote past the end of the
buffer**. It now checks for gaps first and lets numpy cast the values.

**Non-native byte order.** `Nullable(">i4")` kept the `>`, but the NA pattern is
written natively, so NA was the bytes `00 00 00 80` — read big-endian, **128**:
the number 128 was refused while `INT_MIN` went in as an ordinary value. Worse,
`np.sort` calls `PyArray_DescrNewByteorder` for a byte-swapped copy of the dtype;
for a new-style DType that returns `NULL`, and numpy uses it **without
checking** — segfault. Everything is stored native now; big-endian data goes in
and out through numpy's own casts.

**Assigning `np.void`.** Assigning a numpy scalar is a cast from the scalar's
dtype. With no cast registered for `S`/`U`/`V`, `np.bytes_` and records were
refused, and an unstructured `np.void` segfaulted inside numpy. All three types
now have casts in and out.

Along the way: `np.dtype("S3")` is not a singleton, every call makes a new
object, so every dtype comparison by identity (`wrapped == value`) missed for
strings and pushed the data down the slow cast path. They compare with
`PyArray_EquivTypes` now.

## `==` on records used to be silently wrong

numpy has no ufunc loop comparing records; `==` on a plain record array takes a
separate path inside `array_richcompare`, keyed on `type_num == NPY_VOID`. When
`==` or `!=` finds no loop, numpy **swallows** the `UFuncNoLoopError` and returns
an all-`False` array (or all-`True`), with no warning:

```python
a == a    # [False False False]   -- wrong, and nothing says so
```

There is now a dedicated resolve for `equal`/`not_equal` on records: the loop
calls numpy's own `==` on the two value views and then stamps the gaps. Records
of different dtypes raise `TypeError`, and since numpy only swallows
`UFuncNoLoopError`, that one still comes out. Every other ufunc on records still
raises, because only `==` and `!=` are swallowed.

## A numpy bug found along the way

`maximum`/`minimum` used to return wrong results when called through the
borrowed-loop capsule. It looked like "the last element is skipped"; chased to
the end, the cause was entirely different and lives in numpy, not in this dtype.

`loops_minmax.dispatch.c.src` turns a byte stride into an element stride with
`is1 / sizeof(STYPE)`, after asking `npyv_loadable_stride`. But in
`numpy/_core/src/common/simd/simd.h` the divisibility check is wrapped in one
more condition:

```c
if (alignof(npyv_lanetype_##SFX) != sizeof(npyv_lanetype_##SFX) &&
        stride % sizeof(npyv_lanetype_##SFX) != 0) {
    return 0;
}
stride = stride / sizeof(npyv_lanetype_##SFX);
```

For `double` on x86-64, `alignof == sizeof == 8`, so **the left side is false and
the divisibility check never runs**. The division then truncates: stride 9
becomes 1, stride 12 becomes 1, and the SIMD loop walks the array as if it were
contiguous — reading both inputs wrong and writing the output wrong.

The very commit that added this (`94854dbc77`, "BUG: Fix simd loadable stride
logic") states the opposite intent: *"Strides are now guaranteed to be a
multiple and compatible"*. The `alignof != sizeof` condition disables exactly
what it just added.

numpy itself never trips over it, because it buffers into contiguous memory
before calling the loop. Only a third party calling
`ufunc._get_strided_loop(..., fixed_strides=...)` directly — public API since
1.24, and exactly what this prototype uses — can hit it.

A minimal reproducer in ctypes, with no custom dtype involved:
`scratchpad/minmax_raw.py`. The proposed patch (drop the `alignof` clause) is in
`numpy-simd-stride.patch`. **Not reported upstream yet.**

On this side it only takes checking divisibility instead of excluding the two
operations outright. The item size equals `T`'s, so every stride is a multiple
of it and the fast path always applies — 3.1x slower than plain numpy at 100,000
elements and 1.4x at 5,000,000.

## Why float NA is `0x7FFF…` and a *quiet* NaN

R uses `0x7FF00000000007A2` for `NA_real_` (0x7A2 = 1954, Ross Ihaka's year of
birth). That pattern is a **signaling NaN** — the top bit of the significand is
0. R gets away with it because it never reads the IEEE flags; numpy does, and
turns it into `RuntimeWarning: invalid value encountered in add` every time an
operation merely *passes over* a gap. Missing data is not an arithmetic error,
so it must not be reported as one.

The first version kept R's payload and set the quiet bit. The current one drops
the payload and uses **every bit but the sign**: `0x7FFF`, `0x7FFFFFFF`,
`0x7FFFFFFFFFFFFFFF`.

- **One rule for every width**, the same all-ones idea as `UINT_MAX` and the
  all-`0xFF` string cell. `float16` no longer has to shrink a payload into 10
  bits.
- **Still a quiet NaN** (exponent and top significand bit set), so passing over a
  gap raises no `FE_INVALID`.
- **Arithmetic never synthesises it**: `np.nan` is `0x7FF8000000000000`, and the
  NaN x86 produces for `inf - inf` is `0xFFF8000000000000`.
- **The sign is ignored when reading**, so a borrowed loop that flips the sign
  leaves the cell a gap.

## Two traps in borrowing someone else's loop

Going from 14 to 26 binary operations exposed two hidden assumptions the first 14
never touched. Both were **silently wrong**.

**`fmax`/`fmin` swallow NaN.** The output-fixing pass has an optimisation: scan
the output for NaN, and if there is none there is nothing to fix — which rests on
the hardware having put a NaN wherever an input was NA. `fmax` deliberately
returns the *non*-NaN side, so a gap vanishes from the output and the scan skips
exactly the row that needed fixing: `fmax(NA, 2.0)` gave `2.0`. NA is not NaN —
it propagates regardless of what the operation does with NaNs.

`fmax` was not the only one. A later sweep over special values found four more
operations that turn a NaN into a number, all by the book: `pow(1, NaN)` and
`pow(NaN, 0)` are 1 in IEEE 754 and C99, `copysign` reads only the sign bit,
`heaviside(x, NaN)` is 0 or 1 unless `x == 0`, and `hypot(±inf, NaN)` is inf.
They all scan their inputs now (`binop_swallows_nan`), but they do not all
answer NA.

A gap is a value nobody knows, so what matters is whether the answer depends
on it. `fmax(NA, 2)` does (the gap might be larger), and so does
`copysign(1, NA)`: those are NA. `1 ** NA`, `NA ** 0`, `hypot(inf, NA)` and
`heaviside(3, NA)` do not: whatever the gap holds, the answer is the same, and
IEEE 754 returns a number exactly in those cases for exactly that reason. So
for `power`, `float_power`, `hypot` and `heaviside` a gap is stamped only where
the output came out NaN (`binop_keeps_determined`). This is the same reasoning
as Kleene logic (`NA & False == False`), and R (`1^NA == 1`, `NA^0 == 1`) and
pandas (`1 ** pd.NA == 1`) agree; SQL does not (`POWER(1, NULL)` is NULL).
Complex operands take the masked path and always give NA.

Integer `power` had its own bug: an integer gap is `INT_MIN`, which the loop
read as a negative exponent, so `2 ** NA` raised "Integers to negative integer
powers are not allowed". It now takes the masked path, and
`nullable_int_power_rules` gives back the two determined cases, `1 ** NA` and
`NA ** 0`. A real negative exponent still raises, as for a plain array.

**`<` and `>` are *signaling* comparisons.** IEEE 754 has them raise
`FE_INVALID` even on a quiet NaN, while `==` and plain arithmetic stay silent —
verified in `scratchpad/qnan.c`. The bit-pattern layout lets the loop run
straight over a gap, so any operation that compares its two operands internally
(`logaddexp`, `logaddexp2`) reported `invalid value encountered` for data that
**is not there**. Those take the `where=` path instead. The same sweep found
that complex kernels do this almost everywhere: division scales by the larger
component, `power` special-cases its arguments, and ordering compares
lexicographically with `<`. For complex operands only `add`, `subtract`,
`multiply`, `==` and `!=` run straight over a gap.

Both lists are empirical, not derivable, so the test suite sweeps *every*
operation, taking the names straight from the C table: an operation missing from
a list turns a test red instead of turning up as a strange warning in someone's
output. Verified by mutation — switching off each guard turns 2, 4 and 10 tests
red respectively. `test_every_binop_keeps_a_gap` adds the special values
(0, ±0, 1, ±1, 2, 0.5, ±inf, NaN) on either side of a gap, for every float and
complex width, and fails on a lost gap or on any floating-point warning;
removing the four new entries turns 5 of its cases red, removing the complex
rule 4.

## The second numpy bug: an assertion too strict for non-legacy DTypes

`Nullable[f8]` in the **flag layout** (now archived) plus a Python literal, on an
array larger than `NPY_BUFSIZE` (8192), **aborts** the numpy dev build:

```
can_cast_pyscalar_scalar_to: Assertion `NPY_DT_is_legacy(NPY_DTYPE(to))' failed.
```

All three are needed: a non-legacy DType, the other operand being a **Python
literal** (writing `np.float64(1.0)` does not trip it), and the array being split
into buffered chunks — raising `np.setbufsize` above the array size makes it go
away. The current layout does not hit it, because the item size equals `T`'s and
is properly aligned, so no buffering is needed.

But reading further into that function shows **the assertion is the wrong part,
not the code under it**. The three shortcuts at the top
(`ISCOMPLEX`/`ISFLOAT`/`ISINTEGER`) all key on `type_num` and all return `false`
for this descriptor (`num = -1`); the final branch is generic — it builds a
descriptor for the scalar and calls the common `PyArray_CanCastTypeTo`, which
works for any DType.

Verified: removing that one `assert` line and rebuilding numpy makes every array
size work, with results matching plain numpy. The patch is in
`numpy-pyscalar-assert.patch`.

It only affects builds with assertions on; in a released numpy `assert` is
disabled and the generic path was already correct. **Not reported upstream yet.**

## Three-valued logic

`logical_and` and `logical_or` do not merely propagate NA — sometimes the answer
is known even though one side is missing:

```
   AND |  T    F   NA          OR  |  T    F   NA          XOR |  T    F   NA
     T |  T    F   NA            T |  T    T    T            T |  F    T   NA
     F |  F    F    F            F |  T    F   NA            F |  T    F   NA
    NA | NA    F   NA           NA |  T   NA   NA           NA | NA   NA   NA
```

Same as SQL, R and pandas. These are the only three ufuncs that need a loop of
their own, because the validity of the result depends on the *values*, not only
on the validity of the inputs.

## Why a gap keeps nothing of the old value

NA means *there is no value*, so the bytes of a gap are padding, not data. A gap
**is** the NA pattern: `filled()`, `tobytes()`, hashing and pickling are all
deterministic, and there is no path by which the old value can come back out.
This is what `numpy.ma` does not do, and it is the root of nearly all of its
bugs:

- Every leak bug in `numpy.ma` ends with the value under the mask reaching the
  user **looking like real data**. A canonical NA pattern cannot be mistaken for
  one.
- Deterministic results: keeping whatever was computed would make `tobytes()`,
  hashes and pickles depend on evaluation order and compiler optimisations.
- Values a user masked out do not travel with the array into a file or over the
  wire.

The padding bytes in a missing record are written `0x00` for the same reason. R
chose this direction for floats; here it applies to every type.

## Why there is `matmul` but no `np.dot`

`A @ B` works: `matmul` is a gufunc, it goes through the new DType API, so the
loop receives both the descriptors and the core dimensions. `C[i,j]` is a sum
over `k`, so it is missing exactly when row `i` of A **or** column `j` of B has a
gap anywhere — decided per row and per column rather than per cell, which is two
scans of size `n` and `m` instead of `n*m`.

`np.dot` does not, and not for lack of trying. It goes through the old `dotfunc`
slot:

```c
void dotfunc(void *ip1, npy_intp is1, void *ip2, npy_intp is2,
             void *op, npy_intp n, void *arr);
```

`arr` is the only way to reach the descriptor, and only the descriptor says which
type is wrapped and what its NA pattern is. numpy passes **NULL**:

```c
dot(it1->dataptr, is1, it2->dataptr, is2, op, l, NULL);
/* multiarraymodule.c, PyArray_MatrixProduct2 */
```

I implemented it and it segfaulted on exactly the line that reads `arr`. numpy
also releases the GIL around the loop, so calling back into Python to look
anything up is out too. That ABI predates parametric dtypes, and a parametric
dtype cannot use it. The slot is left empty and numpy reports
`dot not available for this type` — a clean error, which is right.

## The `NA` scalar

Indexing returns an ordinary Python value when the element has one, and the `NA`
singleton when it does not. Whatever someone can do with `a[0]` they will sooner
or later do with `a[1]`, so the singleton has to behave rather than explode.

```python
NA + 1      # NA        every operation touching NA gives NA
NA > 1      # NA
NA == NA    # NA        two things you do not know are not thereby equal
a[1] is NA  # True      identity still answers, and NA is still hashable
bool(NA)    # TypeError
float(NA)   # TypeError
```

`bool(NA)` **used to return `True`** — Python's default for an object with no
`__bool__` — so `if a[i]:` quietly treated a hole in the data as a yes. There is
no right answer to "is the value you do not have true?", so it must refuse rather
than invent one. R says *"missing value where TRUE/FALSE needed"*, and
`bool(pd.NA)` raises too.

Fixing this also turned `np.histogram`'s confusing `TypeError` about
`NAType.__sub__` into a message that names the actual problem.

## Reducing over nothing

An empty array and an array that `skipna` emptied are **the same situation**, so
they have to give the same answer — at one point they did not, and that was a bug.

| | |
|---|---|
| `sum` → `0`, `prod` → `1` | there is an identity element, so this is the right answer rather than an invention; R and numpy agree |
| `mean`, `median`, `min`, `max`, `std`, `var`, `quantile` → **NA** | no identity element: the mean of no numbers is *unknown* |

`NaN` must **not** be returned here. In this dtype NaN is an ordinary value that
a column may legitimately hold, so returning NaN would say "the answer is the
number not-a-number" rather than "unknown". `nd.median` of an empty array used to
return exactly that, with two `RuntimeWarning`s attached.

A note in an older test of mine claimed *"R returns NA for
`sum(c(NA,NA), na.rm=TRUE)`"* — **wrong**, R returns `0`. Fixed.

## `argmax` with gaps

A gap could be the largest value, so `argmax` answers with the position of the
**first gap** — numpy's rule for NaN. That keeps `a[a.argmax()]` NA exactly
when `a.max()` is. It used to skip the gaps, which was IGNORED behaviour hiding
in a MISSING dtype: `a.max()` said NA while `a[a.argmax()]` said 5.0. Skipping
is now asked for by name:

```python
a = [3.0, 1.0, NA, 5.0]
a.argmax()                    # 2     the gap
nd.argmax(a, skipna=True)     # 3     position in the original array
```

An all-NA array gives `0`, exactly what numpy returns for an array of all `NaT`
or all `nan`; `nd.argmax(a, skipna=True)` gives NA there. Raising would be better, but this slot cannot: `_PyArray_ArgMinMaxCommon`
**ignores the return value** and never looks at the error state. Originally the
loop also ran with the GIL released, so touching the error state there was a
segfault — tried, and it was. The dtype now declares `NPY_NEEDS_PYAPI` so the GIL
is held, but an exception set there would still only surface as a `SystemError`
instead of a message. Nothing is invented either way: `a[a.argmax()]` is NA.

Same family of limitation as `dotfunc`: an old ABI with no channel for errors.

## Checking for memory leaks

`scratchpad/leakcheck.py`. It measures with `sys.getallocatedblocks()` — the
number of blocks live in CPython's allocator, so one leaked `PyObject` shows up
immediately as `+1`. Far more precise than RSS; the first attempt used RSS and it
produced two false alarms.

At **N = 40,000 iterations**, leaking one block per call would give `+40,000`.
The largest number measured is `+62`, which is arena noise. It also tracks
`sys.getrefcount` of the **shared** objects — the descriptor, `wrapped`, `NA`,
each ufunc, the DType class itself — because leaking a reference to one of those
allocates no new block and the block count would not see it. None of them grew.

It covers the fast path, the slow `where=` path, the reduction path, the gufunc
path, and 8 **error** paths, which is where refcount bugs like to hide in the
`goto fail` branches.

**The measurement itself is verified by mutation.** Deliberately removing
`PyMem_Free(rowna/colna)` in `matmul` and `Py_DECREF(res)` in the slow path:

```
M @ M                blocks  +40000    (2 blocks/call, matching 2 PyMem_Free)
logaddexp (slow)     blocks  +20001    (1 block/call, matching 1 Py_DECREF)
```

Both were caught immediately. Without that step, "no leaks" is just a sentence.

**The `S`/`U`/`V` round**: `scratchpad/leak_suv.py` measures 18 new paths —
`sort`/`argmax` on `S`/`U`/`V`/records, five raising paths (reserved value, a
cast landing on NA, `nonzero` on a large array), width-changing casts,
big-endian, assigning `np.void` and records. The largest at 3,000 iterations is
`+17`; the two worst were re-measured at 12,000 and 48,000 iterations and stayed
flat, so nothing grows with the number of calls.

**The record round**: `scratchpad/leak_records.py` measures 13 paths — creating
and destroying descriptors (allocating and freeing the leaf-field list), two
refusal paths (an object field, a record of nothing but padding), rebuilding a
record containing `longdouble`, assignment, casting in and out, `sort`/`argmax`,
`nonzero` raising, `isna`/`filled`. Plus three comparison paths (`==`/`!=`,
records of different dtypes). The largest is `+33` at 3,000 iterations, flat
again at 12,000 and 48,000; the refcounts of the dtype and of `NA` did not move.

**ASAN + UBSAN.** A separate build with `-fsanitize=address,undefined`, running
the whole test suite and `leak_records.py`: not a single report. This is how
the `na_bytes[8]` 8-byte over-read with `complex128` was caught, before it was
widened to 16.
