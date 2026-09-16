# NA patterns by wrapped dtype

| Group | Type | T (bytes) | Nullable (bytes) | NA stored as | Note |
|---|---|---:|---:|---|---|
| Reserved value | `bool` | 1 | 1 | `0x02` | never produced by numpy |
| | `int8` | 1 | 1 | `0x80` | `INT8_MIN` |
| | `int16` | 2 | 2 | `0x8000` | `INT16_MIN` |
| | `int32` | 4 | 4 | `0x80000000` | `INT32_MIN` |
| | `int64` | 8 | 8 | `0x8000000000000000` | `INT64_MIN` |
| | `uint8` | 1 | 1 | `0xFF` | `UINT8_MAX` |
| | `uint16` | 2 | 2 | `0xFFFF` | `UINT16_MAX` |
| | `uint32` | 4 | 4 | `0xFFFFFFFF` | `UINT32_MAX` |
| | `uint64` | 8 | 8 | `0xFFFFFFFFFFFFFFFF` | `UINT64_MAX` |
| | `float16` | 2 | 2 | `0x7FFF` | quiet NaN; sign ignored |
| | `float32` | 4 | 4 | `0x7FFFFFFF` | quiet NaN; sign ignored |
| | `float64` | 8 | 8 | `0x7FFFFFFFFFFFFFFF` | quiet NaN; sign ignored |
| | `complex64` | 8 | 8 | `0x7FFFFFFF` | `float32`; in both halves |
| | `complex128` | 16 | 16 | `0x7FFFFFFFFFFFFFFF` | `float64`; in both halves |
| | `datetime64` | 8 | 8 | `0x8000000000000000` | `NaT` |
| | `timedelta64` | 8 | 8 | `0x8000000000000000` | `NaT` |
| | `S<n>` | n | n | `0xFF…FF` | all n bytes |
| | `U<n>` | 4n | 4n | `0x0000FFFF` | per character; U+FFFF, a noncharacter |
| | `V<n>` | n | n | `0xFF…FF` | all n bytes |
| | structured | sum of fields | sum of fields | per field | each field's own NA; padding `0x00` |
| Stored as double | `longdouble` | 16 | 8 | `0x7FFFFFFFFFFFFFFF` | as `float64`; warns |
| | `clongdouble` | 32 | 16 | `0x7FFFFFFFFFFFFFFF` | as `complex128`; in both halves; warns |
| Rejected | `object` | 8 | — | — | `TypeError` |
| | `StringDType` | 16 | — | — | `TypeError`; has its own `na_object` |
| | `S`, `U`, `V` with no length | 0 | — | — | `TypeError`; no cell to fill |
| | bare subarray, e.g. `(i4, (2,))` | n × base | — | — | `TypeError`; fine as a record field |

Hex values are the value of the stored type, not bytes in memory order: on
little-endian x86-64 an `int32` NA sits in memory as `00 00 00 80`. Everything
is stored in native byte order, whatever was asked for: `Nullable(">i4")` is
`Nullable(int32)`, and big-endian data is swapped on the way in and out.

## What each kind gives up

**`bool`** — the byte `0x02`. NumPy only ever writes 0 or 1 into a bool, so
nothing real is given up.

**`int8` … `int64`** — `INT_MIN`, the same choice R makes. A signed range is
lopsided by one, so this is the one value it can spare.

**`uint8` … `uint64`** — `UINT_MAX`, the mirror of `INT_MIN`. Every bit pattern
of an unsigned int is a real number, so one has to go; with `uint8` that is 255,
often a white pixel. A bias shift (store `x - 2**(n-1)`, reserve `INT_MIN`) was
the alternative and is worse twice over: it gives up 0 instead, and the cells
then live in a shifted frame, so a borrowed loop adds the right number in the
wrong frame -- `200 + 100` lands on the cell that means 172. Reserving the top
value keeps each cell equal to its value, so numpy's own uint loops run
unchanged.

**`float16`, `float32`, `float64`, `complex64`, `complex128`** — every bit set
except the sign: `0x7FFF`, `0x7FFFFFFF`, `0x7FFFFFFFFFFFFFFF`. One rule for
every width, the same all-ones idea as `UINT_MAX`. It is a *quiet* NaN, so
merely passing over a gap raises no `FE_INVALID`, and the sign is ignored when
reading, so a gap whose sign a loop flipped is still a gap. Ordinary arithmetic
never synthesises it -- `np.nan` is `0x7FF8000000000000` and the NaN x86 makes
for `inf - inf` is `0xFFF8000000000000` -- which is what keeps an ordinary NaN
in the data distinct from a missing value. Complex carries the pattern in both
halves and is read from the real one.

**`datetime64`, `timedelta64`** — `INT64_MIN`, which numpy already calls `NaT`.
Free: nothing legal is taken away, and `filled()` hands back a real `NaT` rather
than an invented date.

**`S<n>`, `V<n>`** — every byte `0xFF`, at any width. That is a legal byte
string, given up the way `UINT_MAX` is, but `0xFF` never occurs in ASCII or
UTF-8 text. Only the whole cell counts: `b"\xffab"` is an ordinary value. The
reserved cell is not valid text, so `astype("U")` on a raw view of the values
fails to decode; through the dtype it never comes up, because a gap reads as
`NA`.

**`U<n>`** — every character `U+FFFF`, a *noncharacter*: Unicode sets those
aside for a program's internal use and says they are never to be interchanged.
Filling the cell with `0xFFFFFFFF` instead would give up nothing at all, since
`chr()` refuses it, but a cell read raw then comes back as a `str` whose maximum
character is out of range, and even `len()` on it raises `SystemError`. `U+FFFD`
would be worse still: it is the replacement character, which every decode with
`errors="replace"` produces, so it is real data. `U+FFFF` keeps every cell a
well-formed string. Here too only the whole cell counts: `"\uffffq"` is a value.

**structured (records)** — every field holding the NA of its own type:
`(i4, f8, S3)` is missing when it holds
`(0x80000000, 0x7FFFFFFFFFFFFFFF, 0xFFFFFF)`. A record with only some fields
reserved is an ordinary value, so the only value given up is the one whose every
field was already reserved. Padding is written as `0x00` but never read: numpy
copies records field by field, so padding at the destination is whatever the
buffer held. Nested records and subarray fields are unrolled field by field.
Filling the cell with `0xFF` instead was rejected: it gives up `(-1, -1)` for two
`int32` fields, and the `int32`, `bool` and `datetime64` fields of such a cell
would read back as `-1`, `True` and 1969 -- values, not gaps.

**`longdouble`, `clongdouble`** — no counterpart of their own: stored as
`float64` and `complex128`, which keeps about 15 significant digits and a range
of about ±1.8e308. Beyond that values are rounded, or become `inf` or `0`, and a
`LongDoubleWarning` says so; where `long double` already is a double (Windows)
nothing is lost and there is no warning. Every result stays `float64` or
`complex128`: a `longdouble` operand mixed into an operation is rounded on the
way in, and warns again there, never promoted back on the way out. Inside a
record the same substitution applies, which moves the fields after it.

**Rejected** — `object` holds references, which this dtype does not track.
`StringDType` brings a missing value of its own (`na_object`) and owns heap
allocations only it knows how to free. `S`, `U` and `V` with no length have no
cell to fill, and a bare subarray dtype is not one element -- as a record field
it works.

The T sizes in the table are for x86-64 Linux.

An earlier version also had a flag layout, the value followed by a validity
byte. Nothing common needed it once strings and records got patterns, and it
is kept in `archive/flag-layout/`.
