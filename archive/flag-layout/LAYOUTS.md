# NA layouts by wrapped dtype

| Layout | Type | T (bytes) | Nullable (bytes) | NA stored as | Note |
|---|---|---:|---:|---|---|
| Bitpattern | `bool` | 1 | 1 | `0x02` | never produced by numpy |
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
| | `U<n>` | 4n | 4n | `0xFFFFFFFF…` | per character; not a code point |
| | `V<n>` | n | n | `0xFF…FF` | all n bytes |
| | structured | sum of fields | sum of fields | per field | each field's own NA; padding `0x00` |
| Flag | any, via `FlagLayout(T)` | T | T + 1 | `0x00…00` | value zeroed; flag byte 0 |
| Stored as double | `longdouble` | 16 | 8 | `0x7FFFFFFFFFFFFFFF` | as `float64`; warns |
| | `clongdouble` | 32 | 16 | `0x7FFFFFFFFFFFFFFF` | as `complex128`; in both halves; warns |
| Rejected | `object` | 8 | — | — | `TypeError` |
| | `StringDType` | 16 | — | — | `TypeError` |

Hex values are the value of the stored type, not bytes in memory order: on
little-endian x86-64 an `int32` NA sits in memory as `00 00 00 80`.

`longdouble` and `clongdouble` have no Nullable counterpart. They are stored as
`float64` and `complex128`, which keeps about 15 significant digits and a range
of about ±1.8e308; values beyond that are rounded, or become `inf` or `0`, and a
`LongDoubleWarning` says so. Where `long double` already is a double (Windows),
nothing is lost and there is no warning. The T sizes shown are for x86-64 Linux.

Float and complex NA is every bit set except the sign. The sign is ignored when
reading, so a gap whose sign was flipped is still a gap.

For `S`, `U` and `V`, NA is the whole cell filled with `0xFF`, at any width.
In `U` that is the code unit `0xFFFFFFFF`, which is not a Unicode code point,
so no string is given up. In `S` and `V` it is a legal byte string, given up
the way `UINT_MAX` is; `0xFF` never occurs in ASCII or UTF-8 text. Only the
whole cell counts: `b"\xffab"` is an ordinary value. Storing the reserved value
raises, and so does a cast that would produce it.

A record is NA when every field holds its own type's NA: `(i4, f8, S3)` holds
`(0x80000000, 0x7FFFFFFFFFFFFFFF, 0xFFFFFF)`, and padding bytes are `0x00`. A
record with only some fields reserved is an ordinary value, so the one value
given up is the record whose every field is reserved already. Nested records
and subarray fields are unrolled field by field. `longdouble` fields are stored
as `float64`, which moves the fields after them; `object` fields are rejected.

`Nullable()` picks the flag layout only where there is no cell to fill: `S`
with no length, or a bare subarray dtype outside a record. `FlagLayout(T)`
asks for it explicitly.
