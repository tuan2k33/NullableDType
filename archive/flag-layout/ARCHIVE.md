# Archived: the two-layout version

A snapshot of nulldtype from before the flag layout was removed. At that point
`Nullable(T)` had two storage layouts behind one name:

- **Bitpattern** (`src/sentinel.inc`): NA is a reserved value of `T`, so the
  item size equals `T`'s. This is what the current code keeps.
- **Flag** (`src/nulldtype.c`): the value followed by one validity byte, `T + 1`
  bytes, with the value bytes zeroed under a gap. It reserves no value, but
  breaks alignment and runs 2-3.5x slower.

The flag layout was the fallback for types with no spare value. Once `S`, `U`,
`V` and records got bit patterns, `Nullable()` no longer picked it for anything
common, and every C fix had to be made twice. It was removed; the measurements
and experiments behind it are in `README.md` here.

This copy builds and tests on its own, against the same numpy dev build:

```bash
./run_tests.sh    # 783 passed when archived
```

`FlagLayout(T)` and `BitpatternLayout(T)` ask for a layout by name.
