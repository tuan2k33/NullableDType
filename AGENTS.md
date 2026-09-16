# AGENTS.md

Guidance for coding agents working in this repository.

## What this is

A prototype NumPy DType, `Nullable(T)`, that stores missing values *inside*
the dtype as a reserved bit pattern of `T` (see `LAYOUTS.md`). The goal is a
replacement for `numpy.ma` built on the NEP 41/42/43 DType API, on **stock
numpy** — never patch numpy to make something here work.

- `src/nulldtype.c` — the whole C extension (`_nulldtype`): the DType, the `NA`
  scalar, casts, ufunc loops (borrowed from the wrapped dtype through the
  `numpy_1.24_ufunc_call_info` capsule), Kleene logic, records, `S`/`U`/`V`.
- `nulldtype.py` — the Python layer, imported as `nd`: `Nullable`, `isna`,
  `filled`, the `nd.*` reductions with `skipna=`/`axis=`, and a module
  `__getattr__` that forwards every other name to numpy (`nd.sort is np.sort`)
  and refuses misleading ones (`nd.nansum`) through `_REFUSED`.
- `test_basic.py` — the whole test suite.
- `scratchpad/` — measurement scripts: `compare_ma.py` (generates the tables in
  `VS-NUMPY-MA.md`), `leakcheck.py`, `leak_suv.py`, `leak_records.py`,
  reproducers for numpy bugs.
- `archive/flag-layout/` — frozen snapshot of the old two-layout version. It
  builds and tests on its own; do not change its behaviour.
- `numpy-*.patch` — proposed upstream fixes for numpy bugs found here. The
  project must not depend on them.

## Build and test

Built against a local numpy dev build (paths are hard-coded in the scripts):

```bash
./build.sh                          # compile src/nulldtype.c -> _nulldtype*.so
./run_tests.sh                      # build + pytest test_basic.py
./run_tests.sh -k argmax            # extra args go to pytest
```

Python is `/mnt/c/Dev/projects/numpy/.venv/bin/python3`, with
`PYTHONPATH=/mnt/c/Dev/projects/numpy/build-install/usr/lib/python3/dist-packages:.`
and `-P`.

After any change to `src/nulldtype.c`, also:

1. build with `-O1 -g -fsanitize=address,undefined` into a scratch directory and
   run the whole suite with `LD_PRELOAD` of `libasan.so` and `libubsan.so`
   (`ASAN_OPTIONS=detect_leaks=0`); there must be no report;
2. run a leak-scaling check over the changed paths (`scratchpad/leak*.py`, or a
   new one in the same style): block counts and refcounts must stay flat as the
   iteration count grows.

Memory safety in the C code is treated as critical — check it every time.

After changing behaviour visible in the comparison, rerun
`scratchpad/compare_ma.py` and splice its output into `VS-NUMPY-MA.md` between
the fixture block and `## How to read this`.

## Semantics to preserve

The design follows **MISSING** semantics (R/SQL), not **IGNORED** (`numpy.ma`):
a gap is a value nobody knows.

- Anything that depends on a gap is NA: arithmetic, comparisons (`NA == NA` is
  NA), reductions, `argmax` (points at the first gap, numpy's NaN rule),
  `nd.isin` against a set containing a gap.
- Skipping is always explicit: `skipna=False` by default on every `nd`
  reduction, `nd.all`/`nd.any`, `nd.argmax`/`nd.argmin`, `nd.cumsum`/`nd.cumprod`.
- `logical_and/or/xor` and `&`, `|`, `^` on `Nullable(bool)` are Kleene.
- Nothing invents or drops NA silently: casting onto the reserved value or out
  of a gap raises; `bool(NA)` raises; arithmetic landing on the reserved value
  raises.
- NA is not NaN. Over nothing, `sum`/`prod` give their identity and everything
  else gives NA, never NaN.
- Kept conventions: `sort` puts NA last, `nd.unique` keeps one NA, `nd.count`
  counts values.
- When a numpy function cannot work on a Nullable array (`np.all`, `np.dot`,
  `np.median` returns wrong numbers), provide an `nd.*` counterpart rather than
  hijacking numpy.

## Conventions

- All code, comments, docstrings and documents are in English.
- Match the surrounding style: comments explain *why*, usually with the
  measured reason (e.g. which numpy code path forces the choice).
- Every behaviour fix gets a test that fails without it.
- Keep `README.md`, `LAYOUTS.md` and `VS-NUMPY-MA.md` in step with the code,
  including the test count in `README.md`.
- Upstream numpy communications (issues, PRs) are drafted only; the owner
  posts them.
