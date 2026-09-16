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
- `README.md` (using it), `DISCUSSION.md` (how and why, measurements, bugs
  found), `LAYOUTS.md` (NA patterns), `VS-NUMPY-MA.md` (generated comparison),
  `NUMPY-PATCHES.md` (where numpy's array functions fail, and upstream patch
  sketches).
- `scratchpad/` — measurement scripts: `compare_ma.py` (generates the tables in
  `VS-NUMPY-MA.md`), `leakcheck.py`, `leak_suv.py`, `leak_records.py`,
  reproducers for numpy bugs.
- `archive/flag-layout/` — frozen snapshot of the old two-layout version. It
  builds and tests on its own; do not change its behaviour.
- `LICENSE.txt` — BSD 3-Clause, as NumPy.
- `numpy-*.patch` — proposed upstream fixes for numpy bugs found here. The
  project must not depend on them.

## Build and test

Requires Python >= 3.12 and numpy >= 2.5. Packaging is `pyproject.toml` + `setup.py` (setuptools).

```bash
pip install .                       # normal install
./build.sh                          # build _nulldtype*.so in place
./run_tests.sh                      # build in place + pytest test_basic.py
./run_tests.sh -k argmax            # extra args go to pytest
```

The scripts use `PYTHON` (default `python3`) and prepend `NUMPY_SITE` to
`PYTHONPATH` when set; `local.env` (gitignored) supplies both on the owner's
machine, which points at a numpy dev tree. CI (`.github/workflows/ci.yml`)
tests numpy 2.5, the latest release and the nightly wheels, plus an
ASAN + UBSAN job. `.github/workflows/release.yml` builds wheels with
cibuildwheel (also on `ci/**` branches) and publishes a `v*` tag to PyPI
through trusted publishing.

After any change to `src/nulldtype.c`, also:

1. build with sanitizers and run the whole suite, as the CI job does:
   `CFLAGS="-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined" ./build.sh`,
   then pytest with `LD_PRELOAD` of `libasan.so` and `libubsan.so` and
   `ASAN_OPTIONS=detect_leaks=0`; there must be no report;
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
- Keep the documents in step with the code, including the test count in
  `README.md`. Usage belongs in `README.md`; reasoning, measurements and bug
  stories in `DISCUSSION.md`.
- Upstream numpy communications (issues, PRs) are drafted only; the owner
  posts them.
