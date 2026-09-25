# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Quick reference

```bash
./run_tests.sh                 # build _nulldtype in place + pytest test_basic.py
./run_tests.sh -k argmax       # single test / subset: extra args go to pytest
./build.sh                     # build only (honours PYTHON, NUMPY_SITE, local.env)
```

Architecture in one line: `src/nulldtype.c` is the whole C extension (`_nulldtype`: the DType, NA scalar, casts, ufunc loops borrowed from the wrapped dtype); `nulldtype.py` is the thin Python layer (`nd.*`) that forwards everything else to numpy. Targets stock numpy 2.5+ only — never patch numpy.

Any change to `src/nulldtype.c` requires the ASAN/UBSAN run and a leak-scaling check (details in AGENTS.md).
