"""Leak check for nulldtype operations.  Run this script and look for 
"LEAKY" in the output.  If you see any, please report it to the developers.
"""
import gc, sys, numpy as np, nulldtype as nd
from nulldtype import NA

N = 40_000

def blocks():
    gc.collect()
    return sys.getallocatedblocks()

def run(fn, n):
    for _ in range(n):
        try: fn()
        except Exception: pass

bad = []
for L in (nd.Nullable,):
    dt = L(np.float64); db = L(np.bool_)
    a = np.arange(64.0).astype(dt); a[3] = NA
    b = np.arange(64.0).astype(dt)
    M = np.arange(16.0).reshape(4, 4).astype(dt)
    t = np.array([True] * 8, dtype=db); t[2] = NA

    watched = {
        "dtype": dt, "dtype.wrapped": dt.wrapped, "NA": NA,
        "np.add": np.add, "np.matmul": np.matmul, "np.sqrt": np.sqrt,
        "np.logical_or": np.logical_or, "np.maximum": np.maximum,
        "DType class": type(dt), "array a": a,
    }
    import numpy._core.umath as um
    watched["umath.clip"] = um.clip

    cases = [
        ("a + b",                lambda: a + b),
        ("a + 1.0",              lambda: a + 1.0),
        ("np.sqrt(a)",           lambda: np.sqrt(a)),
        ("np.add.reduce(a)",     lambda: np.add.reduce(a)),
        ("np.add.accumulate(a)", lambda: np.add.accumulate(a)),
        ("np.clip(a,1,50)",      lambda: np.clip(a, 1.0, 50.0)),
        ("M @ M",                lambda: M @ M),
        ("np.logical_or(t,t)",   lambda: np.logical_or(t, t)),
        ("np.maximum(a,b)",      lambda: np.maximum(a, b)),
        ("np.fmax(a,b)",         lambda: np.fmax(a, b)),
        ("np.logaddexp(a,b)",    lambda: np.logaddexp(a, b)),
        ("a[3] getitem",         lambda: a[3]),
        ("a[3] = NA",            lambda: a.__setitem__(3, NA)),
        ("L(np.float64)",        lambda: L(np.float64)),
        ("ERR astype NA",        lambda: a.astype(np.float64)),
        ("ERR np.dot",           lambda: np.dot(a, a)),
        ("ERR mask NA",          lambda: a[a > 2.0]),
    ]
    print(f"\n{L.__name__}   (N = {N:,})")
    for name, fn in cases:
        run(fn, 1000)
        rc0 = {k: sys.getrefcount(v) for k, v in watched.items()}
        b0 = blocks()
        run(fn, N)
        db_ = blocks() - b0
        rc = {k: sys.getrefcount(v) - rc0[k] for k, v in watched.items()}
        grown = {k: v for k, v in rc.items() if v > 2}
        leaky = db_ > 500 or grown
        if leaky:
            bad.append((L.__name__, name, db_, grown))
        print(f"  {'LEAKY' if leaky else 'ok   '} {name:24s} blocks {db_:+6d}"
              f"   refcount {grown if grown else 'unchanged'}")

print()
if bad:
    print("LEAKED:")
    for row in bad:
        print("   ", row)
else:
    print("NO LEAKS FOUND")
    print(f"  Leaked: {N:,} blocks per operation; the highest observed was {max(db_ for _, _, db_, _ in bad) if bad else 0} blocks.")
