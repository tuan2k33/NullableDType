"""Leak scaling for the wrapped-loop cache: block counts and refcounts must stay
flat as the iteration count grows, including while the cache evicts (more
distinct strides and dtypes than it has slots).

    python scratchpad/leak_loopcache.py
"""
import gc, sys, numpy as np, nulldtype as nd

def blocks():
    gc.collect()
    return sys.getallocatedblocks()

def workload():
    for t in ("f8", "f4", "i4", "i8", "u2"):
        dt = nd.Nullable(t)
        a = np.arange(200).astype(dt)
        a[5] = nd.NA
        for step in range(1, 12):              # many strides -> evictions
            v = a[::step]
            v + v
            v * v
            np.maximum(v, v)
        a + a[::-1]
        a.sum()
        np.add.accumulate(a)
    f = np.arange(50.0).astype(nd.Nullable("f8"))
    for _ in range(50):
        f + f

workload()                                     # warm up: fills the cache
before_refs = sys.gettotalrefcount() if hasattr(sys, "gettotalrefcount") else None
sizes = []
for n in (50, 200, 800):
    b0 = blocks()
    for _ in range(n):
        workload()
    sizes.append((n, blocks() - b0))
print(sizes)
growth = [d for _, d in sizes]
bad = growth[-1] > growth[0] + 50            # not proportional to n
print("LEAKY" if bad else "flat")
sys.exit(1 if bad else 0)
