"""Time `a + a` plain vs Nullable, and the raw cost of scanning for NA.

    python scratchpad/bench_fixup.py
"""
import timeit, numpy as np, nulldtype as nd
def best(stmt, n):
    t = timeit.Timer(stmt)
    loops = max(1, int(2e6 // n))
    return min(t.repeat(9, loops)) / loops * 1e6   # microseconds
print(f"numpy {np.__version__}")
print(f"{'dtype':6} {'n':>9} {'plain':>9} {'nullable':>9} {'ratio':>6} | {'scan ==':>9} {'scan isnan':>10} {'1 gap':>9}")
for t, u in ((np.float64, np.uint64), (np.int64, np.uint64), (np.int32, np.uint32)):
    for n in (1_000, 100_000, 1_000_000, 10_000_000):
        p = (np.arange(n) % 1000).astype(t)
        a = p.astype(nd.Nullable(t)); o = np.empty_like(a); po = np.empty_like(p)
        g = a.copy(); g[n // 2] = nd.NA
        v = a.view(u); pat = u(np.iinfo(u).max >> 1) if t is np.float64 else u(np.iinfo(t).min % (1 << (8*np.dtype(t).itemsize)))
        tp = best(lambda: np.add(p, p, out=po), n)
        tn = best(lambda: np.add(a, a, out=o), n)
        ts = best(lambda: (v == pat).any(), n)
        ti = best(lambda: np.isnan(po).any(), n) if t is np.float64 else float('nan')
        tg = best(lambda: np.add(g, g, out=o), n)
        print(f"{np.dtype(t).name:6} {n:9d} {tp:9.1f} {tn:9.1f} {tn/tp:6.1f} | {ts:9.1f} {ti:10.1f} {tg:9.1f}")
