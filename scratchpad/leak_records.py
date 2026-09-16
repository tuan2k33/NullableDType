"""Allocation growth on the record paths of the bitpattern layout."""
import gc
import sys
import warnings

import numpy as np
import nulldtype as nd

REC = np.dtype([("i", "i4"), ("f", "f8"), ("s", "S3"), ("t", "M8[s]")], align=True)
INNER = np.dtype([("x", "i2"), ("y", "f4")])
NESTED = np.dtype([("p", INNER), ("q", "f8", (3,)), ("r", INNER, (2,))])
LD = np.dtype([("a", "i2"), ("g", "g"), ("z", "G", (2,))], align=True)
OBJ = np.dtype([("a", "i4"), ("b", "O")])
PAD = np.dtype({"names": [], "formats": [], "itemsize": 4})


def blocks(fn, n=3000):
    for _ in range(100):
        fn()
    gc.collect()
    b0 = sys.getallocatedblocks()
    for _ in range(n):
        fn()
    gc.collect()
    return sys.getallocatedblocks() - b0


def raises(fn, exc=(ValueError, TypeError)):
    def g():
        try:
            fn()
        except exc:
            return
        raise AssertionError("did not raise")
    return g


a = np.zeros(4, dtype=nd.Nullable(REC))
a[0] = (3, 1.5, b"ab", 5)
a[1] = nd.NA
a[2] = (1, 2.5, b"x", 7)
nested = np.zeros(3, dtype=nd.Nullable(NESTED))
nested[1] = nd.NA
gap = np.frombuffer(nested[1:2].tobytes(), dtype=NESTED).copy()
plain = np.array([(1, 2.0, b"a", 1), (3, 4.0, b"b", 2)], REC)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", nd.LongDoubleWarning)
    ld_plain = np.zeros(2, dtype=LD)


def set_valid():
    a[3] = (9, 9.0, b"z", 9)


def set_all_na():
    b = np.zeros(2, dtype=nd.Nullable(REC))
    b[...] = nd.NA


def ld_descr():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nd.LongDoubleWarning)
        return ld_plain.astype(nd.Nullable(LD))


cases = {
    "make Nullable(record) descr": lambda: nd.Nullable(REC),
    "make nested descr": lambda: nd.Nullable(NESTED),
    "object field refused": raises(lambda: nd.Nullable(OBJ)),
    "padding-only refused": raises(lambda: nd.Nullable(PAD)),
    "long double fields rebuilt + cast": ld_descr,
    "getitem value / NA": lambda: (a[0], a[1]),
    "setitem tuple": set_valid,
    "a[...] = NA": set_all_na,
    "cast NA record in raises": raises(lambda: gap.astype(nd.Nullable(NESTED))),
    "cast in and out": lambda: plain.astype(nd.Nullable(REC)).astype(REC),
    "sort / argmax / argmin": lambda: (np.sort(a), a.argmax(), a.argmin()),
    "nonzero raises": raises(lambda: np.nonzero(a)),
    "isna / filled": lambda: (nd.isna(nested), nd.filled(a, (0, 0.0, b"", 0))),
}
refs = {"REC": REC, "NA": nd.NA, "INNER": INNER}
before = {k: sys.getrefcount(v) for k, v in refs.items()}
worst = 0
for name, fn in cases.items():
    d = blocks(fn)
    worst = max(worst, d)
    print(f"{name:36s} {d:+d} blocks / 3000")
after = {k: sys.getrefcount(v) for k, v in refs.items()}
print("refcounts before", before, "after", after)
print("worst", worst)
