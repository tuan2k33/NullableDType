import gc, sys
import numpy as np
import nulldtype as nd

def blocks(fn, n=3000):
    for _ in range(100):
        fn()
    gc.collect(); b0 = sys.getallocatedblocks()
    for _ in range(n):
        fn()
    gc.collect()
    return sys.getallocatedblocks() - b0

def raises(fn, exc=ValueError):
    def g():
        try:
            fn()
        except exc:
            return
        raise AssertionError("did not raise")
    return g

def filled(dt, items):
    a = np.zeros(len(items), dtype=dt)
    for i, v in enumerate(items):
        a[i] = v
    return a

rec = np.dtype([("a", "i4"), ("b", "f8")])
s3 = filled(nd.Nullable("S3"), [b"zz", nd.NA, b"ab", b"q"])
u10f = filled(nd.Nullable("U10"), ["zz", nd.NA, "ab", "q"])
v4 = filled(nd.Nullable("V4"), [np.void(b"zzzz"), nd.NA, np.void(b"abcd")])
recs = filled(nd.Nullable(rec), [(2, 1.0), nd.NA, (1, 5.0)])
big_na = np.zeros(2000, dtype=nd.Nullable(np.float64)); big_na[1500] = nd.NA
s5 = filled(nd.Nullable("S5"), [b"\xff\xff\xffab"])
i8 = np.array([2**31], dtype=np.int64)
i4s = np.array([1, -2, 300], dtype=np.int32).astype(nd.Nullable(np.int32))
i4f = np.array([1, -2, 300], dtype=np.int32).astype(nd.Nullable(np.int32))
bigf8 = np.array([3.0, 1.0], dtype=">f8")
vflag = np.zeros(2, dtype=nd.Nullable("V4"))
rflag = np.zeros(2, dtype=nd.Nullable(rec))
rscalar = np.array((3, 4.0), rec)[()]
plain_recs = np.array([(1, 2.0), (5, 6.0)], rec)
sa = filled(nd.Nullable("S3"), [b"ab", nd.NA, b"x"])
sb = filled(nd.Nullable("S3"), [b"ab", b"q", nd.NA])

def set_reserved():
    s3[0] = b"\xff\xff\xff"

def set_void():
    vflag[0] = np.void(b"zz\xfe\xff")

def set_rec():
    rflag[0] = rscalar
    rflag[:] = plain_recs

cases = {
    "sort S3": lambda: np.sort(s3),
    "argsort U10": lambda: np.argsort(u10f),
    "argmax/argmin V4": lambda: (v4.argmax(), v4.argmin()),
    "sort + argmax records": lambda: (np.sort(recs), recs.argmax()),
    "nonzero records raises": raises(lambda: np.nonzero(recs)),
    "nonzero 2000 f8 raises": raises(lambda: np.nonzero(big_na)),
    "setitem reserved S3 raises": raises(set_reserved),
    "cast S5->S3 lands on NA raises": raises(lambda: s5.astype(nd.Nullable("S3"))),
    "cast int64->Nullable(int32) raises": raises(lambda: i8.astype(nd.Nullable(np.int32))),
    "i4 -> f8 / i16": lambda: (i4s.astype(np.float64), i4s.astype(np.int16)),
    "i4 -> i16 again": lambda: (i4f.astype(np.float64), i4f.astype(np.int16)),
    "from Nullable NA raises": raises(lambda: s3.astype("S3")),
    "construct Nullable('>i4')": lambda: nd.Nullable(">i4"),
    "big-endian f8 in and out": lambda: bigf8.astype(nd.Nullable(">f8")).astype(">f8"),
    "np.void scalar assign": set_void,
    "record scalar/array assign": set_rec,
    "S3 == and +": lambda: (sa == sb, sa + sb),
    "isna U10 / S3": lambda: (nd.isna(u10f), nd.isna(s3)),
}
worst = 0
for name, fn in cases.items():
    d = blocks(fn)
    worst = max(worst, d)
    print(f"{name:38s} {d:+d} blocks over 3000 calls")
print("worst", worst)
