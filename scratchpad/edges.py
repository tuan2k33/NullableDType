import numpy as np, nulldtype as nd
from nulldtype import NA

for L in (nd.Nullable,):
    dt = L(np.float64)
    print(L.__name__)
    def t(tag, fn):
        try:
            r = fn()
            print(f"    {tag:28s} {str(r)[:52]}")
        except Exception as e:
            print(f"    {tag:28s} {type(e).__name__}: {str(e)[:46]}")
    empty = np.array([], dtype=dt)
    zerod = np.array(5.0, dtype=dt)
    zerona = np.array(5.0, dtype=dt); zerona[()] = NA
    allna = np.array([1.0, 2.0], dtype=dt); allna[0] = allna[1] = NA
    big2d = np.zeros((0, 3), dtype=dt)

    t("empty + empty",      lambda: empty + empty)
    t("empty.sum()",        lambda: empty.sum())
    t("nd.median(empty)",   lambda: nd.median(empty))
    t("nd.unique(empty)",   lambda: nd.unique(empty))
    t("empty @ empty",      lambda: empty @ empty)
    t("(0,3) @ (3,2)",      lambda: big2d @ np.zeros((3, 2), dtype=dt))
    t("0-d + 1.0",          lambda: zerod + 1.0)
    t("0-d NA + 1.0",       lambda: zerona + 1.0)
    t("0-d item",           lambda: zerona[()])
    t("np.sqrt(0-d NA)",    lambda: np.sqrt(zerona))
    t("allna.sum()",        lambda: allna.sum())
    t("nd.unique(allna)",   lambda: nd.unique(allna))
    t("np.sort(allna)",     lambda: np.sort(allna))
    t("allna.argmax()",     lambda: allna.argmax())
    t("np.clip(empty,0,1)", lambda: np.clip(empty, 0.0, 1.0))
    t("reshape(0)",         lambda: empty.reshape(0))
    t("np.add.reduce(empty)", lambda: np.add.reduce(empty))
