import numpy as np, nulldtype as nd, warnings
from nulldtype import NA
warnings.simplefilter("ignore")

dt = nd.Nullable(np.float64)
def mk(vals=(1.0, 2.0, 3.0, 4.0), gap=2):
    a = np.array(list(vals), dtype=dt)
    if gap is not None: a[gap] = NA
    return a

M = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=dt)
ok, bad = [], []
def t(tag, fn):
    try:
        r = fn()
        ok.append(f"  {tag:24s} {str(r)[:56]}")
    except Exception as e:
        bad.append(f"  {tag:24s} {type(e).__name__}: {str(e)[:52]}")

# shape / structural
for tag, fn in [
    ("reshape",      lambda: mk().reshape(2,2)),
    ("transpose",    lambda: M.T),
    ("ravel",        lambda: M.ravel()),
    ("concatenate",  lambda: np.concatenate([mk(), mk()])),
    ("stack",        lambda: np.stack([mk(), mk()])),
    ("repeat",       lambda: np.repeat(mk(), 2)),
    ("tile",         lambda: np.tile(mk(), 2)),
    ("roll",         lambda: np.roll(mk(), 1)),
    ("flip",         lambda: np.flip(mk())),
    ("where",        lambda: np.where(np.array([1,0,1,0],bool), mk(), mk())),
    ("take",         lambda: np.take(mk(), [0,2])),
    ("put/setitem",  lambda: (lambda a: (a.__setitem__(0, 9.0), a)[1])(mk())),
    ("fancy index",  lambda: mk()[[0,2,3]]),
    ("bool index",   lambda: mk()[nd.notna(mk())]),
    ("resize/pad",   lambda: np.pad(mk(), (1,1))),
    ("insert",       lambda: np.insert(mk(), 1, 9.0)),
    ("delete",       lambda: np.delete(mk(), 1)),
    ("searchsorted", lambda: np.searchsorted(np.sort(mk()), 2.0)),
    ("astype f8",    lambda: mk(gap=None).astype(np.float64)),
    ("tolist",       lambda: mk().tolist()),
    ("copy",         lambda: mk().copy()),
    ("np.array_equal", lambda: np.array_equal(mk(), mk())),
    ("np.diff",      lambda: np.diff(mk())),
    ("cumsum",       lambda: np.cumsum(mk())),
    ("cumprod",      lambda: np.cumprod(mk())),
    ("np.sort",      lambda: np.sort(mk())),
    ("argsort",      lambda: np.argsort(mk())),
    ("np.dot",       lambda: np.dot(mk(), mk())),
    ("matmul",       lambda: M @ M),
    ("np.clip",      lambda: np.clip(mk(), 1.5, 3.5)),
    ("np.round",     lambda: np.round(mk())),
    ("np.abs",       lambda: np.abs(mk())),
    ("einsum",       lambda: np.einsum('i->', mk())),
    ("np.histogram", lambda: np.histogram(mk())),
    ("np.interp",    lambda: np.interp(0.5, [0,1], [0,1])),
    ("2D sum axis=0",lambda: M.sum(axis=0)),
    ("np.save/load", lambda: (np.save('/tmp/_t.npy', mk()), np.load('/tmp/_t.npy', allow_pickle=True))[1]),
    ("repr 2D",      lambda: repr(M)),
    ("nonzero",      lambda: np.nonzero(mk(gap=None))),
    ("count_nonzero",lambda: np.count_nonzero(mk(gap=None))),
]:
    t(tag, fn)

print("=== CHAY DUOC ===");  print("\n".join(ok))
print("\n=== KHONG ===");    print("\n".join(bad))
