"""Reproduce every numpy.ma behaviour listed in NUMPY-MA-PITFALLS.md.

    python scratchpad/ma_pitfalls.py

Each line is `label | result | warnings`.  `m` is [3.0, 1.0, 12345.0, 5.0] with
the third element masked; 12345.0 is there to show where it leaks.
"""
import io
import warnings

import numpy as np

ma = np.ma


def m():
    return ma.masked_array([3.0, 1.0, 12345.0, 5.0], mask=[0, 0, 1, 0])


def run(label, fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = repr(fn()).replace("\n", " ")
        except Exception as e:
            out = f"raise {type(e).__name__}: {str(e)[:70]}"
    ws = "; ".join(sorted({f"{w.category.__name__}: {str(w.message)[:45]}"
                           for w in caught}))
    print(f"{label:36s} | {' '.join(out.split())[:100]:100s} | {ws}")


def view_nomask():
    parent = m()
    child = parent[1:]
    child.mask = ma.nomask
    return parent.mask


def save_load():
    f = io.BytesIO()
    np.save(f, m())
    f.seek(0)
    return np.load(f)


def copy_fill_value():
    x = ma.masked_array([1.0], fill_value=3.0)
    x.copy().set_fill_value(7.0)
    return x.fill_value


def iadd_nan():
    x = ma.array([1.0])
    x += np.nan
    return x.mask


print(f"numpy {np.__version__}\n")
print("-- leaks")
run("np.asarray(m)[2]", lambda: np.asarray(m())[2])
run("np.array(m)[2]", lambda: np.array(m())[2])
run("m.data[2]", lambda: m().data[2])
run("np.concatenate([m, m]) mask", lambda: ma.getmask(np.concatenate([m(), m()])))
run("np.append(m, 1.)[2]", lambda: np.append(m(), 1.0)[2])
run("np.stack([m, m])[0, 2]", lambda: np.stack([m(), m()])[0, 2])
run("np.where(m > 2, m, 0)", lambda: np.where(m() > 2, m(), 0.0))
run("np.pad(m, 1)", lambda: np.pad(m(), 1))
run("np.save / np.load", save_load)
run("m.astype(int).data[2]", lambda: m().astype(int).data[2])
run("(m + 1).data[2]", lambda: (m() + 1).data[2])
run("parent mask after view.mask=nomask", view_nomask)

print("\n-- silently wrong")
run("np.dot(m, m)", lambda: np.dot(m(), m()))
run("ma.dot(m, m)", lambda: ma.dot(m(), m()))
run("np.interp(2.5, [0,1,2,3], m)", lambda: np.interp(2.5, [0, 1, 2, 3], m()))
run("np.histogram(m, 2) edges", lambda: np.histogram(m(), 2)[1])
run("np.count_nonzero([0, (1)])", lambda: np.count_nonzero(ma.masked_array([0.0, 1.0], mask=[0, 1])))
run("np.polyfit, (2, 99) masked", lambda: np.polyfit([0, 1, 2], ma.masked_array([0.0, 1.0, 99.0], mask=[0, 0, 1]), 1))
run("np.trapezoid(m)", lambda: np.trapezoid(m()))
run("arange(3)[masked index]", lambda: np.arange(3)[ma.masked_array([0, 1, 2], mask=[0, 1, 0])])
run("ma.isin([1, (2)], [2])", lambda: ma.isin(ma.masked_array([1, 2], mask=[0, 1]), [2]))
run("np.ptp, all masked", lambda: np.ptp(ma.masked_array([1.0, 2.0], mask=[1, 1])))
run("(0-d masked == 1).dtype", lambda: (ma.masked_array(1, mask=True) == 1).dtype)

print("\n-- raises")
run("np.quantile(m, .5)", lambda: np.quantile(m(), 0.5))
run("np.nanmean, all masked", lambda: np.nanmean(ma.masked_array([1.0, 2.0], mask=[1, 1])))
run("int(masked element)", lambda: int(ma.masked_array([1], mask=[1])[0]))

print("\n-- surprising")
run("ma.array([1.]) / 0", lambda: ma.array([1.0]) / 0)
run("ma.array([0.]) / 0", lambda: ma.array([0.0]) / 0)
run("ma.log(ma.array([-1.]))", lambda: ma.log(ma.array([-1.0])))
run("ma.array([1e308]) * 10", lambda: ma.array([1e308]) * 10)
run("x += nan -> mask", iadd_nan)
run("np.log on a masked negative", lambda: np.log(ma.masked_array([-1.0, 1.0], mask=[1, 0])))
run("float(masked element)", lambda: float(ma.masked_array([1.0], mask=[1])[0]))
run(".all(), all masked", lambda: ma.masked_array([False], mask=[1]).all())
run(".any(), all masked", lambda: ma.masked_array([True], mask=[1]).any())
run(".argmax(), all masked", lambda: ma.masked_array([1.0, 2.0], mask=[1, 1]).argmax())
run("fill_value after copy().set", copy_fill_value)
run("np.median(m)", lambda: np.median(m()))
