# numpy.ma pitfalls

This is a catalogue of `numpy.ma` behaviours that leak masked data, give wrong
answers without a warning, raise, or surprise. Each entry gives the cause and
the matching numpy issue. Every result below was reproduced on numpy 2.5.3
with `scratchpad/ma_pitfalls.py`.

The array used throughout:

```python
m = np.ma.masked_array([3.0, 1.0, 12345.0, 5.0], mask=[0, 0, 1, 0])
```

12345.0 sits under the mask so that it is obvious wherever it escapes.

## Leaks: the masked value comes out looking like data

| Call | Result | Cause | Issue |
|---|---|---|---|
| `np.asarray(m)[2]`, `np.array(m)[2]`, `m.data[2]` | `12345.0` | the value under the mask is kept, and `asarray` drops the subclass and its mask with it | [#26669](https://github.com/numpy/numpy/issues/26669) |
| `np.concatenate([m, m])` | a MaskedArray whose mask is `False`: 12345.0 is visible | the function does not know about the mask, and `MaskedArray` does not implement `__array_function__` | [#22338](https://github.com/numpy/numpy/issues/22338) |
| `np.append(m, 1.)[2]`, `np.stack([m, m])[0, 2]` | `12345.0`, unmasked | as above | [#22338](https://github.com/numpy/numpy/issues/22338) |
| `np.where(m > 2, m, 0)` | plain `[3, 0, 12345, 5]` | as above; the result is a plain ndarray | [#18675](https://github.com/numpy/numpy/issues/18675) |
| `np.pad(m, 1)` | plain array containing 12345 | as above | [#8881](https://github.com/numpy/numpy/issues/8881) |
| `np.save` then `np.load` | mask gone, 12345 is data | `.npy` stores one buffer | [#18134](https://github.com/numpy/numpy/issues/18134) |
| `m.astype(int).data[2]` | `12345` | the cast runs on the hidden value | — |
| `(m + 1).data[2]` | `12345.0`, not 12346 | under the mask the first operand's data is copied back, by convention | — |
| `child.mask = nomask` on a slice | the **parent** loses its mask and shows 12345 | views share the parent's mask array | [#7781](https://github.com/numpy/numpy/issues/7781) |

## Silently wrong

| Call | Result | Expected | Cause | Issue |
|---|---|---|---|---|
| `np.dot(m, m)` | `152399060.0` (includes 12345²) | 35, which `ma.dot` gives | `np.dot` goes straight to C and ignores the mask | [#5739](https://github.com/numpy/numpy/issues/5739) |
| `np.interp(2.5, [0, 1, 2, 3], m)` | `6175.0` | interpolated from the unmasked points | as above | — |
| `np.histogram(m, 2)` | bin edges up to 12345 | built from 3, 1, 5 | as above | [#10019](https://github.com/numpy/numpy/issues/10019) |
| `np.count_nonzero([0, (1)])` | `1` | 0 | counts the masked element | [#18573](https://github.com/numpy/numpy/issues/18573) |
| `np.polyfit` with a masked point (2, 99) | slope `49.5` | 1.0 | fits the hidden value | [#15807](https://github.com/numpy/numpy/issues/15807), [#9193](https://github.com/numpy/numpy/issues/9193) |
| `np.trapezoid(m)` | `2.0` | not meaningful | integrates filled or unmasked data | [#9735](https://github.com/numpy/numpy/issues/9735) |
| `arr[masked_index]` | the masked index is used | skipped or refused | numpy reads the data buffer | [#20933](https://github.com/numpy/numpy/issues/20933) |
| `ma.isin([1, (2)], [2])` | `[False, False]`, unmasked | the masked element masked | the result does not carry the mask | [#19877](https://github.com/numpy/numpy/issues/19877) |
| `np.ptp` of an all-masked array | `1.0`, **unmasked** | masked | the reduction runs on the hidden values | [#27674](https://github.com/numpy/numpy/issues/27674) |
| `(masked 0-d) == 1` | dtype `float64` | bool | typing rules of the `masked` constant | [#21944](https://github.com/numpy/numpy/issues/21944) |

## Raises

| Call | Error | Cause | Issue |
|---|---|---|---|
| `np.quantile(m, .5)` | `ValueError: output array is read-only`, after "partition will ignore the mask" | a pure-Python numpy function meets the subclass | [#14716](https://github.com/numpy/numpy/issues/14716), [#4767](https://github.com/numpy/numpy/issues/4767) |
| `np.nanmean` of an all-masked array | `ValueError: output array is read-only` | as above | [#29117](https://github.com/numpy/numpy/issues/29117), [#17310](https://github.com/numpy/numpy/issues/17310) |
| `int(masked element)` | `MaskError` | while `float()` of the same element gives `nan` (see below) | [#11674](https://github.com/numpy/numpy/issues/11674) |

## Surprising

| Call | Result | Problem | Issue |
|---|---|---|---|
| `ma.array([1.]) / 0`, `0 / 0`, `ma.log(-1)` | masked automatically | a real arithmetic error becomes a "missing" element | [#5384](https://github.com/numpy/numpy/issues/5384), [#22347](https://github.com/numpy/numpy/issues/22347) |
| `ma.array([1e308]) * 10` | `inf`, **not** masked, with a warning | inconsistent with division | [#8003](https://github.com/numpy/numpy/issues/8003) |
| `x += nan` | NaN is not masked | inconsistent with `0 / 0` | [#20506](https://github.com/numpy/numpy/issues/20506) |
| `np.log` of a negative that **is** masked | `RuntimeWarning: invalid value` | the hidden value is still computed | [#4959](https://github.com/numpy/numpy/issues/4959) |
| `float(masked element)` | `nan`, with a warning | while `int()` raises | [#11674](https://github.com/numpy/numpy/issues/11674) |
| `.all()` / `.any()` of an all-masked array | `masked` | the meaning over nothing is not settled | [#11123](https://github.com/numpy/numpy/issues/11123) |
| `.argmax()` of an all-masked array | `0` | points at a masked element | [#12316](https://github.com/numpy/numpy/issues/12316) |
| `x.copy().set_fill_value(7)` | the **original's** `fill_value` becomes 7 | copies share `fill_value` | [#7329](https://github.com/numpy/numpy/issues/7329) |
| `np.median(m)` | correct (`3.0`) but warns "partition will ignore the mask" | correct only through a special case | [#7330](https://github.com/numpy/numpy/issues/7330) |

## Root causes

1. **A subclass of ndarray that changes its meaning.** A numpy function that
   does not know about masks either drops the mask (`asarray`, `where`, `pad`,
   `save`) or computes with the hidden values (`dot`, `interp`, `histogram`,
   `polyfit`). `MaskedArray` implements no `__array_function__` to intercept
   them.
2. **The value under the mask is kept (IGNORED semantics).** Any path that
   reads `.data` hands it out as if it were data.
3. **Computation runs on the hidden values too.** This gives spurious
   warnings, and whatever is left under the mask is arbitrary.
4. **Invalid results are masked automatically.** Real errors disappear, and
   the rule differs from one operation to the next.
5. **Shared state.** Views share the mask with their parent, and copies share
   `fill_value` with the original.
6. **The all-masked case is unsettled.** `all`, `any`, `argmax`, `ptp` and
   `nanmean` each do something different.

## Why not just re-apply the mask to the output?

That is what `np.ma` already does for elementwise operations: it computes on
the data, then ORs the masks for the result. It stops working in three places.

- **Functions outside `np.ma` do not know there is a mask.** `np.pad`,
  `np.where`, `np.save` and `np.dot` start with `np.asarray`, and no step
  afterwards puts a mask back. Every function would need its own wrapper, and
  numpy has hundreds; `np.ma` wraps some of them.
- **The output mask is not always derivable, and the result may already be
  contaminated.** Re-masking works when output element `i` comes from input
  element `i`. For `np.dot`, the masked value is already inside the sum. For
  `np.histogram`, `np.median` and `np.polyfit`, it has already shaped the
  bins, the order or the fit. The mask has to be applied before and during the
  computation, not after it.
- **A leak is data leaving the container.** `np.asarray(m)`, `m.data` and
  `np.save` hand raw data to the user or to a file. There is no output left to
  re-mask, and because `np.ma` keeps the masked value, what comes out looks
  real.

## Does erasing the masked data fix it?

Only half of it. Here is the same array with the value under the mask
overwritten by 0, next to a Nullable array holding NA there (numpy 2.5.3):

| Call | `np.ma`, masked value erased to 0 | Nullable |
|---|---|---|
| `np.asarray(x)[2]` | `0.0` | `NA` |
| `np.mean(np.asarray(x))` | `2.25`, wrong | `NA` |
| `np.histogram(x, 2)` | edges from 0: wrong | raises: `bool(NA)` has no answer |
| `np.polyfit([0, 1, 2, 3], x, 1)` | `[0.5, 1.5]`, fits the 0: wrong | raises: NA cannot become a float |
| `np.dot(x, x)` | `35.0`, right only because 0 adds nothing | refuses (see `NUMPY-PATCHES.md`); `@` works |
| `np.pad(x, 1)` | `[0, 3, 1, 0, 5, 0]`: the gap is indistinguishable from padding | `[0.0 3.0 1.0 NA 5.0 0.0]` |
| `np.where(x > 2, x, 0)` | `[3, 0, 0, 5]` | raises `DTypePromotionError` (the Python-scalar gap in `README.md`) |

Erasing stops the **real value** from leaking, but once the mask is lost the
erased cell still reads as a plausible number: a 0 that is wrong. Two things
are needed together:

1. **Nothing to leak.** The masked value is not stored.
2. **The marker travels with the data.** Missingness lives in the element's
   own bytes, so any path that copies the data copies the NA with it. Code
   that does not understand NA then either propagates it (ufunc loops,
   `np.pad`) or refuses (`bool(NA)`, casting out), instead of computing with
   a stand-in.

A mask kept beside the data can only ever have the first property. A
bit-pattern dtype has both. What remains for it are the numpy functions that
decide things in Python on sorted data (`np.median`, `np.quantile`), which
return wrong numbers and need hooks in numpy (`NUMPY-PATCHES.md`), and a few
that raise where an answer would be possible.
