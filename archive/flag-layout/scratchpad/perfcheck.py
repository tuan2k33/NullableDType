import numpy as np, timeit, nulldtype as nd
def t(fn, number=5):
    return min(timeit.repeat(fn, number=number, repeat=7)) / number

print(f"{'n':>10} {'op':>10} {'numpy':>9} {'mau bit':>9} {'co':>9}   bit/numpy")
for n in (1_000, 100_000, 5_000_000):
    plain = np.random.rand(n)
    sent = plain.astype(nd.BitpatternLayout(np.float64))
    flag = plain.astype(nd.FlagLayout(np.float64))
    for label, fn in (("a + a", lambda x: x + x), ("a + 1.0", lambda x: x + 1.0),
                      ("sqrt", lambda x: np.sqrt(x))):
        b = t(lambda: fn(plain)); s = t(lambda: fn(sent)); f = t(lambda: fn(flag))
        print(f"{n:>10,} {label:>10} {b*1e6:8.1f}u {s*1e6:8.1f}u {f*1e6:8.1f}u   {s/b:6.2f}x")
print()
print("itemsize sau khi sua promoter (truoc day scalar lam tut xuong 9):")
sent = np.random.rand(100).astype(nd.BitpatternLayout(np.float64))
print("  sent+sent", (sent+sent).itemsize, " sent+1.0", (sent+1.0).itemsize, " sqrt", np.sqrt(sent).itemsize)
