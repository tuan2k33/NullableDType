import ctypes, numpy as np
print("numpy", np.__version__)

PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
PyCapsule_GetPointer.restype = ctypes.c_void_p
PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

LOOP = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_void_p),
                        ctypes.POINTER(ctypes.c_ssize_t),
                        ctypes.POINTER(ctypes.c_ssize_t),
                        ctypes.c_void_p)

def get_loop(ufname, stride):
    uf = getattr(np, ufname)
    f8 = np.dtype('f8')
    _, cap = uf._resolve_dtypes_and_context((f8, f8, f8))
    uf._get_strided_loop(cap, fixed_strides=(stride, stride, stride))
    p = PyCapsule_GetPointer(cap, b"numpy_1.24_ufunc_call_info")
    arr = (ctypes.c_void_p * 3).from_address(p)   # strided_loop, context, auxdata
    return cap, LOOP(arr[0]), arr[1], arr[2]

def call(ufname, stride, n, aval, bval):
    cap, loop, ctx, aux = get_loop(ufname, stride)
    nb = stride*n + 64
    A = (ctypes.c_char*nb)(); B = (ctypes.c_char*nb)(); O = (ctypes.c_char*nb)()
    def poke(buf,i,v):
        ctypes.cast(ctypes.byref(buf, i*stride), ctypes.POINTER(ctypes.c_double))[0]=v
    def peek(buf,i):
        return ctypes.cast(ctypes.byref(buf, i*stride), ctypes.POINTER(ctypes.c_double))[0]
    for i in range(n):
        poke(A,i,aval); poke(B,i,bval); poke(O,i,-999.0)
    data = (ctypes.c_void_p*3)(ctypes.cast(A, ctypes.c_void_p).value,
                               ctypes.cast(B, ctypes.c_void_p).value,
                               ctypes.cast(O, ctypes.c_void_p).value)
    dims = (ctypes.c_ssize_t*1)(n)
    strides = (ctypes.c_ssize_t*3)(stride, stride, stride)
    rc = loop(ctx, data, dims, strides, aux)
    assert rc == 0, rc
    return [peek(O,i) for i in range(n)]

for name in ["add","maximum","minimum","fmax","fmin","multiply"]:
    exp_one = float(getattr(np,name)(np.float64(100.0), np.float64(-1.0)))
    for stride in (8,9,12,16,24):
        for n in (1,2,3,4,5,7,8,9,16,17,33):
            got = call(name, stride, n, 100.0, -1.0)
            if any(g != exp_one for g in got):
                print(f"MISMATCH {name:8s} stride={stride:3d} n={n:3d} -> {got}")
print("done")
