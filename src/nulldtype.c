/*
 * Nullable[T] -- a parametric NumPy DType that gives any other dtype a notion
 * of "missing".  NA is a reserved value of `T` itself, so an element costs
 * exactly what `T` costs, the values stay contiguous and aligned, and the
 * wrapped dtype's SIMD loops keep working at full speed.
 *
 *   bool           the byte 2, which numpy never produces
 *   intN           INT_MIN
 *   uintN          UINT_MAX
 *   float, complex every bit but the sign (0x7FFF...), both halves for
 *                  complex; long double is stored as double
 *   datetime64,    INT64_MIN, which numpy already calls NaT
 *   timedelta64
 *   S, V           every byte 0xFF, at any width
 *   U              every character U+FFFF, the noncharacter Unicode sets
 *                  aside for internal use
 *   records        every field its own type's NA, padding zeroed; missing
 *                  only when every field is
 *
 * The full table with byte sizes is in LAYOUTS.md.  NumPy already ships one of
 * these: `NaT` for datetime64.  This applies the idea to every dtype.
 *
 * NA is not left to hardware NaN propagation.  The loops read the inputs and
 * write NA into the output themselves, so `nan + NA` is NA whichever side it
 * is on, and the scheme does not depend on the CPU preserving NaN payloads.
 *
 * An earlier version had a second layout -- the value followed by a validity
 * byte -- for types with no spare value.  Once strings, raw bytes and records
 * got patterns nothing common needed it; it lives on in archive/flag-layout.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/dtype_api.h"

static PyArray_DTypeMeta NullableDType;

/* the plain dtypes a Nullable one is the common dtype of */
#define N_WRAPPABLE 16
static const int wrappable_typenums[N_WRAPPABLE] = {
    NPY_BOOL, NPY_INT8, NPY_UINT8, NPY_INT16, NPY_UINT16, NPY_INT32,
    NPY_UINT32, NPY_INT64, NPY_UINT64, NPY_FLOAT16, NPY_FLOAT32, NPY_FLOAT64,
    NPY_COMPLEX64, NPY_COMPLEX128, NPY_DATETIME, NPY_TIMEDELTA,
};


/* The Python-level marker for a missing value: `nulldtype.NA` */
static PyObject *NA_singleton = NULL;

static int same_value_dtype(PyArray_Descr *a, PyArray_Descr *b);


/*
 * Nullable has no long double.  `longdouble` and `clongdouble` are stored as
 * `float64` and `complex128`: one float format on every platform, where the
 * real long double is x87 80-bit on x86-64, IEEE quad on aarch64 and a plain
 * double under MSVC.  A `LongDoubleWarning` says so wherever the substitution
 * loses anything; where long double already is a double it stays silent.
 *
 * Long doubles inside a record are replaced the same way.
 *
 * Returns a new reference: the substitute, or `wrapped` itself.
 */
static PyObject *LongDoubleWarning = NULL;

static int
contains_long_double(PyArray_Descr *descr)
{
    if (descr->type_num == NPY_LONGDOUBLE || descr->type_num == NPY_CLONGDOUBLE) {
        return 1;
    }
    if (PyDataType_HASSUBARRAY(descr)) {
        return contains_long_double(PyDataType_SUBARRAY(descr)->base);
    }
    if (PyDataType_HASFIELDS(descr)) {
        PyObject *key, *info;
        Py_ssize_t pos = 0;
        while (PyDict_Next(PyDataType_FIELDS(descr), &pos, &key, &info)) {
            if (contains_long_double((PyArray_Descr *)PyTuple_GET_ITEM(info, 0))) {
                return 1;
            }
        }
    }
    return 0;
}


/*
 * `descr` with every long double in it replaced by a double, as a new reference;
 * `*lossy` is set when that loses precision.  A record is rebuilt from its
 * field list, so the fields after a replaced one move down.  NumPy casts one
 * record to another field by field, in order, so values still land right.
 */
static PyArray_Descr *
without_long_double(PyArray_Descr *descr, int *lossy)
{
    if (descr->type_num == NPY_LONGDOUBLE || descr->type_num == NPY_CLONGDOUBLE) {
        PyArray_Descr *res = PyArray_DescrFromType(
                descr->type_num == NPY_LONGDOUBLE ? NPY_DOUBLE : NPY_CDOUBLE);
        if (res != NULL && descr->elsize > res->elsize) {
            *lossy = 1;
        }
        return res;
    }
    if (!contains_long_double(descr)) {
        Py_INCREF(descr);
        return descr;
    }
    PyObject *spec = NULL;
    if (PyDataType_HASSUBARRAY(descr)) {
        PyArray_ArrayDescr *sub = PyDataType_SUBARRAY(descr);
        PyArray_Descr *base = without_long_double(sub->base, lossy);
        if (base == NULL) {
            return NULL;
        }
        spec = Py_BuildValue("(NO)", (PyObject *)base, sub->shape);
    }
    else {
        PyObject *names = PyDataType_NAMES(descr);
        PyObject *fields = PyDataType_FIELDS(descr);
        Py_ssize_t n = PyTuple_GET_SIZE(names);
        spec = PyList_New(n);
        for (Py_ssize_t i = 0; spec != NULL && i < n; i++) {
            PyObject *name = PyTuple_GET_ITEM(names, i);
            PyObject *info = PyDict_GetItemWithError(fields, name);
            if (info == NULL) {
                if (!PyErr_Occurred()) {
                    PyErr_SetString(PyExc_RuntimeError,
                            "a record field is missing from its fields");
                }
                Py_CLEAR(spec);
                break;
            }
            PyArray_Descr *field = without_long_double(
                    (PyArray_Descr *)PyTuple_GET_ITEM(info, 0), lossy);
            if (field == NULL) {
                Py_CLEAR(spec);
                break;
            }
            PyObject *title = (PyTuple_GET_SIZE(info) > 2)
                    ? PyTuple_GET_ITEM(info, 2) : Py_None;
            PyObject *item = (title != Py_None)
                    ? Py_BuildValue("((OO)N)", title, name, (PyObject *)field)
                    : Py_BuildValue("(ON)", name, (PyObject *)field);
            if (item == NULL) {
                Py_CLEAR(spec);
                break;
            }
            PyList_SET_ITEM(spec, i, item);
        }
    }
    if (spec == NULL) {
        return NULL;
    }
    PyArray_Descr *res = NULL;
    int ok = (descr->flags & NPY_ALIGNED_STRUCT)
            ? PyArray_DescrAlignConverter(spec, &res)
            : PyArray_DescrConverter(spec, &res);
    Py_DECREF(spec);
    return ok ? res : NULL;
}


static PyArray_Descr *
substitute_long_double(PyArray_Descr *wrapped)
{
    const char *what;
    if (wrapped->type_num == NPY_LONGDOUBLE) {
        what = "longdouble is stored as float64";
    }
    else if (wrapped->type_num == NPY_CLONGDOUBLE) {
        what = "clongdouble is stored as complex128";
    }
    else if (PyDataType_HASFIELDS(wrapped) && contains_long_double(wrapped)) {
        what = "longdouble and clongdouble fields are stored as float64 and "
               "complex128";
    }
    else {
        Py_INCREF(wrapped);
        return wrapped;
    }
    int lossy = 0;
    PyArray_Descr *stored = without_long_double(wrapped, &lossy);
    if (stored == NULL) {
        return NULL;
    }
    if (lossy) {
        if (PyErr_WarnFormat(
                LongDoubleWarning ? LongDoubleWarning : PyExc_UserWarning, 1,
                "%s: Nullable has no long double, so values keep about 15 "
                "significant digits and a range of about +-1.8e308; beyond "
                "that they are rounded, or become inf or 0", what) < 0) {
            Py_DECREF(stored);
            return NULL;
        }
    }
    return stored;
}


/*
 * What a Nullable dtype stores for a requested dtype: long double substituted as
 * above, in native byte order.  Keeping `>i4` as given got two things wrong.
 * The reserved patterns are written in native order, so the NA bytes
 * 00 00 00 80 meant 128 to a big-endian reader and INT_MIN went in as an
 * ordinary value.  And `np.sort` asks for a byte-swapped copy of the array's
 * dtype, which a new-style DType cannot give; numpy does not check for NULL
 * there and crashed.  Big-endian data still comes in and goes out: numpy's own
 * casts swap it.
 *
 * Returns a new reference.
 */
static PyArray_Descr *
stored_dtype(PyArray_Descr *wrapped)
{
    PyArray_Descr *stored = substitute_long_double(wrapped);
    /* a record's own byte order is `|`; its fields carry theirs */
    if (stored == NULL
            || (PyArray_ISNBO(stored->byteorder) && !PyDataType_HASFIELDS(stored))) {
        return stored;
    }
    PyArray_Descr *native = PyArray_DescrNewByteorder(stored, NPY_NATIVE);
    Py_DECREF(stored);
    return native;
}


static int
is_wrappable_typenum(int type_num)
{
    for (int i = 0; i < N_WRAPPABLE; i++) {
        if (wrappable_typenums[i] == type_num) {
            return 1;
        }
    }
    return 0;
}


static PyObject *np_copyto = NULL;
static PyObject *np_asarray = NULL;
static PyObject *np_concatenate = NULL;


/*
 * np.copyto(dst, src, casting="unsafe", where=where); `where` may be NULL.
 * Unsafe because the caller has already decided the cast is wanted.
 */
static int
copy_values_where(PyObject *dst, PyObject *src, PyObject *where)
{
    PyObject *args = PyTuple_Pack(2, dst, src);
    PyObject *kwargs = (where != NULL)
            ? Py_BuildValue("{s:s,s:O}", "casting", "unsafe", "where", where)
            : Py_BuildValue("{s:s}", "casting", "unsafe");
    PyObject *res = (args && kwargs)
            ? PyObject_Call(np_copyto, args, kwargs) : NULL;
    Py_XDECREF(args); Py_XDECREF(kwargs);
    if (res == NULL) {
        return -1;
    }
    Py_DECREF(res);
    return 0;
}


static int
copy_values(PyObject *dst, PyObject *src)
{
    return copy_values_where(dst, src, NULL);
}


/*
 * Two descriptors that hold the same values.  `S3` has no singleton -- every
 * `np.dtype("S3")` is a new object -- so the identity checks that do for the
 * numeric types would send two equal string dtypes down the casting path.
 */
static int
same_value_dtype(PyArray_Descr *a, PyArray_Descr *b)
{
    return a == b || PyArray_EquivTypes(a, b);
}


/*
 * The wrapped dtype's `compare`, callable on an element where it lies: the
 * bytes may be misaligned, and there is no array of the wrapped dtype to pass.
 *
 * The numeric compares ignore their array argument, so NULL does for them.  The
 * flexible ones do not -- STRING_compare and UNICODE_compare read the item size
 * off it, VOID_compare the fields -- and a NULL there segfaulted `np.sort`.
 * Strings and raw bytes are compared here directly, with the same order numpy
 * uses.  A structured value gets a stack array standing in for the real one,
 * numpy's own trick inside VOID_compare, which also takes care of alignment.
 */
static int
wrapped_compare(PyArray_Descr *wrapped, const char *a, const char *b)
{
    npy_intp n = wrapped->elsize;

    if (wrapped->type_num == NPY_UNICODE) {
        /* always native byte order: see `stored_dtype` */
        for (npy_intp i = 0; i + 4 <= n; i += 4) {
            npy_ucs4 x, y;
            memcpy(&x, a + i, 4);
            memcpy(&y, b + i, 4);
            if (x != y) {
                return (x < y) ? -1 : 1;
            }
        }
        return 0;
    }
    if (wrapped->type_num == NPY_STRING
            || (wrapped->type_num == NPY_VOID && !PyDataType_HASFIELDS(wrapped))) {
        int c = memcmp(a, b, (size_t)n);
        return (c > 0) - (c < 0);
    }
    PyArray_ArrFuncs *funcs = PyDataType_GetArrFuncs(wrapped);
    if (funcs->compare == NULL) {
        return 0;
    }
    if (wrapped->type_num == NPY_VOID) {
        /*
         * VOID_compare asks the memory handler for scratch buffers, which needs
         * the GIL.  It is held: both Nullable descriptors carry
         * NPY_NEEDS_PYAPI, so `sort` and `argmax` do not release it.
         */
        PyArrayObject_fields dummy;
        memset(&dummy, 0, sizeof(dummy));
        dummy.descr = wrapped;
        return funcs->compare(a, b, (PyArrayObject *)&dummy);
    }
    npy_clongdouble x, y;
    if (n > (npy_intp)sizeof(x)) {
        return 0;
    }
    memcpy(&x, a, (size_t)n);
    memcpy(&y, b, (size_t)n);
    return funcs->compare(&x, &y, NULL);
}


/*
 * Truth of one wrapped value, under the same constraints as `wrapped_compare`.
 * Strings and raw bytes are true when any byte is set -- what numpy answers
 * for them too, whatever the alignment.
 */
static npy_bool
wrapped_nonzero(PyArray_Descr *wrapped, const char *data)
{
    npy_intp n = wrapped->elsize;
    PyArray_ArrFuncs *funcs = PyDataType_GetArrFuncs(wrapped);

    if (PyDataType_HASFIELDS(wrapped)) {
        if (funcs->nonzero == NULL) {
            return NPY_FALSE;
        }
        PyArrayObject_fields dummy;
        memset(&dummy, 0, sizeof(dummy));
        dummy.descr = wrapped;
        return funcs->nonzero((char *)data, (PyArrayObject *)&dummy);
    }
    if (wrapped->type_num == NPY_STRING || wrapped->type_num == NPY_UNICODE
            || wrapped->type_num == NPY_VOID) {
        for (npy_intp i = 0; i < n; i++) {
            if (data[i] != 0) {
                return NPY_TRUE;
            }
        }
        return NPY_FALSE;
    }
    if (funcs->nonzero == NULL) {
        return NPY_FALSE;
    }
    /*
     * The wrapped dtype's `nonzero` falls back to `copyswap` when the array is
     * not "behaved", and an element handed to us need not be aligned.  A
     * new-style DType has no `copyswap`, so that fallback would call a null
     * pointer.  Copy into an aligned buffer and pass NULL, which is the
     * documented "just read it" path.
     */
    npy_clongdouble aligned;
    if (n > (npy_intp)sizeof(aligned)) {
        return NPY_FALSE;
    }
    memcpy(&aligned, data, (size_t)n);
    return funcs->nonzero((char *)&aligned, NULL);
}


static PyArray_DTypeMeta *
dtypemeta_from_typenum(int typenum)
{
    PyArray_Descr *descr = PyArray_DescrFromType(typenum);
    if (descr == NULL) {
        return NULL;
    }
    PyArray_DTypeMeta *dt = (PyArray_DTypeMeta *)Py_TYPE(descr);
    Py_DECREF(descr);
    return dt;
}


/* -------------------------------------------------------------- ufuncs */
/*
 * No arithmetic is reimplemented.  The values of a Nullable[T] array are a
 * plain array of T, so every loop hands them to T's own ufunc as zero-copy
 * views and only deals with the gaps itself.
 */

#define N_BINOPS 29
static const char *binop_names[N_BINOPS] = {
    "add", "subtract", "multiply", "true_divide",
    "floor_divide", "power", "maximum", "minimum",
    /* comparisons come for free: `resolve_dtypes` reports a bool output, so
     * the result is Nullable[bool] without a line of special casing */
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    /* the rest are ordinary two-in/one-out loops and cost nothing but a name */
    "remainder", "fmod", "arctan2", "hypot",
    "logaddexp", "logaddexp2", "copysign", "nextafter",
    "fmax", "fmin", "heaviside", "float_power",
    /* two answers in one name each, see `binop_kleene_op` */
    "bitwise_and", "bitwise_or", "bitwise_xor",
};
static PyObject *binop_ufuncs[N_BINOPS];


/*
 * Py_EQ or Py_NE when `ufunc` is `equal` or `not_equal`, otherwise -1.
 *
 * Records need this.  NumPy has no ufunc loop comparing them -- a plain record
 * array's `==` takes a special path inside `array_richcompare` -- and when `==`
 * finds no loop it returns an all-False array instead of raising.  Nullable
 * records got `[False, False, False]` for `a == a`, with no warning at all.
 */
static int
binop_rich_op(PyObject *ufunc)
{
    for (int k = 0; k < N_BINOPS; k++) {
        if (binop_ufuncs[k] == ufunc) {
            if (strcmp(binop_names[k], "equal") == 0) {
                return Py_EQ;
            }
            if (strcmp(binop_names[k], "not_equal") == 0) {
                return Py_NE;
            }
            return -1;
        }
    }
    return -1;
}

/*
 * Each op needs its own C slot table, so the index list is written once here
 * and expanded wherever a table is built.  Adding an op
 * means adding a name above and one `X()` below; the assert catches a mismatch
 * at compile time instead of at import.
 */
#define BINOP_INDICES(X) \
    X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) \
    X(10) X(11) X(12) X(13) X(14) X(15) X(16) X(17) X(18) X(19) \
    X(20) X(21) X(22) X(23) X(24) X(25) X(26) X(27) X(28)

#define COUNT_ONE(I) + 1
/*
 * A compile-time check that also builds where `_Static_assert` does not
 * (MSVC in its default C mode): an array of size -1 is an error.
 */
#define NULLABLE_STATIC_ASSERT(COND, NAME) typedef char NAME[(COND) ? 1 : -1]
/* BINOP_INDICES is out of step with binop_names */
NULLABLE_STATIC_ASSERT(0 BINOP_INDICES(COUNT_ONE) == N_BINOPS, binop_indices_in_step);


/* zero-copy 1-D view over the value fields of one operand */
static PyObject *
value_view(PyArray_Descr *wrapped, char *data, npy_intp n, npy_intp stride)
{
    npy_intp dims[1] = {n};
    npy_intp strides[1] = {stride};
    Py_INCREF(wrapped);
    return PyArray_NewFromDescr(&PyArray_Type, wrapped, 1, dims, strides,
                                data, NPY_ARRAY_WRITEABLE, NULL);
}


/* ---------------------------------------------------------------- matmul */
/*
 * `matmul` is a gufunc, signature `(n?,k),(k,m?)->(n?,m?)`, so the loop gets
 * the core dimensions and their strides on top of the outer ones.  It is also
 * the only way in: `np.dot` goes through the legacy `dotfunc` slot, which is
 * handed a NULL array pointer (multiarraymodule.c, PyArray_MatrixProduct2), so
 * a *parametric* dtype has no way to learn its own element layout there.
 *
 * `C[i,j]` is a sum over `k`, so it is missing exactly when row `i` of A or
 * column `j` of B holds a gap anywhere.  That is decided per row and per
 * column, not per cell: two scans of size n and m instead of n*m.  The values
 * themselves come from numpy's own matmul on raw views -- whatever it computes
 * for a cell we are about to mark NA is discarded.
 */
static PyObject *np_matmul = NULL;

static PyObject *
value_view_2d(PyArray_Descr *w, char *data,
              npy_intp d0, npy_intp d1, npy_intp s0, npy_intp s1)
{
    npy_intp dims[2] = {d0, d1};
    npy_intp strides[2] = {s0, s1};
    Py_INCREF(w);
    return PyArray_NewFromDescr(&PyArray_Type, w, 2, dims, strides,
                                data, NPY_ARRAY_WRITEABLE, NULL);
}


/* run numpy's matmul over the value fields; the caller stamps the gaps */
static int
matmul_values(PyArray_Descr *wa, PyArray_Descr *wb, PyArray_Descr *wc,
        char *pa, char *pb, char *pc,
        npy_intp m, npy_intp n, npy_intp pdim,
        npy_intp a_m, npy_intp a_n, npy_intp b_n, npy_intp b_p,
        npy_intp c_m, npy_intp c_p)
{
    PyObject *A = value_view_2d(wa, pa, m, n, a_m, a_n);
    PyObject *B = value_view_2d(wb, pb, n, pdim, b_n, b_p);
    PyObject *C = value_view_2d(wc, pc, m, pdim, c_m, c_p);
    if (A == NULL || B == NULL || C == NULL) {
        Py_XDECREF(A); Py_XDECREF(B); Py_XDECREF(C);
        return -1;
    }
    PyObject *res = PyObject_CallFunctionObjArgs(np_matmul, A, B, C, NULL);
    Py_DECREF(A); Py_DECREF(B); Py_DECREF(C);
    if (res == NULL) {
        return -1;
    }
    Py_DECREF(res);
    return 0;
}


/* ------------------------------------------------------------------ clip */
/*
 * `clip` is the one three-input ufunc worth having.  It works exactly like the
 * binary ops -- hand the values to numpy's own `clip` and stamp the
 * gaps ourselves -- but with three operands instead of two, so the
 * two-in/one-out machinery above cannot be reused as is.
 *
 * A bound that is missing makes the answer missing: "keep this between 1 and
 * something I do not know" has no answer.
 */
static PyObject *clip_ufunc = NULL;

/*
 * Unary ufuncs: T's own ufunc runs on the values and the gaps are carried
 * over.  A unary op cannot create or remove missingness.
 */
#define N_UNOPS 41
static const char *unop_names[N_UNOPS] = {
    "negative", "absolute", "sqrt", "square", "reciprocal", "exp",
    "log", "sin", "cos", "floor", "ceil", "sign",
    /* asking whether a value we do not have is NaN can only answer NA */
    "isnan", "isinf", "isfinite",
    /* `conjugate` is what makes `std` and `var` work */
    "conjugate", "rint", "trunc",
    /* the remaining elementwise math, free for the same reason */
    "tan", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",
    "log2", "log10", "log1p", "expm1", "exp2", "cbrt",
    "degrees", "radians", "positive", "signbit", "fabs",
    /* `~` is bitwise NOT on ints and plain NOT on bool; both just carry NA */
    "invert", "logical_not",
};
static PyObject *unop_ufuncs[N_UNOPS];

#define UNOP_INDICES(X) \
    X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) \
    X(10) X(11) X(12) X(13) X(14) X(15) X(16) X(17) X(18) X(19) \
    X(20) X(21) X(22) X(23) X(24) X(25) X(26) X(27) X(28) X(29) \
    X(30) X(31) X(32) X(33) X(34) X(35) X(36) X(37) X(38) X(39) X(40)

/* UNOP_INDICES is out of step with unop_names */
NULLABLE_STATIC_ASSERT(0 UNOP_INDICES(COUNT_ONE) == N_UNOPS, unop_indices_in_step);


/* ------------------------------------------------------- Kleene logic */
/*
 * `logical_and` and `logical_or` cannot just propagate NA: sometimes the
 * answer is known even though one side is missing.
 *
 *      NA & False = False      NA | True  = True
 *      NA & True  = NA         NA | False = NA
 *      NA ^ x     = NA         (xor is never decided by one side)
 *
 * This is the three-valued logic of SQL, R and pandas.
 */

#define KLEENE_AND 0
#define KLEENE_OR  1
#define KLEENE_XOR 2

static const char *kleene_names[3] = {
    "logical_and", "logical_or", "logical_xor"};
static PyObject *kleene_ufuncs[3];


/* a contiguous bool copy of the values of one operand */
static PyArrayObject *
truthiness(PyArray_Descr *descr, char *data, npy_intp n, npy_intp stride)
{
    PyObject *view = value_view(descr, data, n, stride);
    if (view == NULL) {
        return NULL;
    }
    PyObject *as_bool = PyObject_CallMethod(view, "astype", "s", "bool");
    Py_DECREF(view);
    return (PyArrayObject *)as_bool;
}


/* first fields of NumPy's `ufunc_call_info`, see umath/ufunc_object.c */
typedef struct {
    PyArrayMethod_StridedLoop *strided_loop;
    PyArrayMethod_Context *context;
    NpyAuxData *auxdata;
    npy_bool requires_pyapi;
    npy_bool no_floatingpoint_errors;
} nullable_call_info;


/*
 * Ops whose scalar kernel compares its operands with `<` or `>`.  Those are
 * *signaling* predicates: IEEE 754 makes them raise FE_INVALID even on a quiet
 * NaN, where `==` and plain arithmetic stay silent.  A float gap is a NaN,
 * and the wrapped loop normally runs straight over it, so for these ops a gap would report "invalid value encountered" -- an arithmetic
 * error for data that simply is not there.  They take the masked path instead.
 * The list is empirical; `test_no_spurious_fp_warning_over_gaps` sweeps every
 * registered op, so an op that belongs here but is missing shows up as a test
 * failure rather than as a stray warning in someone's output.
 */
static int
binop_signals_on_gap(int idx)
{
    const char *n = binop_names[idx];
    return strcmp(n, "logaddexp") == 0 || strcmp(n, "logaddexp2") == 0;
}

/*
 * Ops that return the non-NaN operand instead of propagating.  A float gap is
 * a NaN, so for these a gap vanishes from the output and cannot be recovered
 * from the result alone.
 */
static int
binop_swallows_nan(int idx)
{
    const char *n = binop_names[idx];
    return strcmp(n, "fmax") == 0 || strcmp(n, "fmin") == 0;
}

/*
 * numpy's `maximum`/`minimum` loops take a SIMD path that turns the byte
 * stride into an element stride with `stride / sizeof(T)`.  The guard that is
 * supposed to reject a stride which is not a multiple of `sizeof(T)` only runs
 * when `alignof(T) != sizeof(T)`, which is false for every type wrapped on this
 * platform, so the division silently truncates (9 / 8 -> 1) and the loop walks
 * the array as if it were contiguous.  numpy never trips over this itself
 * because it buffers into contiguous memory first; the loops here are called
 * directly with the operands' own strides, so they must check.
 * See numpy/_core/src/common/simd/simd.h, NPYV_IMPL_MAXSTRIDE.
 */
static int
binop_needs_whole_element_strides(int idx)
{
    const char *n = binop_names[idx];
    return strcmp(n, "maximum") == 0 || strcmp(n, "minimum") == 0
            || strcmp(n, "fmax") == 0 || strcmp(n, "fmin") == 0;
}


/*
 * `Nullable[T] op T` has no loop of its own; the promoter tells NumPy to retry
 * with every operand seen as Nullable, and the cast `T -> Nullable[T]` then does
 * the rest.
 */
static int
nullable_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const NPY_UNUSED(op_dtypes[3]),
        PyArray_DTypeMeta *const signature[3],
        PyArray_DTypeMeta *new_op_dtypes[3])
{
    for (int i = 0; i < 3; i++) {
        PyArray_DTypeMeta *new = (signature[i] != NULL)
                ? signature[i] : &NullableDType;
        Py_INCREF(new);
        new_op_dtypes[i] = new;
    }
    return 0;
}


/*
 * `clip` has three inputs, so the promoter fills four slots.  Only the first
 * position is pinned and the rest are `None`, which numpy reads as "anything";
 * that covers `np.clip(a, lo, hi)` for every shape of bound.
 *
 * Exactly one wildcard, deliberately.  Pinning the second or third position as
 * well makes two promoters match a call like `clip(Nullable, Nullable, 5.0)`,
 * and numpy answers an ambiguous match with "could not find a loop" rather
 * than picking one.  The case left out is a *plain* array clipped by Nullable
 * bounds, which raises cleanly; `a.astype(nd.Nullable(...))` first.
 */
static int
nullable_promoter4(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const NPY_UNUSED(op_dtypes[4]),
        PyArray_DTypeMeta *const signature[4],
        PyArray_DTypeMeta *new_op_dtypes[4])
{
    for (int i = 0; i < 4; i++) {
        PyArray_DTypeMeta *new = (signature[i] != NULL)
                ? signature[i] : &NullableDType;
        Py_INCREF(new);
        new_op_dtypes[i] = new;
    }
    return 0;
}


static int
register_clip_promoter(PyArray_DTypeMeta *dtype)
{
    PyObject *capsule = PyCapsule_New(
            &nullable_promoter4, "numpy._ufunc_promoter", NULL);
    if (capsule == NULL) {
        return -1;
    }
    PyObject *dtypes = PyTuple_Pack(4, (PyObject *)dtype,
                                    Py_None, Py_None, Py_None);
    if (dtypes == NULL) {
        Py_DECREF(capsule);
        return -1;
    }
    int res = PyUFunc_AddPromoter(clip_ufunc, dtypes, capsule);
    Py_DECREF(dtypes);
    Py_DECREF(capsule);
    return res;
}
static int
register_promoter(PyObject *ufunc,
        PyArray_DTypeMeta *left, PyArray_DTypeMeta *right)
{
    PyObject *capsule = PyCapsule_New(
            &nullable_promoter, "numpy._ufunc_promoter", NULL);
    if (capsule == NULL) {
        return -1;
    }
    PyObject *dtypes = PyTuple_Pack(3, (PyObject *)left, (PyObject *)right,
                                    Py_None);
    if (dtypes == NULL) {
        Py_DECREF(capsule);
        return -1;
    }
    int res = PyUFunc_AddPromoter(ufunc, dtypes, capsule);
    Py_DECREF(dtypes);
    Py_DECREF(capsule);
    return res;
}


/* ------------------------------------------------------------ scalar NA */

typedef struct {
    PyObject_HEAD
} NAObject;

static PyObject *
na_repr(PyObject *NPY_UNUSED(self))
{
    return PyUnicode_FromString("NA");
}


/*
 * Arithmetic on the scalar.
 *
 * Indexing an array hands back a plain Python value for a present element and
 * this singleton for a missing one, so anything a caller can do with `a[0]`
 * they will sooner or later do with `a[1]`.  Without these the two branches
 * behave completely differently -- `a[0] + 1` is a number and `a[1] + 1` is a
 * TypeError -- which is not "missing propagates", it is a crash.
 *
 * Every operation involving NA is NA.  Same as R, same as `pd.NA`, and the
 * same answer the array loops already give elementwise.
 */
static PyObject *
na_return(PyObject *NPY_UNUSED(a), PyObject *NPY_UNUSED(b))
{
    Py_INCREF(NA_singleton);
    return NA_singleton;
}

static PyObject *
na_return_1(PyObject *NPY_UNUSED(a))
{
    Py_INCREF(NA_singleton);
    return NA_singleton;
}

static PyObject *
na_return_3(PyObject *NPY_UNUSED(a), PyObject *NPY_UNUSED(b),
            PyObject *NPY_UNUSED(c))
{
    Py_INCREF(NA_singleton);
    return NA_singleton;
}

/*
 * Truth value.  `bool(NA)` used to be True, inherited from Python's default
 * for an object with no `__bool__`, which quietly made `if a[i]:` treat a hole
 * in the data as a yes.  There is no right answer to "is the value you do not
 * have true?", so refuse instead of inventing one -- R raises "missing value
 * where TRUE/FALSE needed" and `bool(pd.NA)` raises too.
 */
static int
na_bool(PyObject *NPY_UNUSED(self))
{
    PyErr_SetString(PyExc_TypeError,
            "the truth value of NA is undefined; use nd.isna(), or "
            "nd.filled(x, ...) to say what a missing value should count as");
    return -1;
}

/*
 * Comparison.  `NA == NA` is NA, not True: two things you do not know are not
 * thereby known to be equal.  `is` still answers the identity question, which
 * is what `a[i] is NA` relies on.
 */
static PyObject *
na_richcompare(PyObject *NPY_UNUSED(a), PyObject *NPY_UNUSED(b),
               int NPY_UNUSED(op))
{
    Py_INCREF(NA_singleton);
    return NA_singleton;
}

static Py_hash_t
na_hash(PyObject *NPY_UNUSED(self))
{
    return 1954;          /* the payload, for no reason beyond consistency */
}

static PyNumberMethods na_as_number = {
    .nb_add = na_return,
    .nb_subtract = na_return,
    .nb_multiply = na_return,
    .nb_remainder = na_return,
    .nb_divmod = na_return,
    .nb_power = na_return_3,
    .nb_negative = na_return_1,
    .nb_positive = na_return_1,
    .nb_absolute = na_return_1,
    .nb_bool = na_bool,
    .nb_invert = na_return_1,
    .nb_lshift = na_return,
    .nb_rshift = na_return,
    .nb_and = na_return,
    .nb_xor = na_return,
    .nb_or = na_return,
    .nb_floor_divide = na_return,
    .nb_true_divide = na_return,
    .nb_matrix_multiply = na_return,
};

static PyTypeObject NAType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "nulldtype.NAType",
    .tp_basicsize = sizeof(NAObject),
    .tp_repr = na_repr,
    .tp_str = na_repr,
    .tp_as_number = &na_as_number,
    .tp_richcompare = na_richcompare,
    /*
     * Still hashable.  Python drops `tp_hash` when a type defines rich
     * comparison, but NA is a singleton: hashing it by identity stays
     * consistent, and a dict or set checks identity before `==`.  `pd.NA` is
     * hashable for the same reason.
     */
    .tp_hash = na_hash,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
};


/* ------------------------------------------------------------ the dtype */

/*
 * Room for the widest fixed NA pattern, complex128's.  This
 * used to be 8, and `nullable_put_na` copies `elsize` bytes out of it, so a
 * complex128 NA read 8 bytes past the end of the descriptor -- AddressSanitizer:
 * heap-buffer-overflow, READ of size 16, 0 bytes after the 112-byte object.
 * The imaginary half came out as zeros only because that memory happened to be.
 */
#define NA_PATTERN_MAX 16

#define NA_KIND_EXACT 0     /* ints, bool, datetime: a plain bit pattern */
#define NA_KIND_F4    1     /* float32: all bits but the sign */
#define NA_KIND_F8    2
#define NA_KIND_F2    3     /* float16 */
#define NA_KIND_C8    4     /* complex: the pattern lives in the real half */
#define NA_KIND_C16   5
#define NA_KIND_FILL  6     /* S, V: every byte 0xFF, `na_bytes` unused */
#define NA_KIND_UCS4  8     /* U: every code unit U+FFFF, `na_bytes` unused */
#define NA_KIND_RECORD 7    /* every field its own NA, see `nullable_leaf` */

/*
 * A record's NA is every field holding the NA of its own type -- INT_MIN in an
 * int32 field, 0x7FFF... in a float64 one, 0xFF bytes in an S field -- and the
 * record is missing only when all of them do.  Filling the record with 0xFF
 * instead would give up (-1, -1) for two int32 fields, and the int32, bool and
 * datetime64 fields of such a gap would read back as -1, True and 1969: values,
 * not gaps.  This way a field taken out of a gap is a gap of its own type, and
 * the only value given up is the record whose every field is already reserved.
 *
 * The fields are flattened once, when the descriptor is made, into leaves:
 * scalar fields with their offsets, nested records and subarrays unrolled.
 * Padding belongs to no leaf, so it never decides anything.
 */
typedef struct {
    npy_intp offset;
    npy_intp count;             /* > 1 for a subarray: that many in a row */
    int elsize;                 /* of one of them */
    int kind;                   /* never NA_KIND_RECORD */
    char na_bytes[NA_PATTERN_MAX];
} nullable_leaf;

typedef struct {
    PyArray_Descr base;
    PyArray_Descr *wrapped;
    int kind;
    char na_bytes[NA_PATTERN_MAX];
    nullable_leaf *leaves;      /* records only, owned */
    npy_intp n_leaves;
} NullableDescr;

#define NULLABLE_DESCR(d) ((NullableDescr *)(d))

/*
 * Float NA: every bit set except the sign -- 0x7FFF, 0x7FFFFFFF,
 * 0x7FFFFFFFFFFFFFFF.  One rule for every width, the same all-ones idea as
 * UINT_MAX.  It is a quiet NaN (exponent and top mantissa bit set), so an
 * operation that merely touches a gap raises no FE_INVALID, and ordinary
 * arithmetic never produces it: `np.nan` is 0x7FF8000000000000 and the NaN x86
 * generates for inf - inf is 0xFFF8000000000000.  `nullable_is_na` ignores the
 * sign, so a gap whose sign a borrowed loop flipped still reads as a gap.
 */
#define NA_F8_BITS 0x7FFFFFFFFFFFFFFFULL
#define NA_F4_BITS 0x7FFFFFFFU
#define NA_F2_BITS 0x7FFFU

/* U: the noncharacter U+FFFF, see `nullable_pattern` */
#define NA_UCS4 0x0000FFFFU


static int
nullable_pattern(int type_num, int *kind, char *na_bytes, int elsize)
{
    memset(na_bytes, 0, NA_PATTERN_MAX);
    switch (type_num) {
        case NPY_STRING: case NPY_VOID:
            /*
             * The whole cell filled with 0xFF, whatever its width, so there is
             * nothing to keep in `na_bytes`.  It is a legal byte string, given
             * up like UINT_MAX, but 0xFF never occurs in ASCII or UTF-8 text.
             * A V with fields never gets here: records go through
             * `collect_leaves`.
             */
            *kind = NA_KIND_FILL;
            return 0;
        case NPY_UNICODE:
            /*
             * Every code unit U+FFFF, a *noncharacter*: Unicode sets those
             * aside for a program's internal use and says they are never to be
             * interchanged, which is exactly this job.  The cell stays a valid
             * string, so `len`, `ord`, `encode` and `np.strings.*` all work on
             * it if someone views the values raw.  U+FFFFFFFF would give up
             * nothing at all -- `chr()` refuses it -- but numpy then hands back
             * a `str` whose maximum character is out of range, and even `len()`
             * on that raises SystemError.  A gap must not be a broken object.
             */
            *kind = NA_KIND_UCS4;
            return 0;
        default:
            break;
    }
    if (elsize > NA_PATTERN_MAX) {
        return -1;
    }
    switch (type_num) {
        case NPY_FLOAT64: {
            npy_uint64 bits = NA_F8_BITS;
            memcpy(na_bytes, &bits, 8);
            *kind = NA_KIND_F8;
            return 0;
        }
        case NPY_FLOAT32: {
            npy_uint32 bits = NA_F4_BITS;
            memcpy(na_bytes, &bits, 4);
            *kind = NA_KIND_F4;
            return 0;
        }
        case NPY_BOOL:
            na_bytes[0] = 2;
            *kind = NA_KIND_EXACT;
            return 0;
        case NPY_FLOAT16: {
            npy_uint16 bits = NA_F2_BITS;
            memcpy(na_bytes, &bits, 2);
            *kind = NA_KIND_F2;
            return 0;
        }
        case NPY_COMPLEX128: {
            /*
             * Both halves get the pattern, so whatever the arithmetic does
             * with them the result still carries a NaN; `nullable_is_na` only
             * reads the real half, which is where a payload survives best.
             */
            npy_uint64 bits = NA_F8_BITS;
            memcpy(na_bytes, &bits, 8);
            memcpy(na_bytes + 8, &bits, 8);
            *kind = NA_KIND_C16;
            return 0;
        }
        case NPY_COMPLEX64: {
            npy_uint32 bits = NA_F4_BITS;
            memcpy(na_bytes, &bits, 4);
            memcpy(na_bytes + 4, &bits, 4);
            *kind = NA_KIND_C8;
            return 0;
        }
        case NPY_INT8: case NPY_INT16: case NPY_INT32: case NPY_INT64: {
            /* INT_MIN is 0x80 in the top byte, zeros below */
            na_bytes[elsize - 1] = (char)0x80;
            *kind = NA_KIND_EXACT;
            return 0;
        }
        case NPY_UINT8: case NPY_UINT16: case NPY_UINT32: case NPY_UINT64:
            /*
             * UINT_MAX, the mirror of INT_MIN: every pattern of an unsigned
             * int is a real number, so one has to be given up.  A bias shift
             * (store x - 2**(n-1), reserve INT_MIN) was the alternative and is
             * worse twice over: it gives up 0 instead of UINT_MAX, and the
             * cells then live in a shifted frame, so a borrowed loop adds up
             * the right number in the wrong frame -- 200 + 100 lands on the
             * cell that means 172.  Reserving the top value keeps each cell
             * equal to its value, so numpy's own uint loops run unchanged.
             */
            memset(na_bytes, 0xFF, (size_t)elsize);
            *kind = NA_KIND_EXACT;
            return 0;
        case NPY_DATETIME: case NPY_TIMEDELTA: {
            /*
             * Free: numpy already spells NaT as INT64_MIN in these, so the
             * reserved pattern is the one the type itself already reserves.
             * Nothing legal is taken away, and `filled()` hands back a real
             * NaT rather than an invented date.
             */
            na_bytes[elsize - 1] = (char)0x80;
            *kind = NA_KIND_EXACT;
            return 0;
        }
        default:
            return -1;
    }
}


/*
 * Flatten `descr`, placed at `offset` and repeated `count` times, into leaves.
 * Returns 0, 1 when some field has no pattern (an object, `S0`), or -1 with an
 * error set.
 */
static int
collect_leaves(PyArray_Descr *descr, npy_intp offset, npy_intp count,
               nullable_leaf **leaves, npy_intp *n, npy_intp *cap)
{
    if (PyDataType_HASFIELDS(descr)) {
        PyObject *names = PyDataType_NAMES(descr);
        PyObject *fields = PyDataType_FIELDS(descr);
        for (npy_intp r = 0; r < count; r++) {
            for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(names); i++) {
                PyObject *info = PyDict_GetItemWithError(
                        fields, PyTuple_GET_ITEM(names, i));
                if (info == NULL) {
                    if (!PyErr_Occurred()) {
                        PyErr_SetString(PyExc_RuntimeError,
                                "a record field is missing from its fields");
                    }
                    return -1;
                }
                npy_intp field_offset = PyLong_AsSsize_t(PyTuple_GET_ITEM(info, 1));
                if (field_offset == -1 && PyErr_Occurred()) {
                    return -1;
                }
                int rc = collect_leaves(
                        (PyArray_Descr *)PyTuple_GET_ITEM(info, 0),
                        offset + r * descr->elsize + field_offset, 1,
                        leaves, n, cap);
                if (rc != 0) {
                    return rc;
                }
            }
        }
        return 0;
    }
    if (PyDataType_HASSUBARRAY(descr)) {
        PyArray_Descr *base = PyDataType_SUBARRAY(descr)->base;
        if (base->elsize <= 0) {
            return 1;
        }
        return collect_leaves(base, offset, count * (descr->elsize / base->elsize),
                              leaves, n, cap);
    }
    int kind;
    char na_bytes[NA_PATTERN_MAX];
    if (descr->elsize <= 0
            || nullable_pattern(descr->type_num, &kind, na_bytes, descr->elsize) < 0) {
        return 1;
    }
    if (*n == *cap) {
        npy_intp new_cap = (*cap == 0) ? 8 : 2 * *cap;
        nullable_leaf *grown = PyMem_Realloc(
                *leaves, (size_t)new_cap * sizeof(nullable_leaf));
        if (grown == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        *leaves = grown;
        *cap = new_cap;
    }
    nullable_leaf *leaf = &(*leaves)[(*n)++];
    leaf->offset = offset;
    leaf->count = count;
    leaf->elsize = descr->elsize;
    leaf->kind = kind;
    memcpy(leaf->na_bytes, na_bytes, NA_PATTERN_MAX);
    return 0;
}


static NPY_INLINE int
pattern_is_na(int kind, npy_intp elsize, const char *na_bytes, const char *data)
{
    if (kind == NA_KIND_F8) {
        npy_uint64 u;
        memcpy(&u, data, 8);
        return (u & 0x7FFFFFFFFFFFFFFFULL) == 0x7FFFFFFFFFFFFFFFULL;
    }
    if (kind == NA_KIND_F4) {
        npy_uint32 u;
        memcpy(&u, data, 4);
        return (u & 0x7FFFFFFFU) == 0x7FFFFFFFU;
    }
    if (kind == NA_KIND_F2) {
        npy_uint16 u;
        memcpy(&u, data, 2);
        return (u & 0x7FFFU) == 0x7FFFU;
    }
    if (kind == NA_KIND_C16) {
        npy_uint64 u;
        memcpy(&u, data, 8);          /* real half only */
        return (u & 0x7FFFFFFFFFFFFFFFULL) == 0x7FFFFFFFFFFFFFFFULL;
    }
    if (kind == NA_KIND_C8) {
        npy_uint32 u;
        memcpy(&u, data, 4);
        return (u & 0x7FFFFFFFU) == 0x7FFFFFFFU;
    }
    if (kind == NA_KIND_FILL) {
        /* the whole cell; a real value almost always stops at the first byte */
        const unsigned char *p = (const unsigned char *)data;
        for (npy_intp i = 0; i < elsize; i++) {
            if (p[i] != 0xFF) {
                return 0;
            }
        }
        return 1;
    }
    if (kind == NA_KIND_UCS4) {
        /* every character, so one real character settles it */
        for (npy_intp i = 0; i + 4 <= elsize; i += 4) {
            npy_uint32 u;
            memcpy(&u, data + i, 4);
            if (u != NA_UCS4) {
                return 0;
            }
        }
        return 1;
    }
    return memcmp(data, na_bytes, (size_t)elsize) == 0;
}


static NPY_INLINE void
pattern_put_na(int kind, npy_intp elsize, const char *na_bytes, char *data)
{
    if (kind == NA_KIND_FILL) {
        memset(data, 0xFF, (size_t)elsize);
        return;
    }
    if (kind == NA_KIND_UCS4) {
        for (npy_intp i = 0; i + 4 <= elsize; i += 4) {
            npy_uint32 u = NA_UCS4;
            memcpy(data + i, &u, 4);
        }
        return;
    }
    memcpy(data, na_bytes, (size_t)elsize);
}


static NPY_INLINE int
nullable_is_na(const NullableDescr *descr, const char *data)
{
    if (descr->kind != NA_KIND_RECORD) {
        return pattern_is_na(descr->kind, descr->base.elsize, descr->na_bytes,
                             data);
    }
    /* missing only if every field is; the first field with a value settles it */
    for (npy_intp j = 0; j < descr->n_leaves; j++) {
        const nullable_leaf *leaf = &descr->leaves[j];
        for (npy_intp k = 0; k < leaf->count; k++) {
            if (!pattern_is_na(leaf->kind, leaf->elsize, leaf->na_bytes,
                               data + leaf->offset + k * leaf->elsize)) {
                return 0;
            }
        }
    }
    return 1;
}


static NPY_INLINE void
nullable_put_na(const NullableDescr *descr, char *data)
{
    if (descr->kind != NA_KIND_RECORD) {
        pattern_put_na(descr->kind, descr->base.elsize, descr->na_bytes, data);
        return;
    }
    /* padding zeroed too, so every gap has the same `tobytes()` */
    memset(data, 0, (size_t)descr->base.elsize);
    for (npy_intp j = 0; j < descr->n_leaves; j++) {
        const nullable_leaf *leaf = &descr->leaves[j];
        for (npy_intp k = 0; k < leaf->count; k++) {
            pattern_put_na(leaf->kind, leaf->elsize, leaf->na_bytes,
                           data + leaf->offset + k * leaf->elsize);
        }
    }
}


static PyArray_Descr *new_nullable_descr_impl(PyArray_Descr *wrapped);

/* long double and byte order are settled first, see `stored_dtype` */
static PyArray_Descr *
new_nullable_descr(PyArray_Descr *wrapped)
{
    PyArray_Descr *stored = stored_dtype(wrapped);
    if (stored == NULL) {
        return NULL;
    }
    PyArray_Descr *res = new_nullable_descr_impl(stored);
    Py_DECREF(stored);
    return res;
}


static PyArray_Descr *
new_nullable_descr_impl(PyArray_Descr *wrapped)
{
    if (wrapped->type_num == NPY_VSTRING) {
        /*
         * StringDType already has a missing value of its own (NEP 55), and its
         * elements own heap allocations that only it knows how to free, which
         * a wrapper cannot do through the public API.
         */
        PyErr_SetString(PyExc_TypeError,
                "Nullable[T] does not wrap StringDType, which brings its own "
                "missing value: np.dtypes.StringDType(na_object=np.nan)");
        return NULL;
    }
    if (PyDataType_REFCHK(wrapped)) {
        PyErr_SetString(PyExc_TypeError,
                "Nullable[T] does not support dtypes holding references yet");
        return NULL;
    }
    PyArray_Descr *canonical = NULL;
    if (wrapped->type_num >= 0) {
        canonical = PyArray_DescrFromType(wrapped->type_num);
        if (canonical == NULL) {
            return NULL;
        }
        /*
         * `EquivTypes` and not an elsize check: `M8` and `M8[D]` are both 8
         * bytes, so a size comparison would swap a datetime's unit away and
         * leave a descriptor nothing can cast to.
         */
        if (PyArray_EquivTypes(canonical, wrapped)) {
            wrapped = canonical;
        }
    }
    int kind = NA_KIND_EXACT;
    char na_bytes[NA_PATTERN_MAX];
    memset(na_bytes, 0, NA_PATTERN_MAX);
    nullable_leaf *leaves = NULL;
    npy_intp n_leaves = 0;
    int usable;
    if (wrapped->elsize <= 0 || PyDataType_HASSUBARRAY(wrapped)) {
        /* `S0` has no cell to fill; a bare subarray dtype is not one element */
        usable = 0;
    }
    else if (PyDataType_HASFIELDS(wrapped)) {
        npy_intp cap = 0;
        int rc = collect_leaves(wrapped, 0, 1, &leaves, &n_leaves, &cap);
        if (rc < 0) {
            PyMem_Free(leaves);
            Py_XDECREF(canonical);
            return NULL;
        }
        /* a record of nothing but padding has nowhere to put NA */
        usable = (rc == 0 && n_leaves > 0);
        kind = NA_KIND_RECORD;
    }
    else {
        usable = nullable_pattern(wrapped->type_num, &kind, na_bytes,
                                 wrapped->elsize) == 0;
    }
    if (!usable) {
        PyMem_Free(leaves);
        PyErr_Format(PyExc_TypeError,
                "Nullable[T] has no spare value to reserve in %R", wrapped);
        Py_XDECREF(canonical);
        return NULL;
    }
    NullableDescr *new = (NullableDescr *)PyArrayDescr_Type.tp_new(
            (PyTypeObject *)&NullableDType, NULL, NULL);
    if (new == NULL) {
        PyMem_Free(leaves);
        Py_XDECREF(canonical);
        return NULL;
    }
    Py_INCREF(wrapped);
    Py_XDECREF(canonical);
    new->wrapped = wrapped;
    new->kind = kind;
    new->leaves = leaves;
    new->n_leaves = n_leaves;
    memcpy(new->na_bytes, na_bytes, NA_PATTERN_MAX);
    new->base.elsize = wrapped->elsize;          /* no extra byte at all */
    new->base.alignment = wrapped->alignment;
    /* NPY_NEEDS_PYAPI: see `new_nullable_descr_impl` */
    new->base.flags = NPY_USE_GETITEM | NPY_USE_SETITEM | NPY_NEEDS_INIT
                      | NPY_NEEDS_PYAPI;
    new->base.byteorder = wrapped->byteorder;
    return (PyArray_Descr *)new;
}


static void
nullable_dealloc(NullableDescr *self)
{
    Py_CLEAR(self->wrapped);
    PyMem_Free(self->leaves);
    self->leaves = NULL;
    PyArrayDescr_Type.tp_dealloc((PyObject *)self);
}


static PyObject *
nullable_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"dtype", NULL};
    PyObject *dtype_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:Nullable", kwlist,
                                     &dtype_obj)) {
        return NULL;
    }
    PyArray_Descr *wrapped = NULL;
    if (!PyArray_DescrConverter(dtype_obj, &wrapped)) {
        return NULL;
    }
    PyObject *res = (PyObject *)new_nullable_descr(wrapped);
    Py_DECREF(wrapped);
    return res;
}


static PyObject *
nullable_repr(NullableDescr *self)
{
    return PyUnicode_FromFormat("Nullable(%R)", (PyObject *)self->wrapped);
}


static PyObject *
nullable_reduce(NullableDescr *self, PyObject *NPY_UNUSED(args))
{
    return Py_BuildValue("O(O)", Py_TYPE(self), (PyObject *)self->wrapped);
}


static PyObject *
nullable_get_wrapped(NullableDescr *self, void *NPY_UNUSED(closure))
{
    Py_INCREF(self->wrapped);
    return (PyObject *)self->wrapped;
}


static PyMethodDef nullable_methods[] = {
    {"__reduce__", (PyCFunction)nullable_reduce, METH_NOARGS, "pickle support"},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef nullable_getset[] = {
    {"wrapped", (getter)nullable_get_wrapped, NULL,
     "the dtype whose values this one stores", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};


/* ------------------------------------------------------------ dtype slots */

static PyArray_Descr *
nullable_default_descr(PyArray_DTypeMeta *NPY_UNUSED(cls))
{
    PyArray_Descr *f8 = PyArray_DescrFromType(NPY_DOUBLE);
    PyArray_Descr *res = new_nullable_descr(f8);
    Py_DECREF(f8);
    return res;
}


static PyArray_Descr *
nullable_discover(PyArray_DTypeMeta *cls, PyObject *NPY_UNUSED(obj))
{
    return nullable_default_descr(cls);
}


static PyArray_DTypeMeta *
nullable_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    /*
     * A Python `int`, `float` or `complex` is deliberately *not* handled here.
     * Saying yes makes `np.result_type(dtype, 1.0)` work -- and with it
     * `np.select` -- but it also takes the weak-scalar rule of NEP 50 away from
     * every ufunc: numpy stops adopting the array's dtype for the scalar and
     * reaches for this DType's default instead, so `Nullable[f4] + 1.0` came
     * out float64 and `Nullable[i8] // 2` came out float64 too.  Measured, not
     * guessed.  The promoters keep the scalar path value-correct; the price is
     * that `result_type` with a Python scalar refuses.
     */
    if (other == &NullableDType || is_wrappable_typenum(other->type_num)) {
        Py_INCREF(cls);
        return cls;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


static PyArray_Descr *
nullable_common_instance(PyArray_Descr *descr1, PyArray_Descr *descr2)
{
    if (NULLABLE_DESCR(descr1)->wrapped == NULLABLE_DESCR(descr2)->wrapped) {
        Py_INCREF(descr1);
        return descr1;
    }
    PyArray_Descr *value = PyArray_PromoteTypes(
            NULLABLE_DESCR(descr1)->wrapped, NULLABLE_DESCR(descr2)->wrapped);
    if (value == NULL) {
        return NULL;
    }
    PyArray_Descr *res = new_nullable_descr(value);
    Py_DECREF(value);
    return res;
}


static PyArray_Descr *
nullable_ensure_canonical(PyArray_Descr *self)
{
    Py_INCREF(self);
    return self;
}


static int
nullable_setitem(PyArray_Descr *descr, PyObject *obj, char *dataptr)
{
    NullableDescr *self = NULLABLE_DESCR(descr);
    if (obj == NA_singleton || obj == Py_None) {
        nullable_put_na(self, dataptr);
        return 0;
    }
    PyArray_Descr *wrapped = self->wrapped;
    Py_INCREF(wrapped);
    PyArrayObject *tmp = (PyArrayObject *)PyArray_FromAny(
            obj, wrapped, 0, 0, NPY_ARRAY_FORCECAST, NULL);
    if (tmp == NULL) {
        return -1;
    }
    if (PyArray_NDIM(tmp) != 0) {
        Py_DECREF(tmp);
        PyErr_SetString(PyExc_ValueError,
                "setting an array element with a sequence");
        return -1;
    }
    memcpy(dataptr, PyArray_DATA(tmp), (size_t)wrapped->elsize);
    Py_DECREF(tmp);
    if (nullable_is_na(self, dataptr)) {
        PyErr_Format(PyExc_ValueError,
                "%R is the value reserved to mean NA in %R", obj, descr);
        return -1;
    }
    return 0;
}


static PyObject *
nullable_getitem(PyArray_Descr *descr, char *dataptr)
{
    NullableDescr *self = NULLABLE_DESCR(descr);
    if (nullable_is_na(self, dataptr)) {
        Py_INCREF(NA_singleton);
        return NA_singleton;
    }
    return PyArray_Scalar(dataptr, self->wrapped, NULL);
}


/* ----------------------------------------------------------------- casts */

static NPY_CASTING
nullable_to_nullable_resolve(
        struct PyArrayMethodObject_tag *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[2]),
        PyArray_Descr *const given_descrs[2], PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    if (given_descrs[1] == NULL) {
        Py_INCREF(given_descrs[0]);
        loop_descrs[1] = given_descrs[0];
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }
    if (!same_value_dtype(NULLABLE_DESCR(loop_descrs[0])->wrapped,
                          NULLABLE_DESCR(loop_descrs[1])->wrapped)) {
        return NPY_UNSAFE_CASTING;      /* handled by the loop below */
    }
    *view_offset = 0;
    return NPY_NO_CASTING;
}


/*
 * After the values of a cast into a Nullable array have been written: a cell
 * whose source had a value but which now reads as NA was produced by the cast
 * itself -- an int64 wrapping onto INT32_MIN, an `S5` truncated to three 0xFF
 * bytes.  Left alone it would silently become a gap, so refuse instead.
 * `src_na` says whether source cell `i` was already missing; NULL means none
 * were.  Missing source cells get the destination's own NA stamped over them.
 */
static int
nullable_stamp_and_check(NullableDescr *to, char *out, npy_intp out_stride,
        npy_intp N, int (*src_na)(void *, npy_intp), void *src)
{
    for (npy_intp i = 0; i < N; i++) {
        char *cell = out + i * out_stride;
        if (src_na != NULL && src_na(src, i)) {
            nullable_put_na(to, cell);
        }
        else if (nullable_is_na(to, cell)) {
            PyErr_Format(PyExc_ValueError,
                    "the cast produced the value reserved to mean NA in %R",
                    (PyObject *)to);
            return -1;
        }
    }
    return 0;
}


typedef struct {
    NullableDescr *descr;
    const char *data;
    npy_intp stride;
} nullable_source;


static int
nullable_source_na(void *src, npy_intp i)
{
    nullable_source *s = (nullable_source *)src;
    return nullable_is_na(s->descr, s->data + i * s->stride);
}


static int
nullable_to_nullable_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    NullableDescr *from = NULLABLE_DESCR(context->descriptors[0]);
    NullableDescr *to = NULLABLE_DESCR(context->descriptors[1]);
    char *in = data[0], *out = data[1];
    size_t itemsize = (size_t)from->base.elsize;

    if (same_value_dtype(from->wrapped, to->wrapped)) {
        if (strides[0] == (npy_intp)itemsize && strides[1] == (npy_intp)itemsize) {
            memcpy(out, in, itemsize * (size_t)N);
            return 0;
        }
        while (N--) {
            memcpy(out, in, itemsize);
            in += strides[0];
            out += strides[1];
        }
        return 0;
    }

    /*
     * Different value dtypes: NumPy casts the cells that hold a value, and only
     * those.  A gap's bits are no value of anything; casting a float gap to an
     * int would warn about an invalid value that is not there.
     */
    npy_intp dims[1] = {N};
    PyObject *valid = PyArray_SimpleNew(1, dims, NPY_BOOL);
    if (valid == NULL) {
        return -1;
    }
    char *mask = PyArray_DATA((PyArrayObject *)valid);
    int any = 0;
    for (npy_intp i = 0; i < N; i++) {
        mask[i] = !nullable_is_na(from, data[0] + i * strides[0]);
        any |= mask[i];
    }
    int rc = 0;
    if (any) {
        PyObject *a = value_view(from->wrapped, in, N, strides[0]);
        PyObject *b = value_view(to->wrapped, out, N, strides[1]);
        rc = (a && b) ? copy_values_where(b, a, valid) : -1;
        Py_XDECREF(a); Py_XDECREF(b);
    }
    Py_DECREF(valid);
    if (rc < 0) {
        return -1;
    }
    nullable_source src = {from, data[0], strides[0]};
    return nullable_stamp_and_check(to, data[1], strides[1], N,
                                    &nullable_source_na, &src);
}


static NPY_CASTING
to_nullable_resolve(
        struct PyArrayMethodObject_tag *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[2]),
        PyArray_Descr *const given_descrs[2], PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = new_nullable_descr(given_descrs[0]);
        if (loop_descrs[1] == NULL) {
            Py_CLEAR(loop_descrs[0]);
            return (NPY_CASTING)-1;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }
    /* one value of the range disappears, so this is never a safe cast */
    return NPY_SAME_KIND_CASTING;
}


static int
to_nullable_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    PyArray_Descr *src = context->descriptors[0];
    NullableDescr *dst = NULLABLE_DESCR(context->descriptors[1]);

    if (same_value_dtype(src, dst->wrapped)) {
        for (npy_intp i = 0; i < N; i++) {
            char *in = data[0] + i * strides[0];
            if (nullable_is_na(dst, in)) {
                PyErr_Format(PyExc_ValueError,
                        "the array holds the value reserved to mean NA in %R",
                        (PyObject *)dst);
                return -1;
            }
            memcpy(data[1] + i * strides[1], in, (size_t)dst->base.elsize);
        }
        return 0;
    }
    PyObject *a = value_view(src, data[0], N, strides[0]);
    PyObject *b = value_view(dst->wrapped, data[1], N, strides[1]);
    if (a == NULL || b == NULL) {
        Py_XDECREF(a); Py_XDECREF(b);
        return -1;
    }
    int rc = (a && b) ? copy_values(b, a) : -1;
    Py_XDECREF(a); Py_XDECREF(b);
    if (rc < 0) {
        return -1;
    }
    return nullable_stamp_and_check(dst, data[1], strides[1], N, NULL, NULL);
}


static NPY_CASTING
from_nullable_resolve(
        struct PyArrayMethodObject_tag *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[2]),
        PyArray_Descr *const given_descrs[2], PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    PyArray_Descr *wrapped = NULLABLE_DESCR(given_descrs[0])->wrapped;
    if (given_descrs[1] == NULL) {
        Py_INCREF(wrapped);
        loop_descrs[1] = wrapped;
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }
    /*
     * `same_kind`, not `unsafe`: dropping NA is still refused, but by the loop,
     * which raises on the first missing element.  Keeping it at `unsafe` only
     * meant numpy would not even try -- `np.isin` writes `|=` into a plain bool
     * array and gave up there, on data that had nothing missing in it.
     */
    return NPY_SAME_KIND_CASTING;
}


static int
from_nullable_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    NullableDescr *src = NULLABLE_DESCR(context->descriptors[0]);
    PyArray_Descr *dst = context->descriptors[1];
    size_t itemsize = (size_t)src->base.elsize;

    for (npy_intp i = 0; i < N; i++) {
        if (nullable_is_na(src, data[0] + i * strides[0])) {
            PyErr_Format(PyExc_ValueError,
                    "cannot convert a missing value to %R; "
                    "fill it first, for example with `filled()`",
                    (PyObject *)src->wrapped);
            return -1;
        }
    }
    if (same_value_dtype(src->wrapped, dst)) {
        for (npy_intp i = 0; i < N; i++) {
            memcpy(data[1] + i * strides[1], data[0] + i * strides[0], itemsize);
        }
        return 0;
    }
    /*
     * `astype(np.float64)` on Nullable(int32) lands here with an 8-byte
     * destination.  Copying the 4 source bytes as they were left half of every
     * double as garbage, and `astype(np.int16)` wrote past the end of the
     * output buffer.  Let NumPy cast the values instead.
     */
    PyObject *a = value_view(src->wrapped, data[0], N, strides[0]);
    PyObject *b = value_view(dst, data[1], N, strides[1]);
    int rc = (a && b) ? copy_values(b, a) : -1;
    Py_XDECREF(a); Py_XDECREF(b);
    return rc;
}


/*
 * Stamping NA over the result is the hot loop, so it must vectorise.  The
 * dispatch on the dtype happens once, outside; inside there is nothing but a
 * comparison the compiler can turn into SIMD.
 */
/*
 * Can an operation *compute* the reserved value out of operands that both had
 * one?  For ints it can: `2**31 - 1 + 1` wraps exactly onto INT_MIN, `0u - 1`
 * onto UINT_MAX, and `np.strings.add` can fill a cell with 0xFF.  Such a cell
 * would read back as a gap nobody wrote -- the one silent path left -- so the
 * loops refuse it instead.
 *
 * Floats and complex are exempt: their pattern is a quiet NaN with every bit
 * set, and hardware only ever propagates a NaN it was given, it does not
 * synthesise that one (`np.nan` is 0x7FF8..., `inf - inf` is 0xFFF8...).
 */
static int
na_can_be_computed(int kind)
{
    return kind != NA_KIND_F8 && kind != NA_KIND_F4 && kind != NA_KIND_F2
        && kind != NA_KIND_C8 && kind != NA_KIND_C16;
}


static int
nullable_computed_na(const NullableDescr *out)
{
    PyErr_Format(PyExc_ValueError,
            "the operation produced %R's reserved value, which would read "
            "back as a missing value; compute in a wider dtype",
            (PyObject *)out);
    return -1;
}


#define NA_FIXUP(CTYPE, MASKEXPR, CHECK)                                      \
    do {                                                                      \
        const CTYPE *p0 = (const CTYPE *)data[0];                             \
        const CTYPE *p1 = (const CTYPE *)data[1];                             \
        CTYPE *po = (CTYPE *)data[2];                                         \
        CTYPE na;                                                             \
        memcpy(&na, out->na_bytes, sizeof(CTYPE));                            \
        for (npy_intp i = 0; i < N; i++) {                                    \
            CTYPE x = p0[i], y = p1[i];                                       \
            if (MASKEXPR) {                                                   \
                po[i] = na;                                                   \
            }                                                                 \
            else if (CHECK) {                                                 \
                made |= (po[i] == na);                                        \
            }                                                                 \
        }                                                                     \
    } while (0)

static int
nullable_fixup(NullableDescr *in0, NullableDescr *in1, NullableDescr *out,
        char *const data[], npy_intp const strides[], npy_intp N,
        int propagates_nan)
{
    npy_intp item = out->base.elsize;
    int check = na_can_be_computed(out->kind);
    int made = 0;
    int packed = (strides[0] == in0->base.elsize)
              && (strides[1] == in1->base.elsize)
              && (strides[2] == item)
              && in0->kind == out->kind && in1->kind == out->kind;

    if (packed) {
        /*
         * For floats the hardware has already put a NaN everywhere an input
         * was NA, so a single pass over the freshly written output — still hot
         * in cache — tells us whether there is anything to fix at all.  Most
         * arrays have no gaps and stop here.
         *
         * This rests on the op propagating NaN.  `fmax`/`fmin` do not: they
         * return the non-NaN side on purpose, so a gap leaves no trace in the
         * output and this scan would skip a fixup that is needed.  They are
         * flagged by `binop_swallows_nan`, which turns this scan off and
         * forces the full pass; anything else added to `binop_names` that
         * swallows NaN must be listed there too.
         */
        if (propagates_nan && out->kind == NA_KIND_F8) {
            const double *po = (const double *)data[2];
            int any = 0;
            for (npy_intp i = 0; i < N; i++) {
                any |= (po[i] != po[i]);
            }
            if (!any) {
                return 0;
            }
        }
        else if (propagates_nan && out->kind == NA_KIND_F4) {
            const float *po = (const float *)data[2];
            int any = 0;
            for (npy_intp i = 0; i < N; i++) {
                any |= (po[i] != po[i]);
            }
            if (!any) {
                return 0;
            }
        }
        switch (out->kind) {
            case NA_KIND_F8:
                NA_FIXUP(npy_uint64,
                    ((x & 0x7FFFFFFFFFFFFFFFULL) == 0x7FFFFFFFFFFFFFFFULL) ||
                    ((y & 0x7FFFFFFFFFFFFFFFULL) == 0x7FFFFFFFFFFFFFFFULL), 0);
                return 0;
            case NA_KIND_F4:
                NA_FIXUP(npy_uint32,
                    ((x & 0x7FFFFFFFU) == 0x7FFFFFFFU) ||
                    ((y & 0x7FFFFFFFU) == 0x7FFFFFFFU), 0);
                return 0;
            case NA_KIND_EXACT:
                if (item == 8) { NA_FIXUP(npy_uint64, x == na || y == na, 1); goto done; }
                if (item == 4) { NA_FIXUP(npy_uint32, x == na || y == na, 1); goto done; }
                if (item == 2) { NA_FIXUP(npy_uint16, x == na || y == na, 1); goto done; }
                if (item == 1) { NA_FIXUP(npy_uint8,  x == na || y == na, 1); goto done; }
                break;
            default:
                /* f16 and complex: correctness first, via the loop below */
                break;
        }
    }
    for (npy_intp i = 0; i < N; i++) {
        if (nullable_is_na(in0, data[0] + i * strides[0])
                || nullable_is_na(in1, data[1] + i * strides[1])) {
            nullable_put_na(out, data[2] + i * strides[2]);
        }
        else if (check && nullable_is_na(out, data[2] + i * strides[2])) {
            made = 1;
        }
    }
  done:
    return made ? nullable_computed_na(out) : 0;
}


/* ------------------------------------------------------- ufunc loops */
/*
 * The values are contiguous and aligned, so the wrapped loop runs at full
 * speed on the whole chunk.  Afterwards we stamp NA over the positions where
 * an input was missing.  That second pass is what makes `nan + NA` come out
 * as NA regardless of operand order, and it means we never rely on the CPU
 * carrying a NaN payload through the arithmetic.
 */

/*
 * A reduction hands us the accumulator as both first input and output at
 * stride 0.  Element-wise code would keep only the last write, so the whole
 * chunk goes to `ufunc.reduce`.  NA propagates: one gap makes the result a gap.
 */
static int
nullable_reduce_chunk(int idx, PyArrayMethod_Context *context,
        char *const data[], npy_intp N, npy_intp const strides[])
{
    NullableDescr *acc = NULLABLE_DESCR(context->descriptors[0]);
    NullableDescr *in = NULLABLE_DESCR(context->descriptors[1]);

    int missing = nullable_is_na(acc, data[0]);
    for (npy_intp i = 0; i < N && !missing; i++) {
        missing = nullable_is_na(in, data[1] + i * strides[1]);
    }
    if (missing) {
        nullable_put_na(acc, data[2]);
        return 0;
    }
    PyObject *values = value_view(in->wrapped, data[1], N, strides[1]);
    PyObject *initial = PyArray_Scalar(data[0], acc->wrapped, NULL);
    PyObject *reduce = PyObject_GetAttrString(binop_ufuncs[idx], "reduce");
    PyObject *args = (values != NULL) ? PyTuple_Pack(1, values) : NULL;
    PyObject *kwargs = (initial != NULL)
            ? Py_BuildValue("{s:O}", "initial", initial) : NULL;
    PyObject *res = (reduce && args && kwargs)
            ? PyObject_Call(reduce, args, kwargs) : NULL;
    Py_XDECREF(values); Py_XDECREF(initial);
    Py_XDECREF(reduce); Py_XDECREF(args); Py_XDECREF(kwargs);
    if (res == NULL) {
        return -1;
    }
    PyObject *slot = value_view(acc->wrapped, data[2], 1, 0);
    int rc = (slot != NULL) ? copy_values(slot, res) : -1;
    Py_XDECREF(slot);
    Py_DECREF(res);
    if (rc == 0 && na_can_be_computed(acc->kind)
            && nullable_is_na(acc, data[2])) {
        return nullable_computed_na(acc);       /* the sum wrapped onto it */
    }
    return rc;
}


/* `accumulate`: everything after the first gap is a gap too */
static int
nullable_accumulate_chunk(int idx, PyArrayMethod_Context *context,
        char *const data[], npy_intp N, npy_intp const strides[])
{
    NullableDescr *acc = NULLABLE_DESCR(context->descriptors[0]);
    NullableDescr *in = NULLABLE_DESCR(context->descriptors[1]);
    NullableDescr *out = NULLABLE_DESCR(context->descriptors[2]);

    npy_intp known = 0;
    if (!nullable_is_na(acc, data[0])) {
        while (known < N && !nullable_is_na(in, data[1] + known * strides[1])) {
            known++;
        }
    }
    for (npy_intp i = known; i < N; i++) {
        nullable_put_na(out, data[2] + i * strides[2]);
    }
    if (known == 0) {
        return 0;
    }
    PyObject *first = PyArray_Scalar(data[0], acc->wrapped, NULL);
    PyObject *head = (first != NULL)
            ? PyObject_CallFunction(np_asarray, "([O])", first) : NULL;
    PyObject *tail = value_view(in->wrapped, data[1], known, strides[1]);
    Py_XDECREF(first);
    PyObject *joined = (head && tail)
            ? PyObject_CallFunction(np_concatenate, "((OO))", head, tail) : NULL;
    Py_XDECREF(head); Py_XDECREF(tail);
    if (joined == NULL) {
        return -1;
    }
    PyObject *running = PyObject_CallMethod(
            binop_ufuncs[idx], "accumulate", "O", joined);
    Py_DECREF(joined);
    if (running == NULL) {
        return -1;
    }
    PyObject *without_head = PySequence_GetSlice(running, 1, known + 1);
    Py_DECREF(running);
    if (without_head == NULL) {
        return -1;
    }
    PyObject *slot = value_view(out->wrapped, data[2], known, strides[2]);
    int rc = (slot != NULL) ? copy_values(slot, without_head) : -1;
    Py_XDECREF(slot);
    Py_DECREF(without_head);
    return rc;
}


/* `==` / `!=` on records: a Nullable(bool) out, see `binop_rich_op` */
static NPY_CASTING
nullable_record_compare_resolve(PyArray_Descr *const given_descrs[3],
        PyArray_Descr *loop_descrs[3])
{
    PyArray_Descr *w0 = NULLABLE_DESCR(given_descrs[0])->wrapped;
    PyArray_Descr *w1 = NULLABLE_DESCR(given_descrs[1])->wrapped;
    if (!same_value_dtype(w0, w1)) {
        PyErr_Format(PyExc_TypeError,
                "cannot compare records of different dtypes: %R and %R",
                (PyObject *)w0, (PyObject *)w1);
        return (NPY_CASTING)-1;
    }
    PyArray_Descr *bool_descr = PyArray_DescrFromType(NPY_BOOL);
    if (bool_descr == NULL) {
        return (NPY_CASTING)-1;
    }
    loop_descrs[2] = new_nullable_descr(bool_descr);
    Py_DECREF(bool_descr);
    if (loop_descrs[2] == NULL) {
        return (NPY_CASTING)-1;
    }
    for (int i = 0; i < 2; i++) {
        Py_INCREF(given_descrs[i]);
        loop_descrs[i] = given_descrs[i];
    }
    return NPY_NO_CASTING;
}


/*
 * Compare the value views with NumPy's own record `==`, which is the special
 * path `array_richcompare` takes for plain records, then stamp the gaps.
 */
static int
nullable_record_compare(int op, PyArrayMethod_Context *context,
        char *const data[], npy_intp N, npy_intp const strides[])
{
    NullableDescr *in0 = NULLABLE_DESCR(context->descriptors[0]);
    NullableDescr *in1 = NULLABLE_DESCR(context->descriptors[1]);
    NullableDescr *out = NULLABLE_DESCR(context->descriptors[2]);

    PyObject *a = value_view(in0->wrapped, data[0], N, strides[0]);
    PyObject *b = value_view(in1->wrapped, data[1], N, strides[1]);
    PyObject *res = (a && b) ? PyObject_RichCompare(a, b, op) : NULL;
    Py_XDECREF(a); Py_XDECREF(b);
    if (res == NULL) {
        return -1;
    }
    PyObject *o = value_view(out->wrapped, data[2], N, strides[2]);
    int rc = (o != NULL) ? copy_values(o, res) : -1;
    Py_XDECREF(o);
    Py_DECREF(res);
    if (rc < 0) {
        return -1;
    }
    for (npy_intp i = 0; i < N; i++) {
        if (nullable_is_na(in0, data[0] + i * strides[0])
                || nullable_is_na(in1, data[1] + i * strides[1])) {
            nullable_put_na(out, data[2] + i * strides[2]);
        }
    }
    return 0;
}


static NPY_CASTING
nullable_binary_resolve_impl(PyObject *ufunc,
        PyArray_Descr *const given_descrs[3], PyArray_Descr *loop_descrs[3])
{
    if (NULLABLE_DESCR(given_descrs[0])->kind == NA_KIND_RECORD
            || NULLABLE_DESCR(given_descrs[1])->kind == NA_KIND_RECORD) {
        if (binop_rich_op(ufunc) >= 0) {
            return nullable_record_compare_resolve(given_descrs, loop_descrs);
        }
        /* anything else falls through; `resolve_dtypes` refuses records */
    }
    PyObject *query = Py_BuildValue("(OOO)",
            (PyObject *)NULLABLE_DESCR(given_descrs[0])->wrapped,
            (PyObject *)NULLABLE_DESCR(given_descrs[1])->wrapped,
            Py_None);
    if (query == NULL) {
        return (NPY_CASTING)-1;
    }
    PyObject *resolved = PyObject_CallMethod(
            ufunc, "resolve_dtypes", "(O)", query);
    Py_DECREF(query);
    if (resolved == NULL) {
        return (NPY_CASTING)-1;
    }
    NPY_CASTING casting = NPY_NO_CASTING;
    for (int i = 0; i < 3; i++) {
        PyArray_Descr *value = (PyArray_Descr *)PyTuple_GET_ITEM(resolved, i);
        /*
         * Reuse the descriptor we were handed when it already matches.  A new
         * object here would look like a different dtype to NumPy, which would
         * then buffer and cast every operand for nothing.
         */
        if (i < 3 && given_descrs[i] != NULL
                && same_value_dtype(NULLABLE_DESCR(given_descrs[i])->wrapped,
                                    value)) {
            Py_INCREF(given_descrs[i]);
            loop_descrs[i] = (PyArray_Descr *)given_descrs[i];
            continue;
        }
        loop_descrs[i] = new_nullable_descr(value);
        if (loop_descrs[i] == NULL) {
            for (int j = 0; j < i; j++) {
                Py_CLEAR(loop_descrs[j]);
            }
            Py_DECREF(resolved);
            return (NPY_CASTING)-1;
        }
        if (i < 2
                && !same_value_dtype(NULLABLE_DESCR(given_descrs[i])->wrapped,
                                     value)) {
            casting = NPY_SAFE_CASTING;
        }
    }
    Py_DECREF(resolved);
    return casting;
}


static int
nullable_binary_loop_impl(int idx, PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[])
{
    npy_intp N = dimensions[0];
    NullableDescr *in0 = NULLABLE_DESCR(context->descriptors[0]);
    NullableDescr *in1 = NULLABLE_DESCR(context->descriptors[1]);
    NullableDescr *out = NULLABLE_DESCR(context->descriptors[2]);

    if (in0->kind == NA_KIND_RECORD) {
        /* only `==` and `!=` resolve for records */
        return nullable_record_compare(binop_rich_op(binop_ufuncs[idx]),
                                       context, data, N, strides);
    }
    if (strides[0] == 0 && data[0] == data[2]) {
        return nullable_reduce_chunk(idx, context, data, N, strides);
    }
    if (strides[0] == strides[2] && data[0] == data[2] - strides[2]) {
        return nullable_accumulate_chunk(idx, context, data, N, strides);
    }

    PyObject *a = value_view(in0->wrapped, data[0], N, strides[0]);
    PyObject *b = value_view(in1->wrapped, data[1], N, strides[1]);
    PyObject *o = value_view(out->wrapped, data[2], N, strides[2]);
    if (a == NULL || b == NULL || o == NULL) {
        Py_XDECREF(a); Py_XDECREF(b); Py_XDECREF(o);
        return -1;
    }
    /*
     * Unlike the fast path this masks the gaps off.  It costs a bool array per
     * chunk, but it is the only way to guarantee that a value which does not
     * exist never raises a floating point warning of its own -- and it is what
     * makes this the correct fallback for `binop_signals_on_gap`.
     */
    npy_intp dims[1] = {N};
    PyObject *both = PyArray_SimpleNew(1, dims, NPY_BOOL);
    if (both == NULL) {
        Py_DECREF(a); Py_DECREF(b); Py_DECREF(o);
        return -1;
    }
    char *mask = PyArray_DATA((PyArrayObject *)both);
    for (npy_intp i = 0; i < N; i++) {
        mask[i] = !nullable_is_na(in0, data[0] + i * strides[0])
               && !nullable_is_na(in1, data[1] + i * strides[1]);
    }

    PyObject *args = PyTuple_Pack(2, a, b);
    PyObject *kwargs = Py_BuildValue("{s:O,s:O}", "out", o, "where", both);
    PyObject *res = (args && kwargs)
            ? PyObject_Call(binop_ufuncs[idx], args, kwargs) : NULL;
    Py_XDECREF(args); Py_XDECREF(kwargs);
    Py_DECREF(a); Py_DECREF(b); Py_DECREF(o);
    Py_DECREF(both);
    if (res == NULL) {
        return -1;
    }
    Py_DECREF(res);

    /*
     * `where=` left the gap lanes untouched, so the output holds whatever the
     * buffer held before -- never a reliable NaN.  The pre-scan must not run.
     */
    return nullable_fixup(in0, in1, out, data, strides, N, 0);
}


/*
 * The fast path: values are contiguous and aligned, so the wrapped loop runs
 * on the whole chunk at native speed, gaps included -- a gap holds a NaN or
 * INT_MIN, and computing with those is harmless and quiet.  One pass afterwards
 * stamps NA wherever an input had one.
 */
typedef struct {
    NpyAuxData base;
    int idx;
    PyObject *capsule;
    PyArrayMethod_StridedLoop *inner;
    PyArrayMethod_Context *inner_context;
    NpyAuxData *inner_auxdata;
} nullable_auxdata;


static void
nullable_auxdata_free(NpyAuxData *data)
{
    nullable_auxdata *aux = (nullable_auxdata *)data;
    Py_XDECREF(aux->capsule);
    PyMem_Free(aux);
}


static int
nullable_fast_loop(PyArrayMethod_Context *context, char *const data[],
        npy_intp const dimensions[], npy_intp const strides[],
        NpyAuxData *auxdata)
{
    nullable_auxdata *aux = (nullable_auxdata *)auxdata;
    npy_intp N = dimensions[0];

    if (strides[0] == 0 && data[0] == data[2]) {
        return nullable_reduce_chunk(aux->idx, context, data, N, strides);
    }
    if (strides[0] == strides[2] && data[0] == data[2] - strides[2]) {
        return nullable_accumulate_chunk(aux->idx, context, data, N, strides);
    }

    npy_intp count = N;
    if (aux->inner(aux->inner_context, data, &count, strides,
                   aux->inner_auxdata) < 0) {
        return -1;
    }
    return nullable_fixup(NULLABLE_DESCR(context->descriptors[0]),
                          NULLABLE_DESCR(context->descriptors[1]),
                          NULLABLE_DESCR(context->descriptors[2]),
                          data, strides, N, !binop_swallows_nan(aux->idx));
}


static int
nullable_get_loop_impl(int idx, PyArrayMethod_Context *context,
        int NPY_UNUSED(aligned), int NPY_UNUSED(move_references),
        const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_auxdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    *flags = NPY_METH_REQUIRES_PYAPI;

    /* records have no loop to borrow; `nullable_record_compare` does it */
    if (NULLABLE_DESCR(context->descriptors[0])->kind == NA_KIND_RECORD) {
        *out_auxdata = NULL;
        return 1;
    }
    /* these must not run over a gap at all; the masked path handles them */
    if (binop_signals_on_gap(idx)) {
        *out_auxdata = NULL;
        return 1;
    }

    /* see `binop_needs_whole_element_strides` */
    if (binop_needs_whole_element_strides(idx)) {
        npy_intp elsize = NULLABLE_DESCR(context->descriptors[0])->wrapped->elsize;
        if (strides[0] % elsize != 0 || strides[1] % elsize != 0
                || strides[2] % elsize != 0) {
            *out_auxdata = NULL;
            return 1;
        }
    }
    PyObject *ufunc = binop_ufuncs[idx];
    PyObject *query = Py_BuildValue("(OOO)",
            (PyObject *)NULLABLE_DESCR(context->descriptors[0])->wrapped,
            (PyObject *)NULLABLE_DESCR(context->descriptors[1])->wrapped,
            (PyObject *)NULLABLE_DESCR(context->descriptors[2])->wrapped);
    if (query == NULL) {
        return -1;
    }
    PyObject *resolved = PyObject_CallMethod(
            ufunc, "_resolve_dtypes_and_context", "(O)", query);
    Py_DECREF(query);
    if (resolved == NULL || !PyTuple_Check(resolved)
            || PyTuple_GET_SIZE(resolved) != 2) {
        Py_XDECREF(resolved);
        PyErr_Clear();
        *out_auxdata = NULL;
        return 1;
    }
    PyObject *capsule = PyTuple_GET_ITEM(resolved, 1);
    Py_INCREF(capsule);
    Py_DECREF(resolved);

    PyObject *fixed = Py_BuildValue("(nnn)", strides[0], strides[1], strides[2]);
    PyObject *kwargs = (fixed != NULL)
            ? Py_BuildValue("{s:O}", "fixed_strides", fixed) : NULL;
    PyObject *args = (kwargs != NULL) ? PyTuple_Pack(1, capsule) : NULL;
    PyObject *meth = (args != NULL)
            ? PyObject_GetAttrString(ufunc, "_get_strided_loop") : NULL;
    PyObject *res = (meth != NULL) ? PyObject_Call(meth, args, kwargs) : NULL;
    Py_XDECREF(fixed); Py_XDECREF(kwargs); Py_XDECREF(args); Py_XDECREF(meth);
    if (res == NULL) {
        Py_DECREF(capsule);
        PyErr_Clear();
        *out_auxdata = NULL;
        return 1;
    }
    Py_DECREF(res);

    nullable_call_info *info = PyCapsule_GetPointer(
            capsule, "numpy_1.24_ufunc_call_info");
    if (info == NULL || info->strided_loop == NULL) {
        Py_DECREF(capsule);
        PyErr_Clear();
        *out_auxdata = NULL;
        return 1;
    }
    nullable_auxdata *aux = PyMem_Calloc(1, sizeof(nullable_auxdata));
    if (aux == NULL) {
        Py_DECREF(capsule);
        PyErr_NoMemory();
        return -1;
    }
    aux->base.free = &nullable_auxdata_free;
    aux->idx = idx;
    aux->capsule = capsule;
    aux->inner = info->strided_loop;
    aux->inner_context = info->context;
    aux->inner_auxdata = info->auxdata;
    *out_loop = &nullable_fast_loop;
    *out_auxdata = (NpyAuxData *)aux;
    return 0;
}


static int
nullable_reduction_initial_impl(int idx, PyArrayMethod_Context *context,
        npy_bool NPY_UNUSED(reduction_is_empty), void *initial)
{
    PyObject *identity = PyObject_GetAttrString(binop_ufuncs[idx], "identity");
    if (identity == NULL) {
        return -1;
    }
    if (identity == Py_None) {
        Py_DECREF(identity);
        return 0;
    }
    NullableDescr *out = NULLABLE_DESCR(context->descriptors[0]);
    PyObject *slot = value_view(out->wrapped, (char *)initial, 1, 0);
    int rc = (slot == NULL) ? -1 : copy_values(slot, identity);
    Py_XDECREF(slot);
    Py_DECREF(identity);
    /*
     * Two ways the identity does not fit the wrapped dtype, both of them
     * `bitwise_and`'s -1: in a `uint8` cell NumPy refuses to put it at all, and
     * where it does fit it is UINT_MAX, the reserved value -- folding from a
     * gap would make every such reduction missing.  Either way the honest
     * answer is "no identity"; numpy then starts from the first element.
     */
    if (rc < 0) {
        PyErr_Clear();
        return 0;
    }
    return nullable_is_na(out, (char *)initial) ? 0 : 1;
}


/* the Kleene loops live further down, past the ordinary binary ones */
static NPY_CASTING nullable_kleene_resolve_impl(
        PyArray_Descr *const given_descrs[3], PyArray_Descr *loop_descrs[3]);
static int nullable_kleene_loop_impl(int op, PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[]);
static int nullable_kleene_initial_impl(int op, PyArrayMethod_Context *context,
        npy_bool reduction_is_empty, void *initial);


/*
 * `&`, `|` and `^` need two different answers.  On Nullable[bool] they follow
 * Kleene -- `NA & False` is False, because whatever the missing value turns out
 * to be the answer is already settled -- and on Nullable[int] they propagate
 * like any other arithmetic.  One loop cannot do both, but by the time a loop
 * is chosen the wrapped dtype is known, so bool operands go to the Kleene
 * loops and everything else to the ordinary binary one.
 */
static int
binop_kleene_op(int idx)
{
    const char *n = binop_names[idx];
    if (strcmp(n, "bitwise_and") == 0) {
        return KLEENE_AND;
    }
    if (strcmp(n, "bitwise_or") == 0) {
        return KLEENE_OR;
    }
    if (strcmp(n, "bitwise_xor") == 0) {
        return KLEENE_XOR;
    }
    return -1;
}


/* both sides have to be bool; `bool & int64` is integer arithmetic */
static int
binop_wants_kleene(int idx, PyArray_Descr *const descrs[2])
{
    return binop_kleene_op(idx) >= 0
        && NULLABLE_DESCR(descrs[0])->wrapped->type_num == NPY_BOOL
        && NULLABLE_DESCR(descrs[1])->wrapped->type_num == NPY_BOOL;
}


#define NULLABLE_BINOP(IDX)                                                   \
static NPY_CASTING                                                            \
nullable_resolve_##IDX(struct PyArrayMethodObject_tag *NPY_UNUSED(self),      \
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),                       \
        PyArray_Descr *const given_descrs[3], PyArray_Descr *loop_descrs[3],  \
        npy_intp *NPY_UNUSED(view_offset))                                    \
{                                                                             \
    if (binop_wants_kleene(IDX, given_descrs)) {                              \
        return nullable_kleene_resolve_impl(given_descrs, loop_descrs);       \
    }                                                                         \
    return nullable_binary_resolve_impl(binop_ufuncs[IDX], given_descrs,      \
                                        loop_descrs);                         \
}                                                                             \
static int                                                                    \
nullable_loop_##IDX(PyArrayMethod_Context *context, char *const data[],       \
        npy_intp const dimensions[], npy_intp const strides[],                \
        NpyAuxData *NPY_UNUSED(auxdata))                                      \
{                                                                             \
    if (binop_wants_kleene(IDX, context->descriptors)) {                      \
        return nullable_kleene_loop_impl(binop_kleene_op(IDX), context, data, \
                                         dimensions, strides);                \
    }                                                                         \
    return nullable_binary_loop_impl(IDX, context, data, dimensions, strides);\
}                                                                             \
static int                                                                    \
nullable_get_loop_##IDX(PyArrayMethod_Context *context, int aligned,          \
        int move_references, const npy_intp *strides,                         \
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_auxdata,       \
        NPY_ARRAYMETHOD_FLAGS *flags)                                         \
{                                                                             \
    if (binop_wants_kleene(IDX, context->descriptors)) {                      \
        *flags = NPY_METH_REQUIRES_PYAPI;                                     \
        *out_loop = &nullable_loop_##IDX;                                     \
        *out_auxdata = NULL;                                                  \
        return 0;                                                             \
    }                                                                         \
    int res = nullable_get_loop_impl(IDX, context, aligned, move_references,  \
                                     strides, out_loop, out_auxdata, flags);  \
    if (res == 1) {                                                           \
        *out_loop = &nullable_loop_##IDX;                                     \
        return 0;                                                             \
    }                                                                         \
    return res;                                                               \
}                                                                             \
static int                                                                    \
nullable_initial_##IDX(PyArrayMethod_Context *context,                        \
        npy_bool reduction_is_empty, void *initial)                           \
{                                                                             \
    if (binop_wants_kleene(IDX, context->descriptors)) {                      \
        return nullable_kleene_initial_impl(binop_kleene_op(IDX), context,    \
                                            reduction_is_empty, initial);     \
    }                                                                         \
    return nullable_reduction_initial_impl(IDX, context,                      \
                                           reduction_is_empty, initial);      \
}                                                                             \
static PyType_Slot nullable_slots_##IDX[] = {                                 \
    {NPY_METH_resolve_descriptors, &nullable_resolve_##IDX},                  \
    {NPY_METH_get_loop, &nullable_get_loop_##IDX},                            \
    {NPY_METH_get_reduction_initial, &nullable_initial_##IDX},                \
    {0, NULL}                                                                 \
};

BINOP_INDICES(NULLABLE_BINOP)

#define NULLABLE_SLOTS_PTR(I) nullable_slots_##I,
static PyType_Slot *nullable_binop_slots[N_BINOPS] = { BINOP_INDICES(NULLABLE_SLOTS_PTR) };
#undef NULLABLE_SLOTS_PTR


/* ------------------------------------------------------------ unary loops */

static NPY_CASTING
nullable_unary_resolve_impl(int idx,
        PyArray_Descr *const given_descrs[2], PyArray_Descr *loop_descrs[2])
{
    PyObject *query = Py_BuildValue("(OO)",
            (PyObject *)NULLABLE_DESCR(given_descrs[0])->wrapped, Py_None);
    if (query == NULL) {
        return (NPY_CASTING)-1;
    }
    PyObject *resolved = PyObject_CallMethod(
            unop_ufuncs[idx], "resolve_dtypes", "(O)", query);
    Py_DECREF(query);
    if (resolved == NULL) {
        return (NPY_CASTING)-1;
    }
    NPY_CASTING casting = NPY_NO_CASTING;
    for (int i = 0; i < 2; i++) {
        PyArray_Descr *value = (PyArray_Descr *)PyTuple_GET_ITEM(resolved, i);
        if (given_descrs[i] != NULL
                && same_value_dtype(NULLABLE_DESCR(given_descrs[i])->wrapped,
                                    value)) {
            Py_INCREF(given_descrs[i]);
            loop_descrs[i] = (PyArray_Descr *)given_descrs[i];
            continue;
        }
        loop_descrs[i] = new_nullable_descr(value);
        if (loop_descrs[i] == NULL) {
            for (int j = 0; j < i; j++) {
                Py_CLEAR(loop_descrs[j]);
            }
            Py_DECREF(resolved);
            return (NPY_CASTING)-1;
        }
        if (i == 0) {
            casting = NPY_SAFE_CASTING;
        }
    }
    Py_DECREF(resolved);
    return casting;
}


static int
nullable_unary_loop_impl(int idx, PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[])
{
    npy_intp N = dimensions[0];
    NullableDescr *in = NULLABLE_DESCR(context->descriptors[0]);
    NullableDescr *out = NULLABLE_DESCR(context->descriptors[1]);

    PyObject *a = value_view(in->wrapped, data[0], N, strides[0]);
    PyObject *o = value_view(out->wrapped, data[1], N, strides[1]);
    if (a == NULL || o == NULL) {
        Py_XDECREF(a); Py_XDECREF(o);
        return -1;
    }
    PyObject *args = PyTuple_Pack(1, a);
    PyObject *kwargs = Py_BuildValue("{s:O}", "out", o);
    PyObject *res = (args && kwargs)
            ? PyObject_Call(unop_ufuncs[idx], args, kwargs) : NULL;
    Py_XDECREF(args); Py_XDECREF(kwargs);
    Py_DECREF(a); Py_DECREF(o);
    if (res == NULL) {
        return -1;
    }
    Py_DECREF(res);

    /* a unary op cannot create or remove a gap, it just carries it over */
    int check = na_can_be_computed(out->kind);
    for (npy_intp i = 0; i < N; i++) {
        if (nullable_is_na(in, data[0] + i * strides[0])) {
            nullable_put_na(out, data[1] + i * strides[1]);
        }
        else if (check && nullable_is_na(out, data[1] + i * strides[1])) {
            return nullable_computed_na(out);
        }
    }
    return 0;
}


#define NULLABLE_UNOP(IDX)                                                    \
static NPY_CASTING                                                            \
nullable_ures_##IDX(struct PyArrayMethodObject_tag *NPY_UNUSED(self),         \
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[2]),                       \
        PyArray_Descr *const given_descrs[2], PyArray_Descr *loop_descrs[2],  \
        npy_intp *NPY_UNUSED(view_offset))                                    \
{                                                                             \
    return nullable_unary_resolve_impl(IDX, given_descrs, loop_descrs);       \
}                                                                             \
static int                                                                    \
nullable_uloop_##IDX(PyArrayMethod_Context *context, char *const data[],      \
        npy_intp const dimensions[], npy_intp const strides[],                \
        NpyAuxData *NPY_UNUSED(auxdata))                                      \
{                                                                             \
    return nullable_unary_loop_impl(IDX, context, data, dimensions, strides); \
}                                                                             \
static PyType_Slot nullable_uslots_##IDX[] = {                                \
    {NPY_METH_resolve_descriptors, &nullable_ures_##IDX},                     \
    {NPY_METH_strided_loop, &nullable_uloop_##IDX},                           \
    {NPY_METH_unaligned_strided_loop, &nullable_uloop_##IDX},                 \
    {0, NULL}                                                                 \
};

UNOP_INDICES(NULLABLE_UNOP)

#define NULLABLE_USLOTS_PTR(I) nullable_uslots_##I,
static PyType_Slot *nullable_unop_slots[N_UNOPS] = { UNOP_INDICES(NULLABLE_USLOTS_PTR) };
#undef NULLABLE_USLOTS_PTR

static PyArray_DTypeMeta *unop_dtypes[2] = {NULL, NULL};
static PyArrayMethod_Spec unop_specs[N_UNOPS];


/* ---------------------------------------------------------------- matmul */

static NPY_CASTING
nullable_matmul_resolve(struct PyArrayMethodObject_tag *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_Descr *const given_descrs[3], PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    return nullable_binary_resolve_impl(np_matmul, given_descrs, loop_descrs);
}


static int
nullable_matmul_loop(PyArrayMethod_Context *context, char *const data[],
        npy_intp const dimensions[], npy_intp const strides[],
        NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp outer = dimensions[0];
    npy_intp m = dimensions[1], n = dimensions[2], pdim = dimensions[3];
    npy_intp o0 = strides[0], o1 = strides[1], o2 = strides[2];
    npy_intp a_m = strides[3], a_n = strides[4];
    npy_intp b_n = strides[5], b_p = strides[6];
    npy_intp c_m = strides[7], c_p = strides[8];

    NullableDescr *da = NULLABLE_DESCR(context->descriptors[0]);
    NullableDescr *db = NULLABLE_DESCR(context->descriptors[1]);
    NullableDescr *dc = NULLABLE_DESCR(context->descriptors[2]);

    char *rowna = PyMem_Malloc((size_t)(m > 0 ? m : 1));
    char *colna = PyMem_Malloc((size_t)(pdim > 0 ? pdim : 1));
    if (rowna == NULL || colna == NULL) {
        PyMem_Free(rowna); PyMem_Free(colna);
        PyErr_NoMemory();
        return -1;
    }

    for (npy_intp it = 0; it < outer; it++) {
        char *pa = data[0] + it * o0;
        char *pb = data[1] + it * o1;
        char *pc = data[2] + it * o2;

        if (matmul_values(da->wrapped, db->wrapped, dc->wrapped,
                          pa, pb, pc, m, n, pdim,
                          a_m, a_n, b_n, b_p, c_m, c_p) < 0) {
            PyMem_Free(rowna); PyMem_Free(colna);
            return -1;
        }

        for (npy_intp i = 0; i < m; i++) {
            char bad = 0;
            for (npy_intp k = 0; k < n; k++) {
                bad |= nullable_is_na(da, pa + i * a_m + k * a_n);
            }
            rowna[i] = bad;
        }
        for (npy_intp j = 0; j < pdim; j++) {
            char bad = 0;
            for (npy_intp k = 0; k < n; k++) {
                bad |= nullable_is_na(db, pb + k * b_n + j * b_p);
            }
            colna[j] = bad;
        }
        int check = na_can_be_computed(dc->kind);
        for (npy_intp i = 0; i < m; i++) {
            for (npy_intp j = 0; j < pdim; j++) {
                char *cell = pc + i * c_m + j * c_p;
                if (rowna[i] || colna[j]) {
                    nullable_put_na(dc, cell);
                }
                else if (check && nullable_is_na(dc, cell)) {
                    PyMem_Free(rowna); PyMem_Free(colna);
                    return nullable_computed_na(dc);
                }
            }
        }
    }
    PyMem_Free(rowna); PyMem_Free(colna);
    return 0;
}


static PyType_Slot nullable_matmul_slots[] = {
    {NPY_METH_resolve_descriptors, &nullable_matmul_resolve},
    {NPY_METH_strided_loop, &nullable_matmul_loop},
    {NPY_METH_unaligned_strided_loop, &nullable_matmul_loop},
    {0, NULL},
};


/* ------------------------------------------------------------------ clip */

static NPY_CASTING
nullable_clip_resolve(struct PyArrayMethodObject_tag *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[4]),
        PyArray_Descr *const given_descrs[4], PyArray_Descr *loop_descrs[4],
        npy_intp *NPY_UNUSED(view_offset))
{
    PyObject *query = Py_BuildValue("(OOOO)",
            (PyObject *)NULLABLE_DESCR(given_descrs[0])->wrapped,
            (PyObject *)NULLABLE_DESCR(given_descrs[1])->wrapped,
            (PyObject *)NULLABLE_DESCR(given_descrs[2])->wrapped,
            Py_None);
    if (query == NULL) {
        return (NPY_CASTING)-1;
    }
    PyObject *resolved = PyObject_CallMethod(
            clip_ufunc, "resolve_dtypes", "(O)", query);
    Py_DECREF(query);
    if (resolved == NULL) {
        return (NPY_CASTING)-1;
    }
    NPY_CASTING casting = NPY_NO_CASTING;
    for (int i = 0; i < 4; i++) {
        PyArray_Descr *value_descr =
                (PyArray_Descr *)PyTuple_GET_ITEM(resolved, i);
        loop_descrs[i] = new_nullable_descr(value_descr);
        if (loop_descrs[i] == NULL) {
            for (int j = 0; j < i; j++) {
                Py_CLEAR(loop_descrs[j]);
            }
            Py_DECREF(resolved);
            return (NPY_CASTING)-1;
        }
        if (i < 3
                && !same_value_dtype(NULLABLE_DESCR(given_descrs[i])->wrapped,
                                     value_descr)) {
            casting = NPY_SAFE_CASTING;
        }
    }
    Py_DECREF(resolved);
    return casting;
}


static int
nullable_clip_loop(PyArrayMethod_Context *context, char *const data[],
        npy_intp const dimensions[], npy_intp const strides[],
        NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    NullableDescr *d[4];
    PyObject *v[4] = {NULL, NULL, NULL, NULL};
    for (int i = 0; i < 4; i++) {
        d[i] = NULLABLE_DESCR(context->descriptors[i]);
    }
    for (int i = 0; i < 4; i++) {
        v[i] = value_view(d[i]->wrapped, data[i], N, strides[i]);
        if (v[i] == NULL) {
            goto fail;
        }
    }

    npy_intp dims[1] = {N};
    PyObject *mask = PyArray_SimpleNew(1, dims, NPY_BOOL);
    if (mask == NULL) {
        goto fail;
    }
    char *m = PyArray_DATA((PyArrayObject *)mask);
    for (npy_intp i = 0; i < N; i++) {
        m[i] = !nullable_is_na(d[0], data[0] + i * strides[0])
            && !nullable_is_na(d[1], data[1] + i * strides[1])
            && !nullable_is_na(d[2], data[2] + i * strides[2]);
    }

    PyObject *args = PyTuple_Pack(3, v[0], v[1], v[2]);
    PyObject *kwargs = Py_BuildValue("{s:O,s:O}", "out", v[3], "where", mask);
    PyObject *res = (args && kwargs)
            ? PyObject_Call(clip_ufunc, args, kwargs) : NULL;
    Py_XDECREF(args); Py_XDECREF(kwargs);
    if (res == NULL) {
        Py_DECREF(mask);
        goto fail;
    }
    Py_DECREF(res);

    /* `where=` skipped the gaps, so those output slots still hold whatever
     * was in the buffer; stamp the reserved pattern over them. */
    for (npy_intp i = 0; i < N; i++) {
        if (!m[i]) {
            nullable_put_na(d[3], data[3] + i * strides[3]);
        }
    }
    Py_DECREF(mask);
    for (int i = 0; i < 4; i++) {
        Py_DECREF(v[i]);
    }
    return 0;

fail:
    for (int i = 0; i < 4; i++) {
        Py_XDECREF(v[i]);
    }
    return -1;
}


static PyType_Slot nullable_clip_slots[] = {
    {NPY_METH_resolve_descriptors, &nullable_clip_resolve},
    {NPY_METH_strided_loop, &nullable_clip_loop},
    {NPY_METH_unaligned_strided_loop, &nullable_clip_loop},
    {0, NULL},
};


/* ------------------------------------------------------- Kleene logic */

static NPY_CASTING
nullable_kleene_resolve_impl(PyArray_Descr *const given_descrs[3],
        PyArray_Descr *loop_descrs[3])
{
    /*
     * Everything -- both inputs and the output -- is Nullable[bool].  Leaving
     * the inputs as they came resolved `logical_and` on Nullable[f8] to
     * (f8, f8, bool), and `reduce` refuses that: it needs one dtype for all
     * three.  That is what stopped `np.all` and `np.any` on anything but a
     * bool array.  NumPy casts the operands, exactly as it does for plain
     * arrays, where `np.all([1.5, 0.0])` is also a bool reduction.
     */
    PyArray_Descr *bool_descr = PyArray_DescrFromType(NPY_BOOL);
    if (bool_descr == NULL) {
        return (NPY_CASTING)-1;
    }
    PyArray_Descr *nullable_bool = new_nullable_descr(bool_descr);
    Py_DECREF(bool_descr);
    if (nullable_bool == NULL) {
        return (NPY_CASTING)-1;
    }
    NPY_CASTING casting = NPY_NO_CASTING;
    for (int i = 0; i < 3; i++) {
        if (i < 2 && given_descrs[i] != NULL
                && NULLABLE_DESCR(given_descrs[i])->wrapped->type_num != NPY_BOOL) {
            casting = NPY_SAFE_CASTING;
        }
        Py_INCREF(nullable_bool);
        loop_descrs[i] = nullable_bool;
    }
    Py_DECREF(nullable_bool);
    return casting;
}


/*
 * One chunk of a Kleene `reduce`.
 *
 * A settling value -- False for AND, True for OR -- fixes the answer for good;
 * no later gap can unfix it, because whatever the missing element turns out to
 * be, the decision is already made.  Otherwise any gap makes the result NA.
 */
static int
nullable_kleene_reduce_chunk(int op, PyArrayMethod_Context *context,
        char *const data[], npy_intp N, npy_intp const strides[])
{
    NullableDescr *acc = NULLABLE_DESCR(context->descriptors[0]);
    NullableDescr *in = NULLABLE_DESCR(context->descriptors[1]);
    NullableDescr *out = NULLABLE_DESCR(context->descriptors[2]);

    PyArrayObject *tb = truthiness(in->wrapped, data[1], N, strides[1]);
    if (tb == NULL) {
        return -1;
    }
    const char *vals = PyArray_DATA(tb);

    int valid = !nullable_is_na(acc, data[0]);
    char value = valid ? (*data[0] != 0) : 0;
    char settles = (op == KLEENE_AND) ? 0 : 1;

    for (npy_intp i = 0; i < N; i++) {
        int b_valid = !nullable_is_na(in, data[1] + i * strides[1]);
        char b = vals[i];

        if (op == KLEENE_XOR) {
            valid = valid && b_valid;
            value = valid ? (value != b) : 0;
            continue;
        }
        if (valid && value == settles) {
            break;
        }
        if (b_valid && b == settles) {
            value = settles;
            valid = 1;
            break;
        }
        if (!b_valid) {
            valid = 0;
        }
        else if (valid) {
            value = (op == KLEENE_AND) ? (value && b) : (value || b);
        }
    }
    Py_DECREF(tb);

    if (valid) {
        *data[2] = value;
    }
    else {
        nullable_put_na(out, data[2]);
    }
    return 0;
}


static int
nullable_kleene_loop_impl(int op, PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[])
{
    npy_intp N = dimensions[0];
    NullableDescr *in0 = NULLABLE_DESCR(context->descriptors[0]);
    NullableDescr *in1 = NULLABLE_DESCR(context->descriptors[1]);
    NullableDescr *out = NULLABLE_DESCR(context->descriptors[2]);

    /*
     * A reduction hands us the accumulator as first input *and* output, both
     * at stride 0.  Running the elementwise code over that writes the same
     * slot N times while still reading the accumulator's original value, so
     * only the last element would survive.  Fold the run sequentially instead.
     */
    if (strides[0] == 0 && data[0] == data[2]) {
        return nullable_kleene_reduce_chunk(op, context, data, N, strides);
    }

    PyArrayObject *ta = truthiness(in0->wrapped, data[0], N, strides[0]);
    PyArrayObject *tb = truthiness(in1->wrapped, data[1], N, strides[1]);
    if (ta == NULL || tb == NULL) {
        Py_XDECREF(ta); Py_XDECREF(tb);
        return -1;
    }
    const char *av = PyArray_DATA(ta);
    const char *bv = PyArray_DATA(tb);

    for (npy_intp i = 0; i < N; i++) {
        int a_valid = !nullable_is_na(in0, data[0] + i * strides[0]);
        int b_valid = !nullable_is_na(in1, data[1] + i * strides[1]);
        char a = av[i], b = bv[i];
        char *slot = data[2] + i * strides[2];
        char value = 0;
        int known = 0;

        switch (op) {
            case KLEENE_AND:
                if ((a_valid && !a) || (b_valid && !b)) { value = 0; known = 1; }
                else if (a_valid && b_valid) { value = 1; known = 1; }
                break;
            case KLEENE_OR:
                if ((a_valid && a) || (b_valid && b)) { value = 1; known = 1; }
                else if (a_valid && b_valid) { value = 0; known = 1; }
                break;
            default:
                known = a_valid && b_valid;
                value = known ? (a != b) : 0;
                break;
        }
        if (known) {
            *slot = value;
        }
        else {
            nullable_put_na(out, slot);
        }
    }
    Py_DECREF(ta); Py_DECREF(tb);
    return 0;
}


static int
nullable_kleene_initial_impl(int op, PyArrayMethod_Context *NPY_UNUSED(context),
        npy_bool NPY_UNUSED(reduction_is_empty), void *initial)
{
    ((char *)initial)[0] = (op == KLEENE_AND);
    return 1;
}


#define NULLABLE_KLEENE(OP)                                                   \
static NPY_CASTING                                                            \
kleene_res_##OP(struct PyArrayMethodObject_tag *NPY_UNUSED(self),        \
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),                       \
        PyArray_Descr *const given_descrs[3], PyArray_Descr *loop_descrs[3],  \
        npy_intp *NPY_UNUSED(view_offset))                                    \
{                                                                             \
    return nullable_kleene_resolve_impl(given_descrs, loop_descrs);           \
}                                                                             \
static int                                                                    \
kleene_loop_##OP(PyArrayMethod_Context *context, char *const data[],     \
        npy_intp const dimensions[], npy_intp const strides[],                \
        NpyAuxData *NPY_UNUSED(auxdata))                                      \
{                                                                             \
    return nullable_kleene_loop_impl(OP, context, data, dimensions, strides); \
}                                                                             \
static int                                                                    \
kleene_init_##OP(PyArrayMethod_Context *context,                         \
        npy_bool reduction_is_empty, void *initial)                           \
{                                                                             \
    return nullable_kleene_initial_impl(OP, context, reduction_is_empty,      \
                                        initial);                             \
}                                                                             \
static PyType_Slot kleene_slots_##OP[] = {                               \
    {NPY_METH_resolve_descriptors, &kleene_res_##OP},                    \
    {NPY_METH_strided_loop, &kleene_loop_##OP},                          \
    {NPY_METH_unaligned_strided_loop, &kleene_loop_##OP},                \
    {NPY_METH_get_reduction_initial, &kleene_init_##OP},                 \
    {0, NULL}                                                                 \
};

NULLABLE_KLEENE(0)
NULLABLE_KLEENE(1)
NULLABLE_KLEENE(2)

static PyType_Slot *kleene_slots[3] = {
    kleene_slots_0, kleene_slots_1, kleene_slots_2};
static PyArrayMethod_Spec kleene_specs[3];


/* ------------------------------------------------------------ dtype setup */

/* NA sorts last and two NAs are equal, as in R */
static int
nullable_compare(const void *a, const void *b, void *arr)
{
    NullableDescr *descr = NULLABLE_DESCR(PyArray_DESCR((PyArrayObject *)arr));
    int na_a = nullable_is_na(descr, (const char *)a);
    int na_b = nullable_is_na(descr, (const char *)b);
    if (na_a || na_b) {
        return na_a ? (na_b ? 0 : 1) : -1;
    }
    return wrapped_compare(descr->wrapped, (const char *)a, (const char *)b);
}


static npy_bool
nullable_nonzero(void *data, void *arr)
{
    NullableDescr *descr = NULLABLE_DESCR(PyArray_DESCR((PyArrayObject *)arr));
    if (nullable_is_na(descr, (const char *)data)) {
        PyErr_SetString(PyExc_ValueError,
                "cannot take the truth value of a missing element; "
                "fill it first, for example with `filled(False)`");
        return NPY_FALSE;
    }
    return wrapped_nonzero(descr->wrapped, (const char *)data);
}


/*
 * argmax / argmin.  A gap is a value nobody knows -- it may be the largest as
 * well as the smallest -- so once one is present the answer is unknown, and the
 * slot says so the only way it can: by pointing at the first gap.  Then
 * `a[a.argmax()]` is NA exactly when `a.max()` is.  That is numpy's own rule
 * for NaN (`np.argmax([3, nan, 5])` is 1); skipping, like `np.nanargmax`, is
 * `nd.argmax(a, skipna=True)`.
 *
 * It used to skip gaps, which answered a MISSING question with IGNORED
 * semantics: `a.max()` was NA while `a[a.argmax()]` was a number.
 *
 * The slot cannot raise instead: `_PyArray_ArgMinMaxCommon` ignores its return
 * value and never looks at the error state.
 */
static int
nullable_arg_extreme(void *data, npy_intp n, npy_intp *ind, void *arr, int sign)
{
    NullableDescr *descr = NULLABLE_DESCR(PyArray_DESCR((PyArrayObject *)arr));
    npy_intp item = descr->base.elsize;
    char *ptr = (char *)data;
    npy_intp best = 0;
    for (npy_intp i = 0; i < n; i++) {
        char *cell = ptr + i * item;
        if (nullable_is_na(descr, cell)) {
            *ind = i;
            return 0;
        }
        if (i > 0 && sign * nullable_compare(cell, ptr + best * item, arr) > 0) {
            best = i;
        }
    }
    *ind = best;
    return 0;
}


static int
nullable_argmax(void *data, npy_intp n, npy_intp *max_ind, void *arr)
{
    return nullable_arg_extreme(data, n, max_ind, arr, 1);
}


static int
nullable_argmin(void *data, npy_intp n, npy_intp *min_ind, void *arr)
{
    return nullable_arg_extreme(data, n, min_ind, arr, -1);
}


/*
 * `np.arange(..., dtype=Nullable(T))`.  NumPy writes the first two elements and
 * asks the dtype to count out the rest; without this slot it refuses the dtype
 * outright.  The values of a Nullable array are laid out exactly like T's, so
 * T's own `fill` runs over the very same buffer.
 *
 * The GIL is held here -- the descriptor carries NPY_NEEDS_PYAPI -- and
 * `PyArray_Arange` checks the error state afterwards, so this can raise.
 */
static int
nullable_fill(void *buffer, npy_intp length, void *arr)
{
    NullableDescr *descr = NULLABLE_DESCR(PyArray_DESCR((PyArrayObject *)arr));
    npy_intp itemsize = descr->base.elsize;
    PyArray_ArrFuncs *funcs = PyDataType_GetArrFuncs(descr->wrapped);

    if (funcs->fill == NULL) {
        PyErr_Format(PyExc_TypeError, "%R cannot count out a range",
                     (PyObject *)descr->wrapped);
        return -1;
    }
    if (length < 2) {
        return 0;
    }
    for (npy_intp i = 0; i < 2; i++) {
        if (nullable_is_na(descr, (char *)buffer + i * itemsize)) {
            PyErr_SetString(PyExc_ValueError,
                    "cannot count out a range from a missing value");
            return -1;
        }
    }
    if (funcs->fill(buffer, length, NULL) < 0) {
        return -1;
    }
    /* the step can land on the reserved value; that must not become a gap */
    for (npy_intp i = 2; i < length; i++) {
        if (nullable_is_na(descr, (char *)buffer + i * itemsize)) {
            PyErr_Format(PyExc_ValueError,
                    "the range reached the value reserved to mean NA in %R",
                    (PyObject *)descr);
            return -1;
        }
    }
    return 0;
}


static PyType_Slot NullableDType_slots[] = {
    {NPY_DT_discover_descr_from_pyobject, &nullable_discover},
    {NPY_DT_default_descr, &nullable_default_descr},
    {NPY_DT_common_dtype, &nullable_common_dtype},
    {NPY_DT_common_instance, &nullable_common_instance},
    {NPY_DT_ensure_canonical, &nullable_ensure_canonical},
    {NPY_DT_setitem, &nullable_setitem},
    {NPY_DT_getitem, &nullable_getitem},
    {NPY_DT_PyArray_ArrFuncs_compare, &nullable_compare},
    {NPY_DT_PyArray_ArrFuncs_nonzero, &nullable_nonzero},
    {NPY_DT_PyArray_ArrFuncs_argmax, &nullable_argmax},
    {NPY_DT_PyArray_ArrFuncs_argmin, &nullable_argmin},
    {NPY_DT_PyArray_ArrFuncs_fill, &nullable_fill},
    {0, NULL}
};

static PyArray_DTypeMeta NullableDType = {{{
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "nulldtype.NullableDType",
        .tp_basicsize = sizeof(NullableDescr),
        .tp_new = nullable_new,
        .tp_dealloc = (destructor)nullable_dealloc,
        .tp_methods = nullable_methods,
        .tp_getset = nullable_getset,
        .tp_repr = (reprfunc)nullable_repr,
        .tp_str = (reprfunc)nullable_repr,
    }},
};


static PyArray_DTypeMeta *self_dtypes[2] = {NULL, NULL};
static PyType_Slot self_slots[] = {
    {NPY_METH_resolve_descriptors, &nullable_to_nullable_resolve},
    {NPY_METH_strided_loop, &nullable_to_nullable_loop},
    {NPY_METH_unaligned_strided_loop, &nullable_to_nullable_loop},
    {0, NULL}
};
static PyArrayMethod_Spec NullableSelfCast = {
    .name = "nullable_to_nullable", .nin = 1, .nout = 1,
    .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
    .casting = NPY_NO_CASTING,
    .dtypes = self_dtypes, .slots = self_slots,
};

static PyType_Slot to_slots[] = {
    {NPY_METH_resolve_descriptors, &to_nullable_resolve},
    {NPY_METH_strided_loop, &to_nullable_loop},
    {NPY_METH_unaligned_strided_loop, &to_nullable_loop},
    {0, NULL}
};
static PyType_Slot from_slots[] = {
    {NPY_METH_resolve_descriptors, &from_nullable_resolve},
    {NPY_METH_strided_loop, &from_nullable_loop},
    {NPY_METH_unaligned_strided_loop, &from_nullable_loop},
    {0, NULL}
};

/* the plain dtypes that cast in and out, and that promoters pair with */
#define N_NULLABLE_TYPES 21
static const int nullable_typenums[N_NULLABLE_TYPES] = {
    NPY_BOOL, NPY_INT8, NPY_INT16, NPY_INT32, NPY_INT64,
    NPY_UINT8, NPY_UINT16, NPY_UINT32, NPY_UINT64,
    NPY_FLOAT16, NPY_FLOAT32, NPY_FLOAT64,
    NPY_COMPLEX64, NPY_COMPLEX128,
    /* free: numpy already reserves INT64_MIN in these and calls it NaT */
    NPY_DATETIME, NPY_TIMEDELTA,
    /*
     * No patterns of their own -- stored as float64 and complex128, see
     * `substitute_long_double` -- but plain arrays of them still cast in.
     */
    NPY_LONGDOUBLE, NPY_CLONGDOUBLE,
    /* any width; VOID covers records too, which `new_nullable_descr` refuses */
    NPY_STRING, NPY_UNICODE, NPY_VOID,
};
static PyArray_DTypeMeta *cast_dtypes[N_NULLABLE_TYPES][2][2];
static PyArrayMethod_Spec cast_specs[N_NULLABLE_TYPES][2];
static PyArrayMethod_Spec *all_casts[N_NULLABLE_TYPES * 2 + 2];
static PyArray_DTypeMeta *binop_dtypes[3] = {NULL, NULL, NULL};
static PyArrayMethod_Spec binop_specs[N_BINOPS];


static int
init_nullable_dtype(PyObject *m)
{
    ((PyObject *)&NullableDType)->ob_type = &PyArrayDTypeMeta_Type;
    ((PyTypeObject *)&NullableDType)->tp_base = &PyArrayDescr_Type;
    if (PyType_Ready((PyTypeObject *)&NullableDType) < 0) {
        return -1;
    }

    self_dtypes[0] = &NullableDType;
    self_dtypes[1] = &NullableDType;
    int n = 0;
    all_casts[n++] = &NullableSelfCast;
    for (int i = 0; i < N_NULLABLE_TYPES; i++) {
        PyArray_DTypeMeta *plain = dtypemeta_from_typenum(nullable_typenums[i]);
        if (plain == NULL) {
            return -1;
        }
        cast_dtypes[i][0][0] = plain;
        cast_dtypes[i][0][1] = &NullableDType;
        cast_specs[i][0] = (PyArrayMethod_Spec){
            .name = "to_nullable", .nin = 1, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_SAME_KIND_CASTING,
            .dtypes = cast_dtypes[i][0], .slots = to_slots,
        };
        all_casts[n++] = &cast_specs[i][0];

        cast_dtypes[i][1][0] = &NullableDType;
        cast_dtypes[i][1][1] = plain;
        cast_specs[i][1] = (PyArrayMethod_Spec){
            .name = "from_nullable", .nin = 1, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_UNSAFE_CASTING,
            .dtypes = cast_dtypes[i][1], .slots = from_slots,
        };
        all_casts[n++] = &cast_specs[i][1];
    }
    all_casts[n] = NULL;

    PyArrayDTypeMeta_Spec spec = {
        .typeobj = &NAType,
        .flags = NPY_DT_PARAMETRIC,
        .casts = all_casts,
        .slots = NullableDType_slots,
    };
    if (PyArrayInitDTypeMeta_FromSpec(&NullableDType, &spec) < 0) {
        return -1;
    }

    binop_dtypes[0] = &NullableDType;
    binop_dtypes[1] = &NullableDType;
    binop_dtypes[2] = &NullableDType;
    PyArray_DTypeMeta *pyscalars[3] = {
        &PyArray_PyLongDType, &PyArray_PyFloatDType, &PyArray_PyComplexDType};
    for (int k = 0; k < N_BINOPS; k++) {
        /*
         * numpy's own rule for its legacy loops: an op with an identity, or
         * declared "reorderable none" like `maximum`, may be reduced over
         * several axes at once.  Without the flag even `a.sum()` on a 2-d
         * array refuses.  NA propagation keeps such an op associative and
         * commutative, and so does the Kleene table on bool.
         */
        int reorderable =
                ((PyUFuncObject *)binop_ufuncs[k])->identity != PyUFunc_None;
        binop_specs[k] = (PyArrayMethod_Spec){
            .name = binop_names[k], .nin = 2, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI
                     | (reorderable ? NPY_METH_IS_REORDERABLE : 0),
            .casting = NPY_NO_CASTING,
            .dtypes = binop_dtypes, .slots = nullable_binop_slots[k],
        };
        if (PyUFunc_AddLoopFromSpec(binop_ufuncs[k], &binop_specs[k]) < 0) {
            return -1;
        }
        for (int i = 0; i < N_NULLABLE_TYPES; i++) {
            PyArray_DTypeMeta *plain =
                    dtypemeta_from_typenum(nullable_typenums[i]);
            if (plain == NULL
                    || register_promoter(binop_ufuncs[k], &NullableDType, plain) < 0
                    || register_promoter(binop_ufuncs[k], plain, &NullableDType) < 0) {
                return -1;
            }
        }
        for (int i = 0; i < 3; i++) {
            if (register_promoter(binop_ufuncs[k], &NullableDType, pyscalars[i]) < 0
                    || register_promoter(binop_ufuncs[k], pyscalars[i], &NullableDType) < 0) {
                return -1;
            }
        }
    }

    unop_dtypes[0] = &NullableDType;
    unop_dtypes[1] = &NullableDType;
    for (int k = 0; k < N_UNOPS; k++) {
        unop_specs[k] = (PyArrayMethod_Spec){
            .name = unop_names[k], .nin = 1, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_NO_CASTING,
            .dtypes = unop_dtypes, .slots = nullable_unop_slots[k],
        };
        if (PyUFunc_AddLoopFromSpec(unop_ufuncs[k], &unop_specs[k]) < 0) {
            return -1;
        }
    }

    {
        static PyArrayMethod_Spec matmul_spec;
        matmul_spec = (PyArrayMethod_Spec){
            .name = "matmul", .nin = 2, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_NO_CASTING,
            .dtypes = binop_dtypes, .slots = nullable_matmul_slots,
        };
        if (PyUFunc_AddLoopFromSpec(np_matmul, &matmul_spec) < 0) {
            return -1;
        }
    }

    {
        static PyArray_DTypeMeta *clip_dtypes[4];
        static PyArrayMethod_Spec clip_spec;
        for (int i = 0; i < 4; i++) {
            clip_dtypes[i] = &NullableDType;
        }
        clip_spec = (PyArrayMethod_Spec){
            .name = "clip", .nin = 3, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_NO_CASTING,
            .dtypes = clip_dtypes, .slots = nullable_clip_slots,
        };
        if (PyUFunc_AddLoopFromSpec(clip_ufunc, &clip_spec) < 0
                || register_clip_promoter(&NullableDType) < 0) {
            return -1;
        }
    }

    for (int k = 0; k < 3; k++) {
        kleene_specs[k] = (PyArrayMethod_Spec){
            .name = kleene_names[k], .nin = 2, .nout = 1,
            /* Kleene and/or/xor are associative and commutative */
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI
                     | NPY_METH_IS_REORDERABLE,
            .casting = NPY_NO_CASTING,
            .dtypes = binop_dtypes, .slots = kleene_slots[k],
        };
        if (PyUFunc_AddLoopFromSpec(kleene_ufuncs[k], &kleene_specs[k]) < 0) {
            return -1;
        }
        for (int i = 0; i < N_NULLABLE_TYPES; i++) {
            PyArray_DTypeMeta *plain =
                    dtypemeta_from_typenum(nullable_typenums[i]);
            if (plain == NULL
                    || register_promoter(kleene_ufuncs[k], &NullableDType, plain) < 0
                    || register_promoter(kleene_ufuncs[k], plain, &NullableDType) < 0) {
                return -1;
            }
        }
    }

    if (PyModule_AddObjectRef(m, "NullableDType", (PyObject *)&NullableDType) < 0
            || PyModule_AddObjectRef(m, "Nullable", (PyObject *)&NullableDType) < 0) {
        return -1;
    }
    return 0;
}


/* ---------------------------------------------------------------- module */

/* the numpy callables and ufuncs the loops hand their work to */
static int
init_ufunc_tables(void)
{
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) {
        return -1;
    }
    int ok = (np_copyto = PyObject_GetAttrString(numpy, "copyto")) != NULL
          && (np_asarray = PyObject_GetAttrString(numpy, "asarray")) != NULL
          && (np_concatenate = PyObject_GetAttrString(numpy, "concatenate")) != NULL
          && (np_matmul = PyObject_GetAttrString(numpy, "matmul")) != NULL;
    for (int k = 0; ok && k < N_BINOPS; k++) {
        ok = (binop_ufuncs[k] = PyObject_GetAttrString(numpy, binop_names[k])) != NULL;
    }
    for (int k = 0; ok && k < N_UNOPS; k++) {
        ok = (unop_ufuncs[k] = PyObject_GetAttrString(numpy, unop_names[k])) != NULL;
    }
    for (int k = 0; ok && k < 3; k++) {
        ok = (kleene_ufuncs[k] = PyObject_GetAttrString(numpy, kleene_names[k])) != NULL;
    }
    Py_DECREF(numpy);
    if (!ok) {
        return -1;
    }
    /* `clip` the ufunc, not `np.clip` the Python function */
    PyObject *umath = PyImport_ImportModule("numpy._core.umath");
    if (umath == NULL) {
        return -1;
    }
    clip_ufunc = PyObject_GetAttrString(umath, "clip");
    Py_DECREF(umath);
    return (clip_ufunc != NULL) ? 0 : -1;
}


static struct PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_nulldtype",
    .m_size = -1,
};


PyMODINIT_FUNC
PyInit__nulldtype(void)
{
    if (_import_array() < 0) {
        return NULL;
    }
    if (_import_umath() < 0) {
        return NULL;
    }

    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }

    if (PyType_Ready(&NAType) < 0) {
        goto fail;
    }
    NA_singleton = PyObject_CallNoArgs((PyObject *)&NAType);
    if (NA_singleton == NULL
            || PyModule_AddObjectRef(m, "NA", NA_singleton) < 0) {
        goto fail;
    }
    LongDoubleWarning = PyErr_NewExceptionWithDoc(
            "nulldtype.LongDoubleWarning",
            "longdouble or clongdouble was stored as float64 or complex128, "
            "which can lose precision.",
            PyExc_UserWarning, NULL);
    if (LongDoubleWarning == NULL
            || PyModule_AddObjectRef(m, "LongDoubleWarning",
                                     LongDoubleWarning) < 0) {
        goto fail;
    }

    /*
     * The op tables, so the test suite sweeps exactly what is registered
     * instead of a hand-kept copy that can drift out of date.
     */
    {
        PyObject *bl = PyList_New(N_BINOPS), *ul = PyList_New(N_UNOPS);
        if (bl == NULL || ul == NULL) {
            Py_XDECREF(bl); Py_XDECREF(ul);
            goto fail;
        }
        for (int k = 0; k < N_BINOPS; k++) {
            PyList_SET_ITEM(bl, k, PyUnicode_FromString(binop_names[k]));
        }
        for (int k = 0; k < N_UNOPS; k++) {
            PyList_SET_ITEM(ul, k, PyUnicode_FromString(unop_names[k]));
        }
        int bad = PyModule_AddObject(m, "binop_names", bl) < 0
               || PyModule_AddObject(m, "unop_names", ul) < 0;
        if (bad) {
            goto fail;
        }
    }

    if (init_ufunc_tables() < 0 || init_nullable_dtype(m) < 0) {
        goto fail;
    }
    return m;

  fail:
    Py_DECREF(m);
    return NULL;
}
