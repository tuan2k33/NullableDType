/*
 * Nullable[T] — a parametric DType that adds a validity flag to any other dtype.
 *
 * Layout is interleaved, one element is `{ value bytes..., valid byte }`, so an
 * array of Nullable[T] keeps a single buffer as NumPy requires, and the value
 * fields form a strided array that T's own loops can consume directly.
 *
 *
 * Why the bytes under a missing value are always zero
 * ---------------------------------------------------
 * NA means there is no value, so the bytes of a missing element are padding,
 * never data.  We write zeros over them and never read them back.  That is a
 * deliberate choice, and it is the one thing `numpy.ma` does not do:
 *
 *  - Every leak bug in `numpy.ma` (gh-9750, `bool()`, `complex()`, casts to
 *    plain arrays) ends with the value under the mask reaching the user, where
 *    it looks like real data.  A canonical zero cannot be mistaken for one.
 *  - Results stay deterministic.  Keeping whatever was computed would make
 *    `tobytes()`, hashes and pickles depend on evaluation order and on
 *    compiler optimisations.
 *  - Values a user masked out do not travel into files or over the wire.
 *  - Since the binary loops pass `where=` to the wrapped ufunc, a missing
 *    element is never computed at all; without the memset its slot would hold
 *    whatever the (possibly uninitialised) output buffer happened to contain.
 *
 * R makes the opposite trade for floats: `NA_real_` is a NaN with a dedicated
 * payload, so even raw bytes identify it.  That only works for dtypes with
 * spare bit patterns, and Nullable[T] has to work for every T.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/dtype_api.h"

typedef struct {
    PyArray_Descr base;
    PyArray_Descr *wrapped;   /* the dtype whose values we store */
} NullableDescr;

static PyArray_DTypeMeta NullableDType;
/* defined in sentinel.inc, declared here so the shared promoter can name it */
static PyArray_DTypeMeta SentinelDType;

/* built at import time, one pair of casts per wrapped dtype we support */
#define N_WRAPPABLE 16
static const int wrappable_typenums[N_WRAPPABLE] = {
    NPY_BOOL, NPY_INT8, NPY_UINT8, NPY_INT16, NPY_UINT16, NPY_INT32,
    NPY_UINT32, NPY_INT64, NPY_UINT64, NPY_FLOAT16, NPY_FLOAT32, NPY_FLOAT64,
    NPY_COMPLEX64, NPY_COMPLEX128, NPY_DATETIME, NPY_TIMEDELTA,
};


#define NULLABLE_DESCR(d) ((NullableDescr *)(d))
/* offset of the validity byte inside one element */
#define VALID_OFFSET(d) (NULLABLE_DESCR(d)->wrapped->elsize)

/* The Python-level marker for a missing value: `nullable_dtype.NA` */
static PyObject *NA_singleton = NULL;

static int same_value_dtype(PyArray_Descr *a, PyArray_Descr *b);


/*
 * How many bytes one element takes: the value, then its validity byte.
 *
 * Rounding this up to a power of two is tempting -- a 9-byte stride is the
 * worst case for a vector unit -- and it was tried.  Measured on uint64 add,
 * 16 bytes is 1.04-1.21x faster while the array fits in cache and
 * 1.6-1.9x *slower* once it does not, for 78% more memory.  The slowdown
 * tracks 16/9 = 1.78 almost exactly, so it is pure bandwidth and no amount of
 * tuning will move it.  Padding buys a little where speed does not matter and
 * costs a lot where it does; the tight layout stays.
 */
static npy_intp
nullable_padded_size(PyArray_Descr *wrapped)
{
    return wrapped->elsize + 1;
}


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
 * What both layouts store for a requested dtype: long double substituted as
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


static PyArray_Descr *new_nullable_descr_impl(PyArray_Descr *wrapped);

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
    if (PyDataType_REFCHK(wrapped)) {
        PyErr_SetString(PyExc_TypeError,
                "Nullable[T] does not support dtypes holding references yet");
        return NULL;
    }
    /*
     * Always store the canonical instance of the wrapped dtype.  Unpickling
     * hands back a fresh `float64` descriptor rather than the singleton, and
     * everything here compares wrapped dtypes by identity.
     */
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
    NullableDescr *new = (NullableDescr *)PyArrayDescr_Type.tp_new(
            (PyTypeObject *)&NullableDType, NULL, NULL);
    if (new == NULL) {
        Py_XDECREF(canonical);
        return NULL;
    }
    Py_INCREF(wrapped);
    Py_XDECREF(canonical);
    new->wrapped = wrapped;
    new->base.elsize = nullable_padded_size(wrapped);
    new->base.alignment = wrapped->alignment;
    /*
     * NPY_NEEDS_PYAPI keeps the GIL held around the legacy slots.  `nonzero`
     * raises on a gap, and numpy calls it with the GIL released once an array
     * has more than 500 elements unless this flag is set: a segfault, on every
     * wrapped dtype.  `sort` and `argmax` on records need it too, because
     * VOID_compare asks the memory handler for scratch buffers.
     */
    new->base.flags = NPY_USE_GETITEM | NPY_USE_SETITEM | NPY_NEEDS_INIT
                      | NPY_NEEDS_PYAPI;
    new->base.byteorder = wrapped->byteorder;
    return (PyArray_Descr *)new;
}


static void
nullable_dealloc(NullableDescr *self)
{
    Py_CLEAR(self->wrapped);
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
nullable_get_wrapped(NullableDescr *self, void *NPY_UNUSED(closure))
{
    Py_INCREF(self->wrapped);
    return (PyObject *)self->wrapped;
}


static PyObject *
nullable_reduce(NullableDescr *self, PyObject *NPY_UNUSED(args))
{
    /* rebuild as `Nullable(wrapped)`; the default dtype pickling refuses */
    return Py_BuildValue("O(O)", Py_TYPE(self), (PyObject *)self->wrapped);
}


static PyMethodDef nullable_methods[] = {
    {"__reduce__", (PyCFunction)nullable_reduce, METH_NOARGS,
     "pickle support"},
    {NULL, NULL, 0, NULL}
};


static PyGetSetDef nullable_getset[] = {
    {"wrapped", (getter)nullable_get_wrapped, NULL,
     "the dtype whose values this one stores", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};


static PyObject *
nullable_repr(NullableDescr *self)
{
    return PyUnicode_FromFormat("Nullable(%R)", (PyObject *)self->wrapped);
}


/* ---------------------------------------------------------------- slots */

static PyArray_Descr *
nullable_default_descr(PyArray_DTypeMeta *NPY_UNUSED(cls))
{
    PyArray_Descr *f8 = PyArray_DescrFromType(NPY_DOUBLE);
    PyArray_Descr *res = new_nullable_descr(f8);
    Py_DECREF(f8);
    return res;
}


static PyArray_Descr *
nullable_discover_from_pyobject(PyArray_DTypeMeta *cls, PyObject *NPY_UNUSED(obj))
{
    return nullable_default_descr(cls);
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


static PyArray_DTypeMeta *
nullable_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    /* Nullable[T] wins over the plain dtype it can wrap. */
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
    if (same_value_dtype(NULLABLE_DESCR(descr1)->wrapped,
                         NULLABLE_DESCR(descr2)->wrapped)) {
        Py_INCREF(descr1);
        return descr1;
    }
    PyErr_SetString(PyExc_TypeError,
            "cannot combine two Nullable dtypes wrapping different dtypes yet");
    return NULL;
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
    npy_intp voff = VALID_OFFSET(descr);

    if (obj == NA_singleton || obj == Py_None) {
        memset(dataptr, 0, voff);      /* canonical fill: never observable */
        dataptr[voff] = 0;             /* invalid */
        return 0;
    }

    /* Let the wrapped dtype do the conversion, via a temporary 0-d array. */
    PyArray_Descr *wrapped = NULLABLE_DESCR(descr)->wrapped;
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
    memcpy(dataptr, PyArray_DATA(tmp), voff);
    dataptr[voff] = 1;                 /* valid */
    Py_DECREF(tmp);
    return 0;
}


static PyObject *
nullable_getitem(PyArray_Descr *descr, char *dataptr)
{
    if (dataptr[VALID_OFFSET(descr)] == 0) {
        Py_INCREF(NA_singleton);
        return NA_singleton;
    }
    return PyArray_Scalar(dataptr, NULLABLE_DESCR(descr)->wrapped, NULL);
}


static PyObject *np_copyto = NULL;
static PyObject *np_asarray = NULL;
static PyObject *np_concatenate = NULL;
static PyObject *value_view(PyArray_Descr *wrapped, char *data,
                            npy_intp n, npy_intp stride);
static PyArray_Descr *unwrap(PyArray_Descr *descr);


/* np.copyto(dst, src, casting="unsafe"): values under a gap are meaningless */
static int
copy_values(PyObject *dst, PyObject *src)
{
    PyObject *args = PyTuple_Pack(2, dst, src);
    PyObject *kwargs = Py_BuildValue("{s:s}", "casting", "unsafe");
    PyObject *res = (args && kwargs)
            ? PyObject_Call(np_copyto, args, kwargs) : NULL;
    Py_XDECREF(args); Py_XDECREF(kwargs);
    if (res == NULL) {
        return -1;
    }
    Py_DECREF(res);
    return 0;
}

/* ------------------------------------------------------------ self cast */

static NPY_CASTING
nullable_to_nullable_resolve(
        struct PyArrayMethodObject_tag *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[2]),
        PyArray_Descr *const given_descrs[2],
        PyArray_Descr *loop_descrs[2],
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

    PyArray_Descr *from = NULLABLE_DESCR(loop_descrs[0])->wrapped;
    PyArray_Descr *to = NULLABLE_DESCR(loop_descrs[1])->wrapped;
    if (!same_value_dtype(from, to)) {
        /* the values need casting; the validity byte just rides along */
        return PyArray_CanCastTypeTo(from, to, NPY_SAFE_CASTING)
                ? NPY_SAFE_CASTING : NPY_UNSAFE_CASTING;
    }
    *view_offset = 0;
    return NPY_NO_CASTING;
}


static int
nullable_to_nullable_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in = data[0], *out = data[1];
    npy_intp in_stride = strides[0], out_stride = strides[1];
    PyArray_Descr *from = NULLABLE_DESCR(context->descriptors[0])->wrapped;
    PyArray_Descr *to = NULLABLE_DESCR(context->descriptors[1])->wrapped;

    if (same_value_dtype(from, to)) {
        size_t itemsize = (size_t)context->descriptors[0]->elsize;
        /* one bulk copy when both sides are packed, instead of N small ones */
        if (in_stride == (npy_intp)itemsize && out_stride == (npy_intp)itemsize) {
            memcpy(out, in, itemsize * (size_t)N);
            return 0;
        }
        while (N--) {
            memcpy(out, in, itemsize);
            in += in_stride;
            out += out_stride;
        }
        return 0;
    }

    /* different value dtypes: cast the values, carry the validity over */
    PyObject *a = value_view(from, in, N, in_stride);
    PyObject *b = value_view(to, out, N, out_stride);
    if (a == NULL || b == NULL) {
        Py_XDECREF(a); Py_XDECREF(b);
        return -1;
    }
    PyObject *res = PyObject_CallFunction(np_copyto, "OO", b, a);
    Py_DECREF(a); Py_DECREF(b);
    if (res == NULL) {
        return -1;
    }
    Py_DECREF(res);

    for (npy_intp i = 0; i < N; i++) {
        out[i * out_stride + to->elsize] = in[i * in_stride + from->elsize];
    }
    return 0;
}


static PyArray_DTypeMeta *nullable_casting_dtypes[2] = {NULL, NULL};

static PyType_Slot nullable_to_nullable_slots[] = {
    {NPY_METH_resolve_descriptors, &nullable_to_nullable_resolve},
    {NPY_METH_strided_loop, &nullable_to_nullable_loop},
    {NPY_METH_unaligned_strided_loop, &nullable_to_nullable_loop},
    {0, NULL}
};

static PyArrayMethod_Spec NullableToNullableCastSpec = {
    .name = "nullable_to_nullable",
    .nin = 1,
    .nout = 1,
    .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
    .casting = NPY_NO_CASTING,
    .dtypes = nullable_casting_dtypes,
    .slots = nullable_to_nullable_slots,
};


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
 * The wrapped dtype's `compare`, callable on an element of either layout: the
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
     * not "behaved", and flag-layout elements are misaligned by design (stride
     * is itemsize + 1).  A new-style DType has no `copyswap`, so that fallback
     * would call a null pointer.  Copy into an aligned buffer and pass NULL,
     * which is the documented "just read it" path.
     */
    npy_clongdouble aligned;
    if (n > (npy_intp)sizeof(aligned)) {
        return NPY_FALSE;
    }
    memcpy(&aligned, data, (size_t)n);
    return funcs->nonzero((char *)&aligned, NULL);
}


/*
 * Ordering for `sort`, `argsort` and friends.  Missing values sort last, and
 * two of them compare equal, which is what R does.
 */
static int
nullable_compare(const void *a, const void *b, void *arr)
{
    PyArray_Descr *descr = PyArray_DESCR((PyArrayObject *)arr);
    PyArray_Descr *wrapped = NULLABLE_DESCR(descr)->wrapped;
    npy_intp voff = wrapped->elsize;
    char a_valid = ((const char *)a)[voff];
    char b_valid = ((const char *)b)[voff];

    if (!a_valid || !b_valid) {
        return a_valid ? -1 : (b_valid ? 1 : 0);
    }
    return wrapped_compare(wrapped, (const char *)a, (const char *)b);
}


/*
 * `np.zeros` should give a real zero, not a gap.  Without this slot the buffer
 * is only zero-filled, which leaves the validity byte at 0 and makes every
 * element missing.  `np.empty` keeps that behaviour on purpose: uninitialised
 * memory is better described as missing than as garbage.
 */
static int
nullable_fill_zero(void *NPY_UNUSED(traverse_context),
        const PyArray_Descr *descr, char *data, npy_intp size,
        npy_intp stride, NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp voff = NULLABLE_DESCR(descr)->wrapped->elsize;
    while (size--) {
        memset(data, 0, (size_t)voff);
        data[voff] = 1;
        data += stride;
    }
    return 0;
}


static int
nullable_get_fill_zero_loop(void *NPY_UNUSED(traverse_context),
        const PyArray_Descr *NPY_UNUSED(descr), int NPY_UNUSED(aligned),
        npy_intp NPY_UNUSED(fixed_stride),
        PyArrayMethod_TraverseLoop **out_loop, NpyAuxData **out_auxdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    *flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    *out_loop = &nullable_fill_zero;
    *out_auxdata = NULL;
    return 0;
}


/*
 * Truth of a single element, used by `nonzero` and by boolean indexing.
 * Without this slot NumPy dereferences a null pointer and crashes.  A missing
 * value has no truth to report, so refuse rather than guess.
 */
static npy_bool
nullable_nonzero(void *data, void *arr)
{
    PyArray_Descr *descr = PyArray_DESCR((PyArrayObject *)arr);
    PyArray_Descr *wrapped = NULLABLE_DESCR(descr)->wrapped;
    if (((char *)data)[wrapped->elsize] == 0) {
        PyErr_SetString(PyExc_ValueError,
                "cannot take the truth value of a missing element; "
                "fill it first, for example with `filled(False)`");
        return NPY_FALSE;
    }
    return wrapped_nonzero(wrapped, (const char *)data);
}


/* argmax / argmin, with missing values never winning */
static int
nullable_argmax(void *data, npy_intp n, npy_intp *max_ind, void *arr)
{
    PyArray_Descr *descr = PyArray_DESCR((PyArrayObject *)arr);
    npy_intp itemsize = descr->elsize;
    npy_intp voff = VALID_OFFSET(descr);
    char *ptr = (char *)data;
    npy_intp best = -1;

    for (npy_intp i = 0; i < n; i++) {
        if (ptr[i * itemsize + voff] == 0) {
            continue;                       /* a gap can never be the maximum */
        }
        if (best < 0 || nullable_compare(ptr + i * itemsize,
                                         ptr + best * itemsize, arr) > 0) {
            best = i;
        }
    }
    /*
     * All gaps: answer 0, which is what numpy itself answers for an array of
     * all-NaT or all-nan.  Raising would be better, but this slot cannot: the
     * caller (_PyArray_ArgMinMaxCommon) ignores the return value and never
     * looks at the error state, so an exception set here would come out as a
     * SystemError instead of a message.  Nothing is invented either way --
     * `a[a.argmax()]` is NA.
     */
    *max_ind = (best < 0) ? 0 : best;
    return 0;
}


static int
nullable_argmin(void *data, npy_intp n, npy_intp *min_ind, void *arr)
{
    PyArray_Descr *descr = PyArray_DESCR((PyArrayObject *)arr);
    npy_intp itemsize = descr->elsize;
    npy_intp voff = VALID_OFFSET(descr);
    char *ptr = (char *)data;
    npy_intp best = -1;

    for (npy_intp i = 0; i < n; i++) {
        if (ptr[i * itemsize + voff] == 0) {
            continue;
        }
        if (best < 0 || nullable_compare(ptr + i * itemsize,
                                         ptr + best * itemsize, arr) < 0) {
            best = i;
        }
    }
    /*
     * All gaps: answer 0, which is what numpy itself answers for an array of
     * all-NaT or all-nan.  Raising would be better, but this slot cannot: the
     * caller (_PyArray_ArgMinMaxCommon) ignores the return value and never
     * looks at the error state, so an exception set here would come out as a
     * SystemError instead of a message.  Nothing is invented either way --
     * `a[a.argmax()]` is NA.
     */
    *min_ind = (best < 0) ? 0 : best;
    return 0;
}


static PyType_Slot NullableDType_slots[] = {
    {NPY_DT_discover_descr_from_pyobject, &nullable_discover_from_pyobject},
    {NPY_DT_default_descr, &nullable_default_descr},
    {NPY_DT_common_dtype, &nullable_common_dtype},
    {NPY_DT_common_instance, &nullable_common_instance},
    {NPY_DT_ensure_canonical, &nullable_ensure_canonical},
    {NPY_DT_setitem, &nullable_setitem},
    {NPY_DT_getitem, &nullable_getitem},
    {NPY_DT_PyArray_ArrFuncs_compare, &nullable_compare},
    {NPY_DT_get_fill_zero_loop, &nullable_get_fill_zero_loop},
    {NPY_DT_PyArray_ArrFuncs_nonzero, &nullable_nonzero},
    {NPY_DT_PyArray_ArrFuncs_argmax, &nullable_argmax},
    {NPY_DT_PyArray_ArrFuncs_argmin, &nullable_argmin},
    {0, NULL}
};


/* -------------------------------------------------- casts T <-> Nullable[T] */



static NPY_CASTING
to_nullable_resolve(
        struct PyArrayMethodObject_tag *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[2]),
        PyArray_Descr *const given_descrs[2],
        PyArray_Descr *loop_descrs[2],
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
        if (!same_value_dtype(NULLABLE_DESCR(given_descrs[1])->wrapped,
                              given_descrs[0])) {
            /* the values still have to be cast; report that safety honestly */
            return PyArray_CanCastTypeTo(given_descrs[0],
                        NULLABLE_DESCR(given_descrs[1])->wrapped,
                        NPY_SAFE_CASTING)
                    ? NPY_SAFE_CASTING : NPY_UNSAFE_CASTING;
        }
    }
    /* adding a validity byte never loses information */
    return NPY_SAFE_CASTING;
}


static int
to_nullable_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    PyArray_Descr *src = context->descriptors[0];
    PyArray_Descr *dst = NULLABLE_DESCR(context->descriptors[1])->wrapped;
    size_t itemsize = (size_t)dst->elsize;
    char *in = data[0], *out = data[1];

    if (same_value_dtype(src, dst)) {
        while (N--) {
            memcpy(out, in, itemsize);
            out[itemsize] = 1;          /* everything coming in is valid */
            in += strides[0];
            out += strides[1];
        }
        return 0;
    }

    /* the value dtype changes too, so let NumPy cast the value fields */
    PyObject *a = value_view(src, data[0], N, strides[0]);
    PyObject *b = value_view(dst, data[1], N, strides[1]);
    if (a == NULL || b == NULL) {
        Py_XDECREF(a); Py_XDECREF(b);
        return -1;
    }
    PyObject *res = PyObject_CallFunction(np_copyto, "OO", b, a);
    Py_DECREF(a); Py_DECREF(b);
    if (res == NULL) {
        return -1;
    }
    Py_DECREF(res);

    for (npy_intp i = 0; i < N; i++) {
        out[i * strides[1] + itemsize] = 1;
    }
    return 0;
}


static NPY_CASTING
from_nullable_resolve(
        struct PyArrayMethodObject_tag *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[2]),
        PyArray_Descr *const given_descrs[2],
        PyArray_Descr *loop_descrs[2],
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
        /* another dtype is fine too: the loop lets NumPy cast the values */
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }
    /* dropping the mask can lose information, so never do it implicitly */
    return NPY_UNSAFE_CASTING;
}


static int
from_nullable_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    PyArray_Descr *wrapped = NULLABLE_DESCR(context->descriptors[0])->wrapped;
    PyArray_Descr *dst = context->descriptors[1];
    size_t itemsize = (size_t)wrapped->elsize;

    for (npy_intp i = 0; i < N; i++) {
        if (data[0][i * strides[0] + itemsize] == 0) {
            PyErr_Format(PyExc_ValueError,
                    "cannot convert a missing value to %R; "
                    "fill it first, e.g. with `filled()`", wrapped);
            return -1;
        }
    }
    if (same_value_dtype(wrapped, dst)) {
        for (npy_intp i = 0; i < N; i++) {
            memcpy(data[1] + i * strides[1], data[0] + i * strides[0], itemsize);
        }
        return 0;
    }
    /* another plain dtype -- wider, narrower, big-endian: NumPy casts */
    PyObject *a = value_view(wrapped, data[0], N, strides[0]);
    PyObject *b = value_view(dst, data[1], N, strides[1]);
    int rc = (a && b) ? copy_values(b, a) : -1;
    Py_XDECREF(a); Py_XDECREF(b);
    return rc;
}


static PyType_Slot to_nullable_slots[] = {
    {NPY_METH_resolve_descriptors, &to_nullable_resolve},
    {NPY_METH_strided_loop, &to_nullable_loop},
    {NPY_METH_unaligned_strided_loop, &to_nullable_loop},
    {0, NULL}
};

static PyType_Slot from_nullable_slots[] = {
    {NPY_METH_resolve_descriptors, &from_nullable_resolve},
    {NPY_METH_strided_loop, &from_nullable_loop},
    {NPY_METH_unaligned_strided_loop, &from_nullable_loop},
    {0, NULL}
};

/*
 * S, U and V get casts in and out as well, though nothing else is registered
 * for them the way it is for numbers.  Assigning a numpy scalar is a cast from
 * the scalar's dtype: without one, `np.bytes_` and records were refused and an
 * unstructured `np.void` segfaulted inside numpy.  Records stay on this layout,
 * so they need it most.
 */
#define N_FLAG_FLEXIBLE 3
static const int flag_flexible_typenums[N_FLAG_FLEXIBLE] = {
    NPY_STRING, NPY_UNICODE, NPY_VOID,
};
#define N_FLAG_CASTS (N_WRAPPABLE + N_FLAG_FLEXIBLE)
static PyArray_DTypeMeta *cast_dtypes[N_FLAG_CASTS][2][2];
static PyArrayMethod_Spec cast_specs[N_FLAG_CASTS][2];
static PyArrayMethod_Spec *all_casts[N_FLAG_CASTS * 2 + 2];


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


static int
build_casts(void)
{
    int n = 0;
    all_casts[n++] = &NullableToNullableCastSpec;

    for (int i = 0; i < N_FLAG_CASTS; i++) {
        PyArray_DTypeMeta *wrapped_dt = dtypemeta_from_typenum(
                i < N_WRAPPABLE ? wrappable_typenums[i]
                                : flag_flexible_typenums[i - N_WRAPPABLE]);
        if (wrapped_dt == NULL) {
            return -1;
        }
        cast_dtypes[i][0][0] = wrapped_dt;
        cast_dtypes[i][0][1] = &NullableDType;
        cast_specs[i][0] = (PyArrayMethod_Spec){
            .name = "to_nullable",
            .nin = 1, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_SAFE_CASTING,
            .dtypes = cast_dtypes[i][0],
            .slots = to_nullable_slots,
        };
        all_casts[n++] = &cast_specs[i][0];

        cast_dtypes[i][1][0] = &NullableDType;
        cast_dtypes[i][1][1] = wrapped_dt;
        cast_specs[i][1] = (PyArrayMethod_Spec){
            .name = "from_nullable",
            .nin = 1, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_UNSAFE_CASTING,
            .dtypes = cast_dtypes[i][1],
            .slots = from_nullable_slots,
        };
        all_casts[n++] = &cast_specs[i][1];
    }
    all_casts[n] = NULL;
    return 0;
}


/* --------------------------------------------------------- binary loops */
/*
 * The proof of the design: we do not reimplement any arithmetic.  The value
 * fields of an interleaved Nullable[T] array form a strided array of T, so we
 * hand them to T's own machinery as zero-copy views and only merge the
 * validity bytes ourselves.  One implementation serves every binary ufunc.
 */

#define N_BINOPS 26
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
    /*
     * Deliberately absent: `bitwise_and` / `bitwise_or` / `bitwise_xor`.
     * On Nullable[bool] those have to follow Kleene (`NA & False = False`),
     * on Nullable[int] they have to propagate.  One loop cannot do both
     * without branching on the wrapped kind, and shipping the wrong truth
     * table quietly is exactly what this prototype exists to avoid.
     */
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
 * and expanded wherever a table is built -- for both layouts.  Adding an op
 * means adding a name above and one `X()` below; the assert catches a mismatch
 * at compile time instead of at import.
 */
#define BINOP_INDICES(X) \
    X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) \
    X(10) X(11) X(12) X(13) X(14) X(15) X(16) X(17) X(18) X(19) \
    X(20) X(21) X(22) X(23) X(24) X(25)

#define COUNT_ONE(I) + 1
_Static_assert(0 BINOP_INDICES(COUNT_ONE) == N_BINOPS,
               "BINOP_INDICES is out of step with binop_names");


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


/* zero-copy 1-D bool view over the validity bytes of one operand */
static PyObject *
valid_view(char *data, npy_intp value_size, npy_intp n, npy_intp stride)
{
    npy_intp dims[1] = {n};
    npy_intp strides[1] = {stride};
    PyArray_Descr *b = PyArray_DescrFromType(NPY_BOOL);
    if (b == NULL) {
        return NULL;
    }
    return PyArray_NewFromDescr(&PyArray_Type, b, 1, dims, strides,
                                data + value_size, NPY_ARRAY_WRITEABLE, NULL);
}


static PyArray_Descr *
unwrap(PyArray_Descr *descr)
{
    if (Py_TYPE(descr) == (PyTypeObject *)&NullableDType) {
        return NULLABLE_DESCR(descr)->wrapped;
    }
    return descr;   /* a promoter sent us a plain operand */
}


/*
 * Ask the wrapped ufunc what dtypes it would use, then wrap its answer.  That
 * is how `Nullable[i8] / Nullable[i8]` correctly becomes `Nullable[f8]`.
 */
static NPY_CASTING
nullable_binary_resolve_impl(PyObject *ufunc,
        PyArray_Descr *const given_descrs[3], PyArray_Descr *loop_descrs[3])
{
    /*
     * Refuse rather than let `==` come out all False (see `binop_rich_op`).
     * Only the bitpattern layout compares records; this one is on its way out.
     */
    if (binop_rich_op(ufunc) >= 0
            && (PyDataType_HASFIELDS(unwrap((PyArray_Descr *)given_descrs[0]))
                || PyDataType_HASFIELDS(unwrap((PyArray_Descr *)given_descrs[1])))) {
        PyErr_SetString(PyExc_TypeError,
                "FlagLayout cannot compare records; use Nullable(), which "
                "stores them as bit patterns");
        return (NPY_CASTING)-1;
    }
    PyObject *query = Py_BuildValue("(OOO)",
            (PyObject *)unwrap((PyArray_Descr *)given_descrs[0]),
            (PyObject *)unwrap((PyArray_Descr *)given_descrs[1]),
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
        if (i < 2 && unwrap((PyArray_Descr *)given_descrs[i]) != value_descr) {
            casting = NPY_SAFE_CASTING;   /* NumPy will cast the operand */
        }
    }
    Py_DECREF(resolved);
    return casting;
}


/*
 * One chunk of `ufunc.reduce`.  NA propagates: a missing element anywhere
 * makes the whole reduction missing, which is R's default (`na.rm = FALSE`).
 * To skip instead, reduce over `where=notna(a)` with an explicit `initial`.
 */
static int
nullable_reduce_chunk(int idx, PyArrayMethod_Context *context,
        char *const data[], npy_intp N, npy_intp const strides[])
{
    PyArray_Descr *acc_descr = NULLABLE_DESCR(context->descriptors[0])->wrapped;
    PyArray_Descr *in_descr = NULLABLE_DESCR(context->descriptors[1])->wrapped;
    npy_intp va = acc_descr->elsize, vi = in_descr->elsize;

    int missing = !data[0][va];
    for (npy_intp i = 0; i < N && !missing; i++) {
        missing = !data[1][i * strides[1] + vi];
    }
    if (missing) {
        memset(data[2], 0, (size_t)va);
        data[2][va] = 0;
        return 0;
    }

    PyObject *values = value_view(in_descr, data[1], N, strides[1]);
    PyObject *initial = PyArray_Scalar(data[0], acc_descr, NULL);
    if (values == NULL || initial == NULL) {
        Py_XDECREF(values); Py_XDECREF(initial);
        return -1;
    }
    PyObject *reduce = PyObject_GetAttrString(binop_ufuncs[idx], "reduce");
    PyObject *args = PyTuple_Pack(1, values);
    PyObject *kwargs = Py_BuildValue("{s:O}", "initial", initial);
    PyObject *res = (reduce && args && kwargs)
            ? PyObject_Call(reduce, args, kwargs) : NULL;
    Py_XDECREF(reduce); Py_XDECREF(args); Py_XDECREF(kwargs);
    Py_DECREF(values);
    Py_DECREF(initial);
    if (res == NULL) {
        return -1;
    }

    PyObject *slot = value_view(acc_descr, data[2], 1, 0);
    PyObject *done = (slot != NULL)
            ? PyObject_CallFunction(np_copyto, "OO", slot, res) : NULL;
    Py_XDECREF(slot);
    Py_DECREF(res);
    if (done == NULL) {
        return -1;
    }
    Py_DECREF(done);
    data[2][va] = 1;
    return 0;
}


/*
 * One chunk of `ufunc.accumulate`.  Every element after the first missing one
 * is missing too, so we only accumulate up to that point and never compute on
 * a value that does not exist.
 */
static int
nullable_accumulate_chunk(int idx, PyArrayMethod_Context *context,
        char *const data[], npy_intp N, npy_intp const strides[])
{
    PyArray_Descr *acc_descr = NULLABLE_DESCR(context->descriptors[0])->wrapped;
    PyArray_Descr *in_descr = NULLABLE_DESCR(context->descriptors[1])->wrapped;
    PyArray_Descr *out_descr = NULLABLE_DESCR(context->descriptors[2])->wrapped;
    npy_intp va = acc_descr->elsize, vi = in_descr->elsize;
    npy_intp vo = out_descr->elsize;

    /* how far the running value stays known */
    npy_intp known = 0;
    if (data[0][va]) {
        while (known < N && data[1][known * strides[1] + vi]) {
            known++;
        }
    }

    /* everything past that is missing */
    for (npy_intp i = known; i < N; i++) {
        memset(data[2] + i * strides[2], 0, (size_t)vo);
        data[2][i * strides[2] + vo] = 0;
    }
    if (known == 0) {
        return 0;
    }

    /*
     * `[acc, x0, ..., xk-1]` accumulated gives `[acc, acc op x0, ...]`, so the
     * tail of that is exactly what the output wants.
     */
    PyObject *acc = PyArray_Scalar(data[0], acc_descr, NULL);
    PyObject *head = (acc != NULL) ? PyObject_CallFunction(
            np_asarray, "([O])", acc) : NULL;
    PyObject *tail = value_view(in_descr, data[1], known, strides[1]);
    Py_XDECREF(acc);
    if (head == NULL || tail == NULL) {
        Py_XDECREF(head); Py_XDECREF(tail);
        return -1;
    }
    PyObject *joined = PyObject_CallFunction(np_concatenate, "((OO))",
                                             head, tail);
    Py_DECREF(head); Py_DECREF(tail);
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

    PyObject *out = value_view(out_descr, data[2], known, strides[2]);
    PyObject *done = (out != NULL) ? PyObject_CallFunction(
            np_copyto, "OO", out, without_head) : NULL;
    Py_XDECREF(out);
    Py_DECREF(without_head);
    if (done == NULL) {
        return -1;
    }
    Py_DECREF(done);

    for (npy_intp i = 0; i < known; i++) {
        data[2][i * strides[2] + vo] = 1;
    }
    return 0;
}


static int
nullable_binary_loop_impl(int idx, PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[])
{
    npy_intp N = dimensions[0];
    PyArray_Descr *in0 = NULLABLE_DESCR(context->descriptors[0])->wrapped;
    PyArray_Descr *in1 = NULLABLE_DESCR(context->descriptors[1])->wrapped;
    PyArray_Descr *out = NULLABLE_DESCR(context->descriptors[2])->wrapped;

    /*
     * A reduction calls us with the accumulator as both first input and
     * output, at stride 0.  Element-wise code would only keep the last write,
     * so hand the whole chunk to `ufunc.reduce` instead.
     */
    if (strides[0] == 0 && data[0] == data[2]) {
        return nullable_reduce_chunk(idx, context, data, N, strides);
    }
    /* `accumulate` feeds each output back in as the next input */
    if (strides[0] == strides[2] && data[0] == data[2] - strides[2]) {
        return nullable_accumulate_chunk(idx, context, data, N, strides);
    }

    /* 1. let T compute the values, through its own ufunc loop */
    PyObject *a = value_view(in0, data[0], N, strides[0]);
    PyObject *b = value_view(in1, data[1], N, strides[1]);
    PyObject *o = value_view(out, data[2], N, strides[2]);
    if (a == NULL || b == NULL || o == NULL) {
        Py_XDECREF(a); Py_XDECREF(b); Py_XDECREF(o);
        return -1;
    }
    /*
     * `where=` keeps the loop away from the missing entries, so a value that
     * does not exist can never raise a floating point warning of its own.
     */
    /*
     * Build the `where=` mask straight into a fresh bool array.  Doing the AND
     * here instead of with a second ufunc call saves a dispatch and a
     * temporary per chunk.
     */
    npy_intp v0 = in0->elsize, v1 = in1->elsize, vo = out->elsize;
    npy_intp dims[1] = {N};
    PyObject *both = PyArray_SimpleNew(1, dims, NPY_BOOL);
    if (both == NULL) {
        Py_DECREF(a); Py_DECREF(b); Py_DECREF(o);
        return -1;
    }
    char *mask = PyArray_DATA((PyArrayObject *)both);
    for (npy_intp i = 0; i < N; i++) {
        mask[i] = data[0][i * strides[0] + v0] & data[1][i * strides[1] + v1];
    }

    PyObject *args = PyTuple_Pack(2, a, b);
    PyObject *kwargs = Py_BuildValue("{s:O,s:O}", "out", o, "where", both);
    PyObject *res = (args && kwargs)
            ? PyObject_Call(binop_ufuncs[idx], args, kwargs) : NULL;
    Py_XDECREF(args); Py_XDECREF(kwargs);
    Py_DECREF(a); Py_DECREF(b); Py_DECREF(o);
    if (res == NULL) {
        Py_DECREF(both);
        return -1;
    }
    Py_DECREF(res);

    /* 2. carry the validity over, and keep missing values unobservable */
    for (npy_intp i = 0; i < N; i++) {
        data[2][i * strides[2] + vo] = mask[i];
        if (!mask[i]) {
            memset(data[2] + i * strides[2], 0, (size_t)vo);
        }
    }
    Py_DECREF(both);
    return 0;
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

    PyArray_Descr *wa = NULLABLE_DESCR(context->descriptors[0])->wrapped;
    PyArray_Descr *wb = NULLABLE_DESCR(context->descriptors[1])->wrapped;
    PyArray_Descr *wc = NULLABLE_DESCR(context->descriptors[2])->wrapped;
    npy_intp va = wa->elsize, vb = wb->elsize, vc = wc->elsize;

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

        if (matmul_values(wa, wb, wc, pa, pb, pc, m, n, pdim,
                          a_m, a_n, b_n, b_p, c_m, c_p) < 0) {
            PyMem_Free(rowna); PyMem_Free(colna);
            return -1;
        }

        for (npy_intp i = 0; i < m; i++) {
            char bad = 0;
            for (npy_intp k = 0; k < n; k++) {
                bad |= !pa[i * a_m + k * a_n + va];
            }
            rowna[i] = bad;
        }
        for (npy_intp j = 0; j < pdim; j++) {
            char bad = 0;
            for (npy_intp k = 0; k < n; k++) {
                bad |= !pb[k * b_n + j * b_p + vb];
            }
            colna[j] = bad;
        }
        for (npy_intp i = 0; i < m; i++) {
            for (npy_intp j = 0; j < pdim; j++) {
                char *slot = pc + i * c_m + j * c_p;
                if (rowna[i] || colna[j]) {
                    memset(slot, 0, (size_t)vc);
                    slot[vc] = 0;
                }
                else {
                    slot[vc] = 1;
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
/*
 * `clip` is the one three-input ufunc worth having.  It works exactly like the
 * binary ops -- hand the value fields to numpy's own `clip` and merge the
 * validity ourselves -- but with three operands instead of two, so the
 * two-in/one-out machinery above cannot be reused as is.
 *
 * A bound that is missing makes the answer missing: "keep this between 1 and
 * something I do not know" has no answer.
 */
static PyObject *clip_ufunc = NULL;

static NPY_CASTING
nullable_clip_resolve(struct PyArrayMethodObject_tag *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[4]),
        PyArray_Descr *const given_descrs[4], PyArray_Descr *loop_descrs[4],
        npy_intp *NPY_UNUSED(view_offset))
{
    PyObject *query = Py_BuildValue("(OOOO)",
            (PyObject *)unwrap((PyArray_Descr *)given_descrs[0]),
            (PyObject *)unwrap((PyArray_Descr *)given_descrs[1]),
            (PyObject *)unwrap((PyArray_Descr *)given_descrs[2]),
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
        if (i < 3 && unwrap((PyArray_Descr *)given_descrs[i]) != value_descr) {
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
    PyArray_Descr *w[4];
    for (int i = 0; i < 4; i++) {
        w[i] = NULLABLE_DESCR(context->descriptors[i])->wrapped;
    }

    PyObject *v[4] = {NULL, NULL, NULL, NULL};
    for (int i = 0; i < 4; i++) {
        v[i] = value_view(w[i], data[i], N, strides[i]);
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
        m[i] = data[0][i * strides[0] + w[0]->elsize]
             & data[1][i * strides[1] + w[1]->elsize]
             & data[2][i * strides[2] + w[2]->elsize];
    }

    PyObject *args = PyTuple_Pack(3, v[0], v[1], v[2]);
    PyObject *kwargs = Py_BuildValue("{s:O,s:O}", "out", v[3], "where", mask);
    PyObject *res = (args && kwargs)
            ? PyObject_Call(clip_ufunc, args, kwargs) : NULL;
    Py_XDECREF(args); Py_XDECREF(kwargs); Py_DECREF(mask);
    if (res == NULL) {
        goto fail;
    }
    Py_DECREF(res);

    npy_intp vo = w[3]->elsize;
    for (npy_intp i = 0; i < N; i++) {
        data[3][i * strides[3] + vo] = m[i];
        if (!m[i]) {
            memset(data[3] + i * strides[3], 0, (size_t)vo);
        }
    }
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


/* ---------------------------------------------------------- unary loops */
/*
 * Same idea as the binary loops: hand the value fields to T's own ufunc with
 * `where=` so missing entries are never touched, then carry the validity over
 * unchanged.  A unary op cannot create or remove missingness.
 */

#define N_UNOPS 39
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
};
static PyObject *unop_ufuncs[N_UNOPS];

#define UNOP_INDICES(X) \
    X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) \
    X(10) X(11) X(12) X(13) X(14) X(15) X(16) X(17) X(18) X(19) \
    X(20) X(21) X(22) X(23) X(24) X(25) X(26) X(27) X(28) X(29) \
    X(30) X(31) X(32) X(33) X(34) X(35) X(36) X(37) X(38)

_Static_assert(0 UNOP_INDICES(COUNT_ONE) == N_UNOPS,
               "UNOP_INDICES is out of step with unop_names");


static NPY_CASTING
nullable_unary_resolve_impl(int idx,
        PyArray_Descr *const given_descrs[2], PyArray_Descr *loop_descrs[2])
{
    PyObject *query = Py_BuildValue("(OO)",
            (PyObject *)unwrap((PyArray_Descr *)given_descrs[0]), Py_None);
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
        if (i == 0 && unwrap((PyArray_Descr *)given_descrs[0]) != value_descr) {
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
    PyArray_Descr *in = NULLABLE_DESCR(context->descriptors[0])->wrapped;
    PyArray_Descr *out = NULLABLE_DESCR(context->descriptors[1])->wrapped;
    npy_intp vi = in->elsize, vo = out->elsize;

    PyObject *a = value_view(in, data[0], N, strides[0]);
    PyObject *o = value_view(out, data[1], N, strides[1]);
    PyObject *m = valid_view(data[0], vi, N, strides[0]);
    if (a == NULL || o == NULL || m == NULL) {
        Py_XDECREF(a); Py_XDECREF(o); Py_XDECREF(m);
        return -1;
    }
    PyObject *args = PyTuple_Pack(1, a);
    PyObject *kwargs = Py_BuildValue("{s:O,s:O}", "out", o, "where", m);
    PyObject *res = (args && kwargs)
            ? PyObject_Call(unop_ufuncs[idx], args, kwargs) : NULL;
    Py_XDECREF(args); Py_XDECREF(kwargs);
    Py_DECREF(a); Py_DECREF(o); Py_DECREF(m);
    if (res == NULL) {
        return -1;
    }
    Py_DECREF(res);

    for (npy_intp i = 0; i < N; i++) {
        char valid = data[0][i * strides[0] + vi];
        data[1][i * strides[1] + vo] = valid;
        if (!valid) {
            memset(data[1] + i * strides[1], 0, (size_t)vo);
        }
    }
    return 0;
}


#define NULLABLE_UNOP(IDX)                                                    \
static NPY_CASTING                                                            \
nullable_uresolve_##IDX(struct PyArrayMethodObject_tag *NPY_UNUSED(self),     \
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
    {NPY_METH_resolve_descriptors, &nullable_uresolve_##IDX},                 \
    {NPY_METH_strided_loop, &nullable_uloop_##IDX},                           \
    {NPY_METH_unaligned_strided_loop, &nullable_uloop_##IDX},                 \
    {0, NULL}                                                                 \
};

UNOP_INDICES(NULLABLE_UNOP)

#define NULLABLE_USLOTS_PTR(I) nullable_uslots_##I,
static PyType_Slot *unop_slots[N_UNOPS] = { UNOP_INDICES(NULLABLE_USLOTS_PTR) };
#undef NULLABLE_USLOTS_PTR

static PyArray_DTypeMeta *unop_dtypes[2] = {NULL, NULL};
static PyArrayMethod_Spec unop_specs[N_UNOPS];


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


/*
 * `np.all` and `np.any` are `logical_and.reduce` and `logical_or.reduce`, so
 * the Kleene rules have to hold while folding as well: one known False settles
 * `all`, one known True settles `any`, and only then does a gap win.
 */
static int
kleene_reduce_chunk(int op, PyArrayMethod_Context *context,
        char *const data[], npy_intp N, npy_intp const strides[])
{
    PyArray_Descr *in_descr = NULLABLE_DESCR(context->descriptors[1])->wrapped;
    npy_intp vi = in_descr->elsize;

    PyArrayObject *tb = truthiness(in_descr, data[1], N, strides[1]);
    if (tb == NULL) {
        return -1;
    }
    const char *vals = PyArray_DATA(tb);

    char value = data[0][0], valid = data[0][1];
    char settles = (op == KLEENE_AND) ? 0 : 1;   /* False settles AND */

    for (npy_intp i = 0; i < N; i++) {
        char b_valid = data[1][i * strides[1] + vi];
        char b = vals[i];

        if (op == KLEENE_XOR) {
            valid = valid && b_valid;
            value = valid ? (value != b) : 0;
            continue;
        }
        /*
         * Once a settling value has been seen the answer is fixed and no
         * later gap can unfix it: `any([True, NA, False])` is True, because
         * whatever the missing element turns out to be, a True is already
         * there.  Without this the clause below wiped out a decision that had
         * already been made, and the result came out order-dependent.
         */
        if (valid && value == settles) {
            break;
        }
        if (b_valid && b == settles) {
            value = settles;                     /* decided, gaps no longer matter */
            valid = 1;
            break;
        }
        if (!b_valid) {
            valid = 0;                           /* unknown so far */
        }
        else if (valid) {
            value = (op == KLEENE_AND) ? (value && b) : (value || b);
        }
    }
    Py_DECREF(tb);

    data[2][0] = valid ? value : 0;
    data[2][1] = valid;
    return 0;
}


static int
kleene_loop_impl(int op, PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[])
{
    npy_intp N = dimensions[0];

    if (strides[0] == 0 && data[0] == data[2]) {
        return kleene_reduce_chunk(op, context, data, N, strides);
    }
    PyArray_Descr *in0 = NULLABLE_DESCR(context->descriptors[0])->wrapped;
    PyArray_Descr *in1 = NULLABLE_DESCR(context->descriptors[1])->wrapped;
    npy_intp v0 = in0->elsize, v1 = in1->elsize;

    PyArrayObject *ta = truthiness(in0, data[0], N, strides[0]);
    PyArrayObject *tb = truthiness(in1, data[1], N, strides[1]);
    if (ta == NULL || tb == NULL) {
        Py_XDECREF(ta); Py_XDECREF(tb);
        return -1;
    }
    const char *a_vals = PyArray_DATA(ta);
    const char *b_vals = PyArray_DATA(tb);

    for (npy_intp i = 0; i < N; i++) {
        char a_valid = data[0][i * strides[0] + v0];
        char b_valid = data[1][i * strides[1] + v1];
        char a = a_vals[i], b = b_vals[i];
        char value = 0, valid = 0;

        switch (op) {
            case KLEENE_AND:
                /* one known False settles it, whatever the other side is */
                if ((a_valid && !a) || (b_valid && !b)) {
                    value = 0;
                    valid = 1;
                }
                else if (a_valid && b_valid) {
                    value = 1;
                    valid = 1;
                }
                break;
            case KLEENE_OR:
                /* one known True settles it */
                if ((a_valid && a) || (b_valid && b)) {
                    value = 1;
                    valid = 1;
                }
                else if (a_valid && b_valid) {
                    value = 0;
                    valid = 1;
                }
                break;
            default:  /* KLEENE_XOR */
                valid = a_valid && b_valid;
                value = valid ? (a != b) : 0;
                break;
        }
        data[2][i * strides[2]] = value;
        data[2][i * strides[2] + 1] = valid;
    }

    Py_DECREF(ta); Py_DECREF(tb);
    return 0;
}


static NPY_CASTING
kleene_resolve_impl(int op,
        PyArray_Descr *const given_descrs[3], PyArray_Descr *loop_descrs[3])
{
    /* inputs keep their value dtype, the answer is always a nullable bool */
    PyArray_Descr *bool_descr = PyArray_DescrFromType(NPY_BOOL);
    if (bool_descr == NULL) {
        return (NPY_CASTING)-1;
    }
    for (int i = 0; i < 2; i++) {
        PyArray_Descr *value_descr = unwrap((PyArray_Descr *)given_descrs[i]);
        loop_descrs[i] = new_nullable_descr(value_descr);
        if (loop_descrs[i] == NULL) {
            Py_DECREF(bool_descr);
            for (int j = 0; j < i; j++) {
                Py_CLEAR(loop_descrs[j]);
            }
            return (NPY_CASTING)-1;
        }
    }
    loop_descrs[2] = new_nullable_descr(bool_descr);
    Py_DECREF(bool_descr);
    if (loop_descrs[2] == NULL) {
        Py_CLEAR(loop_descrs[0]);
        Py_CLEAR(loop_descrs[1]);
        return (NPY_CASTING)-1;
    }
    (void)op;
    return NPY_SAFE_CASTING;
}


/* `all` starts from True, `any` and `xor` from False */
static int
kleene_initial_impl(int op, PyArrayMethod_Context *NPY_UNUSED(context),
        npy_bool NPY_UNUSED(reduction_is_empty), void *initial)
{
    ((char *)initial)[0] = (op == KLEENE_AND);
    ((char *)initial)[1] = 1;
    return 1;
}


#define NULLABLE_KLEENE(OP)                                                   \
static NPY_CASTING                                                            \
kleene_resolve_##OP(struct PyArrayMethodObject_tag *NPY_UNUSED(self),         \
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),                       \
        PyArray_Descr *const given_descrs[3], PyArray_Descr *loop_descrs[3],  \
        npy_intp *NPY_UNUSED(view_offset))                                    \
{                                                                             \
    return kleene_resolve_impl(OP, given_descrs, loop_descrs);                \
}                                                                             \
static int                                                                    \
kleene_loop_##OP(PyArrayMethod_Context *context, char *const data[],          \
        npy_intp const dimensions[], npy_intp const strides[],                \
        NpyAuxData *NPY_UNUSED(auxdata))                                      \
{                                                                             \
    return kleene_loop_impl(OP, context, data, dimensions, strides);          \
}                                                                             \
static int                                                                    \
kleene_initial_##OP(PyArrayMethod_Context *context,                           \
        npy_bool reduction_is_empty, void *initial)                           \
{                                                                             \
    return kleene_initial_impl(OP, context, reduction_is_empty, initial);     \
}                                                                             \
static PyType_Slot kleene_slots_##OP[] = {                                    \
    {NPY_METH_resolve_descriptors, &kleene_resolve_##OP},                     \
    {NPY_METH_strided_loop, &kleene_loop_##OP},                               \
    {NPY_METH_unaligned_strided_loop, &kleene_loop_##OP},                     \
    {NPY_METH_get_reduction_initial, &kleene_initial_##OP},                   \
    {0, NULL}                                                                 \
};

NULLABLE_KLEENE(0)
NULLABLE_KLEENE(1)
NULLABLE_KLEENE(2)

static PyType_Slot *kleene_slots[3] = {
    kleene_slots_0, kleene_slots_1, kleene_slots_2};
static PyArrayMethod_Spec kleene_specs[3];


/*
 * The identity of a reduction, e.g. 0 for `add`.  Without it NumPy refuses
 * `reduce(..., where=...)` unless the caller also passes `initial=`, which is
 * exactly the "skip missing values" call we want to be easy.
 */
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
        return 0;               /* `maximum` and friends have none */
    }
    PyArray_Descr *value_descr =
            NULLABLE_DESCR(context->descriptors[0])->wrapped;
    PyObject *slot = value_view(value_descr, (char *)initial, 1, 0);
    PyObject *done = (slot != NULL)
            ? PyObject_CallFunction(np_copyto, "OO", slot, identity) : NULL;
    Py_XDECREF(slot);
    Py_DECREF(identity);
    if (done == NULL) {
        return -1;
    }
    Py_DECREF(done);
    ((char *)initial)[value_descr->elsize] = 1;   /* the identity is a value */
    return 1;
}


/* ------------------------------------------------- the fast path: no Python */
/*
 * `NPY_METH_get_loop` runs once per operation, not once per chunk.  That is
 * where we ask the wrapped ufunc for its C inner loop, through the documented
 * `numpy_1.24_ufunc_call_info` capsule, and keep it for the whole call.  The
 * chunk loop below then touches no Python at all: it walks runs of present
 * values and hands each run straight to T's own loop.
 */

/* first fields of NumPy's `ufunc_call_info`, see umath/ufunc_object.c */
typedef struct {
    PyArrayMethod_StridedLoop *strided_loop;
    PyArrayMethod_Context *context;
    NpyAuxData *auxdata;
    npy_bool requires_pyapi;
    npy_bool no_floatingpoint_errors;
} nullable_call_info;


typedef struct {
    NpyAuxData base;
    int idx;
    PyObject *capsule;                  /* owns everything below */
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

    /* reductions and accumulations keep their own paths */
    if ((strides[0] == 0 && data[0] == data[2])
            || (strides[0] == strides[2] && data[0] == data[2] - strides[2])) {
        return nullable_binary_loop_impl(aux->idx, context, data,
                                         dimensions, strides);
    }

    npy_intp v0 = NULLABLE_DESCR(context->descriptors[0])->wrapped->elsize;
    npy_intp v1 = NULLABLE_DESCR(context->descriptors[1])->wrapped->elsize;
    npy_intp vo = NULLABLE_DESCR(context->descriptors[2])->wrapped->elsize;

    npy_intp i = 0;
    while (i < N) {
        /* gaps: nothing to compute, and nothing observable to leave behind */
        while (i < N && !(data[0][i * strides[0] + v0]
                          && data[1][i * strides[1] + v1])) {
            memset(data[2] + i * strides[2], 0, (size_t)vo);
            data[2][i * strides[2] + vo] = 0;
            i++;
        }
        npy_intp start = i;
        while (i < N && data[0][i * strides[0] + v0]
                     && data[1][i * strides[1] + v1]) {
            i++;
        }
        npy_intp run = i - start;
        if (run == 0) {
            continue;
        }
        char *args[3] = {
            data[0] + start * strides[0],
            data[1] + start * strides[1],
            data[2] + start * strides[2],
        };
        if (aux->inner(aux->inner_context, args, &run,
                       strides, aux->inner_auxdata) < 0) {
            return -1;
        }
        for (npy_intp j = start; j < i; j++) {
            data[2][j * strides[2] + vo] = 1;
        }
    }
    return 0;
}


/*
 * Ops whose numpy loop takes the broken SIMD stride path (see the long note at
 * `nullable_get_loop_impl`).  Matched by name rather than by index so that
 * reordering `binop_names` cannot silently un-protect one of them.
 */
/*
 * Ops whose scalar kernel compares its operands with `<` or `>`.  Those are
 * *signaling* predicates: IEEE 754 makes them raise FE_INVALID even on a quiet
 * NaN, where `==` and plain arithmetic stay silent.  Sentinel keeps a NaN in
 * every gap and normally lets the wrapped loop run straight over it, so for
 * these ops a gap would report "invalid value encountered" -- an arithmetic
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
 * Ops that return the non-NaN operand instead of propagating.  Sentinel stores
 * NA as a NaN, so for these a gap vanishes from the output and the layout
 * cannot recover it from the result alone.
 */
static int
binop_swallows_nan(int idx)
{
    const char *n = binop_names[idx];
    return strcmp(n, "fmax") == 0 || strcmp(n, "fmin") == 0;
}

static int
binop_needs_whole_element_strides(int idx)
{
    const char *n = binop_names[idx];
    return strcmp(n, "maximum") == 0 || strcmp(n, "minimum") == 0
            || strcmp(n, "fmax") == 0 || strcmp(n, "fmin") == 0;
}


static int
nullable_get_loop_impl(int idx, PyArrayMethod_Context *context,
        int NPY_UNUSED(aligned), int NPY_UNUSED(move_references),
        const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_auxdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    *flags = NPY_METH_REQUIRES_PYAPI;

    /*
     * numpy's `maximum`/`minimum` loops take a SIMD path that turns the byte
     * stride into an element stride with `stride / sizeof(T)`.  The guard that
     * is supposed to reject a stride which is not a multiple of `sizeof(T)`
     * only runs when `alignof(T) != sizeof(T)`, which is false for every type
     * we wrap on this platform, so the division silently truncates (9 / 8 -> 1)
     * and the loop walks the array as if it were contiguous.  numpy never trips
     * over this itself because it buffers into contiguous memory first; we call
     * the loop directly with our own strides, so we must check ourselves.
     * See numpy/_core/src/common/simd/simd.h, NPYV_IMPL_MAXSTRIDE.
     */
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
    if (resolved == NULL || !PyTuple_Check(resolved) ||
            PyTuple_GET_SIZE(resolved) != 2) {
        /* fall back to the loop that goes through Python */
        Py_XDECREF(resolved);
        PyErr_Clear();
        *out_loop = &nullable_fast_loop;   /* replaced below */
        goto fallback;
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
        goto fallback;
    }
    Py_DECREF(res);

    nullable_call_info *info = PyCapsule_GetPointer(
            capsule, "numpy_1.24_ufunc_call_info");
    if (info == NULL || info->strided_loop == NULL) {
        Py_DECREF(capsule);
        PyErr_Clear();
        goto fallback;
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

  fallback:
    *out_auxdata = NULL;
    return 1;      /* caller substitutes the Python-based loop */
}


/* one thin pair of functions per ufunc, all sharing the code above */
#define NULLABLE_BINOP(IDX)                                                   \
static NPY_CASTING                                                            \
nullable_resolve_##IDX(struct PyArrayMethodObject_tag *NPY_UNUSED(self),      \
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),                       \
        PyArray_Descr *const given_descrs[3], PyArray_Descr *loop_descrs[3],  \
        npy_intp *NPY_UNUSED(view_offset))                                    \
{                                                                             \
    return nullable_binary_resolve_impl(binop_ufuncs[IDX], given_descrs,      \
                                        loop_descrs);                         \
}                                                                             \
static int                                                                    \
nullable_loop_##IDX(PyArrayMethod_Context *context, char *const data[],       \
        npy_intp const dimensions[], npy_intp const strides[],                \
        NpyAuxData *NPY_UNUSED(auxdata))                                      \
{                                                                             \
    return nullable_binary_loop_impl(IDX, context, data, dimensions, strides);\
}                                                                             \
static int                                                                    \
nullable_initial_##IDX(PyArrayMethod_Context *context,                        \
        npy_bool reduction_is_empty, void *initial)                           \
{                                                                             \
    return nullable_reduction_initial_impl(IDX, context,                      \
                                           reduction_is_empty, initial);      \
}                                                                             \
static int                                                                    \
nullable_get_loop_##IDX(PyArrayMethod_Context *context, int aligned,          \
        int move_references, const npy_intp *strides,                         \
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_auxdata,       \
        NPY_ARRAYMETHOD_FLAGS *flags)                                         \
{                                                                             \
    int res = nullable_get_loop_impl(IDX, context, aligned, move_references,  \
                                     strides, out_loop, out_auxdata, flags);  \
    if (res == 1) {                     /* no C loop to borrow */             \
        *out_loop = &nullable_loop_##IDX;                                     \
        return 0;                                                             \
    }                                                                         \
    return res;                                                               \
}                                                                             \
static PyType_Slot nullable_slots_##IDX[] = {                                 \
    {NPY_METH_resolve_descriptors, &nullable_resolve_##IDX},                  \
    {NPY_METH_get_loop, &nullable_get_loop_##IDX},                            \
    {NPY_METH_get_reduction_initial, &nullable_initial_##IDX},                \
    {0, NULL}                                                                 \
};

BINOP_INDICES(NULLABLE_BINOP)

#define NULLABLE_SLOTS_PTR(I) nullable_slots_##I,
static PyType_Slot *binop_slots[N_BINOPS] = { BINOP_INDICES(NULLABLE_SLOTS_PTR) };
#undef NULLABLE_SLOTS_PTR

static PyArray_DTypeMeta *binop_dtypes[3] = {NULL, NULL, NULL};
static PyArrayMethod_Spec binop_specs[N_BINOPS];


/*
 * `Nullable[T] op T` has no loop of its own; the promoter tells NumPy to
 * retry with every operand seen as Nullable, and the safe `T -> Nullable[T]`
 * cast then does the rest.
 */
/*
 * Which layout the result should use: whichever one the operands already have.
 *
 * Both layouts share this promoter, so hardcoding one of them meant
 * `Sentinel[f8] + 1.0` came back in the flag layout -- right answer, wrong
 * (bigger, slower) dtype, and silent because both print as `Nullable(...)`.
 */
static PyArray_DTypeMeta *
layout_of(PyArray_DTypeMeta *const op_dtypes[], int n)
{
    for (int i = 0; i < n; i++) {
        if (op_dtypes[i] == &SentinelDType || op_dtypes[i] == &NullableDType) {
            return op_dtypes[i];
        }
    }
    return &NullableDType;
}


static int
nullable_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[3],
        PyArray_DTypeMeta *const signature[3],
        PyArray_DTypeMeta *new_op_dtypes[3])
{
    PyArray_DTypeMeta *layout = layout_of(op_dtypes, 3);
    for (int i = 0; i < 3; i++) {
        PyArray_DTypeMeta *new = layout;
        if (signature[i] != NULL) {
            new = signature[i];
        }
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
        PyArray_DTypeMeta *const op_dtypes[4],
        PyArray_DTypeMeta *const signature[4],
        PyArray_DTypeMeta *new_op_dtypes[4])
{
    PyArray_DTypeMeta *layout = layout_of(op_dtypes, 4);
    for (int i = 0; i < 4; i++) {
        PyArray_DTypeMeta *new = layout;
        if (signature[i] != NULL) {
            new = signature[i];
        }
        Py_INCREF(new);
        new_op_dtypes[i] = new;
    }
    return 0;
}


static int
register_clip_promoter(PyArray_DTypeMeta *layout)
{
    PyObject *capsule = PyCapsule_New(
            &nullable_promoter4, "numpy._ufunc_promoter", NULL);
    if (capsule == NULL) {
        return -1;
    }
    PyObject *dtypes = PyTuple_Pack(4, (PyObject *)layout,
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


/* register every binary ufunc, plus the promoters that feed it */
static int
init_binops(void)
{
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) {
        return -1;
    }
    np_copyto = PyObject_GetAttrString(numpy, "copyto");
    np_asarray = PyObject_GetAttrString(numpy, "asarray");
    np_concatenate = PyObject_GetAttrString(numpy, "concatenate");
    if (np_copyto == NULL || np_asarray == NULL || np_concatenate == NULL) {
        Py_DECREF(numpy);
        return -1;
    }

    PyArray_DTypeMeta *pyscalars[3] = {
        &PyArray_PyLongDType, &PyArray_PyFloatDType, &PyArray_PyComplexDType};

    binop_dtypes[0] = &NullableDType;
    binop_dtypes[1] = &NullableDType;
    binop_dtypes[2] = &NullableDType;

    for (int k = 0; k < N_BINOPS; k++) {
        binop_ufuncs[k] = PyObject_GetAttrString(numpy, binop_names[k]);
        if (binop_ufuncs[k] == NULL) {
            Py_DECREF(numpy);
            return -1;
        }
        binop_specs[k] = (PyArrayMethod_Spec){
            .name = binop_names[k],
            .nin = 2, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_NO_CASTING,
            .dtypes = binop_dtypes,
            .slots = binop_slots[k],
        };
        if (PyUFunc_AddLoopFromSpec(binop_ufuncs[k], &binop_specs[k]) < 0) {
            Py_DECREF(numpy);
            return -1;
        }
        for (int i = 0; i < N_WRAPPABLE; i++) {
            PyArray_DTypeMeta *plain =
                    dtypemeta_from_typenum(wrappable_typenums[i]);
            if (plain == NULL
                    || register_promoter(binop_ufuncs[k], &NullableDType, plain) < 0
                    || register_promoter(binop_ufuncs[k], plain, &NullableDType) < 0) {
                Py_DECREF(numpy);
                return -1;
            }
        }
        for (int i = 0; i < 3; i++) {
            if (register_promoter(binop_ufuncs[k], &NullableDType, pyscalars[i]) < 0
                    || register_promoter(binop_ufuncs[k], pyscalars[i], &NullableDType) < 0) {
                Py_DECREF(numpy);
                return -1;
            }
        }
    }

    /* unary ufuncs: validity simply rides along */
    unop_dtypes[0] = &NullableDType;
    unop_dtypes[1] = &NullableDType;
    for (int k = 0; k < N_UNOPS; k++) {
        unop_ufuncs[k] = PyObject_GetAttrString(numpy, unop_names[k]);
        if (unop_ufuncs[k] == NULL) {
            Py_DECREF(numpy);
            return -1;
        }
        unop_specs[k] = (PyArrayMethod_Spec){
            .name = unop_names[k],
            .nin = 1, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_NO_CASTING,
            .dtypes = unop_dtypes,
            .slots = unop_slots[k],
        };
        if (PyUFunc_AddLoopFromSpec(unop_ufuncs[k], &unop_specs[k]) < 0) {
            Py_DECREF(numpy);
            return -1;
        }
    }

    /* matmul: a gufunc, so its loop also receives the core dimensions */
    {
        np_matmul = PyObject_GetAttrString(numpy, "matmul");
        if (np_matmul == NULL) {
            Py_DECREF(numpy);
            return -1;
        }
        static PyArrayMethod_Spec matmul_spec;
        matmul_spec = (PyArrayMethod_Spec){
            .name = "matmul",
            .nin = 2, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_NO_CASTING,
            .dtypes = binop_dtypes,
            .slots = nullable_matmul_slots,
        };
        if (PyUFunc_AddLoopFromSpec(np_matmul, &matmul_spec) < 0) {
            Py_DECREF(numpy);
            return -1;
        }
    }

    /* clip: three inputs, one output */
    {
        PyObject *umath = PyImport_ImportModule("numpy._core.umath");
        clip_ufunc = (umath != NULL)
                ? PyObject_GetAttrString(umath, "clip") : NULL;
        Py_XDECREF(umath);
        if (clip_ufunc == NULL) {
            Py_DECREF(numpy);
            return -1;
        }
        static PyArray_DTypeMeta *clip_dtypes[4];
        static PyArrayMethod_Spec clip_spec;
        for (int i = 0; i < 4; i++) {
            clip_dtypes[i] = &NullableDType;
        }
        clip_spec = (PyArrayMethod_Spec){
            .name = "clip",
            .nin = 3, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_NO_CASTING,
            .dtypes = clip_dtypes,
            .slots = nullable_clip_slots,
        };
        if (PyUFunc_AddLoopFromSpec(clip_ufunc, &clip_spec) < 0) {
            Py_DECREF(numpy);
            return -1;
        }
        if (register_clip_promoter(&NullableDType) < 0) {
            Py_DECREF(numpy);
            return -1;
        }
    }

    /* the three-valued logic ops need their own loops */
    binop_dtypes[0] = &NullableDType;
    for (int k = 0; k < 3; k++) {
        kleene_ufuncs[k] = PyObject_GetAttrString(numpy, kleene_names[k]);
        if (kleene_ufuncs[k] == NULL) {
            Py_DECREF(numpy);
            return -1;
        }
        kleene_specs[k] = (PyArrayMethod_Spec){
            .name = kleene_names[k],
            .nin = 2, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_NO_CASTING,
            .dtypes = binop_dtypes,
            .slots = kleene_slots[k],
        };
        if (PyUFunc_AddLoopFromSpec(kleene_ufuncs[k], &kleene_specs[k]) < 0) {
            Py_DECREF(numpy);
            return -1;
        }
        for (int i = 0; i < N_WRAPPABLE; i++) {
            PyArray_DTypeMeta *plain =
                    dtypemeta_from_typenum(wrappable_typenums[i]);
            if (plain == NULL
                    || register_promoter(kleene_ufuncs[k], &NullableDType, plain) < 0
                    || register_promoter(kleene_ufuncs[k], plain, &NullableDType) < 0) {
                Py_DECREF(numpy);
                return -1;
            }
        }
        for (int i = 0; i < 3; i++) {
            if (register_promoter(kleene_ufuncs[k], &NullableDType, pyscalars[i]) < 0
                    || register_promoter(kleene_ufuncs[k], pyscalars[i], &NullableDType) < 0) {
                Py_DECREF(numpy);
                return -1;
            }
        }
    }

    Py_DECREF(numpy);
    return 0;
}


static PyArray_DTypeMeta NullableDType = {{{
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "nulldtype.FlagDType",
        .tp_basicsize = sizeof(NullableDescr),
        .tp_new = nullable_new,
        .tp_dealloc = (destructor)nullable_dealloc,
        .tp_methods = nullable_methods,
        .tp_getset = nullable_getset,
        .tp_repr = (reprfunc)nullable_repr,
        .tp_str = (reprfunc)nullable_repr,
    }},
};


/* ---------------------------------------------------------------- module */

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


#include "sentinel.inc"


static PyTypeObject NullableScalarType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "nulldtype._FlagScalar",
    .tp_basicsize = sizeof(PyObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
};


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

    if (PyType_Ready(&NAType) < 0 || PyType_Ready(&NullableScalarType) < 0) {
        goto fail;
    }
    NA_singleton = PyObject_CallNoArgs((PyObject *)&NAType);
    if (NA_singleton == NULL) {
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
    if (PyModule_AddObjectRef(m, "NA", NA_singleton) < 0) {
        goto fail;
    }

    nullable_casting_dtypes[0] = &NullableDType;
    nullable_casting_dtypes[1] = &NullableDType;
    if (build_casts() < 0) {
        goto fail;
    }
    PyArrayMethod_Spec **casts = all_casts;
    PyArrayDTypeMeta_Spec spec = {
        .typeobj = &NAType,
        .flags = NPY_DT_PARAMETRIC,
        .casts = casts,
        .slots = NullableDType_slots,
    };

    ((PyObject *)&NullableDType)->ob_type = &PyArrayDTypeMeta_Type;
    ((PyTypeObject *)&NullableDType)->tp_base = &PyArrayDescr_Type;
    if (PyType_Ready((PyTypeObject *)&NullableDType) < 0) {
        goto fail;
    }
    if (PyArrayInitDTypeMeta_FromSpec(&NullableDType, &spec) < 0) {
        goto fail;
    }

    if (PyModule_AddObjectRef(m, "NullableDType", (PyObject *)&NullableDType) < 0) {
        goto fail;
    }
    /* the name users type: `Nullable(np.float64)` */
    if (PyModule_AddObjectRef(m, "Nullable", (PyObject *)&NullableDType) < 0) {
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

    if (init_binops() < 0) {
        goto fail;
    }
    if (init_sentinel_dtype(m) < 0) {
        goto fail;
    }

    return m;

  fail:
    Py_DECREF(m);
    return NULL;
}
