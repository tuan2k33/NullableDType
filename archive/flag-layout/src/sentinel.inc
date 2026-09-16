/* ==========================================================================
 * Sentinel[T] — the same idea as Nullable[T], but NA is a reserved value of
 * `T` itself instead of an extra byte.  The element size is unchanged, so the
 * values stay contiguous and aligned and the wrapped dtype's SIMD loops keep
 * working at full speed.
 *
 *   bool           the byte 2, which numpy never produces
 *   intN           INT_MIN
 *   uintN          UINT_MAX
 *   float, complex every bit but the sign (0x7FFF...), both halves for
 *                  complex; long double is stored as double
 *   datetime64,    INT64_MIN, which numpy already calls NaT
 *   timedelta64
 *   S, U, V        every byte 0xFF, at any width; in U that is the code unit
 *                  0xFFFFFFFF, which is no code point at all
 *   records        every field its own type's NA, padding zeroed; missing
 *                  only when every field is
 *
 * The full table with byte sizes is in LAYOUTS.md.
 *
 * NumPy already ships one of these: `NaT` for datetime64.  This just applies
 * the idea to every numeric dtype.
 *
 * NA is not left to hardware NaN propagation.  The loops read the inputs and
 * write NA into the output themselves, so `nan + NA` is NA whichever side it
 * is on, and the scheme does not depend on the CPU preserving NaN payloads.
 * ========================================================================== */

/*
 * Room for the NA pattern of the widest bitpattern type, complex128.  This
 * used to be 8, and `sentinel_put_na` copies `elsize` bytes out of it, so a
 * complex128 NA read 8 bytes past the end of the descriptor -- AddressSanitizer:
 * heap-buffer-overflow, READ of size 16, 0 bytes after the 112-byte object.
 * The imaginary half came out as zeros only because that memory happened to be.
 */
#define SENT_NA_MAX 16

#define SENT_KIND_EXACT 0     /* ints, bool, datetime: a plain bit pattern */
#define SENT_KIND_F4    1     /* float32: all bits but the sign */
#define SENT_KIND_F8    2
#define SENT_KIND_F2    3     /* float16 */
#define SENT_KIND_C8    4     /* complex: the pattern lives in the real half */
#define SENT_KIND_C16   5
#define SENT_KIND_FILL  6     /* S, U, V: every byte 0xFF, `na_bytes` unused */
#define SENT_KIND_RECORD 7    /* every field its own NA, see `sentinel_leaf` */

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
    int kind;                   /* never SENT_KIND_RECORD */
    char na_bytes[SENT_NA_MAX];
} sentinel_leaf;

typedef struct {
    PyArray_Descr base;
    PyArray_Descr *wrapped;
    int kind;
    char na_bytes[SENT_NA_MAX];
    sentinel_leaf *leaves;      /* records only, owned */
    npy_intp n_leaves;
} SentinelDescr;

static PyArray_DTypeMeta SentinelDType;

#define SENTINEL_DESCR(d) ((SentinelDescr *)(d))

/*
 * Float NA: every bit set except the sign -- 0x7FFF, 0x7FFFFFFF,
 * 0x7FFFFFFFFFFFFFFF.  One rule for every width, the same all-ones idea as
 * UINT_MAX.  It is a quiet NaN (exponent and top mantissa bit set), so an
 * operation that merely touches a gap raises no FE_INVALID, and ordinary
 * arithmetic never produces it: `np.nan` is 0x7FF8000000000000 and the NaN x86
 * generates for inf - inf is 0xFFF8000000000000.  `sentinel_is_na` ignores the
 * sign, so a gap whose sign a borrowed loop flipped still reads as a gap.
 */
#define NA_F8_BITS 0x7FFFFFFFFFFFFFFFULL
#define NA_F4_BITS 0x7FFFFFFFU
#define NA_F2_BITS 0x7FFFU


static int
sentinel_layout(int type_num, int *kind, char *na_bytes, int elsize)
{
    memset(na_bytes, 0, SENT_NA_MAX);
    switch (type_num) {
        case NPY_STRING: case NPY_UNICODE: case NPY_VOID:
            /*
             * The whole cell filled with 0xFF, whatever its width, so there is
             * nothing to keep in `na_bytes`.  In U it is the code unit
             * 0xFFFFFFFF, which `chr()` refuses, so no string is given up; in
             * S and V it is a legal byte string, given up like UINT_MAX.  A V
             * with fields never gets here: records go through `collect_leaves`.
             */
            *kind = SENT_KIND_FILL;
            return 0;
        default:
            break;
    }
    if (elsize > SENT_NA_MAX) {
        return -1;
    }
    switch (type_num) {
        case NPY_FLOAT64: {
            npy_uint64 bits = NA_F8_BITS;
            memcpy(na_bytes, &bits, 8);
            *kind = SENT_KIND_F8;
            return 0;
        }
        case NPY_FLOAT32: {
            npy_uint32 bits = NA_F4_BITS;
            memcpy(na_bytes, &bits, 4);
            *kind = SENT_KIND_F4;
            return 0;
        }
        case NPY_BOOL:
            na_bytes[0] = 2;
            *kind = SENT_KIND_EXACT;
            return 0;
        case NPY_FLOAT16: {
            npy_uint16 bits = NA_F2_BITS;
            memcpy(na_bytes, &bits, 2);
            *kind = SENT_KIND_F2;
            return 0;
        }
        case NPY_COMPLEX128: {
            /*
             * Both halves get the pattern, so whatever the arithmetic does
             * with them the result still carries a NaN; `sentinel_is_na` only
             * reads the real half, which is where a payload survives best.
             */
            npy_uint64 bits = NA_F8_BITS;
            memcpy(na_bytes, &bits, 8);
            memcpy(na_bytes + 8, &bits, 8);
            *kind = SENT_KIND_C16;
            return 0;
        }
        case NPY_COMPLEX64: {
            npy_uint32 bits = NA_F4_BITS;
            memcpy(na_bytes, &bits, 4);
            memcpy(na_bytes + 4, &bits, 4);
            *kind = SENT_KIND_C8;
            return 0;
        }
        case NPY_INT8: case NPY_INT16: case NPY_INT32: case NPY_INT64: {
            /* INT_MIN is 0x80 in the top byte, zeros below */
            na_bytes[elsize - 1] = (char)0x80;
            *kind = SENT_KIND_EXACT;
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
            *kind = SENT_KIND_EXACT;
            return 0;
        case NPY_DATETIME: case NPY_TIMEDELTA: {
            /*
             * Free: numpy already spells NaT as INT64_MIN in these, so the
             * reserved pattern is the one the type itself already reserves.
             * Nothing legal is taken away, and `filled()` hands back a real
             * NaT rather than an invented date.
             */
            na_bytes[elsize - 1] = (char)0x80;
            *kind = SENT_KIND_EXACT;
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
               sentinel_leaf **leaves, npy_intp *n, npy_intp *cap)
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
    char na_bytes[SENT_NA_MAX];
    if (descr->elsize <= 0
            || sentinel_layout(descr->type_num, &kind, na_bytes, descr->elsize) < 0) {
        return 1;
    }
    if (*n == *cap) {
        npy_intp new_cap = (*cap == 0) ? 8 : 2 * *cap;
        sentinel_leaf *grown = PyMem_Realloc(
                *leaves, (size_t)new_cap * sizeof(sentinel_leaf));
        if (grown == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        *leaves = grown;
        *cap = new_cap;
    }
    sentinel_leaf *leaf = &(*leaves)[(*n)++];
    leaf->offset = offset;
    leaf->count = count;
    leaf->elsize = descr->elsize;
    leaf->kind = kind;
    memcpy(leaf->na_bytes, na_bytes, SENT_NA_MAX);
    return 0;
}


static NPY_INLINE int
pattern_is_na(int kind, npy_intp elsize, const char *na_bytes, const char *data)
{
    if (kind == SENT_KIND_F8) {
        npy_uint64 u;
        memcpy(&u, data, 8);
        return (u & 0x7FFFFFFFFFFFFFFFULL) == 0x7FFFFFFFFFFFFFFFULL;
    }
    if (kind == SENT_KIND_F4) {
        npy_uint32 u;
        memcpy(&u, data, 4);
        return (u & 0x7FFFFFFFU) == 0x7FFFFFFFU;
    }
    if (kind == SENT_KIND_F2) {
        npy_uint16 u;
        memcpy(&u, data, 2);
        return (u & 0x7FFFU) == 0x7FFFU;
    }
    if (kind == SENT_KIND_C16) {
        npy_uint64 u;
        memcpy(&u, data, 8);          /* real half only */
        return (u & 0x7FFFFFFFFFFFFFFFULL) == 0x7FFFFFFFFFFFFFFFULL;
    }
    if (kind == SENT_KIND_C8) {
        npy_uint32 u;
        memcpy(&u, data, 4);
        return (u & 0x7FFFFFFFU) == 0x7FFFFFFFU;
    }
    if (kind == SENT_KIND_FILL) {
        /* the whole cell; a real value almost always stops at the first byte */
        const unsigned char *p = (const unsigned char *)data;
        for (npy_intp i = 0; i < elsize; i++) {
            if (p[i] != 0xFF) {
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
    if (kind == SENT_KIND_FILL) {
        memset(data, 0xFF, (size_t)elsize);
        return;
    }
    memcpy(data, na_bytes, (size_t)elsize);
}


static NPY_INLINE int
sentinel_is_na(const SentinelDescr *descr, const char *data)
{
    if (descr->kind != SENT_KIND_RECORD) {
        return pattern_is_na(descr->kind, descr->base.elsize, descr->na_bytes,
                             data);
    }
    /* missing only if every field is; the first field with a value settles it */
    for (npy_intp j = 0; j < descr->n_leaves; j++) {
        const sentinel_leaf *leaf = &descr->leaves[j];
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
sentinel_put_na(const SentinelDescr *descr, char *data)
{
    if (descr->kind != SENT_KIND_RECORD) {
        pattern_put_na(descr->kind, descr->base.elsize, descr->na_bytes, data);
        return;
    }
    /* padding zeroed too, so every gap has the same `tobytes()` */
    memset(data, 0, (size_t)descr->base.elsize);
    for (npy_intp j = 0; j < descr->n_leaves; j++) {
        const sentinel_leaf *leaf = &descr->leaves[j];
        for (npy_intp k = 0; k < leaf->count; k++) {
            pattern_put_na(leaf->kind, leaf->elsize, leaf->na_bytes,
                           data + leaf->offset + k * leaf->elsize);
        }
    }
}


static PyArray_Descr *new_sentinel_descr_impl(PyArray_Descr *wrapped);

/* the same stored dtype as the flag layout, see `stored_dtype` */
static PyArray_Descr *
new_sentinel_descr(PyArray_Descr *wrapped)
{
    PyArray_Descr *stored = stored_dtype(wrapped);
    if (stored == NULL) {
        return NULL;
    }
    PyArray_Descr *res = new_sentinel_descr_impl(stored);
    Py_DECREF(stored);
    return res;
}


static PyArray_Descr *
new_sentinel_descr_impl(PyArray_Descr *wrapped)
{
    PyArray_Descr *canonical = NULL;
    if (wrapped->type_num >= 0) {
        canonical = PyArray_DescrFromType(wrapped->type_num);
        if (canonical == NULL) {
            return NULL;
        }
        /* see the note in `new_nullable_descr`: `M8` and `M8[D]` are both 8 */
        if (PyArray_EquivTypes(canonical, wrapped)) {
            wrapped = canonical;
        }
    }
    int kind = SENT_KIND_EXACT;
    char na_bytes[SENT_NA_MAX];
    memset(na_bytes, 0, SENT_NA_MAX);
    sentinel_leaf *leaves = NULL;
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
        kind = SENT_KIND_RECORD;
    }
    else {
        usable = sentinel_layout(wrapped->type_num, &kind, na_bytes,
                                 wrapped->elsize) == 0;
    }
    if (!usable) {
        PyMem_Free(leaves);
        PyErr_Format(PyExc_TypeError,
                "Sentinel[T] has no spare value to reserve in %R; "
                "Nullable[T] works for any dtype", wrapped);
        Py_XDECREF(canonical);
        return NULL;
    }
    SentinelDescr *new = (SentinelDescr *)PyArrayDescr_Type.tp_new(
            (PyTypeObject *)&SentinelDType, NULL, NULL);
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
    memcpy(new->na_bytes, na_bytes, SENT_NA_MAX);
    new->base.elsize = wrapped->elsize;          /* no extra byte at all */
    new->base.alignment = wrapped->alignment;
    /* NPY_NEEDS_PYAPI: see `new_nullable_descr_impl` */
    new->base.flags = NPY_USE_GETITEM | NPY_USE_SETITEM | NPY_NEEDS_INIT
                      | NPY_NEEDS_PYAPI;
    new->base.byteorder = wrapped->byteorder;
    return (PyArray_Descr *)new;
}


static void
sentinel_dealloc(SentinelDescr *self)
{
    Py_CLEAR(self->wrapped);
    PyMem_Free(self->leaves);
    self->leaves = NULL;
    PyArrayDescr_Type.tp_dealloc((PyObject *)self);
}


static PyObject *
sentinel_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"dtype", NULL};
    PyObject *dtype_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:Sentinel", kwlist,
                                     &dtype_obj)) {
        return NULL;
    }
    PyArray_Descr *wrapped = NULL;
    if (!PyArray_DescrConverter(dtype_obj, &wrapped)) {
        return NULL;
    }
    PyObject *res = (PyObject *)new_sentinel_descr(wrapped);
    Py_DECREF(wrapped);
    return res;
}


static PyObject *
sentinel_repr(SentinelDescr *self)
{
    /*
     * Both layouts print the same way.  Which one `Nullable()` picked is an
     * implementation detail of the wrapped type, not something a caller chose,
     * so putting it in the repr would only invite people to depend on it.
     */
    return PyUnicode_FromFormat("Nullable(%R)", (PyObject *)self->wrapped);
}


static PyObject *
sentinel_reduce(SentinelDescr *self, PyObject *NPY_UNUSED(args))
{
    return Py_BuildValue("O(O)", Py_TYPE(self), (PyObject *)self->wrapped);
}


static PyObject *
sentinel_get_wrapped(SentinelDescr *self, void *NPY_UNUSED(closure))
{
    Py_INCREF(self->wrapped);
    return (PyObject *)self->wrapped;
}


static PyMethodDef sentinel_methods[] = {
    {"__reduce__", (PyCFunction)sentinel_reduce, METH_NOARGS, "pickle support"},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef sentinel_getset[] = {
    {"wrapped", (getter)sentinel_get_wrapped, NULL,
     "the dtype whose values this one stores", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};


/* ------------------------------------------------------------ dtype slots */

static PyArray_Descr *
sentinel_default_descr(PyArray_DTypeMeta *NPY_UNUSED(cls))
{
    PyArray_Descr *f8 = PyArray_DescrFromType(NPY_DOUBLE);
    PyArray_Descr *res = new_sentinel_descr(f8);
    Py_DECREF(f8);
    return res;
}


static PyArray_Descr *
sentinel_discover(PyArray_DTypeMeta *cls, PyObject *NPY_UNUSED(obj))
{
    return sentinel_default_descr(cls);
}


static PyArray_DTypeMeta *
sentinel_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    if (other == &SentinelDType || is_wrappable_typenum(other->type_num)) {
        Py_INCREF(cls);
        return cls;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


static PyArray_Descr *
sentinel_common_instance(PyArray_Descr *descr1, PyArray_Descr *descr2)
{
    if (SENTINEL_DESCR(descr1)->wrapped == SENTINEL_DESCR(descr2)->wrapped) {
        Py_INCREF(descr1);
        return descr1;
    }
    PyArray_Descr *value = PyArray_PromoteTypes(
            SENTINEL_DESCR(descr1)->wrapped, SENTINEL_DESCR(descr2)->wrapped);
    if (value == NULL) {
        return NULL;
    }
    PyArray_Descr *res = new_sentinel_descr(value);
    Py_DECREF(value);
    return res;
}


static PyArray_Descr *
sentinel_ensure_canonical(PyArray_Descr *self)
{
    Py_INCREF(self);
    return self;
}


static int
sentinel_setitem(PyArray_Descr *descr, PyObject *obj, char *dataptr)
{
    SentinelDescr *self = SENTINEL_DESCR(descr);
    if (obj == NA_singleton || obj == Py_None) {
        sentinel_put_na(self, dataptr);
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
    if (sentinel_is_na(self, dataptr)) {
        PyErr_Format(PyExc_ValueError,
                "%R is the value reserved to mean NA in %R", obj, descr);
        return -1;
    }
    return 0;
}


static PyObject *
sentinel_getitem(PyArray_Descr *descr, char *dataptr)
{
    SentinelDescr *self = SENTINEL_DESCR(descr);
    if (sentinel_is_na(self, dataptr)) {
        Py_INCREF(NA_singleton);
        return NA_singleton;
    }
    return PyArray_Scalar(dataptr, self->wrapped, NULL);
}


/* ----------------------------------------------------------------- casts */

static NPY_CASTING
sentinel_to_sentinel_resolve(
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
    if (!same_value_dtype(SENTINEL_DESCR(loop_descrs[0])->wrapped,
                          SENTINEL_DESCR(loop_descrs[1])->wrapped)) {
        return NPY_UNSAFE_CASTING;      /* handled by the loop below */
    }
    *view_offset = 0;
    return NPY_NO_CASTING;
}


/*
 * After the values of a cast into a Sentinel array have been written: a cell
 * whose source had a value but which now reads as NA was produced by the cast
 * itself -- an int64 wrapping onto INT32_MIN, an `S5` truncated to three 0xFF
 * bytes.  Left alone it would silently become a gap, so refuse instead.
 * `src_na` says whether source cell `i` was already missing; NULL means none
 * were.  Missing source cells get the destination's own NA stamped over them.
 */
static int
sentinel_stamp_and_check(SentinelDescr *to, char *out, npy_intp out_stride,
        npy_intp N, int (*src_na)(void *, npy_intp), void *src)
{
    for (npy_intp i = 0; i < N; i++) {
        char *cell = out + i * out_stride;
        if (src_na != NULL && src_na(src, i)) {
            sentinel_put_na(to, cell);
        }
        else if (sentinel_is_na(to, cell)) {
            PyErr_Format(PyExc_ValueError,
                    "the cast produced the value reserved to mean NA in %R",
                    (PyObject *)to);
            return -1;
        }
    }
    return 0;
}


typedef struct {
    SentinelDescr *descr;
    const char *data;
    npy_intp stride;
} sentinel_source;


static int
sentinel_source_na(void *src, npy_intp i)
{
    sentinel_source *s = (sentinel_source *)src;
    return sentinel_is_na(s->descr, s->data + i * s->stride);
}


static int
sentinel_to_sentinel_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    SentinelDescr *from = SENTINEL_DESCR(context->descriptors[0]);
    SentinelDescr *to = SENTINEL_DESCR(context->descriptors[1]);
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

    /* different value dtypes: cast the values, then restamp the gaps */
    PyObject *a = value_view(from->wrapped, in, N, strides[0]);
    PyObject *b = value_view(to->wrapped, out, N, strides[1]);
    if (a == NULL || b == NULL) {
        Py_XDECREF(a); Py_XDECREF(b);
        return -1;
    }
    int rc = (a && b) ? copy_values(b, a) : -1;
    Py_XDECREF(a); Py_XDECREF(b);
    if (rc < 0) {
        return -1;
    }
    sentinel_source src = {from, data[0], strides[0]};
    return sentinel_stamp_and_check(to, data[1], strides[1], N,
                                    &sentinel_source_na, &src);
}


static NPY_CASTING
to_sentinel_resolve(
        struct PyArrayMethodObject_tag *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[2]),
        PyArray_Descr *const given_descrs[2], PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = new_sentinel_descr(given_descrs[0]);
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
to_sentinel_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    PyArray_Descr *src = context->descriptors[0];
    SentinelDescr *dst = SENTINEL_DESCR(context->descriptors[1]);

    if (same_value_dtype(src, dst->wrapped)) {
        for (npy_intp i = 0; i < N; i++) {
            char *in = data[0] + i * strides[0];
            if (sentinel_is_na(dst, in)) {
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
    return sentinel_stamp_and_check(dst, data[1], strides[1], N, NULL, NULL);
}


static NPY_CASTING
from_sentinel_resolve(
        struct PyArrayMethodObject_tag *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[2]),
        PyArray_Descr *const given_descrs[2], PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    PyArray_Descr *wrapped = SENTINEL_DESCR(given_descrs[0])->wrapped;
    if (given_descrs[1] == NULL) {
        Py_INCREF(wrapped);
        loop_descrs[1] = wrapped;
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }
    return NPY_UNSAFE_CASTING;
}


static int
from_sentinel_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    SentinelDescr *src = SENTINEL_DESCR(context->descriptors[0]);
    PyArray_Descr *dst = context->descriptors[1];
    size_t itemsize = (size_t)src->base.elsize;

    for (npy_intp i = 0; i < N; i++) {
        if (sentinel_is_na(src, data[0] + i * strides[0])) {
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
#define SENT_FIXUP(CTYPE, MASKEXPR)                                           \
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
        }                                                                     \
    } while (0)

static void
sentinel_fixup(SentinelDescr *in0, SentinelDescr *in1, SentinelDescr *out,
        char *const data[], npy_intp const strides[], npy_intp N,
        int propagates_nan)
{
    npy_intp item = out->base.elsize;
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
        if (propagates_nan && out->kind == SENT_KIND_F8) {
            const double *po = (const double *)data[2];
            int any = 0;
            for (npy_intp i = 0; i < N; i++) {
                any |= (po[i] != po[i]);
            }
            if (!any) {
                return;
            }
        }
        else if (propagates_nan && out->kind == SENT_KIND_F4) {
            const float *po = (const float *)data[2];
            int any = 0;
            for (npy_intp i = 0; i < N; i++) {
                any |= (po[i] != po[i]);
            }
            if (!any) {
                return;
            }
        }
        switch (out->kind) {
            case SENT_KIND_F8:
                SENT_FIXUP(npy_uint64,
                    ((x & 0x7FFFFFFFFFFFFFFFULL) == 0x7FFFFFFFFFFFFFFFULL) ||
                    ((y & 0x7FFFFFFFFFFFFFFFULL) == 0x7FFFFFFFFFFFFFFFULL));
                return;
            case SENT_KIND_F4:
                SENT_FIXUP(npy_uint32,
                    ((x & 0x7FFFFFFFU) == 0x7FFFFFFFU) ||
                    ((y & 0x7FFFFFFFU) == 0x7FFFFFFFU));
                return;
            case SENT_KIND_EXACT:
                if (item == 8) { SENT_FIXUP(npy_uint64, x == na || y == na); return; }
                if (item == 4) { SENT_FIXUP(npy_uint32, x == na || y == na); return; }
                if (item == 2) { SENT_FIXUP(npy_uint16, x == na || y == na); return; }
                if (item == 1) { SENT_FIXUP(npy_uint8,  x == na || y == na); return; }
                break;
            default:
                /* f16 and complex: correctness first, via the loop below */
                break;
        }
    }
    for (npy_intp i = 0; i < N; i++) {
        if (sentinel_is_na(in0, data[0] + i * strides[0])
                || sentinel_is_na(in1, data[1] + i * strides[1])) {
            sentinel_put_na(out, data[2] + i * strides[2]);
        }
    }
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
sentinel_reduce_chunk(int idx, PyArrayMethod_Context *context,
        char *const data[], npy_intp N, npy_intp const strides[])
{
    SentinelDescr *acc = SENTINEL_DESCR(context->descriptors[0]);
    SentinelDescr *in = SENTINEL_DESCR(context->descriptors[1]);

    int missing = sentinel_is_na(acc, data[0]);
    for (npy_intp i = 0; i < N && !missing; i++) {
        missing = sentinel_is_na(in, data[1] + i * strides[1]);
    }
    if (missing) {
        sentinel_put_na(acc, data[2]);
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
    return rc;
}


/* `accumulate`: everything after the first gap is a gap too */
static int
sentinel_accumulate_chunk(int idx, PyArrayMethod_Context *context,
        char *const data[], npy_intp N, npy_intp const strides[])
{
    SentinelDescr *acc = SENTINEL_DESCR(context->descriptors[0]);
    SentinelDescr *in = SENTINEL_DESCR(context->descriptors[1]);
    SentinelDescr *out = SENTINEL_DESCR(context->descriptors[2]);

    npy_intp known = 0;
    if (!sentinel_is_na(acc, data[0])) {
        while (known < N && !sentinel_is_na(in, data[1] + known * strides[1])) {
            known++;
        }
    }
    for (npy_intp i = known; i < N; i++) {
        sentinel_put_na(out, data[2] + i * strides[2]);
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
sentinel_record_compare_resolve(PyArray_Descr *const given_descrs[3],
        PyArray_Descr *loop_descrs[3])
{
    PyArray_Descr *w0 = SENTINEL_DESCR(given_descrs[0])->wrapped;
    PyArray_Descr *w1 = SENTINEL_DESCR(given_descrs[1])->wrapped;
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
    loop_descrs[2] = new_sentinel_descr(bool_descr);
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
sentinel_record_compare(int op, PyArrayMethod_Context *context,
        char *const data[], npy_intp N, npy_intp const strides[])
{
    SentinelDescr *in0 = SENTINEL_DESCR(context->descriptors[0]);
    SentinelDescr *in1 = SENTINEL_DESCR(context->descriptors[1]);
    SentinelDescr *out = SENTINEL_DESCR(context->descriptors[2]);

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
        if (sentinel_is_na(in0, data[0] + i * strides[0])
                || sentinel_is_na(in1, data[1] + i * strides[1])) {
            sentinel_put_na(out, data[2] + i * strides[2]);
        }
    }
    return 0;
}


static NPY_CASTING
sentinel_binary_resolve_impl(PyObject *ufunc,
        PyArray_Descr *const given_descrs[3], PyArray_Descr *loop_descrs[3])
{
    if (SENTINEL_DESCR(given_descrs[0])->kind == SENT_KIND_RECORD
            || SENTINEL_DESCR(given_descrs[1])->kind == SENT_KIND_RECORD) {
        if (binop_rich_op(ufunc) >= 0) {
            return sentinel_record_compare_resolve(given_descrs, loop_descrs);
        }
        /* anything else falls through; `resolve_dtypes` refuses records */
    }
    PyObject *query = Py_BuildValue("(OOO)",
            (PyObject *)SENTINEL_DESCR(given_descrs[0])->wrapped,
            (PyObject *)SENTINEL_DESCR(given_descrs[1])->wrapped,
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
                && same_value_dtype(SENTINEL_DESCR(given_descrs[i])->wrapped,
                                    value)) {
            Py_INCREF(given_descrs[i]);
            loop_descrs[i] = (PyArray_Descr *)given_descrs[i];
            continue;
        }
        loop_descrs[i] = new_sentinel_descr(value);
        if (loop_descrs[i] == NULL) {
            for (int j = 0; j < i; j++) {
                Py_CLEAR(loop_descrs[j]);
            }
            Py_DECREF(resolved);
            return (NPY_CASTING)-1;
        }
        if (i < 2
                && !same_value_dtype(SENTINEL_DESCR(given_descrs[i])->wrapped,
                                     value)) {
            casting = NPY_SAFE_CASTING;
        }
    }
    Py_DECREF(resolved);
    return casting;
}


static int
sentinel_binary_loop_impl(int idx, PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[])
{
    npy_intp N = dimensions[0];
    SentinelDescr *in0 = SENTINEL_DESCR(context->descriptors[0]);
    SentinelDescr *in1 = SENTINEL_DESCR(context->descriptors[1]);
    SentinelDescr *out = SENTINEL_DESCR(context->descriptors[2]);

    if (in0->kind == SENT_KIND_RECORD) {
        /* only `==` and `!=` resolve for records */
        return sentinel_record_compare(binop_rich_op(binop_ufuncs[idx]),
                                       context, data, N, strides);
    }
    if (strides[0] == 0 && data[0] == data[2]) {
        return sentinel_reduce_chunk(idx, context, data, N, strides);
    }
    if (strides[0] == strides[2] && data[0] == data[2] - strides[2]) {
        return sentinel_accumulate_chunk(idx, context, data, N, strides);
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
        mask[i] = !sentinel_is_na(in0, data[0] + i * strides[0])
               && !sentinel_is_na(in1, data[1] + i * strides[1]);
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
    sentinel_fixup(in0, in1, out, data, strides, N, 0);
    return 0;
}


/*
 * The fast path: values are contiguous and aligned, so the wrapped loop runs
 * on the whole chunk at native speed.  Unlike the flag layout we do not even
 * need `where=`: a gap holds a NaN or INT_MIN, and computing with those is
 * harmless and quiet.  One pass afterwards stamps NA where an input had one.
 */
typedef struct {
    NpyAuxData base;
    int idx;
    PyObject *capsule;
    PyArrayMethod_StridedLoop *inner;
    PyArrayMethod_Context *inner_context;
    NpyAuxData *inner_auxdata;
} sentinel_auxdata;


static void
sentinel_auxdata_free(NpyAuxData *data)
{
    sentinel_auxdata *aux = (sentinel_auxdata *)data;
    Py_XDECREF(aux->capsule);
    PyMem_Free(aux);
}


static int
sentinel_fast_loop(PyArrayMethod_Context *context, char *const data[],
        npy_intp const dimensions[], npy_intp const strides[],
        NpyAuxData *auxdata)
{
    sentinel_auxdata *aux = (sentinel_auxdata *)auxdata;
    npy_intp N = dimensions[0];

    if (strides[0] == 0 && data[0] == data[2]) {
        return sentinel_reduce_chunk(aux->idx, context, data, N, strides);
    }
    if (strides[0] == strides[2] && data[0] == data[2] - strides[2]) {
        return sentinel_accumulate_chunk(aux->idx, context, data, N, strides);
    }

    npy_intp count = N;
    if (aux->inner(aux->inner_context, data, &count, strides,
                   aux->inner_auxdata) < 0) {
        return -1;
    }
    sentinel_fixup(SENTINEL_DESCR(context->descriptors[0]),
                   SENTINEL_DESCR(context->descriptors[1]),
                   SENTINEL_DESCR(context->descriptors[2]),
                   data, strides, N, !binop_swallows_nan(aux->idx));
    return 0;
}


static int
sentinel_get_loop_impl(int idx, PyArrayMethod_Context *context,
        int NPY_UNUSED(aligned), int NPY_UNUSED(move_references),
        const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_auxdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    *flags = NPY_METH_REQUIRES_PYAPI;

    /* records have no loop to borrow; `sentinel_record_compare` does it */
    if (SENTINEL_DESCR(context->descriptors[0])->kind == SENT_KIND_RECORD) {
        *out_auxdata = NULL;
        return 1;
    }
    /* these must not run over a gap at all; the masked path handles them */
    if (binop_signals_on_gap(idx)) {
        *out_auxdata = NULL;
        return 1;
    }

    /* see the note on minmax in nulldtype.c */
    if (binop_needs_whole_element_strides(idx)) {
        npy_intp elsize = SENTINEL_DESCR(context->descriptors[0])->wrapped->elsize;
        if (strides[0] % elsize != 0 || strides[1] % elsize != 0
                || strides[2] % elsize != 0) {
            *out_auxdata = NULL;
            return 1;
        }
    }
    PyObject *ufunc = binop_ufuncs[idx];
    PyObject *query = Py_BuildValue("(OOO)",
            (PyObject *)SENTINEL_DESCR(context->descriptors[0])->wrapped,
            (PyObject *)SENTINEL_DESCR(context->descriptors[1])->wrapped,
            (PyObject *)SENTINEL_DESCR(context->descriptors[2])->wrapped);
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
    sentinel_auxdata *aux = PyMem_Calloc(1, sizeof(sentinel_auxdata));
    if (aux == NULL) {
        Py_DECREF(capsule);
        PyErr_NoMemory();
        return -1;
    }
    aux->base.free = &sentinel_auxdata_free;
    aux->idx = idx;
    aux->capsule = capsule;
    aux->inner = info->strided_loop;
    aux->inner_context = info->context;
    aux->inner_auxdata = info->auxdata;
    *out_loop = &sentinel_fast_loop;
    *out_auxdata = (NpyAuxData *)aux;
    return 0;
}


static int
sentinel_reduction_initial_impl(int idx, PyArrayMethod_Context *context,
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
    SentinelDescr *out = SENTINEL_DESCR(context->descriptors[0]);
    PyObject *slot = value_view(out->wrapped, (char *)initial, 1, 0);
    int rc = (slot != NULL) ? copy_values(slot, identity) : -1;
    Py_XDECREF(slot);
    Py_DECREF(identity);
    return (rc < 0) ? -1 : 1;
}


#define SENTINEL_BINOP(IDX)                                                   \
static NPY_CASTING                                                            \
sentinel_resolve_##IDX(struct PyArrayMethodObject_tag *NPY_UNUSED(self),      \
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),                       \
        PyArray_Descr *const given_descrs[3], PyArray_Descr *loop_descrs[3],  \
        npy_intp *NPY_UNUSED(view_offset))                                    \
{                                                                             \
    return sentinel_binary_resolve_impl(binop_ufuncs[IDX], given_descrs,       \
                                        loop_descrs);      \
}                                                                             \
static int                                                                    \
sentinel_loop_##IDX(PyArrayMethod_Context *context, char *const data[],       \
        npy_intp const dimensions[], npy_intp const strides[],                \
        NpyAuxData *NPY_UNUSED(auxdata))                                      \
{                                                                             \
    return sentinel_binary_loop_impl(IDX, context, data, dimensions, strides);\
}                                                                             \
static int                                                                    \
sentinel_get_loop_##IDX(PyArrayMethod_Context *context, int aligned,          \
        int move_references, const npy_intp *strides,                         \
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_auxdata,       \
        NPY_ARRAYMETHOD_FLAGS *flags)                                         \
{                                                                             \
    int res = sentinel_get_loop_impl(IDX, context, aligned, move_references,  \
                                     strides, out_loop, out_auxdata, flags);  \
    if (res == 1) {                                                           \
        *out_loop = &sentinel_loop_##IDX;                                     \
        return 0;                                                             \
    }                                                                         \
    return res;                                                               \
}                                                                             \
static int                                                                    \
sentinel_initial_##IDX(PyArrayMethod_Context *context,                        \
        npy_bool reduction_is_empty, void *initial)                           \
{                                                                             \
    return sentinel_reduction_initial_impl(IDX, context,                      \
                                           reduction_is_empty, initial);      \
}                                                                             \
static PyType_Slot sentinel_slots_##IDX[] = {                                 \
    {NPY_METH_resolve_descriptors, &sentinel_resolve_##IDX},                  \
    {NPY_METH_get_loop, &sentinel_get_loop_##IDX},                            \
    {NPY_METH_get_reduction_initial, &sentinel_initial_##IDX},                \
    {0, NULL}                                                                 \
};

BINOP_INDICES(SENTINEL_BINOP)

#define SENTINEL_SLOTS_PTR(I) sentinel_slots_##I,
static PyType_Slot *sentinel_binop_slots[N_BINOPS] = { BINOP_INDICES(SENTINEL_SLOTS_PTR) };
#undef SENTINEL_SLOTS_PTR


/* --------------------------------------------- casts between the two layouts */
/*
 * `NA` is a scalar of NullableDType, so assigning it into a Sentinel array is
 * resolved as a cast from one layout to the other.  These also let a user move
 * an array between the two representations.
 */

static NPY_CASTING
nullable_to_sentinel_resolve(
        struct PyArrayMethodObject_tag *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[2]),
        PyArray_Descr *const given_descrs[2], PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = new_sentinel_descr(
                NULLABLE_DESCR(given_descrs[0])->wrapped);
        if (loop_descrs[1] == NULL) {
            Py_CLEAR(loop_descrs[0]);
            return (NPY_CASTING)-1;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }
    return NPY_SAME_KIND_CASTING;
}


typedef struct {
    const char *data;
    npy_intp stride;
    npy_intp voff;
} flag_source;


static int
flag_source_na(void *src, npy_intp i)
{
    flag_source *s = (flag_source *)src;
    return s->data[i * s->stride + s->voff] == 0;
}


static int
nullable_to_sentinel_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    PyArray_Descr *from = NULLABLE_DESCR(context->descriptors[0])->wrapped;
    SentinelDescr *to = SENTINEL_DESCR(context->descriptors[1]);
    npy_intp voff = from->elsize;

    if (!same_value_dtype(from, to->wrapped)) {
        /* let NumPy cast the values; the gaps are stamped below */
        PyObject *a = value_view(from, data[0], N, strides[0]);
        PyObject *b = value_view(to->wrapped, data[1], N, strides[1]);
        int rc = (a && b) ? copy_values(b, a) : -1;
        Py_XDECREF(a); Py_XDECREF(b);
        if (rc < 0) {
            return -1;
        }
    }
    else {
        for (npy_intp i = 0; i < N; i++) {
            memcpy(data[1] + i * strides[1], data[0] + i * strides[0],
                   (size_t)voff);
        }
    }
    /* a valid INT_MIN in the flag layout has no place in this one */
    flag_source src = {data[0], strides[0], voff};
    return sentinel_stamp_and_check(to, data[1], strides[1], N,
                                    &flag_source_na, &src);
}


static NPY_CASTING
sentinel_to_nullable_resolve(
        struct PyArrayMethodObject_tag *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[2]),
        PyArray_Descr *const given_descrs[2], PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = new_nullable_descr(
                SENTINEL_DESCR(given_descrs[0])->wrapped);
        if (loop_descrs[1] == NULL) {
            Py_CLEAR(loop_descrs[0]);
            return (NPY_CASTING)-1;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }
    return NPY_SAFE_CASTING;      /* every sentinel value fits the flag layout */
}


static int
sentinel_to_nullable_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    SentinelDescr *from = SENTINEL_DESCR(context->descriptors[0]);
    PyArray_Descr *to = NULLABLE_DESCR(context->descriptors[1])->wrapped;
    npy_intp voff = to->elsize;

    if (!same_value_dtype(from->wrapped, to)) {
        PyObject *a = value_view(from->wrapped, data[0], N, strides[0]);
        PyObject *b = value_view(to, data[1], N, strides[1]);
        int rc = (a && b) ? copy_values(b, a) : -1;
        Py_XDECREF(a); Py_XDECREF(b);
        if (rc < 0) {
            return -1;
        }
        for (npy_intp i = 0; i < N; i++) {
            char *out = data[1] + i * strides[1];
            int na = sentinel_is_na(from, data[0] + i * strides[0]);
            if (na) {
                memset(out, 0, (size_t)voff);
            }
            out[voff] = na ? 0 : 1;
        }
        return 0;
    }
    for (npy_intp i = 0; i < N; i++) {
        char *in = data[0] + i * strides[0];
        char *out = data[1] + i * strides[1];
        if (sentinel_is_na(from, in)) {
            memset(out, 0, (size_t)voff);
            out[voff] = 0;
        }
        else {
            memcpy(out, in, (size_t)voff);
            out[voff] = 1;
        }
    }
    return 0;
}


static PyArray_DTypeMeta *n2s_dtypes[2], *s2n_dtypes[2];
static PyType_Slot n2s_slots[] = {
    {NPY_METH_resolve_descriptors, &nullable_to_sentinel_resolve},
    {NPY_METH_strided_loop, &nullable_to_sentinel_loop},
    {NPY_METH_unaligned_strided_loop, &nullable_to_sentinel_loop},
    {0, NULL}
};
static PyType_Slot s2n_slots[] = {
    {NPY_METH_resolve_descriptors, &sentinel_to_nullable_resolve},
    {NPY_METH_strided_loop, &sentinel_to_nullable_loop},
    {NPY_METH_unaligned_strided_loop, &sentinel_to_nullable_loop},
    {0, NULL}
};
static PyArrayMethod_Spec N2SCast = {
    .name = "nullable_to_sentinel", .nin = 1, .nout = 1,
    .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
    .casting = NPY_SAME_KIND_CASTING, .dtypes = n2s_dtypes, .slots = n2s_slots,
};
static PyArrayMethod_Spec S2NCast = {
    .name = "sentinel_to_nullable", .nin = 1, .nout = 1,
    .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
    .casting = NPY_SAFE_CASTING, .dtypes = s2n_dtypes, .slots = s2n_slots,
};


/* ------------------------------------------------------------ unary loops */

static NPY_CASTING
sentinel_unary_resolve_impl(int idx,
        PyArray_Descr *const given_descrs[2], PyArray_Descr *loop_descrs[2])
{
    PyObject *query = Py_BuildValue("(OO)",
            (PyObject *)SENTINEL_DESCR(given_descrs[0])->wrapped, Py_None);
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
                && same_value_dtype(SENTINEL_DESCR(given_descrs[i])->wrapped,
                                    value)) {
            Py_INCREF(given_descrs[i]);
            loop_descrs[i] = (PyArray_Descr *)given_descrs[i];
            continue;
        }
        loop_descrs[i] = new_sentinel_descr(value);
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
sentinel_unary_loop_impl(int idx, PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[])
{
    npy_intp N = dimensions[0];
    SentinelDescr *in = SENTINEL_DESCR(context->descriptors[0]);
    SentinelDescr *out = SENTINEL_DESCR(context->descriptors[1]);

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
    for (npy_intp i = 0; i < N; i++) {
        if (sentinel_is_na(in, data[0] + i * strides[0])) {
            sentinel_put_na(out, data[1] + i * strides[1]);
        }
    }
    return 0;
}


#define SENTINEL_UNOP(IDX)                                                    \
static NPY_CASTING                                                            \
sentinel_ures_##IDX(struct PyArrayMethodObject_tag *NPY_UNUSED(self),         \
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[2]),                       \
        PyArray_Descr *const given_descrs[2], PyArray_Descr *loop_descrs[2],  \
        npy_intp *NPY_UNUSED(view_offset))                                    \
{                                                                             \
    return sentinel_unary_resolve_impl(IDX, given_descrs, loop_descrs);       \
}                                                                             \
static int                                                                    \
sentinel_uloop_##IDX(PyArrayMethod_Context *context, char *const data[],      \
        npy_intp const dimensions[], npy_intp const strides[],                \
        NpyAuxData *NPY_UNUSED(auxdata))                                      \
{                                                                             \
    return sentinel_unary_loop_impl(IDX, context, data, dimensions, strides); \
}                                                                             \
static PyType_Slot sentinel_uslots_##IDX[] = {                                \
    {NPY_METH_resolve_descriptors, &sentinel_ures_##IDX},                     \
    {NPY_METH_strided_loop, &sentinel_uloop_##IDX},                           \
    {NPY_METH_unaligned_strided_loop, &sentinel_uloop_##IDX},                 \
    {0, NULL}                                                                 \
};

UNOP_INDICES(SENTINEL_UNOP)

#define SENTINEL_USLOTS_PTR(I) sentinel_uslots_##I,
static PyType_Slot *sentinel_unop_slots[N_UNOPS] = { UNOP_INDICES(SENTINEL_USLOTS_PTR) };
#undef SENTINEL_USLOTS_PTR

static PyArray_DTypeMeta *sent_unop_dtypes[2] = {NULL, NULL};
static PyArrayMethod_Spec sent_unop_specs[N_UNOPS];


/* ---------------------------------------------------------------- matmul */
/* Same structure as the flag layout's; only how a gap is spotted differs. */

static NPY_CASTING
sentinel_matmul_resolve(struct PyArrayMethodObject_tag *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_Descr *const given_descrs[3], PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    return sentinel_binary_resolve_impl(np_matmul, given_descrs, loop_descrs);
}


static int
sentinel_matmul_loop(PyArrayMethod_Context *context, char *const data[],
        npy_intp const dimensions[], npy_intp const strides[],
        NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp outer = dimensions[0];
    npy_intp m = dimensions[1], n = dimensions[2], pdim = dimensions[3];
    npy_intp o0 = strides[0], o1 = strides[1], o2 = strides[2];
    npy_intp a_m = strides[3], a_n = strides[4];
    npy_intp b_n = strides[5], b_p = strides[6];
    npy_intp c_m = strides[7], c_p = strides[8];

    SentinelDescr *da = SENTINEL_DESCR(context->descriptors[0]);
    SentinelDescr *db = SENTINEL_DESCR(context->descriptors[1]);
    SentinelDescr *dc = SENTINEL_DESCR(context->descriptors[2]);

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
                bad |= sentinel_is_na(da, pa + i * a_m + k * a_n);
            }
            rowna[i] = bad;
        }
        for (npy_intp j = 0; j < pdim; j++) {
            char bad = 0;
            for (npy_intp k = 0; k < n; k++) {
                bad |= sentinel_is_na(db, pb + k * b_n + j * b_p);
            }
            colna[j] = bad;
        }
        for (npy_intp i = 0; i < m; i++) {
            for (npy_intp j = 0; j < pdim; j++) {
                if (rowna[i] || colna[j]) {
                    sentinel_put_na(dc, pc + i * c_m + j * c_p);
                }
            }
        }
    }
    PyMem_Free(rowna); PyMem_Free(colna);
    return 0;
}


static PyType_Slot sentinel_matmul_slots[] = {
    {NPY_METH_resolve_descriptors, &sentinel_matmul_resolve},
    {NPY_METH_strided_loop, &sentinel_matmul_loop},
    {NPY_METH_unaligned_strided_loop, &sentinel_matmul_loop},
    {0, NULL},
};


/* ------------------------------------------------------------------ clip */
/* Same shape as the flag layout's clip; only how a gap is spotted differs. */

static NPY_CASTING
sentinel_clip_resolve(struct PyArrayMethodObject_tag *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[4]),
        PyArray_Descr *const given_descrs[4], PyArray_Descr *loop_descrs[4],
        npy_intp *NPY_UNUSED(view_offset))
{
    PyObject *query = Py_BuildValue("(OOOO)",
            (PyObject *)SENTINEL_DESCR(given_descrs[0])->wrapped,
            (PyObject *)SENTINEL_DESCR(given_descrs[1])->wrapped,
            (PyObject *)SENTINEL_DESCR(given_descrs[2])->wrapped,
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
        loop_descrs[i] = new_sentinel_descr(value_descr);
        if (loop_descrs[i] == NULL) {
            for (int j = 0; j < i; j++) {
                Py_CLEAR(loop_descrs[j]);
            }
            Py_DECREF(resolved);
            return (NPY_CASTING)-1;
        }
        if (i < 3
                && !same_value_dtype(SENTINEL_DESCR(given_descrs[i])->wrapped,
                                     value_descr)) {
            casting = NPY_SAFE_CASTING;
        }
    }
    Py_DECREF(resolved);
    return casting;
}


static int
sentinel_clip_loop(PyArrayMethod_Context *context, char *const data[],
        npy_intp const dimensions[], npy_intp const strides[],
        NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    SentinelDescr *d[4];
    PyObject *v[4] = {NULL, NULL, NULL, NULL};
    for (int i = 0; i < 4; i++) {
        d[i] = SENTINEL_DESCR(context->descriptors[i]);
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
        m[i] = !sentinel_is_na(d[0], data[0] + i * strides[0])
            && !sentinel_is_na(d[1], data[1] + i * strides[1])
            && !sentinel_is_na(d[2], data[2] + i * strides[2]);
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
            sentinel_put_na(d[3], data[3] + i * strides[3]);
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


static PyType_Slot sentinel_clip_slots[] = {
    {NPY_METH_resolve_descriptors, &sentinel_clip_resolve},
    {NPY_METH_strided_loop, &sentinel_clip_loop},
    {NPY_METH_unaligned_strided_loop, &sentinel_clip_loop},
    {0, NULL},
};


/* ------------------------------------------------------- Kleene logic */

static NPY_CASTING
sentinel_kleene_resolve_impl(PyArray_Descr *const given_descrs[3],
        PyArray_Descr *loop_descrs[3])
{
    PyArray_Descr *bool_descr = PyArray_DescrFromType(NPY_BOOL);
    if (bool_descr == NULL) {
        return (NPY_CASTING)-1;
    }
    for (int i = 0; i < 2; i++) {
        Py_INCREF(given_descrs[i]);
        loop_descrs[i] = (PyArray_Descr *)given_descrs[i];
    }
    loop_descrs[2] = new_sentinel_descr(bool_descr);
    Py_DECREF(bool_descr);
    if (loop_descrs[2] == NULL) {
        Py_CLEAR(loop_descrs[0]);
        Py_CLEAR(loop_descrs[1]);
        return (NPY_CASTING)-1;
    }
    return NPY_SAFE_CASTING;
}


/*
 * One chunk of a Kleene `reduce`.
 *
 * A settling value -- False for AND, True for OR -- fixes the answer for good;
 * no later gap can unfix it, because whatever the missing element turns out to
 * be, the decision is already made.  Otherwise any gap makes the result NA.
 */
static int
sentinel_kleene_reduce_chunk(int op, PyArrayMethod_Context *context,
        char *const data[], npy_intp N, npy_intp const strides[])
{
    SentinelDescr *acc = SENTINEL_DESCR(context->descriptors[0]);
    SentinelDescr *in = SENTINEL_DESCR(context->descriptors[1]);
    SentinelDescr *out = SENTINEL_DESCR(context->descriptors[2]);

    PyArrayObject *tb = truthiness(in->wrapped, data[1], N, strides[1]);
    if (tb == NULL) {
        return -1;
    }
    const char *vals = PyArray_DATA(tb);

    int valid = !sentinel_is_na(acc, data[0]);
    char value = valid ? (*data[0] != 0) : 0;
    char settles = (op == KLEENE_AND) ? 0 : 1;

    for (npy_intp i = 0; i < N; i++) {
        int b_valid = !sentinel_is_na(in, data[1] + i * strides[1]);
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
        sentinel_put_na(out, data[2]);
    }
    return 0;
}


static int
sentinel_kleene_loop_impl(int op, PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[])
{
    npy_intp N = dimensions[0];
    SentinelDescr *in0 = SENTINEL_DESCR(context->descriptors[0]);
    SentinelDescr *in1 = SENTINEL_DESCR(context->descriptors[1]);
    SentinelDescr *out = SENTINEL_DESCR(context->descriptors[2]);

    /*
     * A reduction hands us the accumulator as first input *and* output, both
     * at stride 0.  Running the elementwise code over that writes the same
     * slot N times while still reading the accumulator's original value, so
     * only the last element would survive.  Fold the run sequentially instead.
     */
    if (strides[0] == 0 && data[0] == data[2]) {
        return sentinel_kleene_reduce_chunk(op, context, data, N, strides);
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
        int a_valid = !sentinel_is_na(in0, data[0] + i * strides[0]);
        int b_valid = !sentinel_is_na(in1, data[1] + i * strides[1]);
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
            sentinel_put_na(out, slot);
        }
    }
    Py_DECREF(ta); Py_DECREF(tb);
    return 0;
}


static int
sentinel_kleene_initial_impl(int op, PyArrayMethod_Context *NPY_UNUSED(context),
        npy_bool NPY_UNUSED(reduction_is_empty), void *initial)
{
    ((char *)initial)[0] = (op == KLEENE_AND);
    return 1;
}


#define SENTINEL_KLEENE(OP)                                                   \
static NPY_CASTING                                                            \
sent_kleene_res_##OP(struct PyArrayMethodObject_tag *NPY_UNUSED(self),        \
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),                       \
        PyArray_Descr *const given_descrs[3], PyArray_Descr *loop_descrs[3],  \
        npy_intp *NPY_UNUSED(view_offset))                                    \
{                                                                             \
    return sentinel_kleene_resolve_impl(given_descrs, loop_descrs);           \
}                                                                             \
static int                                                                    \
sent_kleene_loop_##OP(PyArrayMethod_Context *context, char *const data[],     \
        npy_intp const dimensions[], npy_intp const strides[],                \
        NpyAuxData *NPY_UNUSED(auxdata))                                      \
{                                                                             \
    return sentinel_kleene_loop_impl(OP, context, data, dimensions, strides); \
}                                                                             \
static int                                                                    \
sent_kleene_init_##OP(PyArrayMethod_Context *context,                         \
        npy_bool reduction_is_empty, void *initial)                           \
{                                                                             \
    return sentinel_kleene_initial_impl(OP, context, reduction_is_empty,      \
                                        initial);                             \
}                                                                             \
static PyType_Slot sent_kleene_slots_##OP[] = {                               \
    {NPY_METH_resolve_descriptors, &sent_kleene_res_##OP},                    \
    {NPY_METH_strided_loop, &sent_kleene_loop_##OP},                          \
    {NPY_METH_unaligned_strided_loop, &sent_kleene_loop_##OP},                \
    {NPY_METH_get_reduction_initial, &sent_kleene_init_##OP},                 \
    {0, NULL}                                                                 \
};

SENTINEL_KLEENE(0)
SENTINEL_KLEENE(1)
SENTINEL_KLEENE(2)

static PyType_Slot *sent_kleene_slots[3] = {
    sent_kleene_slots_0, sent_kleene_slots_1, sent_kleene_slots_2};
static PyArrayMethod_Spec sent_kleene_specs[3];


/* ------------------------------------------------------------ dtype setup */

/*
 * NumPy insists every DType names a scalar type, and two DTypes may not share
 * one.  `NAType` already belongs to NullableDType, so Sentinel gets a private
 * placeholder.  It is never handed to users: `getitem` still returns the one
 * `NA` singleton, so `x is NA` works for both dtypes.
 */
static PyTypeObject SentinelScalarType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "nulldtype._BitpatternScalar",
    .tp_basicsize = sizeof(PyObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
};


/* NA sorts last, two NAs are equal — same rule as the flag layout */
static int
sentinel_compare(const void *a, const void *b, void *arr)
{
    SentinelDescr *descr = SENTINEL_DESCR(PyArray_DESCR((PyArrayObject *)arr));
    int na_a = sentinel_is_na(descr, (const char *)a);
    int na_b = sentinel_is_na(descr, (const char *)b);
    if (na_a || na_b) {
        return na_a ? (na_b ? 0 : 1) : -1;
    }
    return wrapped_compare(descr->wrapped, (const char *)a, (const char *)b);
}


static npy_bool
sentinel_nonzero(void *data, void *arr)
{
    SentinelDescr *descr = SENTINEL_DESCR(PyArray_DESCR((PyArrayObject *)arr));
    if (sentinel_is_na(descr, (const char *)data)) {
        PyErr_SetString(PyExc_ValueError,
                "cannot take the truth value of a missing element; "
                "fill it first, for example with `filled(False)`");
        return NPY_FALSE;
    }
    return wrapped_nonzero(descr->wrapped, (const char *)data);
}


static int
sentinel_argmax(void *data, npy_intp n, npy_intp *max_ind, void *arr)
{
    SentinelDescr *descr = SENTINEL_DESCR(PyArray_DESCR((PyArrayObject *)arr));
    npy_intp item = descr->base.elsize;
    char *ptr = (char *)data;
    npy_intp best = -1;
    for (npy_intp i = 0; i < n; i++) {
        if (sentinel_is_na(descr, ptr + i * item)) {
            continue;
        }
        if (best < 0 || sentinel_compare(ptr + i * item,
                                         ptr + best * item, arr) > 0) {
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
sentinel_argmin(void *data, npy_intp n, npy_intp *min_ind, void *arr)
{
    SentinelDescr *descr = SENTINEL_DESCR(PyArray_DESCR((PyArrayObject *)arr));
    npy_intp item = descr->base.elsize;
    char *ptr = (char *)data;
    npy_intp best = -1;
    for (npy_intp i = 0; i < n; i++) {
        if (sentinel_is_na(descr, ptr + i * item)) {
            continue;
        }
        if (best < 0 || sentinel_compare(ptr + i * item,
                                         ptr + best * item, arr) < 0) {
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


static PyType_Slot SentinelDType_slots[] = {
    {NPY_DT_discover_descr_from_pyobject, &sentinel_discover},
    {NPY_DT_default_descr, &sentinel_default_descr},
    {NPY_DT_common_dtype, &sentinel_common_dtype},
    {NPY_DT_common_instance, &sentinel_common_instance},
    {NPY_DT_ensure_canonical, &sentinel_ensure_canonical},
    {NPY_DT_setitem, &sentinel_setitem},
    {NPY_DT_getitem, &sentinel_getitem},
    {NPY_DT_PyArray_ArrFuncs_compare, &sentinel_compare},
    {NPY_DT_PyArray_ArrFuncs_nonzero, &sentinel_nonzero},
    {NPY_DT_PyArray_ArrFuncs_argmax, &sentinel_argmax},
    {NPY_DT_PyArray_ArrFuncs_argmin, &sentinel_argmin},
    {0, NULL}
};

static PyArray_DTypeMeta SentinelDType = {{{
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "nulldtype.BitpatternDType",
        .tp_basicsize = sizeof(SentinelDescr),
        .tp_new = sentinel_new,
        .tp_dealloc = (destructor)sentinel_dealloc,
        .tp_methods = sentinel_methods,
        .tp_getset = sentinel_getset,
        .tp_repr = (reprfunc)sentinel_repr,
        .tp_str = (reprfunc)sentinel_repr,
    }},
};


static PyArray_DTypeMeta *sent_self_dtypes[2] = {NULL, NULL};
static PyType_Slot sent_self_slots[] = {
    {NPY_METH_resolve_descriptors, &sentinel_to_sentinel_resolve},
    {NPY_METH_strided_loop, &sentinel_to_sentinel_loop},
    {NPY_METH_unaligned_strided_loop, &sentinel_to_sentinel_loop},
    {0, NULL}
};
static PyArrayMethod_Spec SentSelfCast = {
    .name = "sentinel_to_sentinel", .nin = 1, .nout = 1,
    .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
    .casting = NPY_NO_CASTING,
    .dtypes = sent_self_dtypes, .slots = sent_self_slots,
};

static PyType_Slot sent_to_slots[] = {
    {NPY_METH_resolve_descriptors, &to_sentinel_resolve},
    {NPY_METH_strided_loop, &to_sentinel_loop},
    {NPY_METH_unaligned_strided_loop, &to_sentinel_loop},
    {0, NULL}
};
static PyType_Slot sent_from_slots[] = {
    {NPY_METH_resolve_descriptors, &from_sentinel_resolve},
    {NPY_METH_strided_loop, &from_sentinel_loop},
    {NPY_METH_unaligned_strided_loop, &from_sentinel_loop},
    {0, NULL}
};

/* the dtypes a sentinel can be built from: those with a spare value */
#define N_SENTINEL_TYPES 21
static const int sentinel_typenums[N_SENTINEL_TYPES] = {
    NPY_BOOL, NPY_INT8, NPY_INT16, NPY_INT32, NPY_INT64,
    NPY_UINT8, NPY_UINT16, NPY_UINT32, NPY_UINT64,
    NPY_FLOAT16, NPY_FLOAT32, NPY_FLOAT64,
    NPY_COMPLEX64, NPY_COMPLEX128,
    /* free: numpy already reserves INT64_MIN in these and calls it NaT */
    NPY_DATETIME, NPY_TIMEDELTA,
    /*
     * Not layouts of their own -- stored as float64 and complex128, see
     * `substitute_long_double` -- but plain arrays of them still cast in.
     */
    NPY_LONGDOUBLE, NPY_CLONGDOUBLE,
    /* any width; VOID covers records too, which `new_sentinel_descr` refuses */
    NPY_STRING, NPY_UNICODE, NPY_VOID,
};
static PyArray_DTypeMeta *sent_cast_dtypes[N_SENTINEL_TYPES][2][2];
static PyArrayMethod_Spec sent_cast_specs[N_SENTINEL_TYPES][2];
static PyArrayMethod_Spec *sent_all_casts[N_SENTINEL_TYPES * 2 + 4];
static PyArray_DTypeMeta *sent_binop_dtypes[3] = {NULL, NULL, NULL};
static PyArrayMethod_Spec sent_binop_specs[N_BINOPS];


static int
init_sentinel_dtype(PyObject *m)
{
    if (PyType_Ready(&SentinelScalarType) < 0) {
        return -1;
    }
    ((PyObject *)&SentinelDType)->ob_type = &PyArrayDTypeMeta_Type;
    ((PyTypeObject *)&SentinelDType)->tp_base = &PyArrayDescr_Type;
    if (PyType_Ready((PyTypeObject *)&SentinelDType) < 0) {
        return -1;
    }

    sent_self_dtypes[0] = &SentinelDType;
    sent_self_dtypes[1] = &SentinelDType;
    int n = 0;
    sent_all_casts[n++] = &SentSelfCast;
    for (int i = 0; i < N_SENTINEL_TYPES; i++) {
        PyArray_DTypeMeta *plain = dtypemeta_from_typenum(sentinel_typenums[i]);
        if (plain == NULL) {
            return -1;
        }
        sent_cast_dtypes[i][0][0] = plain;
        sent_cast_dtypes[i][0][1] = &SentinelDType;
        sent_cast_specs[i][0] = (PyArrayMethod_Spec){
            .name = "to_sentinel", .nin = 1, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_SAME_KIND_CASTING,
            .dtypes = sent_cast_dtypes[i][0], .slots = sent_to_slots,
        };
        sent_all_casts[n++] = &sent_cast_specs[i][0];

        sent_cast_dtypes[i][1][0] = &SentinelDType;
        sent_cast_dtypes[i][1][1] = plain;
        sent_cast_specs[i][1] = (PyArrayMethod_Spec){
            .name = "from_sentinel", .nin = 1, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_UNSAFE_CASTING,
            .dtypes = sent_cast_dtypes[i][1], .slots = sent_from_slots,
        };
        sent_all_casts[n++] = &sent_cast_specs[i][1];
    }
    n2s_dtypes[0] = &NullableDType; n2s_dtypes[1] = &SentinelDType;
    s2n_dtypes[0] = &SentinelDType; s2n_dtypes[1] = &NullableDType;
    sent_all_casts[n++] = &N2SCast;
    sent_all_casts[n++] = &S2NCast;
    sent_all_casts[n] = NULL;

    PyArrayDTypeMeta_Spec spec = {
        .typeobj = &SentinelScalarType,
        .flags = NPY_DT_PARAMETRIC,
        .casts = sent_all_casts,
        .slots = SentinelDType_slots,
    };
    if (PyArrayInitDTypeMeta_FromSpec(&SentinelDType, &spec) < 0) {
        return -1;
    }

    sent_binop_dtypes[0] = &SentinelDType;
    sent_binop_dtypes[1] = &SentinelDType;
    sent_binop_dtypes[2] = &SentinelDType;
    PyArray_DTypeMeta *pyscalars[3] = {
        &PyArray_PyLongDType, &PyArray_PyFloatDType, &PyArray_PyComplexDType};
    for (int k = 0; k < N_BINOPS; k++) {
        sent_binop_specs[k] = (PyArrayMethod_Spec){
            .name = binop_names[k], .nin = 2, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_NO_CASTING,
            .dtypes = sent_binop_dtypes, .slots = sentinel_binop_slots[k],
        };
        if (PyUFunc_AddLoopFromSpec(binop_ufuncs[k], &sent_binop_specs[k]) < 0) {
            return -1;
        }
        for (int i = 0; i < N_SENTINEL_TYPES; i++) {
            PyArray_DTypeMeta *plain =
                    dtypemeta_from_typenum(sentinel_typenums[i]);
            if (plain == NULL
                    || register_promoter(binop_ufuncs[k], &SentinelDType, plain) < 0
                    || register_promoter(binop_ufuncs[k], plain, &SentinelDType) < 0) {
                return -1;
            }
        }
        for (int i = 0; i < 3; i++) {
            if (register_promoter(binop_ufuncs[k], &SentinelDType, pyscalars[i]) < 0
                    || register_promoter(binop_ufuncs[k], pyscalars[i], &SentinelDType) < 0) {
                return -1;
            }
        }
    }

    sent_unop_dtypes[0] = &SentinelDType;
    sent_unop_dtypes[1] = &SentinelDType;
    for (int k = 0; k < N_UNOPS; k++) {
        sent_unop_specs[k] = (PyArrayMethod_Spec){
            .name = unop_names[k], .nin = 1, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_NO_CASTING,
            .dtypes = sent_unop_dtypes, .slots = sentinel_unop_slots[k],
        };
        if (PyUFunc_AddLoopFromSpec(unop_ufuncs[k], &sent_unop_specs[k]) < 0) {
            return -1;
        }
    }

    {
        static PyArrayMethod_Spec sent_matmul_spec;
        sent_matmul_spec = (PyArrayMethod_Spec){
            .name = "matmul", .nin = 2, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_NO_CASTING,
            .dtypes = sent_binop_dtypes, .slots = sentinel_matmul_slots,
        };
        if (PyUFunc_AddLoopFromSpec(np_matmul, &sent_matmul_spec) < 0) {
            return -1;
        }
    }

    {
        static PyArray_DTypeMeta *sent_clip_dtypes[4];
        static PyArrayMethod_Spec sent_clip_spec;
        for (int i = 0; i < 4; i++) {
            sent_clip_dtypes[i] = &SentinelDType;
        }
        sent_clip_spec = (PyArrayMethod_Spec){
            .name = "clip", .nin = 3, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_NO_CASTING,
            .dtypes = sent_clip_dtypes, .slots = sentinel_clip_slots,
        };
        if (PyUFunc_AddLoopFromSpec(clip_ufunc, &sent_clip_spec) < 0
                || register_clip_promoter(&SentinelDType) < 0) {
            return -1;
        }
    }

    for (int k = 0; k < 3; k++) {
        sent_kleene_specs[k] = (PyArrayMethod_Spec){
            .name = kleene_names[k], .nin = 2, .nout = 1,
            .flags = NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI,
            .casting = NPY_NO_CASTING,
            .dtypes = sent_binop_dtypes, .slots = sent_kleene_slots[k],
        };
        if (PyUFunc_AddLoopFromSpec(kleene_ufuncs[k], &sent_kleene_specs[k]) < 0) {
            return -1;
        }
        for (int i = 0; i < N_SENTINEL_TYPES; i++) {
            PyArray_DTypeMeta *plain =
                    dtypemeta_from_typenum(sentinel_typenums[i]);
            if (plain == NULL
                    || register_promoter(kleene_ufuncs[k], &SentinelDType, plain) < 0
                    || register_promoter(kleene_ufuncs[k], plain, &SentinelDType) < 0) {
                return -1;
            }
        }
    }

    if (PyModule_AddObjectRef(m, "SentinelDType", (PyObject *)&SentinelDType) < 0
            || PyModule_AddObjectRef(m, "Sentinel", (PyObject *)&SentinelDType) < 0) {
        return -1;
    }
    return 0;
}
