/*
 * _units_accel.c — C-accelerated physical units for xue
 *
 * Dimension exponents packed into int64 for single-instruction comparison.
 * Quantity arithmetic avoids Python object creation on the hot path.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <math.h>
#include <stdint.h>

/* ── Forward declarations ─────────────────────────────────────────── */

typedef struct DimensionObject DimensionObject;
typedef struct UnitObject UnitObject;

/* QuantityObject must be fully defined before UnitObject methods use it */
typedef struct QuantityObject {
    PyObject_HEAD
    double value;
    UnitObject *unit;
} QuantityObject;

static PyTypeObject DimensionType;
static PyTypeObject UnitType;
static PyTypeObject QuantityType;
static PyObject *DimensionError;

/* ── Dimension packing helpers ────────────────────────────────────── */

static const char *DIM_LABELS[] = {"m", "kg", "s", "A", "K", "mol", "cd"};

static inline int64_t dim_pack7(int e0, int e1, int e2, int e3,
                                int e4, int e5, int e6) {
    return ((int64_t)(uint8_t)(int8_t)e0)       |
           ((int64_t)(uint8_t)(int8_t)e1 << 8)  |
           ((int64_t)(uint8_t)(int8_t)e2 << 16) |
           ((int64_t)(uint8_t)(int8_t)e3 << 24) |
           ((int64_t)(uint8_t)(int8_t)e4 << 32) |
           ((int64_t)(uint8_t)(int8_t)e5 << 40) |
           ((int64_t)(uint8_t)(int8_t)e6 << 48);
}

static inline int8_t dim_get(int64_t packed, int idx) {
    return (int8_t)((packed >> (idx * 8)) & 0xFF);
}

static inline int64_t dim_add(int64_t a, int64_t b) {
    int64_t r = 0;
    for (int i = 0; i < 7; i++) {
        int8_t sum = dim_get(a, i) + dim_get(b, i);
        r |= ((int64_t)(uint8_t)sum) << (i * 8);
    }
    return r;
}

static inline int64_t dim_sub(int64_t a, int64_t b) {
    int64_t r = 0;
    for (int i = 0; i < 7; i++) {
        int8_t diff = dim_get(a, i) - dim_get(b, i);
        r |= ((int64_t)(uint8_t)diff) << (i * 8);
    }
    return r;
}

static inline int64_t dim_scale(int64_t a, int n) {
    int64_t r = 0;
    for (int i = 0; i < 7; i++) {
        int8_t v = dim_get(a, i) * n;
        r |= ((int64_t)(uint8_t)v) << (i * 8);
    }
    return r;
}

static inline int64_t dim_negate(int64_t a) {
    return dim_scale(a, -1);
}

/* ── Dimension cache ──────────────────────────────────────────────── */

#define DIM_CACHE_SIZE 128
static DimensionObject *dim_cache[DIM_CACHE_SIZE];

static DimensionObject *Dimension_get_cached(int64_t packed, PyObject *name);

/* ══════════════════════════════════════════════════════════════════
   DimensionObject
   ══════════════════════════════════════════════════════════════════ */

struct DimensionObject {
    PyObject_HEAD
    int64_t packed;
    PyObject *name;   /* str or None */
};

static void Dimension_dealloc(DimensionObject *self) {
    Py_XDECREF(self->name);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Dimension_new(PyTypeObject *type, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"exponents", "name", NULL};
    PyObject *exponents = NULL;
    PyObject *name = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "|OO", kwlist, &exponents, &name))
        return NULL;

    int e[7] = {0};
    if (exponents && exponents != Py_None) {
        PyObject *seq = PySequence_Fast(exponents, "exponents must be a sequence");
        if (!seq) return NULL;
        Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
        if (n != 7) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "exponents must have exactly 7 elements");
            return NULL;
        }
        for (int i = 0; i < 7; i++) {
            e[i] = (int)PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
        }
        Py_DECREF(seq);
        if (PyErr_Occurred()) return NULL;
    }

    int64_t packed = dim_pack7(e[0], e[1], e[2], e[3], e[4], e[5], e[6]);

    PyObject *name_str = Py_None;
    if (name != Py_None && PyUnicode_Check(name)) {
        name_str = name;
    }

    /* Check cache (only for unnamed dimensions) */
    if (name_str == Py_None) {
        unsigned int idx = (unsigned int)((uint64_t)packed % DIM_CACHE_SIZE);
        DimensionObject *cached = dim_cache[idx];
        if (cached && cached->packed == packed && cached->name == Py_None) {
            Py_INCREF(cached);
            return (PyObject *)cached;
        }
    }

    DimensionObject *self = (DimensionObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->packed = packed;
    Py_INCREF(name_str);
    self->name = name_str;

    /* Cache unnamed dimensions */
    if (name_str == Py_None) {
        unsigned int idx = (unsigned int)((uint64_t)packed % DIM_CACHE_SIZE);
        DimensionObject *old = dim_cache[idx];
        Py_INCREF(self);
        dim_cache[idx] = self;
        Py_XDECREF(old);
    }

    return (PyObject *)self;
}

static DimensionObject *Dimension_from_packed(int64_t packed) {
    /* Fast creation from packed value, uses cache */
    unsigned int idx = (unsigned int)((uint64_t)packed % DIM_CACHE_SIZE);
    DimensionObject *cached = dim_cache[idx];
    if (cached && cached->packed == packed && cached->name == Py_None) {
        Py_INCREF(cached);
        return cached;
    }

    DimensionObject *d = PyObject_New(DimensionObject, &DimensionType);
    if (!d) return NULL;
    d->packed = packed;
    Py_INCREF(Py_None);
    d->name = Py_None;

    DimensionObject *old = dim_cache[idx];
    Py_INCREF(d);
    dim_cache[idx] = d;
    Py_XDECREF(old);

    return d;
}

static PyObject *Dimension_repr(DimensionObject *self) {
    if (self->name != Py_None)
        return PyUnicode_FromFormat("%S", self->name);

    /* Build "m*kg*s^-2" style string */
    char buf[256];
    int pos = 0;
    int first = 1;
    for (int i = 0; i < 7; i++) {
        int8_t exp = dim_get(self->packed, i);
        if (exp == 0) continue;
        if (!first && pos < 250) buf[pos++] = '*';
        first = 0;
        if (exp == 1)
            pos += snprintf(buf + pos, sizeof(buf) - pos, "%s", DIM_LABELS[i]);
        else
            pos += snprintf(buf + pos, sizeof(buf) - pos, "%s^%d", DIM_LABELS[i], exp);
    }
    if (first)
        return PyUnicode_FromString("dimensionless");
    buf[pos] = '\0';
    return PyUnicode_FromString(buf);
}

static Py_hash_t Dimension_hash(DimensionObject *self) {
    Py_hash_t h = (Py_hash_t)self->packed;
    if (h == -1) h = -2;
    return h;
}

static PyObject *Dimension_richcompare(PyObject *a, PyObject *b, int op) {
    if (!PyObject_TypeCheck(a, &DimensionType) || !PyObject_TypeCheck(b, &DimensionType))
        Py_RETURN_NOTIMPLEMENTED;
    DimensionObject *da = (DimensionObject *)a;
    DimensionObject *db = (DimensionObject *)b;
    int eq = (da->packed == db->packed);
    if (op == Py_EQ) return PyBool_FromLong(eq);
    if (op == Py_NE) return PyBool_FromLong(!eq);
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject *Dimension_nb_multiply(PyObject *va, PyObject *vb) {
    if (!PyObject_TypeCheck(va, &DimensionType) || !PyObject_TypeCheck(vb, &DimensionType))
        Py_RETURN_NOTIMPLEMENTED;
    int64_t r = dim_add(((DimensionObject *)va)->packed, ((DimensionObject *)vb)->packed);
    return (PyObject *)Dimension_from_packed(r);
}

static PyObject *Dimension_nb_truediv(PyObject *va, PyObject *vb) {
    if (!PyObject_TypeCheck(va, &DimensionType) || !PyObject_TypeCheck(vb, &DimensionType))
        Py_RETURN_NOTIMPLEMENTED;
    int64_t r = dim_sub(((DimensionObject *)va)->packed, ((DimensionObject *)vb)->packed);
    return (PyObject *)Dimension_from_packed(r);
}

static PyObject *Dimension_nb_power(PyObject *base, PyObject *exp, PyObject *mod) {
    if (!PyObject_TypeCheck(base, &DimensionType) || !PyLong_Check(exp))
        Py_RETURN_NOTIMPLEMENTED;
    int n = (int)PyLong_AsLong(exp);
    int64_t r = dim_scale(((DimensionObject *)base)->packed, n);
    return (PyObject *)Dimension_from_packed(r);
}

static PyObject *Dimension_get_is_dimensionless(DimensionObject *self, void *closure) {
    return PyBool_FromLong(self->packed == 0);
}

/* Expose _exponents as a tuple for compatibility */
static PyObject *Dimension_get_exponents(DimensionObject *self, void *closure) {
    PyObject *t = PyTuple_New(7);
    if (!t) return NULL;
    for (int i = 0; i < 7; i++)
        PyTuple_SET_ITEM(t, i, PyLong_FromLong(dim_get(self->packed, i)));
    return t;
}

static PyGetSetDef Dimension_getset[] = {
    {"is_dimensionless", (getter)Dimension_get_is_dimensionless, NULL, NULL, NULL},
    {"_exponents", (getter)Dimension_get_exponents, NULL, NULL, NULL},
    {NULL}
};

static PyNumberMethods Dimension_as_number = {
    .nb_multiply = Dimension_nb_multiply,
    .nb_true_divide = Dimension_nb_truediv,
    .nb_power = Dimension_nb_power,
};

static PyTypeObject DimensionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "xue._units_accel.Dimension",
    .tp_basicsize = sizeof(DimensionObject),
    .tp_dealloc = (destructor)Dimension_dealloc,
    .tp_repr = (reprfunc)Dimension_repr,
    .tp_as_number = &Dimension_as_number,
    .tp_hash = (hashfunc)Dimension_hash,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_richcompare = Dimension_richcompare,
    .tp_getset = Dimension_getset,
    .tp_new = Dimension_new,
    .tp_doc = "Physical dimension with packed exponents.",
};

/* ══════════════════════════════════════════════════════════════════
   UnitObject
   ══════════════════════════════════════════════════════════════════ */

#define UNIT_CACHE_SIZE 256
static UnitObject *unit_cache[UNIT_CACHE_SIZE];

struct UnitObject {
    PyObject_HEAD
    DimensionObject *dimension;
    double scale;
    PyObject *name;
    PyObject *symbol;
};

static void Unit_dealloc(UnitObject *self) {
    Py_XDECREF(self->dimension);
    Py_XDECREF(self->name);
    Py_XDECREF(self->symbol);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Unit_new(PyTypeObject *type, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"dimension", "scale", "name", "symbol", NULL};
    DimensionObject *dim;
    double scale = 1.0;
    PyObject *name = NULL;
    PyObject *symbol = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O!|dOO", kwlist,
            &DimensionType, &dim, &scale, &name, &symbol))
        return NULL;

    UnitObject *self = (UnitObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;

    Py_INCREF(dim);
    self->dimension = dim;
    self->scale = scale;

    if (name && PyUnicode_Check(name)) { Py_INCREF(name); self->name = name; }
    else { Py_INCREF(Py_None); self->name = Py_None; }

    if (symbol && PyUnicode_Check(symbol)) { Py_INCREF(symbol); self->symbol = symbol; }
    else if (name && PyUnicode_Check(name)) { Py_INCREF(name); self->symbol = name; }
    else { Py_INCREF(Py_None); self->symbol = Py_None; }

    return (PyObject *)self;
}

static UnitObject *Unit_create(DimensionObject *dim, double scale, PyObject *symbol) {
    /* Fast internal creation */
    UnitObject *u = PyObject_New(UnitObject, &UnitType);
    if (!u) return NULL;
    Py_INCREF(dim);
    u->dimension = dim;
    u->scale = scale;
    Py_INCREF(symbol);
    u->name = symbol;
    Py_INCREF(symbol);
    u->symbol = symbol;
    return u;
}

static UnitObject *Unit_cached(int64_t dim_packed, double scale, PyObject *symbol) {
    /* Cache lookup by packed dim + scale */
    uint64_t h = (uint64_t)dim_packed;
    uint64_t sb;
    memcpy(&sb, &scale, sizeof(sb));
    h ^= sb;
    unsigned int idx = (unsigned int)(h % UNIT_CACHE_SIZE);

    UnitObject *cached = unit_cache[idx];
    if (cached && cached->dimension->packed == dim_packed && cached->scale == scale) {
        Py_INCREF(cached);
        return cached;
    }

    DimensionObject *dim = Dimension_from_packed(dim_packed);
    if (!dim) return NULL;
    UnitObject *u = Unit_create(dim, scale, symbol);
    Py_DECREF(dim);
    if (!u) return NULL;

    UnitObject *old = unit_cache[idx];
    Py_INCREF(u);
    unit_cache[idx] = u;
    Py_XDECREF(old);

    return u;
}

static PyObject *Unit_repr(UnitObject *self) {
    if (self->symbol != Py_None)
        return PyUnicode_FromFormat("%S", self->symbol);
    return Dimension_repr(self->dimension);
}

static Py_hash_t Unit_hash(UnitObject *self) {
    /* Round scale to 10 decimal places for hash stability */
    double rs = round(self->scale * 1e10) / 1e10;
    uint64_t sb;
    memcpy(&sb, &rs, sizeof(sb));
    Py_hash_t h = (Py_hash_t)(self->dimension->packed ^ (int64_t)sb);
    if (h == -1) h = -2;
    return h;
}

static PyObject *Unit_richcompare(PyObject *a, PyObject *b, int op) {
    if (!PyObject_TypeCheck(a, &UnitType) || !PyObject_TypeCheck(b, &UnitType))
        Py_RETURN_NOTIMPLEMENTED;
    UnitObject *ua = (UnitObject *)a;
    UnitObject *ub = (UnitObject *)b;
    int eq = (ua->dimension->packed == ub->dimension->packed) &&
             fabs(ua->scale - ub->scale) < 1e-9 * fmax(fabs(ua->scale), fabs(ub->scale));
    if (op == Py_EQ) return PyBool_FromLong(eq);
    if (op == Py_NE) return PyBool_FromLong(!eq);
    Py_RETURN_NOTIMPLEMENTED;
}

/* Unit * Unit → Unit,  scalar * Unit → Quantity,  Unit * scalar → Quantity */
static PyObject *Unit_nb_multiply(PyObject *va, PyObject *vb) {
    if (PyObject_TypeCheck(va, &UnitType) && PyObject_TypeCheck(vb, &UnitType)) {
        UnitObject *a = (UnitObject *)va, *b = (UnitObject *)vb;
        int64_t dp = dim_add(a->dimension->packed, b->dimension->packed);
        double sc = a->scale * b->scale;
        PyObject *sym = PyUnicode_FromFormat("%S*%S", a->symbol, b->symbol);
        UnitObject *r = Unit_cached(dp, sc, sym);
        Py_DECREF(sym);
        return (PyObject *)r;
    }
    /* scalar * Unit or Unit * scalar → Quantity */
    UnitObject *unit = NULL;
    double val = 0;
    if (PyObject_TypeCheck(va, &UnitType) && (PyFloat_Check(vb) || PyLong_Check(vb))) {
        unit = (UnitObject *)va;
        val = PyFloat_AsDouble(vb);
    } else if (PyObject_TypeCheck(vb, &UnitType) && (PyFloat_Check(va) || PyLong_Check(va))) {
        unit = (UnitObject *)vb;
        val = PyFloat_AsDouble(va);
    }
    if (unit) {
        if (PyErr_Occurred()) return NULL;
        QuantityObject *q = PyObject_New(QuantityObject, &QuantityType);
        if (!q) return NULL;
        q->value = val;
        Py_INCREF(unit);
        q->unit = unit;
        return (PyObject *)q;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject *Unit_nb_truediv(PyObject *va, PyObject *vb) {
    if (!PyObject_TypeCheck(va, &UnitType) || !PyObject_TypeCheck(vb, &UnitType))
        Py_RETURN_NOTIMPLEMENTED;
    UnitObject *a = (UnitObject *)va, *b = (UnitObject *)vb;
    int64_t dp = dim_sub(a->dimension->packed, b->dimension->packed);
    double sc = a->scale / b->scale;
    PyObject *sym = PyUnicode_FromFormat("%S/%S", a->symbol, b->symbol);
    UnitObject *r = Unit_cached(dp, sc, sym);
    Py_DECREF(sym);
    return (PyObject *)r;
}

static PyObject *Unit_nb_power(PyObject *base, PyObject *exp, PyObject *mod) {
    if (!PyObject_TypeCheck(base, &UnitType) || !PyLong_Check(exp))
        Py_RETURN_NOTIMPLEMENTED;
    UnitObject *u = (UnitObject *)base;
    int n = (int)PyLong_AsLong(exp);
    int64_t dp = dim_scale(u->dimension->packed, n);
    double sc = pow(u->scale, n);
    PyObject *sym = PyUnicode_FromFormat("%S^%d", u->symbol, n);
    UnitObject *r = Unit_cached(dp, sc, sym);
    Py_DECREF(sym);
    return (PyObject *)r;
}

static PyMemberDef Unit_members[] = {
    {"dimension", T_OBJECT_EX, offsetof(UnitObject, dimension), READONLY, NULL},
    {"scale", T_DOUBLE, offsetof(UnitObject, scale), READONLY, NULL},
    {"name", T_OBJECT_EX, offsetof(UnitObject, name), READONLY, NULL},
    {"symbol", T_OBJECT_EX, offsetof(UnitObject, symbol), READONLY, NULL},
    {NULL}
};

static PyNumberMethods Unit_as_number = {
    .nb_multiply = Unit_nb_multiply,
    .nb_true_divide = Unit_nb_truediv,
    .nb_power = Unit_nb_power,
};

static PyTypeObject UnitType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "xue._units_accel.Unit",
    .tp_basicsize = sizeof(UnitObject),
    .tp_dealloc = (destructor)Unit_dealloc,
    .tp_repr = (reprfunc)Unit_repr,
    .tp_as_number = &Unit_as_number,
    .tp_hash = (hashfunc)Unit_hash,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_richcompare = Unit_richcompare,
    .tp_members = Unit_members,
    .tp_new = Unit_new,
    .tp_doc = "Physical unit with dimension and scale.",
};

/* ══════════════════════════════════════════════════════════════════
   QuantityObject
   ══════════════════════════════════════════════════════════════════ */

/* (QuantityObject defined above in forward declarations) */

static void Quantity_dealloc(QuantityObject *self) {
    Py_XDECREF(self->unit);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Quantity_new(PyTypeObject *type, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"value", "unit", NULL};
    double value;
    UnitObject *unit;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "dO!", kwlist,
            &value, &UnitType, &unit))
        return NULL;

    QuantityObject *self = (QuantityObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->value = value;
    Py_INCREF(unit);
    self->unit = unit;
    return (PyObject *)self;
}

static inline QuantityObject *Quantity_fast(double value, UnitObject *unit) {
    QuantityObject *q = PyObject_New(QuantityObject, &QuantityType);
    if (!q) return NULL;
    q->value = value;
    Py_INCREF(unit);
    q->unit = unit;
    return q;
}

static int Quantity_check_compat(QuantityObject *a, QuantityObject *b) {
    if (a->unit->dimension->packed != b->unit->dimension->packed) {
        PyErr_Format(DimensionError,
            "Cannot combine %R with %R",
            (PyObject *)a->unit->dimension,
            (PyObject *)b->unit->dimension);
        return -1;
    }
    return 0;
}

static PyObject *Quantity_to(QuantityObject *self, PyObject *args) {
    UnitObject *target;
    if (!PyArg_ParseTuple(args, "O!", &UnitType, &target))
        return NULL;
    if (self->unit->dimension->packed != target->dimension->packed) {
        PyErr_Format(DimensionError,
            "Cannot convert %R to %R",
            (PyObject *)self->unit->dimension,
            (PyObject *)target->dimension);
        return NULL;
    }
    double result = self->value * (self->unit->scale / target->scale);
    return PyFloat_FromDouble(result);
}

static PyObject *Quantity_repr(QuantityObject *self) {
    return PyUnicode_FromFormat("Quantity(%R, %R)",
        PyFloat_FromDouble(self->value), (PyObject *)self->unit);
}

static PyObject *Quantity_format(QuantityObject *self, PyObject *args) {
    const char *spec = "";
    if (!PyArg_ParseTuple(args, "|s", &spec))
        return NULL;
    if (spec[0]) {
        PyObject *fmt = PyUnicode_FromFormat("{:%s}", spec);
        if (!fmt) return NULL;
        PyObject *val = PyFloat_FromDouble(self->value);
        PyObject *formatted = PyObject_Format(val, NULL);
        /* Use the spec properly */
        Py_DECREF(fmt);
        Py_DECREF(val);

        PyObject *spec_obj = PyUnicode_FromString(spec);
        val = PyFloat_FromDouble(self->value);
        formatted = PyObject_Format(val, spec_obj);
        Py_DECREF(spec_obj);
        Py_DECREF(val);
        if (!formatted) return NULL;
        PyObject *result = PyUnicode_FromFormat("%S %S", formatted, self->unit->symbol);
        Py_DECREF(formatted);
        return result;
    }
    return PyUnicode_FromFormat("%S %S",
        PyFloat_FromDouble(self->value), self->unit->symbol);
}

/* ── Quantity number methods ──────────────────────────────────────── */

static PyObject *Quantity_nb_add(PyObject *va, PyObject *vb) {
    if (!PyObject_TypeCheck(va, &QuantityType) || !PyObject_TypeCheck(vb, &QuantityType))
        Py_RETURN_NOTIMPLEMENTED;
    QuantityObject *a = (QuantityObject *)va, *b = (QuantityObject *)vb;
    if (Quantity_check_compat(a, b) < 0) return NULL;
    double converted = b->value * (b->unit->scale / a->unit->scale);
    return (PyObject *)Quantity_fast(a->value + converted, a->unit);
}

static PyObject *Quantity_nb_subtract(PyObject *va, PyObject *vb) {
    if (!PyObject_TypeCheck(va, &QuantityType) || !PyObject_TypeCheck(vb, &QuantityType))
        Py_RETURN_NOTIMPLEMENTED;
    QuantityObject *a = (QuantityObject *)va, *b = (QuantityObject *)vb;
    if (Quantity_check_compat(a, b) < 0) return NULL;
    double converted = b->value * (b->unit->scale / a->unit->scale);
    return (PyObject *)Quantity_fast(a->value - converted, a->unit);
}

static PyObject *Quantity_nb_multiply(PyObject *va, PyObject *vb) {
    /* Quantity * Quantity */
    if (PyObject_TypeCheck(va, &QuantityType) && PyObject_TypeCheck(vb, &QuantityType)) {
        QuantityObject *a = (QuantityObject *)va, *b = (QuantityObject *)vb;
        PyObject *new_unit = Unit_nb_multiply((PyObject *)a->unit, (PyObject *)b->unit);
        if (!new_unit) return NULL;
        QuantityObject *r = Quantity_fast(a->value * b->value, (UnitObject *)new_unit);
        Py_DECREF(new_unit);
        return (PyObject *)r;
    }
    /* Quantity * scalar or scalar * Quantity */
    QuantityObject *q = NULL;
    double s = 0;
    if (PyObject_TypeCheck(va, &QuantityType) && (PyFloat_Check(vb) || PyLong_Check(vb))) {
        q = (QuantityObject *)va;
        s = PyFloat_AsDouble(vb);
    } else if (PyObject_TypeCheck(vb, &QuantityType) && (PyFloat_Check(va) || PyLong_Check(va))) {
        q = (QuantityObject *)vb;
        s = PyFloat_AsDouble(va);
    }
    if (q) {
        if (PyErr_Occurred()) return NULL;
        return (PyObject *)Quantity_fast(q->value * s, q->unit);
    }
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject *Quantity_nb_truediv(PyObject *va, PyObject *vb) {
    /* Quantity / Quantity */
    if (PyObject_TypeCheck(va, &QuantityType) && PyObject_TypeCheck(vb, &QuantityType)) {
        QuantityObject *a = (QuantityObject *)va, *b = (QuantityObject *)vb;
        PyObject *new_unit = Unit_nb_truediv((PyObject *)a->unit, (PyObject *)b->unit);
        if (!new_unit) return NULL;
        QuantityObject *r = Quantity_fast(a->value / b->value, (UnitObject *)new_unit);
        Py_DECREF(new_unit);
        return (PyObject *)r;
    }
    /* Quantity / scalar */
    if (PyObject_TypeCheck(va, &QuantityType) && (PyFloat_Check(vb) || PyLong_Check(vb))) {
        QuantityObject *q = (QuantityObject *)va;
        double s = PyFloat_AsDouble(vb);
        if (PyErr_Occurred()) return NULL;
        return (PyObject *)Quantity_fast(q->value / s, q->unit);
    }
    /* scalar / Quantity */
    if (PyObject_TypeCheck(vb, &QuantityType) && (PyFloat_Check(va) || PyLong_Check(va))) {
        QuantityObject *q = (QuantityObject *)vb;
        double s = PyFloat_AsDouble(va);
        if (PyErr_Occurred()) return NULL;
        int64_t inv_dim = dim_negate(q->unit->dimension->packed);
        double inv_scale = 1.0 / q->unit->scale;
        PyObject *sym = PyUnicode_FromFormat("1/%S", q->unit->symbol);
        UnitObject *inv_unit = Unit_cached(inv_dim, inv_scale, sym);
        Py_DECREF(sym);
        if (!inv_unit) return NULL;
        QuantityObject *r = Quantity_fast(s / q->value, inv_unit);
        Py_DECREF(inv_unit);
        return (PyObject *)r;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject *Quantity_nb_power(PyObject *base, PyObject *exp, PyObject *mod) {
    if (!PyObject_TypeCheck(base, &QuantityType) || !PyLong_Check(exp))
        Py_RETURN_NOTIMPLEMENTED;
    QuantityObject *q = (QuantityObject *)base;
    int n = (int)PyLong_AsLong(exp);
    PyObject *new_unit = Unit_nb_power((PyObject *)q->unit, exp, Py_None);
    if (!new_unit) return NULL;
    QuantityObject *r = Quantity_fast(pow(q->value, n), (UnitObject *)new_unit);
    Py_DECREF(new_unit);
    return (PyObject *)r;
}

static PyObject *Quantity_nb_negative(PyObject *v) {
    QuantityObject *q = (QuantityObject *)v;
    return (PyObject *)Quantity_fast(-q->value, q->unit);
}

static PyObject *Quantity_nb_absolute(PyObject *v) {
    QuantityObject *q = (QuantityObject *)v;
    return (PyObject *)Quantity_fast(fabs(q->value), q->unit);
}

static PyObject *Quantity_richcompare(PyObject *va, PyObject *vb, int op) {
    if (!PyObject_TypeCheck(va, &QuantityType) || !PyObject_TypeCheck(vb, &QuantityType))
        Py_RETURN_NOTIMPLEMENTED;
    QuantityObject *a = (QuantityObject *)va, *b = (QuantityObject *)vb;

    if (op == Py_EQ) {
        if (a->unit->dimension->packed != b->unit->dimension->packed)
            Py_RETURN_FALSE;
        double av = a->value * a->unit->scale;
        double bv = b->value * b->unit->scale;
        return PyBool_FromLong(fabs(av - bv) <= 1e-9 * fmax(fabs(av), fmax(fabs(bv), 1e-15)));
    }
    if (op == Py_NE) {
        PyObject *eq = Quantity_richcompare(va, vb, Py_EQ);
        if (!eq) return NULL;
        int is_eq = (eq == Py_True);
        Py_DECREF(eq);
        return PyBool_FromLong(!is_eq);
    }

    if (Quantity_check_compat(a, b) < 0) return NULL;
    double av = a->value * a->unit->scale;
    double bv = b->value * b->unit->scale;
    int result;
    switch (op) {
        case Py_LT: result = av < bv; break;
        case Py_LE: result = av <= bv; break;
        case Py_GT: result = av > bv; break;
        case Py_GE: result = av >= bv; break;
        default: Py_RETURN_NOTIMPLEMENTED;
    }
    return PyBool_FromLong(result);
}

static PyMemberDef Quantity_members[] = {
    {"value", T_DOUBLE, offsetof(QuantityObject, value), READONLY, NULL},
    {"unit", T_OBJECT_EX, offsetof(QuantityObject, unit), READONLY, NULL},
    {NULL}
};

static PyMethodDef Quantity_methods[] = {
    {"to", (PyCFunction)Quantity_to, METH_VARARGS, "Convert to target unit."},
    {"__format__", (PyCFunction)Quantity_format, METH_VARARGS, NULL},
    {NULL}
};

static PyNumberMethods Quantity_as_number = {
    .nb_add = Quantity_nb_add,
    .nb_subtract = Quantity_nb_subtract,
    .nb_multiply = Quantity_nb_multiply,
    .nb_true_divide = Quantity_nb_truediv,
    .nb_power = Quantity_nb_power,
    .nb_negative = Quantity_nb_negative,
    .nb_absolute = Quantity_nb_absolute,
};

static PyTypeObject QuantityType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "xue._units_accel.Quantity",
    .tp_basicsize = sizeof(QuantityObject),
    .tp_dealloc = (destructor)Quantity_dealloc,
    .tp_repr = (reprfunc)Quantity_repr,
    .tp_as_number = &Quantity_as_number,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_richcompare = Quantity_richcompare,
    .tp_members = Quantity_members,
    .tp_methods = Quantity_methods,
    .tp_new = Quantity_new,
    .tp_doc = "Numeric value with physical unit.",
};

/* ══════════════════════════════════════════════════════════════════
   Module definition
   ══════════════════════════════════════════════════════════════════ */

static PyMethodDef module_methods[] = {
    {NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_units_accel",
    "C-accelerated physical units for xue",
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__units_accel(void) {
    if (PyType_Ready(&DimensionType) < 0) return NULL;
    if (PyType_Ready(&UnitType) < 0) return NULL;
    if (PyType_Ready(&QuantityType) < 0) return NULL;

    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;

    /* Create DimensionError */
    DimensionError = PyErr_NewException("xue._units_accel.DimensionError",
                                         PyExc_TypeError, NULL);
    if (!DimensionError) { Py_DECREF(m); return NULL; }
    Py_INCREF(DimensionError);
    PyModule_AddObject(m, "DimensionError", DimensionError);

    Py_INCREF(&DimensionType);
    PyModule_AddObject(m, "Dimension", (PyObject *)&DimensionType);
    Py_INCREF(&UnitType);
    PyModule_AddObject(m, "Unit", (PyObject *)&UnitType);
    Py_INCREF(&QuantityType);
    PyModule_AddObject(m, "Quantity", (PyObject *)&QuantityType);

    /* Clear caches */
    memset(dim_cache, 0, sizeof(dim_cache));
    memset(unit_cache, 0, sizeof(unit_cache));

    return m;
}
