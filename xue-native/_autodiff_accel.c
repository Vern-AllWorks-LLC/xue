/*
 * _autodiff_accel.c — C-accelerated reverse-mode automatic differentiation
 *
 * Uses tree-based computation graph with C-level topological sort and
 * switch-dispatch backward rules. No Python closures created per operation.
 *
 * Technique: standard Wengert list / reverse-mode AD (Speelpenning 1980).
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* ── Operation types ──────────────────────────────────────────────── */

typedef enum {
    OP_NONE = 0,
    OP_ADD,           /* a + b */
    OP_ADD_CONST,     /* a + c */
    OP_SUB,           /* a - b */
    OP_SUB_CONST,     /* a - c */
    OP_RSUB_CONST,    /* c - a */
    OP_MUL,           /* a * b */
    OP_MUL_CONST,     /* a * c */
    OP_DIV_CONST,     /* a / c */
    OP_RDIV_CONST,    /* c / a */
    OP_POW_CONST,     /* a ** c */
    OP_NEG,           /* -a */
    OP_EXP,           /* exp(a) */
    OP_LOG,           /* log(a) */
    OP_SIN,           /* sin(a) */
    OP_COS,           /* cos(a) */
    OP_TAN,           /* tan(a) */
    OP_TANH,          /* tanh(a) */
    OP_SIGMOID,       /* sigmoid(a) */
    OP_RELU,          /* relu(a) */
    OP_ABS,           /* abs(a) */
} OpType;

/* ── VariableObject ───────────────────────────────────────────────── */

typedef struct VariableObject {
    PyObject_HEAD
    double data;
    double grad;
    OpType op;
    struct VariableObject *child1;  /* first operand (may be NULL) */
    struct VariableObject *child2;  /* second operand (may be NULL) */
    double cached;                  /* cached constant or intermediate */
    int requires_grad;
    PyObject *name;                 /* str or None */
} VariableObject;

static PyTypeObject VariableType;

/* Forward declarations */
static VariableObject *Variable_create(double data, OpType op,
                                       VariableObject *c1, VariableObject *c2,
                                       double cached);
static VariableObject *ensure_variable(PyObject *x);

/* ── Fast allocation ──────────────────────────────────────────────── */

static VariableObject *Variable_create(double data, OpType op,
                                       VariableObject *c1, VariableObject *c2,
                                       double cached) {
    VariableObject *v = PyObject_New(VariableObject, &VariableType);
    if (!v) return NULL;
    v->data = data;
    v->grad = 0.0;
    v->op = op;
    v->child1 = c1;  Py_XINCREF(c1);
    v->child2 = c2;  Py_XINCREF(c2);
    v->cached = cached;
    v->requires_grad = 1;
    Py_INCREF(Py_None);
    v->name = Py_None;
    return v;
}

static VariableObject *ensure_variable(PyObject *x) {
    if (PyObject_TypeCheck(x, &VariableType)) {
        Py_INCREF(x);
        return (VariableObject *)x;
    }
    double val = PyFloat_AsDouble(x);
    if (PyErr_Occurred()) return NULL;
    VariableObject *v = Variable_create(val, OP_NONE, NULL, NULL, 0.0);
    if (v) v->requires_grad = 0;
    return v;
}

/* ── Type methods ─────────────────────────────────────────────────── */

static void Variable_dealloc(VariableObject *self) {
    Py_XDECREF(self->child1);
    Py_XDECREF(self->child2);
    Py_XDECREF(self->name);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Variable_new(PyTypeObject *type, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"data", "name", "requires_grad", NULL};
    double data;
    PyObject *name = Py_None;
    int requires_grad = 1;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "d|Op", kwlist,
            &data, &name, &requires_grad))
        return NULL;

    VariableObject *self = (VariableObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->data = data;
    self->grad = 0.0;
    self->op = OP_NONE;
    self->child1 = NULL;
    self->child2 = NULL;
    self->cached = 0.0;
    self->requires_grad = requires_grad;
    Py_INCREF(name);
    self->name = name;
    return (PyObject *)self;
}

static PyObject *Variable_repr(VariableObject *self) {
    if (self->name != Py_None)
        return PyUnicode_FromFormat("Variable(%g, name=%R)", self->data, self->name);
    return PyUnicode_FromFormat("Variable(%g)", self->data);
}

static PyObject *Variable_float(VariableObject *self) {
    return PyFloat_FromDouble(self->data);
}

static PyObject *Variable_int(VariableObject *self) {
    return PyLong_FromDouble(self->data);
}

/* ── Backward pass ────────────────────────────────────────────────── */

/*
 * C-level topological sort using an explicit stack.
 * Avoids Python recursion and set/list overhead.
 */
typedef struct {
    VariableObject **items;
    Py_ssize_t count;
    Py_ssize_t capacity;
} VarList;

static int varlist_init(VarList *vl, Py_ssize_t cap) {
    vl->items = (VariableObject **)PyMem_Malloc(sizeof(VariableObject *) * cap);
    if (!vl->items) { PyErr_NoMemory(); return -1; }
    vl->count = 0;
    vl->capacity = cap;
    return 0;
}

static int varlist_append(VarList *vl, VariableObject *v) {
    if (vl->count >= vl->capacity) {
        Py_ssize_t newcap = vl->capacity * 2;
        VariableObject **newitems = (VariableObject **)PyMem_Realloc(
            vl->items, sizeof(VariableObject *) * newcap);
        if (!newitems) { PyErr_NoMemory(); return -1; }
        vl->items = newitems;
        vl->capacity = newcap;
    }
    vl->items[vl->count++] = v;
    return 0;
}

static void varlist_free(VarList *vl) {
    PyMem_Free(vl->items);
}

/*
 * Iterative topological sort using an explicit stack.
 * Each stack frame tracks: node, phase (0=enter, 1=child1 done, 2=child2 done).
 */
typedef struct {
    VariableObject *node;
    int phase;
} TopoFrame;

static int build_topo(VariableObject *root, VarList *topo) {
    /* Use a hash set for visited (based on pointer identity) */
    PyObject *visited = PySet_New(NULL);
    if (!visited) return -1;

    /* Explicit stack */
    Py_ssize_t stack_cap = 64;
    TopoFrame *stack = (TopoFrame *)PyMem_Malloc(sizeof(TopoFrame) * stack_cap);
    if (!stack) { Py_DECREF(visited); PyErr_NoMemory(); return -1; }
    Py_ssize_t stack_top = 0;

    stack[stack_top++] = (TopoFrame){root, 0};

    while (stack_top > 0) {
        TopoFrame *frame = &stack[stack_top - 1];
        VariableObject *node = frame->node;

        if (frame->phase == 0) {
            /* Check if visited */
            PyObject *key = PyLong_FromVoidPtr(node);
            if (!key) goto error;
            int in_set = PySet_Contains(visited, key);
            if (in_set) {
                Py_DECREF(key);
                stack_top--;
                continue;
            }
            PySet_Add(visited, key);
            Py_DECREF(key);
            frame->phase = 1;

            /* Push child1 if exists */
            if (node->child1) {
                if (stack_top >= stack_cap) {
                    stack_cap *= 2;
                    TopoFrame *ns = (TopoFrame *)PyMem_Realloc(stack, sizeof(TopoFrame) * stack_cap);
                    if (!ns) { PyErr_NoMemory(); goto error; }
                    stack = ns;
                    frame = &stack[stack_top - 1]; /* realloc may move */
                }
                stack[stack_top++] = (TopoFrame){node->child1, 0};
            }
        } else if (frame->phase == 1) {
            frame->phase = 2;
            /* Push child2 if exists */
            if (node->child2) {
                if (stack_top >= stack_cap) {
                    stack_cap *= 2;
                    TopoFrame *ns = (TopoFrame *)PyMem_Realloc(stack, sizeof(TopoFrame) * stack_cap);
                    if (!ns) { PyErr_NoMemory(); goto error; }
                    stack = ns;
                    frame = &stack[stack_top - 1];
                }
                stack[stack_top++] = (TopoFrame){node->child2, 0};
            }
        } else {
            /* phase == 2: both children done, add to topo */
            if (varlist_append(topo, node) < 0) goto error;
            stack_top--;
        }
    }

    PyMem_Free(stack);
    Py_DECREF(visited);
    return 0;

error:
    PyMem_Free(stack);
    Py_DECREF(visited);
    return -1;
}

/*
 * Apply backward rules based on op type.
 * This is the key performance win: no Python function calls, just a switch.
 */
static void apply_backward(VariableObject *v) {
    double g = v->grad;
    VariableObject *c1 = v->child1;
    VariableObject *c2 = v->child2;

    switch (v->op) {
    case OP_NONE:
        break;
    case OP_ADD:
        c1->grad += g;
        c2->grad += g;
        break;
    case OP_ADD_CONST:
        c1->grad += g;
        break;
    case OP_SUB:
        c1->grad += g;
        c2->grad += -g;
        break;
    case OP_SUB_CONST:
        c1->grad += g;
        break;
    case OP_RSUB_CONST:
        c1->grad += -g;
        break;
    case OP_MUL:
        c1->grad += c2->data * g;
        c2->grad += c1->data * g;
        break;
    case OP_MUL_CONST:
        c1->grad += v->cached * g;  /* cached = constant multiplier */
        break;
    case OP_DIV_CONST:
        c1->grad += g / v->cached;
        break;
    case OP_RDIV_CONST:
        /* c / self: d/dself = -c / self^2 */
        c1->grad += -v->cached / (c1->data * c1->data) * g;
        break;
    case OP_POW_CONST:
        /* self^n: d/dself = n * self^(n-1) */
        c1->grad += v->cached * pow(c1->data, v->cached - 1.0) * g;
        break;
    case OP_NEG:
        c1->grad += -g;
        break;
    case OP_EXP:
        c1->grad += v->data * g;  /* exp(x)' = exp(x) */
        break;
    case OP_LOG:
        c1->grad += (1.0 / c1->data) * g;
        break;
    case OP_SIN:
        c1->grad += cos(c1->data) * g;
        break;
    case OP_COS:
        c1->grad += -sin(c1->data) * g;
        break;
    case OP_TAN:
        {
            double c = cos(c1->data);
            c1->grad += (1.0 / (c * c)) * g;
        }
        break;
    case OP_TANH:
        /* cached = tanh(x) */
        c1->grad += (1.0 - v->cached * v->cached) * g;
        break;
    case OP_SIGMOID:
        /* cached = sigmoid(x) */
        c1->grad += v->cached * (1.0 - v->cached) * g;
        break;
    case OP_RELU:
        c1->grad += (c1->data > 0 ? 1.0 : 0.0) * g;
        break;
    case OP_ABS:
        c1->grad += (c1->data >= 0 ? 1.0 : -1.0) * g;
        break;
    }
}

static PyObject *Variable_backward(VariableObject *self, PyObject *Py_UNUSED(args)) {
    VarList topo;
    if (varlist_init(&topo, 64) < 0) return NULL;
    if (build_topo(self, &topo) < 0) {
        varlist_free(&topo);
        return NULL;
    }

    /* Reset all grads */
    for (Py_ssize_t i = 0; i < topo.count; i++)
        topo.items[i]->grad = 0.0;

    /* Seed */
    self->grad = 1.0;

    /* Reverse traversal */
    for (Py_ssize_t i = topo.count - 1; i >= 0; i--)
        apply_backward(topo.items[i]);

    varlist_free(&topo);
    Py_RETURN_NONE;
}

static PyObject *Variable_zero_grad(VariableObject *self, PyObject *Py_UNUSED(args)) {
    self->grad = 0.0;
    Py_RETURN_NONE;
}

/* ── Arithmetic operators ─────────────────────────────────────────── */

static PyObject *Variable_nb_add(PyObject *va, PyObject *vb) {
    /* Variable + scalar */
    if (PyObject_TypeCheck(va, &VariableType) && (PyFloat_Check(vb) || PyLong_Check(vb))) {
        VariableObject *a = (VariableObject *)va;
        double c = PyFloat_AsDouble(vb);
        return (PyObject *)Variable_create(a->data + c, OP_ADD_CONST, a, NULL, c);
    }
    /* scalar + Variable */
    if ((PyFloat_Check(va) || PyLong_Check(va)) && PyObject_TypeCheck(vb, &VariableType)) {
        VariableObject *b = (VariableObject *)vb;
        double c = PyFloat_AsDouble(va);
        return (PyObject *)Variable_create(b->data + c, OP_ADD_CONST, b, NULL, c);
    }
    /* Variable + Variable */
    if (PyObject_TypeCheck(va, &VariableType) && PyObject_TypeCheck(vb, &VariableType)) {
        VariableObject *a = (VariableObject *)va, *b = (VariableObject *)vb;
        return (PyObject *)Variable_create(a->data + b->data, OP_ADD, a, b, 0.0);
    }
    /* Variable + other Variable-like */
    if (PyObject_TypeCheck(va, &VariableType)) {
        VariableObject *b = ensure_variable(vb);
        if (!b) Py_RETURN_NOTIMPLEMENTED;
        VariableObject *a = (VariableObject *)va;
        VariableObject *r = Variable_create(a->data + b->data, OP_ADD, a, b, 0.0);
        Py_DECREF(b);
        return (PyObject *)r;
    }
    if (PyObject_TypeCheck(vb, &VariableType)) {
        VariableObject *a = ensure_variable(va);
        if (!a) Py_RETURN_NOTIMPLEMENTED;
        VariableObject *b = (VariableObject *)vb;
        VariableObject *r = Variable_create(a->data + b->data, OP_ADD, a, b, 0.0);
        Py_DECREF(a);
        return (PyObject *)r;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject *Variable_nb_subtract(PyObject *va, PyObject *vb) {
    /* Variable - scalar */
    if (PyObject_TypeCheck(va, &VariableType) && (PyFloat_Check(vb) || PyLong_Check(vb))) {
        VariableObject *a = (VariableObject *)va;
        double c = PyFloat_AsDouble(vb);
        return (PyObject *)Variable_create(a->data - c, OP_SUB_CONST, a, NULL, c);
    }
    /* scalar - Variable */
    if ((PyFloat_Check(va) || PyLong_Check(va)) && PyObject_TypeCheck(vb, &VariableType)) {
        VariableObject *b = (VariableObject *)vb;
        double c = PyFloat_AsDouble(va);
        return (PyObject *)Variable_create(c - b->data, OP_RSUB_CONST, b, NULL, c);
    }
    /* Variable - Variable */
    if (PyObject_TypeCheck(va, &VariableType) && PyObject_TypeCheck(vb, &VariableType)) {
        VariableObject *a = (VariableObject *)va, *b = (VariableObject *)vb;
        return (PyObject *)Variable_create(a->data - b->data, OP_SUB, a, b, 0.0);
    }
    if (PyObject_TypeCheck(va, &VariableType)) {
        VariableObject *b = ensure_variable(vb);
        if (!b) Py_RETURN_NOTIMPLEMENTED;
        VariableObject *a = (VariableObject *)va;
        VariableObject *r = Variable_create(a->data - b->data, OP_SUB, a, b, 0.0);
        Py_DECREF(b);
        return (PyObject *)r;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject *Variable_nb_multiply(PyObject *va, PyObject *vb) {
    /* Variable * scalar */
    if (PyObject_TypeCheck(va, &VariableType) && (PyFloat_Check(vb) || PyLong_Check(vb))) {
        VariableObject *a = (VariableObject *)va;
        double c = PyFloat_AsDouble(vb);
        return (PyObject *)Variable_create(a->data * c, OP_MUL_CONST, a, NULL, c);
    }
    /* scalar * Variable */
    if ((PyFloat_Check(va) || PyLong_Check(va)) && PyObject_TypeCheck(vb, &VariableType)) {
        VariableObject *b = (VariableObject *)vb;
        double c = PyFloat_AsDouble(va);
        return (PyObject *)Variable_create(b->data * c, OP_MUL_CONST, b, NULL, c);
    }
    /* Variable * Variable */
    if (PyObject_TypeCheck(va, &VariableType) && PyObject_TypeCheck(vb, &VariableType)) {
        VariableObject *a = (VariableObject *)va, *b = (VariableObject *)vb;
        return (PyObject *)Variable_create(a->data * b->data, OP_MUL, a, b, 0.0);
    }
    if (PyObject_TypeCheck(va, &VariableType)) {
        VariableObject *b = ensure_variable(vb);
        if (!b) Py_RETURN_NOTIMPLEMENTED;
        VariableObject *a = (VariableObject *)va;
        VariableObject *r = Variable_create(a->data * b->data, OP_MUL, a, b, 0.0);
        Py_DECREF(b);
        return (PyObject *)r;
    }
    if (PyObject_TypeCheck(vb, &VariableType)) {
        VariableObject *a = ensure_variable(va);
        if (!a) Py_RETURN_NOTIMPLEMENTED;
        VariableObject *b = (VariableObject *)vb;
        VariableObject *r = Variable_create(a->data * b->data, OP_MUL, a, b, 0.0);
        Py_DECREF(a);
        return (PyObject *)r;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject *Variable_nb_truediv(PyObject *va, PyObject *vb) {
    /* Variable / scalar */
    if (PyObject_TypeCheck(va, &VariableType) && (PyFloat_Check(vb) || PyLong_Check(vb))) {
        VariableObject *a = (VariableObject *)va;
        double c = PyFloat_AsDouble(vb);
        return (PyObject *)Variable_create(a->data / c, OP_DIV_CONST, a, NULL, c);
    }
    /* scalar / Variable */
    if ((PyFloat_Check(va) || PyLong_Check(va)) && PyObject_TypeCheck(vb, &VariableType)) {
        VariableObject *b = (VariableObject *)vb;
        double c = PyFloat_AsDouble(va);
        return (PyObject *)Variable_create(c / b->data, OP_RDIV_CONST, b, NULL, c);
    }
    /* Variable / Variable → self * (other ** -1) */
    if (PyObject_TypeCheck(va, &VariableType) && PyObject_TypeCheck(vb, &VariableType)) {
        VariableObject *b = (VariableObject *)vb;
        /* b^(-1) */
        VariableObject *inv = Variable_create(1.0 / b->data, OP_POW_CONST, b, NULL, -1.0);
        if (!inv) return NULL;
        VariableObject *a = (VariableObject *)va;
        VariableObject *r = Variable_create(a->data * inv->data, OP_MUL, a, inv, 0.0);
        Py_DECREF(inv);
        return (PyObject *)r;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject *Variable_nb_power(PyObject *base, PyObject *exponent, PyObject *mod) {
    if (!PyObject_TypeCheck(base, &VariableType))
        Py_RETURN_NOTIMPLEMENTED;
    VariableObject *a = (VariableObject *)base;

    if (PyObject_TypeCheck(exponent, &VariableType)) {
        /* x^y = exp(y * ln(x)) — compose existing ops */
        VariableObject *y = (VariableObject *)exponent;
        /* log(x) */
        VariableObject *lnx = Variable_create(log(a->data), OP_LOG, a, NULL, 0.0);
        if (!lnx) return NULL;
        /* y * ln(x) */
        VariableObject *ylnx = Variable_create(y->data * lnx->data, OP_MUL, y, lnx, 0.0);
        Py_DECREF(lnx);
        if (!ylnx) return NULL;
        /* exp(y * ln(x)) */
        VariableObject *r = Variable_create(exp(ylnx->data), OP_EXP, ylnx, NULL, 0.0);
        Py_DECREF(ylnx);
        return (PyObject *)r;
    }

    double n = PyFloat_AsDouble(exponent);
    if (PyErr_Occurred()) return NULL;
    return (PyObject *)Variable_create(pow(a->data, n), OP_POW_CONST, a, NULL, n);
}

static PyObject *Variable_nb_negative(PyObject *v) {
    VariableObject *a = (VariableObject *)v;
    return (PyObject *)Variable_create(-a->data, OP_NEG, a, NULL, 0.0);
}

/* ── Math methods ─────────────────────────────────────────────────── */

static PyObject *Variable_exp(VariableObject *self, PyObject *Py_UNUSED(args)) {
    double val = exp(self->data);
    return (PyObject *)Variable_create(val, OP_EXP, self, NULL, 0.0);
}

static PyObject *Variable_log(VariableObject *self, PyObject *Py_UNUSED(args)) {
    return (PyObject *)Variable_create(log(self->data), OP_LOG, self, NULL, 0.0);
}

static PyObject *Variable_sin(VariableObject *self, PyObject *Py_UNUSED(args)) {
    return (PyObject *)Variable_create(sin(self->data), OP_SIN, self, NULL, 0.0);
}

static PyObject *Variable_cos(VariableObject *self, PyObject *Py_UNUSED(args)) {
    return (PyObject *)Variable_create(cos(self->data), OP_COS, self, NULL, 0.0);
}

static PyObject *Variable_tan(VariableObject *self, PyObject *Py_UNUSED(args)) {
    return (PyObject *)Variable_create(tan(self->data), OP_TAN, self, NULL, 0.0);
}

static PyObject *Variable_tanh(VariableObject *self, PyObject *Py_UNUSED(args)) {
    double t = tanh(self->data);
    return (PyObject *)Variable_create(t, OP_TANH, self, NULL, t);
}

static PyObject *Variable_sigmoid(VariableObject *self, PyObject *Py_UNUSED(args)) {
    double s = 1.0 / (1.0 + exp(-self->data));
    return (PyObject *)Variable_create(s, OP_SIGMOID, self, NULL, s);
}

static PyObject *Variable_relu(VariableObject *self, PyObject *Py_UNUSED(args)) {
    double val = self->data > 0 ? self->data : 0.0;
    return (PyObject *)Variable_create(val, OP_RELU, self, NULL, 0.0);
}

static PyObject *Variable_abs(VariableObject *self, PyObject *Py_UNUSED(args)) {
    return (PyObject *)Variable_create(fabs(self->data), OP_ABS, self, NULL, 0.0);
}

static PyObject *Variable_sqrt(VariableObject *self, PyObject *Py_UNUSED(args)) {
    /* sqrt(x) = x^0.5 */
    return (PyObject *)Variable_create(sqrt(self->data), OP_POW_CONST, self, NULL, 0.5);
}

/* ── Comparisons ──────────────────────────────────────────────────── */

static PyObject *Variable_richcompare(PyObject *va, PyObject *vb, int op) {
    double a_data, b_data;

    if (PyObject_TypeCheck(va, &VariableType))
        a_data = ((VariableObject *)va)->data;
    else if (PyFloat_Check(va) || PyLong_Check(va))
        a_data = PyFloat_AsDouble(va);
    else
        Py_RETURN_NOTIMPLEMENTED;

    if (PyObject_TypeCheck(vb, &VariableType))
        b_data = ((VariableObject *)vb)->data;
    else if (PyFloat_Check(vb) || PyLong_Check(vb))
        b_data = PyFloat_AsDouble(vb);
    else
        Py_RETURN_NOTIMPLEMENTED;

    int result;
    switch (op) {
    case Py_LT: result = a_data < b_data; break;
    case Py_LE: result = a_data <= b_data; break;
    case Py_GT: result = a_data > b_data; break;
    case Py_GE: result = a_data >= b_data; break;
    case Py_EQ: result = fabs(a_data - b_data) < 1e-9; break;
    case Py_NE: result = fabs(a_data - b_data) >= 1e-9; break;
    default: Py_RETURN_NOTIMPLEMENTED;
    }
    return PyBool_FromLong(result);
}

/* ── Members & methods ────────────────────────────────────────────── */

static PyMemberDef Variable_members[] = {
    {"data", T_DOUBLE, offsetof(VariableObject, data), 0, NULL},
    {"grad", T_DOUBLE, offsetof(VariableObject, grad), 0, NULL},
    {"name", T_OBJECT_EX, offsetof(VariableObject, name), 0, NULL},
    {"_requires_grad", T_BOOL, offsetof(VariableObject, requires_grad), 0, NULL},
    {NULL}
};

/* _children property for compatibility */
static PyObject *Variable_get_children(VariableObject *self, void *closure) {
    if (self->child1 && self->child2) {
        return Py_BuildValue("(OO)", self->child1, self->child2);
    } else if (self->child1) {
        return Py_BuildValue("(O,)", self->child1);
    }
    return PyTuple_New(0);
}

static PyGetSetDef Variable_getset[] = {
    {"_children", (getter)Variable_get_children, NULL, NULL, NULL},
    {NULL}
};

static PyMethodDef Variable_methods[] = {
    {"backward", (PyCFunction)Variable_backward, METH_NOARGS, "Reverse-mode AD backward pass."},
    {"zero_grad", (PyCFunction)Variable_zero_grad, METH_NOARGS, "Reset gradient to zero."},
    {"exp", (PyCFunction)Variable_exp, METH_NOARGS, NULL},
    {"log", (PyCFunction)Variable_log, METH_NOARGS, NULL},
    {"sin", (PyCFunction)Variable_sin, METH_NOARGS, NULL},
    {"cos", (PyCFunction)Variable_cos, METH_NOARGS, NULL},
    {"tan", (PyCFunction)Variable_tan, METH_NOARGS, NULL},
    {"tanh", (PyCFunction)Variable_tanh, METH_NOARGS, NULL},
    {"sigmoid", (PyCFunction)Variable_sigmoid, METH_NOARGS, NULL},
    {"relu", (PyCFunction)Variable_relu, METH_NOARGS, NULL},
    {"abs", (PyCFunction)Variable_abs, METH_NOARGS, NULL},
    {"sqrt", (PyCFunction)Variable_sqrt, METH_NOARGS, NULL},
    {NULL}
};

static PyNumberMethods Variable_as_number = {
    .nb_add = Variable_nb_add,
    .nb_subtract = Variable_nb_subtract,
    .nb_multiply = Variable_nb_multiply,
    .nb_true_divide = Variable_nb_truediv,
    .nb_power = Variable_nb_power,
    .nb_negative = Variable_nb_negative,
    .nb_float = (unaryfunc)Variable_float,
    .nb_int = (unaryfunc)Variable_int,
};

static PyTypeObject VariableType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "xue._autodiff_accel.Variable",
    .tp_basicsize = sizeof(VariableObject),
    .tp_dealloc = (destructor)Variable_dealloc,
    .tp_repr = (reprfunc)Variable_repr,
    .tp_as_number = &Variable_as_number,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_richcompare = Variable_richcompare,
    .tp_members = Variable_members,
    .tp_methods = Variable_methods,
    .tp_getset = Variable_getset,
    .tp_new = Variable_new,
    .tp_doc = "Differentiable variable with C-accelerated backward pass.",
};

/* ══════════════════════════════════════════════════════════════════
   Module-level functions: grad, value_and_grad, jacobian
   ══════════════════════════════════════════════════════════════════ */

static PyObject *mod_grad(PyObject *self, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"f", "argnum", NULL};
    PyObject *f;
    int argnum = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|i", kwlist, &f, &argnum))
        return NULL;

    /* Build a closure that captures f and argnum */
    /* We'll return a Python lambda that calls our C logic */
    /* For simplicity, store f and argnum in a capsule and create a wrapper */

    /* Alternative: return a callable object */
    /* For now, use a simple approach with a Python wrapper */

    /* Create a tuple (f, argnum) and attach it to a PyCFunction */
    PyObject *closure = Py_BuildValue("(Oi)", f, argnum);
    if (!closure) return NULL;

    /* We need a callable. Use a simple approach: create a partial-like object.
       For maximum C performance, implement a custom callable type. */

    /* Simple approach: define a _grad_call function and use PyCFunction with closure */
    /* Actually, the cleanest way is a GradFunc type */

    /* For simplicity, return a Python-level closure via PyRun_String */
    /* Better: implement a C callable type */

    /* Let's just return the closure tuple and have Python wrap it.
       Actually, let's implement a proper C callable. */
    Py_DECREF(closure);

    /* Return a _GradCallable instance */
    /* For now, implement inline. We need f and argnum. */

    /* Simplest correct approach: build a lambda using Python */
    PyObject *mod = PyImport_ImportModule("xue._autodiff_accel");
    if (!mod) return NULL;

    /* Store f and argnum as default args in a nested function */
    PyObject *code = PyUnicode_FromFormat(
        "def _make_grad(f, argnum):\n"
        "    from xue._autodiff_accel import Variable as _V\n"
        "    def grad_fn(*args, **kwargs):\n"
        "        new_args = list(args)\n"
        "        x = _V(float(args[argnum]), name='arg%%d' %% argnum)\n"
        "        new_args[argnum] = x\n"
        "        result = f(*new_args, **kwargs)\n"
        "        if isinstance(result, _V):\n"
        "            result.backward()\n"
        "            return x.grad\n"
        "        return 0.0\n"
        "    grad_fn.__name__ = 'grad(%%s)' %% f.__name__\n"
        "    grad_fn.__qualname__ = grad_fn.__name__\n"
        "    return grad_fn\n"
    );
    Py_DECREF(mod);
    Py_DECREF(code);

    /* Actually this eval approach is ugly. Let me do it properly in C. */
    /* I'll create a simple callable type. */

    /* Forget it — for grad/value_and_grad/jacobian, the overhead is in the
       computation, not in the function call. Just implement them as Python
       wrappers that use the C Variable type. The key win is the C Variable. */

    Py_RETURN_NONE;  /* placeholder — will be overridden by Python wrapper */
}

/* ══════════════════════════════════════════════════════════════════
   GradCallable — C callable for grad() return value
   ══════════════════════════════════════════════════════════════════ */

typedef struct {
    PyObject_HEAD
    PyObject *func;
    int argnum;
} GradCallableObject;

static PyTypeObject GradCallableType;

static void GradCallable_dealloc(GradCallableObject *self) {
    Py_XDECREF(self->func);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *GradCallable_call(GradCallableObject *self,
                                    PyObject *args, PyObject *kwargs) {
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (self->argnum >= nargs) {
        PyErr_Format(PyExc_IndexError, "argnum %d out of range (got %zd args)",
                     self->argnum, nargs);
        return NULL;
    }

    /* Build new args with Variable at argnum position */
    PyObject *new_args = PyList_New(nargs);
    if (!new_args) return NULL;

    for (Py_ssize_t i = 0; i < nargs; i++) {
        PyObject *item = PyTuple_GET_ITEM(args, i);
        Py_INCREF(item);
        PyList_SET_ITEM(new_args, i, item);
    }

    /* Create Variable for the target argument */
    double val = PyFloat_AsDouble(PyTuple_GET_ITEM(args, self->argnum));
    if (PyErr_Occurred()) { Py_DECREF(new_args); return NULL; }

    PyObject *name = PyUnicode_FromFormat("arg%d", self->argnum);
    VariableObject *x = (VariableObject *)Variable_new(&VariableType,
        Py_BuildValue("(d)", val), NULL);
    if (!x) { Py_DECREF(new_args); Py_XDECREF(name); return NULL; }
    Py_XDECREF(x->name);
    x->name = name;

    /* Replace in args list */
    Py_DECREF(PyList_GET_ITEM(new_args, self->argnum));
    Py_INCREF(x);
    PyList_SET_ITEM(new_args, self->argnum, (PyObject *)x);

    /* Convert to tuple for call */
    PyObject *call_args = PyList_AsTuple(new_args);
    Py_DECREF(new_args);
    if (!call_args) { Py_DECREF(x); return NULL; }

    /* Call the function */
    PyObject *result = PyObject_Call(self->func, call_args, kwargs);
    Py_DECREF(call_args);

    if (!result) { Py_DECREF(x); return NULL; }

    if (PyObject_TypeCheck(result, &VariableType)) {
        VariableObject *rv = (VariableObject *)result;
        /* Call backward in C — no Python overhead */
        VarList topo;
        if (varlist_init(&topo, 64) < 0) {
            Py_DECREF(result);
            Py_DECREF(x);
            return NULL;
        }
        if (build_topo(rv, &topo) < 0) {
            varlist_free(&topo);
            Py_DECREF(result);
            Py_DECREF(x);
            return NULL;
        }
        for (Py_ssize_t i = 0; i < topo.count; i++)
            topo.items[i]->grad = 0.0;
        rv->grad = 1.0;
        for (Py_ssize_t i = topo.count - 1; i >= 0; i--)
            apply_backward(topo.items[i]);
        varlist_free(&topo);

        double grad_val = x->grad;
        Py_DECREF(result);
        Py_DECREF(x);
        return PyFloat_FromDouble(grad_val);
    }

    Py_DECREF(result);
    Py_DECREF(x);
    return PyFloat_FromDouble(0.0);
}

static PyObject *GradCallable_get_name(GradCallableObject *self, void *closure) {
    PyObject *fname = PyObject_GetAttrString(self->func, "__name__");
    if (!fname) { PyErr_Clear(); fname = PyUnicode_FromString("<unknown>"); }
    PyObject *result = PyUnicode_FromFormat("grad(%S)", fname);
    Py_DECREF(fname);
    return result;
}

static PyGetSetDef GradCallable_getset[] = {
    {"__name__", (getter)GradCallable_get_name, NULL, NULL, NULL},
    {"__qualname__", (getter)GradCallable_get_name, NULL, NULL, NULL},
    {NULL}
};

static PyTypeObject GradCallableType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "xue._autodiff_accel._GradCallable",
    .tp_basicsize = sizeof(GradCallableObject),
    .tp_dealloc = (destructor)GradCallable_dealloc,
    .tp_call = (ternaryfunc)GradCallable_call,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = GradCallable_getset,
};

/* ── ValueAndGrad callable ────────────────────────────────────────── */

typedef struct {
    PyObject_HEAD
    PyObject *func;
    int argnum;
} ValueAndGradCallableObject;

static PyTypeObject ValueAndGradCallableType;

static void VAGCallable_dealloc(ValueAndGradCallableObject *self) {
    Py_XDECREF(self->func);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *VAGCallable_call(ValueAndGradCallableObject *self,
                                   PyObject *args, PyObject *kwargs) {
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (self->argnum >= nargs) {
        PyErr_Format(PyExc_IndexError, "argnum %d out of range", self->argnum);
        return NULL;
    }

    double val = PyFloat_AsDouble(PyTuple_GET_ITEM(args, self->argnum));
    if (PyErr_Occurred()) return NULL;

    PyObject *name = PyUnicode_FromFormat("arg%d", self->argnum);
    VariableObject *x = (VariableObject *)Variable_new(&VariableType,
        Py_BuildValue("(d)", val), NULL);
    if (!x) { Py_XDECREF(name); return NULL; }
    Py_XDECREF(x->name);
    x->name = name;

    /* Build call args */
    PyObject *new_args = PyList_New(nargs);
    for (Py_ssize_t i = 0; i < nargs; i++) {
        PyObject *item = PyTuple_GET_ITEM(args, i);
        Py_INCREF(item);
        PyList_SET_ITEM(new_args, i, item);
    }
    Py_DECREF(PyList_GET_ITEM(new_args, self->argnum));
    Py_INCREF(x);
    PyList_SET_ITEM(new_args, self->argnum, (PyObject *)x);

    PyObject *call_args = PyList_AsTuple(new_args);
    Py_DECREF(new_args);
    PyObject *result = PyObject_Call(self->func, call_args, kwargs);
    Py_DECREF(call_args);

    if (!result) { Py_DECREF(x); return NULL; }

    if (PyObject_TypeCheck(result, &VariableType)) {
        VariableObject *rv = (VariableObject *)result;
        VarList topo;
        if (varlist_init(&topo, 64) < 0) { Py_DECREF(result); Py_DECREF(x); return NULL; }
        if (build_topo(rv, &topo) < 0) { varlist_free(&topo); Py_DECREF(result); Py_DECREF(x); return NULL; }
        for (Py_ssize_t i = 0; i < topo.count; i++) topo.items[i]->grad = 0.0;
        rv->grad = 1.0;
        for (Py_ssize_t i = topo.count - 1; i >= 0; i--) apply_backward(topo.items[i]);
        varlist_free(&topo);

        double fval = rv->data;
        double gval = x->grad;
        Py_DECREF(result);
        Py_DECREF(x);
        return Py_BuildValue("(dd)", fval, gval);
    }

    double fval = PyFloat_AsDouble(result);
    Py_DECREF(result);
    Py_DECREF(x);
    if (PyErr_Occurred()) return NULL;
    return Py_BuildValue("(dd)", fval, 0.0);
}

static PyTypeObject ValueAndGradCallableType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "xue._autodiff_accel._ValueAndGradCallable",
    .tp_basicsize = sizeof(ValueAndGradCallableObject),
    .tp_dealloc = (destructor)VAGCallable_dealloc,
    .tp_call = (ternaryfunc)VAGCallable_call,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

/* ── Module-level functions ───────────────────────────────────────── */

static PyObject *mod_grad_func(PyObject *self, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"f", "argnum", NULL};
    PyObject *f;
    int argnum = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|i", kwlist, &f, &argnum))
        return NULL;

    GradCallableObject *gc = PyObject_New(GradCallableObject, &GradCallableType);
    if (!gc) return NULL;
    Py_INCREF(f);
    gc->func = f;
    gc->argnum = argnum;
    return (PyObject *)gc;
}

static PyObject *mod_value_and_grad_func(PyObject *self, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"f", "argnum", NULL};
    PyObject *f;
    int argnum = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|i", kwlist, &f, &argnum))
        return NULL;

    ValueAndGradCallableObject *vg = PyObject_New(ValueAndGradCallableObject,
                                                   &ValueAndGradCallableType);
    if (!vg) return NULL;
    Py_INCREF(f);
    vg->func = f;
    vg->argnum = argnum;
    return (PyObject *)vg;
}

static PyObject *mod_jacobian_func(PyObject *self, PyObject *args) {
    PyObject *f;
    PyObject *x_list;
    if (!PyArg_ParseTuple(args, "OO", &f, &x_list))
        return NULL;

    PyObject *x_seq = PySequence_Fast(x_list, "x must be a sequence");
    if (!x_seq) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(x_seq);

    /* Create Variable list */
    PyObject *vars = PyList_New(n);
    VariableObject **var_ptrs = (VariableObject **)PyMem_Malloc(sizeof(VariableObject *) * n);
    if (!vars || !var_ptrs) {
        Py_XDECREF(vars);
        Py_DECREF(x_seq);
        if (var_ptrs) PyMem_Free(var_ptrs);
        return PyErr_NoMemory();
    }

    for (Py_ssize_t i = 0; i < n; i++) {
        double val = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(x_seq, i));
        if (PyErr_Occurred()) {
            Py_DECREF(vars);
            Py_DECREF(x_seq);
            PyMem_Free(var_ptrs);
            return NULL;
        }
        PyObject *name = PyUnicode_FromFormat("x%zd", i);
        VariableObject *v = (VariableObject *)Variable_new(&VariableType,
            Py_BuildValue("(d)", val), NULL);
        if (!v) {
            Py_XDECREF(name);
            Py_DECREF(vars);
            Py_DECREF(x_seq);
            PyMem_Free(var_ptrs);
            return NULL;
        }
        Py_XDECREF(v->name);
        v->name = name;
        var_ptrs[i] = v;
        PyList_SET_ITEM(vars, i, (PyObject *)v);  /* steals ref */
    }
    Py_DECREF(x_seq);

    /* Call f(vars) */
    PyObject *call_args = PyTuple_Pack(1, vars);
    PyObject *outputs = PyObject_Call(f, call_args, NULL);
    Py_DECREF(call_args);
    if (!outputs) { Py_DECREF(vars); PyMem_Free(var_ptrs); return NULL; }

    /* Normalize outputs to a list */
    PyObject *out_list;
    if (PyObject_TypeCheck(outputs, &VariableType)) {
        out_list = PyList_New(1);
        PyList_SET_ITEM(out_list, 0, outputs);  /* steals ref */
    } else {
        out_list = PySequence_Fast(outputs, "f must return Variable or list of Variables");
        Py_DECREF(outputs);
        if (!out_list) { Py_DECREF(vars); PyMem_Free(var_ptrs); return NULL; }
    }

    Py_ssize_t m = PySequence_Fast_GET_SIZE(out_list);

    /* Build Jacobian */
    PyObject *J = PyList_New(m);
    for (Py_ssize_t i = 0; i < m; i++) {
        PyObject *out_i = PySequence_Fast_GET_ITEM(out_list, i);
        if (!PyObject_TypeCheck(out_i, &VariableType)) {
            PyErr_SetString(PyExc_TypeError, "Output must be Variable");
            Py_DECREF(J); Py_DECREF(out_list); Py_DECREF(vars);
            PyMem_Free(var_ptrs);
            return NULL;
        }
        VariableObject *ov = (VariableObject *)out_i;

        /* Reset grads */
        for (Py_ssize_t j = 0; j < n; j++)
            var_ptrs[j]->grad = 0.0;

        /* Backward */
        VarList topo;
        if (varlist_init(&topo, 64) < 0) {
            Py_DECREF(J); Py_DECREF(out_list); Py_DECREF(vars);
            PyMem_Free(var_ptrs);
            return NULL;
        }
        if (build_topo(ov, &topo) < 0) {
            varlist_free(&topo);
            Py_DECREF(J); Py_DECREF(out_list); Py_DECREF(vars);
            PyMem_Free(var_ptrs);
            return NULL;
        }
        for (Py_ssize_t k = 0; k < topo.count; k++)
            topo.items[k]->grad = 0.0;
        ov->grad = 1.0;
        for (Py_ssize_t k = topo.count - 1; k >= 0; k--)
            apply_backward(topo.items[k]);
        varlist_free(&topo);

        /* Collect row */
        PyObject *row = PyList_New(n);
        for (Py_ssize_t j = 0; j < n; j++)
            PyList_SET_ITEM(row, j, PyFloat_FromDouble(var_ptrs[j]->grad));
        PyList_SET_ITEM(J, i, row);
    }

    Py_DECREF(out_list);
    Py_DECREF(vars);
    PyMem_Free(var_ptrs);
    return J;
}

/* ── Module definition ────────────────────────────────────────────── */

static PyMethodDef module_methods[] = {
    {"grad", (PyCFunction)mod_grad_func, METH_VARARGS | METH_KEYWORDS,
     "Return gradient function."},
    {"value_and_grad", (PyCFunction)mod_value_and_grad_func, METH_VARARGS | METH_KEYWORDS,
     "Return function that computes both value and gradient."},
    {"jacobian", (PyCFunction)mod_jacobian_func, METH_VARARGS,
     "Compute Jacobian matrix."},
    {NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_autodiff_accel",
    "C-accelerated automatic differentiation for xue",
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__autodiff_accel(void) {
    if (PyType_Ready(&VariableType) < 0) return NULL;
    if (PyType_Ready(&GradCallableType) < 0) return NULL;
    if (PyType_Ready(&ValueAndGradCallableType) < 0) return NULL;

    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;

    Py_INCREF(&VariableType);
    PyModule_AddObject(m, "Variable", (PyObject *)&VariableType);

    return m;
}
