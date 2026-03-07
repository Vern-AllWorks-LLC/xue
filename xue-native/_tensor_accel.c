/*
 * _tensor_accel.c — C-accelerated tensor operations with SIMD and CUDA
 *
 * Uses raw C arrays for storage, AVX2 SIMD for elementwise ops and matmul,
 * and optional CUDA GPU kernels loaded at runtime via dlopen.
 *
 * Techniques: standard numerical computing (BLAS-era, 1970s+),
 * SSE/AVX intrinsics on published Intel ISA.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* ── SIMD detection ───────────────────────────────────────────────── */

#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
  #define HAVE_AVX2 1
  #define HAVE_NEON 0
#elif defined(__aarch64__) || defined(_M_ARM64)
  #include <arm_neon.h>
  #define HAVE_AVX2 0
  #define HAVE_NEON 1
#else
  #define HAVE_AVX2 0
  #define HAVE_NEON 0
#endif

/* ── Cross-platform dynamic loading ──────────────────────────────── */

#ifdef _WIN32
  #include <windows.h>
  typedef HMODULE lib_handle_t;
  #define lib_open(name)      LoadLibraryA(name)
  #define lib_sym(h, sym)     ((void *)GetProcAddress(h, sym))
  #define lib_close(h)        FreeLibrary(h)
#else
  #include <dlfcn.h>
  typedef void *lib_handle_t;
  #define lib_open(name)      dlopen(name, RTLD_LAZY)
  #define lib_sym(h, sym)     dlsym(h, sym)
  #define lib_close(h)        dlclose(h)
#endif

typedef struct {
    lib_handle_t handle;   /* handle to libcudart */
    int available;         /* 1 if CUDA is usable */
    /* Function pointers */
    int (*cudaMalloc)(void **devPtr, size_t size);
    int (*cudaFree)(void *devPtr);
    int (*cudaMemcpy)(void *dst, const void *src, size_t count, int kind);
    int (*cudaGetDeviceCount)(int *count);
    /* Custom kernel library */
    lib_handle_t kernel_handle;  /* handle to _xue_cuda_kernels */
    void (*cuda_elementwise_add)(double *out, const double *a, const double *b, int n);
    void (*cuda_elementwise_sub)(double *out, const double *a, const double *b, int n);
    void (*cuda_elementwise_mul)(double *out, const double *a, const double *b, int n);
    void (*cuda_elementwise_div)(double *out, const double *a, const double *b, int n);
    void (*cuda_scalar_mul)(double *out, const double *a, double s, int n);
    void (*cuda_matmul)(double *out, const double *a, const double *b, int m, int k, int n);
} CudaRuntime;

static CudaRuntime cuda_rt = {0};

static void init_cuda(void) {
    if (cuda_rt.handle) return;  /* already initialized */

    /* Platform-specific CUDA runtime library names */
#ifdef _WIN32
    cuda_rt.handle = lib_open("cudart64_12.dll");
    if (!cuda_rt.handle) cuda_rt.handle = lib_open("cudart64_11.dll");
    if (!cuda_rt.handle) cuda_rt.handle = lib_open("cudart.dll");
#elif defined(__APPLE__)
    cuda_rt.handle = lib_open("libcudart.dylib");
#else
    cuda_rt.handle = lib_open("libcudart.so");
    if (!cuda_rt.handle) cuda_rt.handle = lib_open("libcudart.so.12");
#endif
    if (!cuda_rt.handle) {
        cuda_rt.available = 0;
        return;
    }

    cuda_rt.cudaMalloc = lib_sym(cuda_rt.handle, "cudaMalloc");
    cuda_rt.cudaFree = lib_sym(cuda_rt.handle, "cudaFree");
    cuda_rt.cudaMemcpy = lib_sym(cuda_rt.handle, "cudaMemcpy");
    cuda_rt.cudaGetDeviceCount = lib_sym(cuda_rt.handle, "cudaGetDeviceCount");

    if (!cuda_rt.cudaMalloc || !cuda_rt.cudaFree || !cuda_rt.cudaMemcpy ||
        !cuda_rt.cudaGetDeviceCount) {
        lib_close(cuda_rt.handle);
        cuda_rt.handle = (lib_handle_t)0;
        cuda_rt.available = 0;
        return;
    }

    /* Check for actual GPU */
    int count = 0;
    cuda_rt.cudaGetDeviceCount(&count);
    if (count <= 0) {
        cuda_rt.available = 0;
        return;
    }

    /* Try to load kernel library */
#ifdef _WIN32
    #define KERNEL_LIB "_xue_cuda_kernels.dll"
#elif defined(__APPLE__)
    #define KERNEL_LIB "_xue_cuda_kernels.dylib"
#else
    #define KERNEL_LIB "_xue_cuda_kernels.so"
#endif
    cuda_rt.kernel_handle = lib_open("./" KERNEL_LIB);
    if (!cuda_rt.kernel_handle) {
        Py_BEGIN_ALLOW_THREADS
        cuda_rt.kernel_handle = lib_open(KERNEL_LIB);
        Py_END_ALLOW_THREADS
    }
    if (cuda_rt.kernel_handle) {
        cuda_rt.cuda_elementwise_add = lib_sym(cuda_rt.kernel_handle, "cuda_elementwise_add");
        cuda_rt.cuda_elementwise_sub = lib_sym(cuda_rt.kernel_handle, "cuda_elementwise_sub");
        cuda_rt.cuda_elementwise_mul = lib_sym(cuda_rt.kernel_handle, "cuda_elementwise_mul");
        cuda_rt.cuda_elementwise_div = lib_sym(cuda_rt.kernel_handle, "cuda_elementwise_div");
        cuda_rt.cuda_scalar_mul = lib_sym(cuda_rt.kernel_handle, "cuda_scalar_mul");
        cuda_rt.cuda_matmul = lib_sym(cuda_rt.kernel_handle, "cuda_matmul");
    }

    cuda_rt.available = 1;
}

/* cudaMemcpyKind enum values */
#define CUDA_MEMCPY_H2D 1
#define CUDA_MEMCPY_D2H 2
#define CUDA_MEMCPY_D2D 3

/* ── DType ────────────────────────────────────────────────────────── */

typedef struct {
    PyObject_HEAD
    char name[16];
    int size;
    char typecode;
} DTypeObject;

static PyTypeObject DTypeType;

static void DType_dealloc(DTypeObject *self) {
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *DType_repr(DTypeObject *self) {
    return PyUnicode_FromString(self->name);
}

static PyObject *DType_richcompare(PyObject *a, PyObject *b, int op) {
    if (!PyObject_TypeCheck(a, &DTypeType) || !PyObject_TypeCheck(b, &DTypeType))
        Py_RETURN_NOTIMPLEMENTED;
    int eq = (strcmp(((DTypeObject *)a)->name, ((DTypeObject *)b)->name) == 0);
    if (op == Py_EQ) return PyBool_FromLong(eq);
    if (op == Py_NE) return PyBool_FromLong(!eq);
    Py_RETURN_NOTIMPLEMENTED;
}

static Py_hash_t DType_hash(DTypeObject *self) {
    /* Simple hash of name */
    Py_hash_t h = 0;
    for (const char *p = self->name; *p; p++)
        h = h * 31 + *p;
    if (h == -1) h = -2;
    return h;
}

static PyMemberDef DType_members[] = {
    {"size", T_INT, offsetof(DTypeObject, size), READONLY, NULL},
    {NULL}
};

static PyObject *DType_get_name(DTypeObject *self, void *closure) {
    return PyUnicode_FromString(self->name);
}

static PyGetSetDef DType_getset[] = {
    {"name", (getter)DType_get_name, NULL, NULL, NULL},
    {NULL}
};

static PyTypeObject DTypeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "xue._tensor_accel.DType",
    .tp_basicsize = sizeof(DTypeObject),
    .tp_dealloc = (destructor)DType_dealloc,
    .tp_repr = (reprfunc)DType_repr,
    .tp_hash = (hashfunc)DType_hash,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_richcompare = DType_richcompare,
    .tp_members = DType_members,
    .tp_getset = DType_getset,
};

/* Pre-built DType instances */
static DTypeObject *dt_float32;
static DTypeObject *dt_float64;
static DTypeObject *dt_int32;
static DTypeObject *dt_int64;
static DTypeObject *dt_int8;
static DTypeObject *dt_int16;
static DTypeObject *dt_uint8;
static DTypeObject *dt_float16;
static DTypeObject *dt_bool;

static DTypeObject *make_dtype(const char *name, int size, char typecode) {
    DTypeObject *d = PyObject_New(DTypeObject, &DTypeType);
    if (!d) return NULL;
    strncpy(d->name, name, sizeof(d->name) - 1);
    d->name[sizeof(d->name) - 1] = '\0';
    d->size = size;
    d->typecode = typecode;
    return d;
}

/* ── ShapeError ───────────────────────────────────────────────────── */

static PyObject *ShapeError;

/* ══════════════════════════════════════════════════════════════════
   TensorObject — raw C array with SIMD acceleration
   ══════════════════════════════════════════════════════════════════ */

typedef struct {
    PyObject_HEAD
    double *data;         /* raw C array (always float64 internally) */
    Py_ssize_t *shape;    /* shape array */
    Py_ssize_t *strides;  /* stride array (in elements, not bytes) */
    int ndim;
    Py_ssize_t size;      /* total number of elements */
    Py_ssize_t offset;    /* offset into data (for views) */
    DTypeObject *dtype;
    int owns_data;        /* 1 if we should free data on dealloc */
    /* GPU fields */
    double *gpu_data;     /* device pointer, NULL if not on GPU */
    int on_gpu;
} TensorObject;

static PyTypeObject TensorType;

/* Forward declarations */
static TensorObject *Tensor_empty(Py_ssize_t *shape, int ndim, DTypeObject *dtype);
static void compute_strides(Py_ssize_t *shape, Py_ssize_t *strides, int ndim);
static int flatten_data(PyObject *data, double *buf, Py_ssize_t *pos,
                        Py_ssize_t *shape, int *ndim, int depth, int max_depth);

/* ── Allocation helpers ───────────────────────────────────────────── */

static void compute_strides(Py_ssize_t *shape, Py_ssize_t *strides, int ndim) {
    Py_ssize_t stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

static TensorObject *Tensor_empty(Py_ssize_t *shape, int ndim, DTypeObject *dtype) {
    TensorObject *t = PyObject_New(TensorObject, &TensorType);
    if (!t) return NULL;

    Py_ssize_t size = 1;
    for (int i = 0; i < ndim; i++) size *= shape[i];

    t->data = (double *)PyMem_Calloc(size, sizeof(double));
    t->shape = (Py_ssize_t *)PyMem_Malloc(sizeof(Py_ssize_t) * ndim);
    t->strides = (Py_ssize_t *)PyMem_Malloc(sizeof(Py_ssize_t) * ndim);
    if (!t->data || !t->shape || !t->strides) {
        PyMem_Free(t->data);
        PyMem_Free(t->shape);
        PyMem_Free(t->strides);
        Py_TYPE(t)->tp_free((PyObject *)t);
        return (TensorObject *)PyErr_NoMemory();
    }

    memcpy(t->shape, shape, sizeof(Py_ssize_t) * ndim);
    compute_strides(shape, t->strides, ndim);
    t->ndim = ndim;
    t->size = size;
    t->offset = 0;
    Py_INCREF(dtype);
    t->dtype = dtype;
    t->owns_data = 1;
    t->gpu_data = NULL;
    t->on_gpu = 0;
    return t;
}

static void Tensor_dealloc(TensorObject *self) {
    if (self->owns_data && self->data)
        PyMem_Free(self->data);
    PyMem_Free(self->shape);
    PyMem_Free(self->strides);
    Py_XDECREF(self->dtype);
    if (self->gpu_data && cuda_rt.available && cuda_rt.cudaFree)
        cuda_rt.cudaFree(self->gpu_data);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* ── Flatten nested Python lists into C double array ──────────────── */

/* First pass: infer shape */
static int infer_shape(PyObject *data, Py_ssize_t *shape, int *ndim, int depth) {
    if (PyFloat_Check(data) || PyLong_Check(data)) {
        *ndim = depth;
        return 0;
    }
    if (!PyList_Check(data) && !PyTuple_Check(data)) {
        PyErr_SetString(PyExc_TypeError, "Cannot create tensor from this type");
        return -1;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(data);
    shape[depth] = n;
    if (n == 0) {
        *ndim = depth + 1;
        return 0;
    }
    return infer_shape(PySequence_Fast_GET_ITEM(data, 0), shape, ndim, depth + 1);
}

/* Second pass: copy data */
static int copy_flat(PyObject *data, double *buf, Py_ssize_t *pos) {
    if (PyFloat_Check(data) || PyLong_Check(data)) {
        buf[(*pos)++] = PyFloat_AsDouble(data);
        return PyErr_Occurred() ? -1 : 0;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(data);
    for (Py_ssize_t i = 0; i < n; i++) {
        if (copy_flat(PySequence_Fast_GET_ITEM(data, i), buf, pos) < 0)
            return -1;
    }
    return 0;
}

/* ── Constructor ──────────────────────────────────────────────────── */

static PyObject *Tensor_new(PyTypeObject *type, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"data", "shape", "dtype", NULL};
    PyObject *data = NULL;
    PyObject *shape_arg = NULL;
    DTypeObject *dtype = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "|OOO!", kwlist,
            &data, &shape_arg, &DTypeType, &dtype))
        return NULL;

    if (!dtype) dtype = dt_float64;

    if (data && data != Py_None) {
        /* Ensure it's a fast sequence for recursive processing */
        PyObject *seq = data;
        int need_decref = 0;
        if (!PyList_Check(data) && !PyTuple_Check(data)) {
            if (PyFloat_Check(data) || PyLong_Check(data)) {
                /* Scalar */
                Py_ssize_t sh[] = {1};  /* treat as 1-element */
                if (shape_arg) {
                    /* Use provided shape */
                } else {
                    TensorObject *t = Tensor_empty(sh, 0, dtype);
                    if (!t) return NULL;
                    /* Actually scalars... let's treat as shape () */
                    /* For compatibility with existing tests, wrap in 1D if no shape given */
                    /* The Python version infers (,) for scalar */
                    t->data[0] = PyFloat_AsDouble(data);
                    return (PyObject *)t;
                }
            }
            seq = PySequence_Fast(data, "data must be a nested list/tuple");
            if (!seq) return NULL;
            need_decref = 1;
        }

        /* Infer shape */
        Py_ssize_t inferred_shape[32];
        int ndim = 0;
        if (infer_shape(seq, inferred_shape, &ndim, 0) < 0) {
            if (need_decref) Py_DECREF(seq);
            return NULL;
        }

        /* Validate against provided shape */
        Py_ssize_t *use_shape = inferred_shape;
        int use_ndim = ndim;
        Py_ssize_t explicit_shape[32];

        if (shape_arg && shape_arg != Py_None) {
            PyObject *shape_seq = PySequence_Fast(shape_arg, "shape must be a sequence");
            if (!shape_seq) { if (need_decref) Py_DECREF(seq); return NULL; }
            use_ndim = (int)PySequence_Fast_GET_SIZE(shape_seq);
            Py_ssize_t explicit_size = 1;
            for (int i = 0; i < use_ndim; i++) {
                explicit_shape[i] = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(shape_seq, i));
                explicit_size *= explicit_shape[i];
            }
            Py_DECREF(shape_seq);

            /* Check total size matches */
            Py_ssize_t data_size = 1;
            for (int i = 0; i < ndim; i++) data_size *= inferred_shape[i];
            if (data_size != explicit_size) {
                PyErr_Format(ShapeError,
                    "Data has %zd elements but shape requires %zd",
                    data_size, explicit_size);
                if (need_decref) Py_DECREF(seq);
                return NULL;
            }
            use_shape = explicit_shape;
        }

        /* Allocate tensor */
        TensorObject *t = Tensor_empty(use_shape, use_ndim, dtype);
        if (!t) { if (need_decref) Py_DECREF(seq); return NULL; }

        /* Copy data */
        Py_ssize_t pos = 0;
        if (copy_flat(seq, t->data, &pos) < 0) {
            if (need_decref) Py_DECREF(seq);
            Py_DECREF(t);
            return NULL;
        }

        if (need_decref) Py_DECREF(seq);
        return (PyObject *)t;
    }

    /* No data — create zeros with given shape */
    if (!shape_arg || shape_arg == Py_None) {
        PyErr_SetString(PyExc_ValueError, "Either data or shape must be provided");
        return NULL;
    }

    PyObject *shape_seq = PySequence_Fast(shape_arg, "shape must be a sequence");
    if (!shape_seq) return NULL;
    int ndim = (int)PySequence_Fast_GET_SIZE(shape_seq);
    Py_ssize_t shape[32];
    for (int i = 0; i < ndim; i++)
        shape[i] = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(shape_seq, i));
    Py_DECREF(shape_seq);
    if (PyErr_Occurred()) return NULL;

    return (PyObject *)Tensor_empty(shape, ndim, dtype);
}

/* ── SIMD elementwise operations ──────────────────────────────────── */

#if HAVE_AVX2

/* ── AVX2: 4 doubles per cycle (x86_64) ──────────────────────────── */

static void simd_add(double * restrict out, const double * restrict a,
                     const double * restrict b, Py_ssize_t n) {
    Py_ssize_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_add_pd(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] + b[i];
}

static void simd_sub(double * restrict out, const double * restrict a,
                     const double * restrict b, Py_ssize_t n) {
    Py_ssize_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_sub_pd(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] - b[i];
}

static void simd_mul(double * restrict out, const double * restrict a,
                     const double * restrict b, Py_ssize_t n) {
    Py_ssize_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_mul_pd(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] * b[i];
}

static void simd_div(double * restrict out, const double * restrict a,
                     const double * restrict b, Py_ssize_t n) {
    Py_ssize_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_div_pd(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] / b[i];
}

static void simd_scalar_mul(double * restrict out, const double * restrict a,
                            double s, Py_ssize_t n) {
    __m256d vs = _mm256_set1_pd(s);
    Py_ssize_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        _mm256_storeu_pd(out + i, _mm256_mul_pd(va, vs));
    }
    for (; i < n; i++) out[i] = a[i] * s;
}

static void simd_scalar_add(double * restrict out, const double * restrict a,
                            double s, Py_ssize_t n) {
    __m256d vs = _mm256_set1_pd(s);
    Py_ssize_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        _mm256_storeu_pd(out + i, _mm256_add_pd(va, vs));
    }
    for (; i < n; i++) out[i] = a[i] + s;
}

static void simd_neg(double * restrict out, const double * restrict a, Py_ssize_t n) {
    __m256d zero = _mm256_setzero_pd();
    Py_ssize_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        _mm256_storeu_pd(out + i, _mm256_sub_pd(zero, va));
    }
    for (; i < n; i++) out[i] = -a[i];
}

static double simd_sum(const double *a, Py_ssize_t n) {
    __m256d vsum = _mm256_setzero_pd();
    Py_ssize_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        vsum = _mm256_add_pd(vsum, va);
    }
    double tmp[4];
    _mm256_storeu_pd(tmp, vsum);
    double s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; i++) s += a[i];
    return s;
}

#elif HAVE_NEON

/* ── NEON: 2 doubles per cycle (ARM64 / Apple Silicon) ───────────── */

static void simd_add(double * restrict out, const double * restrict a,
                     const double * restrict b, Py_ssize_t n) {
    Py_ssize_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(out + i, vaddq_f64(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] + b[i];
}

static void simd_sub(double * restrict out, const double * restrict a,
                     const double * restrict b, Py_ssize_t n) {
    Py_ssize_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(out + i, vsubq_f64(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] - b[i];
}

static void simd_mul(double * restrict out, const double * restrict a,
                     const double * restrict b, Py_ssize_t n) {
    Py_ssize_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(out + i, vmulq_f64(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] * b[i];
}

static void simd_div(double * restrict out, const double * restrict a,
                     const double * restrict b, Py_ssize_t n) {
    Py_ssize_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(out + i, vdivq_f64(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] / b[i];
}

static void simd_scalar_mul(double * restrict out, const double * restrict a,
                            double s, Py_ssize_t n) {
    float64x2_t vs = vdupq_n_f64(s);
    Py_ssize_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        vst1q_f64(out + i, vmulq_f64(va, vs));
    }
    for (; i < n; i++) out[i] = a[i] * s;
}

static void simd_scalar_add(double * restrict out, const double * restrict a,
                            double s, Py_ssize_t n) {
    float64x2_t vs = vdupq_n_f64(s);
    Py_ssize_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        vst1q_f64(out + i, vaddq_f64(va, vs));
    }
    for (; i < n; i++) out[i] = a[i] + s;
}

static void simd_neg(double * restrict out, const double * restrict a, Py_ssize_t n) {
    Py_ssize_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        vst1q_f64(out + i, vnegq_f64(va));
    }
    for (; i < n; i++) out[i] = -a[i];
}

static double simd_sum(const double *a, Py_ssize_t n) {
    float64x2_t vsum = vdupq_n_f64(0.0);
    Py_ssize_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        vsum = vaddq_f64(vsum, va);
    }
    double s = vgetq_lane_f64(vsum, 0) + vgetq_lane_f64(vsum, 1);
    for (; i < n; i++) s += a[i];
    return s;
}

#else

/* ── Scalar fallbacks (any platform) ─────────────────────────────── */

static void simd_add(double *out, const double *a, const double *b, Py_ssize_t n) {
    for (Py_ssize_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}
static void simd_sub(double *out, const double *a, const double *b, Py_ssize_t n) {
    for (Py_ssize_t i = 0; i < n; i++) out[i] = a[i] - b[i];
}
static void simd_mul(double *out, const double *a, const double *b, Py_ssize_t n) {
    for (Py_ssize_t i = 0; i < n; i++) out[i] = a[i] * b[i];
}
static void simd_div(double *out, const double *a, const double *b, Py_ssize_t n) {
    for (Py_ssize_t i = 0; i < n; i++) out[i] = a[i] / b[i];
}
static void simd_scalar_mul(double *out, const double *a, double s, Py_ssize_t n) {
    for (Py_ssize_t i = 0; i < n; i++) out[i] = a[i] * s;
}
static void simd_scalar_add(double *out, const double *a, double s, Py_ssize_t n) {
    for (Py_ssize_t i = 0; i < n; i++) out[i] = a[i] + s;
}
static void simd_neg(double *out, const double *a, Py_ssize_t n) {
    for (Py_ssize_t i = 0; i < n; i++) out[i] = -a[i];
}
static double simd_sum(const double *a, Py_ssize_t n) {
    double s = 0;
    for (Py_ssize_t i = 0; i < n; i++) s += a[i];
    return s;
}
#endif

/* ── SIMD tiled matrix multiply ───────────────────────────────────── */

#define TILE_SIZE 32

static void matmul_tiled(double * restrict C,
                         const double * restrict A,
                         const double * restrict B,
                         Py_ssize_t M, Py_ssize_t K, Py_ssize_t N) {
    /* Zero output */
    memset(C, 0, M * N * sizeof(double));

    /* Tiled matmul for cache efficiency */
    for (Py_ssize_t i0 = 0; i0 < M; i0 += TILE_SIZE) {
        Py_ssize_t iend = (i0 + TILE_SIZE < M) ? i0 + TILE_SIZE : M;
        for (Py_ssize_t k0 = 0; k0 < K; k0 += TILE_SIZE) {
            Py_ssize_t kend = (k0 + TILE_SIZE < K) ? k0 + TILE_SIZE : K;
            for (Py_ssize_t j0 = 0; j0 < N; j0 += TILE_SIZE) {
                Py_ssize_t jend = (j0 + TILE_SIZE < N) ? j0 + TILE_SIZE : N;

                for (Py_ssize_t i = i0; i < iend; i++) {
                    for (Py_ssize_t k = k0; k < kend; k++) {
                        double a_ik = A[i * K + k];
#if HAVE_AVX2
                        __m256d va = _mm256_set1_pd(a_ik);
                        Py_ssize_t j = j0;
                        for (; j + 4 <= jend; j += 4) {
                            __m256d vb = _mm256_loadu_pd(&B[k * N + j]);
                            __m256d vc = _mm256_loadu_pd(&C[i * N + j]);
                            _mm256_storeu_pd(&C[i * N + j],
                                _mm256_add_pd(vc, _mm256_mul_pd(va, vb)));
                        }
                        for (; j < jend; j++)
                            C[i * N + j] += a_ik * B[k * N + j];
#elif HAVE_NEON
                        float64x2_t va = vdupq_n_f64(a_ik);
                        Py_ssize_t j = j0;
                        for (; j + 2 <= jend; j += 2) {
                            float64x2_t vb = vld1q_f64(&B[k * N + j]);
                            float64x2_t vc = vld1q_f64(&C[i * N + j]);
                            vst1q_f64(&C[i * N + j],
                                vaddq_f64(vc, vmulq_f64(va, vb)));
                        }
                        for (; j < jend; j++)
                            C[i * N + j] += a_ik * B[k * N + j];
#else
                        for (Py_ssize_t j = j0; j < jend; j++)
                            C[i * N + j] += a_ik * B[k * N + j];
#endif
                    }
                }
            }
        }
    }
}

/* ── Tensor properties ────────────────────────────────────────────── */

static PyObject *Tensor_get_shape(TensorObject *self, void *closure) {
    PyObject *t = PyTuple_New(self->ndim);
    for (int i = 0; i < self->ndim; i++)
        PyTuple_SET_ITEM(t, i, PyLong_FromSsize_t(self->shape[i]));
    return t;
}

static PyObject *Tensor_get_dtype(TensorObject *self, void *closure) {
    Py_INCREF(self->dtype);
    return (PyObject *)self->dtype;
}

static PyObject *Tensor_get_ndim(TensorObject *self, void *closure) {
    return PyLong_FromLong(self->ndim);
}

static PyObject *Tensor_get_size(TensorObject *self, void *closure) {
    return PyLong_FromSsize_t(self->size);
}

/* ── Element access ───────────────────────────────────────────────── */

static Py_ssize_t Tensor_flat_index(TensorObject *self, Py_ssize_t *indices) {
    Py_ssize_t idx = self->offset;
    for (int i = 0; i < self->ndim; i++)
        idx += indices[i] * self->strides[i];
    return idx;
}

static PyObject *Tensor_getitem(TensorObject *self, PyObject *key) {
    if (PyLong_Check(key)) {
        if (self->ndim == 1) {
            Py_ssize_t idx = PyLong_AsSsize_t(key);
            return PyFloat_FromDouble(self->data[self->offset + idx * self->strides[0]]);
        }
        /* Return a sub-tensor (row) */
        Py_ssize_t idx = PyLong_AsSsize_t(key);
        if (idx < 0 || idx >= self->shape[0]) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return NULL;
        }
        TensorObject *sub = Tensor_empty(self->shape + 1, self->ndim - 1, self->dtype);
        if (!sub) return NULL;
        Py_ssize_t row_size = self->strides[0];
        memcpy(sub->data, self->data + self->offset + idx * row_size,
               row_size * sizeof(double));
        return (PyObject *)sub;
    }

    if (PyTuple_Check(key)) {
        Py_ssize_t n = PyTuple_GET_SIZE(key);

        /* Check for slices */
        int has_slice = 0;
        for (Py_ssize_t i = 0; i < n; i++) {
            if (PySlice_Check(PyTuple_GET_ITEM(key, i))) { has_slice = 1; break; }
        }

        if (!has_slice) {
            /* Direct element access */
            if (n != self->ndim) {
                PyErr_Format(PyExc_IndexError, "Expected %d indices, got %zd", self->ndim, n);
                return NULL;
            }
            Py_ssize_t indices[32];
            for (Py_ssize_t i = 0; i < n; i++) {
                indices[i] = PyLong_AsSsize_t(PyTuple_GET_ITEM(key, i));
                if (PyErr_Occurred()) return NULL;
            }
            return PyFloat_FromDouble(self->data[Tensor_flat_index(self, indices)]);
        }

        /* Slice access — build new tensor */
        Py_ssize_t new_shape[32];
        int new_ndim = 0;
        Py_ssize_t starts[32], stops[32], steps[32];
        int is_index[32];  /* 1 if dimension is indexed (not sliced) */

        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *k = PyTuple_GET_ITEM(key, i);
            if (PyLong_Check(k)) {
                starts[i] = PyLong_AsSsize_t(k);
                stops[i] = starts[i] + 1;
                steps[i] = 1;
                is_index[i] = 1;
            } else if (PySlice_Check(k)) {
                Py_ssize_t length;
                PySlice_GetIndicesEx(k, self->shape[i],
                    &starts[i], &stops[i], &steps[i], &length);
                new_shape[new_ndim++] = length;
                is_index[i] = 0;
            }
        }
        /* Pad remaining dims */
        for (int i = (int)n; i < self->ndim; i++) {
            starts[i] = 0;
            stops[i] = self->shape[i];
            steps[i] = 1;
            is_index[i] = 0;
            new_shape[new_ndim++] = self->shape[i];
        }

        TensorObject *result = Tensor_empty(new_shape, new_ndim, self->dtype);
        if (!result) return NULL;

        /* Copy data using recursive index iteration */
        Py_ssize_t src_idx[32] = {0};
        Py_ssize_t dst_pos = 0;

        /* Iterative approach for arbitrary dimensions */
        int total_dims = (n > self->ndim) ? (int)n : self->ndim;
        Py_ssize_t cur[32];
        for (int i = 0; i < total_dims; i++) cur[i] = starts[i];

        int done = 0;
        while (!done) {
            /* Compute flat source index */
            Py_ssize_t flat = self->offset;
            for (int i = 0; i < total_dims; i++)
                flat += cur[i] * self->strides[i];
            result->data[dst_pos++] = self->data[flat];

            /* Advance indices (rightmost first) */
            int dim = total_dims - 1;
            while (dim >= 0) {
                cur[dim] += steps[dim];
                if (cur[dim] < stops[dim]) break;
                cur[dim] = starts[dim];
                dim--;
            }
            if (dim < 0) done = 1;
        }

        return (PyObject *)result;
    }

    if (PySlice_Check(key)) {
        /* Single slice on first dimension */
        PyObject *tuple = PyTuple_Pack(1, key);
        PyObject *result = Tensor_getitem(self, tuple);
        Py_DECREF(tuple);
        return result;
    }

    PyErr_SetString(PyExc_TypeError, "Invalid index type");
    return NULL;
}

static int Tensor_setitem(TensorObject *self, PyObject *key, PyObject *value) {
    if (!PyTuple_Check(key) || PyTuple_GET_SIZE(key) != self->ndim) {
        PyErr_SetString(PyExc_TypeError, "Invalid index for assignment");
        return -1;
    }
    Py_ssize_t indices[32];
    for (int i = 0; i < self->ndim; i++) {
        indices[i] = PyLong_AsSsize_t(PyTuple_GET_ITEM(key, i));
        if (PyErr_Occurred()) return -1;
    }
    double val = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) return -1;
    self->data[Tensor_flat_index(self, indices)] = val;
    return 0;
}

/* ── Arithmetic ───────────────────────────────────────────────────── */

static int check_shapes(TensorObject *a, TensorObject *b) {
    if (a->ndim != b->ndim) goto mismatch;
    for (int i = 0; i < a->ndim; i++)
        if (a->shape[i] != b->shape[i]) goto mismatch;
    return 0;
mismatch:
    PyErr_Format(ShapeError, "Shape mismatch in elementwise operation");
    return -1;
}

static PyObject *Tensor_nb_add(PyObject *va, PyObject *vb) {
    if (PyObject_TypeCheck(va, &TensorType) && (PyFloat_Check(vb) || PyLong_Check(vb))) {
        TensorObject *a = (TensorObject *)va;
        double s = PyFloat_AsDouble(vb);
        TensorObject *r = Tensor_empty(a->shape, a->ndim, a->dtype);
        if (!r) return NULL;
        Py_BEGIN_ALLOW_THREADS
        simd_scalar_add(r->data, a->data + a->offset, s, a->size);
        Py_END_ALLOW_THREADS
        return (PyObject *)r;
    }
    if ((PyFloat_Check(va) || PyLong_Check(va)) && PyObject_TypeCheck(vb, &TensorType)) {
        TensorObject *b = (TensorObject *)vb;
        double s = PyFloat_AsDouble(va);
        TensorObject *r = Tensor_empty(b->shape, b->ndim, b->dtype);
        if (!r) return NULL;
        Py_BEGIN_ALLOW_THREADS
        simd_scalar_add(r->data, b->data + b->offset, s, b->size);
        Py_END_ALLOW_THREADS
        return (PyObject *)r;
    }
    if (!PyObject_TypeCheck(va, &TensorType) || !PyObject_TypeCheck(vb, &TensorType))
        Py_RETURN_NOTIMPLEMENTED;
    TensorObject *a = (TensorObject *)va, *b = (TensorObject *)vb;
    if (check_shapes(a, b) < 0) return NULL;
    TensorObject *r = Tensor_empty(a->shape, a->ndim, a->dtype);
    if (!r) return NULL;

    /* Use GPU if both on GPU */
    if (a->on_gpu && b->on_gpu && cuda_rt.available && cuda_rt.cuda_elementwise_add) {
        Py_BEGIN_ALLOW_THREADS
        cuda_rt.cuda_elementwise_add(r->gpu_data, a->gpu_data, b->gpu_data, (int)a->size);
        Py_END_ALLOW_THREADS
        r->on_gpu = 1;
        return (PyObject *)r;
    }

    Py_BEGIN_ALLOW_THREADS
    simd_add(r->data, a->data + a->offset, b->data + b->offset, a->size);
    Py_END_ALLOW_THREADS
    return (PyObject *)r;
}

static PyObject *Tensor_nb_subtract(PyObject *va, PyObject *vb) {
    if (PyObject_TypeCheck(va, &TensorType) && (PyFloat_Check(vb) || PyLong_Check(vb))) {
        TensorObject *a = (TensorObject *)va;
        double s = PyFloat_AsDouble(vb);
        TensorObject *r = Tensor_empty(a->shape, a->ndim, a->dtype);
        if (!r) return NULL;
        Py_BEGIN_ALLOW_THREADS
        simd_scalar_add(r->data, a->data + a->offset, -s, a->size);
        Py_END_ALLOW_THREADS
        return (PyObject *)r;
    }
    if ((PyFloat_Check(va) || PyLong_Check(va)) && PyObject_TypeCheck(vb, &TensorType)) {
        TensorObject *b = (TensorObject *)vb;
        double s = PyFloat_AsDouble(va);
        TensorObject *r = Tensor_empty(b->shape, b->ndim, b->dtype);
        if (!r) return NULL;
        /* s - b[i] */
        Py_BEGIN_ALLOW_THREADS
        simd_neg(r->data, b->data + b->offset, b->size);
        simd_scalar_add(r->data, r->data, s, b->size);
        Py_END_ALLOW_THREADS
        return (PyObject *)r;
    }
    if (!PyObject_TypeCheck(va, &TensorType) || !PyObject_TypeCheck(vb, &TensorType))
        Py_RETURN_NOTIMPLEMENTED;
    TensorObject *a = (TensorObject *)va, *b = (TensorObject *)vb;
    if (check_shapes(a, b) < 0) return NULL;
    TensorObject *r = Tensor_empty(a->shape, a->ndim, a->dtype);
    if (!r) return NULL;
    Py_BEGIN_ALLOW_THREADS
    simd_sub(r->data, a->data + a->offset, b->data + b->offset, a->size);
    Py_END_ALLOW_THREADS
    return (PyObject *)r;
}

static PyObject *Tensor_nb_multiply(PyObject *va, PyObject *vb) {
    TensorObject *t = NULL;
    double s = 0;
    if (PyObject_TypeCheck(va, &TensorType) && (PyFloat_Check(vb) || PyLong_Check(vb))) {
        t = (TensorObject *)va; s = PyFloat_AsDouble(vb);
    } else if ((PyFloat_Check(va) || PyLong_Check(va)) && PyObject_TypeCheck(vb, &TensorType)) {
        t = (TensorObject *)vb; s = PyFloat_AsDouble(va);
    }
    if (t) {
        TensorObject *r = Tensor_empty(t->shape, t->ndim, t->dtype);
        if (!r) return NULL;
        Py_BEGIN_ALLOW_THREADS
        simd_scalar_mul(r->data, t->data + t->offset, s, t->size);
        Py_END_ALLOW_THREADS
        return (PyObject *)r;
    }
    if (!PyObject_TypeCheck(va, &TensorType) || !PyObject_TypeCheck(vb, &TensorType))
        Py_RETURN_NOTIMPLEMENTED;
    TensorObject *a = (TensorObject *)va, *b = (TensorObject *)vb;
    if (check_shapes(a, b) < 0) return NULL;
    TensorObject *r = Tensor_empty(a->shape, a->ndim, a->dtype);
    if (!r) return NULL;
    Py_BEGIN_ALLOW_THREADS
    simd_mul(r->data, a->data + a->offset, b->data + b->offset, a->size);
    Py_END_ALLOW_THREADS
    return (PyObject *)r;
}

static PyObject *Tensor_nb_truediv(PyObject *va, PyObject *vb) {
    if (PyObject_TypeCheck(va, &TensorType) && (PyFloat_Check(vb) || PyLong_Check(vb))) {
        TensorObject *a = (TensorObject *)va;
        double s = 1.0 / PyFloat_AsDouble(vb);
        TensorObject *r = Tensor_empty(a->shape, a->ndim, a->dtype);
        if (!r) return NULL;
        Py_BEGIN_ALLOW_THREADS
        simd_scalar_mul(r->data, a->data + a->offset, s, a->size);
        Py_END_ALLOW_THREADS
        return (PyObject *)r;
    }
    if (!PyObject_TypeCheck(va, &TensorType) || !PyObject_TypeCheck(vb, &TensorType))
        Py_RETURN_NOTIMPLEMENTED;
    TensorObject *a = (TensorObject *)va, *b = (TensorObject *)vb;
    if (check_shapes(a, b) < 0) return NULL;
    TensorObject *r = Tensor_empty(a->shape, a->ndim, a->dtype);
    if (!r) return NULL;
    Py_BEGIN_ALLOW_THREADS
    simd_div(r->data, a->data + a->offset, b->data + b->offset, a->size);
    Py_END_ALLOW_THREADS
    return (PyObject *)r;
}

static PyObject *Tensor_nb_negative(PyObject *v) {
    TensorObject *a = (TensorObject *)v;
    TensorObject *r = Tensor_empty(a->shape, a->ndim, a->dtype);
    if (!r) return NULL;
    Py_BEGIN_ALLOW_THREADS
    simd_neg(r->data, a->data + a->offset, a->size);
    Py_END_ALLOW_THREADS
    return (PyObject *)r;
}

/* ── Matrix multiply ──────────────────────────────────────────────── */

static PyObject *Tensor_matmul(PyObject *va, PyObject *vb) {
    if (!PyObject_TypeCheck(va, &TensorType) || !PyObject_TypeCheck(vb, &TensorType))
        Py_RETURN_NOTIMPLEMENTED;
    TensorObject *a = (TensorObject *)va, *b = (TensorObject *)vb;

    if (a->ndim < 2 || b->ndim < 2) {
        PyErr_SetString(ShapeError, "Matrix multiply requires at least 2D tensors");
        return NULL;
    }
    Py_ssize_t m = a->shape[a->ndim - 2];
    Py_ssize_t k = a->shape[a->ndim - 1];
    Py_ssize_t n = b->shape[b->ndim - 1];

    if (k != b->shape[b->ndim - 2]) {
        PyErr_Format(ShapeError,
            "Matrix multiply shape mismatch: inner dimensions %zd != %zd", k, b->shape[b->ndim - 2]);
        return NULL;
    }

    Py_ssize_t result_shape[2] = {m, n};
    TensorObject *r = Tensor_empty(result_shape, 2, a->dtype);
    if (!r) return NULL;

    /* Use GPU matmul if available and both on GPU */
    if (a->on_gpu && b->on_gpu && cuda_rt.available && cuda_rt.cuda_matmul) {
        Py_BEGIN_ALLOW_THREADS
        cuda_rt.cuda_matmul(r->gpu_data, a->gpu_data, b->gpu_data,
                           (int)m, (int)k, (int)n);
        Py_END_ALLOW_THREADS
        r->on_gpu = 1;
        return (PyObject *)r;
    }

    /* CPU tiled SIMD matmul */
    Py_BEGIN_ALLOW_THREADS
    matmul_tiled(r->data, a->data + a->offset, b->data + b->offset, m, k, n);
    Py_END_ALLOW_THREADS
    return (PyObject *)r;
}

/* ── Shape operations ─────────────────────────────────────────────── */

static PyObject *Tensor_reshape(TensorObject *self, PyObject *args) {
    Py_ssize_t new_shape[32];
    int new_ndim;

    if (PyTuple_GET_SIZE(args) == 1) {
        PyObject *arg0 = PyTuple_GET_ITEM(args, 0);
        if (PyList_Check(arg0) || PyTuple_Check(arg0)) {
            PyObject *seq = PySequence_Fast(arg0, "shape");
            new_ndim = (int)PySequence_Fast_GET_SIZE(seq);
            for (int i = 0; i < new_ndim; i++)
                new_shape[i] = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(seq, i));
            Py_DECREF(seq);
        } else {
            new_ndim = (int)PyTuple_GET_SIZE(args);
            for (int i = 0; i < new_ndim; i++)
                new_shape[i] = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, i));
        }
    } else {
        new_ndim = (int)PyTuple_GET_SIZE(args);
        for (int i = 0; i < new_ndim; i++)
            new_shape[i] = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, i));
    }
    if (PyErr_Occurred()) return NULL;

    Py_ssize_t new_size = 1;
    for (int i = 0; i < new_ndim; i++) new_size *= new_shape[i];
    if (new_size != self->size) {
        PyErr_Format(ShapeError,
            "Cannot reshape (%zd elements) to (%zd elements)", self->size, new_size);
        return NULL;
    }

    TensorObject *r = Tensor_empty(new_shape, new_ndim, self->dtype);
    if (!r) return NULL;
    memcpy(r->data, self->data + self->offset, self->size * sizeof(double));
    return (PyObject *)r;
}

static PyObject *Tensor_transpose(TensorObject *self, PyObject *Py_UNUSED(args)) {
    if (self->ndim != 2) {
        PyErr_Format(ShapeError, "Transpose requires 2D tensor, got %dD", self->ndim);
        return NULL;
    }
    Py_ssize_t m = self->shape[0], n = self->shape[1];
    Py_ssize_t new_shape[2] = {n, m};
    TensorObject *r = Tensor_empty(new_shape, 2, self->dtype);
    if (!r) return NULL;

    const double *src = self->data + self->offset;
    double *dst = r->data;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < m; i++)
        for (Py_ssize_t j = 0; j < n; j++)
            dst[j * m + i] = src[i * n + j];
    Py_END_ALLOW_THREADS
    return (PyObject *)r;
}

static PyObject *Tensor_get_T(TensorObject *self, void *closure) {
    return Tensor_transpose(self, NULL);
}

/* ── Reductions ───────────────────────────────────────────────────── */

static PyObject *Tensor_sum(TensorObject *self, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"axis", NULL};
    PyObject *axis_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "|O", kwlist, &axis_obj))
        return NULL;

    if (axis_obj == Py_None) {
        double s;
        Py_BEGIN_ALLOW_THREADS
        s = simd_sum(self->data + self->offset, self->size);
        Py_END_ALLOW_THREADS
        return PyFloat_FromDouble(s);
    }
    PyErr_SetString(PyExc_NotImplementedError, "Axis-specific sum not yet implemented");
    return NULL;
}

static PyObject *Tensor_mean(TensorObject *self, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"axis", NULL};
    PyObject *axis_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "|O", kwlist, &axis_obj))
        return NULL;

    if (axis_obj == Py_None) {
        double s;
        Py_BEGIN_ALLOW_THREADS
        s = simd_sum(self->data + self->offset, self->size);
        Py_END_ALLOW_THREADS
        return PyFloat_FromDouble(s / self->size);
    }
    PyErr_SetString(PyExc_NotImplementedError, "Axis-specific mean not yet implemented");
    return NULL;
}

static PyObject *Tensor_max(TensorObject *self, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"axis", NULL};
    PyObject *axis_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "|O", kwlist, &axis_obj))
        return NULL;

    if (axis_obj == Py_None) {
        double mx = self->data[self->offset];
        for (Py_ssize_t i = 1; i < self->size; i++) {
            double v = self->data[self->offset + i];
            if (v > mx) mx = v;
        }
        return PyFloat_FromDouble(mx);
    }
    PyErr_SetString(PyExc_NotImplementedError, "Axis-specific max not yet implemented");
    return NULL;
}

static PyObject *Tensor_min(TensorObject *self, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"axis", NULL};
    PyObject *axis_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "|O", kwlist, &axis_obj))
        return NULL;

    if (axis_obj == Py_None) {
        double mn = self->data[self->offset];
        for (Py_ssize_t i = 1; i < self->size; i++) {
            double v = self->data[self->offset + i];
            if (v < mn) mn = v;
        }
        return PyFloat_FromDouble(mn);
    }
    PyErr_SetString(PyExc_NotImplementedError, "Axis-specific min not yet implemented");
    return NULL;
}

/* ── tolist ───────────────────────────────────────────────────────── */

static PyObject *Tensor_tolist_recursive(const double *data, const Py_ssize_t *shape,
                                          const Py_ssize_t *strides, int ndim,
                                          Py_ssize_t offset) {
    if (ndim == 1) {
        PyObject *lst = PyList_New(shape[0]);
        for (Py_ssize_t i = 0; i < shape[0]; i++)
            PyList_SET_ITEM(lst, i, PyFloat_FromDouble(data[offset + i * strides[0]]));
        return lst;
    }
    PyObject *lst = PyList_New(shape[0]);
    for (Py_ssize_t i = 0; i < shape[0]; i++) {
        PyObject *sub = Tensor_tolist_recursive(data, shape + 1, strides + 1,
                                                 ndim - 1, offset + i * strides[0]);
        if (!sub) { Py_DECREF(lst); return NULL; }
        PyList_SET_ITEM(lst, i, sub);
    }
    return lst;
}

static PyObject *Tensor_tolist(TensorObject *self, PyObject *Py_UNUSED(args)) {
    if (self->ndim == 0) {
        return PyFloat_FromDouble(self->data[self->offset]);
    }
    return Tensor_tolist_recursive(self->data, self->shape, self->strides,
                                    self->ndim, self->offset);
}

/* ── repr ─────────────────────────────────────────────────────────── */

static PyObject *Tensor_repr(TensorObject *self) {
    PyObject *data_list = Tensor_tolist(self, NULL);
    if (!data_list) return NULL;

    PyObject *shape_list = PyList_New(self->ndim);
    for (int i = 0; i < self->ndim; i++)
        PyList_SET_ITEM(shape_list, i, PyLong_FromSsize_t(self->shape[i]));

    PyObject *result = PyUnicode_FromFormat("Tensor(%R, shape=%R, dtype=%s)",
        data_list, shape_list, self->dtype->name);
    Py_DECREF(data_list);
    Py_DECREF(shape_list);
    return result;
}

static Py_ssize_t Tensor_length(TensorObject *self) {
    return self->shape[0];
}

/* ── Equality ─────────────────────────────────────────────────────── */

static PyObject *Tensor_richcompare(PyObject *va, PyObject *vb, int op) {
    if (!PyObject_TypeCheck(va, &TensorType) || !PyObject_TypeCheck(vb, &TensorType))
        Py_RETURN_NOTIMPLEMENTED;
    if (op != Py_EQ && op != Py_NE)
        Py_RETURN_NOTIMPLEMENTED;
    TensorObject *a = (TensorObject *)va, *b = (TensorObject *)vb;
    int eq = 1;
    if (a->ndim != b->ndim || a->size != b->size) { eq = 0; goto done; }
    for (int i = 0; i < a->ndim; i++)
        if (a->shape[i] != b->shape[i]) { eq = 0; goto done; }
    for (Py_ssize_t i = 0; i < a->size; i++)
        if (a->data[a->offset + i] != b->data[b->offset + i]) { eq = 0; goto done; }
done:
    if (op == Py_NE) eq = !eq;
    return PyBool_FromLong(eq);
}

/* ── GPU methods ──────────────────────────────────────────────────── */

static PyObject *Tensor_to_gpu(TensorObject *self, PyObject *Py_UNUSED(args)) {
    if (!cuda_rt.available) {
        PyErr_SetString(PyExc_RuntimeError, "CUDA not available (no GPU detected)");
        return NULL;
    }
    if (self->on_gpu) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    size_t nbytes = self->size * sizeof(double);
    double *gpu_ptr = NULL;
    int err = cuda_rt.cudaMalloc((void **)&gpu_ptr, nbytes);
    if (err != 0) {
        PyErr_Format(PyExc_RuntimeError, "cudaMalloc failed (error %d)", err);
        return NULL;
    }
    err = cuda_rt.cudaMemcpy(gpu_ptr, self->data + self->offset, nbytes, CUDA_MEMCPY_H2D);
    if (err != 0) {
        cuda_rt.cudaFree(gpu_ptr);
        PyErr_Format(PyExc_RuntimeError, "cudaMemcpy H2D failed (error %d)", err);
        return NULL;
    }

    /* Create a new tensor that references GPU memory */
    TensorObject *gt = Tensor_empty(self->shape, self->ndim, self->dtype);
    if (!gt) { cuda_rt.cudaFree(gpu_ptr); return NULL; }
    gt->gpu_data = gpu_ptr;
    gt->on_gpu = 1;
    /* Keep CPU data as well for fallback */
    memcpy(gt->data, self->data + self->offset, nbytes);
    return (PyObject *)gt;
}

static PyObject *Tensor_to_cpu(TensorObject *self, PyObject *Py_UNUSED(args)) {
    if (!self->on_gpu) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    /* Copy GPU data back to CPU */
    size_t nbytes = self->size * sizeof(double);
    TensorObject *ct = Tensor_empty(self->shape, self->ndim, self->dtype);
    if (!ct) return NULL;
    int err = cuda_rt.cudaMemcpy(ct->data, self->gpu_data, nbytes, CUDA_MEMCPY_D2H);
    if (err != 0) {
        Py_DECREF(ct);
        PyErr_Format(PyExc_RuntimeError, "cudaMemcpy D2H failed (error %d)", err);
        return NULL;
    }
    return (PyObject *)ct;
}

static PyObject *Tensor_gpu_available(PyObject *self, PyObject *Py_UNUSED(args)) {
    return PyBool_FromLong(cuda_rt.available);
}

/* ── class_getitem for Tensor[[3,4], float32] syntax ──────────────── */

typedef struct {
    PyObject_HEAD
    Py_ssize_t shape[32];
    int ndim;
    DTypeObject *dtype;
} TensorFactoryObject;

static PyTypeObject TensorFactoryType;

static void TensorFactory_dealloc(TensorFactoryObject *self) {
    Py_XDECREF(self->dtype);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *TensorFactory_call(TensorFactoryObject *self,
                                     PyObject *args, PyObject *kwargs) {
    PyObject *data;
    if (!PyArg_ParseTuple(args, "O", &data))
        return NULL;

    /* Create shape tuple to pass to Tensor_new */
    PyObject *shape_list = PyList_New(self->ndim);
    for (int i = 0; i < self->ndim; i++)
        PyList_SET_ITEM(shape_list, i, PyLong_FromSsize_t(self->shape[i]));

    PyObject *ctor_args = Py_BuildValue("(OOO)", data, shape_list, self->dtype);
    Py_DECREF(shape_list);
    PyObject *result = Tensor_new(&TensorType, ctor_args, NULL);
    Py_DECREF(ctor_args);
    return result;
}

static PyObject *TensorFactory_repr(TensorFactoryObject *self) {
    PyObject *shape_list = PyList_New(self->ndim);
    for (int i = 0; i < self->ndim; i++)
        PyList_SET_ITEM(shape_list, i, PyLong_FromSsize_t(self->shape[i]));
    PyObject *result = PyUnicode_FromFormat("Tensor[%R, %s]", shape_list, self->dtype->name);
    Py_DECREF(shape_list);
    return result;
}

static PyObject *TensorFactory_zeros(TensorFactoryObject *self, PyObject *Py_UNUSED(args)) {
    return (PyObject *)Tensor_empty(self->shape, self->ndim, self->dtype);
}

static PyObject *TensorFactory_ones(TensorFactoryObject *self, PyObject *Py_UNUSED(args)) {
    TensorObject *t = Tensor_empty(self->shape, self->ndim, self->dtype);
    if (!t) return NULL;
    for (Py_ssize_t i = 0; i < t->size; i++) t->data[i] = 1.0;
    return (PyObject *)t;
}

static PyMethodDef TensorFactory_methods[] = {
    {"zeros", (PyCFunction)TensorFactory_zeros, METH_NOARGS, NULL},
    {"ones", (PyCFunction)TensorFactory_ones, METH_NOARGS, NULL},
    {NULL}
};

static PyTypeObject TensorFactoryType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "xue._tensor_accel._TensorFactory",
    .tp_basicsize = sizeof(TensorFactoryObject),
    .tp_dealloc = (destructor)TensorFactory_dealloc,
    .tp_repr = (reprfunc)TensorFactory_repr,
    .tp_call = (ternaryfunc)TensorFactory_call,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = TensorFactory_methods,
};

static PyObject *Tensor_class_getitem(PyObject *cls, PyObject *params) {
    if (!PyTuple_Check(params) || PyTuple_GET_SIZE(params) != 2) {
        PyErr_SetString(PyExc_TypeError,
            "Tensor requires [shape, dtype], e.g. Tensor[[3, 4], float32]");
        return NULL;
    }

    PyObject *shape_arg = PyTuple_GET_ITEM(params, 0);
    PyObject *dtype_arg = PyTuple_GET_ITEM(params, 1);

    if (!PyObject_TypeCheck(dtype_arg, &DTypeType)) {
        PyErr_SetString(PyExc_TypeError, "Second argument must be a DType");
        return NULL;
    }

    PyObject *shape_seq = PySequence_Fast(shape_arg, "Shape must be a list/tuple");
    if (!shape_seq) return NULL;

    TensorFactoryObject *f = PyObject_New(TensorFactoryObject, &TensorFactoryType);
    if (!f) { Py_DECREF(shape_seq); return NULL; }

    f->ndim = (int)PySequence_Fast_GET_SIZE(shape_seq);
    for (int i = 0; i < f->ndim; i++)
        f->shape[i] = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(shape_seq, i));
    Py_DECREF(shape_seq);

    Py_INCREF(dtype_arg);
    f->dtype = (DTypeObject *)dtype_arg;

    return (PyObject *)f;
}

/* ── Type definition ──────────────────────────────────────────────── */

static PyMethodDef Tensor_methods[] = {
    {"reshape", (PyCFunction)Tensor_reshape, METH_VARARGS, NULL},
    {"transpose", (PyCFunction)Tensor_transpose, METH_NOARGS, NULL},
    {"sum", (PyCFunction)Tensor_sum, METH_VARARGS | METH_KEYWORDS, NULL},
    {"mean", (PyCFunction)Tensor_mean, METH_VARARGS | METH_KEYWORDS, NULL},
    {"max", (PyCFunction)Tensor_max, METH_VARARGS | METH_KEYWORDS, NULL},
    {"min", (PyCFunction)Tensor_min, METH_VARARGS | METH_KEYWORDS, NULL},
    {"tolist", (PyCFunction)Tensor_tolist, METH_NOARGS, NULL},
    {"to_gpu", (PyCFunction)Tensor_to_gpu, METH_NOARGS, "Transfer to GPU."},
    {"to_cpu", (PyCFunction)Tensor_to_cpu, METH_NOARGS, "Transfer to CPU."},
    {"__class_getitem__", (PyCFunction)Tensor_class_getitem, METH_O | METH_CLASS, NULL},
    {NULL}
};

static PyGetSetDef Tensor_getset[] = {
    {"shape", (getter)Tensor_get_shape, NULL, NULL, NULL},
    {"dtype", (getter)Tensor_get_dtype, NULL, NULL, NULL},
    {"ndim", (getter)Tensor_get_ndim, NULL, NULL, NULL},
    {"size", (getter)Tensor_get_size, NULL, NULL, NULL},
    {"T", (getter)Tensor_get_T, NULL, NULL, NULL},
    {NULL}
};

static PyNumberMethods Tensor_as_number = {
    .nb_add = Tensor_nb_add,
    .nb_subtract = Tensor_nb_subtract,
    .nb_multiply = Tensor_nb_multiply,
    .nb_true_divide = Tensor_nb_truediv,
    .nb_negative = Tensor_nb_negative,
    .nb_matrix_multiply = Tensor_matmul,
};

static PyMappingMethods Tensor_as_mapping = {
    .mp_length = (lenfunc)Tensor_length,
    .mp_subscript = (binaryfunc)Tensor_getitem,
    .mp_ass_subscript = (objobjargproc)Tensor_setitem,
};

static PyTypeObject TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "xue._tensor_accel.Tensor",
    .tp_basicsize = sizeof(TensorObject),
    .tp_dealloc = (destructor)Tensor_dealloc,
    .tp_repr = (reprfunc)Tensor_repr,
    .tp_as_number = &Tensor_as_number,
    .tp_as_mapping = &Tensor_as_mapping,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_richcompare = Tensor_richcompare,
    .tp_methods = Tensor_methods,
    .tp_getset = Tensor_getset,
    .tp_new = Tensor_new,
    .tp_doc = "C-accelerated tensor with SIMD and optional CUDA GPU support.",
};

/* ══════════════════════════════════════════════════════════════════
   Module-level factory functions
   ══════════════════════════════════════════════════════════════════ */

static PyObject *mod_zeros(PyObject *self, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"shape", "dtype", NULL};
    PyObject *shape_arg;
    DTypeObject *dtype = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|O!", kwlist,
            &shape_arg, &DTypeType, &dtype))
        return NULL;
    if (!dtype) dtype = dt_float64;

    PyObject *seq = PySequence_Fast(shape_arg, "shape");
    if (!seq) return NULL;
    int ndim = (int)PySequence_Fast_GET_SIZE(seq);
    Py_ssize_t shape[32];
    for (int i = 0; i < ndim; i++)
        shape[i] = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(seq, i));
    Py_DECREF(seq);
    return (PyObject *)Tensor_empty(shape, ndim, dtype);
}

static PyObject *mod_ones(PyObject *self, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"shape", "dtype", NULL};
    PyObject *shape_arg;
    DTypeObject *dtype = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|O!", kwlist,
            &shape_arg, &DTypeType, &dtype))
        return NULL;
    if (!dtype) dtype = dt_float64;

    PyObject *seq = PySequence_Fast(shape_arg, "shape");
    if (!seq) return NULL;
    int ndim = (int)PySequence_Fast_GET_SIZE(seq);
    Py_ssize_t shape[32];
    for (int i = 0; i < ndim; i++)
        shape[i] = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(seq, i));
    Py_DECREF(seq);

    TensorObject *t = Tensor_empty(shape, ndim, dtype);
    if (!t) return NULL;
    for (Py_ssize_t i = 0; i < t->size; i++) t->data[i] = 1.0;
    return (PyObject *)t;
}

static PyObject *mod_eye(PyObject *self, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"n", "dtype", NULL};
    int n;
    DTypeObject *dtype = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "i|O!", kwlist,
            &n, &DTypeType, &dtype))
        return NULL;
    if (!dtype) dtype = dt_float64;

    Py_ssize_t shape[2] = {n, n};
    TensorObject *t = Tensor_empty(shape, 2, dtype);
    if (!t) return NULL;
    for (int i = 0; i < n; i++) t->data[i * n + i] = 1.0;
    return (PyObject *)t;
}

static PyObject *mod_arange(PyObject *self, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"start", "stop", "step", "dtype", NULL};
    double start, stop = -1e308, step = 1.0;
    DTypeObject *dtype = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "d|ddO!", kwlist,
            &start, &stop, &step, &DTypeType, &dtype))
        return NULL;
    if (!dtype) dtype = dt_float64;

    if (stop == -1e308) {
        stop = start;
        start = 0;
    }

    Py_ssize_t count = 0;
    double v = start;
    while (v < stop) { count++; v += step; }

    Py_ssize_t shape[1] = {count};
    TensorObject *t = Tensor_empty(shape, 1, dtype);
    if (!t) return NULL;
    v = start;
    for (Py_ssize_t i = 0; i < count; i++) {
        t->data[i] = v;
        v += step;
    }
    return (PyObject *)t;
}

/* ══════════════════════════════════════════════════════════════════
   Module definition
   ══════════════════════════════════════════════════════════════════ */

static PyMethodDef module_methods[] = {
    {"zeros", (PyCFunction)mod_zeros, METH_VARARGS | METH_KEYWORDS, NULL},
    {"ones", (PyCFunction)mod_ones, METH_VARARGS | METH_KEYWORDS, NULL},
    {"eye", (PyCFunction)mod_eye, METH_VARARGS | METH_KEYWORDS, NULL},
    {"arange", (PyCFunction)mod_arange, METH_VARARGS | METH_KEYWORDS, NULL},
    {"gpu_available", (PyCFunction)Tensor_gpu_available, METH_NOARGS,
     "Check if CUDA GPU is available."},
    {NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_tensor_accel",
    "C-accelerated tensor with SIMD and CUDA GPU support for xue",
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__tensor_accel(void) {
    if (PyType_Ready(&DTypeType) < 0) return NULL;
    if (PyType_Ready(&TensorType) < 0) return NULL;
    if (PyType_Ready(&TensorFactoryType) < 0) return NULL;

    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;

    /* ShapeError */
    ShapeError = PyErr_NewException("xue._tensor_accel.ShapeError",
                                     PyExc_ValueError, NULL);
    Py_INCREF(ShapeError);
    PyModule_AddObject(m, "ShapeError", ShapeError);

    /* DType instances */
    dt_float16 = make_dtype("float16", 2, 'e'); PyModule_AddObject(m, "float16", (PyObject *)dt_float16);
    dt_float32 = make_dtype("float32", 4, 'f'); PyModule_AddObject(m, "float32", (PyObject *)dt_float32);
    dt_float64 = make_dtype("float64", 8, 'd'); PyModule_AddObject(m, "float64", (PyObject *)dt_float64);
    dt_int8 = make_dtype("int8", 1, 'b');       PyModule_AddObject(m, "int8", (PyObject *)dt_int8);
    dt_int16 = make_dtype("int16", 2, 'h');     PyModule_AddObject(m, "int16", (PyObject *)dt_int16);
    dt_int32 = make_dtype("int32", 4, 'i');     PyModule_AddObject(m, "int32", (PyObject *)dt_int32);
    dt_int64 = make_dtype("int64", 8, 'q');     PyModule_AddObject(m, "int64", (PyObject *)dt_int64);
    dt_uint8 = make_dtype("uint8", 1, 'B');     PyModule_AddObject(m, "uint8", (PyObject *)dt_uint8);
    dt_bool = make_dtype("bool", 1, 'b');       PyModule_AddObject(m, "bool_", (PyObject *)dt_bool);

    /* Types */
    Py_INCREF(&DTypeType);
    PyModule_AddObject(m, "DType", (PyObject *)&DTypeType);
    Py_INCREF(&TensorType);
    PyModule_AddObject(m, "Tensor", (PyObject *)&TensorType);

    /* Initialize CUDA runtime */
    init_cuda();
    PyModule_AddIntConstant(m, "CUDA_AVAILABLE", cuda_rt.available);

    return m;
}
