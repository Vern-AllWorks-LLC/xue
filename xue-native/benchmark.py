"""Benchmark: C-accelerated vs pure Python implementations."""
import sys, time, os

# Force pure-Python by hiding C extensions temporarily
XUE_DIR = os.path.dirname(os.path.abspath(__file__))

def bench(name, fn, n=100000):
    """Run fn() n times and return ops/sec."""
    start = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - start
    ops = n / elapsed
    return elapsed, ops

def run_units_bench():
    print("\n=== Units Benchmark (Quantity arithmetic) ===")

    # Pure Python
    sys.modules.pop('xue.units', None)
    sys.modules.pop('xue._units_accel', None)
    import importlib

    # Import pure Python versions directly
    from xue import units as units_mod
    # Get the Python classes before C override
    # We need to reimport without C accel
    import types
    exec_ns = {}
    src = open(os.path.join(os.path.dirname(units_mod.__file__), 'units.py')).read()
    # Remove the C accel override section
    marker = "# ── C-accelerated override"
    idx = src.find(marker)
    if idx >= 0:
        pure_src = src[:idx]
    else:
        pure_src = src
    exec(compile(pure_src, '<pure_units>', 'exec'), exec_ns)

    Py_meters = exec_ns['meters']
    Py_seconds = exec_ns['seconds']
    Py_kg = exec_ns['kg']
    Py_kilometers = exec_ns['kilometers']

    # C-accelerated
    from xue._units_accel import Dimension, Unit, Quantity
    C_L = Dimension((1,0,0,0,0,0,0))
    C_M = Dimension((0,1,0,0,0,0,0))
    C_T = Dimension((0,0,1,0,0,0,0))
    C_meters = Unit(C_L, 1.0, "meter", "m")
    C_seconds = Unit(C_T, 1.0, "second", "s")
    C_kg = Unit(C_M, 1.0, "kilogram", "kg")
    C_kilometers = Unit(C_L, 1000.0, "kilometer", "km")

    N = 200000

    # Quantity creation
    t1, ops1 = bench("Py create", lambda: 100.0 * Py_meters, N)
    t2, ops2 = bench("C  create", lambda: 100.0 * C_meters, N)
    print(f"  Create:   Python {ops1:>10.0f} ops/s | C {ops2:>10.0f} ops/s | speedup {ops2/ops1:.1f}x")

    # Quantity add
    pd = 100.0 * Py_meters
    cd = Quantity(100.0, C_meters)
    pd2 = 50.0 * Py_meters
    cd2 = Quantity(50.0, C_meters)
    t1, ops1 = bench("Py add", lambda: pd + pd2, N)
    t2, ops2 = bench("C  add", lambda: cd + cd2, N)
    print(f"  Add:      Python {ops1:>10.0f} ops/s | C {ops2:>10.0f} ops/s | speedup {ops2/ops1:.1f}x")

    # Quantity mul
    pm = 10.0 * Py_kg
    cm = Quantity(10.0, C_kg)
    t1, ops1 = bench("Py mul", lambda: pd * pm, N)
    t2, ops2 = bench("C  mul", lambda: cd * cm, N)
    print(f"  Mul:      Python {ops1:>10.0f} ops/s | C {ops2:>10.0f} ops/s | speedup {ops2/ops1:.1f}x")

    # Quantity div
    pt = 2.0 * Py_seconds
    ct = Quantity(2.0, C_seconds)
    t1, ops1 = bench("Py div", lambda: pd / pt, N)
    t2, ops2 = bench("C  div", lambda: cd / ct, N)
    print(f"  Div:      Python {ops1:>10.0f} ops/s | C {ops2:>10.0f} ops/s | speedup {ops2/ops1:.1f}x")

    # Conversion
    pkm = 5000.0 * Py_meters
    ckm = Quantity(5000.0, C_meters)
    t1, ops1 = bench("Py conv", lambda: pkm.to(Py_kilometers), N)
    t2, ops2 = bench("C  conv", lambda: ckm.to(C_kilometers), N)
    print(f"  Convert:  Python {ops1:>10.0f} ops/s | C {ops2:>10.0f} ops/s | speedup {ops2/ops1:.1f}x")

def run_autodiff_bench():
    print("\n=== Autodiff Benchmark (Variable arithmetic + backward) ===")

    # Pure Python Variable
    src = open(os.path.join(os.path.dirname(__file__), '..', 'xue-python', 'Lib', 'xue', 'autodiff.py')).read()
    marker = "# ── C-accelerated override"
    idx = src.find(marker)
    if idx >= 0:
        pure_src = src[:idx]
    else:
        pure_src = src
    exec_ns = {}
    exec(compile(pure_src, '<pure_autodiff>', 'exec'), exec_ns)
    PyVariable = exec_ns['Variable']

    # C Variable
    from xue._autodiff_accel import Variable as CVariable

    N = 100000

    # Simple expression: x^2 + 2*x + 1, backward
    def py_expr():
        x = PyVariable(3.0)
        y = x ** 2 + 2 * x + 1
        y.backward()
        return x.grad

    def c_expr():
        x = CVariable(3.0)
        y = x ** 2 + 2 * x + 1
        y.backward()
        return x.grad

    t1, ops1 = bench("Py expr+backward", py_expr, N)
    t2, ops2 = bench("C  expr+backward", c_expr, N)
    print(f"  x²+2x+1: Python {ops1:>10.0f} ops/s | C {ops2:>10.0f} ops/s | speedup {ops2/ops1:.1f}x")

    # Larger expression: MLP-like
    def py_mlp():
        x = PyVariable(0.5)
        h = (x * PyVariable(0.3) + PyVariable(0.1)).tanh()
        o = (h * PyVariable(0.7) + PyVariable(-0.2)).sigmoid()
        o.backward()
        return x.grad

    def c_mlp():
        x = CVariable(0.5)
        h = (x * CVariable(0.3) + CVariable(0.1)).tanh()
        o = (h * CVariable(0.7) + CVariable(-0.2)).sigmoid()
        o.backward()
        return x.grad

    t1, ops1 = bench("Py MLP-like", py_mlp, N)
    t2, ops2 = bench("C  MLP-like", c_mlp, N)
    print(f"  MLP-like: Python {ops1:>10.0f} ops/s | C {ops2:>10.0f} ops/s | speedup {ops2/ops1:.1f}x")

def run_tensor_bench():
    print("\n=== Tensor Benchmark (SIMD elementwise + matmul) ===")

    src = open(os.path.join(os.path.dirname(__file__), '..', 'xue-python', 'Lib', 'xue', 'tensor.py')).read()
    marker = "# ── C-accelerated override"
    idx = src.find(marker)
    if idx >= 0:
        pure_src = src[:idx]
    else:
        pure_src = src
    exec_ns = {}
    exec(compile(pure_src, '<pure_tensor>', 'exec'), exec_ns)
    PyTensor = exec_ns['Tensor']
    Py_float64 = exec_ns['float64']

    from xue._tensor_accel import Tensor as CTensor, float64 as C_float64

    # Small tensors (100x100)
    import random
    random.seed(42)
    data_100 = [[random.random() for _ in range(100)] for _ in range(100)]

    pa = PyTensor(data=data_100, dtype=Py_float64)
    pb = PyTensor(data=data_100, dtype=Py_float64)
    ca = CTensor(data=data_100)
    cb = CTensor(data=data_100)

    N_small = 1000

    t1, ops1 = bench("Py add 100x100", lambda: pa + pb, N_small)
    t2, ops2 = bench("C  add 100x100", lambda: ca + cb, N_small)
    print(f"  Add 100x100:  Python {ops1:>8.0f} ops/s | C {ops2:>8.0f} ops/s | speedup {ops2/ops1:.1f}x")

    t1, ops1 = bench("Py mul 100x100", lambda: pa * pb, N_small)
    t2, ops2 = bench("C  mul 100x100", lambda: ca * cb, N_small)
    print(f"  Mul 100x100:  Python {ops1:>8.0f} ops/s | C {ops2:>8.0f} ops/s | speedup {ops2/ops1:.1f}x")

    # Matmul
    N_mm = 100
    t1, ops1 = bench("Py matmul 100x100", lambda: pa @ pb, N_mm)
    t2, ops2 = bench("C  matmul 100x100", lambda: ca @ cb, N_mm)
    print(f"  Matmul 100x100: Python {ops1:>6.0f} ops/s | C {ops2:>6.0f} ops/s | speedup {ops2/ops1:.1f}x")

    # Larger matmul (256x256)
    data_256 = [[random.random() for _ in range(256)] for _ in range(256)]
    pa256 = PyTensor(data=data_256, dtype=Py_float64)
    ca256 = CTensor(data=data_256)
    N_big = 10
    t1, ops1 = bench("Py matmul 256x256", lambda: pa256 @ pa256, N_big)
    t2, ops2 = bench("C  matmul 256x256", lambda: ca256 @ ca256, N_big)
    print(f"  Matmul 256x256: Python {ops1:>6.1f} ops/s | C {ops2:>6.1f} ops/s | speedup {ops2/ops1:.1f}x")

    # Sum reduction
    t1, ops1 = bench("Py sum 100x100", lambda: pa.sum(), N_small)
    t2, ops2 = bench("C  sum 100x100", lambda: ca.sum(), N_small)
    print(f"  Sum 100x100:  Python {ops1:>8.0f} ops/s | C {ops2:>8.0f} ops/s | speedup {ops2/ops1:.1f}x")


if __name__ == "__main__":
    print(f"Python: {sys.version}")
    run_units_bench()
    run_autodiff_bench()
    run_tensor_bench()

    # GPU status
    print("\n=== GPU Status ===")
    from xue._tensor_accel import Tensor
    try:
        t = Tensor(data=[[1,2],[3,4]])
        gt = t.to_gpu()
        print("  GPU: Available, transfer succeeded")
    except RuntimeError as e:
        print(f"  GPU: {e}")
