# Xue

Xue (xue-python) is an AI-native enhanced Python distribution from OpenCan.ai, built as a standard CPython fork of 3.12 and 3.14 (free-threaded). It ships an extra standard-library package, `xue`, with 11 modules for scientific computing, AI/ML, robotics, and safety-critical programming that stock Python lacks. All existing Python packages work unchanged; performance-critical modules (units, autodiff, tensor) automatically load C-extension backends with AVX2/NEON SIMD and optional CUDA, falling back to pure Python when unavailable.

## Type System & Safety

### `xue.result` — Result & Option types

Rust-inspired `Result[T, E]` and `Option[T]` types that make error and absence paths explicit in signatures instead of relying on exceptions or silent `None`. Supports pattern matching and method chaining (`map`, `and_then`, `unwrap_or`, `or_else`).

```python
from xue.result import Ok, Err, Some, Nothing

def parse_int(s):
    try:
        return Ok(int(s))
    except ValueError:
        return Err(f"cannot parse '{s}'")

value = parse_int("10").map(lambda x: x * 2).unwrap_or(0)   # 20
name = Some({"name": "Alice"}).map(lambda u: u["name"]).unwrap_or("?")
```

**Use case:** Make every failure path visible at the type level in parsing, I/O, and validation code so callers must handle errors deliberately.

### `xue.secret` — leak-proof secret wrapper

Wraps API keys, tokens, and passwords so they never appear in `repr`, `str`, logs, or JSON. The raw value is only retrievable via an explicit `.expose()` call, and equality uses constant-time comparison to resist timing attacks.

```python
from xue.secret import Secret

api_key = Secret("sk-abc123def456")
print(api_key)                 # Secret(****)
token = api_key.expose()       # "sk-abc123def456"

if api_key == Secret("sk-abc123def456"):   # constant-time compare
    print("Authenticated")
```

**Use case:** Prevent accidental disclosure of credentials through application logs or error reports. Set `XUE_SCRUB_SECRETS=1` to also redact values from tracebacks.

### `xue.contracts` — design by contract

Adds `@requires` (preconditions), `@ensures` (postconditions), and `@invariant` (class invariants) decorators that raise `ContractError` when violated. Checking can be turned off for zero overhead in production.

```python
from xue.contracts import requires, ensures, invariant

@requires(lambda a, b: b != 0, "divisor must not be zero")
@ensures(lambda result, a, b: abs(result * b - a) < 1e-10)
def divide(a, b):
    return a / b

@invariant(lambda self: self.balance >= 0, "balance non-negative")
class Account:
    def __init__(self, balance): self.balance = balance
```

**Use case:** Enforce correctness conditions in safety-critical and financial code; disable with `XUE_CONTRACTS=0` or `set_enabled(False)` for hot paths.

### `xue.strict` — runtime type checking

The `@checked` decorator validates arguments and return values against function annotations at the call boundary, raising `StrictTypeError` on mismatch. `strict_module(__name__)` applies it to an entire module.

```python
from xue.strict import checked, strict_module

@checked
def compute(x: float, y: float) -> float:
    return x + y

compute(1.0, 2.0)     # OK -> 3.0
compute("a", "b")     # StrictTypeError: x: expected float, got str

strict_module(__name__)   # check every annotated function here
```

**Use case:** Catch type mismatches at function boundaries during development; enable everywhere with `XUE_STRICT=1` or `python --strict`.

## Scientific Computing

### `xue.units` — SI physical units with dimension checking

A complete SI unit system with runtime dimensional analysis. Arithmetic on `Quantity` values tracks units and raises `DimensionError` on incompatible operations. A C extension packs the 7 SI dimension exponents into one 64-bit integer, making arithmetic 9–15x faster than pure Python.

```python
from xue.units import meters, seconds, kg, kilometers, degrees, radians

distance = 100 * meters
time = 9.58 * seconds
speed = distance / time          # Quantity(10.44..., m/s)

distance + time                  # DimensionError!
distance.to(kilometers)          # 0.1
(180 * degrees).to(radians)      # 3.14159...
```

**Use case:** Eliminate unit-confusion bugs (the classic meters-vs-feet failure) in robotics, physics, and engineering calculations.

### `xue.autodiff` — reverse-mode automatic differentiation

Computes exact gradients, Jacobians, and higher-order derivatives by building a computation graph through operator overloading — no symbolic math or finite differences. Offers a graph API (`Variable.backward()`) and a functional API (`grad`, `value_and_grad`, `jacobian`). C-accelerated to 9–11x.

```python
from xue.autodiff import Variable, grad, jacobian

x = Variable(3.0, name="x")
y = Variable(2.0, name="y")
z = x ** 2 + 2 * x * y + y ** 2
z.backward()
print(x.grad, y.grad)            # 10.0 10.0

df = grad(lambda x: x ** 3 + 2 * x)
print(df(3.0))                   # 29.0
J = jacobian(lambda v: [v[0] + v[1], v[0] * v[1]], [2.0, 3.0])
```

**Use case:** Train custom models, run gradient-based optimization, and compute sensitivities in scientific code without a full deep-learning framework.

### `xue.tensor` — shape-checked tensors with SIMD + GPU

N-dimensional tensors with runtime shape validation (`ShapeError` on mismatch), AVX2/NEON SIMD-accelerated elementwise ops and tiled matrix multiply, and optional CUDA GPU offload via `.to_gpu()`/`.to_cpu()`. Operations release the GIL for true parallelism on the 3.14t free-threaded build.

```python
from xue.tensor import Tensor, float32, zeros, eye, arange

a = Tensor[[2, 3], float32]([[1,2,3], [4,5,6]])
b = Tensor([[1, 2], [3, 4], [5, 6]])     # (3, 2)
c = a @ b                                # shape-checked (2, 2)
print(c.shape, c.sum(), c.T.shape)

I = eye(3); r = arange(0, 10, 0.5)
gpu = a.to_gpu()                         # CUDA if available
```

**Use case:** High-throughput numerical kernels and ML inference where shape errors must be caught early and SIMD/GPU acceleration matters.

## Language & Runtime Enhancements

### `xue.dispatch` — Julia-style multiple dispatch

The `@multimethod` decorator selects an implementation based on the runtime types of *all* arguments, not just `self`. Multiple functions share a name and are resolved by their annotated signatures.

```python
from xue.dispatch import multimethod

@multimethod
def add(a: int, b: int) -> int: return a + b

@multimethod
def add(a: list, b: list) -> list: return a + b

add(1, 2)         # 3
add([1], [2])     # [1, 2]
```

**Use case:** Express clean numeric/geometric operations that vary by operand type combination without long `isinstance` chains.

### `xue.unicode_ops` — Unicode mathematical operators

Registers a source codec (`# coding: xue-unicode`) that translates Unicode math symbols to Python equivalents during parsing — e.g. ≤ → `<=`, ≠ → `!=`, ∧ → `and`, ÷ → `/`, √ → `math.sqrt`, π/τ to their constants.

```python
# coding: xue-unicode
if x ≤ 10 ∧ y ≠ 0:
    result = x ÷ y

# Or activate programmatically:
import xue.unicode_ops
xue.unicode_ops.register()
```

**Use case:** Write mathematical and scientific code that reads like the equations it implements.

## AI & Security Integration

### `xue.sandbox` — capability-based sandboxed imports

Restricts what untrusted code can do by importing it under a `SandboxPolicy` that gates network, filesystem, subprocess, and ctypes access; forbidden access raises `SandboxViolation`. Also supports a `with sandbox(...)` context manager and hash-based package verification (`register_trusted_hash`, `verify_package`).

```python
from xue.sandbox import sandboxed_import, SandboxPolicy, sandbox

policy = SandboxPolicy(allow_network=False, allow_filesystem=False,
                       allow_subprocess=False, allow_ctypes=False)
plugin = sandboxed_import("untrusted_plugin", policy=policy)

with sandbox(allow_network=False, allow_subprocess=False):
    import some_plugin
```

**Use case:** Safely load third-party plugins or AI-generated code while denying it access to the network, disk, and process spawning.

### `xue.llmhook` — AI-powered diagnostics

Optional LLM integration (Ollama, OpenAI, or a Unix socket backend) that explains exceptions in plain language and answers questions about your code. Zero overhead when disabled.

```python
from xue.llmhook import configure, explain, ask

configure(backend="http",
          url="http://localhost:11434/api/generate", model="llama3")
try:
    result = compute(data)
except Exception as e:
    print(explain(e))            # human-readable fix suggestion

answer = ask("Why is this function returning None?")
```

**Use case:** Get inline, AI-generated explanations and fixes for runtime errors during development; enable with `python --llm` or `XUE_LLM_HOOK=1`.

## Building for macOS, Linux & Windows

Xue builds from source like CPython itself — a Python interpreter cannot be cross-compiled, so build natively on each platform. The result is a relocatable interpreter tree plus the C-accelerated `units`, `autodiff` and `tensor` backends under your chosen prefix.

### Linux

```python
# build deps: gcc, make, libssl/zlib/readline headers
git clone https://github.com/Vern-AllWorks-LLC/xue.git
cd xue/xue-python
./configure --prefix=/opt/xue --enable-optimizations
make -j$(nproc)
sudo make install
tar -czf xue-linux-x86_64.tar.gz -C /opt xue
```

**Tip:** Build on Ubuntu 20.04 / manylinux for broad glibc compatibility. Add `--disable-gil` for the free-threaded 3.14t build.

### macOS

```python
brew install openssl readline xz
git clone https://github.com/Vern-AllWorks-LLC/xue.git
cd xue/xue-python
./configure --prefix=/opt/xue --enable-optimizations \
  --with-openssl=$(brew --prefix openssl)
make -j$(sysctl -n hw.ncpu)
make install
```

**Tip:** Build separately on Apple Silicon and Intel, or pass `--enable-universalsdk --with-universal-archs=universal2` to cover both.

### Windows

```python
:: Visual Studio 2022 with "Desktop development with C++"
git clone https://github.com/Vern-AllWorks-LLC/xue.git
cd xue\xue-python
PCbuild\build.bat -c Release -p x64
:: interpreter + extensions land in PCbuild\amd64\
```

**Tip:** Package the `PCbuild\amd64` output as a .zip, or build an installer with `Tools\msi`.

### Automated builds (GitHub Actions)

An `ubuntu` / `macos-14` / `macos-13` / `windows` runner matrix runs `./configure && make` on Unix and `PCbuild\build.bat` on Windows, then uploads the packaged interpreter as a release artifact.

**Use case:** Ship signed per-OS Xue builds on every tagged release without maintaining build machines.

---

## Documentation

Full documentation: <https://vernallworks.com/docs-xue.php>

## License

See the `LICENSE` file. Built on CPython (PSF License).
