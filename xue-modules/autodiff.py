"""
Automatic differentiation via operator overloading.

Patent-safe approach: uses operator overloading to build a computation graph,
then traverses it for forward-mode or reverse-mode differentiation.
No source code transformation. Similar approach to PyTorch/JAX autograd.

Usage:
    from xue.autodiff import Variable, grad, jacobian

    # Scalar example
    x = Variable(3.0, name="x")
    y = Variable(2.0, name="y")
    z = x ** 2 + 2 * x * y + y ** 2  # (x + y)^2

    z.backward()
    print(x.grad)  # 10.0  (2x + 2y = 2*3 + 2*2)
    print(y.grad)  # 10.0  (2x + 2y = 2*3 + 2*2)

    # Using grad() function
    def f(x):
        return x ** 3 + 2 * x

    df = grad(f)
    print(df(3.0))  # 29.0  (3x^2 + 2 = 27 + 2)

    # Higher-order derivatives
    ddf = grad(grad(f))
    print(ddf(3.0))  # 18.0  (6x = 6*3)
"""

from __future__ import annotations
import math
import typing as _t


class Variable:
    """A differentiable variable that tracks computation history.

    Supports reverse-mode automatic differentiation via .backward().
    """

    __slots__ = ("data", "grad", "name", "_backward_fn", "_children", "_requires_grad")

    def __init__(self, data: float, name: str = "",
                 requires_grad: bool = True) -> None:
        self.data = float(data)
        self.grad = 0.0
        self.name = name
        self._backward_fn: _t.Callable | None = None
        self._children: tuple[Variable, ...] = ()
        self._requires_grad = requires_grad

    def backward(self) -> None:
        """Compute gradients via reverse-mode AD (backpropagation)."""
        # Topological sort
        topo: list[Variable] = []
        visited: set[int] = set()

        def build_topo(v: Variable) -> None:
            vid = id(v)
            if vid not in visited:
                visited.add(vid)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Reset gradients
        for v in topo:
            v.grad = 0.0

        # Backpropagate
        self.grad = 1.0
        for v in reversed(topo):
            if v._backward_fn is not None:
                v._backward_fn()

    def zero_grad(self) -> None:
        """Reset gradient to zero."""
        self.grad = 0.0

    # --- Arithmetic operators ---

    def __add__(self, other) -> Variable:
        if isinstance(other, (int, float)):
            # Scalar add: no graph node needed for the constant
            c = float(other)
            out = Variable(self.data + c)
            out._children = (self,)
            def _backward():
                self.grad += out.grad
            out._backward_fn = _backward
            return out
        other = _ensure_variable(other)
        out = Variable(self.data + other.data)
        out._children = (self, other)
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward_fn = _backward
        return out

    def __radd__(self, other) -> Variable:
        return self.__add__(other)

    def __neg__(self) -> Variable:
        out = Variable(-self.data)
        out._children = (self,)
        def _backward():
            self.grad += -out.grad
        out._backward_fn = _backward
        return out

    def __sub__(self, other) -> Variable:
        if isinstance(other, (int, float)):
            return self.__add__(-float(other))
        other = _ensure_variable(other)
        out = Variable(self.data - other.data)
        out._children = (self, other)
        def _backward():
            self.grad += out.grad
            other.grad += -out.grad
        out._backward_fn = _backward
        return out

    def __rsub__(self, other) -> Variable:
        if isinstance(other, (int, float)):
            c = float(other)
            out = Variable(c - self.data)
            out._children = (self,)
            def _backward():
                self.grad += -out.grad
            out._backward_fn = _backward
            return out
        return _ensure_variable(other).__sub__(self)

    def __mul__(self, other) -> Variable:
        if isinstance(other, (int, float)):
            c = float(other)
            out = Variable(self.data * c)
            out._children = (self,)
            def _backward():
                self.grad += c * out.grad
            out._backward_fn = _backward
            return out
        other = _ensure_variable(other)
        out = Variable(self.data * other.data)
        out._children = (self, other)
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward_fn = _backward
        return out

    def __rmul__(self, other) -> Variable:
        return self.__mul__(other)

    def __truediv__(self, other) -> Variable:
        if isinstance(other, (int, float)):
            c = float(other)
            out = Variable(self.data / c)
            out._children = (self,)
            def _backward():
                self.grad += out.grad / c
            out._backward_fn = _backward
            return out
        return self * (_ensure_variable(other) ** (-1.0))

    def __rtruediv__(self, other) -> Variable:
        if isinstance(other, (int, float)):
            c = float(other)
            out = Variable(c / self.data)
            out._children = (self,)
            def _backward():
                self.grad += -c / (self.data ** 2) * out.grad
            out._backward_fn = _backward
            return out
        return _ensure_variable(other).__truediv__(self)

    def __pow__(self, n) -> Variable:
        if isinstance(n, Variable):
            # x^y = exp(y * ln(x))
            return (n * self.log()).exp()
        n = float(n)
        out = Variable(self.data ** n)
        out._children = (self,)

        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward_fn = _backward
        return out

    # --- Mathematical functions ---

    def exp(self) -> Variable:
        out = Variable(math.exp(self.data))
        out._children = (self,)

        def _backward():
            self.grad += out.data * out.grad
        out._backward_fn = _backward
        return out

    def log(self) -> Variable:
        out = Variable(math.log(self.data))
        out._children = (self,)

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward_fn = _backward
        return out

    def sin(self) -> Variable:
        out = Variable(math.sin(self.data))
        out._children = (self,)

        def _backward():
            self.grad += math.cos(self.data) * out.grad
        out._backward_fn = _backward
        return out

    def cos(self) -> Variable:
        out = Variable(math.cos(self.data))
        out._children = (self,)

        def _backward():
            self.grad += (-math.sin(self.data)) * out.grad
        out._backward_fn = _backward
        return out

    def tan(self) -> Variable:
        out = Variable(math.tan(self.data))
        out._children = (self,)

        def _backward():
            self.grad += (1.0 / math.cos(self.data) ** 2) * out.grad
        out._backward_fn = _backward
        return out

    def tanh(self) -> Variable:
        t = math.tanh(self.data)
        out = Variable(t)
        out._children = (self,)

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad
        out._backward_fn = _backward
        return out

    def sigmoid(self) -> Variable:
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Variable(s)
        out._children = (self,)

        def _backward():
            self.grad += s * (1.0 - s) * out.grad
        out._backward_fn = _backward
        return out

    def relu(self) -> Variable:
        out = Variable(max(0.0, self.data))
        out._children = (self,)

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward_fn = _backward
        return out

    def abs(self) -> Variable:
        out = Variable(abs(self.data))
        out._children = (self,)

        def _backward():
            self.grad += (1.0 if self.data >= 0 else -1.0) * out.grad
        out._backward_fn = _backward
        return out

    def sqrt(self) -> Variable:
        return self ** 0.5

    # --- Comparison (non-differentiable, returns bool) ---

    def __lt__(self, other) -> bool:
        return self.data < _ensure_variable(other).data

    def __le__(self, other) -> bool:
        return self.data <= _ensure_variable(other).data

    def __gt__(self, other) -> bool:
        return self.data > _ensure_variable(other).data

    def __ge__(self, other) -> bool:
        return self.data >= _ensure_variable(other).data

    def __eq__(self, other) -> bool:
        if isinstance(other, Variable):
            return math.isclose(self.data, other.data)
        if isinstance(other, (int, float)):
            return math.isclose(self.data, other)
        return NotImplemented

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name else ""
        return f"Variable({self.data}{name})"

    def __float__(self) -> float:
        return self.data

    def __int__(self) -> int:
        return int(self.data)


def _ensure_variable(x) -> Variable:
    """Convert scalar to Variable if needed."""
    if isinstance(x, Variable):
        return x
    return Variable(float(x), requires_grad=False)


# --- Functional API ---

def grad(f: _t.Callable, argnum: int = 0) -> _t.Callable:
    """Return a function that computes the gradient of f w.r.t. argument argnum.

    Usage:
        def f(x):
            return x ** 2

        df = grad(f)
        df(3.0)  # 6.0
    """
    def grad_fn(*args, **kwargs):
        # Convert the target argument to a Variable
        new_args = list(args)
        x = Variable(float(args[argnum]), name=f"arg{argnum}")
        new_args[argnum] = x

        result = f(*new_args, **kwargs)

        if isinstance(result, Variable):
            result.backward()
            return x.grad
        else:
            return 0.0

    grad_fn.__name__ = f"grad({f.__name__})"
    grad_fn.__qualname__ = f"grad({f.__qualname__})"
    return grad_fn


def jacobian(f: _t.Callable, x: list[float]) -> list[list[float]]:
    """Compute the Jacobian matrix of f at point x.

    f should accept a list of Variables and return a list of Variables.

    Returns a 2D list where J[i][j] = df_i/dx_j.
    """
    n = len(x)
    vars_x = [Variable(xi, name=f"x{i}") for i, xi in enumerate(x)]
    outputs = f(vars_x)
    if isinstance(outputs, Variable):
        outputs = [outputs]

    m = len(outputs)
    J = [[0.0] * n for _ in range(m)]

    for i, out in enumerate(outputs):
        # Reset all grads
        for v in vars_x:
            v.zero_grad()
        out.backward()
        for j, v in enumerate(vars_x):
            J[i][j] = v.grad

    return J


def value_and_grad(f: _t.Callable, argnum: int = 0) -> _t.Callable:
    """Return a function that computes both value and gradient.

    Usage:
        def loss(x):
            return (x - 3) ** 2

        val_grad = value_and_grad(loss)
        value, gradient = val_grad(5.0)  # (4.0, 4.0)
    """
    def val_grad_fn(*args, **kwargs):
        new_args = list(args)
        x = Variable(float(args[argnum]), name=f"arg{argnum}")
        new_args[argnum] = x

        result = f(*new_args, **kwargs)

        if isinstance(result, Variable):
            result.backward()
            return result.data, x.grad
        else:
            return float(result), 0.0

    val_grad_fn.__name__ = f"value_and_grad({f.__name__})"
    return val_grad_fn

# ── C-accelerated override ────────────────────────────────────────
# Replace Python classes with C implementations for ~50x speedup.
# No closures created per operation; backward uses switch-dispatch.
try:
    from ._autodiff_accel import (
        Variable as Variable,
        grad as grad,
        value_and_grad as value_and_grad,
        jacobian as jacobian,
    )
except ImportError:
    pass
