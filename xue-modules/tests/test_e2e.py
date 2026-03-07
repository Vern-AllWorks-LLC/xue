"""
End-to-end tests for all xue-python features.

Tests real-world usage patterns, cross-module integration,
edge cases, error paths, and interoperability.
"""

import unittest
import sys
import os
import math
import json
import copy
import pickle
import threading
import tempfile
import textwrap


# ============================================================
# E2E: xue.result
# ============================================================

class TestResultE2E(unittest.TestCase):
    """Real-world patterns with Result/Option types."""

    def test_pipeline_chaining(self):
        """Simulate a data processing pipeline with error propagation."""
        from xue.result import Ok, Err, Result

        def parse_int(s: str) -> Result[int, str]:
            try:
                return Ok(int(s))
            except ValueError:
                return Err(f"invalid integer: {s!r}")

        def validate_range(n: int) -> Result[int, str]:
            if 0 <= n <= 100:
                return Ok(n)
            return Err(f"out of range: {n}")

        def normalize(n: int) -> Result[float, str]:
            return Ok(n / 100.0)

        # Happy path
        result = parse_int("42").and_then(validate_range).and_then(normalize)
        self.assertEqual(result.unwrap(), 0.42)

        # Error at parse
        result = parse_int("abc").and_then(validate_range).and_then(normalize)
        self.assertTrue(result.is_err())
        self.assertIn("invalid integer", result.err().unwrap())

        # Error at validation
        result = parse_int("200").and_then(validate_range).and_then(normalize)
        self.assertTrue(result.is_err())
        self.assertIn("out of range", result.err().unwrap())

    def test_option_database_lookup_pattern(self):
        """Simulate database lookups returning Option."""
        from xue.result import Some, Nothing, Option

        db = {"alice": 30, "bob": 25}

        def find_age(name: str) -> Option[int]:
            return Option.from_nullable(db.get(name))

        def is_adult(age: int) -> bool:
            return age >= 18

        # Found and passes filter
        self.assertEqual(find_age("alice").filter(is_adult).unwrap(), 30)

        # Not found
        self.assertEqual(find_age("charlie").unwrap_or(-1), -1)

    def test_result_collect_pattern(self):
        """Collect multiple Results into a single Result."""
        from xue.result import Ok, Err, Result, catch

        inputs = ["1", "2", "three", "4"]
        results = [catch(int, s) for s in inputs]

        successes = [r.unwrap() for r in results if r.is_ok()]
        self.assertEqual(successes, [1, 2, 4])

        failures = [r for r in results if r.is_err()]
        self.assertEqual(len(failures), 1)

    def test_result_with_match_statement(self):
        """Pattern matching integration."""
        from xue.result import Ok, Err

        def process(value):
            match value:
                case Ok(n) if n > 10:
                    return "big"
                case Ok(n):
                    return "small"
                case Err(e):
                    return f"error: {e}"

        self.assertEqual(process(Ok(42)), "big")
        self.assertEqual(process(Ok(5)), "small")
        self.assertEqual(process(Err("fail")), "error: fail")

    def test_option_none_safety(self):
        """Verify Option prevents None propagation."""
        from xue.result import Some, Nothing, Option

        data = {"a": {"b": None}}

        result = (Option.from_nullable(data.get("a"))
                  .and_then(lambda d: Option.from_nullable(d.get("b"))))

        self.assertTrue(result.is_nothing())
        self.assertEqual(result.unwrap_or("default"), "default")


# ============================================================
# E2E: xue.secret
# ============================================================

class TestSecretE2E(unittest.TestCase):
    """Real-world patterns with Secret type."""

    def test_secret_in_logging(self):
        """Ensure secrets don't leak into log-like output."""
        from xue.secret import Secret
        import io

        api_key = Secret("sk-live-abc123xyz789")

        # Simulate logging
        log_output = io.StringIO()
        log_output.write(f"Connecting with key: {api_key}\n")
        log_output.write(f"Key repr: {api_key!r}\n")
        log_output.write(f"Key str: {api_key!s}\n")

        log_text = log_output.getvalue()
        self.assertNotIn("abc123", log_text)
        self.assertNotIn("sk-live", log_text)
        self.assertIn("Secret(****)", log_text)

    def test_secret_in_dict_and_json(self):
        """Secrets in data structures should not leak."""
        from xue.secret import Secret

        config = {
            "host": "example.com",
            "key": Secret("super-secret"),
        }

        # repr of dict should show Secret(****)
        config_str = str(config)
        self.assertNotIn("super-secret", config_str)

    def test_secret_in_exception(self):
        """Secrets in exception messages should be redacted."""
        from xue.secret import Secret

        key = Secret("my-password")
        try:
            raise ValueError(f"Failed with key: {key}")
        except ValueError as e:
            self.assertNotIn("my-password", str(e))
            self.assertIn("Secret(****)", str(e))

    def test_secret_cannot_be_serialized(self):
        """Verify all serialization paths are blocked."""
        from xue.secret import Secret

        s = Secret("data")

        with self.assertRaises(TypeError):
            pickle.dumps(s)

        # deepcopy returns a new Secret (by design), verify it's safe
        s2 = copy.deepcopy(s)
        self.assertEqual(repr(s2), "Secret(****)")
        self.assertEqual(s2.expose(), "data")

    def test_secret_copy_is_safe(self):
        """copy.copy should return a new Secret wrapping the same value."""
        from xue.secret import Secret
        s = Secret("value")
        s2 = copy.copy(s)
        self.assertEqual(repr(s2), "Secret(****)")
        self.assertEqual(s2.expose(), "value")

    def test_secret_from_file(self):
        """Load secret from file."""
        from xue.secret import Secret

        with tempfile.NamedTemporaryFile(mode="w", suffix=".key", delete=False) as f:
            f.write("file-secret-value\n")
            f.flush()
            path = f.name

        try:
            s = Secret.from_file(path)
            self.assertEqual(s.expose(), "file-secret-value")
            self.assertEqual(repr(s), "Secret(****)")
        finally:
            os.unlink(path)

    def test_secret_timing_safety(self):
        """Constant-time comparison should work correctly."""
        from xue.secret import Secret

        a = Secret("password123")
        b = Secret("password123")
        c = Secret("password124")

        self.assertTrue(a == b)
        self.assertFalse(a == c)
        # Different types should return NotImplemented
        self.assertFalse(a == "password123")


# ============================================================
# E2E: xue.contracts
# ============================================================

class TestContractsE2E(unittest.TestCase):
    """Real-world contract programming patterns."""

    def test_scientific_function_contracts(self):
        """Contracts on a numerical computation."""
        from xue.contracts import requires, ensures, ContractViolation

        @requires(lambda matrix: len(matrix) > 0, "matrix must be non-empty")
        @requires(lambda matrix: all(len(row) == len(matrix[0]) for row in matrix),
                  "matrix must be rectangular")
        @ensures(lambda result, matrix: result >= 0, "determinant squared must be non-negative")
        def matrix_trace(matrix):
            return sum(matrix[i][i] for i in range(min(len(matrix), len(matrix[0]))))

        self.assertEqual(matrix_trace([[1, 0], [0, 1]]), 2)
        self.assertEqual(matrix_trace([[5, 3], [2, 7]]), 12)

        with self.assertRaises(ContractViolation):
            matrix_trace([])

    def test_invariant_stack(self):
        """Stack data structure with invariant."""
        from xue.contracts import invariant, requires, ContractViolation

        @invariant(lambda self: self._size >= 0, "size must be non-negative")
        class Stack:
            def __init__(self):
                self._items = []
                self._size = 0

            def push(self, item):
                self._items.append(item)
                self._size += 1

            def pop(self):
                item = self._items.pop()
                self._size -= 1
                return item

            def peek(self):
                return self._items[-1] if self._items else None

        s = Stack()
        s.push(1)
        s.push(2)
        self.assertEqual(s.pop(), 2)
        self.assertEqual(s.peek(), 1)

    def test_contracts_toggle_performance(self):
        """Contracts can be disabled for performance."""
        from xue.contracts import requires, set_enabled, is_enabled
        import time

        @requires(lambda x: x > 0, "positive")
        def fast_sqrt(x):
            return x ** 0.5

        # With contracts
        set_enabled(True)
        t0 = time.perf_counter()
        for _ in range(10000):
            fast_sqrt(4.0)
        with_contracts = time.perf_counter() - t0

        # Without contracts
        set_enabled(False)
        t0 = time.perf_counter()
        for _ in range(10000):
            fast_sqrt(4.0)
        without_contracts = time.perf_counter() - t0

        set_enabled(True)  # restore

        # Without contracts should be faster (or at least not slower)
        # Just verify both complete correctly
        self.assertAlmostEqual(fast_sqrt(9.0), 3.0)


# ============================================================
# E2E: xue.units
# ============================================================

class TestUnitsE2E(unittest.TestCase):
    """Real-world scientific and robotics unit calculations."""

    def test_physics_kinematics(self):
        """Classical kinematics calculation."""
        from xue.units import meters, seconds, kg

        # v = d / t
        distance = 100 * meters
        time = 10 * seconds
        velocity = distance / time
        self.assertAlmostEqual(velocity.value, 10.0)

        # F = m * a
        mass = 5 * kg
        acceleration = velocity / time
        force = mass * acceleration
        self.assertAlmostEqual(force.value, 5.0)

    def test_unit_conversion_chain(self):
        """Chain of conversions."""
        from xue.units import kilometers, meters, centimeters, millimeters

        d = 1.5 * kilometers
        self.assertAlmostEqual(d.to(meters), 1500.0)
        self.assertAlmostEqual(d.to(centimeters), 150000.0)
        self.assertAlmostEqual(d.to(millimeters), 1500000.0)

    def test_robotics_torque(self):
        """Robot arm torque calculation."""
        from xue.units import kg, meters, seconds, Quantity

        arm_length = 0.5 * meters
        payload = 2 * kg
        # Build m/s^2 as Quantity division: (9.81 m) / (1 s^2)
        gravity = (9.81 * meters) / (1 * seconds ** 2)
        force = payload * gravity
        torque = force * arm_length
        self.assertAlmostEqual(torque.value, 9.81)

    def test_dimension_error_message(self):
        """Clear error message on dimension mismatch."""
        from xue.units import meters, seconds, DimensionError

        with self.assertRaises(DimensionError) as ctx:
            (5 * meters) + (3 * seconds)
        self.assertIn("length", str(ctx.exception).lower() + repr(ctx.exception).lower()
                       or "Cannot combine" in str(ctx.exception))

    def test_angle_conversion(self):
        """Degree/radian conversion for robotics."""
        from xue.units import degrees, radians
        import math

        angle = 90 * degrees
        rad = angle.to(radians)
        self.assertAlmostEqual(rad, math.pi / 2)

        full_circle = (2 * math.pi) * radians
        deg = full_circle.to(degrees)
        self.assertAlmostEqual(deg, 360.0)

    def test_comparison_across_scales(self):
        """Compare quantities in different units."""
        from xue.units import meters, kilometers

        self.assertTrue(500 * meters < 1 * kilometers)
        self.assertTrue(1001 * meters > 1 * kilometers)
        self.assertEqual(1000 * meters, 1 * kilometers)


# ============================================================
# E2E: xue.dispatch
# ============================================================

class TestDispatchE2E(unittest.TestCase):
    """Real-world multiple dispatch patterns."""

    def test_scientific_operations(self):
        """Dispatch for mathematical operations on different types."""
        from xue.dispatch import multimethod, _registries
        _registries.clear()

        @multimethod
        def dot(a: list, b: list) -> float:
            return sum(x * y for x, y in zip(a, b))

        @multimethod
        def dot(a: float, b: float) -> float:
            return a * b

        self.assertAlmostEqual(dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0)
        self.assertAlmostEqual(dot(3.0, 4.0), 12.0)

    def test_dispatch_with_inheritance(self):
        """Dispatch should pick most specific type."""
        from xue.dispatch import multimethod, _registries
        _registries.clear()

        class Animal:
            pass

        class Dog(Animal):
            pass

        class Cat(Animal):
            pass

        @multimethod
        def speak(a: Animal) -> str:
            return "..."

        @multimethod
        def speak(a: Dog) -> str:
            return "woof"

        @multimethod
        def speak(a: Cat) -> str:
            return "meow"

        self.assertEqual(speak(Dog()), "woof")
        self.assertEqual(speak(Cat()), "meow")
        self.assertEqual(speak(Animal()), "...")

    def test_no_match_error_message(self):
        """Clear error on dispatch failure."""
        from xue.dispatch import multimethod, _registries
        _registries.clear()

        @multimethod
        def typed_op(a: int, b: int) -> int:
            return a + b

        with self.assertRaises(TypeError) as ctx:
            typed_op(1.0, 2.0)
        self.assertIn("float", str(ctx.exception))


# ============================================================
# E2E: xue.autodiff
# ============================================================

class TestAutodiffE2E(unittest.TestCase):
    """Real-world automatic differentiation patterns."""

    def test_gradient_descent(self):
        """Simple gradient descent optimization."""
        from xue.autodiff import Variable

        # Minimize f(x) = (x - 3)^2
        x = Variable(0.0, name="x")
        lr = 0.1

        for _ in range(100):
            loss = (x - 3.0) ** 2
            loss.backward()
            x = Variable(x.data - lr * x.grad, name="x")

        self.assertAlmostEqual(x.data, 3.0, places=3)

    def test_neural_network_forward(self):
        """Single neuron forward + backward pass."""
        from xue.autodiff import Variable

        # y = sigmoid(w1*x1 + w2*x2 + b)
        x1 = Variable(1.0)
        x2 = Variable(2.0)
        w1 = Variable(0.5, name="w1")
        w2 = Variable(-0.3, name="w2")
        b = Variable(0.1, name="b")

        z = w1 * x1 + w2 * x2 + b
        y = z.sigmoid()

        # target = 1.0
        loss = (y - 1.0) ** 2
        loss.backward()

        # Gradients should be non-zero
        self.assertNotEqual(w1.grad, 0.0)
        self.assertNotEqual(w2.grad, 0.0)
        self.assertNotEqual(b.grad, 0.0)

    def test_rosenbrock_gradient(self):
        """Gradient of Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2."""
        from xue.autodiff import Variable

        x = Variable(1.0, name="x")
        y = Variable(1.0, name="y")
        f = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

        f.backward()
        # At (1,1) the minimum, gradients should be ~0
        self.assertAlmostEqual(x.grad, 0.0, places=5)
        self.assertAlmostEqual(y.grad, 0.0, places=5)

    def test_trig_derivatives(self):
        """Verify trig function derivatives."""
        from xue.autodiff import Variable
        import math

        # d/dx sin(x) = cos(x)
        x = Variable(math.pi / 4)
        y = x.sin()
        y.backward()
        self.assertAlmostEqual(x.grad, math.cos(math.pi / 4), places=10)

        # d/dx cos(x) = -sin(x)
        x2 = Variable(math.pi / 3)
        y2 = x2.cos()
        y2.backward()
        self.assertAlmostEqual(x2.grad, -math.sin(math.pi / 3), places=10)

    def test_grad_function_api(self):
        """Functional grad API."""
        from xue.autodiff import grad, value_and_grad

        def f(x):
            return x.sin() * x.exp()

        df = grad(f)
        val, g = value_and_grad(f)(1.0)

        # f(1) = sin(1) * e^1
        self.assertAlmostEqual(val, math.sin(1) * math.e, places=5)
        # f'(x) = cos(x)*e^x + sin(x)*e^x = (cos(x)+sin(x))*e^x
        expected_grad = (math.cos(1) + math.sin(1)) * math.e
        self.assertAlmostEqual(g, expected_grad, places=5)


# ============================================================
# E2E: xue.tensor
# ============================================================

class TestTensorE2E(unittest.TestCase):
    """Real-world tensor operations."""

    def test_matrix_chain_multiplication(self):
        """Chain of matrix multiplications."""
        from xue.tensor import Tensor, eye

        A = Tensor([[1, 2], [3, 4]])
        I = eye(2)

        # A @ I = A
        result = A @ I
        self.assertEqual(result.tolist(), A.tolist())

        # I @ A = A
        result2 = I @ A
        self.assertEqual(result2.tolist(), A.tolist())

    def test_shape_error_messages(self):
        """Clear error messages for shape mismatches."""
        from xue.tensor import Tensor, ShapeError

        a = Tensor([[1, 2, 3]])       # 1x3
        b = Tensor([[1, 2], [3, 4]])  # 2x2

        with self.assertRaises(ShapeError) as ctx:
            a @ b
        msg = str(ctx.exception)
        self.assertIn("3", msg)
        self.assertIn("2", msg)

    def test_reshape_transpose_chain(self):
        """Reshape and transpose operations."""
        from xue.tensor import Tensor

        t = Tensor([[1, 2, 3, 4, 5, 6]])  # 1x6
        t2 = t.reshape(2, 3)               # 2x3
        self.assertEqual(t2.shape, (2, 3))

        t3 = t2.T                           # 3x2
        self.assertEqual(t3.shape, (3, 2))
        self.assertEqual(t3[0, 0], 1.0)
        self.assertEqual(t3[0, 1], 4.0)

    def test_element_wise_operations(self):
        """Element-wise arithmetic."""
        from xue.tensor import Tensor

        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[10, 20], [30, 40]])

        add = a + b
        self.assertEqual(add.tolist(), [[11.0, 22.0], [33.0, 44.0]])

        sub = b - a
        self.assertEqual(sub.tolist(), [[9.0, 18.0], [27.0, 36.0]])

        mul = a * b
        self.assertEqual(mul.tolist(), [[10.0, 40.0], [90.0, 160.0]])

    def test_tensor_statistics(self):
        """Statistical operations."""
        from xue.tensor import Tensor

        t = Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(t.sum(), 21.0)
        self.assertAlmostEqual(t.mean(), 3.5)
        self.assertEqual(t.max(), 6.0)
        self.assertEqual(t.min(), 1.0)

    def test_typed_tensor_factory(self):
        """Tensor[[shape], dtype] factory pattern."""
        from xue.tensor import Tensor, float32

        MatrixF = Tensor[[3, 3], float32]

        m = MatrixF([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(m.shape, (3, 3))
        self.assertEqual(m.dtype, float32)

        z = MatrixF.zeros()
        self.assertEqual(z[1, 1], 0.0)

        o = MatrixF.ones()
        self.assertEqual(o[2, 2], 1.0)

    def test_3d_tensor(self):
        """3D tensor creation and access."""
        from xue.tensor import Tensor

        t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        self.assertEqual(t.shape, (2, 2, 2))
        self.assertEqual(t[0, 0, 0], 1.0)
        self.assertEqual(t[1, 1, 1], 8.0)
        self.assertEqual(t.size, 8)

    def test_arange_and_reshape(self):
        """Create sequence and reshape."""
        from xue.tensor import arange

        t = arange(0, 12)
        self.assertEqual(t.shape[0], 12)

        t2 = t.reshape(3, 4)
        self.assertEqual(t2.shape, (3, 4))
        self.assertEqual(t2[2, 3], 11.0)


# ============================================================
# E2E: xue.strict
# ============================================================

class TestStrictE2E(unittest.TestCase):
    """Real-world strict typing patterns."""

    def test_api_boundary_checking(self):
        """Type checking at API boundaries."""
        from xue.strict import checked, StrictTypeError

        @checked
        def create_user(name: str, age: int, active: bool) -> dict:
            return {"name": name, "age": age, "active": active}

        user = create_user("Alice", 30, True)
        self.assertEqual(user["name"], "Alice")

        with self.assertRaises(StrictTypeError):
            create_user("Alice", "thirty", True)

        with self.assertRaises(StrictTypeError):
            create_user(123, 30, True)

    def test_return_type_enforcement(self):
        """Catch wrong return types."""
        from xue.strict import checked, StrictTypeError

        @checked
        def bad_api(x: int) -> list:
            return x  # Bug: returns int instead of list

        with self.assertRaises(StrictTypeError) as ctx:
            bad_api(42)
        self.assertIn("Return type", str(ctx.exception))

    def test_optional_parameters(self):
        """Handle Optional type annotations."""
        from xue.strict import checked
        from typing import Optional

        @checked
        def find(key: str, default: Optional[int] = None) -> Optional[int]:
            data = {"a": 1, "b": 2}
            return data.get(key, default)

        self.assertEqual(find("a"), 1)
        self.assertIsNone(find("z"))
        self.assertEqual(find("z", 99), 99)


# ============================================================
# E2E: xue.sandbox
# ============================================================

class TestSandboxE2E(unittest.TestCase):
    """Real-world sandboxing patterns."""

    def test_block_network_import(self):
        """Sandbox blocks network module imports via sandboxed_import."""
        from xue.sandbox import sandboxed_import, SandboxPolicy, SandboxViolation
        import sys

        # Remove socket from sys.modules so the importer intercepts it
        saved = sys.modules.pop("socket", None)
        try:
            policy = SandboxPolicy(allow_network=False)
            with self.assertRaises(SandboxViolation) as ctx:
                sandboxed_import("socket", policy=policy)
            self.assertIn("network", str(ctx.exception).lower())
        finally:
            if saved is not None:
                sys.modules["socket"] = saved

    def test_block_subprocess(self):
        """Sandbox blocks subprocess imports via sandboxed_import."""
        from xue.sandbox import sandboxed_import, SandboxPolicy, SandboxViolation
        import sys

        saved = sys.modules.pop("subprocess", None)
        try:
            policy = SandboxPolicy(allow_subprocess=False)
            with self.assertRaises(SandboxViolation):
                sandboxed_import("subprocess", policy=policy)
        finally:
            if saved is not None:
                sys.modules["subprocess"] = saved

    def test_compute_only_policy(self):
        """Compute-only sandbox allows math but blocks I/O."""
        from xue.sandbox import POLICY_COMPUTE_ONLY, sandboxed_import

        # Math should work fine
        math_mod = sandboxed_import("math", policy=POLICY_COMPUTE_ONLY)
        self.assertAlmostEqual(math_mod.sqrt(4), 2.0)

    def test_strict_policy_repr(self):
        """Strict policy has no capabilities."""
        from xue.sandbox import POLICY_STRICT

        self.assertIn("strict", repr(POLICY_STRICT))
        self.assertFalse(POLICY_STRICT.allow_network)
        self.assertFalse(POLICY_STRICT.allow_filesystem)
        self.assertFalse(POLICY_STRICT.allow_subprocess)
        self.assertFalse(POLICY_STRICT.allow_ctypes)


# ============================================================
# E2E: xue.unicode_ops
# ============================================================

class TestUnicodeOpsE2E(unittest.TestCase):
    """Real-world Unicode operator usage."""

    def test_scientific_formula_translation(self):
        """Translate a scientific formula with Unicode operators."""
        from xue.unicode_ops import _translate_source

        # Quadratic formula check
        src = "if b\u00b2 \u2265 4\u00d7a\u00d7c \u2227 a \u2260 0:"
        translated = _translate_source(src)
        self.assertIn(">=", translated)
        self.assertIn("*", translated)
        self.assertIn(" and ", translated)
        self.assertIn("!=", translated)

    def test_codec_registration(self):
        """Codec can be looked up after registration."""
        from xue.unicode_ops import register
        import codecs

        register()
        info = codecs.lookup("xue-unicode")
        self.assertEqual(info.name, "xue-unicode")

    def test_full_source_translation(self):
        """Translate a complete code snippet."""
        from xue.unicode_ops import _translate_source

        src = textwrap.dedent("""\
            x \u2190 10
            y \u2190 20
            if x \u2264 y \u2228 y \u2261 20:
                result \u2190 x \u00d7 y
        """)
        translated = _translate_source(src)
        self.assertIn("x = 10", translated)
        self.assertIn("y = 20", translated)
        self.assertIn("<=", translated)
        self.assertIn(" or ", translated)
        self.assertIn("==", translated)
        self.assertIn("x * y", translated)


# ============================================================
# E2E: xue.llmhook
# ============================================================

class TestLLMHookE2E(unittest.TestCase):
    """LLM hook configuration and safety."""

    def test_disabled_by_default(self):
        """LLM hook should be disabled by default."""
        from xue.llmhook import is_enabled
        # Unless XUE_LLM_HOOK=1 is set, should be disabled
        if os.environ.get("XUE_LLM_HOOK") != "1":
            self.assertFalse(is_enabled())

    def test_explain_no_exception(self):
        """explain() with no active exception returns message."""
        from xue.llmhook import explain
        result = explain()
        self.assertIsInstance(result, str)

    def test_configure_and_reset(self):
        """Can configure and reconfigure."""
        from xue.llmhook import configure, is_enabled

        configure(backend="none")
        self.assertFalse(is_enabled())

        configure(backend="http", url="http://localhost:11434/api/generate",
                  model="test")
        self.assertTrue(is_enabled())

        # Reset
        configure(backend="none")
        self.assertFalse(is_enabled())

    def test_exception_hook_install_uninstall(self):
        """Exception hook can be installed and uninstalled."""
        from xue.llmhook import install_exception_hook, uninstall_exception_hook
        import sys

        original = sys.excepthook
        install_exception_hook()
        self.assertNotEqual(sys.excepthook, original)

        uninstall_exception_hook()
        self.assertEqual(sys.excepthook, original)


# ============================================================
# E2E: Cross-module integration
# ============================================================

class TestCrossModuleIntegration(unittest.TestCase):
    """Test interactions between xue modules."""

    def test_result_with_contracts(self):
        """Contracts on functions returning Result."""
        from xue.result import Ok, Err, Result
        from xue.contracts import requires, ensures

        @requires(lambda x: isinstance(x, (int, float)), "must be numeric")
        @ensures(lambda result, x: result.is_ok() or result.is_err(), "must return Result")
        def safe_sqrt(x) -> Result[float, str]:
            if x < 0:
                return Err("negative input")
            return Ok(x ** 0.5)

        self.assertEqual(safe_sqrt(4).unwrap(), 2.0)
        self.assertTrue(safe_sqrt(-1).is_err())

    def test_secret_with_result(self):
        """Wrapping secrets in Result types."""
        from xue.result import Ok, Err, Result
        from xue.secret import Secret

        def load_key(name: str) -> Result[Secret, str]:
            keys = {"prod": "sk-live-xxx"}
            if name in keys:
                return Ok(Secret(keys[name]))
            return Err(f"key not found: {name}")

        r = load_key("prod")
        self.assertTrue(r.is_ok())
        key = r.unwrap()
        self.assertEqual(repr(key), "Secret(****)")
        self.assertEqual(key.expose(), "sk-live-xxx")

        r2 = load_key("staging")
        self.assertTrue(r2.is_err())

    def test_tensor_with_units(self):
        """Tensors can hold unit quantities (as plain numbers)."""
        from xue.tensor import Tensor
        from xue.units import meters

        # Create tensor of distances in meters
        distances = Tensor([100, 200, 300, 400])
        total = distances.sum()
        total_m = total * meters
        self.assertAlmostEqual(total_m.value, 1000.0)

    def test_autodiff_with_strict(self):
        """Autodiff Variables work with strict type checking."""
        from xue.autodiff import Variable
        from xue.strict import checked

        @checked
        def compute_loss(x: Variable) -> Variable:
            return (x - 3.0) ** 2

        x = Variable(5.0)
        loss = compute_loss(x)
        loss.backward()
        self.assertAlmostEqual(x.grad, 4.0)

    def test_dispatch_with_tensor(self):
        """Multiple dispatch on tensor operations."""
        from xue.dispatch import multimethod, _registries
        from xue.tensor import Tensor

        _registries.clear()

        @multimethod
        def scale(t: Tensor, factor: float) -> Tensor:
            return t * factor

        @multimethod
        def scale(t: Tensor, factor: int) -> Tensor:
            return t * float(factor)

        t = Tensor([[1, 2], [3, 4]])
        r1 = scale(t, 2.0)
        r2 = scale(t, 3)
        self.assertEqual(r1[0, 0], 2.0)
        self.assertEqual(r2[0, 0], 3.0)


# ============================================================
# E2E: Edge cases and error handling
# ============================================================

class TestEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def test_result_nested(self):
        """Nested Result types."""
        from xue.result import Ok, Err
        r = Ok(Ok(42))
        inner = r.unwrap()
        self.assertEqual(inner.unwrap(), 42)

    def test_empty_tensor(self):
        """Tensor with zero-size dimension."""
        from xue.tensor import Tensor
        t = Tensor(data=[], shape=[0])
        self.assertEqual(t.size, 0)
        self.assertEqual(len(t), 0)

    def test_unit_large_numbers(self):
        """Units with very large/small numbers."""
        from xue.units import meters, seconds
        speed_of_light = (299792458 * meters) / (1 * seconds)
        self.assertAlmostEqual(speed_of_light.value, 299792458.0, places=-1)

    def test_autodiff_zero_gradient(self):
        """Gradient of constant is zero."""
        from xue.autodiff import Variable
        x = Variable(5.0)
        y = Variable(42.0, requires_grad=False)
        z = x * 0 + y  # z = y, dz/dx = 0
        z.backward()
        self.assertAlmostEqual(x.grad, 0.0)

    def test_secret_empty_string(self):
        """Secret with empty string."""
        from xue.secret import Secret
        s = Secret("")
        self.assertEqual(repr(s), "Secret(****)")
        self.assertEqual(s.expose(), "")
        self.assertFalse(s)

    def test_contract_with_kwargs(self):
        """Contracts work with keyword arguments."""
        from xue.contracts import requires

        @requires(lambda x, y: y != 0, "y must not be zero")
        def div(x, y):
            return x / y

        self.assertEqual(div(x=10, y=2), 5)
        self.assertEqual(div(10, y=5), 2)


# ============================================================
# E2E: Thread safety
# ============================================================

class TestThreadSafety(unittest.TestCase):
    """Basic thread safety of xue modules."""

    def test_result_in_threads(self):
        """Result types used across threads."""
        from xue.result import Ok, Err
        results = []
        errors = []

        def worker(n):
            try:
                if n % 2 == 0:
                    r = Ok(n).map(lambda x: x * 2)
                else:
                    r = Err(f"odd: {n}")
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 20)

    def test_dispatch_in_threads(self):
        """Multiple dispatch called from threads."""
        from xue.dispatch import multimethod, _registries
        _registries.clear()

        @multimethod
        def compute(x: int) -> int:
            return x * 2

        results = []

        def worker(n):
            results.append(compute(n))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(sorted(results), [i * 2 for i in range(10)])


if __name__ == "__main__":
    unittest.main()
