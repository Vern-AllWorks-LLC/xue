"""Tests for xue.dispatch — Multiple dispatch."""

import unittest
from xue.dispatch import multimethod, _registries


class TestMultimethod(unittest.TestCase):
    def setUp(self):
        # Clear registry between tests
        _registries.clear()

    def test_basic_dispatch(self):
        @multimethod
        def add(a: int, b: int) -> int:
            return a + b

        @multimethod
        def add(a: float, b: float) -> float:
            return a + b

        @multimethod
        def add(a: str, b: str) -> str:
            return a + b

        self.assertEqual(add(1, 2), 3)
        self.assertAlmostEqual(add(1.5, 2.5), 4.0)
        self.assertEqual(add("hello ", "world"), "hello world")

    def test_subclass_dispatch(self):
        @multimethod
        def describe(x: object) -> str:
            return "object"

        @multimethod
        def describe(x: int) -> str:
            return "integer"

        self.assertEqual(describe(42), "integer")
        self.assertEqual(describe("hello"), "object")

    def test_no_match_raises(self):
        @multimethod
        def typed_func(a: int, b: int) -> int:
            return a + b

        with self.assertRaises(TypeError):
            typed_func("a", "b")

    def test_different_arity(self):
        @multimethod
        def norm(x: float) -> float:
            return abs(x)

        @multimethod
        def norm(x: float, y: float) -> float:
            return (x ** 2 + y ** 2) ** 0.5

        self.assertEqual(norm(-5.0), 5.0)
        self.assertAlmostEqual(norm(3.0, 4.0), 5.0)

    def test_repr(self):
        @multimethod
        def f(x: int) -> int:
            return x

        self.assertIn("multimethod", repr(f))
        self.assertIn("1 implementation", repr(f))


if __name__ == "__main__":
    unittest.main()
