"""Tests for xue.contracts — @requires, @ensures, @invariant."""

import unittest
from xue.contracts import requires, ensures, invariant, ContractViolation, set_enabled


class TestRequires(unittest.TestCase):
    def test_precondition_passes(self):
        @requires(lambda x: x > 0, "x must be positive")
        def sqrt(x):
            return x ** 0.5

        self.assertAlmostEqual(sqrt(4), 2.0)

    def test_precondition_fails(self):
        @requires(lambda x: x > 0, "x must be positive")
        def sqrt(x):
            return x ** 0.5

        with self.assertRaises(ContractViolation) as ctx:
            sqrt(-1)
        self.assertIn("Precondition", str(ctx.exception))
        self.assertIn("x must be positive", str(ctx.exception))

    def test_disabled(self):
        @requires(lambda x: x > 0, "x must be positive")
        def sqrt(x):
            return x ** 0.5

        set_enabled(False)
        try:
            # Should not raise even with invalid input
            sqrt(-1)  # returns nan, but no contract error
        finally:
            set_enabled(True)


class TestEnsures(unittest.TestCase):
    def test_postcondition_passes(self):
        @ensures(lambda result, x: result >= 0, "result must be non-negative")
        def abs_val(x):
            return abs(x)

        self.assertEqual(abs_val(-5), 5)

    def test_postcondition_fails(self):
        @ensures(lambda result, x: result >= 0, "result must be non-negative")
        def buggy_abs(x):
            return x  # Bug: doesn't actually take absolute value

        with self.assertRaises(ContractViolation) as ctx:
            buggy_abs(-5)
        self.assertIn("Postcondition", str(ctx.exception))


class TestInvariant(unittest.TestCase):
    def test_invariant_holds(self):
        @invariant(lambda self: self.balance >= 0, "balance must be non-negative")
        class Account:
            def __init__(self, balance):
                self.balance = balance

            def deposit(self, amount):
                self.balance += amount

        acct = Account(100)
        acct.deposit(50)
        self.assertEqual(acct.balance, 150)

    def test_invariant_violated_in_init(self):
        @invariant(lambda self: self.balance >= 0, "balance must be non-negative")
        class Account:
            def __init__(self, balance):
                self.balance = balance

        with self.assertRaises(ContractViolation):
            Account(-100)

    def test_invariant_violated_in_method(self):
        @invariant(lambda self: self.balance >= 0, "balance must be non-negative")
        class Account:
            def __init__(self, balance):
                self.balance = balance

            def withdraw(self, amount):
                self.balance -= amount

        acct = Account(50)
        with self.assertRaises(ContractViolation):
            acct.withdraw(100)


class TestCombined(unittest.TestCase):
    def test_requires_and_ensures(self):
        @requires(lambda a, b: b != 0, "divisor must not be zero")
        @ensures(lambda result, a, b: isinstance(result, float), "must return float")
        def divide(a: float, b: float) -> float:
            return a / b

        self.assertEqual(divide(10, 2), 5.0)

        with self.assertRaises(ContractViolation):
            divide(10, 0)


if __name__ == "__main__":
    unittest.main()
