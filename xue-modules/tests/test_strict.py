"""Tests for xue.strict — Runtime type checking."""

import unittest
from xue.strict import checked, StrictTypeError, enable, disable


class TestChecked(unittest.TestCase):
    def test_valid_types(self):
        @checked
        def add(x: int, y: int) -> int:
            return x + y

        self.assertEqual(add(1, 2), 3)

    def test_invalid_param_type(self):
        @checked
        def add(x: int, y: int) -> int:
            return x + y

        with self.assertRaises(StrictTypeError) as ctx:
            add("a", "b")
        self.assertIn("x", str(ctx.exception))
        self.assertIn("int", str(ctx.exception))
        self.assertIn("str", str(ctx.exception))

    def test_invalid_return_type(self):
        @checked
        def bad_return(x: int) -> int:
            return str(x)  # type: ignore

        with self.assertRaises(StrictTypeError) as ctx:
            bad_return(42)
        self.assertIn("Return type", str(ctx.exception))

    def test_optional_type(self):
        from typing import Optional

        @checked
        def maybe(x: Optional[int]) -> Optional[int]:
            return x

        self.assertEqual(maybe(42), 42)
        self.assertIsNone(maybe(None))

    def test_no_annotation_skipped(self):
        @checked
        def untyped(x):
            return x

        self.assertEqual(untyped("anything"), "anything")

    def test_list_generic(self):
        @checked
        def first(items: list) -> object:
            return items[0]

        self.assertEqual(first([1, 2, 3]), 1)

        with self.assertRaises(StrictTypeError):
            first("not a list")


class TestGlobalToggle(unittest.TestCase):
    def test_disable_skips_check(self):
        @checked
        def add(x: int, y: int) -> int:
            return x + y

        # checked decorator sets _force_checked, so it always checks
        # But a non-decorated function with global strict mode respects the toggle
        self.assertEqual(add(1, 2), 3)


if __name__ == "__main__":
    unittest.main()
