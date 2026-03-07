"""Tests for xue.unicode_ops — Unicode operator aliases."""

import unittest
from xue.unicode_ops import _translate_source, register


class TestTranslation(unittest.TestCase):
    def test_comparison_operators(self):
        self.assertEqual(_translate_source("x \u2264 10"), "x <= 10")
        self.assertEqual(_translate_source("x \u2265 10"), "x >= 10")
        self.assertEqual(_translate_source("x \u2260 y"), "x != y")
        self.assertEqual(_translate_source("x \u2261 y"), "x == y")

    def test_arithmetic_operators(self):
        self.assertEqual(_translate_source("a \u00d7 b"), "a * b")
        self.assertEqual(_translate_source("a \u00f7 b"), "a / b")

    def test_logical_operators(self):
        self.assertEqual(_translate_source("a \u2227 b"), "a  and  b")
        self.assertEqual(_translate_source("a \u2228 b"), "a  or  b")
        self.assertEqual(_translate_source("\u00ac x"), "not  x")

    def test_membership_operators(self):
        self.assertEqual(_translate_source("x \u2208 items"), "x  in  items")
        self.assertEqual(_translate_source("x \u2209 items"), "x  not in  items")

    def test_assignment(self):
        self.assertEqual(_translate_source("x \u2190 5"), "x = 5")
        self.assertEqual(_translate_source("x \u2254 5"), "x := 5")

    def test_constants(self):
        self.assertEqual(_translate_source("\u221e"), "float('inf')")

    def test_mixed_code(self):
        src = "if x \u2264 10 \u2227 y \u2260 0:\n    result \u2190 x \u00d7 y"
        expected = "if x <= 10  and  y != 0:\n    result = x * y"
        self.assertEqual(_translate_source(src), expected)

    def test_no_change_on_ascii(self):
        src = "x = 10 + 20"
        self.assertEqual(_translate_source(src), src)

    def test_unicode_identifiers_preserved(self):
        # Greek letters should NOT be translated (they're valid Python identifiers)
        src = "\u03b1 = 0.01"  # α = 0.01
        self.assertEqual(_translate_source(src), src)


class TestCodecRegistration(unittest.TestCase):
    def test_register(self):
        register()
        # Should not raise
        import codecs
        info = codecs.lookup("xue-unicode")
        self.assertIsNotNone(info)


if __name__ == "__main__":
    unittest.main()
