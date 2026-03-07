"""Tests for xue.units — Physical units type system."""

import unittest
import math
from xue.units import (
    meters, seconds, kg, kilometers, centimeters, minutes, hours,
    newtons, joules, watts, degrees, radians,
    Quantity, DimensionError, Unit,
)


class TestQuantityArithmetic(unittest.TestCase):
    def test_create(self):
        d = 100 * meters
        self.assertIsInstance(d, Quantity)
        self.assertEqual(d.value, 100)

    def test_add_same_unit(self):
        d = 100 * meters + 50 * meters
        self.assertEqual(d.value, 150)

    def test_add_compatible_units(self):
        d = 1 * kilometers + 500 * meters
        self.assertAlmostEqual(d.value, 1.5)  # 1 km + 0.5 km

    def test_add_incompatible_raises(self):
        with self.assertRaises(DimensionError):
            (5 * meters) + (3 * seconds)

    def test_multiply(self):
        d = 10 * meters
        t = 2 * seconds
        v = d / t
        self.assertAlmostEqual(v.value, 5.0)

    def test_force(self):
        mass = 10 * kg
        accel = (5 * meters) / ((1 * seconds) ** 2)
        force = mass * accel
        self.assertAlmostEqual(force.value, 50.0)

    def test_scalar_multiply(self):
        d = 10 * meters
        self.assertEqual((d * 3).value, 30)
        self.assertEqual((3 * d).value, 30)

    def test_negate(self):
        d = -(5 * meters)
        self.assertEqual(d.value, -5)


class TestConversion(unittest.TestCase):
    def test_meters_to_km(self):
        d = 5000 * meters
        self.assertAlmostEqual(d.to(kilometers), 5.0)

    def test_km_to_meters(self):
        d = 2.5 * kilometers
        self.assertAlmostEqual(d.to(meters), 2500.0)

    def test_minutes_to_seconds(self):
        t = 5 * minutes
        self.assertAlmostEqual(t.to(seconds), 300.0)

    def test_incompatible_conversion_raises(self):
        with self.assertRaises(DimensionError):
            (5 * meters).to(seconds)


class TestComparison(unittest.TestCase):
    def test_less_than(self):
        self.assertTrue(5 * meters < 10 * meters)
        self.assertFalse(10 * meters < 5 * meters)

    def test_cross_unit_comparison(self):
        self.assertTrue(500 * meters < 1 * kilometers)

    def test_equality(self):
        self.assertEqual(1000 * meters, 1 * kilometers)

    def test_incompatible_comparison_raises(self):
        with self.assertRaises(DimensionError):
            (5 * meters) < (3 * seconds)


class TestAngles(unittest.TestCase):
    def test_degrees_to_radians(self):
        angle = 180 * degrees
        self.assertAlmostEqual(angle.to(radians), math.pi)

    def test_radians_to_degrees(self):
        angle = math.pi * radians
        self.assertAlmostEqual(angle.to(degrees), 180.0)


class TestRepr(unittest.TestCase):
    def test_quantity_repr(self):
        d = 5 * meters
        self.assertIn("5", repr(d))
        self.assertIn("m", repr(d))

    def test_format(self):
        d = 3.14159 * meters
        s = f"{d:.2f}"
        self.assertIn("3.14", s)


if __name__ == "__main__":
    unittest.main()
