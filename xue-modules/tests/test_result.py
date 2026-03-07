"""Tests for xue.result — Result[T,E] and Option[T] types."""

import unittest
from xue.result import Ok, Err, Result, Some, Nothing, Option, UnwrapError, catch


class TestOk(unittest.TestCase):
    def test_is_ok(self):
        r = Ok(42)
        self.assertTrue(r.is_ok())
        self.assertFalse(r.is_err())

    def test_unwrap(self):
        self.assertEqual(Ok(42).unwrap(), 42)

    def test_unwrap_or(self):
        self.assertEqual(Ok(42).unwrap_or(0), 42)

    def test_map(self):
        r = Ok(5).map(lambda x: x * 2)
        self.assertEqual(r.unwrap(), 10)

    def test_and_then(self):
        r = Ok(5).and_then(lambda x: Ok(x + 1))
        self.assertEqual(r.unwrap(), 6)

    def test_and_then_err(self):
        r = Ok(5).and_then(lambda x: Err("fail"))
        self.assertTrue(r.is_err())

    def test_bool(self):
        self.assertTrue(Ok(1))

    def test_iter(self):
        self.assertEqual(list(Ok(42)), [42])

    def test_repr(self):
        self.assertEqual(repr(Ok(42)), "Ok(42)")

    def test_eq(self):
        self.assertEqual(Ok(1), Ok(1))
        self.assertNotEqual(Ok(1), Ok(2))

    def test_match(self):
        r: Result[int, str] = Ok(42)
        match r:
            case Ok(v):
                self.assertEqual(v, 42)
            case _:
                self.fail("Should match Ok")


class TestErr(unittest.TestCase):
    def test_is_err(self):
        r = Err("fail")
        self.assertFalse(r.is_ok())
        self.assertTrue(r.is_err())

    def test_unwrap_raises(self):
        with self.assertRaises(UnwrapError):
            Err("fail").unwrap()

    def test_unwrap_or(self):
        self.assertEqual(Err("fail").unwrap_or(0), 0)

    def test_map_noop(self):
        r = Err("fail").map(lambda x: x * 2)
        self.assertTrue(r.is_err())

    def test_map_err(self):
        r = Err("fail").map_err(lambda e: e.upper())
        self.assertEqual(r.err().unwrap(), "FAIL")

    def test_bool(self):
        self.assertFalse(Err("x"))

    def test_iter(self):
        self.assertEqual(list(Err("x")), [])

    def test_match(self):
        r: Result[int, str] = Err("oops")
        match r:
            case Err(e):
                self.assertEqual(e, "oops")
            case _:
                self.fail("Should match Err")


class TestOption(unittest.TestCase):
    def test_some(self):
        o = Some(42)
        self.assertTrue(o.is_some())
        self.assertFalse(o.is_nothing())
        self.assertEqual(o.unwrap(), 42)

    def test_nothing(self):
        o = Nothing()
        self.assertFalse(o.is_some())
        self.assertTrue(o.is_nothing())
        with self.assertRaises(UnwrapError):
            o.unwrap()

    def test_nothing_singleton(self):
        self.assertIs(Nothing(), Nothing())

    def test_some_rejects_none(self):
        with self.assertRaises(ValueError):
            Some(None)

    def test_from_nullable(self):
        self.assertEqual(Option.from_nullable(42), Some(42))
        self.assertEqual(Option.from_nullable(None), Nothing())

    def test_map(self):
        self.assertEqual(Some(5).map(lambda x: x * 2).unwrap(), 10)
        self.assertEqual(Nothing().map(lambda x: x * 2), Nothing())

    def test_ok_or(self):
        self.assertEqual(Some(5).ok_or("err"), Ok(5))
        self.assertEqual(Nothing().ok_or("err"), Err("err"))

    def test_filter(self):
        self.assertEqual(Some(5).filter(lambda x: x > 3).unwrap(), 5)
        self.assertEqual(Some(5).filter(lambda x: x > 10), Nothing())


class TestCatch(unittest.TestCase):
    def test_catch_success(self):
        r = catch(int, "42")
        self.assertEqual(r.unwrap(), 42)

    def test_catch_failure(self):
        r = catch(int, "not_a_number")
        self.assertTrue(r.is_err())
        self.assertIsInstance(r.err().unwrap(), ValueError)


if __name__ == "__main__":
    unittest.main()
