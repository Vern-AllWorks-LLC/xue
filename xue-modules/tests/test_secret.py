"""Tests for xue.secret — Secret[str] type."""

import unittest
import pickle
from xue.secret import Secret, SecretBytes


class TestSecret(unittest.TestCase):
    def test_repr_redacted(self):
        s = Secret("my-api-key")
        self.assertEqual(repr(s), "Secret(****)")
        self.assertEqual(str(s), "Secret(****)")

    def test_format_redacted(self):
        s = Secret("my-api-key")
        self.assertEqual(f"{s}", "Secret(****)")
        self.assertEqual(f"{s:>20}", "Secret(****)")

    def test_expose(self):
        s = Secret("my-api-key")
        self.assertEqual(s.expose(), "my-api-key")

    def test_constant_time_equality(self):
        a = Secret("secret123")
        b = Secret("secret123")
        c = Secret("different")
        self.assertTrue(a == b)
        self.assertFalse(a == c)

    def test_no_pickle(self):
        s = Secret("key")
        with self.assertRaises(TypeError):
            pickle.dumps(s)

    def test_map(self):
        s = Secret("hello")
        s2 = s.map(str.upper)
        self.assertEqual(s2.expose(), "HELLO")
        self.assertEqual(repr(s2), "Secret(****)")

    def test_bool(self):
        self.assertTrue(Secret("nonempty"))
        self.assertFalse(Secret(""))

    def test_nested_secret(self):
        s = Secret(Secret("deep"))
        self.assertEqual(s.expose(), "deep")

    def test_from_env(self):
        import os
        os.environ["XUE_TEST_SECRET"] = "test_value"
        s = Secret.from_env("XUE_TEST_SECRET")
        self.assertEqual(s.expose(), "test_value")
        del os.environ["XUE_TEST_SECRET"]

    def test_from_env_missing(self):
        with self.assertRaises(KeyError):
            Secret.from_env("XUE_NONEXISTENT_VAR_12345")

    def test_from_env_default(self):
        s = Secret.from_env("XUE_NONEXISTENT_VAR_12345", default="fallback")
        self.assertEqual(s.expose(), "fallback")


class TestSecretBytes(unittest.TestCase):
    def test_bytes_secret(self):
        s = SecretBytes(b"binary-key")
        self.assertEqual(repr(s), "Secret(****)")
        self.assertEqual(s.expose(), b"binary-key")

    def test_rejects_string(self):
        with self.assertRaises(TypeError):
            SecretBytes("not bytes")


if __name__ == "__main__":
    unittest.main()
