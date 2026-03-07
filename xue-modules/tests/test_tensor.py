"""Tests for xue.tensor — Tensor type with runtime shape validation."""

import unittest
from xue.tensor import Tensor, float32, float64, int32, ShapeError, zeros, ones, eye, arange


class TestTensorCreation(unittest.TestCase):
    def test_from_list(self):
        t = Tensor([[1, 2], [3, 4]])
        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(t[0, 0], 1.0)
        self.assertEqual(t[1, 1], 4.0)

    def test_from_flat(self):
        t = Tensor([1, 2, 3, 4])
        self.assertEqual(t.shape, (4,))  # 1D tensor

    def test_typed_constructor(self):
        factory = Tensor[[3, 2], float32]
        t = factory([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(t.shape, (3, 2))
        self.assertEqual(t.dtype, float32)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ShapeError):
            Tensor(data=[[1, 2, 3]], shape=[2, 2])

    def test_zeros(self):
        t = zeros((3, 4))
        self.assertEqual(t.shape, (3, 4))
        self.assertEqual(t[0, 0], 0.0)

    def test_ones(self):
        t = ones((2, 3))
        self.assertEqual(t[1, 2], 1.0)

    def test_eye(self):
        t = eye(3)
        self.assertEqual(t[0, 0], 1.0)
        self.assertEqual(t[0, 1], 0.0)
        self.assertEqual(t[2, 2], 1.0)

    def test_arange(self):
        t = arange(0, 5)
        self.assertEqual(t.shape[0], 5)


class TestTensorArithmetic(unittest.TestCase):
    def test_add(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        c = a + b
        self.assertEqual(c[0, 0], 6.0)
        self.assertEqual(c[1, 1], 12.0)

    def test_scalar_mul(self):
        a = Tensor([[1, 2], [3, 4]])
        b = a * 2
        self.assertEqual(b[0, 1], 4.0)

    def test_sub(self):
        a = Tensor([[5, 6], [7, 8]])
        b = Tensor([[1, 2], [3, 4]])
        c = a - b
        self.assertEqual(c[0, 0], 4.0)

    def test_shape_mismatch_add_raises(self):
        a = Tensor([[1, 2]])
        b = Tensor([[1, 2, 3]])
        with self.assertRaises(ShapeError):
            a + b


class TestMatmul(unittest.TestCase):
    def test_basic_matmul(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        c = a @ b
        self.assertEqual(c.shape, (2, 2))
        self.assertAlmostEqual(c[0, 0], 19.0)  # 1*5 + 2*7
        self.assertAlmostEqual(c[0, 1], 22.0)  # 1*6 + 2*8
        self.assertAlmostEqual(c[1, 0], 43.0)  # 3*5 + 4*7
        self.assertAlmostEqual(c[1, 1], 50.0)  # 3*6 + 4*8

    def test_shape_checking(self):
        a = Tensor([[1, 2, 3]])         # 1x3
        b = Tensor([[1, 2], [3, 4]])    # 2x2
        with self.assertRaises(ShapeError) as ctx:
            a @ b
        self.assertIn("3", str(ctx.exception))
        self.assertIn("2", str(ctx.exception))

    def test_non_square(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])   # 2x3
        b = Tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
        c = a @ b
        self.assertEqual(c.shape, (2, 2))


class TestTensorOps(unittest.TestCase):
    def test_reshape(self):
        t = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2x4
        t2 = t.reshape(4, 2)
        self.assertEqual(t2.shape, (4, 2))

    def test_reshape_invalid(self):
        t = Tensor([[1, 2], [3, 4]])  # 4 elements
        with self.assertRaises(ShapeError):
            t.reshape(3, 3)  # needs 9 elements

    def test_transpose(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
        t2 = t.T
        self.assertEqual(t2.shape, (3, 2))
        self.assertEqual(t2[0, 0], 1.0)
        self.assertEqual(t2[2, 1], 6.0)

    def test_sum(self):
        t = Tensor([[1, 2], [3, 4]])
        self.assertEqual(t.sum(), 10.0)

    def test_mean(self):
        t = Tensor([[1, 2], [3, 4]])
        self.assertEqual(t.mean(), 2.5)

    def test_tolist(self):
        t = Tensor([[1, 2], [3, 4]])
        self.assertEqual(t.tolist(), [[1.0, 2.0], [3.0, 4.0]])


if __name__ == "__main__":
    unittest.main()
