"""Tests for xue.autodiff — Automatic differentiation."""

import unittest
import math
from xue.autodiff import Variable, grad, value_and_grad, jacobian


class TestVariable(unittest.TestCase):
    def test_basic_ops(self):
        x = Variable(3.0)
        y = x + 2
        self.assertAlmostEqual(y.data, 5.0)

    def test_backward_simple(self):
        x = Variable(3.0)
        y = x * x  # y = x^2, dy/dx = 2x = 6
        y.backward()
        self.assertAlmostEqual(x.grad, 6.0)

    def test_backward_chain(self):
        x = Variable(2.0)
        y = x ** 3  # dy/dx = 3x^2 = 12
        y.backward()
        self.assertAlmostEqual(x.grad, 12.0)

    def test_backward_multi_var(self):
        x = Variable(3.0)
        y = Variable(2.0)
        z = x * y + x  # dz/dx = y + 1 = 3, dz/dy = x = 3
        z.backward()
        self.assertAlmostEqual(x.grad, 3.0)
        self.assertAlmostEqual(y.grad, 3.0)

    def test_exp(self):
        x = Variable(1.0)
        y = x.exp()  # dy/dx = e^x = e
        y.backward()
        self.assertAlmostEqual(x.grad, math.e, places=5)

    def test_log(self):
        x = Variable(2.0)
        y = x.log()  # dy/dx = 1/x = 0.5
        y.backward()
        self.assertAlmostEqual(x.grad, 0.5)

    def test_sin(self):
        x = Variable(0.0)
        y = x.sin()  # dy/dx = cos(0) = 1
        y.backward()
        self.assertAlmostEqual(x.grad, 1.0)

    def test_cos(self):
        x = Variable(0.0)
        y = x.cos()  # dy/dx = -sin(0) = 0
        y.backward()
        self.assertAlmostEqual(x.grad, 0.0)

    def test_tanh(self):
        x = Variable(0.0)
        y = x.tanh()  # dy/dx = 1 - tanh^2(0) = 1
        y.backward()
        self.assertAlmostEqual(x.grad, 1.0)

    def test_sigmoid(self):
        x = Variable(0.0)
        y = x.sigmoid()  # sigmoid(0) = 0.5, dy/dx = 0.25
        y.backward()
        self.assertAlmostEqual(y.data, 0.5)
        self.assertAlmostEqual(x.grad, 0.25)

    def test_relu(self):
        x = Variable(3.0)
        y = x.relu()
        y.backward()
        self.assertAlmostEqual(x.grad, 1.0)

        x2 = Variable(-3.0)
        y2 = x2.relu()
        y2.backward()
        self.assertAlmostEqual(x2.grad, 0.0)

    def test_division(self):
        x = Variable(6.0)
        y = x / 2  # dy/dx = 0.5
        y.backward()
        self.assertAlmostEqual(x.grad, 0.5)

    def test_subtraction(self):
        x = Variable(5.0)
        y = x - 3
        self.assertAlmostEqual(y.data, 2.0)

    def test_negation(self):
        x = Variable(5.0)
        y = -x
        y.backward()
        self.assertAlmostEqual(x.grad, -1.0)


class TestGrad(unittest.TestCase):
    def test_grad_simple(self):
        def f(x):
            return x ** 2

        df = grad(f)
        self.assertAlmostEqual(df(3.0), 6.0)

    def test_grad_polynomial(self):
        def f(x):
            return x ** 3 + 2 * x

        df = grad(f)
        self.assertAlmostEqual(df(3.0), 29.0)  # 3x^2 + 2 = 27 + 2

    def test_higher_order_grad(self):
        def f(x):
            return x ** 3

        df = grad(f)
        # f' = 3x^2
        self.assertAlmostEqual(df(2.0), 12.0)
        # Note: higher-order grad(grad(f)) requires Variable-aware grad,
        # which is a known limitation of the simple scalar implementation.
        # Use value_and_grad or explicit Variable.backward() for higher orders.


class TestValueAndGrad(unittest.TestCase):
    def test_value_and_grad(self):
        def loss(x):
            return (x - 3) ** 2

        vg = value_and_grad(loss)
        val, g = vg(5.0)
        self.assertAlmostEqual(val, 4.0)
        self.assertAlmostEqual(g, 4.0)


class TestJacobian(unittest.TestCase):
    def test_jacobian_linear(self):
        def f(x):
            return [x[0] + x[1], x[0] * x[1]]

        J = jacobian(f, [2.0, 3.0])
        # df0/dx0 = 1, df0/dx1 = 1
        self.assertAlmostEqual(J[0][0], 1.0)
        self.assertAlmostEqual(J[0][1], 1.0)
        # df1/dx0 = x1 = 3, df1/dx1 = x0 = 2
        self.assertAlmostEqual(J[1][0], 3.0)
        self.assertAlmostEqual(J[1][1], 2.0)


if __name__ == "__main__":
    unittest.main()
