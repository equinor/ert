from tests import ResTest
from ecl.util.util import DoubleVector
from res.util import polyfit


class StatTest(ResTest):
    def test_polyfit(self):
        x_list = DoubleVector()
        y_list = DoubleVector()
        S = DoubleVector()

        A = 7.25
        B = -4
        C = 0.025

        x = 0
        dx = 0.1
        for i in range(100):
            y = A + B * x + C * x * x
            x_list.append(x)
            y_list.append(y)

            x += dx
            S.append(1.0)

        beta = polyfit(3, x_list, y_list, None)

        self.assertAlmostEqual(A, beta[0])
        self.assertAlmostEqual(B, beta[1])
        self.assertAlmostEqual(C, beta[2])
