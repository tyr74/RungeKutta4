import math
from RungeKuttaOrder4 import RungeKuttaOrder4


def func(x, y):
    return 1 / x


if __name__ == "__main__":
    RK4 = RungeKuttaOrder4(func=func)
    print(RK4.approx_value(n=1000000, x0=1, y0=0, xf=2))
    print(math.log(2, math.e))
