from typing import Callable, Tuple


CLASSIC = 0


class RungeKuttaOrder4:
    def __init__(self, func: Callable[[float, float], float], mode: int = CLASSIC):
        """
        Initializes the Runge-Kutta differential equation approximation of order 4
        :param func: the function defined as dy/dx = func(x, y)
        :param mode: yet to be implemented, different types of Runge-Kutta approximation
        """
        self.mode = mode
        self.func = func

    def approx_step(self, h: float, x: float, y: float) -> float:
        """
        Provides a numerical approximation of a function based on a first order differential equation.
        :param h: step size
        :param x: initial x value
        :param y: initial y value
        :return: approximated y at x + h
        """
        k1 = self.func(x, y)
        k2 = self.func(x + (h/2), y + (h/2) * k1)
        k3 = self.func(x + (h/2), y + (h/2) * k2)
        k4 = self.func(x + h, y + h * k3)

        return y + (h/6) * (k1 + (2 * k2) + (2 * k3) + k4)

    def approx_value(self, n: int = 1000, x0: float = 0, y0: float = 1, xf: float = 1) -> Tuple[float, float]:
        """
        Approximates the value of a differential equation from initial conditions
        :param n: number of steps
        :param x0: initial x value
        :param y0: initial y value
        :param xf: desired x value
        :return: coordinates of final value
        """
        h = (xf - x0) / n
        x = x0
        y = y0
        while x < xf:
            y = self.approx_step(h=h, x=x, y=y)
            x += h
        return xf, y
