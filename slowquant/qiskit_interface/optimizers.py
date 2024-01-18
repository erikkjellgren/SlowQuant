from collections.abc import Callable

import numpy as np


class Result:
    def __init__(self):
        self.x = None
        self.fun = None


class RotaSolve:
    def __init__(self, maxiter: int = 30, tol: float = 1e-6, callback: Callable | None = None) -> None:
        self._callback = callback
        self.max_iterations = maxiter
        self.threshold = tol
        self.max_fail = 3

    def minimize(self, f: Callable[[list[float]], float], x: list[float], jac=None):
        f_best = 10**20
        x_best = x.copy()
        fails = 0
        res = Result()
        for _ in range(self.max_iterations):
            phi = 0.0
            for i in range(len(x)):
                x[i] = phi
                f0 = f(x)
                x[i] = phi + np.pi / 4
                fp = f(x)
                x[i] = phi - np.pi / 4
                fm = f(x)
                x[i] = phi - np.pi / 4 - np.arctan2(2 * f0 - fp - fm, fp - fm) / 2
                while x[i] < np.pi:
                    x[i] += 2 * np.pi
                while x[i] > np.pi:
                    x[i] -= 2 * np.pi
            f_new = f(x)
            if abs(f_best - f_new) < self.threshold:
                f_best = f_new
                x_best = x.copy()
                break
            if (f_new - f_best) > 0.0:
                fails += 1
            else:
                f_best = f_new
                x_best = x.copy()
            if fails == self.max_fail:
                break
            if self._callback is not None:
                self._callback(x)
        res.x = np.array(x_best)
        res.fun = f_best
        return res
