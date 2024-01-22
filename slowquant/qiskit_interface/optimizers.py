from collections.abc import Callable
from functools import partial

import numpy as np
import scipy


class Result:
    def __init__(self):
        self.x: np.ndarray | None = None
        self.fun: float | None = None


class RotoSolve:
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
            for i in range(len(x)):
                e_vals = get_energy_evals(f, x, i)
                f_reconstructed = partial(reconstructed_f, energy_vals=e_vals)
                vecf = np.vectorize(f_reconstructed)
                values = vecf(np.linspace(-np.pi, np.pi, int(1e4)))
                minidx = np.argmin(values)
                theta = np.linspace(-np.pi, np.pi, int(1e4))[minidx]
                res = scipy.optimize.minimize(f_reconstructed, x0=[theta], method="BFGS", tol=1e-12)
                x[i] = res.x[0]
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


def get_energy_evals(f, x, idx) -> list[float]:
    e_vals = []
    x = x.copy()
    R = 2
    for mu in [-2, -1, 0, 1, 2]:
        x_mu = 2 * mu / (2 * R + 1) * np.pi
        x[idx] = x_mu
        e_vals.append(f(x))
    return e_vals


def reconstructed_f(x: float, energy_vals: list[float]) -> float:
    R = 2
    e = 0
    for i, mu in enumerate([-2, -1, 0, 1, 2]):
        x_mu = 2 * mu / (2 * R + 1) * np.pi
        e += energy_vals[i] * (-1) ** mu / np.sin(x / 2 - x_mu / 2)
    return np.sin((2 * R + 1) / 2 * x) / (2 * R + 1) * e
