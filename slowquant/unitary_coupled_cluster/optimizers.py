import time
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import numpy as np
import scipy


class Result:
    """Result class for optimizers."""

    def __init__(self) -> None:
        """Initialize result class."""
        self.x: np.ndarray
        self.fun: float


class Optimizers:
    """Optimizers class."""

    _start: float
    _iteration: int

    def __init__(
        self,
        fun: Callable[[list[float]], float],
        method: str,
        grad: Callable[[list[float]], np.ndarray] | None = None,
        maxiter: int = 1000,
        tol: float = 10e-8,
        is_silent: bool = False,
    ) -> None:
        """Initialize optimizer class.

        Args:
            fun: Function to minimize.
            method: Optimization method.
            grad: Gradient of function.
            maxiter: Maximum iterations.
            tol: Convergence tolerence.
            is_silent: Supress progress output.
        """
        self.fun = fun
        self.grad = grad
        self.method = method.lower()
        self.maxiter = maxiter
        self.tol = tol
        self.is_silent = is_silent

    def _print_progress(
        self, x: Sequence[float], fun: Callable[[list[float]], float], silent: bool = False
    ) -> None:
        """Print progress during optimization.

        Args:
            x: Parameters.
            fun: Function.
            silent: Silence progress print.
        """
        time_str = f"{time.time() - self._start:7.2f}"
        if not silent:
            e_str = f"{fun(list(x)):3.16f}"
            print(
                f"--------{str(self._iteration + 1).center(11)} | {time_str.center(18)} | {e_str.center(27)}"
            )
            self._iteration += 1
            self._start = time.time()

    def minimize(self, x0: Sequence[float], extra_options: dict[str, Any] | None = None) -> Result:
        """Minimize function.

        extra_options:
            * R dict[str, int]: Order parameter needed for Rotosolve.
            * param_names Sequence[str]: Names of parameters needed for Rotosolve.

        Args:
            x0: Starting value of changable parameters.
            extra_options: Extra options for optimizers.
        """
        self._start = time.time()
        self._iteration = 0
        print_progress = partial(self._print_progress, fun=self.fun, silent=self.is_silent)
        if self.method in ("slsqp", "l-bfgs-b"):
            if self.grad is not None:
                res = scipy.optimize.minimize(
                    self.fun,
                    x0,
                    jac=self.grad,
                    method=self.method,
                    tol=self.tol,
                    callback=print_progress,
                    options={"maxiter": self.maxiter},
                )
            else:
                res = scipy.optimize.minimize(
                    self.fun,
                    x0,
                    method=self.method,
                    tol=self.tol,
                    callback=print_progress,
                    options={"maxiter": self.maxiter},
                )
        elif self.method in ("cobyla",):
            res = scipy.optimize.minimize(
                self.fun,
                x0,
                method=self.method,
                tol=self.tol,
                callback=print_progress,
                options={"maxiter": self.maxiter},
            )
        elif self.method in ("rotosolve",):
            if not isinstance(extra_options, dict):
                raise TypeError("extra_options is not set, but is required for RotoSolve")
            if "R" not in extra_options:
                raise ValueError(f"Expected option 'R' in extra_options, got {extra_options.keys()}")
            if "param_names" not in extra_options:
                raise ValueError(
                    f"Expected option 'param_names' in extra_options, got {extra_options.keys()}"
                )
            optimizer = RotoSolve(
                extra_options["R"],
                extra_options["param_names"],
                maxiter=self.maxiter,
                tol=self.tol,
                callback=print_progress,
            )
            res = optimizer.minimize(self.fun, x0)

        else:
            raise ValueError(f"Got an unkonwn optimizer {self.method}")
        result = Result()
        result.x = res.x
        result.fun = res.fun
        return result


class RotoSolve:
    r"""Rotosolve optimizer.

    Implemenation of Rotosolver assuming three eigenvalues for generators.
    This works for fermionic generators of the type:

    .. math::
        \hat{G}_{pq} = \hat{a}^\dagger_p \hat{a}_q - \hat{a}_q^\dagger \hat{a}_p

    and,

    .. math::
        \hat{G}_{pqrs} = \hat{a}^\dagger_p \hat{a}^\dagger_q \hat{a}_r \hat{a}_s - \hat{a}^\dagger_s \hat{a}^\dagger_r \hat{a}_p \hat{a}_q

    Rotosolve works by exactly reconstrucing the energy function in a single parameter:

    .. math::
        E(x) = \frac{\sin\left(\frac{2R+1}{2}x\right)}{2R+1}\sum_{\mu=-R}^{R}E(x_\mu)\frac{(-1)^\mu}{\sin\left(\frac{x - x_\mu}{2}\right)}

    With :math:`R` being the number of different positive differences between eigenvalues, and :math:`x_\mu=\frac{2\mu}{2R+1}\pi`.

    After the function :math:`E(x)` have been reconstruced the global minima of the function can be found classically.

    #. 10.22331/q-2021-01-28-391, Algorithm 1
    #. 10.22331/q-2022-03-30-677, Eq. (57)
    """

    def __init__(
        self,
        R: dict[str, int],
        param_names: Sequence[str],
        maxiter: int = 30,
        tol: float = 1e-6,
        callback: Callable[[list[float]], None] | None = None,
    ) -> None:
        """Initialize Rotosolver.

        Args:
            R: R parameter used for the function reconstruction.
            param_names: Names of parameters, used to index R.
            maxiter: Maximum number of iterations (sweeps).
            tol: Convergence tolerence.
            callback: Callback function, takes only x (parameters) as an argument.
        """
        self._callback = callback
        self.max_iterations = maxiter
        self.threshold = tol
        self.max_fail = 3
        self._R = R
        self._param_names = param_names

    def minimize(self, f: Callable[[list[float]], float], x0: Sequence[float]) -> Result:
        """Run minimization.

        Args:
            f: Function to be minimzed, can only take one argument.
            x: Changable parameters of f.

        Returns:
            Minimization results.
        """
        f_best = float(10**20)
        x = list(x0).copy()
        x_best = x.copy()
        fails = 0
        res = Result()
        for _ in range(self.max_iterations):
            for i, par_name in enumerate(self._param_names):
                e_vals = get_energy_evals(f, x, i, self._R[par_name])
                f_reconstructed = partial(reconstructed_f, energy_vals=e_vals, R=self._R[par_name])
                vecf = np.vectorize(f_reconstructed)
                values = vecf(np.linspace(-np.pi, np.pi, int(1e4)))
                theta = np.linspace(-np.pi, np.pi, int(1e4))[np.argmin(values)]
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


def get_energy_evals(f: Callable[[list[float]], float], x: list[float], idx: int, R: int) -> list[float]:
    """Evaluate the function in all points needed for the reconstrucing in Rotosolve.

    Args:
        f: Function to evaluate.
        x: Parameters of f.
        idx: Index of parameter to be changed.
        R: Parameter to control how many points are needed.

    Returns:
        All needed function evaluations.
    """
    e_vals = []
    x = x.copy()
    for mu in range(-R, R + 1):
        x_mu = 2 * mu / (2 * R + 1) * np.pi
        x[idx] = x_mu
        e_vals.append(f(x))
    return e_vals


def reconstructed_f(x: float, energy_vals: list[float], R: int) -> float:
    r"""Reconstructed the function in terms of sin-functions.

    .. math::
        E(x) = \frac{\sin\left(\frac{2R+1}{2}x\right)}{2R+1}\sum_{\mu=-R}^{R}E(x_\mu)\frac{(-1)^\mu}{\sin\left(\frac{x - x_\mu}{2}\right)}

    For better numerical stability the implemented form is instead:

    .. math::
        E(x) = \sum_{\mu=-R}^{R}E(x_\mu)\frac{\mathrm{sinc}\left(\frac{2R+1}{2}(x-x_\mu)\right)}{\mathrm{sinc}\left(\frac{1}{2}(x-x_\mu)\right)}

    #. 10.22331/q-2022-03-30-677, Eq. (57)
    #. https://pennylane.ai/qml/demos/tutorial_general_parshift/, 2024-03-14

    Args:
        x: Function variable.
        energy_vals: Pre-calculated points of original function.
        R: Parameter to control how many points are needed.

    Returns:
        Function value in x.
    """
    e = 0.0
    for i, mu in enumerate(list(range(-R, R + 1))):
        x_mu = 2 * mu / (2 * R + 1) * np.pi
        e += (
            energy_vals[i]
            * np.sinc((2 * R + 1) / 2 * (x - x_mu) / np.pi)
            / (np.sinc(1 / 2 * (x - x_mu) / np.pi))
        )
    return e
