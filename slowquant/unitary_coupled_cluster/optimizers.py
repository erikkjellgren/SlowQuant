import time
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import numba as nb
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
        if not silent:
            e_str = f"{fun(list(x)):3.16f}"
            time_str = f"{time.time() - self._start:7.2f}"
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
        if self.method in ("bfgs", "l-bfgs-b", "slsqp"):
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
        elif self.method in ("cobyla", "cobyqa"):
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
            if "f_rotosolve_optimized" in extra_options:
                res = optimizer.minimize(
                    self.fun, x0, f_rotosolve_optimized=extra_options["f_rotosolve_optimized"]
                )
            else:
                res = optimizer.minimize(self.fun, x0)
        elif self.method in ("rotosolve_2d",):
            if not isinstance(extra_options, dict):
                raise TypeError("extra_options is not set, but is required for RotoSolve")
            if "R" not in extra_options:
                raise ValueError(f"Expected option 'R' in extra_options, got {extra_options.keys()}")
            if "param_names" not in extra_options:
                raise ValueError(
                    f"Expected option 'param_names' in extra_options, got {extra_options.keys()}"
                )
            optimizer = RotoSolve2D(
                extra_options["R"],
                extra_options["param_names"],
                maxiter=self.maxiter,
                tol=self.tol,
                callback=print_progress,
            )
            if "f_rotosolve2d_optimized" in extra_options:
                res = optimizer.minimize(
                    self.fun, x0, f_rotosolve2d_optimized=extra_options["f_rotosolve2d_optimized"]
                )
            else:
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

    def minimize(
        self,
        f: Callable[[list[float]], float],
        x0: Sequence[float],
        f_rotosolve_optimized: None | Callable[[list[float], list[float], int], list[float]] = None,
    ) -> Result:
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
                # Get the energy for specific values of theta_i, defined by the _R parameter.
                if f_rotosolve_optimized is not None:
                    e_vals = get_energy_evals_optimized(f_rotosolve_optimized, x, i, self._R[par_name])
                else:
                    e_vals = get_energy_evals(f, x, i, self._R[par_name])
                # Do an analytic construction of the energy as a function of theta_i.
                f_reconstructed = partial(reconstructed_f, energy_vals=e_vals, R=self._R[par_name])
                # Evaluate the energy in many points.
                values = f_reconstructed(np.linspace(-np.pi, np.pi, int(1e4)))
                # Find the theta_i that gives the lowest energy.
                theta = np.linspace(-np.pi, np.pi, int(1e4))[np.argmin(values)]
                # Run an optimization on the theta_i that gives to the lowest energy in the previous step.
                # This is to get more digits precision in value of theta_i.
                res = scipy.optimize.minimize(f_reconstructed, x0=[theta], method="BFGS", tol=1e-12)
                x[i] = res.x[0]
                while x[i] < np.pi:
                    x[i] += 2 * np.pi
                while x[i] > np.pi:
                    x[i] -= 2 * np.pi
            f_tmp = f(x)
            f_new = float(np.mean(f_tmp))
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
    """Evaluate the function in all points needed for the reconstruction in Rotosolve.

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
    return e_vals  # type: ignore


def get_energy_evals_optimized(
    f: Callable[[list[float], list[float], int], list[float]], x: list[float], idx: int, R: int
) -> list[float]:
    """Evaluate the function in all points needed for the reconstruction in Rotosolve.

    Args:
        f: Function to evaluate.
        x: Parameters of f.
        idx: Index of parameter to be changed.
        R: Parameter to control how many points are needed.

    Returns:
        All needed function evaluations.
    """
    theta_diffs = []
    for mu in range(-R, R + 1):
        theta_diffs.append(2 * mu / (2 * R + 1) * np.pi)
    return f(x, theta_diffs, idx)


def get_energy_evals2d_optimized(
        f: Callable[[list[float], list[float], list[float], int, int], list[float]], x: list[float], idx1: int, idx2: int, R1: int, R2: int,
) -> list[float]:
    """
    """
    if idx1 <= idx2:
        raise ValueError(f"The first index must be larger than the second index, got idx1,idx2={idx1},{idx2}")
    theta1_diffs = []
    theta2_diffs = []
    for mu1 in range(-R1, R1 + 1):
        theta1_diffs.append(2 * mu1 / (2 * R1 + 1) * np.pi)
    for mu2 in range(-R2, R2 + 1):
        theta2_diffs.append(2 * mu2 / (2 * R2 + 1) * np.pi)
    return f(x, theta1_diffs, theta2_diffs, idx1, idx2)


@nb.jit(nopython=True)
def reconstructed_f(x_vals: np.ndarray, energy_vals: list[float], R: int) -> np.ndarray:
    r"""Reconstructed the function in terms of sin-functions.

    .. math::
        E(x) = \frac{\sin\left(\frac{2R+1}{2}x\right)}{2R+1}\sum_{\mu=-R}^{R}E(x_\mu)\frac{(-1)^\mu}{\sin\left(\frac{x - x_\mu}{2}\right)}

    For better numerical stability the implemented form is instead:

    .. math::
        E(x) = \sum_{\mu=-R}^{R}E(x_\mu)\frac{\mathrm{sinc}\left(\frac{2R+1}{2}(x-x_\mu)\right)}{\mathrm{sinc}\left(\frac{1}{2}(x-x_\mu)\right)}

    #. 10.22331/q-2022-03-30-677, Eq. (57)
    #. https://pennylane.ai/qml/demos/tutorial_general_parshift/, 2024-03-14

    Args:
        x_vals: List of points to evaluate the function in.
        energy_vals: Pre-calculated points of original function.
        R: Parameter to control how many points are needed.

    Returns:
        Function value in list of points.
    """
    e = np.zeros(len(x_vals))
    # Single state case
    for i, mu in enumerate(list(range(-R, R + 1))):
        x_mu = 2 * mu / (2 * R + 1) * np.pi
        for j, x in enumerate(x_vals):
            e[j] += (
                energy_vals[i]
                * np.sinc((2 * R + 1) / 2 * (x - x_mu) / np.pi)
                / (np.sinc(1 / 2 * (x - x_mu) / np.pi))
            )
    return e


class RotoSolve2D:
    r"""
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

    def minimize(
        self,
        f: Callable[[list[float]], float],
        x0: Sequence[float],
        f_rotosolve2d_optimized: None | Callable[[list[float], list[float], int], list[float]] = None,
    ) -> Result:
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

        num_of_params = len(self._param_names)

        for _ in range(self.max_iterations):

            # Generate all possible pairs (i, j, R(i), R(j)) where i < j.
            par_pairs = [(i, j, self._param_names[i], self._param_names[j]) for i in range(0, num_of_params)
                     for j in range(i + 1, num_of_params)]
            np.random.shuffle(par_pairs)

            for i, j, par_name1, par_name2 in par_pairs:
                # For the optimized classical energy calculation of rotosolve 2D,
                # It is assumed that the first parameters index is the largest.
                if i < j:
                    i, j = j, i
                    par_name1, par_name2 = par_name2, par_name1
                # Get the energy for specific values of theta_i and theta_j, defined by the _R parameter.
                if f_rotosolve2d_optimized is not None:
                    e_vals = get_energy_evals2d_optimized(f_rotosolve2d_optimized, x, i, j, self._R[par_name1], self._R[par_name2])
                else:
                    e_vals = get_energy_evals_2d(f, x, i, j, self._R[par_name1], self._R[par_name2])
                # Do an analytic construction of the energy as a function of theta_i and theta_j.
                coeffs = get_coefficients_2d(e_vals, self._R[par_name1], self._R[par_name2])
                f_reconstructed = partial(reconstructed_f_2d, coeffs=coeffs, R1=self._R[par_name1], R2=self._R[par_name2])
                g_reconstructed = partial(reconstructed_grad_2d, coeffs=coeffs, R1=self._R[par_name1], R2=self._R[par_name2])

                e_best = 10**20
                theta1_best = 0.0
                theta2_best = 0.0
                # Nyquist-Shannon sampling theorem, https://arxiv.org/pdf/2409.05939
                for theta_i in np.linspace(-np.pi, np.pi, 2*self._R[par_name1]+1):
                    for theta_j in np.linspace(-np.pi, np.pi, 2*self._R[par_name2]+1):
                        res: Any = scipy.optimize.minimize(f_reconstructed, x0=np.array([theta_i, theta_j]), jac=g_reconstructed, method="BFGS", tol=1e-12)
                        if res.fun < e_best:
                            e_best = res.fun
                            theta1_best = res.x[0]
                            theta2_best = res.x[1]
                x[i] = theta1_best
                x[j] = theta2_best
                while x[i] < np.pi:
                    x[i] += 2 * np.pi
                while x[i] > np.pi:
                    x[i] -= 2 * np.pi

                while x[j] < np.pi:
                    x[j] += 2 * np.pi
                while x[j] > np.pi:
                    x[j] -= 2 * np.pi
            f_tmp = f(x)
            f_new = float(np.mean(f_tmp))
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


def get_energy_evals_2d(f: Callable[[list[float]], float], x: list[float], idx1: int, idx2: int, R1: int, R2: int) -> list[float]:
    """
    """
    e_vals = []
    x = x.copy()

    # The number of energy measurements is equal to the number of coefficients.
    for mu1 in range(-R1, R1 + 1):
        x_mu1 = 2 * mu1 / (2 * R1 + 1) * np.pi
        x[idx1] = x_mu1
        for mu2 in range(-R2, R2 + 1):
            x_mu2 = 2 * mu2 / (2 * R2 + 1) * np.pi
            x[idx2] = x_mu2
            e_vals.append(f(x))
    return e_vals  # type: ignore


def get_coefficients_2d(energy_vals: list[float], R1: int, R2: int) -> np.ndarray:
    r"""
    """
    # Number of unique positive differences.
    upd1 = 2 * R1 + 1
    upd2 = 2 * R2 + 1

    # Number of coefficients.
    nc = upd1 * upd2

    angle1_list = [(k / upd1) * (2 * np.pi) for k in range(-R1, R1 + 1)]
    angle2_list = [(k / upd2) * (2 * np.pi) for k in range(-R2, R2 + 1)]

    v_krons = [0] * nc
    row_num = 0
    for k in range(upd1):
        for l in range(upd2):
            v1 = np.zeros((upd1, 1))
            v2 = np.zeros((upd2, 1))
            v1[0] = 1
            for j in range(1, R1 + 1):
                v1[j] = np.cos(j * angle1_list[k])
                v1[R1 + j] = np.sin(j * angle1_list[k])

            v2[0] = 1
            for j in range(1, R2 + 1):
                v2[j] = np.cos(j * angle2_list[l])
                v2[R2 + j] = np.sin(j * angle2_list[l])
            v_krons[row_num] = np.kron(v1, v2).T
            row_num += 1
    A = np.vstack(v_krons)
    return np.linalg.lstsq(A, energy_vals)[0]


@nb.jit(nopython=True)
def reconstructed_f_2d(xy_vals: np.ndarray, coeffs: np.ndarray, R1: int, R2: int) -> float:
    r"""
    """
    # Number of unique positive differences.
    upd1 = 2 * R1 + 1
    upd2 = 2 * R2 + 1
    # Evaluation of function in external values.
    x = xy_vals[0]
    y = xy_vals[1]
    v1 = np.zeros(upd1)
    v2 = np.zeros(upd2)
    v1[0] = 1
    for i in range(1, R1 + 1):
        v1[i] = np.cos(i * x)
        v1[R1 + i] = np.sin(i * x)

    v2[0] = 1
    for i in range(1, R2 + 1):
        v2[i] = np.cos(i * y)
        v2[R2 + i] = np.sin(i * y)

    v_kron = np.kron(v1, v2)
    return np.dot(coeffs, v_kron)


@nb.jit(nopython=True)
def reconstructed_grad_2d(xy_vals: np.ndarray, coeffs: np.ndarray, R1: int, R2: int) -> tuple[float, float]:
    r"""
    """
    # Number of unique positive differences.
    upd1 = 2 * R1 + 1
    upd2 = 2 * R2 + 1
    # Evaluation of function in external values.
    x = xy_vals[0]
    y = xy_vals[1]
    v1 = np.zeros(upd1)
    v2 = np.zeros(upd2)
    v1_grad = np.zeros(upd1)
    v2_grad = np.zeros(upd2)
    v1[0] = 1
    v1_grad[0] = 0
    for i in range(1, R1 + 1):
        v1[i] = np.cos(i * x)
        v1[R1 + i] = np.sin(i * x)
        v1_grad[i] = -i*np.sin(i * x)
        v1_grad[R1 + i] = i*np.cos(i * x)

    v2[0] = 1
    v2_grad[0] = 0
    for i in range(1, R2 + 1):
        v2[i] = np.cos(i * y)
        v2[R2 + i] = np.sin(i * y)
        v2_grad[i] = -i*np.sin(i * y)
        v2_grad[R2 + i] = i*np.cos(i * y)

    v1_kron = np.kron(v1_grad, v2)
    v2_kron = np.kron(v1, v2_grad)
    return np.dot(coeffs, v1_kron), np.dot(coeffs, v2_kron)
