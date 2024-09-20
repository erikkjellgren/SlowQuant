# pylint: disable=too-many-lines
from __future__ import annotations

import time
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import scipy
import scipy.optimize

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.density_matrix import (
    ReducedDenstiyMatrix,
    get_orbital_gradient,
)
from slowquant.unitary_coupled_cluster.operator_matrix import (
    build_operator_matrix,
    construct_ups_state,
    expectation_value,
    get_grad_action,
    get_indexing,
    propagate_unitary,
)
from slowquant.unitary_coupled_cluster.operators import Epq, hamiltonian_0i_0a
from slowquant.unitary_coupled_cluster.util import UpsStructure


class WaveFunctionUPS:
    def __init__(
        self,
        num_spin_orbs: int,
        num_elec: int,
        cas: Sequence[int],
        c_orthonormal: np.ndarray,
        h_ao: np.ndarray,
        g_ao: np.ndarray,
        ansatz: str,
        ansatz_options: dict[str, Any] | None = None,
        include_active_kappa: bool = False,
    ) -> None:
        """Initialize for UPS wave function.

        Args:
            num_spin_orbs: Number of spin orbitals.
            num_elec: Number of electrons.
            cas: CAS(num_active_elec, num_active_orbs),
                 orbitals are counted in spatial basis.
            c_orthonormal: Initial orbital coefficients.
            h_ao: One-electron integrals in AO for Hamiltonian.
            g_ao: Two-electron integrals in AO.
            ansatz: Name of ansatz.
            ansatz_options: Ansatz options.
            include_active_kappa: Include active-active orbital rotations.
        """
        if ansatz_options is None:
            ansatz_options = {}
        if len(cas) != 2:
            raise ValueError(f"cas must have two elements, got {len(cas)} elements.")
        self._c_orthonormal = c_orthonormal
        self.h_ao = h_ao
        self.g_ao = g_ao
        self.inactive_spin_idx = []
        self.virtual_spin_idx = []
        self.active_spin_idx = []
        self.active_occ_spin_idx = []
        self.active_unocc_spin_idx = []
        self.active_spin_idx_shifted = []
        self.active_occ_spin_idx_shifted = []
        self.active_unocc_spin_idx_shifted = []
        self.active_idx_shifted = []
        self.active_occ_idx_shifted = []
        self.active_unocc_idx_shifted = []
        self.num_elec = num_elec
        self.num_elec_alpha = num_elec // 2
        self.num_elec_beta = num_elec // 2
        self.num_spin_orbs = num_spin_orbs
        self.num_orbs = num_spin_orbs // 2
        self._include_active_kappa = include_active_kappa
        self.num_active_elec = 0
        self.num_active_spin_orbs = 0
        self.num_inactive_spin_orbs = 0
        self.num_virtual_spin_orbs = 0
        self._rdm1 = None
        self._rdm2 = None
        self._h_mo = None
        self._g_mo = None
        self._energy_elec: float | None = None
        self.ansatz_options = ansatz_options
        active_space = []
        orbital_counter = 0
        for i in range(num_elec - cas[0], num_elec):
            active_space.append(i)
            orbital_counter += 1
        for i in range(num_elec, num_elec + 2 * cas[1] - orbital_counter):
            active_space.append(i)
        for i in range(num_elec):
            if i in active_space:
                self.active_spin_idx.append(i)
                self.active_occ_spin_idx.append(i)
                self.num_active_spin_orbs += 1
                self.num_active_elec += 1
            else:
                self.inactive_spin_idx.append(i)
                self.num_inactive_spin_orbs += 1
        for i in range(num_elec, num_spin_orbs):
            if i in active_space:
                self.active_spin_idx.append(i)
                self.active_unocc_spin_idx.append(i)
                self.num_active_spin_orbs += 1
            else:
                self.virtual_spin_idx.append(i)
                self.num_virtual_spin_orbs += 1
        self.num_active_elec_alpha = self.num_active_elec // 2
        self.num_active_elec_beta = self.num_active_elec // 2
        self.num_inactive_orbs = self.num_inactive_spin_orbs // 2
        self.num_active_orbs = self.num_active_spin_orbs // 2
        self.num_virtual_orbs = self.num_virtual_spin_orbs // 2
        # Contruct spatial idx
        self.inactive_idx: list[int] = []
        self.virtual_idx: list[int] = []
        self.active_idx: list[int] = []
        self.active_occ_idx: list[int] = []
        self.active_unocc_idx: list[int] = []
        for idx in self.inactive_spin_idx:
            if idx // 2 not in self.inactive_idx:
                self.inactive_idx.append(idx // 2)
        for idx in self.active_spin_idx:
            if idx // 2 not in self.active_idx:
                self.active_idx.append(idx // 2)
        for idx in self.virtual_spin_idx:
            if idx // 2 not in self.virtual_idx:
                self.virtual_idx.append(idx // 2)
        for idx in self.active_occ_spin_idx:
            if idx // 2 not in self.active_occ_idx:
                self.active_occ_idx.append(idx // 2)
        for idx in self.active_unocc_spin_idx:
            if idx // 2 not in self.active_unocc_idx:
                self.active_unocc_idx.append(idx // 2)
        # Make shifted indecies
        if len(self.active_spin_idx) != 0:
            active_shift = np.min(self.active_spin_idx)
            for active_idx in self.active_spin_idx:
                self.active_spin_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_occ_spin_idx:
                self.active_occ_spin_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_unocc_spin_idx:
                self.active_unocc_spin_idx_shifted.append(active_idx - active_shift)
        if len(self.active_idx) != 0:
            active_shift = np.min(self.active_idx)
            for active_idx in self.active_idx:
                self.active_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_occ_idx:
                self.active_occ_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_unocc_idx:
                self.active_unocc_idx_shifted.append(active_idx - active_shift)
        # Find non-redundant kappas
        self.kappa = []
        self.kappa_idx = []
        self.kappa_no_activeactive_idx = []
        self.kappa_no_activeactive_idx_dagger = []
        self.kappa_redundant = []
        self.kappa_redundant_idx = []
        self._kappa_old = []
        self._kappa_redundant_old = []
        # kappa can be optimized in spatial basis
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                if p in self.inactive_idx and q in self.inactive_idx:
                    self.kappa_redundant.append(0.0)
                    self._kappa_redundant_old.append(0.0)
                    self.kappa_redundant_idx.append([p, q])
                    continue
                if p in self.virtual_idx and q in self.virtual_idx:
                    self.kappa_redundant.append(0.0)
                    self._kappa_redundant_old.append(0.0)
                    self.kappa_redundant_idx.append([p, q])
                    continue
                if not include_active_kappa:
                    if p in self.active_idx and q in self.active_idx:
                        self.kappa_redundant.append(0.0)
                        self._kappa_redundant_old.append(0.0)
                        self.kappa_redundant_idx.append([p, q])
                        continue
                if not (p in self.active_idx and q in self.active_idx):
                    self.kappa_no_activeactive_idx.append([p, q])
                    self.kappa_no_activeactive_idx_dagger.append([q, p])
                self.kappa.append(0.0)
                self._kappa_old.append(0.0)
                self.kappa_idx.append([p, q])
        # HF like orbital rotation indecies
        self.kappa_hf_like_idx = []
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                if p in self.inactive_idx and q in self.virtual_idx:
                    self.kappa_hf_like_idx.append([p, q])
                elif p in self.inactive_idx and q in self.active_unocc_idx:
                    self.kappa_hf_like_idx.append([p, q])
                elif p in self.active_occ_idx and q in self.virtual_idx:
                    self.kappa_hf_like_idx.append([p, q])
        # Construct determinant basis
        self.idx2det, self.det2idx = get_indexing(
            self.num_active_orbs, self.num_active_elec_alpha, self.num_active_elec_beta
        )
        self.num_det = len(self.idx2det)
        self.csf_coeffs = np.zeros(self.num_det)
        hf_det = int("1" * self.num_active_elec + "0" * (self.num_active_spin_orbs - self.num_active_elec), 2)
        self.csf_coeffs[self.det2idx[hf_det]] = 1
        self.ci_coeffs = np.copy(self.csf_coeffs)
        self.ups_layout = UpsStructure()
        if ansatz.lower() == "tups":
            self.ups_layout.create_tups(self.num_active_orbs, self.ansatz_options)
        elif ansatz.lower() == "qnp":
            self.ansatz_options["do_qnp"] = True
            self.ups_layout.create_tups(self.num_active_orbs, self.ansatz_options)
        elif ansatz.lower() == "fucc":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ups_layout.create_fUCC(self.num_active_orbs, self.num_active_elec, self.ansatz_options)
        elif ansatz.lower() == "ksafupccgsd":
            self.ups_layout.create_kSAfUpCCGSD(self.num_active_orbs, self.ansatz_options)
        elif ansatz.lower() == "duccsd":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ups_layout.create_dUCCSD(self.num_active_orbs, self.num_active_elec, self.ansatz_options)
        elif ansatz.lower() == "ksadupccgsd":
            self.ups_layout.create_kSAdUpCCGSD(self.num_active_orbs, self.ansatz_options)
        else:
            raise ValueError(f"Got unknown ansatz, {ansatz}")
        self._thetas = np.zeros(self.ups_layout.n_params).tolist()

    @property
    def c_orthonormal(self) -> np.ndarray:
        """Get orthonormalization coefficients (MO coefficients).

        Returns:
            Orthonormalization coefficients.
        """
        return self._c_orthonormal

    @c_orthonormal.setter
    def c_orthonormal(self, c: np.ndarray) -> None:
        """Set orthonormalization coefficients.

        Args:
            c: Orthonormalization coefficients.
        """
        self._h_mo = None
        self._g_mo = None
        self._energy_elec = None
        self._c_orthonormal = c

    @property
    def thetas(self) -> list[float]:
        """Get theta values.

        Returns:
            theta values.
        """
        return self._thetas.copy()

    @thetas.setter
    def thetas(self, theta_vals: list[float]) -> None:
        """Set theta values.

        Args:
            theta_vals: theta values.
        """
        if len(theta_vals) != len(self._thetas):
            raise ValueError(f"Expected {len(self._thetas)} theta1 values got {len(theta_vals)}")
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._energy_elec = None
        self._thetas = theta_vals.copy()
        self.ci_coeffs = construct_ups_state(
            self.csf_coeffs,
            self.num_active_orbs,
            self.num_active_elec_alpha,
            self.num_active_elec_beta,
            self.thetas,
            self.ups_layout,
        )

    @property
    def c_trans(self) -> np.ndarray:
        """Get orbital coefficients.

        Returns:
            Orbital coefficients.
        """
        kappa_mat = np.zeros_like(self._c_orthonormal)
        if len(self.kappa) != 0:
            if np.max(np.abs(self.kappa)) > 0.0:
                for kappa_val, (p, q) in zip(self.kappa, self.kappa_idx):
                    kappa_mat[p, q] = kappa_val
                    kappa_mat[q, p] = -kappa_val
        if len(self.kappa_redundant) != 0:
            if np.max(np.abs(self.kappa_redundant)) > 0.0:
                for kappa_val, (p, q) in zip(self.kappa_redundant, self.kappa_redundant_idx):
                    kappa_mat[p, q] = kappa_val
                    kappa_mat[q, p] = -kappa_val
        return np.matmul(self._c_orthonormal, scipy.linalg.expm(-kappa_mat))

    @property
    def h_mo(self) -> np.ndarray:
        """Get one-electron Hamiltonian integrals in MO basis.

        Returns:
            One-electron Hamiltonian integrals in MO basis.
        """
        if self._h_mo is None:
            self._h_mo = one_electron_integral_transform(self.c_trans, self.h_ao)
        return self._h_mo

    @property
    def g_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._g_mo is None:
            self._g_mo = two_electron_integral_transform(self.c_trans, self.g_ao)
        return self._g_mo

    @property
    def rdm1(self) -> np.ndarray:
        """Calcuate one-electron reduced density matrix.

        Returns:
            One-electron reduced density matrix.
        """
        if self._rdm1 is None:
            self._rdm1 = np.zeros((self.num_active_orbs, self.num_active_orbs))
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_idx = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_idx = q - self.num_inactive_orbs
                    val = expectation_value(
                        self.ci_coeffs,
                        [Epq(p, q)],
                        self.ci_coeffs,
                        self.idx2det,
                        self.det2idx,
                        self.num_inactive_orbs,
                        self.num_active_orbs,
                        self.num_inactive_orbs,
                        self.num_active_elec_alpha,
                        self.num_active_elec_beta,
                        self.thetas,
                        self.ups_layout,
                    )
                    self._rdm1[p_idx, q_idx] = val  # type: ignore
                    self._rdm1[q_idx, p_idx] = val  # type: ignore
        return self._rdm1

    @property
    def rdm2(self) -> np.ndarray:
        """Calcuate two-electron reduced density matrix.

        Returns:
            Two-electron reduced density matrix.
        """
        if self._rdm2 is None:
            self._rdm2 = np.zeros(
                (
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                )
            )
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_idx = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_idx = q - self.num_inactive_orbs
                    for r in range(self.num_inactive_orbs, p + 1):
                        r_idx = r - self.num_inactive_orbs
                        if p == q:
                            s_lim = r + 1
                        elif p == r:
                            s_lim = q + 1
                        elif q < r:
                            s_lim = p
                        else:
                            s_lim = p + 1
                        for s in range(self.num_inactive_orbs, s_lim):
                            s_idx = s - self.num_inactive_orbs
                            val = expectation_value(
                                self.ci_coeffs,
                                [Epq(p, q) * Epq(r, s)],
                                self.ci_coeffs,
                                self.idx2det,
                                self.det2idx,
                                self.num_inactive_orbs,
                                self.num_active_orbs,
                                self.num_inactive_orbs,
                                self.num_active_elec_alpha,
                                self.num_active_elec_beta,
                                self.thetas,
                                self.ups_layout,
                            )
                            if q == r:
                                val -= self.rdm1[p_idx, s_idx]
                            self._rdm2[p_idx, q_idx, r_idx, s_idx] = val  # type: ignore
                            self._rdm2[r_idx, s_idx, p_idx, q_idx] = val  # type: ignore
                            self._rdm2[q_idx, p_idx, s_idx, r_idx] = val  # type: ignore
                            self._rdm2[s_idx, r_idx, q_idx, p_idx] = val  # type: ignore
        return self._rdm2

    def check_orthonormality(self, overlap_integral: np.ndarray) -> None:
        r"""Check orthonormality of orbitals.

        .. math::
            \boldsymbol{I} = \boldsymbol{C}_\text{MO}\boldsymbol{S}\boldsymbol{C}_\text{MO}^T

        Args:
            overlap_integral: Overlap integral in AO basis.
        """
        S_ortho = one_electron_integral_transform(self.c_trans, overlap_integral)
        one = np.identity(len(S_ortho))
        diff = np.abs(S_ortho - one)
        print("Max ortho-normal diff:", np.max(diff))

    @property
    def energy_elec(self) -> float:
        """Get the electronic energy.

        Returns:
            Electronic energy.
        """
        if self._energy_elec is None:
            self._energy_elec = energy_ups(self.thetas, False, self)
        return self._energy_elec

    def run_ups(
        self,
        orbital_optimization: bool = False,
        is_silent: bool = False,
        convergence_threshold: float = 10**-10,
        maxiter: int = 10000,
    ) -> None:
        """Run optimization of UPS wave function.

        Args:
            orbital_optimization: Do orbital optimization.
            is_silent: Do not print any output.
            convergence_threshold: Energy threshold for convergence.
            maxiter: Maximum number of iterations.
        """
        e_tot = partial(
            energy_ups,
            orbital_optimized=orbital_optimization,
            wf=self,
        )
        parameter_gradient = partial(
            gradient_ups,
            orbital_optimized=orbital_optimization,
            wf=self,
        )
        global iteration  # pylint: disable=global-variable-undefined
        global start  # pylint: disable=global-variable-undefined
        iteration = 0  # type: ignore
        start = time.time()  # type: ignore

        def print_progress(x: Sequence[float]) -> None:
            """Print progress during energy minimization of wave function.

            Args:
                x: Wave function parameters.
            """
            global iteration  # pylint: disable=global-variable-undefined
            global start  # pylint: disable=global-variable-undefined
            time_str = f"{time.time() - start:7.2f}"  # type: ignore
            e_str = f"{e_tot(x):3.12f}"
            print(f"{str(iteration + 1).center(11)} | {time_str.center(18)} | {e_str.center(27)}")  # type: ignore
            iteration += 1  # type: ignore [name-defined]
            start = time.time()  # type: ignore [name-defined]

        def silent_progress(x: Sequence[float]) -> None:  # pylint: disable=unused-argument
            """Print progress during energy minimization of wave function.

            Args:
                x: Wave function parameters.
            """
            pass  # pylint: disable=unnecessary-pass

        parameters: list[float] = []
        num_kappa = 0
        num_theta = 0
        if orbital_optimization:
            parameters += self.kappa
            num_kappa += len(self.kappa)
        for theta in self.thetas:
            parameters.append(theta)
            num_theta += 1
        if is_silent:
            res = scipy.optimize.minimize(
                e_tot,
                parameters,
                tol=convergence_threshold,
                callback=silent_progress,
                method="SLSQP",
                jac=parameter_gradient,
            )
        else:
            print("### Parameters information:")
            print(f"### Number kappa: {num_kappa}")
            print(f"### Number theta: {num_theta}")
            print(f"### Total parameters: {num_kappa + num_theta}\n")
            print("Iteration # | Iteration time [s] | Electronic energy [Hartree]")
            res = scipy.optimize.minimize(
                e_tot,
                parameters,
                tol=convergence_threshold,
                callback=print_progress,
                method="SLSQP",
                jac=parameter_gradient,
                options={"maxiter": maxiter},
            )
        self._energy_elec = res["fun"]
        param_idx = 0
        if orbital_optimization:
            param_idx += len(self.kappa)
            for i in range(len(self.kappa)):  # pylint: disable=consider-using-enumerate
                self.kappa[i] = 0
                self._kappa_old[i] = 0
            for i in range(len(self.kappa_redundant)):  # pylint: disable=consider-using-enumerate
                self.kappa_redundant[i] = 0
                self._kappa_redundant_old[i] = 0
        self.thetas = res["x"][param_idx : num_theta + param_idx].tolist()


def energy_ups(
    parameters: Sequence[float],
    orbital_optimized: bool,
    wf: WaveFunctionUPS,
) -> float:
    r"""Calculate electronic energy of SA-UPS wave function.

    .. math::
        E = \left<0\left|\hat{H}\right|0\right>

    Args:
        parameters: Sequence of all parameters.
                    Ordered as orbital rotations, active-space excitations.
        orbital_optimized: Do orbital optimization.
        wf: Wave function object.

    Returns:
        Electronic energy.
    """
    kappa = []
    theta = []
    idx_counter = 0
    if orbital_optimized:
        for _ in range(len(wf.kappa_idx)):
            kappa.append(parameters[idx_counter])
            idx_counter += 1
    for par in parameters[idx_counter:]:
        theta.append(par)
    assert len(parameters) == len(kappa) + len(theta)

    kappa_mat = np.zeros_like(wf.c_orthonormal)
    if orbital_optimized:
        for kappa_val, (p, q) in zip(
            np.array(kappa) - np.array(wf._kappa_old), wf.kappa_idx  # pylint: disable=protected-access
        ):
            kappa_mat[p, q] = kappa_val
            kappa_mat[q, p] = -kappa_val
    if len(wf.kappa_redundant) != 0:
        if np.max(np.abs(wf.kappa_redundant)) > 0.0:
            for kappa_val, (p, q) in zip(
                np.array(wf.kappa_redundant)
                - np.array(wf._kappa_redundant_old),  # pylint: disable=protected-access
                wf.kappa_redundant_idx,
            ):
                kappa_mat[p, q] = kappa_val
                kappa_mat[q, p] = -kappa_val
    c_trans = np.matmul(wf.c_orthonormal, scipy.linalg.expm(-kappa_mat))
    if orbital_optimized:
        wf._kappa_old = kappa.copy()  # pylint: disable=protected-access
        wf._kappa_redundant_old = wf.kappa_redundant.copy()  # pylint: disable=protected-access
    # Moving expansion point of kappa
    wf.c_orthonormal = c_trans
    # Add thetas
    wf.thetas = theta
    return expectation_value(
        wf.ci_coeffs,
        [
            hamiltonian_0i_0a(
                wf.h_mo,
                wf.g_mo,
                wf.num_inactive_orbs,
                wf.num_active_orbs,
            )
        ],
        wf.ci_coeffs,
        wf.idx2det,
        wf.det2idx,
        wf.num_inactive_orbs,
        wf.num_active_orbs,
        wf.num_inactive_orbs,
        wf.num_active_elec_alpha,
        wf.num_active_elec_beta,
        wf.thetas,
        wf.ups_layout,
    )


def gradient_ups(
    parameters: Sequence[float],
    orbital_optimized: bool,
    wf: WaveFunctionUPS,
) -> np.ndarray:
    """Calcuate electronic gradient.

    Args:
        parameters: Sequence of all parameters.
                    Ordered as orbital rotations, active-space excitations.
        orbital_optimized: Do orbital optimization.
        wf: Wave function object.

    Returns:
        Electronic gradient.
    """
    number_kappas = 0
    if orbital_optimized:
        number_kappas = len(wf.kappa_idx)
    gradient = np.zeros_like(parameters)
    if orbital_optimized:
        gradient[:number_kappas] = orbital_rotation_gradient(
            wf,
        )
    gradient[number_kappas:] = active_space_parameter_gradient(
        wf,
    )
    return gradient


def orbital_rotation_gradient(
    wf: WaveFunctionUPS,
) -> np.ndarray:
    """Calcuate electronic gradient with respect to orbital rotations.

    Args:
        wf: Wave function object.

    Return:
        Electronic gradient with respect to orbital rotations.
    """
    rdms = ReducedDenstiyMatrix(
        wf.num_inactive_orbs,
        wf.num_active_orbs,
        wf.num_active_orbs,
        rdm1=wf.rdm1,
        rdm2=wf.rdm2,
    )
    gradient = get_orbital_gradient(
        rdms, wf.h_mo, wf.g_mo, wf.kappa_idx, wf.num_inactive_orbs, wf.num_active_orbs
    )
    return gradient


def active_space_parameter_gradient(
    wf: WaveFunctionUPS,
) -> np.ndarray:
    """Calcuate electronic gradient with respect to active space parameters.

    Args:
        wf: Wave function object.

    Returns:
        Electronic gradient with respect to active spae parameters.
    """
    Hamiltonian = build_operator_matrix(
        hamiltonian_0i_0a(
            wf.h_mo,
            wf.g_mo,
            wf.num_inactive_orbs,
            wf.num_active_orbs,
        ).get_folded_operator(wf.num_inactive_orbs, wf.num_active_orbs, wf.num_virtual_orbs),
        wf.idx2det,
        wf.det2idx,
        wf.num_active_orbs,
    )

    gradient_theta = np.zeros_like(wf.thetas)
    bra_vec = construct_ups_state(
        np.matmul(Hamiltonian, wf.ci_coeffs),
        wf.num_active_orbs,
        wf.num_active_elec_alpha,
        wf.num_active_elec_beta,
        wf.thetas,
        wf.ups_layout,
        dagger=True,
    )
    ket_vec = np.copy(wf.csf_coeffs)
    ket_vec_tmp = np.copy(wf.csf_coeffs)
    for i in range(len(wf.thetas)):
        ket_vec_tmp = get_grad_action(
            ket_vec,
            i,
            wf.num_active_orbs,
            wf.num_active_elec_alpha,
            wf.num_active_elec_beta,
            wf.ups_layout,
        )
        gradient_theta[i] += 2 * np.matmul(bra_vec, ket_vec_tmp)
        bra_vec = propagate_unitary(
            bra_vec,
            i,
            wf.num_active_orbs,
            wf.num_active_elec_alpha,
            wf.num_active_elec_beta,
            wf.thetas,
            wf.ups_layout,
        )
        ket_vec = propagate_unitary(
            ket_vec,
            i,
            wf.num_active_orbs,
            wf.num_active_elec_alpha,
            wf.num_active_elec_beta,
            wf.thetas,
            wf.ups_layout,
        )
    return gradient_theta
