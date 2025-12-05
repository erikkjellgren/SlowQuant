from __future__ import annotations

import time
from functools import partial
from typing import Any

import numpy as np
import scipy
import scipy.optimize

from slowquant.molecularintegrals.integralfunctions import (
    generalized_one_electron_transform,
    generalized_two_electron_transform,
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing_generalized
from slowquant.unitary_coupled_cluster.generalized_density_matrix import (
    get_electronic_energy_generalized,
    get_orbital_gradient_generalized_real_imag,
    get_orbital_gradient_generalized,
)
from slowquant.unitary_coupled_cluster.generalized_operators import (
    a_op_spin,
    generalized_hamiltonian_full_space,
)

# from slowquant.unitary_coupled_cluster.generalized_density_matrix import (
#    get_electronic_energy_generalized,
#    get_orbital_gradient_generalized,
# )
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    construct_ups_state,
    expectation_value,
    get_grad_action,
    propagate_state,
    propagate_unitary,
)
from slowquant.unitary_coupled_cluster.operators import G1, G2
from slowquant.unitary_coupled_cluster.operators import generalized_hamiltonian_0i_0a_spinidx
from slowquant.unitary_coupled_cluster.optimizers import Optimizers
from slowquant.unitary_coupled_cluster.util import (
    UpsStructure,
    iterate_t1,
    iterate_t1_generalized,
    iterate_t2,
    iterate_t2_generalized,
)
from slowquant.unitary_coupled_cluster.util import UpsStructure
from slowquant.unitary_coupled_cluster.generalized_density_matrix import get_orbital_gradient_generalized, get_orbital_gradient_generalized_real_imag, get_electronic_energy_generalized, get_orbital_gradient_test_anna

class GeneralizedWaveFunctionUPS:
    def __init__(
        self,
        num_elec: int,
        cas: tuple[tuple[int, int], int],
        mo_coeffs: np.ndarray,
        h_ao: np.ndarray,
        g_ao: np.ndarray,
        ansatz: str,
        ansatz_options: dict[str, Any] | None = None,
        include_active_kappa: bool = False,
    ) -> None:
        """Initialize for UPS wave function.

        Args:
            num_elec: Number of electrons.
            cas: CAS(num_active_elec, num_active_orbs),
                 orbitals are counted in spatial basis.
            mo_coeffs: Initial orbital coefficients.
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
        if len(cas[0]) != 2:
            raise ValueError(
                "Number of electrons in the active space must be specified as a tuple of (alpha, beta)."
            )
        if cas[1] > len(h_ao):
            raise ValueError(
                f"More spatial active orbitls than total orbitals. Got {cas[1]} active orbitals, and {len(h_ao)} total orbitals."
            )
        if np.sum(cas[0]) > num_elec:
            raise ValueError(
                f"More active electrons than total electrons. Got {np.sum(cas[0])} active electrons, and {num_elec} total electrons."
            )
        # Init stuff
        self._c_mo = mo_coeffs
        self._h_ao = h_ao
        self._g_ao = g_ao
        self._rdm1 = None
        self._rdm2 = None
        self._h_mo = None
        self._g_mo = None
        self._energy_elec: float | None = None
        self.ansatz_options = ansatz_options
        self.num_energy_evals = 0
        # Construct spin orbital spaces and indices
        self.inactive_spin_idx = []
        self.virtual_spin_idx = []
        self.active_spin_idx = []
        self.active_occ_spin_idx = []
        self.active_unocc_spin_idx = []
        self.active_spin_idx_shifted = []
        self.active_occ_spin_idx_shifted = []
        self.active_unocc_spin_idx_shifted = []
        self.num_elec = num_elec
        self.num_elec_alpha = (num_elec - np.sum(cas[0])) // 2 + cas[0][0]
        self.num_elec_beta = (num_elec - np.sum(cas[0])) // 2 + cas[0][1]
        self.num_spin_orbs = 2 * len(h_ao)
        self._include_active_kappa = include_active_kappa
        self.num_active_elec_alpha = cas[0][0]
        self.num_active_elec_beta = cas[0][1]
        self.num_active_elec = self.num_active_elec_alpha + self.num_active_elec_beta
        self.num_active_spin_orbs = 2 * cas[1]
        self.num_inactive_spin_orbs = self.num_elec - self.num_active_elec
        self.num_virtual_spin_orbs = 2 * len(h_ao) - self.num_inactive_spin_orbs - self.num_active_spin_orbs
        # Find non-redundant kappas
        self._kappa_real = []
        self._kappa_imag = []
        self.kappa_spin_idx = []
        self.kappa_no_activeactive_spin_idx = []
        self.kappa_no_activeactive_spin_idx_dagger = []
        self._kappa_real_redundant = []
        self._kappa_imag_redundant = []
        self.kappa_redundant_spin_idx = []
        self._kappa_real_old = []
        self._kappa_imag_old = []
        self._kappa_real_redundant_old = []
        self._kappa_imag_redundant_old = []
        for p in range(0, self.num_spin_orbs):
            for q in range(p, self.num_spin_orbs):
                if p in self.inactive_spin_idx and q in self.inactive_spin_idx:
                    self._kappa_real_redundant.append(0.0)
                    self._kappa_imag_redundant.append(0.0)
                    self._kappa_real_redundant_old.append(0.0)
                    self._kappa_imag_redundant_old.append(0.0)
                    self.kappa_redundant_spin_idx.append((p, q))
                    continue
                if p in self.virtual_spin_idx and q in self.virtual_spin_idx:
                    self._kappa_real_redundant.append(0.0)
                    self._kappa_imag_redundant.append(0.0)
                    self._kappa_real_redundant_old.append(0.0)
                    self._kappa_imag_redundant_old.append(0.0)
                    self.kappa_redundant_spin_idx.append((p, q))
                    continue
                if not include_active_kappa:
                    if p in self.active_spin_idx and q in self.active_spin_idx:
                        self._kappa_real_redundant.append(0.0)
                        self._kappa_imag_redundant.append(0.0)
                        self._kappa_real_redundant_old.append(0.0)
                        self._kappa_imag_redundant_old.append(0.0)
                        self.kappa_redundant_spin_idx.append((p, q))
                        continue
                if include_active_kappa:
                    if p in self.active_occ_spin_idx and q in self.active_occ_spin_idx:
                        self._kappa_real_redundant.append(0.0)
                        self._kappa_imag_redundant.append(0.0)
                        self._kappa_real_redundant_old.append(0.0)
                        self._kappa_imag_redundant_old.append(0.0)
                        self.kappa_redundant_spin_idx.append((p, q))
                        continue
                    if p in self.active_unocc_spin_idx and q in self.active_unocc_spin_idx:
                        self._kappa_real_redundant.append(0.0)
                        self._kappa_imag_redundant.append(0.0)
                        self._kappa_real_redundant_old.append(0.0)
                        self._kappa_imag_redundant_old.append(0.0)
                        self.kappa_redundant_spin_idx.append((p, q))
                        continue
                if not (p in self.active_spin_idx and q in self.active_spin_idx):
                    self.kappa_no_activeactive_spin_idx.append((p, q))
                    self.kappa_no_activeactive_spin_idx_dagger.append((q, p))
                self._kappa_real.append(0.0)
                self._kappa_imag.append(0.0)
                self._kappa_real_old.append(0.0)
                self._kappa_imag_old.append(0.0)
                self.kappa_spin_idx.append((p, q))
        # Construct determinant basis
        self.ci_info = get_indexing_generalized(
            self.num_inactive_spin_orbs,
            self.num_active_spin_orbs,
            self.num_virtual_spin_orbs,
            self.num_active_elec_alpha,
            self.num_active_elec_beta,
        )
        self.num_det = len(self.ci_info.idx2det)
        self.csf_coeffs = np.zeros(self.num_det)
        hf_string = ""
        for i in range(self.num_active_spin_orbs // 2):
            if i < self.num_active_elec_alpha:
                hf_string += "1"
            else:
                hf_string += "0"
            if i < self.num_active_elec_beta:
                hf_string += "1"
            else:
                hf_string += "0"
        hf_det = int(hf_string, 2)
        self.csf_coeffs[self.ci_info.det2idx[hf_det]] = 1
        self.ci_coeffs = np.copy(self.csf_coeffs)
        # Construct UPS Structure
        self.ups_layout = UpsStructure()
        # Do the ansatz setup
        #
        # Note: Put here because I do not know here else to put it.
        #       The operator pool currectly does not have the diagonal elements.
        #       F.x. not the p^dagger p single excitation.
        #       These should be there? (At some point)
        #
        if ansatz.lower() == "fuccsd":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ansatz_options["S"] = True
            self.ansatz_options["D"] = True
            self.ups_layout.create_fUCC(
                [],
                [],
                self.active_occ_spin_idx_shifted,
                self.active_unocc_spin_idx_shifted,
                self.num_active_spin_orbs // 2,
                self.ansatz_options,
            )
        elif ansatz.lower() == "fuccsdt":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ansatz_options["S"] = True
            self.ansatz_options["D"] = True
            self.ansatz_options["T"] = True
            self.ups_layout.create_fUCC(
                [],
                [],
                self.active_occ_spin_idx_shifted,
                self.active_unocc_spin_idx_shifted,
                self.num_active_spin_orbs // 2,
                self.ansatz_options,
            )
        elif ansatz.lower() == "fuccsdtq":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ansatz_options["S"] = True
            self.ansatz_options["D"] = True
            self.ansatz_options["T"] = True
            self.ansatz_options["Q"] = True
            self.ups_layout.create_fUCC(
                [],
                [],
                self.active_occ_spin_idx_shifted,
                self.active_unocc_spin_idx_shifted,
                self.num_active_spin_orbs // 2,
                self.ansatz_options,
            )
        elif ansatz.lower() == "fuccsdtq5":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ansatz_options["S"] = True
            self.ansatz_options["D"] = True
            self.ansatz_options["T"] = True
            self.ansatz_options["Q"] = True
            self.ansatz_options["5"] = True
            self.ups_layout.create_fUCC(
                [],
                [],
                self.active_occ_spin_idx_shifted,
                self.active_unocc_spin_idx_shifted,
                self.num_active_spin_orbs // 2,
                self.ansatz_options,
            )
        elif ansatz.lower() == "fuccsdtq56":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ansatz_options["S"] = True
            self.ansatz_options["D"] = True
            self.ansatz_options["T"] = True
            self.ansatz_options["Q"] = True
            self.ansatz_options["5"] = True
            self.ansatz_options["6"] = True
            self.ups_layout.create_fUCC(
                [],
                [],
                self.active_occ_spin_idx_shifted,
                self.active_unocc_spin_idx_shifted,
                self.num_active_spin_orbs // 2,
                self.ansatz_options,
            )
        elif ansatz.lower() == "adapt":
            None
        else:
            raise ValueError(f"Got unknown ansatz, {ansatz}")
        if self.ups_layout.n_params == 0:
            self._thetas_real = []
            self._thetas_imag = []
        else:
            self._thetas_real = np.zeros(self.ups_layout.n_params).tolist()
            self._thetas_imag = np.zeros(self.ups_layout.n_params).tolist()

    @property
    def kappa_real(self) -> list[float]:
        """Get real orbital rotation parameters."""
        return self._kappa_real.copy()

    @property
    def kappa_imag(self) -> list[float]:
        """Get imaginary orbital rotation parameters."""
        return self._kappa_imag.copy()

    def set_kappa_cep(self, k_real: list[float], k_imag: list[float]) -> None:
        """Set orbital rotation parameters, and move current expansion point.

        Args:
            k: orbital rotation parameters.
        """
        self._h_mo = None
        self._g_mo = None
        self._energy_elec = None
        self._kappa_real = k_real.copy()
        self._kappa_imag = k_imag.copy()
        # Move current expansion point.
        self._c_mo = self.c_mo
        self._kappa_real_old = self.kappa_real
        self._kappa_imag_old = self.kappa_imag

    @property
    def thetas_real(self) -> list[float]:
        """Get real theta values.

        Returns:
            theta values.
        """
        return self._thetas_real.copy()

    @property
    def thetas_imag(self) -> list[float]:
        """Get imaginary theta values.

        Returns:
            theta values.
        """
        return self._thetas_imag.copy()

    @property
    def thetas(self) -> list[complex]:
        """Get complex theta values.

        Returns:
            theta values.
        """
        return (np.array(self._thetas_real.copy()) + 1.0j * np.array(self._thetas_imag.copy())).tolist()

    def set_thetas(self, theta_real: list[float], theta_imag: list[float]) -> None:
        """Set theta values.

        Args:
            theta_vals: theta values.
        """
        if len(theta_real) != len(self._thetas_real):
            raise ValueError(f"Expected {len(self._thetas_real)} real theta values got {len(theta_real)}")
        if len(theta_real) != len(self._thetas_real):
            raise ValueError(
                f"Expected {len(self._thetas_imag)} imaginary theta values got {len(theta_imag)}"
            )
        self._rdm1 = None
        self._rdm2 = None
        self._energy_elec = None
        self._thetas_real = theta_real.copy()
        self._thetas_imag = theta_imag.copy()
        self.ci_coeffs = construct_ups_state(
            self.csf_coeffs,
            self.ci_info,
            self.thetas,
            self.ups_layout,
        )

    @property
    def c_mo(self) -> np.ndarray:
        """Get molecular orbital coefficients.

        Returns:
            Molecular orbital coefficients.
        """
        # Construct anti-hermitian kappa matrix
        kappa_mat = np.zeros_like(self._c_mo)
        if len(self.kappa_real) != 0:
            # The MO transformation is calculated as a difference between current kappa and kappa old.
            # This is to make the moving of the expansion point to work with SciPy optimization algorithms.
            # Resetting kappa to zero would mess with any algorithm that has any memory f.x. BFGS.
            if np.max(np.abs(np.array(self.kappa_real) - np.array(self._kappa_real_old))) > 0.0:
                for kappa_val, kappa_old, (p, q) in zip(
                    self.kappa_real, self._kappa_real_old, self.kappa_spin_idx
                ):
                    kappa_mat[p, q] =   kappa_val - kappa_old
                    kappa_mat[q, p] = -(kappa_val - kappa_old)
            if np.max(np.abs(np.array(self.kappa_imag) - np.array(self._kappa_imag_old))) > 0.0:
                for kappa_val, kappa_old, (p, q) in zip(
                    self.kappa_imag, self._kappa_imag_old, self.kappa_spin_idx
                ):
                    kappa_mat[p, q] += (kappa_val - kappa_old) * 1.0j
                    kappa_mat[q, p] += (kappa_val - kappa_old) * 1.0j
        # Apply orbital rotation unitary to MO coefficients
        return np.matmul(self._c_mo, scipy.linalg.expm(-kappa_mat))

    @property
    def h_mo(self) -> np.ndarray:
        """Get one-electron Hamiltonian integrals in MO basis.

        Returns:
            One-electron Hamiltonian integrals in MO basis.
        """
        if self._h_mo is None:
            self._h_mo = generalized_one_electron_transform(self.c_mo, self._h_ao)
        return self._h_mo

    @property
    def g_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._g_mo is None:
            self._g_mo = generalized_two_electron_transform(self.c_mo, self._g_ao)
        return self._g_mo

    @property
    def rdm1(self) -> np.ndarray:
        """Calculate one-electron reduced density matrix in the active space.

        Returns:
            One-electron reduced density matrix.
        """
        # Annika has added dtype=complex
        if self._rdm1 is None:
            self._rdm1 = np.zeros((self.num_active_spin_orbs, self.num_active_spin_orbs), dtype=complex)
            for P in range(
                self.num_inactive_spin_orbs, self.num_inactive_spin_orbs + self.num_active_spin_orbs
            ):
                P_idx = P - self.num_inactive_spin_orbs
                for Q in range(self.num_inactive_spin_orbs, P + 1):
                    Q_idx = Q - self.num_inactive_spin_orbs
                    val = expectation_value(
                        self.ci_coeffs,
                        [(a_op_spin(P, True) * a_op_spin(Q, False))],
                        self.ci_coeffs,
                        self.ci_info,
                    )
                    self._rdm1[P_idx, Q_idx] = val  # type: ignore
                    self._rdm1[Q_idx, P_idx] = val.conjugate()  # type: ignore (1.7.7 EST)
        return self._rdm1

    @property
    def rdm2(self) -> np.ndarray:
        """Calculate two-electron reduced density matrix in the active space.

        Returns:
            Two-electron reduced density matrix.
        """
        # Annika has added dtype=complex
        if self._rdm2 is None:
            self._rdm2 = np.zeros(
                (
                    self.num_active_spin_orbs,
                    self.num_active_spin_orbs,
                    self.num_active_spin_orbs,
                    self.num_active_spin_orbs,
                ),
            dtype=complex)
            for p in range(
                self.num_inactive_spin_orbs, self.num_inactive_spin_orbs + self.num_active_spin_orbs
            ):
                p_idx = p - self.num_inactive_spin_orbs
                for q in range(self.num_inactive_spin_orbs, p + 1):
                    q_idx = q - self.num_inactive_spin_orbs
                    for r in range(self.num_inactive_spin_orbs, p + 1):
                        r_idx = r - self.num_inactive_spin_orbs
                        if p == q:
                            s_lim = r + 1
                        elif p == r:
                            s_lim = q + 1
                        elif q < r:
                            s_lim = p
                        else:
                            s_lim = p + 1
                        for s in range(self.num_inactive_spin_orbs, s_lim):
                            s_idx = s - self.num_inactive_spin_orbs
                            val = expectation_value(
                                self.ci_coeffs,
                                [
                                    (
                                        a_op_spin(p, True)
                                        * a_op_spin(r, True)
                                        * a_op_spin(s, False)
                                        * a_op_spin(q, False)
                                    )
                                ],
                                self.ci_coeffs,
                                self.ci_info,
                            )
                            if q == r:
                                val -= self.rdm1[p_idx, s_idx]
                            self._rdm2[p_idx, q_idx, r_idx, s_idx] = val  # type: ignore
                            self._rdm2[q_idx, p_idx, s_idx, r_idx] = val.conjugate()  # type: ignore
                            self._rdm2[r_idx, s_idx, p_idx, q_idx] = val  # type: ignore
                            self._rdm2[s_idx, r_idx, q_idx, p_idx] = val.conjugate()  # type: ignore
        return self._rdm2


    @property
    def rdm2_symmetry(self) -> np.ndarray:
        """Calculate two-electron reduced density matrix in the active space based on complex spin orbitals.

        Returns:
            Two-electron reduced density matrix.
        """
        if self._rdm2 is None:
            self._rdm2 = np.zeros(
                (
                    self.num_active_spin_orbs,
                    self.num_active_spin_orbs,
                    self.num_active_spin_orbs,
                    self.num_active_spin_orbs,
                )
            )
            for P in range(self.num_inactive_spin_orbs, self.num_inactive_spin_orbs + self.num_active_spin_orbs):
                P_idx = P - self.num_inactive_spin_orbs
                for Q in range(self.num_inactive_spin_orbs, P + 1):
                    Q_idx = Q - self.num_inactive_spin_orbs
                    for R in range(self.num_inactive_spin_orbs, P + 1):
                        R_idx = R - self.num_inactive_spin_orbs
                        if P == Q:
                            S_lim = R + 1
                        elif P == R:
                            S_lim = Q + 1
                        elif Q < R:
                            S_lim = P # Not sure I understand this limit? Why not R?
                        else:
                            S_lim = P + 1
                        for S in range(self.num_inactive_spin_orbs, S_lim):
                            S_idx = S - self.num_inactive_spin_orbs
                            val = expectation_value(
                                self.ci_coeffs,
                                [a_op_spin(P,dagger=True)*a_op_spin(R,dagger=True)*
                                 a_op_spin(S,dagger=False)*a_op_spin(Q,dagger=False)],
                                self.ci_coeffs,
                                self.ci_info,
                                do_folding=False,
                            )
                            #if Q == R: # No comprehendo
                            #    val -= self.rdm1[P_idx, S_idx]

                            self._rdm2[P_idx, Q_idx, R_idx, S_idx] =  val  # type: ignore
                            self._rdm2[Q_idx, P_idx, S_idx, R_idx] =  val.conjugate()  # type: ignore
                            self._rdm2[R_idx, S_idx, P_idx, Q_idx] =  val  # type: ignore
                            self._rdm2[S_idx, R_idx, Q_idx, P_idx] =  val.conjugate()  # type: ignore

                            self._rdm2[R_idx, Q_idx, P_idx, S_idx] = -val  # type: ignore
                            self._rdm2[P_idx, S_idx, R_idx, Q_idx] = -val  # type: ignore
                            
        return self._rdm2

    def check_orthonormality(self, overlap_integral: np.ndarray) -> None:
        r"""Check orthonormality of orbitals.

        .. math::
            \boldsymbol{I} = \boldsymbol{C}_\text{MO}\boldsymbol{S}\boldsymbol{C}_\text{MO}^T

        Args:
            overlap_integral: Overlap integral in AO basis.
        """
        S_ortho = one_electron_integral_transform(self.c_mo, overlap_integral)
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
            self._energy_elec = expectation_value(
                self.ci_coeffs,
                # Skal ændres til generalized_hamiltonian_0i_0a på et tidspunkt.
                # [generalized_hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_spin_orbs, self.num_active_spin_orbs)],
                [generalized_hamiltonian_full_space(self.h_mo, self.g_mo, self.num_spin_orbs)],
                self.ci_coeffs,
                self.ci_info,
            )
        return self._energy_elec

    def run_wf_optimization_2step(
        self,
        optimizer_name: str,
        orbital_optimization: bool = False,
        tol: float = 1e-10,
        maxiter: int = 1000,
        is_silent_subiterations: bool = False,
    ) -> None:
        """Run two step optimization of wave function.

        Args:
            optimizer_name: Name of optimizer.
            orbital_optimization: Perform orbital optimization.
            tol: Convergence tolerance.
            maxiter: Maximum number of iterations.
            is_silent_subiterations: Silence subiterations.
        """
        print("### Parameters information:")
        if orbital_optimization:
            print(f"### Number kappa: {len(self.kappa_real)}")
        print(f"### Number theta: {self.ups_layout.n_params}")
        e_old = 1e12
        print("Full optimization")
        print("Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #")
        for full_iter in range(0, int(maxiter)):
            full_start = time.time()

            # Do ansatz optimization
            if not is_silent_subiterations:
                print("--------Ansatz optimization")
                print(
                    "--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #"
                )
            energy_theta = partial(
                self._calc_energy_optimization,
                theta_optimization=True,
                kappa_optimization=False,
            )
            gradient_theta = partial(
                self._calc_gradient_optimization,
                theta_optimization=True,
                kappa_optimization=False,
            )
            optimizer = Optimizers(
                energy_theta,
                optimizer_name,
                grad=gradient_theta,
                maxiter=maxiter,
                tol=tol,
                is_silent=is_silent_subiterations,
                energy_eval_callback=lambda: self.num_energy_evals,
            )
            self._old_opt_parameters = 2 * np.zeros_like(self.thetas) + 10**20
            self._E_opt_old = 0.0
            thetas = []
            for theta_r, theta_i in zip(self.thetas_real, self.thetas_imag):
                thetas.append(theta_r)
                thetas.append(theta_i)
            res = optimizer.minimize(
                thetas,
                extra_options={"R": self.ups_layout.grad_param_R, "param_names": self.ups_layout.param_names},
            )
            thetas_r = []
            thetas_i = []
            # OBS!! Thetas are in the order 00_real,00_imag,01_real,01_imag,...
            for i in range(len(self.thetas)):
                thetas_r.append(res.x[2 * i])
                thetas_i.append(res.x[2 * i + 1])
            self.set_thetas(thetas_r, thetas_i)

            if orbital_optimization and len(self.kappa_real) != 0:
                if not is_silent_subiterations:
                    print("--------Orbital optimization")
                    print(
                        "--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #"
                    )
                energy_oo = partial(
                    self._calc_energy_optimization,
                    theta_optimization=False,
                    kappa_optimization=True,
                )
                gradient_oo = partial(
                    self._calc_gradient_optimization,
                    theta_optimization=False,
                    kappa_optimization=True,
                )

                optimizer = Optimizers(
                    energy_oo,
                    "l-bfgs-b",
                    grad=gradient_oo,
                    maxiter=maxiter,
                    tol=tol,
                    is_silent=is_silent_subiterations,
                    energy_eval_callback=lambda: self.num_energy_evals,
                )
                self._old_opt_parameters = np.zeros(2 * len(self.kappa_spin_idx)) + 10**20
                self._E_opt_old = 0.0
                res = optimizer.minimize([0.0] * 2 * len(self.kappa_spin_idx))
                for i in range(len(self.kappa_real)):
                    self._kappa_real[i] = 0.0
                    self._kappa_imag[i] = 0.0
                    self._kappa_real_old[i] = 0.0
                    self._kappa_imag_old[i] = 0.0
            else:
                # If there is no orbital optimization, then the algorithm is already converged.
                e_new = res.fun
                if orbital_optimization and len(self.kappa_real) == 0:
                    print(
                        "WARNING: No orbital optimization performed, because there is no non-redundant orbital parameters."
                    )
                break

            e_new = res.fun
            time_str = f"{time.time() - full_start:7.2f}"
            e_str = f"{e_new:3.12f}"
            print(
                f"{str(full_iter + 1).center(11)} | {time_str.center(18)} | {e_str.center(27)} | {str(self.num_energy_evals).center(11)}"
            )
            if abs(e_new - e_old) < tol:
                break
            e_old = e_new
        self._energy_elec = e_new

    def run_wf_optimization_1step(
        self,
        optimizer_name: str,
        orbital_optimization: bool = False,
        tol: float = 1e-10,
        maxiter: int = 1000,
        is_silent: bool = False,
    ) -> None:
        """Run one step optimization of wave function.

        Args:
            optimizer_name: Name of optimizer.
            orbital_optimization: Perform orbital optimization.
            tol: Convergence tolerance.
            maxiter: Maximum number of iterations.
            is_silent: Toggle optimization print.
        """
        if not is_silent:
            print("### Parameters information:")
            if orbital_optimization:
                print(f"### Number kappa: {len(self.kappa_real)}")
            print(f"### Number theta: {self.ups_layout.n_params}")
        if optimizer_name.lower() == "rotosolve":
            if orbital_optimization and len(self.kappa_real) != 0:
                raise ValueError(
                    "Cannot use RotoSolve together with orbital optimization in the one-step solver."
                )

        print("--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #")
        if orbital_optimization:
            if len(self.thetas) > 0:
                energy = partial(
                    self._calc_energy_optimization,
                    theta_optimization=True,
                    kappa_optimization=True,
                )
                gradient = partial(
                    self._calc_gradient_optimization,
                    theta_optimization=True,
                    kappa_optimization=True,
                )
            else:
                energy = partial(
                    self._calc_energy_optimization,
                    theta_optimization=False,
                    kappa_optimization=True,
                )
                gradient = partial(
                    self._calc_gradient_optimization,
                    theta_optimization=False,
                    kappa_optimization=True,
                )
        else:
            energy = partial(
                self._calc_energy_optimization,
                theta_optimization=True,
                kappa_optimization=False,
            )
            gradient = partial(
                self._calc_gradient_optimization,
                theta_optimization=True,
                kappa_optimization=False,
            )
        if orbital_optimization:
            if len(self.thetas) > 0:
                thetas = []
                for theta_r, theta_i in zip(self.thetas_real, self.thetas_imag):
                    thetas.append(theta_r)
                    thetas.append(theta_i)
                parameters = np.zeros(2 * len(self.kappa_real)).tolist() + thetas
            else:
                parameters = np.zeros(2 * len(self.kappa_real)).tolist()
        else:
            thetas = []
            for theta_r, theta_i in zip(self.thetas_real, self.thetas_imag):
                thetas.append(theta_r)
                thetas.append(theta_i)
            parameters = thetas
        optimizer = Optimizers(
            energy,
            optimizer_name,
            grad=gradient,
            maxiter=maxiter,
            tol=tol,
            energy_eval_callback=lambda: self.num_energy_evals,
            is_silent=is_silent,
        )
        self._old_opt_parameters = np.zeros_like(parameters) + 10**20
        self._E_opt_old = 0.0
        res = optimizer.minimize(
            parameters,
            extra_options={"R": self.ups_layout.grad_param_R, "param_names": self.ups_layout.param_names},
        )
        print(res)
        if orbital_optimization:
            if len(self.thetas) > 0:
                thetas_r = []
                thetas_i = []
                for i in range(len(self.thetas)):
                    thetas_r.append(res.x[2 * i + 2 * len(self.kappa_real)])
                    thetas_i.append(res.x[2 * i + 1 + 2 * len(self.kappa_real)])
                self.set_thetas(thetas_r, thetas_i)
            for i in range(len(self.kappa_real)):
                self._kappa_real[i] = 0.0
                self._kappa_imag[i] = 0.0
                self._kappa_real_old[i] = 0.0
                self._kappa_imag_old[i] = 0.0
            else:
                kappa_r=[]
                kappa_i=[]
                for i in range(len(self.kappa_real)):
                    kappa_r.append(res.x[i])
                    kappa_i.append(res.x[i+len(self.kappa_real)])
                self.set_kappa_cep(kappa_r,kappa_i)
        else:
            thetas_r = []
            thetas_i = []
            for i in range(len(self.thetas)):
                thetas_r.append(res.x[2 * i])
                thetas_i.append(res.x[2 * i + 1])
            self.set_thetas(thetas_r, thetas_i)
        self._energy_elec = res.fun

    def do_adapt(
        self,
        operator_pool: list[str],
        maxiter: int = 1000,
        grad_threshold: float = 1e-5,
        orbital_optimization: bool = False,
    ) -> None:
        """Do ADAPT optimization.

        The valid operator pool is,

        - S, singles.
        - D, doubles.
        - GS, generalized singles.
        - GD, generalized doubles.

        Args:
            operator_pool: Which operators to include in the ADAPT.
            maxiter: Maximum iterations.
            grad_threshold: Convergence threshold based on gradient.
            orbital_optimization: Do orbital optimization.
        """
        excitation_pool: list[tuple[int, ...]] = []
        excitation_pool_type = []
        _operator_pool = [x.lower() for x in operator_pool]
        valid_operators = ("s", "d", "gs", "gd")
        for operator in _operator_pool:
            if operator not in valid_operators:
                raise ValueError(f"Got invalid operator for ADAPT, {operator}")
        if "s" in _operator_pool:
            for a, i in iterate_t1(
                self.active_occ_spin_idx_shifted, self.active_unocc_spin_idx_shifted, is_spin_conserving=False
            ):
                excitation_pool.append((i, a))
                excitation_pool_type.append("single")
        if "d" in _operator_pool:
            for a, i, b, j in iterate_t2(
                self.active_occ_spin_idx_shifted, self.active_unocc_spin_idx_shifted, is_spin_conserving=False
            ):
                excitation_pool.append((i, j, a, b))
                excitation_pool_type.append("double")
        if "gs" in _operator_pool:
            for a, i in iterate_t1_generalized(self.num_active_spin_orbs, is_spin_conserving=False):
                excitation_pool.append((i, a))
                excitation_pool_type.append("single")
        if "gd" in _operator_pool:
            for a, i, b, j in iterate_t2_generalized(self.num_active_spin_orbs, is_spin_conserving=False):
                excitation_pool.append((i, j, a, b))
                excitation_pool_type.append("double")

        print(
            "Iteration # | Iteration time [s] | Electronic energy [Hartree] | max|grad| [Hartree] | Operator"
        )
        start = time.time()
        for iteration in range(maxiter):
            Hamiltonian = generalized_hamiltonian_0i_0a(
                self.h_mo,
                self.g_mo,
                self.num_inactive_spin_orbs,
                self.num_active_spin_orbs,
            )
            H_ket = propagate_state(
                [Hamiltonian],
                self.ci_coeffs,
                self.ci_info,
            )
            grad = []

            for idx, exc_type in enumerate(excitation_pool_type):
                if exc_type == "single":
                    (i, a) = np.array(excitation_pool[idx])
                    T = G1(i, a, True)
                elif exc_type == "double":
                    (i, j, a, b) = np.array(excitation_pool[idx])
                    T = G2(i, j, a, b, True)
                else:
                    raise ValueError(f"Got unknown excitation type {exc_type}")
                gr = expectation_value(self.ci_coeffs, [T], H_ket, self.ci_info, do_folding=False)
                gr -= expectation_value(H_ket, [T], self.ci_coeffs, self.ci_info, do_folding=False)
                grad.append(gr)
            if np.max(np.abs(grad)) < grad_threshold:
                break
            max_arg = np.argmax(np.abs(grad))
            self.ups_layout.excitation_indices.append(excitation_pool[max_arg])
            self.ups_layout.excitation_operator_type.append(excitation_pool_type[max_arg])
            self.ups_layout.n_params += 1

            self._thetas_real.append(0.0)
            self._thetas_imag.append(0.0)
            self.run_wf_optimization_1step("bfgs", orbital_optimization=orbital_optimization, is_silent=True)
            time_str = f"{time.time() - start:7.2f}"
            e_str = f"{self.energy_elec:3.12f}"
            grad_str = f"{np.abs(grad[max_arg]):3.12f}"
            print(
                f"{str(iteration + 1).center(11)} | {time_str.center(18)} | {e_str.center(27)} | {grad_str.center(19)} | {excitation_pool_type[max_arg]}{tuple([int(x) for x in excitation_pool[max_arg]])}"
            )

    def _calc_energy_optimization(
        self, parameters: list[float], theta_optimization: bool, kappa_optimization: bool
    ) -> float:
        """Calculate electronic energy.

        Args:
            parameters: Ansatz and orbital rotation parameters.
            theta_optimization: If used in theta optimization.
            kappa_optimization: If used in kappa optimization.

        Returns:
            Electronic energy.
        """
        # Avoid recalculating energy in callback
        if np.max(np.abs(np.array(self._old_opt_parameters) - np.array(parameters))) < 10**-14:
            return self._E_opt_old
        num_kappa = 0
        if kappa_optimization:
            num_kappa = 2 * len(self.kappa_spin_idx)
            kappa_r = []
            kappa_i = []
            for i in range(len(self.kappa_real)):
                kappa_r.append(parameters[i])
                kappa_i.append(parameters[i + len(self.kappa_real)])
            self.set_kappa_cep(kappa_r, kappa_i)
        if theta_optimization:
            thetas_r = []
            thetas_i = []
            for i in range(len(self.thetas)):
                thetas_r.append(parameters[2 * i + num_kappa])
                thetas_i.append(parameters[2 * i + 1 + num_kappa])
            self.set_thetas(thetas_r, thetas_i)
        if kappa_optimization:
            # RDM is more expensive than evaluation of the Hamiltonian.
            # Thus only construct these if orbital-optimization is turned on,
            # since the RDMs will be reused in the oo gradient calculation.
            E = get_electronic_energy_generalized(
                self.h_mo,
                self.g_mo,
                self.num_inactive_spin_orbs,
                self.num_active_spin_orbs,
                self.rdm1,
                self.rdm2,
            )
        else:
            E = expectation_value(
                self.ci_coeffs,
                # [generalized_hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_spin_orbs, self.num_active_spin_orbs)],
                [generalized_hamiltonian_full_space(self.h_mo, self.g_mo, self.num_spin_orbs)],
                self.ci_coeffs,
                self.ci_info,
            )
        self._E_opt_old = E
        self._old_opt_parameters = np.copy(parameters)
        self.num_energy_evals += 1  # count one measurement
        return E

    def _calc_gradient_optimization(
        self, parameters: list[float], theta_optimization: bool, kappa_optimization: bool
    ) -> np.ndarray:
        """Calculate electronic gradient.

        Args:
            parameters: Ansatz and orbital rotation parameters.
            theta_optimization: If used in theta optimization.
            kappa_optimization: If used in kappa optimization.

        Returns:
            Electronic gradient.
        """
        gradient = np.zeros(len(parameters))
        num_kappa = 0
        if kappa_optimization:
            num_kappa = len(self.kappa_spin_idx)
            kappa_r = []
            kappa_i = []
            for i in range(len(self.kappa_real)):
                kappa_r.append(parameters[i])
                kappa_i.append(parameters[i + len(self.kappa_real)])
            self.set_kappa_cep(kappa_r, kappa_i)
        if theta_optimization:
            thetas_r = []
            thetas_i = []
            for i in range(len(self.thetas)):
                thetas_r.append(parameters[2 * i + num_kappa])
                thetas_i.append(parameters[2 * i + 1 + num_kappa])
            self.set_thetas(thetas_r, thetas_i)
        if kappa_optimization:
            gradient[:num_kappa] = get_orbital_gradient_generalized_real_imag(self.h_mo, self.g_mo, 
            self.kappa_spin_idx,
            self.num_inactive_spin_orbs,
            self.num_active_spin_orbs,
            self.rdm1,
            self.rdm2
            )
        if theta_optimization:
            # Hamiltonian = generalized_hamiltonian_0i_0a(
            #    self.h_mo,
            #    self.g_mo,
            #    self.num_inactive_spin_orbs,
            #    self.num_active_spin_orbs,
            # )
            Hamiltonian = generalized_hamiltonian_full_space(
                self.h_mo,
                self.g_mo,
                self.num_spin_orbs,
            )
            # Reference bra state (no differentiations)
            bra_vec = propagate_state(
                [Hamiltonian],
                self.ci_coeffs,
                self.ci_info,
            )
            bra_vec = construct_ups_state(
                bra_vec,
                self.ci_info,
                self.thetas,
                self.ups_layout,
                dagger=True,
            )
            # CSF reference state on ket
            ket_vec = np.copy(self.csf_coeffs)
            ket_vec_tmp = np.copy(self.csf_coeffs)
            # Calculate analytical derivative w.r.t. each theta using gradient_action function
            for i in range(len(self.thetas)):
                # Derivative action w.r.t. i-th theta on CSF ket
                ket_vec_tmp = get_grad_action(
                    ket_vec,
                    i,
                    self.ci_info,
                    self.ups_layout,
                )
                gradient[i + num_kappa] += 2 * np.matmul(bra_vec, ket_vec_tmp)
                # Product rule implications on reference bra and CSF ket
                # See 10.48550/arXiv.2303.10825, Eq. 20 (appendix - v1)
                bra_vec = propagate_unitary(
                    bra_vec,
                    i,
                    self.ci_info,
                    self.thetas,
                    self.ups_layout,
                )
                ket_vec = propagate_unitary(
                    ket_vec,
                    i,
                    self.ci_info,
                    self.thetas,
                    self.ups_layout,
                )
            self.num_energy_evals += 2 * np.sum(
                list(self.ups_layout.grad_param_R.values())
            )  # Count energy measurements for all gradients
        return gradient

    @property
    def get_orbital_gradient_generalized_test(self):
        return get_orbital_gradient_generalized(
            self.h_mo,
            self.g_mo,
            self.kappa_spin_idx,
            self.num_inactive_spin_orbs,
            self.num_active_spin_orbs,
            self.rdm1,
            self.rdm2,
        )

    @property
    def get_orbital_gradient_generalized_real_imag(self):
        return get_orbital_gradient_generalized_real_imag(self.h_mo, self.g_mo, self.kappa_spin_idx,
        self.num_inactive_spin_orbs,
        self.num_active_spin_orbs,
        self.rdm1,
        self.rdm2)
        
    @property
    def get_orbital_gradient_generalized_anna(self):
        return get_orbital_gradient_test_anna(self.h_mo, self.g_mo, self.kappa_spin_idx,
        self.num_inactive_spin_orbs,
        self.num_active_spin_orbs,
        self.rdm1,
        self.rdm2)
