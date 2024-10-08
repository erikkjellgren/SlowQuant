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
    Epq_matrix,
    build_operator_matrix,
    construct_ups_state,
    expectation_value_mat,
    get_grad_action,
    get_indexing,
    propagate_unitary,
)
from slowquant.unitary_coupled_cluster.operators import (
    hamiltonian_0i_0a,
    one_elec_op_0i_0a,
)
from slowquant.unitary_coupled_cluster.util import UpsStructure


class WaveFunctionSAUPS:
    def __init__(
        self,
        num_spin_orbs: int,
        num_elec: int,
        cas: Sequence[int],
        c_orthonormal: np.ndarray,
        h_ao: np.ndarray,
        g_ao: np.ndarray,
        states: tuple[list[list[float]], list[list[str]]],
        ansatz: str,
        ansatz_options: dict[str, Any] | None = None,
        include_active_kappa: bool = False,
    ) -> None:
        """Initialize for SA-UPS wave function.

        Args:
            num_spin_orbs: Number of spin orbitals.
            num_elec: Number of electrons.
            cas: CAS(num_active_elec, num_active_orbs),
                 orbitals are counted in spatial basis.
            c_orthonormal: Initial orbital coefficients.
            h_ao: One-electron integrals in AO for Hamiltonian.
            g_ao: Two-electron integrals in AO.
            states: States to include in the state-averaged expansion.
                    Tuple of lists containing weights and determinants.
                    Each state in SA can be constructed of several dets.
            ansatz: Name of ansatz.
            ansatz_options: Ansatz options.
            include_active_kappa: Include active-active orbital rotations.
        """
        if ansatz_options is None:
            ansatz_options = {}
        if len(cas) != 2:
            raise ValueError(f"cas must have two elements, got {len(cas)} elements.")
        # Init stuff
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
        self.ansatz_options = ansatz_options
        # Construct spin orbital spaces and indices
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
        # Make shifted indices
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
        self.kappa_idx_dagger = []
        self.kappa_redundant = []
        self.kappa_redundant_idx = []
        self._kappa_old = []
        self._kappa_redundant_old = []
        # kappa can be optimized in spatial basis
        # Loop over all q>p orb combinations and find redundant kappas
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                # find redundant kappas
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
                # the rest is non-redundant
                self.kappa.append(0.0)
                self._kappa_old.append(0.0)
                self.kappa_idx.append([p, q])
                self.kappa_idx_dagger.append([q, p])
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
        # SA details
        self.num_states = len(states[0])
        self.csf_coeffs = np.zeros((self.num_states, self.num_det))  # state vector for each state in SA
        # Loop over all states in SA procedure
        for i, (coeffs, on_vecs) in enumerate(zip(states[0], states[1])):
            if len(coeffs) != len(on_vecs):
                raise ValueError(
                    f"Mismatch in number of coefficients, {len(coeffs)}, and number of determinants, {len(on_vecs)}. For {coeffs} and {on_vecs}"
                )
            # Loop over all determinants of a given state
            for coeff, on_vec in zip(coeffs, on_vecs):
                if len(on_vec) != self.num_active_spin_orbs:
                    raise ValueError(
                        f"Length of determinant, {len(on_vec)}, does not match number of active spin orbitals, {self.num_active_spin_orbs}. For determinant, {on_vec}"
                    )
                idx = self.det2idx[int(on_vec, 2)]
                self.csf_coeffs[i, idx] = coeff
        self._ci_coeffs = np.copy(self.csf_coeffs)
        for i, coeff_i in enumerate(self.ci_coeffs):
            for j, coeff_j in enumerate(self.ci_coeffs):
                if i == j:
                    if abs(1 - coeff_i @ coeff_j) > 10**-10:
                        raise ValueError(f"state {i} is not normalized got overlap of {coeff_i @ coeff_j}")
                else:
                    if abs(coeff_i @ coeff_j) > 10**-10:
                        raise ValueError(
                            f"state {i} and {j} are not otrhogonal got overlap of {coeff_i @ coeff_j}"
                        )
        # Construct UPS Structure
        self.ups_layout = UpsStructure()
        if ansatz.lower() == "tups":
            self.ups_layout.create_tups(self.num_active_orbs, self.ansatz_options)
        elif ansatz.lower() == "qnp":
            self.ansatz_options["do_qnp"] = True
            self.ups_layout.create_tups(self.num_active_orbs, self.ansatz_options)
        elif ansatz.lower() == "ksafupccgsd":
            self.ansatz_options["SAGS"] = True
            self.ansatz_options["GpD"] = True
            self.ups_layout.create_fUCC(self.num_active_orbs, self.num_active_elec, self.ansatz_options)
        elif ansatz.lower() == "ksasdsfupccgsd":
            self.ansatz_options["GpD"] = True
            self.ups_layout.create_SDSfUCC(self.num_active_orbs, self.num_active_elec, self.ansatz_options)
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
        self._state_energies = None
        self._state_ci_coeffs = None
        self._c_orthonormal = c

    @property
    def ci_coeffs(self) -> list[np.ndarray]:
        """Get CI coefficients.

        Returns:
            State vector.
        """
        if self._ci_coeffs is None:
            tmp = []
            for coeffs in self.csf_coeffs:
                tmp.append(
                    construct_ups_state(
                        coeffs,
                        self.num_active_orbs,
                        self.num_active_elec_alpha,
                        self.num_active_elec_beta,
                        self.thetas,
                        self.ups_layout,
                    )
                )
            self._ci_coeffs = np.array(tmp)
        return self._ci_coeffs  # type: ignore[return-value]

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
        self._state_energies = None
        self._state_ci_coeffs = None
        self._ci_coeffs = None
        self._thetas = theta_vals.copy()

    @property
    def c_trans(self) -> np.ndarray:
        """Get orbital coefficients.

        Returns:
            Orbital coefficients.
        """
        # Construct anti-hermitian kappa matrix
        kappa_mat = np.zeros_like(self._c_orthonormal)
        if len(self.kappa) != 0:
            if np.max(np.abs(self.kappa)) > 0.0:
                for kappa_val, (p, q) in zip(self.kappa, self.kappa_idx):
                    kappa_mat[p, q] = kappa_val
                    kappa_mat[q, p] = -kappa_val
        # Legacy redundant kappa scans
        if len(self.kappa_redundant) != 0:
            if np.max(np.abs(self.kappa_redundant)) > 0.0:
                for kappa_val, (p, q) in zip(self.kappa_redundant, self.kappa_redundant_idx):
                    kappa_mat[p, q] = kappa_val
                    kappa_mat[q, p] = -kappa_val
        # Apply orbital rotation unitary to MO coefficients
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
        """Calculate one-electron reduced density matrix in the active space.

        Returns:
            One-electron reduced density matrix.
        """
        if self._rdm1 is None:
            self._rdm1 = np.zeros((self.num_active_orbs, self.num_active_orbs))
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_idx = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_idx = q - self.num_inactive_orbs
                    val = 0.0
                    Epq_mat = Epq_matrix(
                        p_idx,
                        q_idx,
                        self.num_active_orbs,
                        self.num_active_elec_alpha,
                        self.num_active_elec_beta,
                    ).todense()
                    # Loop over each state in SA
                    for coeffs in self.ci_coeffs:
                        val += expectation_value_mat(
                            coeffs,
                            Epq_mat,
                            coeffs,
                        )
                    val = val / len(self.ci_coeffs)
                    self._rdm1[p_idx, q_idx] = val  # type: ignore
                    self._rdm1[q_idx, p_idx] = val  # type: ignore
        return self._rdm1

    @property
    def rdm2(self) -> np.ndarray:
        """Calculate two-electron reduced density matrix in the active space.

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
            bra_pq = np.zeros_like(self.ci_coeffs)
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_idx = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_idx = q - self.num_inactive_orbs
                    Epq_mat = Epq_matrix(
                        p_idx,
                        q_idx,
                        self.num_active_orbs,
                        self.num_active_elec_alpha,
                        self.num_active_elec_beta,
                    ).todense()
                    # Loop over each state in SA
                    for i, coeff in enumerate(self.ci_coeffs):
                        bra_pq[i] = np.matmul(coeff, Epq_mat)
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
                            Ers_mat = Epq_matrix(
                                r_idx,
                                s_idx,
                                self.num_active_orbs,
                                self.num_active_elec_alpha,
                                self.num_active_elec_beta,
                            ).todense()
                            val = 0.0
                            # Loop over each state in SA
                            for i, coeffs in enumerate(self.ci_coeffs):
                                val += expectation_value_mat(
                                    bra_pq[i],
                                    Ers_mat,
                                    coeffs,
                                )
                            val = val / len(self.ci_coeffs)
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

    def run_saups(
        self,
        orbital_optimization: bool = False,
        is_silent: bool = False,
        convergence_threshold: float = 10**-10,
        maxiter: int = 10000,
    ) -> None:
        """Run optimization of SA-UPS wave function.

        Args:
            orbital_optimization: Do orbital optimization.
            is_silent: Do not print any output.
            convergence_threshold: Energy threshold for convergence.
            maxiter: Maximum number of iterations.
        """
        # Define energy and gradient (partial) functions with parameters as free argument
        e_tot = partial(
            energy_saups,
            orbital_optimized=orbital_optimization,
            wf=self,
        )
        parameter_gradient = partial(
            gradient_saups,
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

        # Init parameters
        parameters: list[float] = []
        num_kappa = 0
        num_theta = 0
        if orbital_optimization:
            parameters += self.kappa
            num_kappa += len(self.kappa)
        parameters = parameters + self.thetas
        num_theta = len(self.thetas)
        # Optimization
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
        # Set kappas to zero (orbitals have been optimized)
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
        # Subspace diagonalization
        self._do_state_ci()

    def _do_state_ci(self) -> None:
        r"""Do subspace diagonalisation.

        #. 10.1103/PhysRevLett.122.230401, Eq. 2
        """
        state_H = np.zeros((self.num_states, self.num_states))
        # Hamiltonian matrix
        Hamiltonian = build_operator_matrix(
            hamiltonian_0i_0a(
                self.h_mo,
                self.g_mo,
                self.num_inactive_orbs,
                self.num_active_orbs,
            ).get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs),
            self.idx2det,
            self.det2idx,
            self.num_active_orbs,
        )
        # Create SA H matrix
        for i, coeff_i in enumerate(self.ci_coeffs):
            for j, coeff_j in enumerate(self.ci_coeffs):
                if j > i:
                    continue
                state_H[i, j] = state_H[j, i] = expectation_value_mat(coeff_i, Hamiltonian, coeff_j)
        # Diagonalize
        eigval, eigvec = scipy.linalg.eig(state_H)
        sorting = np.argsort(eigval)
        self._state_energies = np.real(eigval[sorting])
        self._state_ci_coeffs = np.real(eigvec[:, sorting])

    @property
    def energy_states(self) -> np.ndarray:
        """Get state specific energies.

        Returns:
            State specific energies.
        """
        if self._state_energies is None:
            self._do_state_ci()
        return self._state_energies

    @property
    def excitation_energies(self) -> np.ndarray:
        r"""Get excitation energies.

        .. math::
            \varepsilon_n = E_n - E_0

        Returns:
            Excitation energies.
        """
        energies = np.zeros(self.num_states - 1)
        for i, energy in enumerate(self.energy_states[1:]):
            energies[i] = energy - self.energy_states[0]
        return energies

    def get_transition_property(self, ao_integral: np.ndarray) -> np.ndarray:
        r"""Get transition property with one-electron operator.

        .. math::
            t_n = \left<0\left|\hat{O}\right|n\right>

        Args:
            ao_integral: Operator integrals in AO basis.

        Returns:
            Transition property.
        """
        if self._state_ci_coeffs is None:
            self._do_state_ci()
        if self._state_ci_coeffs is None:
            raise ValueError("_state_ci_coeffs is None")
        # MO integrals
        mo_integral = one_electron_integral_transform(self.c_trans, ao_integral)
        transition_property = np.zeros(self.num_states - 1)
        state_op = np.zeros((self.num_states, self.num_states))
        # One-electron operator matrix
        op = build_operator_matrix(
            one_elec_op_0i_0a(mo_integral, self.num_inactive_orbs, self.num_active_orbs).get_folded_operator(
                self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs
            ),
            self.idx2det,
            self.det2idx,
            self.num_active_orbs,
        )
        for i, coeff_i in enumerate(self.ci_coeffs):
            for j, coeff_j in enumerate(self.ci_coeffs):
                state_op[i, j] = expectation_value_mat(coeff_i, op, coeff_j)
        # Transition between SA states (after diagonalization)
        for i in range(self.num_states - 1):
            transition_property[i] = self._state_ci_coeffs[:, i + 1] @ state_op @ self._state_ci_coeffs[:, 0]
        return transition_property

    def get_oscillator_strenghts(self, dipole_integrals: Sequence[np.ndarray]) -> np.ndarray:
        r"""Get oscillator strengths between ground state and excited states.

        .. math::
            f_n = \frac{2}{3}\varepsilon_n\left|\left<0\left|\hat{\boldsymbol{\mu}}\right|n\right>\right|^2

        Args:
            dipole_integrals: Dipole integrals in AO basis.

        Returns:
            Oscillator strengths.
        """
        transition_dipole_x = self.get_transition_property(dipole_integrals[0])
        transition_dipole_y = self.get_transition_property(dipole_integrals[1])
        transition_dipole_z = self.get_transition_property(dipole_integrals[2])
        osc_strs = np.zeros(self.num_states - 1)
        for idx, (excitation_energy, td_x, td_y, td_z) in enumerate(
            zip(self.excitation_energies, transition_dipole_x, transition_dipole_y, transition_dipole_z)
        ):
            osc_strs[idx] = 2 / 3 * excitation_energy * (td_x**2 + td_y**2 + td_z**2)
        return osc_strs


def energy_saups(
    parameters: Sequence[float],
    orbital_optimized: bool,
    wf: WaveFunctionSAUPS,
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
    # Get kappa and theta parameters separately
    kappa = []
    idx_counter = 0
    if orbital_optimized:
        idx_counter = len(wf.kappa_idx)
        kappa = list(parameters[:idx_counter])
    theta = list(parameters[idx_counter:])
    assert len(parameters) == len(kappa) + len(theta)

    kappa_mat = np.zeros_like(wf.c_orthonormal)
    if orbital_optimized:
        # Build kappa matrix
        for kappa_val, (p, q) in zip(
            np.array(kappa) - np.array(wf._kappa_old), wf.kappa_idx  # pylint: disable=protected-access
        ):
            kappa_mat[p, q] = kappa_val
            kappa_mat[q, p] = -kappa_val
    # Legacy redundant kappa scans
    if len(wf.kappa_redundant) != 0:
        if np.max(np.abs(wf.kappa_redundant)) > 0.0:
            for kappa_val, (p, q) in zip(
                np.array(wf.kappa_redundant)
                - np.array(wf._kappa_redundant_old),  # pylint: disable=protected-access
                wf.kappa_redundant_idx,
            ):
                kappa_mat[p, q] = kappa_val
                kappa_mat[q, p] = -kappa_val
    # Apply orbital rotation unitary
    c_trans = np.matmul(wf.c_orthonormal, scipy.linalg.expm(-kappa_mat))
    if orbital_optimized:
        # Update kappas
        wf._kappa_old = kappa.copy()  # pylint: disable=protected-access
        wf._kappa_redundant_old = wf.kappa_redundant.copy()  # pylint: disable=protected-access
    # Moving expansion point of kappa
    wf.c_orthonormal = c_trans
    # Add thetas
    wf.thetas = theta
    # Hamiltonian matrix
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
    energy = 0.0
    # Energy for each state in SA
    for coeffs in wf.ci_coeffs:
        energy += expectation_value_mat(coeffs, Hamiltonian, coeffs)
    return energy / len(wf.ci_coeffs)


def gradient_saups(
    parameters: Sequence[float],
    orbital_optimized: bool,
    wf: WaveFunctionSAUPS,
) -> np.ndarray:
    """Calculate electronic gradient.

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
    wf: WaveFunctionSAUPS,
) -> np.ndarray:
    """Calculate electronic gradient with respect to orbital rotations using RDMs.

    Args:
        wf: Wave function object.

    Return:
        Electronic gradient with respect to orbital rotations.
    """
    # Analytical gradient via RDMs
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
    wf: WaveFunctionSAUPS,
) -> np.ndarray:
    r"""Calculate electronic gradient with respect to active space parameters.

    #. 10.48550/arXiv.2303.10825, Eq. 17-21 (appendix - v1)

    Args:
        wf: Wave function object.

    Returns:
        Electronic gradient with respect to active space parameters.
    """
    # Hamiltonian matrix
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
    # Reference bra state (no differentiations)
    bra_vec = np.copy(wf.ci_coeffs)
    for i, coeffs in enumerate(bra_vec):
        bra_vec[i] = construct_ups_state(
            np.matmul(Hamiltonian, coeffs),
            wf.num_active_orbs,
            wf.num_active_elec_alpha,
            wf.num_active_elec_beta,
            wf.thetas,
            wf.ups_layout,
            dagger=True,
        )
    # CSF reference state on ket
    ket_vec = np.copy(wf.csf_coeffs)
    ket_vec_tmp = np.copy(wf.csf_coeffs)
    # Calculate analytical derivatice w.r.t. each theta using gradient_action function
    for i in range(len(wf.thetas)):
        # Loop over each state in SA
        for j in range(len(bra_vec)):
            ket_vec_tmp[j] = get_grad_action(
                ket_vec[j],
                i,
                wf.num_active_orbs,
                wf.num_active_elec_alpha,
                wf.num_active_elec_beta,
                wf.ups_layout,
            )
        for bra, ket in zip(bra_vec, ket_vec_tmp):
            gradient_theta[i] += 2 * np.matmul(bra, ket)
        # Product rule implications on reference bra and CSF ket
        # See 10.48550/arXiv.2303.10825, Eq. 20 (appendix - v1)
        for j in range(len(bra_vec)):  # pylint: disable=consider-using-enumerate
            bra_vec[j] = propagate_unitary(
                bra_vec[j],
                i,
                wf.num_active_orbs,
                wf.num_active_elec_alpha,
                wf.num_active_elec_beta,
                wf.thetas,
                wf.ups_layout,
            )
            ket_vec[j] = propagate_unitary(
                ket_vec[j],
                i,
                wf.num_active_orbs,
                wf.num_active_elec_alpha,
                wf.num_active_elec_beta,
                wf.thetas,
                wf.ups_layout,
            )
    return gradient_theta / len(bra_vec)
