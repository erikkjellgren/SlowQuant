# pylint: disable=too-many-lines
from __future__ import annotations

import time
from collections.abc import Sequence
from functools import partial

import numpy as np
import scipy
import scipy.optimize

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.density_matrix import (
    ReducedDenstiyMatrix,
    get_electronic_energy,
    get_orbital_gradient,
)
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operator_matrix import (
    Epq_matrix,
    build_operator_matrix,
    construct_ucc_state,
    expectation_value,
    expectation_value_mat,
    get_indexing,
    propagate_state,
)
from slowquant.unitary_coupled_cluster.operators import (
    Epq,
    G1_sa,
    G2_1_sa,
    G2_2_sa,
    hamiltonian_0i_0a,
)
from slowquant.unitary_coupled_cluster.util import (
    UccStructure,
    iterate_t1_sa,
    iterate_t2_sa,
)


class LagWaveFunctionUCC:
    def __init__(
        self,
        num_spin_orbs: int,
        num_elec: int,
        cas: Sequence[int],
        c_orthonormal: np.ndarray,
        h_ao: np.ndarray,
        g_ao: np.ndarray,
        excitations: str,
        include_active_kappa: bool = False,
    ) -> None:
        """Initialize for UCC wave function.

        Args:
            num_spin_orbs: Number of spin orbitals.
            num_elec: Number of electrons.
            cas: CAS(num_active_elec, num_active_orbs),
                 orbitals are counted in spatial basis.
            c_orthonormal: Initial orbital coefficients.
            h_ao: One-electron integrals in AO for Hamiltonian.
            g_ao: Two-electron integrals in AO.
            excitations: Unitary coupled cluster excitation operators.
            include_active_kappa: Include active-active orbital rotations.
        """
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
        self._rdm1_lag = None
        self._rdm2_lag = None
        self._h_mo = None
        self._g_mo = None
        self._excited_states = None
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
        self._kappa = []
        self.kappa_idx = []
        self._kappa_old = []
        # kappa can be optimized in spatial basis
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                self._kappa.append(0.0)
                self._kappa_old.append(0.0)
                self.kappa_idx.append([p, q])
        # Construct determinant basis
        self.idx2det, self.det2idx = get_indexing(
            self.num_active_orbs, self.num_active_elec_alpha, self.num_active_elec_beta
        )
        self.num_det = len(self.idx2det)
        self.csf_coeffs = np.zeros(self.num_det)
        hf_det = int("1" * self.num_active_elec + "0" * (self.num_active_spin_orbs - self.num_active_elec), 2)
        self.csf_coeffs[self.det2idx[hf_det]] = 1
        self._ci_coeffs = np.copy(self.csf_coeffs)
        # Construct UCC
        self.ucc_layout = UccStructure()
        if "s" in excitations.lower():
            self.ucc_layout.add_sa_singles(self.active_occ_idx_shifted, self.active_unocc_idx_shifted)
        if "d" in excitations.lower():
            self.ucc_layout.add_sa_doubles(self.active_occ_idx_shifted, self.active_unocc_idx_shifted)
        if "t" in excitations.lower():
            self.ucc_layout.add_triples(self.active_occ_spin_idx_shifted, self.active_unocc_spin_idx_shifted)
        if "q" in excitations.lower():
            self.ucc_layout.add_quadruples(
                self.active_occ_spin_idx_shifted, self.active_unocc_spin_idx_shifted
            )
        if "5" in excitations.lower():
            self.ucc_layout.add_quintuples(
                self.active_occ_spin_idx_shifted, self.active_unocc_spin_idx_shifted
            )
        if "6" in excitations.lower():
            self.ucc_layout.add_sextuples(
                self.active_occ_spin_idx_shifted, self.active_unocc_spin_idx_shifted
            )
        self._thetas = np.zeros(self.ucc_layout.n_params).tolist()

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
        self._c_orthonormal = c

    @property
    def kappa(self) -> list[float]:
        return self._kappa.copy()

    @kappa.setter
    def kappa(self, k: list[float]) -> None:
        self._h_mo = None
        self._g_mo = None
        self._energy_elec = None
        self._kappa = k.copy()

    @property
    def ci_coeffs(self) -> np.ndarray:
        """Get CI coefficients.

        Returns:
            State vector.
        """
        if self._ci_coeffs is None:
            self._ci_coeffs = construct_ucc_state(
                self.csf_coeffs,
                self.num_active_orbs,
                self.num_active_elec_alpha,
                self.num_active_elec_beta,
                self._thetas,
                self.ucc_layout,
            )
        return self._ci_coeffs

    @property
    def thetas(self) -> list[float]:
        """Get theta values.

        Returns:
            theta values.
        """
        return self._thetas.copy()

    @thetas.setter
    def thetas(self, theta: list[float]) -> None:
        """Set theta1 values.

        Args:
            theta: theta1 values.
        """
        if len(theta) != len(self._thetas):
            raise ValueError(f"Expected {len(self._thetas)} theta1 values got {len(theta)}")
        self._rdm1 = None
        self._rdm2 = None
        self._rdm1_lag = None
        self._rdm2_lag = None
        self._ci_coeffs = None
        self._excited_states = None
        self._thetas = theta.copy()

    @property
    def c_trans(self) -> np.ndarray:
        """Get orbital coefficients.

        Returns:
            Orbital coefficients.
        """
        kappa_mat = np.zeros_like(self._c_orthonormal)
        for kappa_val, kappa_val_old, (p, q) in zip(self.kappa, self._kappa_old, self.kappa_idx):
            kappa_mat[p, q] = kappa_val - kappa_val_old
            kappa_mat[q, p] = -(kappa_val - kappa_val_old)
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
                        self.num_virtual_orbs,
                        self.num_active_elec_alpha,
                        self.num_active_elec_beta,
                        self.thetas,
                        self.ucc_layout,
                    )
                    self._rdm1[p_idx, q_idx] = val  # type: ignore
                    self._rdm1[q_idx, p_idx] = val  # type: ignore
        return self._rdm1

    @property
    def rdm1_lag(self) -> np.ndarray:
        """Calcuate one-electron reduced density matrix.

        Returns:
            One-electron reduced density matrix.
        """
        if self._rdm1_lag is None:
            self._rdm1_lag = np.zeros((self.num_active_orbs, self.num_active_orbs))
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
                    for state in self.excited_states:
                        val += expectation_value_mat(
                            state,
                            Epq_mat,
                            state,
                        )
                    self._rdm1_lag[p_idx, q_idx] = val  # type: ignore
                    self._rdm1_lag[q_idx, p_idx] = val  # type: ignore
        return self._rdm1_lag

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
                                [Epq(p, q), Epq(r, s)],
                                self.ci_coeffs,
                                self.idx2det,
                                self.det2idx,
                                self.num_inactive_orbs,
                                self.num_active_orbs,
                                self.num_virtual_orbs,
                                self.num_active_elec_alpha,
                                self.num_active_elec_beta,
                                self.thetas,
                                self.ucc_layout,
                            )
                            if q == r:
                                val -= self.rdm1[p_idx, s_idx]
                            self._rdm2[p_idx, q_idx, r_idx, s_idx] = val  # type: ignore
                            self._rdm2[r_idx, s_idx, p_idx, q_idx] = val  # type: ignore
                            self._rdm2[q_idx, p_idx, s_idx, r_idx] = val  # type: ignore
                            self._rdm2[s_idx, r_idx, q_idx, p_idx] = val  # type: ignore
        return self._rdm2

    @property
    def rdm2_lag(self) -> np.ndarray:
        """Calcuate two-electron reduced density matrix.

        Returns:
            Two-electron reduced density matrix.
        """
        if self._rdm2_lag is None:
            self._rdm2_lag = np.zeros(
                (
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                )
            )
            bra_pq = np.zeros_like(self.excited_states)
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
                    for i, state in enumerate(self.excited_states):
                        bra_pq[i] = np.matmul(state, Epq_mat)
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
                            for i, state in enumerate(self.excited_states):
                                val += expectation_value_mat(
                                    bra_pq[i],
                                    Ers_mat,
                                    state,
                                )
                            if q == r:
                                val -= self.rdm1_lag[p_idx, s_idx]
                            self._rdm2_lag[p_idx, q_idx, r_idx, s_idx] = val  # type: ignore
                            self._rdm2_lag[r_idx, s_idx, p_idx, q_idx] = val  # type: ignore
                            self._rdm2_lag[q_idx, p_idx, s_idx, r_idx] = val  # type: ignore
                            self._rdm2_lag[s_idx, r_idx, q_idx, p_idx] = val  # type: ignore
        return self._rdm2_lag

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
    def excited_states(self) -> list[np.ndarray]:
        if self._excited_states is None:
            G_ops: list[FermionicOperator] = []
            if "s" in self.lag_excitations:
                for a, i, _ in iterate_t1_sa(self.active_occ_idx, self.active_unocc_idx):
                    G_ops.append(G1_sa(i, a))
            if "d" in self.lag_excitations:
                for a, i, b, j, _, op_type in iterate_t2_sa(self.active_occ_idx, self.active_unocc_idx):
                    if op_type == 1:
                        G_ops.append(G2_1_sa(i, j, a, b))
                    elif op_type == 2:
                        G_ops.append(G2_2_sa(i, j, a, b))
            self._excited_states = []
            for i, G in enumerate(G_ops):
                if i not in (4, 5, 6):
                    continue
                self._excited_states.append(
                    propagate_state(
                        ["U", G],
                        self.csf_coeffs,
                        self.idx2det,
                        self.det2idx,
                        self.num_inactive_orbs,
                        self.num_active_orbs,
                        self.num_inactive_orbs,
                        self.num_active_elec_alpha,
                        self.num_active_elec_beta,
                        self.thetas,
                        self.ucc_layout,
                    )
                )
        return self._excited_states

    def _move_cep(self) -> None:
        """Move current expansion point."""
        c = self.c_trans
        self._c_orthonormal = c
        self._kappa_old = self.kappa

    def run_lagucc(
        self,
        E0: float,
        excitations: str,
        convergence_threshold: float = 10**-10,
        maxiter: int = 10000,
        save_orbs: None | str = None,
    ) -> None:
        """Run optimization of UCC wave function.

        Args:
            convergence_threshold: Energy threshold for convergence.
            maxiter: Maximum number of iterations.
        """
        self.lag_excitations = excitations.lower()

        global iteration  # pylint: disable=global-variable-undefined
        global start  # pylint: disable=global-variable-undefined
        iteration = 0  # type: ignore
        start = time.time()  # type: ignore

        def print_progress(x: list[float]) -> None:
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

        parameters: list[float] = []
        num_kappa = 0
        num_theta1 = 0
        num_theta2 = 0
        num_theta3 = 0
        num_theta4 = 0
        num_theta5 = 0
        num_theta6 = 0
        parameters += self.kappa
        num_kappa += len(self.kappa)
        for theta in self.thetas:
            parameters.append(theta)
        for exc_type in self.ucc_layout.excitation_operator_type:
            if exc_type == "sa_single":
                num_theta1 += 1
            elif exc_type in ("sa_double_1", "sa_double_2"):
                num_theta2 += 1
            elif exc_type == "triple":
                num_theta3 += 1
            elif exc_type == "quadruple":
                num_theta4 += 1
            elif exc_type == "quintuple":
                num_theta5 += 1
            elif exc_type == "sextuple":
                num_theta6 += 1
            else:
                raise ValueError(f"Got unknown excitation type, {exc_type}")
        print("### Parameters information:")
        print(f"### Number kappa: {num_kappa}")
        print(f"### Number theta1: {num_theta1}")
        print(f"### Number theta2: {num_theta2}")
        print(f"### Number theta3: {num_theta3}")
        print(f"### Number theta4: {num_theta4}")
        print(f"### Number theta5: {num_theta5}")
        print(f"### Number theta6: {num_theta6}")
        print(
            f"### Total parameters: {num_kappa + num_theta1 + num_theta2 + num_theta3 + num_theta4 + num_theta5 + num_theta6}\n"
        )
        print("Iteration # | Iteration time [s] | Electronic energy [Hartree]")
        lag_val = 10**6
        k = 10**12
        for _ in range(1000):
            e_tot = partial(
                energy_lagucc,
                wf=self,
                E0=E0,
                k=k,
            )
            parameter_gradient = partial(
                gradient_lagucc,
                wf=self,
                E0=E0,
                k=k,
            )
            parameters = []
            parameters += self.kappa
            for theta in self.thetas:
                parameters.append(theta)
            res = scipy.optimize.minimize(
                e_tot,
                parameters,
                tol=convergence_threshold,
                callback=print_progress,
                method="BFGS",
                jac=parameter_gradient,
                options={"maxiter": 2000},
            )
            param_idx = len(self.kappa)
            for i in range(len(self.kappa)):  # pylint: disable=consider-using-enumerate
                self.kappa[i] = 0
                self._kappa_old[i] = 0
            self.thetas = res["x"][param_idx:].tolist()
            H = hamiltonian_0i_0a(
                self.h_mo,
                self.g_mo,
                self.num_inactive_orbs,
                self.num_active_orbs,
            )
            E_elec = expectation_value(
                self.ci_coeffs,
                [H],
                self.ci_coeffs,
                self.idx2det,
                self.det2idx,
                self.num_inactive_orbs,
                self.num_active_orbs,
                self.num_inactive_orbs,
                self.num_active_elec_alpha,
                self.num_active_elec_beta,
                self.thetas,
                self.ucc_layout,
            )
            print("Energy elec", E_elec)
            print("Constraint error", abs(E_elec - E0))
            param_idx = len(self.kappa)
            self.kappa = np.zeros_like(self.kappa).tolist()
            self._kappa_old = np.zeros_like(self.kappa).tolist()
            self.thetas = res["x"][param_idx:].tolist()
            if save_orbs is not None:
                np.save(f"{save_orbs}_{iteration}", self.c_trans)
                print("thetas:", self.thetas)
            if res.nfev < 2000:
                break


def energy_lagucc(
    parameters: list[float],
    wf: LagWaveFunctionUCC,
    E0: float,
    k: float,
) -> float:
    r"""Calculate electronic energy of UCC wave function.

    .. math::
        E = \left<0\left|\hat{H}\right|0\right>

    Args:
        parameters: Sequence of all parameters.
                    Ordered as orbital rotations, active-space singles, active-space doubles, ...
        orbital_optimized: Do orbital optimization.
        wf: Wave function object.

    Returns:
        Electronic energy.
    """
    number_kappas = 0
    number_kappas = len(wf.kappa_idx)
    wf.kappa = parameters[:number_kappas]
    wf.thetas = parameters[number_kappas:]
    wf._move_cep()
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
    E_elec = expectation_value_mat(
        wf.ci_coeffs,
        Hamiltonian,
        wf.ci_coeffs,
    )
    E_lag = 0.0
    for state in wf.excited_states:
        E_lag += expectation_value_mat(
            state,
            Hamiltonian,
            state,
        )
    return E_lag + k * (E_elec - E0) ** 2


def gradient_lagucc(
    parameters: list[float],
    wf: LagWaveFunctionUCC,
    E0: float,
    k: float,
) -> np.ndarray:
    """Calcuate electronic gradient.

    Args:
        parameters: Sequence of all parameters.
                    Ordered as orbital rotations, active-space singles, active-space doubles, ...
        orbital_optimized: Do orbital optimization.
        wf: Wave function object.

    Returns:
        Electronic gradient.
    """
    number_kappas = 0
    number_kappas = len(wf.kappa_idx)
    wf.kappa = parameters[:number_kappas]
    gradient = np.zeros_like(parameters)
    wf.thetas = parameters[number_kappas:]
    wf._move_cep()
    gradient[:number_kappas] = orbital_rotation_gradient(
        wf,
        E0,
        k,
    )
    gradient[number_kappas:] = active_space_parameter_gradient(
        wf,
        parameters,
        E0,
        k,
    )
    return gradient


def orbital_rotation_gradient(
    wf: LagWaveFunctionUCC,
    E0: float,
    k: float,
) -> np.ndarray:
    """Calcuate electronic gradient with respect to orbital rotations.

    Args:
        wf: Wave function object.

    Return:
        Electronic gradient with respect to orbital rotations.
    """
    rdms_elec = ReducedDenstiyMatrix(
        wf.num_inactive_orbs,
        wf.num_active_orbs,
        wf.num_active_orbs,
        rdm1=wf.rdm1,
        rdm2=wf.rdm2,
    )
    rdms_lag = ReducedDenstiyMatrix(
        wf.num_inactive_orbs,
        wf.num_active_orbs,
        wf.num_active_orbs,
        rdm1=wf.rdm1_lag,
        rdm2=wf.rdm2_lag,
    )
    dg = get_orbital_gradient(
        rdms_elec, wf.h_mo, wf.g_mo, wf.kappa_idx, wf.num_inactive_orbs, wf.num_active_orbs
    )
    df = get_orbital_gradient(
        rdms_lag, wf.h_mo, wf.g_mo, wf.kappa_idx, wf.num_inactive_orbs, wf.num_active_orbs
    )
    E_elec = get_electronic_energy(rdms_elec, wf.h_mo, wf.g_mo, wf.num_inactive_orbs, wf.num_active_orbs)
    return df + 2 * k * dg * (E_elec - E0)


def active_space_parameter_gradient(
    wf: LagWaveFunctionUCC,
    parameters: list[float],
    E0: float,
    k: float,
) -> np.ndarray:
    """Calcuate electronic gradient with respect to active space parameters.

    Args:
        wf: Wave function object.
        parameters: Sequence of all parameters.
                    Ordered as orbital rotations, active-space singles, active-space doubles, ...
        orbital_optimized: Do orbital optimization.

    Returns:
        Electronic gradient with respect to active spae parameters.
    """
    idx_counter = 0
    for _ in range(len(wf.kappa_idx)):
        idx_counter += 1
    theta_params = parameters[idx_counter:]

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
    gradient_theta = np.zeros_like(theta_params)
    eps = np.finfo(np.float64).eps ** (1 / 2)
    E_elec = expectation_value_mat(wf.ci_coeffs, Hamiltonian, wf.ci_coeffs)
    E_lag = 0.0
    for state in wf.excited_states:
        E_lag += expectation_value_mat(
            state,
            Hamiltonian,
            state,
        )
    for i in range(len(theta_params)):  # pylint: disable=consider-using-enumerate
        sign_step = (theta_params[i] >= 0).astype(float) * 2 - 1  # type: ignore [attr-defined]
        step_size = eps * sign_step * max(1, abs(theta_params[i]))
        theta_params[i] += step_size
        wf.thetas = theta_params
        E_elec_plus = expectation_value_mat(wf.ci_coeffs, Hamiltonian, wf.ci_coeffs)
        E_lag_plus = 0.0
        for state in wf.excited_states:
            E_lag_plus += expectation_value_mat(
                state,
                Hamiltonian,
                state,
            )
        theta_params[i] -= step_size
        wf.thetas = theta_params
        df = (E_lag_plus - E_lag) / step_size
        dg = (E_elec_plus - E_elec) / step_size
        gradient_theta[i] = df + 2 * k * dg * (E_elec - E0)
    return gradient_theta
