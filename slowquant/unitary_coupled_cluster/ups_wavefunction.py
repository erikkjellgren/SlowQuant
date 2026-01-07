from __future__ import annotations

import time
from functools import partial
from typing import Any

import numpy as np
import scipy

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing
from slowquant.unitary_coupled_cluster.density_matrix import (
    get_electronic_energy,
    get_orbital_gradient,
)
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    construct_ups_state,
    expectation_value,
    get_grad_action,
    propagate_state,
    propagate_state_SA,
    propagate_unitary,
    propagate_unitary_SA,
)
from slowquant.unitary_coupled_cluster.operators import (
    G1,
    G2,
    Epq,
    G1_sa,
    G2_sa,
    hamiltonian_0i_0a,
)
from slowquant.unitary_coupled_cluster.optimizers import Optimizers
from slowquant.unitary_coupled_cluster.util import (
    UpsStructure,
    iterate_pair_t2,
    iterate_pair_t2_generalized,
    iterate_t1,
    iterate_t1_generalized,
    iterate_t1_sa,
    iterate_t1_sa_generalized,
    iterate_t2,
    iterate_t2_generalized,
    iterate_t2_sa,
    iterate_t2_sa_generalized,
)


class WaveFunctionUPS:
    def __init__(
        self,
        num_elec: int,
        cas: tuple[int, int] | tuple[tuple[int, int], int],
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
        # Init stuff
        self._c_mo = mo_coeffs
        self._h_ao = h_ao
        self._g_ao = g_ao
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
        if isinstance(cas[0], int):
            self.num_elec_alpha = num_elec // 2
            self.num_elec_beta = num_elec // 2
            self.num_active_elec_alpha = cas[0] // 2
            self.num_active_elec_beta = cas[0] // 2
            if cas[0] % 2 != 0:
                self.num_elec_alpha += 1
                self.num_active_elec_alpha += 1
        else:
            self.num_elec_alpha = (num_elec - np.sum(cas[0])) // 2 + cas[0][0]
            self.num_elec_beta = (num_elec - np.sum(cas[0])) // 2 + cas[0][1]
            self.num_active_elec_alpha = cas[0][0]
            self.num_active_elec_beta = cas[0][1]
        self.num_spin_orbs = 2 * len(h_ao)
        self.num_orbs = len(h_ao)
        self.num_active_elec = np.sum(cas[0])
        self.num_active_spin_orbs = 2 * cas[1]
        self.num_inactive_spin_orbs = self.num_elec - self.num_active_elec
        self.num_virtual_spin_orbs = 2 * len(h_ao) - self.num_inactive_spin_orbs - self.num_active_spin_orbs
        self.num_inactive_orbs = self.num_inactive_spin_orbs // 2
        self.num_active_orbs = self.num_active_spin_orbs // 2
        self.num_virtual_orbs = self.num_virtual_spin_orbs // 2
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._h_mo = None
        self._g_mo = None
        self._energy_elec: float | None = None
        self.ansatz_options = ansatz_options
        self.num_energy_evals = 0
        # Construct spin orbital spaces and indices
        self.inactive_spin_idx = [x for x in range(self.num_inactive_spin_orbs)]
        self.active_spin_idx = [x + self.num_inactive_spin_orbs for x in range(self.num_active_spin_orbs)]
        self.virtual_spin_idx = [
            x + self.num_inactive_spin_orbs + self.num_active_spin_orbs
            for x in range(self.num_virtual_spin_orbs)
        ]
        self.active_occ_spin_idx = []
        for i in range(self.num_active_elec_alpha):
            self.active_occ_spin_idx.append(2 * i + self.num_inactive_spin_orbs)
        for i in range(self.num_active_elec_beta):
            self.active_occ_spin_idx.append(2 * i + 1 + self.num_inactive_spin_orbs)
        self.active_occ_spin_idx.sort()
        self.active_unocc_spin_idx = []
        for i in range(self.num_inactive_spin_orbs, self.num_inactive_spin_orbs + self.num_active_spin_orbs):
            if i not in self.active_occ_spin_idx:
                self.active_unocc_spin_idx.append(i)
        # Construct spatial idx
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
        self._kappa = []
        self.kappa_idx = []
        self.kappa_no_activeactive_idx = []
        self.kappa_no_activeactive_idx_dagger = []
        self.kappa_redundant_idx = []
        self._kappa_old = []
        # kappa can be optimized in spatial basis
        # Loop over all q>p orb combinations and find redundant kappas
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                # find redundant kappas
                if p in self.inactive_idx and q in self.inactive_idx:
                    self.kappa_redundant_idx.append((p, q))
                    continue
                if p in self.virtual_idx and q in self.virtual_idx:
                    self.kappa_redundant_idx.append((p, q))
                    continue
                if not include_active_kappa:
                    if p in self.active_idx and q in self.active_idx:
                        self.kappa_redundant_idx.append((p, q))
                        continue
                if not (p in self.active_idx and q in self.active_idx):
                    self.kappa_no_activeactive_idx.append((p, q))
                    self.kappa_no_activeactive_idx_dagger.append((q, p))
                # the rest is non-redundant
                self._kappa.append(0.0)
                self._kappa_old.append(0.0)
                self.kappa_idx.append((p, q))
        # HF like orbital rotation indices
        self.kappa_hf_like_idx = []
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                if p in self.inactive_idx and q in self.virtual_idx:
                    self.kappa_hf_like_idx.append((p, q))
                elif p in self.inactive_idx and q in self.active_unocc_idx:
                    self.kappa_hf_like_idx.append((p, q))
                elif p in self.active_occ_idx and q in self.virtual_idx:
                    self.kappa_hf_like_idx.append((p, q))
        # Construct determinant basis
        self.ci_info = get_indexing(
            self.num_inactive_orbs,
            self.num_active_orbs,
            self.num_virtual_orbs,
            self.num_active_elec_alpha,
            self.num_active_elec_beta,
        )
        self.num_det = len(self.ci_info.idx2det)
        self.csf_coeffs = np.zeros(self.num_det)
        hf_string = ""
        for i in range(self.num_active_orbs):
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
        if ansatz.lower() == "tups":
            self.ansatz_options["do_tups"] = True
            self.ups_layout.create_tiled(self.num_active_orbs, self.ansatz_options)
        elif ansatz.lower() == "qnp":
            self.ansatz_options["do_qnp"] = True
            self.ups_layout.create_tiled(self.num_active_orbs, self.ansatz_options)
        elif ansatz.lower() == "fuccsd":
            self.ansatz_options["S"] = True
            self.ansatz_options["D"] = True
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ups_layout.create_fUCC(
                self.active_occ_idx_shifted,
                self.active_unocc_idx_shifted,
                self.active_occ_spin_idx_shifted,
                self.active_unocc_spin_idx_shifted,
                self.num_active_orbs,
                self.ansatz_options,
            )
        elif ansatz.lower() == "ksafupccgsd":
            self.ansatz_options["SAGS"] = True
            self.ansatz_options["GpD"] = True
            self.ups_layout.create_fUCC(
                self.active_occ_idx_shifted,
                self.active_unocc_idx_shifted,
                self.active_occ_spin_idx_shifted,
                self.active_unocc_spin_idx_shifted,
                self.num_active_orbs,
                self.ansatz_options,
            )
        elif ansatz.lower() == "sdsfuccsd":
            self.ansatz_options["D"] = True
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ups_layout.create_SDSfUCC(self.num_active_orbs, self.num_active_elec, self.ansatz_options)
        elif ansatz.lower() == "ksasdsfupccgsd":
            self.ansatz_options["GpD"] = True
            self.ups_layout.create_SDSfUCC(self.num_active_orbs, self.num_active_elec, self.ansatz_options)
        elif ansatz.lower() == "fucc":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ups_layout.create_fUCC(
                self.active_occ_idx_shifted,
                self.active_unocc_idx_shifted,
                self.active_occ_spin_idx_shifted,
                self.active_unocc_spin_idx_shifted,
                self.num_active_orbs,
                self.ansatz_options,
            )
        elif ansatz.lower() == "sdsfucc":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ups_layout.create_SDSfUCC(self.num_active_orbs, self.num_active_elec, self.ansatz_options)
        elif ansatz.lower() == "adapt":
            None
        else:
            raise ValueError(f"Got unknown ansatz, {ansatz}")
        if self.ups_layout.n_params == 0:
            self._thetas = []
        else:
            self._thetas = np.zeros(self.ups_layout.n_params).tolist()

    @property
    def kappa(self) -> list[float]:
        """Get orbital rotation parameters."""
        return self._kappa.copy()

    @kappa.setter
    def kappa(self, k: list[float]) -> None:
        """Set orbital rotation parameters, and move current expansion point.

        Args:
            k: orbital rotation parameters.
        """
        self._h_mo = None
        self._g_mo = None
        self._energy_elec = None
        self._kappa = k.copy()
        if isinstance(self._kappa, np.ndarray):
            self._kappa = self._kappa.tolist()
        # Move current expansion point.
        self._c_mo = self.c_mo
        self._kappa_old = self.kappa

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
        if isinstance(self._thetas, np.ndarray):
            self._thetas = self._thetas.tolist()
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
        if len(self.kappa) != 0:
            # The MO transformation is calculated as a difference between current kappa and kappa old.
            # This is to make the moving of the expansion point to work with SciPy optimization algorithms.
            # Resetting kappa to zero would mess with any algorithm that has any memory f.x. BFGS.
            if np.max(np.abs(np.array(self.kappa) - np.array(self._kappa_old))) > 0.0:
                for kappa_val, kappa_old, (p, q) in zip(self.kappa, self._kappa_old, self.kappa_idx):
                    kappa_mat[p, q] = kappa_val - kappa_old
                    kappa_mat[q, p] = -(kappa_val - kappa_old)
        # Apply orbital rotation unitary to MO coefficients
        return np.matmul(self._c_mo, scipy.linalg.expm(-kappa_mat))

    @property
    def h_mo(self) -> np.ndarray:
        """Get one-electron Hamiltonian integrals in MO basis.

        Returns:
            One-electron Hamiltonian integrals in MO basis.
        """
        if self._h_mo is None:
            self._h_mo = one_electron_integral_transform(self.c_mo, self._h_ao)
        return self._h_mo

    @property
    def g_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._g_mo is None:
            self._g_mo = two_electron_integral_transform(self.c_mo, self._g_ao)
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
                    val = expectation_value(
                        self.ci_coeffs,
                        [Epq(p, q)],
                        self.ci_coeffs,
                        self.ci_info,
                    )
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
                                self.ci_info,
                            )
                            if q == r:
                                val -= self.rdm1[p_idx, s_idx]
                            self._rdm2[p_idx, q_idx, r_idx, s_idx] = val  # type: ignore
                            self._rdm2[r_idx, s_idx, p_idx, q_idx] = val  # type: ignore
                            self._rdm2[q_idx, p_idx, s_idx, r_idx] = val  # type: ignore
                            self._rdm2[s_idx, r_idx, q_idx, p_idx] = val  # type: ignore
        return self._rdm2

    @property
    def rdm3(self) -> np.ndarray:
        """Calculate three-electron reduced density matrix in the active space.

        Currently not utilizing the full symmetry.

        Returns:
            Three-electron reduced density matrix.
        """
        if self._rdm3 is None:
            self._rdm3 = np.zeros(
                (
                    self.num_active_orbs,
                    self.num_active_orbs,
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
                        for s in range(self.num_inactive_orbs, p + 1):
                            s_idx = s - self.num_inactive_orbs
                            for t in range(self.num_inactive_orbs, r + 1):
                                t_idx = t - self.num_inactive_orbs
                                for u in range(self.num_inactive_orbs, p + 1):
                                    u_idx = u - self.num_inactive_orbs
                                    val = expectation_value(
                                        self.ci_coeffs,
                                        [Epq(p, q), Epq(r, s), Epq(t, u)],
                                        self.ci_coeffs,
                                        self.ci_info,
                                    )
                                    if t == s:
                                        val -= self.rdm2[p_idx, q_idx, r_idx, u_idx]
                                    if r == q:
                                        val -= self.rdm2[p_idx, s_idx, t_idx, u_idx]
                                    if t == q:
                                        val -= self.rdm2[p_idx, u_idx, r_idx, s_idx]
                                    if t == s and r == q:
                                        val -= self.rdm1[p_idx, u_idx]
                                    self._rdm3[p_idx, q_idx, r_idx, s_idx, t_idx, u_idx] = val  # type: ignore
                                    self._rdm3[p_idx, q_idx, t_idx, u_idx, r_idx, s_idx] = val  # type: ignore
                                    self._rdm3[r_idx, s_idx, p_idx, q_idx, t_idx, u_idx] = val  # type: ignore
                                    self._rdm3[r_idx, s_idx, t_idx, u_idx, p_idx, q_idx] = val  # type: ignore
                                    self._rdm3[t_idx, u_idx, p_idx, q_idx, r_idx, s_idx] = val  # type: ignore
                                    self._rdm3[t_idx, u_idx, r_idx, s_idx, p_idx, q_idx] = val  # type: ignore
                                    self._rdm3[q_idx, p_idx, s_idx, r_idx, u_idx, t_idx] = val  # type: ignore
                                    self._rdm3[q_idx, p_idx, u_idx, t_idx, s_idx, r_idx] = val  # type: ignore
                                    self._rdm3[s_idx, r_idx, q_idx, p_idx, u_idx, t_idx] = val  # type: ignore
                                    self._rdm3[s_idx, r_idx, u_idx, t_idx, q_idx, p_idx] = val  # type: ignore
                                    self._rdm3[u_idx, t_idx, q_idx, p_idx, s_idx, r_idx] = val  # type: ignore
                                    self._rdm3[u_idx, t_idx, s_idx, r_idx, q_idx, p_idx] = val  # type: ignore
        return self._rdm3

    @property
    def rdm4(self) -> np.ndarray:
        """Calculate four-electron reduced density matrix in the active space.

        Currently not utilizing the full symmetry.

        Returns:
            Four-electron reduced density matrix.
        """
        if self._rdm4 is None:
            self._rdm4 = np.zeros(
                (
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
                    self.num_active_orbs,
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
                        for s in range(self.num_inactive_orbs, p + 1):
                            s_idx = s - self.num_inactive_orbs
                            for t in range(self.num_inactive_orbs, r + 1):
                                t_idx = t - self.num_inactive_orbs
                                for u in range(self.num_inactive_orbs, p + 1):
                                    u_idx = u - self.num_inactive_orbs
                                    for m in range(self.num_inactive_orbs, t + 1):
                                        m_idx = m - self.num_inactive_orbs
                                        for n in range(self.num_inactive_orbs, p + 1):
                                            n_idx = n - self.num_inactive_orbs
                                            val = expectation_value(
                                                self.ci_coeffs,
                                                [Epq(p, q), Epq(r, s), Epq(t, u), Epq(m, n)],
                                                self.ci_coeffs,
                                                self.ci_info,
                                            )
                                            if r == q:
                                                val -= self.rdm3[p_idx, s_idx, t_idx, u_idx, m_idx, n_idx]
                                            if t == q:
                                                val -= self.rdm3[p_idx, u_idx, r_idx, s_idx, m_idx, n_idx]
                                            if m == q:
                                                val -= self.rdm3[p_idx, n_idx, r_idx, s_idx, t_idx, u_idx]
                                            if m == u:
                                                val -= self.rdm3[p_idx, q_idx, r_idx, s_idx, t_idx, n_idx]
                                            if t == s:
                                                val -= self.rdm3[p_idx, q_idx, r_idx, u_idx, m_idx, n_idx]
                                            if m == s:
                                                val -= self.rdm3[p_idx, q_idx, r_idx, n_idx, t_idx, u_idx]
                                            if m == u and r == q:
                                                val -= self.rdm2[p_idx, s_idx, t_idx, n_idx]
                                            if m == u and t == q:
                                                val -= self.rdm2[p_idx, n_idx, r_idx, s_idx]
                                            if t == s and m == u:
                                                val -= self.rdm2[p_idx, q_idx, r_idx, n_idx]
                                            if t == s and r == q:
                                                val -= self.rdm2[p_idx, u_idx, m_idx, n_idx]
                                            if t == s and m == q:
                                                val -= self.rdm2[p_idx, n_idx, r_idx, u_idx]
                                            if m == s and r == q:
                                                val -= self.rdm2[p_idx, n_idx, t_idx, u_idx]
                                            if m == s and t == q:
                                                val -= self.rdm2[p_idx, u_idx, r_idx, n_idx]
                                            if m == u and t == s and r == q:
                                                val -= self.rdm1[p_idx, n_idx]
                                            self._rdm4[  # type: ignore
                                                p_idx, q_idx, r_idx, s_idx, t_idx, u_idx, m_idx, n_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                p_idx, q_idx, r_idx, s_idx, m_idx, n_idx, t_idx, u_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                p_idx, q_idx, t_idx, u_idx, r_idx, s_idx, m_idx, n_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                p_idx, q_idx, t_idx, u_idx, m_idx, n_idx, r_idx, s_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                p_idx, q_idx, m_idx, n_idx, r_idx, s_idx, t_idx, u_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                p_idx, q_idx, m_idx, n_idx, t_idx, u_idx, r_idx, s_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                r_idx, s_idx, p_idx, q_idx, t_idx, u_idx, m_idx, n_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                r_idx, s_idx, p_idx, q_idx, m_idx, n_idx, t_idx, u_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                r_idx, s_idx, t_idx, u_idx, p_idx, q_idx, m_idx, n_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                r_idx, s_idx, t_idx, u_idx, m_idx, n_idx, p_idx, q_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                r_idx, s_idx, m_idx, n_idx, p_idx, q_idx, t_idx, u_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                r_idx, s_idx, m_idx, n_idx, t_idx, u_idx, p_idx, q_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                t_idx, u_idx, p_idx, q_idx, r_idx, s_idx, m_idx, n_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                t_idx, u_idx, p_idx, q_idx, m_idx, n_idx, r_idx, s_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                t_idx, u_idx, r_idx, s_idx, p_idx, q_idx, m_idx, n_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                t_idx, u_idx, r_idx, s_idx, m_idx, n_idx, p_idx, q_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                t_idx, u_idx, m_idx, n_idx, p_idx, q_idx, r_idx, s_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                t_idx, u_idx, m_idx, n_idx, r_idx, s_idx, p_idx, q_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                m_idx, n_idx, p_idx, q_idx, r_idx, s_idx, t_idx, u_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                m_idx, n_idx, p_idx, q_idx, t_idx, u_idx, r_idx, s_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                m_idx, n_idx, r_idx, s_idx, p_idx, q_idx, t_idx, u_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                m_idx, n_idx, r_idx, s_idx, t_idx, u_idx, p_idx, q_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                m_idx, n_idx, t_idx, u_idx, p_idx, q_idx, r_idx, s_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                m_idx, n_idx, t_idx, u_idx, r_idx, s_idx, p_idx, q_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                q_idx, p_idx, s_idx, r_idx, u_idx, t_idx, n_idx, m_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                q_idx, p_idx, s_idx, r_idx, n_idx, m_idx, u_idx, t_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                q_idx, p_idx, u_idx, t_idx, s_idx, r_idx, n_idx, m_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                q_idx, p_idx, u_idx, t_idx, n_idx, m_idx, s_idx, r_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                q_idx, p_idx, n_idx, m_idx, s_idx, r_idx, u_idx, t_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                q_idx, p_idx, n_idx, m_idx, u_idx, t_idx, s_idx, r_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                s_idx, r_idx, q_idx, p_idx, u_idx, t_idx, n_idx, m_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                s_idx, r_idx, q_idx, p_idx, n_idx, m_idx, u_idx, t_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                s_idx, r_idx, u_idx, t_idx, q_idx, p_idx, n_idx, m_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                s_idx, r_idx, u_idx, t_idx, n_idx, m_idx, q_idx, p_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                s_idx, r_idx, n_idx, m_idx, q_idx, p_idx, u_idx, t_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                s_idx, r_idx, n_idx, m_idx, u_idx, t_idx, q_idx, p_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                u_idx, t_idx, q_idx, p_idx, s_idx, r_idx, n_idx, m_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                u_idx, t_idx, q_idx, p_idx, n_idx, m_idx, s_idx, r_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                u_idx, t_idx, s_idx, r_idx, q_idx, p_idx, n_idx, m_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                u_idx, t_idx, s_idx, r_idx, n_idx, m_idx, q_idx, p_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                u_idx, t_idx, n_idx, m_idx, q_idx, p_idx, s_idx, r_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                u_idx, t_idx, n_idx, m_idx, s_idx, r_idx, q_idx, p_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                n_idx, m_idx, q_idx, p_idx, s_idx, r_idx, u_idx, t_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                n_idx, m_idx, q_idx, p_idx, u_idx, t_idx, s_idx, r_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                n_idx, m_idx, s_idx, r_idx, q_idx, p_idx, u_idx, t_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                n_idx, m_idx, s_idx, r_idx, u_idx, t_idx, q_idx, p_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                n_idx, m_idx, u_idx, t_idx, q_idx, p_idx, s_idx, r_idx
                                            ] = val
                                            self._rdm4[  # type: ignore
                                                n_idx, m_idx, u_idx, t_idx, s_idx, r_idx, q_idx, p_idx
                                            ] = val
        return self._rdm4

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
                [hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs)],
                self.ci_coeffs,
                self.ci_info,
            )
        return self._energy_elec

    def _get_hamiltonian(self, qiskit_form: bool = False) -> FermionicOperator | dict[str, float]:
        """Return electronic Hamiltonian as FermionicOperator.

        Returns:
            FermionicOperator.
        """
        H = hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs)
        H = H.get_folded_operator(self.num_inactive_orbs, self.num_active_orbs, self.num_virtual_orbs)

        if qiskit_form:
            return H.get_qiskit_form(self.num_active_orbs)
        return H

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
            print(f"### Number kappa: {len(self.kappa)}")
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
            self._old_opt_parameters = np.zeros_like(self.thetas) + 10**20
            self._E_opt_old = 0.0
            if optimizer_name.lower() == "rotosolve":
                res = optimizer.minimize(
                    self.thetas,
                    extra_options={
                        "R": self.ups_layout.grad_param_R,
                        "param_names": self.ups_layout.param_names,
                        "f_rotosolve_optimized": self._calc_energy_rotosolve_optimization,
                    },
                )
            else:
                res = optimizer.minimize(
                    self.thetas,
                )
            self.thetas = res.x.tolist()

            if orbital_optimization and len(self.kappa) != 0:
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
                self._old_opt_parameters = np.zeros(len(self.kappa_idx)) + 10**20
                self._E_opt_old = 0.0
                res = optimizer.minimize([0.0] * len(self.kappa_idx))
                for i in range(len(self.kappa)):
                    self._kappa[i] = 0.0
                    self._kappa_old[i] = 0.0
            else:
                # If there is no orbital optimization, then the algorithm is already converged.
                e_new = res.fun
                if orbital_optimization and len(self.kappa) == 0:
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
                print(f"### Number kappa: {len(self.kappa)}")
            print(f"### Number theta: {self.ups_layout.n_params}")
        if optimizer_name.lower() == "rotosolve":
            if orbital_optimization and len(self.kappa) != 0:
                raise ValueError(
                    "Cannot use RotoSolve together with orbital optimization in the one-step solver."
                )

        if not is_silent:
            print(
                "--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #"
            )
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
                parameters = self.kappa + self.thetas
            else:
                parameters = self.kappa
        else:
            parameters = self.thetas
        optimizer = Optimizers(
            energy,
            optimizer_name,
            grad=gradient,
            maxiter=maxiter,
            tol=tol,
            is_silent=is_silent,
            energy_eval_callback=lambda: self.num_energy_evals,
        )
        self._old_opt_parameters = np.zeros_like(parameters) + 10**20
        self._E_opt_old = 0.0
        if optimizer_name.lower() == "rotosolve":
            res = optimizer.minimize(
                parameters,
                extra_options={
                    "R": self.ups_layout.grad_param_R,
                    "param_names": self.ups_layout.param_names,
                    "f_rotosolve_optimized": self._calc_energy_rotosolve_optimization,
                },
            )
        else:
            res = optimizer.minimize(
                parameters,
            )
        if orbital_optimization:
            self.thetas = res.x[len(self.kappa) :].tolist()
            for i in range(len(self.kappa)):
                self._kappa[i] = 0.0
                self._kappa_old[i] = 0.0
        else:
            self.thetas = res.x.tolist()
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
        - pD, pair doubles.
        - SAS, spin-adapted singles.
        - SAD, spin-adapted doubles.
        - GS, generalized singles.
        - GD, generalized doubles.
        - GpD, generalized pair doubles.
        - SAGS, spin-adapted generalized singles.
        - SAGD, spin-adapted generalized doubles.

        Args:
            operator_pool: Which operators to include in the ADAPT.
            maxiter: Maximum iterations.
            grad_threshold: Convergence threshold based on gradient.
            orbital_optimization: Do orbital optimization.
        """
        excitation_pool: list[tuple[int, ...]] = []
        excitation_pool_type = []
        _operator_pool = [x.lower() for x in operator_pool]
        valid_operators = ("sags", "sagd", "s", "d", "sas", "sad", "gs", "gd", "gpd", "pd")
        for operator in _operator_pool:
            if operator not in valid_operators:
                raise ValueError(f"Got invalid operator for ADAPT, {operator}")
        if "s" in _operator_pool:
            for a, i in iterate_t1(self.active_occ_spin_idx_shifted, self.active_unocc_spin_idx_shifted):
                excitation_pool.append((i, a))
                excitation_pool_type.append("single")
        if "d" in _operator_pool:
            for a, i, b, j in iterate_t2(
                self.active_occ_spin_idx_shifted, self.active_unocc_spin_idx_shifted
            ):
                excitation_pool.append((i, j, a, b))
                excitation_pool_type.append("double")
        if "gs" in _operator_pool:
            for a, i in iterate_t1_generalized(self.num_active_spin_orbs):
                excitation_pool.append((i, a))
                excitation_pool_type.append("single")
        if "gd" in _operator_pool:
            for a, i, b, j in iterate_t2_generalized(self.num_active_spin_orbs):
                excitation_pool.append((i, j, a, b))
                excitation_pool_type.append("double")
        if "pd" in _operator_pool:
            for a, i, b, j in iterate_pair_t2(self.active_occ_idx_shifted, self.active_unocc_idx_shifted):
                excitation_pool.append((i, j, a, b))
                excitation_pool_type.append("double")
        if "gpd" in _operator_pool:
            for a, i, b, j in iterate_pair_t2_generalized(self.num_active_orbs):
                excitation_pool.append((i, j, a, b))
                excitation_pool_type.append("double")
        if "sas" in _operator_pool:
            for a, i, _ in iterate_t1_sa(self.active_occ_idx_shifted, self.active_unocc_idx_shifted):
                excitation_pool.append((i, a))
                excitation_pool_type.append("sa_single")
        if "sad" in _operator_pool:
            for a, i, b, j, _, op_case in iterate_t2_sa(
                self.active_occ_idx_shifted, self.active_unocc_idx_shifted
            ):
                excitation_pool.append((i, j, a, b))
                excitation_pool_type.append(f"sa_double_{op_case}")
        if "sags" in _operator_pool:
            for a, i, _ in iterate_t1_sa_generalized(self.num_active_orbs):
                excitation_pool.append((i, a))
                excitation_pool_type.append("sa_single")
        if "sagd" in _operator_pool:
            for a, i, b, j, _, op_case in iterate_t2_sa_generalized(self.num_active_orbs):
                excitation_pool.append((i, j, a, b))
                excitation_pool_type.append(f"sa_double_{op_case}")

        print(
            "Iteration # | Iteration time [s] | Electronic energy [Hartree] | max|grad| [Hartree] | Operator"
        )
        start = time.time()
        for iteration in range(maxiter):
            Hamiltonian = hamiltonian_0i_0a(
                self.h_mo,
                self.g_mo,
                self.num_inactive_orbs,
                self.num_active_orbs,
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
                elif exc_type in ("sa_single",):
                    (i, a) = np.array(excitation_pool[idx])
                    T = G1_sa(i, a, True)
                elif exc_type in ("sa_double_1",):
                    (i, j, a, b) = np.array(excitation_pool[idx])
                    T = G2_sa(i, j, a, b, 1, True)
                elif exc_type in ("sa_double_2",):
                    (i, j, a, b) = np.array(excitation_pool[idx])
                    T = G2_sa(i, j, a, b, 2, True)
                elif exc_type in ("sa_double_3",):
                    (i, j, a, b) = np.array(excitation_pool[idx])
                    T = G2_sa(i, j, a, b, 3, True)
                elif exc_type in ("sa_double_4",):
                    (i, j, a, b) = np.array(excitation_pool[idx])
                    T = G2_sa(i, j, a, b, 4, True)
                elif exc_type in ("sa_double_5",):
                    (i, j, a, b) = np.array(excitation_pool[idx])
                    T = G2_sa(i, j, a, b, 5, True)
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

            self._thetas.append(0.0)
            self.run_wf_optimization_1step("bfgs", orbital_optimization=orbital_optimization, is_silent=True)
            time_str = f"{time.time() - start:7.2f}"
            e_str = f"{self.energy_elec:3.12f}"
            grad_str = f"{np.abs(grad[max_arg]):3.12f}"
            print(
                f"{str(iteration + 1).center(11)} | {time_str.center(18)} | {e_str.center(27)} | {grad_str.center(19)} | {excitation_pool_type[max_arg]}{tuple([int(x) for x in excitation_pool[max_arg]])}"
            )
            start = time.time()

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
            num_kappa = len(self.kappa_idx)
            self.kappa = parameters[:num_kappa]
        if theta_optimization:
            self.thetas = parameters[num_kappa:]
        if kappa_optimization:
            # RDM is more expensive than evaluation of the Hamiltonian.
            # Thus only construct these if orbital-optimization is turned on,
            # since the RDMs will be reused in the oo gradient calculation.
            E = get_electronic_energy(
                self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs, self.rdm1, self.rdm2
            )
        else:
            E = expectation_value(
                self.ci_coeffs,
                [hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs)],
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
            num_kappa = len(self.kappa_idx)
            self.kappa = parameters[:num_kappa]
        if theta_optimization:
            self.thetas = parameters[num_kappa:]
        if kappa_optimization:
            gradient[:num_kappa] = get_orbital_gradient(
                self.h_mo,
                self.g_mo,
                self.kappa_idx,
                self.num_inactive_orbs,
                self.num_active_orbs,
                self.rdm1,
                self.rdm2,
            )
        if theta_optimization:
            Hamiltonian = hamiltonian_0i_0a(
                self.h_mo,
                self.g_mo,
                self.num_inactive_orbs,
                self.num_active_orbs,
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

    def _calc_energy_rotosolve_optimization(
        self,
        parameters: list[float],
        theta_diffs: list[float],
        theta_idx: int,
    ) -> list[float]:
        """Calculate electronic energy.

        Args:
            parameters: Ansatz parameters.
            theta_diffs: List of theta shifts for RotoSolve.
            theta_idx: Index of theta parameter being optimized.

        Returns:
            Electronic energies for all shifted thetas.
        """
        # copy of parameters
        thetas_local = np.asarray(parameters)

        # Prepare reference state up to theta_idx
        state_vec = np.copy(self.csf_coeffs)
        for i in range(0, theta_idx):
            state_vec = propagate_unitary(state_vec, i, self.ci_info, thetas_local, self.ups_layout)

        n_shifts = len(theta_diffs)
        n_state = state_vec.size

        # Preallocate array for shifted states
        state_vecs = np.empty((n_shifts, n_state), dtype=state_vec.dtype)

        # Propagate unitary with all shifted theta at theta_idx
        theta_tmp = thetas_local.copy()
        for j, theta_diff in enumerate(theta_diffs):
            theta_tmp[theta_idx] = theta_diff
            state_vecs[j, :] = propagate_unitary(
                state_vec, theta_idx, self.ci_info, theta_tmp, self.ups_layout
            )

        # Propagate remaining unitaries for all shifted states in batch using SA propagation
        for i in range(theta_idx + 1, len(thetas_local)):
            state_vecs = propagate_unitary_SA(state_vecs, i, self.ci_info, thetas_local, self.ups_layout)

        Hamiltonian = hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs)
        bra_vec = propagate_state_SA([Hamiltonian], state_vecs, self.ci_info, thetas_local, self.ups_layout)

        energies = []
        for bra, ket in zip(bra_vec, state_vecs):
            energies.append(bra @ ket)
        self.num_energy_evals += len(energies)

        return energies
