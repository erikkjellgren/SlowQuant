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
from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing
from slowquant.unitary_coupled_cluster.density_matrix import (
    ReducedDenstiyMatrix,
    get_electronic_energy,
    get_orbital_gradient,
)
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    construct_ups_state,
    expectation_value,
    get_grad_action,
    propagate_state,
    propagate_unitary,
)
from slowquant.unitary_coupled_cluster.operators import Epq, hamiltonian_0i_0a
from slowquant.unitary_coupled_cluster.optimizers import Optimizers
from slowquant.unitary_coupled_cluster.util import UpsStructure


class WaveFunctionUPS:
    def __init__(
        self,
        num_elec: int,
        cas: Sequence[int],
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
        self.num_elec_alpha = num_elec // 2
        self.num_elec_beta = num_elec // 2
        self.num_spin_orbs = 2 * len(h_ao)
        self.num_orbs = len(h_ao)
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
        for i in range(num_elec, self.num_spin_orbs):
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
        hf_det = int("1" * self.num_active_elec + "0" * (self.num_active_spin_orbs - self.num_active_elec), 2)
        self.csf_coeffs[self.ci_info.det2idx[hf_det]] = 1
        self.ci_coeffs = np.copy(self.csf_coeffs)
        # Construct UPS Structure
        self.ups_layout = UpsStructure()
        if ansatz.lower() == "tups":
            self.ups_layout.create_tups(self.num_active_orbs, self.ansatz_options)
        elif ansatz.lower() == "qnp":
            self.ansatz_options["do_qnp"] = True
            self.ups_layout.create_tups(self.num_active_orbs, self.ansatz_options)
        elif ansatz.lower() == "fuccsd":
            self.ansatz_options["S"] = True
            self.ansatz_options["D"] = True
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ups_layout.create_fUCC(self.num_active_orbs, self.num_active_elec, self.ansatz_options)
        elif ansatz.lower() == "ksafupccgsd":
            self.ansatz_options["SAGS"] = True
            self.ansatz_options["GpD"] = True
            self.ups_layout.create_fUCC(self.num_active_orbs, self.num_active_elec, self.ansatz_options)
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
            self.ups_layout.create_fUCC(self.num_active_orbs, self.num_active_elec, self.ansatz_options)
        elif ansatz.lower() == "sdsfucc":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ups_layout.create_SDSfUCC(self.num_active_orbs, self.num_active_elec, self.ansatz_options)
        else:
            raise ValueError(f"Got unknown ansatz, {ansatz}")
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
                        self.thetas,
                        self.ups_layout,
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
                                        self.thetas,
                                        self.ups_layout,
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
                                                self.thetas,
                                                self.ups_layout,
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
                self.thetas,
                self.ups_layout,
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
            print(f"### Number kappa: {len(self.kappa)}")
        print(f"### Number theta: {self.ups_layout.n_params}")
        e_old = 1e12
        print("Full optimization")
        print("Iteration # | Iteration time [s] | Electronic energy [Hartree]")
        for full_iter in range(0, int(maxiter)):
            full_start = time.time()

            # Do ansatz optimization
            if not is_silent_subiterations:
                print("--------Ansatz optimization")
                print("--------Iteration # | Iteration time [s] | Electronic energy [Hartree]")
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
            )
            self._old_opt_parameters = np.zeros_like(self.thetas) + 10**20
            self._E_opt_old = 0.0
            res = optimizer.minimize(
                self.thetas,
                extra_options={"R": self.ups_layout.grad_param_R, "param_names": self.ups_layout.param_names},
            )
            self.thetas = res.x.tolist()

            if orbital_optimization and len(self.kappa) != 0:
                if not is_silent_subiterations:
                    print("--------Orbital optimization")
                    print("--------Iteration # | Iteration time [s] | Electronic energy [Hartree]")
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
                )
                self._old_opt_parameters = np.zeros(len(self.kappa_idx)) + 10**20
                self._E_opt_old = 0.0
                res = optimizer.minimize([0.0] * len(self.kappa_idx))
                for i in range(len(self.kappa)):  # pylint: disable=consider-using-enumerate
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
            time_str = f"{time.time() - full_start:7.2f}"  # type: ignore
            e_str = f"{e_new:3.12f}"
            print(f"{str(full_iter + 1).center(11)} | {time_str.center(18)} | {e_str.center(27)}")  # type: ignore
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
    ) -> None:
        """Run one step optimization of wave function.

        Args:
            optimizer_name: Name of optimizer.
            orbital_optimization: Perform orbital optimization.
            tol: Convergence tolerance.
            maxiter: Maximum number of iterations.
        """
        print("### Parameters information:")
        if orbital_optimization:
            print(f"### Number kappa: {len(self.kappa)}")
        print(f"### Number theta: {self.ups_layout.n_params}")
        if optimizer_name.lower() == "rotosolve":
            if orbital_optimization and len(self.kappa) != 0:
                raise ValueError(
                    "Cannot use RotoSolve together with orbital optimization in the one-step solver."
                )

        print("--------Iteration # | Iteration time [s] | Electronic energy [Hartree]")
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
        optimizer = Optimizers(energy, optimizer_name, grad=gradient, maxiter=maxiter, tol=tol)
        self._old_opt_parameters = np.zeros_like(parameters) + 10**20
        self._E_opt_old = 0.0
        res = optimizer.minimize(
            parameters,
            extra_options={"R": self.ups_layout.grad_param_R, "param_names": self.ups_layout.param_names},
        )
        if orbital_optimization:
            self.thetas = res.x[len(self.kappa) :].tolist()
            for i in range(len(self.kappa)):  # pylint: disable=consider-using-enumerate
                self._kappa[i] = 0.0
                self._kappa_old[i] = 0.0
        else:
            self.thetas = res.x.tolist()
        self._energy_elec = res.fun

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
            rdms = ReducedDenstiyMatrix(
                self.num_inactive_orbs,
                self.num_active_orbs,
                self.num_virtual_orbs,
                rdm1=self.rdm1,
                rdm2=self.rdm2,
            )
            E = get_electronic_energy(
                rdms, self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs
            )
        else:
            E = expectation_value(
                self.ci_coeffs,
                [hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_orbs, self.num_active_orbs)],
                self.ci_coeffs,
                self.ci_info,
                self.thetas,
                self.ups_layout,
            )
        self._E_opt_old = E
        self._old_opt_parameters = np.copy(parameters)
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
            rdms = ReducedDenstiyMatrix(
                self.num_inactive_orbs,
                self.num_active_orbs,
                self.num_virtual_orbs,
                rdm1=self.rdm1,
                rdm2=self.rdm2,
            )
            gradient[:num_kappa] = get_orbital_gradient(
                rdms, self.h_mo, self.g_mo, self.kappa_idx, self.num_inactive_orbs, self.num_active_orbs
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
                self.thetas,
                self.ups_layout,
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
        return gradient
