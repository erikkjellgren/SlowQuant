# pylint: disable=too-many-lines
from __future__ import annotations

import os
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
    get_orbital_gradient,
)
from slowquant.unitary_coupled_cluster.operator_matrix import (
    build_operator_matrix,
    construct_ucc_state,
    expectation_value,
    expectation_value_mat,
    get_indexing,
)
from slowquant.unitary_coupled_cluster.operators import Epq, hamiltonian_0i_0a
from slowquant.unitary_coupled_cluster.util import UccStructure


class WaveFunctionUCC:
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
        self._rdm3 = None
        self._rdm4 = None
        self._h_mo = None
        self._g_mo = None
        self._energy_elec = None
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
        self.kappa_no_activeactive_idx = []
        self.kappa_no_activeactive_idx_dagger = []
        self.kappa_redundant_idx = []
        self._kappa_old = []
        # kappa can be optimized in spatial basis
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                if p in self.inactive_idx and q in self.inactive_idx:
                    self.kappa_redundant_idx.append([p, q])
                    continue
                if p in self.virtual_idx and q in self.virtual_idx:
                    self.kappa_redundant_idx.append([p, q])
                    continue
                if not include_active_kappa:
                    if p in self.active_idx and q in self.active_idx:
                        self.kappa_redundant_idx.append([p, q])
                        continue
                if not (p in self.active_idx and q in self.active_idx):
                    self.kappa_no_activeactive_idx.append([p, q])
                    self.kappa_no_activeactive_idx_dagger.append([q, p])
                self._kappa.append(0.0)
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
        self._ci_coeffs = np.copy(self.csf_coeffs)
        # Construct UCC
        self._excitations = excitations  # Needed for saving the wave function
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

    def save_wavefunction(self, filename: str, force_overwrite: bool = False) -> None:
        """Save the wave function to a compressed NumPy object.

        Args:
            filename: Filename of compressed NumPy object without file extension.
            force_overwrite: Overwrite file if it already exists.
        """
        if os.path.exists(f"{filename}.npz") and not force_overwrite:
            raise ValueError(f"{filename}.npz already exists and force_overwrite is False.")
        np.savez_compressed(
            f"{filename}.npz",
            thetas=self.thetas,
            c_trans=self.c_trans,
            h_ao=self.h_ao,
            g_ao=self.g_ao,
            excitations=self._excitations,
            num_spin_orbs=self.num_spin_orbs,
            num_elec=self.num_elec,
            num_active_elec=self.num_active_elec,
            num_active_orbs=self.num_active_orbs,
            include_active_kappa=self._include_active_kappa,
            energy_elec=self.energy_elec,
        )

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
        self._rdm3 = None
        self._rdm4 = None
        self._energy_elec = None
        self._ci_coeffs = None
        self._thetas = theta.copy()

    @property
    def c_trans(self) -> np.ndarray:
        """Get orbital coefficients.

        Returns:
            Orbital coefficients.
        """
        kappa_mat = np.zeros_like(self._c_orthonormal)
        if len(self.kappa) != 0:
            if np.max(np.abs(self.kappa)) > 0.0:
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
    def rdm3(self) -> np.ndarray:
        """Calcuate three-electron reduced density matrix.

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
        """Calcuate four-electron reduced density matrix.

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
            self._energy_elec = energy_ucc(self.thetas, False, self)
        return self._energy_elec

    def _move_cep(self) -> None:
        """Move current expansion point."""
        c = self.c_trans
        self._c_orthonormal = c
        self._kappa_old = self.kappa

    def run_ucc(
        self,
        orbital_optimization: bool = False,
        is_silent: bool = False,
        convergence_threshold: float = 10**-10,
        maxiter: int = 10000,
    ) -> None:
        """Run optimization of UCC wave function.

        Args:
            orbital_optimization: Do orbital optimization.
            is_silent: Do not print any output.
            convergence_threshold: Energy threshold for convergence.
            maxiter: Maximum number of iterations.
        """
        e_tot = partial(
            energy_ucc,
            orbital_optimized=orbital_optimization,
            wf=self,
        )
        parameter_gradient = partial(
            gradient_ucc,
            orbital_optimized=orbital_optimization,
            wf=self,
        )
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

        def silent_progress(x: list[float]) -> None:  # pylint: disable=unused-argument
            """Print progress during energy minimization of wave function.

            Args:
                x: Wave function parameters.
            """
            global iteration  # pylint: disable=global-variable-undefined
            iteration += 1  # type: ignore [name-defined]

        parameters: list[float] = []
        num_kappa = 0
        num_theta1 = 0
        num_theta2 = 0
        num_theta3 = 0
        num_theta4 = 0
        num_theta5 = 0
        num_theta6 = 0
        if orbital_optimization:
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
        if is_silent:
            res = scipy.optimize.minimize(
                e_tot,
                parameters,
                tol=convergence_threshold,
                callback=silent_progress,
                method="BFGS",
                jac=parameter_gradient,
            )
        else:
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
            res = scipy.optimize.minimize(
                e_tot,
                parameters,
                tol=convergence_threshold,
                callback=print_progress,
                method="BFGS",
                jac=parameter_gradient,
                options={"maxiter": maxiter},
            )
        self._energy_elec = res["fun"]
        param_idx = 0
        if orbital_optimization:
            param_idx += len(self.kappa)
            self.kappa = np.zeros_like(self.kappa).tolist()
            self._kappa_old = np.zeros_like(self.kappa).tolist()
        self.thetas = res["x"][param_idx:].tolist()


def energy_ucc(
    parameters: list[float],
    orbital_optimized: bool,
    wf: WaveFunctionUCC,
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
    if orbital_optimized:
        number_kappas = len(wf.kappa_idx)
        wf.kappa = parameters[:number_kappas]
    wf.thetas = parameters[number_kappas:]
    wf._move_cep()
    E = expectation_value(
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
        wf.ucc_layout,
    )
    # print(E)
    return E


def gradient_ucc(
    parameters: list[float],
    orbital_optimized: bool,
    wf: WaveFunctionUCC,
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
    if orbital_optimized:
        number_kappas = len(wf.kappa_idx)
        wf.kappa = parameters[:number_kappas]
    gradient = np.zeros_like(parameters)
    wf.thetas = parameters[number_kappas:]
    wf._move_cep()
    if orbital_optimized:
        gradient[:number_kappas] = orbital_rotation_gradient(
            wf,
        )
    gradient[number_kappas:] = active_space_parameter_gradient(
        wf,
        parameters,
        orbital_optimized,
    )
    # print(gradient)
    return gradient


def orbital_rotation_gradient(
    wf: WaveFunctionUCC,
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
    wf: WaveFunctionUCC,
    parameters: list[float],
    orbital_optimized: bool,
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
    if orbital_optimized:
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
    E = expectation_value_mat(wf.ci_coeffs, Hamiltonian, wf.ci_coeffs)
    for i in range(len(theta_params)):  # pylint: disable=consider-using-enumerate
        sign_step = (theta_params[i] >= 0).astype(float) * 2 - 1  # type: ignore [attr-defined]
        step_size = eps * sign_step * max(1, abs(theta_params[i]))
        theta_params[i] += step_size
        wf.thetas = theta_params
        E_plus = expectation_value_mat(wf.ci_coeffs, Hamiltonian, wf.ci_coeffs)
        theta_params[i] -= step_size
        wf.thetas = theta_params
        gradient_theta[i] = (E_plus - E) / step_size
    return gradient_theta


def load_wavefunction(filename: str) -> WaveFunctionUCC:
    """Load wave function from a compressed NumPy object.

    Args:
        filename: Filename of compressed NumPy object without file extension.

    Returns:
        Wave function object.
    """
    dat = np.load(f"{filename}.npz")
    wf = WaveFunctionUCC(
        int(dat["num_spin_orbs"]),
        int(dat["num_elec"]),
        (int(dat["num_active_elec"]), int(dat["num_active_orbs"])),
        dat["c_trans"],
        dat["h_ao"],
        dat["g_ao"],
        str(dat["excitations"]),
        bool(dat["include_active_kappa"]),
    )
    wf.thetas = dat["thetas"]
    energy = energy_ucc(wf.thetas, False, wf)  # pylint: disable=protected-access
    if abs(energy - float(dat["energy_elec"])) > 10**-6:
        raise ValueError(
            f'Calculate energy is different from saved energy: {energy} and {float(dat["energy_elec"])}.'
        )
    wf._energy_elec = energy
    print(f"Electronic energy of loaded wave function is {energy}")
    return wf
