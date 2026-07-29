from __future__ import annotations

import copy
import time
from functools import partial
from typing import Any

import numpy as np
import pyscf
import scipy
import scipy.sparse as ss

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)
from slowquant.SlowQuant import SlowQuant
from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing
from slowquant.unitary_coupled_cluster.density_matrix import (
    get_electronic_energy,
    get_orbital_gradient,
)
from slowquant.unitary_coupled_cluster.integral_manager import IntegralManager
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    build_operator_matrix,
    construct_ucc_state,
    expectation_value,
    get_ucc_T,
    propagate_state,
)
from slowquant.unitary_coupled_cluster.operators import Epq, hamiltonian_0i_0a
from slowquant.unitary_coupled_cluster.optimizers import Optimizers
from slowquant.unitary_coupled_cluster.util import UccStructure


class WaveFunctionUCC:
    def __init__(
        self,
        active_space: tuple[int, int] | tuple[tuple[int, int], int],
        mo_coeffs: np.ndarray,
        integral_generator: SlowQuant | pyscf.gto.mole.Mole,
        excitations: list[str],
        wavefunction_options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize for UCC wave function.

        Wavefunction Options:
            * include_active_kappa [bool]: Include active-active orbital rotations.
                                           (default: True)
            * resolve_unpaired_idx [str]: Specify how to resolve spatial index with respect to occupation of reference determinant.
                                          'both', include spatial index in both occ and unocc idx list.
                                          'occ', include spatial index only in occ idx list.
                                          'unocc', include spatial index only in unocc idx list.
                                          (default: 'both')
            * reference_determiant [str]: Specify a reference determinant for the active space part.
                                          1 specifying occupied orbital and 0 specifying unoccupied orbital.

        Possible excitations:
            * S: Add single excitations.
            * GS: Add generalized single excitations.
            * SAS: Add spin-adapted single excitations.
            * SAGS: Add generalized spin-adapted single excitations.
            * D: Add double excitations.
            * GD: Add generalized double excitations.
            * pD: Add pair double excitations.
            * GpD: Add generalized pair double excitations.
            * SAD: Add spin-adapted doubles.
            * SAGD: Add generalized spin-adapted doubles.

        Args:
            active_space: (num_active_elec, num_active_orbs) or ((num_active_elec_alpha, num_active_elec_beta), num_active_orbs),
                 orbitals are counted in spatial basis.
            mo_coeffs: Initial orbital coefficients.
            integral_generator: Integral generator object.
            excitations: Unitary coupled cluster excitation operators.
            wavefunction_options: Wavefunction options.
        """
        if wavefunction_options is None:
            wavefunction_options = {}
        self.wavefunction_options = copy.deepcopy(wavefunction_options)
        valid_options = (
            "resolve_unpaired_idx",
            "include_active_kappa",
            "reference_determiant",
        )
        for option in wavefunction_options:
            if option not in wavefunction_options:
                raise ValueError(
                    f"Got unknown option for UPS wave function, {option}. Valid options are: {valid_options}"
                )
        # Default options
        self.wavefunction_options.setdefault("resolve_unpaired_idx", "both")
        self.wavefunction_options.setdefault("include_active_kappa", True)
        if len(active_space) != 2:
            raise ValueError(f"cas must have two elements, got {len(active_space)} elements.")
        if isinstance(active_space[0], int):
            if active_space[0] % 2 == 0:
                cas = ((active_space[0] // 2, active_space[0] // 2), active_space[1])
            else:
                # Uneven number of electrons mean one electron must be unpaired.
                cas = ((active_space[0] // 2 + 1, active_space[0] // 2), active_space[1])
        else:
            cas = ((active_space[0][0], active_space[0][1]), active_space[1])
        # Init stuff
        self.int_gen = IntegralManager(integral_generator)
        if np.sum(cas[0]) > self.int_gen.num_elec:
            raise ValueError("More active electrons than total number electrons.")
        if (np.sum(cas[0]) % 2 == 0 and self.int_gen.num_elec % 2 == 1) or (
            np.sum(cas[0]) % 2 == 1 and self.int_gen.num_elec % 2 == 0
        ):
            raise ValueError("Specified CAS gives odd number of electrons in inactive space.")
        self.num_inactive_spin_orbs = self.int_gen.num_elec - int(np.sum(cas[0]))
        self.num_inactive_orbs = self.num_inactive_spin_orbs // 2
        self.num_active_orbs = cas[1]
        self.num_active_spin_orbs = 2 * self.num_active_orbs
        self.num_active_orbs = self.num_active_spin_orbs // 2
        self.num_virtual_orbs = (
            len(self.int_gen.kinetic_energy) - self.num_inactive_orbs - self.num_active_orbs
        )
        if self.num_virtual_orbs < 0:
            raise ValueError(
                "Number of inactive + number of active orbitals is larger than total number of orbitals."
            )
        self.num_virtual_spin_orbs = 2 * self.num_virtual_orbs
        self.num_orbs = self.num_inactive_orbs + self.num_active_orbs + self.num_virtual_orbs
        self.num_spin_orbs = 2 * self.num_orbs
        self.num_active_elec_alpha = cas[0][0]
        self.num_active_elec_beta = cas[0][1]
        self.num_active_elec = self.num_active_elec_alpha + self.num_active_elec_beta
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._h_mo = None
        self._g_mo = None
        self._energy_elec: float | None = None
        self.num_energy_evals = 0
        self._c_mo = mo_coeffs
        # Used when converting to circuit wavefunction.
        self._include_active_kappa = self.wavefunction_options["include_active_kappa"]
        # Reference wave function
        if "reference_determiant" in self.wavefunction_options.keys():
            ref_det = self.wavefunction_options["reference_determiant"]
            if len(ref_det) != self.num_active_spin_orbs:
                raise ValueError(
                    f"Reference determinant is {len(ref_det)} spin orbitals and the active space is {self.num_active_spin_orbs} spin orbitals."
                )
            ref_alpha = 0
            ref_beta = 0
            for i, idx in ref_det:
                if i % 2 == 0 and idx == "1":
                    ref_alpha += 1
                elif idx == "1":
                    ref_beta += 1
            if ref_alpha != self.num_active_elec_alpha or ref_beta != self.num_active_elec_beta:
                raise ValueError(
                    "Number of electrons ({ref_alpha}, {ref_beta}) is different from the active space ({self.num_active_elec_alpha, self.num_active_elec_beta})."
                )
        else:
            ref_det = ""
            for i in range(self.num_active_orbs):
                if i < self.num_active_elec_alpha:
                    ref_det += "1"
                else:
                    ref_det += "0"
                if i < self.num_active_elec_beta:
                    ref_det += "1"
                else:
                    ref_det += "0"
        # Construct spin orbital indices
        self.inactive_spin_idx = [x for x in range(self.num_inactive_spin_orbs)]
        self.active_spin_idx = [x + self.num_inactive_spin_orbs for x in range(self.num_active_spin_orbs)]
        self.virtual_spin_idx = [
            x + self.num_inactive_spin_orbs + self.num_active_spin_orbs
            for x in range(self.num_virtual_spin_orbs)
        ]
        self.active_occ_spin_idx = []
        self.active_unocc_spin_idx = []
        for i, orb_idx in enumerate(self.active_spin_idx):
            if ref_det[i] == "1":
                self.active_occ_spin_idx.append(orb_idx)
            else:
                self.active_unocc_spin_idx.append(orb_idx)
        self.active_spin_idx_shifted = [x - self.num_inactive_spin_orbs for x in self.active_spin_idx]
        self.active_occ_spin_idx_shifted = [x - self.num_inactive_spin_orbs for x in self.active_occ_spin_idx]
        self.active_unocc_spin_idx_shifted = [
            x - self.num_inactive_spin_orbs for x in self.active_unocc_spin_idx
        ]
        # Construct spatial idx
        self.inactive_idx = [x for x in range(self.num_inactive_orbs)]
        self.active_idx = [x + self.num_inactive_orbs for x in range(self.num_active_orbs)]
        self.virtual_idx = [
            x + self.num_inactive_orbs + self.num_active_orbs for x in range(self.num_virtual_orbs)
        ]
        self.active_occ_idx = []
        self.active_unocc_idx = []
        for i, orb_idx in enumerate(self.active_idx):
            if ref_det[2 * i] == "1" and ref_det[2 * i + 1] == "1":
                self.active_occ_idx.append(orb_idx)
            elif ref_det[2 * i] == "0" and ref_det[2 * i + 1] == "0":
                self.active_unocc_idx.append(orb_idx)
            elif self.wavefunction_options["resolve_unpaired_idx"] == "both":
                self.active_occ_idx.append(orb_idx)
                self.active_unocc_idx.append(orb_idx)
            elif self.wavefunction_options["resolve_unpaired_idx"] == "occ":
                self.active_occ_idx.append(orb_idx)
            elif self.wavefunction_options["resolve_unpaired_idx"] == "unocc":
                self.active_unocc_idx.append(orb_idx)
            else:
                raise ValueError(
                    f"Got unknown option for resolve_unpaired_idx, {wavefunction_options['resolve_unpaired_idx']}, excepted 'both', 'occ' or 'unocc'."
                )
        self.active_idx_shifted = [x - self.num_inactive_orbs for x in self.active_idx]
        self.active_occ_idx_shifted = [x - self.num_inactive_orbs for x in self.active_occ_idx]
        self.active_unocc_idx_shifted = [x - self.num_inactive_orbs for x in self.active_unocc_idx]
        # Find non-redundant kappas
        self._kappa = []
        kappa_idx = []
        kappa_no_activeactive_idx = []
        kappa_no_activeactive_idx_dagger = []
        self._kappa_old = []
        # kappa can be optimized in spatial basis
        # Loop over all q>p orb combinations and find redundant kappas
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                # find redundant kappas
                if p in self.inactive_idx and q in self.inactive_idx:
                    continue
                if p in self.virtual_idx and q in self.virtual_idx:
                    continue
                if not self._include_active_kappa:
                    if p in self.active_idx and q in self.active_idx:
                        continue
                if not (p in self.active_idx and q in self.active_idx):
                    kappa_no_activeactive_idx.append((p, q))
                    kappa_no_activeactive_idx_dagger.append((q, p))
                # the rest is non-redundant
                self._kappa.append(0.0)
                self._kappa_old.append(0.0)
                kappa_idx.append((p, q))
        # HF like orbital rotation indices
        kappa_hf_like_idx = []
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                if p in self.inactive_idx and q in self.virtual_idx:
                    kappa_hf_like_idx.append((p, q))
                elif p in self.inactive_idx and q in self.active_unocc_idx:
                    kappa_hf_like_idx.append((p, q))
                elif p in self.active_occ_idx and q in self.virtual_idx:
                    kappa_hf_like_idx.append((p, q))
        self.kappa_idx = np.array(kappa_idx, dtype=int)
        self.kappa_no_activeactive_idx = np.array(kappa_no_activeactive_idx, dtype=int)
        self.kappa_no_activeactive_idx_dagger = np.array(kappa_no_activeactive_idx_dagger, dtype=int)
        self.kappa_hf_like_idx = np.array(kappa_hf_like_idx, dtype=int)
        # Construct determinant basis
        self.ci_info = get_indexing(
            self.num_inactive_orbs,
            self.num_active_orbs,
            self.num_virtual_orbs,
            self.num_active_elec_alpha,
            self.num_active_elec_beta,
        )
        self.num_det = len(self.ci_info.idx2det)
        self.ref_coeffs = np.zeros(self.num_det, dtype=float)
        print("Reference (active) determinant:", ref_det)
        self.ref_coeffs[self.ci_info.det2idx[int(ref_det, 2)]] = 1
        self.ci_coeffs = np.copy(self.ref_coeffs)
        # Construct UCC Structure
        self.ucc_layout = UccStructure()
        self.ucc_layout.add_excitations(
            excitations,
            self.active_occ_idx_shifted,
            self.active_unocc_idx_shifted,
            self.active_occ_spin_idx_shifted,
            self.active_unocc_spin_idx_shifted,
            self.num_active_orbs,
        )
        self._thetas = np.zeros(self.ucc_layout.n_params).tolist()

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
        self.ci_coeffs = construct_ucc_state(
            self.ref_coeffs,
            self.ci_info,
            self.thetas,
            self.ucc_layout,
        )
        self._thetas = theta.copy()

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
            self._h_mo = one_electron_integral_transform(
                self.c_mo, self.int_gen.kinetic_energy + self.int_gen.nuclear_electron_attraction
            )
        return self._h_mo

    @property
    def g_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._g_mo is None:
            self._g_mo = two_electron_integral_transform(self.c_mo, self.int_gen.electron_electron_repulsion)
        return self._g_mo

    @property
    def rdm1(self) -> np.ndarray:
        """Calculate one-electron reduced density matrix in the active space.

        Returns:
            One-electron reduced density matrix.
        """
        if self._rdm1 is None:
            self._rdm1 = np.zeros((self.num_active_orbs, self.num_active_orbs), dtype=float)
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_ = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_ = q - self.num_inactive_orbs
                    val = expectation_value(
                        self.ci_coeffs,
                        [Epq(p, q)],
                        self.ci_coeffs,
                        self.ci_info,
                    )
                    self._rdm1[p_, q_] = val  # type: ignore
                    self._rdm1[q_, p_] = val  # type: ignore
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
                ),
                dtype=float,
            )
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_ = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_ = q - self.num_inactive_orbs
                    for r in range(self.num_inactive_orbs, p + 1):
                        r_ = r - self.num_inactive_orbs
                        if p == q:
                            s_lim = r + 1
                        elif p == r:
                            s_lim = q + 1
                        elif q < r:
                            s_lim = p
                        else:
                            s_lim = p + 1
                        for s in range(self.num_inactive_orbs, s_lim):
                            s_ = s - self.num_inactive_orbs
                            val = expectation_value(
                                self.ci_coeffs,
                                [Epq(p, q) * Epq(r, s)],
                                self.ci_coeffs,
                                self.ci_info,
                            )
                            if q == r:
                                val -= self.rdm1[p_, s_]
                            self._rdm2[p_, q_, r_, s_] = val  # type: ignore
                            self._rdm2[r_, s_, p_, q_] = val  # type: ignore
                            self._rdm2[q_, p_, s_, r_] = val  # type: ignore
                            self._rdm2[s_, r_, q_, p_] = val  # type: ignore
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
                ),
                dtype=float,
            )
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_ = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_ = q - self.num_inactive_orbs
                    for r in range(self.num_inactive_orbs, p + 1):
                        r_ = r - self.num_inactive_orbs
                        for s in range(self.num_inactive_orbs, p + 1):
                            s_ = s - self.num_inactive_orbs
                            for t in range(self.num_inactive_orbs, r + 1):
                                t_ = t - self.num_inactive_orbs
                                for u in range(self.num_inactive_orbs, p + 1):
                                    u_ = u - self.num_inactive_orbs
                                    val = expectation_value(
                                        self.ci_coeffs,
                                        [Epq(p, q), Epq(r, s), Epq(t, u)],
                                        self.ci_coeffs,
                                        self.ci_info,
                                    )
                                    if t == s:
                                        val -= self.rdm2[p_, q_, r_, u_]
                                    if r == q:
                                        val -= self.rdm2[p_, s_, t_, u_]
                                    if t == q:
                                        val -= self.rdm2[p_, u_, r_, s_]
                                    if t == s and r == q:
                                        val -= self.rdm1[p_, u_]
                                    self._rdm3[p_, q_, r_, s_, t_, u_] = val  # type: ignore
                                    self._rdm3[p_, q_, t_, u_, r_, s_] = val  # type: ignore
                                    self._rdm3[r_, s_, p_, q_, t_, u_] = val  # type: ignore
                                    self._rdm3[r_, s_, t_, u_, p_, q_] = val  # type: ignore
                                    self._rdm3[t_, u_, p_, q_, r_, s_] = val  # type: ignore
                                    self._rdm3[t_, u_, r_, s_, p_, q_] = val  # type: ignore
                                    self._rdm3[q_, p_, s_, r_, u_, t_] = val  # type: ignore
                                    self._rdm3[q_, p_, u_, t_, s_, r_] = val  # type: ignore
                                    self._rdm3[s_, r_, q_, p_, u_, t_] = val  # type: ignore
                                    self._rdm3[s_, r_, u_, t_, q_, p_] = val  # type: ignore
                                    self._rdm3[u_, t_, q_, p_, s_, r_] = val  # type: ignore
                                    self._rdm3[u_, t_, s_, r_, q_, p_] = val  # type: ignore
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
                ),
                dtype=float,
            )
            for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                p_ = p - self.num_inactive_orbs
                for q in range(self.num_inactive_orbs, p + 1):
                    q_ = q - self.num_inactive_orbs
                    for r in range(self.num_inactive_orbs, p + 1):
                        r_ = r - self.num_inactive_orbs
                        for s in range(self.num_inactive_orbs, p + 1):
                            s_ = s - self.num_inactive_orbs
                            for t in range(self.num_inactive_orbs, r + 1):
                                t_ = t - self.num_inactive_orbs
                                for u in range(self.num_inactive_orbs, p + 1):
                                    u_ = u - self.num_inactive_orbs
                                    for m in range(self.num_inactive_orbs, t + 1):
                                        m_ = m - self.num_inactive_orbs
                                        for n in range(self.num_inactive_orbs, p + 1):
                                            n_ = n - self.num_inactive_orbs
                                            val = expectation_value(
                                                self.ci_coeffs,
                                                [Epq(p, q), Epq(r, s), Epq(t, u), Epq(m, n)],
                                                self.ci_coeffs,
                                                self.ci_info,
                                            )
                                            if r == q:
                                                val -= self.rdm3[p_, s_, t_, u_, m_, n_]
                                            if t == q:
                                                val -= self.rdm3[p_, u_, r_, s_, m_, n_]
                                            if m == q:
                                                val -= self.rdm3[p_, n_, r_, s_, t_, u_]
                                            if m == u:
                                                val -= self.rdm3[p_, q_, r_, s_, t_, n_]
                                            if t == s:
                                                val -= self.rdm3[p_, q_, r_, u_, m_, n_]
                                            if m == s:
                                                val -= self.rdm3[p_, q_, r_, n_, t_, u_]
                                            if m == u and r == q:
                                                val -= self.rdm2[p_, s_, t_, n_]
                                            if m == u and t == q:
                                                val -= self.rdm2[p_, n_, r_, s_]
                                            if t == s and m == u:
                                                val -= self.rdm2[p_, q_, r_, n_]
                                            if t == s and r == q:
                                                val -= self.rdm2[p_, u_, m_, n_]
                                            if t == s and m == q:
                                                val -= self.rdm2[p_, n_, r_, u_]
                                            if m == s and r == q:
                                                val -= self.rdm2[p_, n_, t_, u_]
                                            if m == s and t == q:
                                                val -= self.rdm2[p_, u_, r_, n_]
                                            if m == u and t == s and r == q:
                                                val -= self.rdm1[p_, n_]
                                            self._rdm4[p_, q_, r_, s_, t_, u_, m_, n_] = val  # type: ignore
                                            self._rdm4[p_, q_, r_, s_, m_, n_, t_, u_] = val  # type: ignore
                                            self._rdm4[p_, q_, t_, u_, r_, s_, m_, n_] = val  # type: ignore
                                            self._rdm4[p_, q_, t_, u_, m_, n_, r_, s_] = val  # type: ignore
                                            self._rdm4[p_, q_, m_, n_, r_, s_, t_, u_] = val  # type: ignore
                                            self._rdm4[p_, q_, m_, n_, t_, u_, r_, s_] = val  # type: ignore
                                            self._rdm4[r_, s_, p_, q_, t_, u_, m_, n_] = val  # type: ignore
                                            self._rdm4[r_, s_, p_, q_, m_, n_, t_, u_] = val  # type: ignore
                                            self._rdm4[r_, s_, t_, u_, p_, q_, m_, n_] = val  # type: ignore
                                            self._rdm4[r_, s_, t_, u_, m_, n_, p_, q_] = val  # type: ignore
                                            self._rdm4[r_, s_, m_, n_, p_, q_, t_, u_] = val  # type: ignore
                                            self._rdm4[r_, s_, m_, n_, t_, u_, p_, q_] = val  # type: ignore
                                            self._rdm4[t_, u_, p_, q_, r_, s_, m_, n_] = val  # type: ignore
                                            self._rdm4[t_, u_, p_, q_, m_, n_, r_, s_] = val  # type: ignore
                                            self._rdm4[t_, u_, r_, s_, p_, q_, m_, n_] = val  # type: ignore
                                            self._rdm4[t_, u_, r_, s_, m_, n_, p_, q_] = val  # type: ignore
                                            self._rdm4[t_, u_, m_, n_, p_, q_, r_, s_] = val  # type: ignore
                                            self._rdm4[t_, u_, m_, n_, r_, s_, p_, q_] = val  # type: ignore
                                            self._rdm4[m_, n_, p_, q_, r_, s_, t_, u_] = val  # type: ignore
                                            self._rdm4[m_, n_, p_, q_, t_, u_, r_, s_] = val  # type: ignore
                                            self._rdm4[m_, n_, r_, s_, p_, q_, t_, u_] = val  # type: ignore
                                            self._rdm4[m_, n_, r_, s_, t_, u_, p_, q_] = val  # type: ignore
                                            self._rdm4[m_, n_, t_, u_, p_, q_, r_, s_] = val  # type: ignore
                                            self._rdm4[m_, n_, t_, u_, r_, s_, p_, q_] = val  # type: ignore
                                            self._rdm4[q_, p_, s_, r_, u_, t_, n_, m_] = val  # type: ignore
                                            self._rdm4[q_, p_, s_, r_, n_, m_, u_, t_] = val  # type: ignore
                                            self._rdm4[q_, p_, u_, t_, s_, r_, n_, m_] = val  # type: ignore
                                            self._rdm4[q_, p_, u_, t_, n_, m_, s_, r_] = val  # type: ignore
                                            self._rdm4[q_, p_, n_, m_, s_, r_, u_, t_] = val  # type: ignore
                                            self._rdm4[q_, p_, n_, m_, u_, t_, s_, r_] = val  # type: ignore
                                            self._rdm4[s_, r_, q_, p_, u_, t_, n_, m_] = val  # type: ignore
                                            self._rdm4[s_, r_, q_, p_, n_, m_, u_, t_] = val  # type: ignore
                                            self._rdm4[s_, r_, u_, t_, q_, p_, n_, m_] = val  # type: ignore
                                            self._rdm4[s_, r_, u_, t_, n_, m_, q_, p_] = val  # type: ignore
                                            self._rdm4[s_, r_, n_, m_, q_, p_, u_, t_] = val  # type: ignore
                                            self._rdm4[s_, r_, n_, m_, u_, t_, q_, p_] = val  # type: ignore
                                            self._rdm4[u_, t_, q_, p_, s_, r_, n_, m_] = val  # type: ignore
                                            self._rdm4[u_, t_, q_, p_, n_, m_, s_, r_] = val  # type: ignore
                                            self._rdm4[u_, t_, s_, r_, q_, p_, n_, m_] = val  # type: ignore
                                            self._rdm4[u_, t_, s_, r_, n_, m_, q_, p_] = val  # type: ignore
                                            self._rdm4[u_, t_, n_, m_, q_, p_, s_, r_] = val  # type: ignore
                                            self._rdm4[u_, t_, n_, m_, s_, r_, q_, p_] = val  # type: ignore
                                            self._rdm4[n_, m_, q_, p_, s_, r_, u_, t_] = val  # type: ignore
                                            self._rdm4[n_, m_, q_, p_, u_, t_, s_, r_] = val  # type: ignore
                                            self._rdm4[n_, m_, s_, r_, q_, p_, u_, t_] = val  # type: ignore
                                            self._rdm4[n_, m_, s_, r_, u_, t_, q_, p_] = val  # type: ignore
                                            self._rdm4[n_, m_, u_, t_, q_, p_, s_, r_] = val  # type: ignore
                                            self._rdm4[n_, m_, u_, t_, s_, r_, q_, p_] = val  # type: ignore
        return self._rdm4

    def check_orthonormality(self) -> None:
        r"""Check orthonormality of orbitals.

        .. math::
            \boldsymbol{I} = \boldsymbol{C}_\text{MO}\boldsymbol{S}\boldsymbol{C}_\text{MO}^T
        """
        S_ortho = one_electron_integral_transform(self.c_mo, self.int_gen.overlap)
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
                [
                    hamiltonian_0i_0a(
                        self.h_mo,
                        self.g_mo,
                        self.num_inactive_orbs,
                        self.num_active_orbs,
                    )
                ],
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
        # Init parameters
        num_kappa = 0
        if orbital_optimization:
            num_kappa = len(self.kappa)
        num_theta1 = 0
        num_theta2 = 0
        num_theta3 = 0
        num_theta4 = 0
        num_theta5 = 0
        num_theta6 = 0
        for exc_type in self.ucc_layout.excitation_operator_type:
            if exc_type == ("sa_single", "single"):
                num_theta1 += 1
            elif exc_type in (
                "sa_double_1",
                "sa_double_2",
                "sa_double_3",
                "sa_double_4",
                "sa_double_5",
                "double",
            ):
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
        # Optimization
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
        e_old = 1e12
        print("Full optimization")
        print("Iteration # | Iteration time [s] | Electronic energy [Hartree]")
        for full_iter in range(0, int(maxiter)):
            full_start = time.time()

            # Do ansatz optimization
            if not is_silent_subiterations:
                print("--------UCC optimization")
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
            res = optimizer.minimize(self.thetas)
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
                for i in range(len(self.kappa)):
                    self.kappa[i] = 0.0
                    self._kappa_old[i] = 0.0
            else:
                # If theres is no orbital optimization, then the algorithm is already converged.
                e_new = res.fun
                if orbital_optimization and len(self.kappa) == 0:
                    print(
                        "WARNING: No orbital optimization performed, because there is no non-redundant orbital parameters"
                    )
                break

            e_new = res.fun
            time_str = f"{time.time() - full_start:7.2f}"
            e_str = f"{e_new:3.12f}"
            print(f"{str(full_iter + 1).center(11)} | {time_str.center(18)} | {e_str.center(27)}")
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
        # Init parameters
        num_kappa = 0
        if orbital_optimization:
            num_kappa = len(self.kappa)
        num_theta1 = 0
        num_theta2 = 0
        num_theta3 = 0
        num_theta4 = 0
        num_theta5 = 0
        num_theta6 = 0
        for exc_type in self.ucc_layout.excitation_operator_type:
            if exc_type == ("sa_single", "single"):
                num_theta1 += 1
            elif exc_type in (
                "sa_double_1",
                "sa_double_2",
                "sa_double_3",
                "sa_double_4",
                "sa_double_5",
                "double",
            ):
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
        # Optimization
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
        )
        if orbital_optimization:
            self.thetas = res.x[len(self.kappa) :].tolist()
            for i in range(len(self.kappa)):
                self._kappa[i] = 0.0
                self._kappa_old[i] = 0.0
        else:
            self.thetas = res.x.tolist()
        self._energy_elec = res.fun

    def _calc_energy_optimization(
        self,
        parameters: list[float],
        theta_optimization: bool,
        kappa_optimization: bool,
    ) -> float:
        r"""Calculate electronic energy of UCC wave function.

        .. math::
            E = \left<0\left|\hat{H}\right|0\right>

        Args:
            parameters: Sequence of all parameters.
                        Ordered as orbital rotations, active-space singles, active-space doubles, ...
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
                self.h_mo,
                self.g_mo,
                self.num_inactive_orbs,
                self.num_active_orbs,
                self.rdm1,
                self.rdm2,
            )
        else:
            E = expectation_value(
                self.ci_coeffs,
                [
                    hamiltonian_0i_0a(
                        self.h_mo,
                        self.g_mo,
                        self.num_inactive_orbs,
                        self.num_active_orbs,
                    )
                ],
                self.ci_coeffs,
                self.ci_info,
            )
        self._E_opt_old = E
        self._old_opt_parameters = np.copy(parameters)
        return E

    def _calc_gradient_optimization(
        self, parameters: list[float], theta_optimization: bool, kappa_optimization: bool
    ) -> np.ndarray:
        r"""Calculate electronic gradient.

        The gradient with respect to the thetas is calculated with finite-difference after applying the product rule.

        .. math::
            \frac{\partial E}{\partial \theta} = 2\left<\frac{\partial \Psi}{\partial \theta}\left|\hat{H}\right|\Psi\right>

        The bra :math:`\left<\frac{\partial \Psi}{\partial \theta}\right|` is constructed using finite-difference.

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
            # Numerical finite difference gradient
            eps = np.finfo(np.float64).eps ** (
                1 / 2
            )  # half-precision of double-precision floating-point numbers
            if eps < 10e-13:
                raise ValueError(f"Cannot perform finite-difference step-size is too small, {eps}")
            Hket = propagate_state(
                [Hamiltonian],
                self.ci_coeffs,
                self.ci_info,
            )
            E = self.ci_coeffs @ Hket
            theta_params = np.zeros_like(self.thetas)
            Tmat = build_operator_matrix(
                get_ucc_T(self.thetas, self.ucc_layout),
                self.ci_info,
            )
            for i in range(len(theta_params)):
                sign_step = (theta_params[i] >= 0).astype(float) * 2 - 1  # type: ignore [attr-defined]
                step_size = eps * sign_step * max(1, abs(theta_params[i]))
                theta_params[i] += step_size
                Tmat_plus = build_operator_matrix(
                    get_ucc_T(theta_params, self.ucc_layout),
                    self.ci_info,
                )
                bra = ss.linalg.expm_multiply(Tmat + Tmat_plus, self.ref_coeffs, traceA=0.0)
                E_plus = bra @ Hket
                theta_params[i] -= step_size
                gradient[i + num_kappa] = 2 * (E_plus - E) / step_size
        return gradient
