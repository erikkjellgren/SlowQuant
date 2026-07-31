from __future__ import annotations

import copy
import time
from functools import partial
from typing import Any

import numpy as np
import pyscf
import scipy

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
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.integral_manager import IntegralManager
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    construct_ups_state,
    expectation_value,
    get_grad_action,
    propagate_state,
    propagate_state_SA,
    propagate_unitary,
    propagate_unitary_SA,
)
from slowquant.unitary_coupled_cluster.operators import Epq, hamiltonian_0i_0a
from slowquant.unitary_coupled_cluster.optimizers import Optimizers
from slowquant.unitary_coupled_cluster.util import UpsStructure


class WaveFunctionUPS:
    def __init__(
        self,
        active_space: tuple[int, int] | tuple[tuple[int, int], int],
        mo_coeffs: np.ndarray,
        integral_generator: SlowQuant | pyscf.gto.mole.Mole,
        ansatz: str,
        ansatz_options: dict[str, Any] | None = None,
        reference_determinant: str | None = None,
        include_active_kappa: bool = True,
        resolve_unpaired_idx: str = "both",
        do_pp: bool = False,
    ) -> None:
        """Initialize for UPS wave function.

        Args:
            active_space: (num_active_elec, num_active_orbs) or ((num_active_elec_alpha, num_active_elec_beta), num_active_orbs),
                 orbitals are counted in spatial basis.
            mo_coeffs: Initial orbital coefficients.
            integral_generator: Integral generator object.
            ansatz: Name of ansatz.
            ansatz_options: Ansatz options.
            reference_determinant: Specify a reference determinant for the active space part.
                                   1 specifying occupied orbital and 0 specifying unoccupied orbital.
            include_active_kappa: Include active-active orbital rotations.
            resolve_unpaired_idx: Specify how to resolve spatial index with respect to occupation of reference determinant.
                                  'both', include spatial index in both occ and unocc idx list.
                                  'occ', include spatial index only in occ idx list.
                                  'unocc', include spatial index only in unocc idx list.
            do_pp: Make perfect-pairing reference determinant. Will also reorder the orbitalers to match the determinant.
        """
        if ansatz_options is None:
            ansatz_options = {}
        self.ansatz_options = copy.deepcopy(ansatz_options)
        if len(active_space) != 2:
            raise ValueError(f"cas must have two elements, got {len(active_space)} elements.")
        if isinstance(active_space[0], int):
            if active_space[0] % 2 == 0:
                cas = ((active_space[0] // 2, active_space[0] // 2), active_space[1])
            else:
                # Odd number of electrons mean one electron must be unpaired.
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
        # Reference wave function
        if do_pp:
            if reference_determinant is not None:
                raise ValueError("Both 'do_pp' and 'reference_determinant'.")
            if self.num_active_elec_alpha != self.num_active_elec_beta:
                raise ValueError(
                    "perfect-pairing is only defined for equal number of alpha and beta electrons."
                )
            # Obtain pp determinant
            pp_det = ""
            spin_orb = 0
            elec_count = self.num_active_elec
            while spin_orb < self.num_active_spin_orbs:
                if (
                    elec_count >= 2
                    and (self.num_active_spin_orbs - spin_orb) >= 4
                    and elec_count <= (self.num_active_spin_orbs - spin_orb - 2)
                ):
                    pp_det += "1100"
                    elec_count -= 2
                    spin_orb += 4
                elif elec_count == 0:
                    pp_det += "0"
                    spin_orb += 1
                elif elec_count != 0:
                    pp_det += "1"
                    spin_orb += 1
                    elec_count -= 1
            if len(pp_det) != self.num_active_spin_orbs or pp_det.count("1") != self.num_active_elec:
                raise ValueError("Perfect pairing determinant violates orbital or electron numbers")

            # Swap mo coefficients to resembles pp layout
            hf_det = "1" * self.num_active_elec + "0" * (self.num_active_spin_orbs - self.num_active_elec)
            hole = [i for i, (h, p) in enumerate(zip(hf_det, pp_det)) if h == "1" and p == "0"]
            part = [i for i, (h, p) in enumerate(zip(hf_det, pp_det)) if h == "0" and p == "1"]
            hole_spatial = sorted(set(i // 2 + self.num_inactive_orbs for i in hole))
            part_spatial = sorted(set(i // 2 + self.num_inactive_orbs for i in part))
            pp_mo_coeffs = mo_coeffs.copy()
            pp_mo_coeffs[:, hole_spatial + part_spatial] = pp_mo_coeffs[:, part_spatial + hole_spatial]
            # Note self._c_mo is set previously but overwritten here.
            self._c_mo = pp_mo_coeffs

            ref_det = pp_det
        elif reference_determinant is not None:
            ref_det = reference_determinant
            if len(ref_det) != self.num_active_spin_orbs:
                raise ValueError(
                    f"Reference determinant is {len(ref_det)} spin orbitals and the active space is {self.num_active_spin_orbs} spin orbitals."
                )
            ref_alpha = 0
            ref_beta = 0
            for i, idx in enumerate(ref_det):
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
            elif resolve_unpaired_idx == "both":
                self.active_occ_idx.append(orb_idx)
                self.active_unocc_idx.append(orb_idx)
            elif resolve_unpaired_idx == "occ":
                self.active_occ_idx.append(orb_idx)
            elif resolve_unpaired_idx == "unocc":
                self.active_unocc_idx.append(orb_idx)
            else:
                raise ValueError(
                    f"Got unknown option for resolve_unpaired_idx, {resolve_unpaired_idx}, excepted 'both', 'occ' or 'unocc'."
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
        # Loop over all q>p orb combinations.
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                if p in self.inactive_idx and q in self.inactive_idx:
                    continue
                if p in self.virtual_idx and q in self.virtual_idx:
                    continue
                if not include_active_kappa:
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
        self._ci_coeffs = np.copy(self.ref_coeffs)
        # Construct UPS Structure
        self.ups_layout = UpsStructure()
        if ansatz.lower() in ("tups", "qnp"):
            if ansatz.lower() == "tups":
                self.ansatz_options["do_tups"] = True
            elif ansatz.lower() == "qnp":
                self.ansatz_options["do_qnp"] = True
            self.ups_layout.create_tiled(self.num_active_orbs, self.ansatz_options)
        elif ansatz.lower() in ("fucc", "fuccsd", "ksafupccgsd", "fuccpd", "safuccsd"):
            # Default options
            self.ansatz_options.setdefault("n_layers", 1)
            self.ansatz_options.setdefault("excitations", [])
            if ansatz.lower() == "fuccsd":
                self.ansatz_options["excitations"].append("S")
                self.ansatz_options["excitations"].append("D")
            elif ansatz.lower() == "ksafupccgsd":
                self.ansatz_options["excitations"].append("SAGS")
                self.ansatz_options["excitations"].append("GpD")
            elif ansatz.lower() == "fuccpd":
                self.ansatz_options["excitations"].append("pD")
            elif ansatz.lower() == "safuccspd":
                self.ansatz_options["excitations"].append("SAS")
                self.ansatz_options["excitations"].append("SAD")
            self.ups_layout.create_fUCC(
                self.active_occ_idx_shifted,
                self.active_unocc_idx_shifted,
                self.active_occ_spin_idx_shifted,
                self.active_unocc_spin_idx_shifted,
                self.num_active_orbs,
                self.ansatz_options,
            )
        elif ansatz.lower() in ("sdsfuccsd", "ksasdsfupccgsd"):
            # Default options
            self.ansatz_options.setdefault("n_layers", 1)
            self.ansatz_options.setdefault("excitations", [])
            if ansatz.lower() == "sdsfuccsd":
                self.ansatz_options["excitations"].append("D")
            elif ansatz.lower() == "ksasdsfupccgsd":
                self.ansatz_options["excitations"].append("GpD")
            self.ups_layout.create_SDSfUCC(
                self.active_occ_idx_shifted,
                self.active_unocc_idx_shifted,
                self.active_occ_spin_idx_shifted,
                self.active_unocc_spin_idx_shifted,
                self.num_active_orbs,
                self.ansatz_options,
            )
        else:
            raise ValueError(f"Got unknown ansatz, {ansatz}")
        self._thetas = np.zeros(self.ups_layout.n_params).tolist()
        # Used when converting to circuit wavefunction.
        self._include_active_kappa = include_active_kappa
        self._ref_det = ref_det
        self._resolve_unpaired_idx = resolve_unpaired_idx

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
    def ci_coeffs(self) -> np.ndarray:
        """Get CI coefficients.

        Returns:
            State vector.
        """
        if self._ci_coeffs is None:
            self._ci_coeffs = construct_ups_state(
                self.ref_coeffs,
                self.ci_info,
                self.thetas,
                self.ups_layout,
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
        self._ci_coeffs = None
        self._thetas = theta_vals.copy()

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
            self._h_mo = one_electron_integral_transform(self.c_mo, self.int_gen.h_ao)
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

    def run_wf_optimization(
        self,
        tol: float = 1e-10,
        maxiter: int = 1000,
        orbital_optimization: bool = False,
        theta_optimization: bool = True,
        one_step_optimizer: str = "BFGS",
        theta_optimizer: str = "BFGS",
        orbital_optimizer: str = "BFGS",
        opt_type: str = "1step",
        is_silent_subiterations: bool = False,
    ) -> None:
        """Run variational optimization of wavefunction.

        Args:
            tol: Tolerance for finishing the optimization.
            maxiter: Maximum number of iterations.
            orbital_optimization: Perform orbital optimization.
            theta_optimization: Perform theta optimization.
            one_step_optimizer: Optimizer used for 1step optimizer.
            theta_optimizer: Optimizer used for theta optimization in 2step optimizer.
            orbital_optimizer: Optimizer used for orbital optimization in 2step optimizer.
            opt_type: Optimization type, can be '1step' or '2step'.
            is_silent_subiterations: Silence sub iterations in 2step.
        """
        if len(self.kappa) == 0 and orbital_optimization:
            print("No kappa parameters turning off orbital optimization.")
            orbital_optimization = False
        if len(self.thetas) == 0 and theta_optimization:
            print("No thetas parameters turning off theta optimization.")
            theta_optimization = False
        if opt_type.lower() == "2step" and (not orbital_optimization or not theta_optimization):
            if not orbital_optimization:
                print("Orbital optimization not requested changing optimizer type to 1step.")
                opt_type = "1step"
            elif not theta_optimization:
                print("theta optimization not requested changing optimizer type to 1step.")
                opt_type = "1step"
        print("### Parameters information:")
        if orbital_optimization:
            print(f"### Number kappa: {len(self.kappa)}")
        if theta_optimization:
            print(f"### Number theta: {self.ups_layout.n_params}")
        if opt_type.lower() == "1step":
            self._run_wf_optimization_1step(
                tol, maxiter, orbital_optimization, theta_optimization, one_step_optimizer
            )
        elif opt_type.lower() == "2step":
            self._run_wf_optimization_2step(
                tol, maxiter, theta_optimizer, orbital_optimizer, is_silent_subiterations
            )
        else:
            raise ValueError(f"Got unknown 'opt_type', {opt_type} excpected '1step' or '2step'.")

    def _run_wf_optimization_2step(
        self,
        tol: float,
        maxiter: int,
        theta_optimizer: str,
        orbital_optimizer: str,
        is_silent_subiterations: bool,
    ) -> None:
        """Run two step optimization of wave function.

        This function should not be called from the outside.
        See 'run_wf_optimization' for argument description.
        """
        e_old = 1e12
        print("Full optimization")
        print("Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #")
        for full_iter in range(0, maxiter):
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
                theta_optimizer,
                grad=gradient_theta,
                maxiter=maxiter,
                tol=tol,
                is_silent=is_silent_subiterations,
                energy_eval_callback=lambda: self.num_energy_evals,
            )
            self._old_opt_parameters = np.zeros_like(self.thetas) + 10**20
            self._E_opt_old = 0.0
            if theta_optimizer.lower() == "rotosolve":
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
                orbital_optimizer,
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

    def _run_wf_optimization_1step(
        self,
        tol: float,
        maxiter: int,
        orbital_optimization: bool,
        theta_optimization: bool,
        one_step_optimizer: str,
    ) -> None:
        """Run one step optimization of wave function.

        This function should not be called from the outside.
        See 'run_wf_optimization' for argument description.
        """
        if one_step_optimizer.lower() == "rotosolve" and orbital_optimization:
            raise ValueError(
                "Cannot use RotoSolve together with orbital optimization in the one-step solver."
            )
        print("--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #")
        energy = partial(
            self._calc_energy_optimization,
            theta_optimization=theta_optimization,
            kappa_optimization=orbital_optimization,
        )
        gradient = partial(
            self._calc_gradient_optimization,
            theta_optimization=theta_optimization,
            kappa_optimization=orbital_optimization,
        )
        if orbital_optimization:
            if theta_optimization:
                parameters = self.kappa + self.thetas
            else:
                parameters = self.kappa
        else:
            parameters = self.thetas
        optimizer = Optimizers(
            energy,
            one_step_optimizer,
            grad=gradient,
            maxiter=maxiter,
            tol=tol,
            energy_eval_callback=lambda: self.num_energy_evals,
        )
        self._old_opt_parameters = np.zeros_like(parameters) + 10**20
        self._E_opt_old = 0.0
        if one_step_optimizer.lower() == "rotosolve":
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
            if theta_optimization:
                self.thetas = res.x[len(self.kappa) :].tolist()
            for i in range(len(self.kappa)):
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
            ket_vec = np.copy(self.ref_coeffs)
            ket_vec_tmp = np.copy(self.ref_coeffs)
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
        state_vec = np.copy(self.ref_coeffs)
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
