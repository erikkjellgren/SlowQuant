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
    two_electron_integral_transform_split,
)
from slowquant.unitary_coupled_cluster.unrestricted_density_matrix import (
    UnrestrictedReducedDensityMatrix,
    get_electronic_energy_unrestricted,
    get_orbital_gradient_unrestricted,
    get_orbital_response_hessian_block_unrestricted,
)
from slowquant.unitary_coupled_cluster.operator_matrix import (
    build_operator_matrix,
    construct_ups_state,
    expectation_value,
    get_grad_action,
    get_indexing,
    propagate_unitary,
)
from slowquant.unitary_coupled_cluster.operators import anni
from slowquant.unitary_coupled_cluster.unrestricted_density_matrix import (
    UnrestrictedReducedDensityMatrix,
    get_electronic_energy_unrestricted,
    get_orbital_gradient_unrestricted,
)
from slowquant.unitary_coupled_cluster.unrestricted_operators import (
    unrestricted_hamiltonian_0i_0a,
    unrestricted_hamiltonian_full_space,
)
from slowquant.unitary_coupled_cluster.util import UpsStructure


class UnrestrictedWaveFunctionUPS:
    def __init__(
        self,
        num_spin_orbs: int,
        num_elec: int,
        cas: tuple[tuple[int, int], int],
        c_orthonormal: np.ndarray,  # tuple[np.ndarray, np.ndarray] ?,
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
                 orbitals are counted in spatial basis. (det her skal rettes til unrestricted)
            c_orthonormal: Initial orbital coefficients. (der skal være to set, alpha og beta)
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
        if len(cas) != 2:
            raise ValueError(
                "Number of electrons in the active space must be specified as a tuple of (alpha, beta)."
            )
        self._c_a_orthonormal = c_orthonormal[0]
        self._c_b_orthonormal = c_orthonormal[1]
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
        self.num_elec_alpha = (num_elec - np.sum(cas[0])) // 2 + cas[0][0]
        self.num_elec_beta = (num_elec - np.sum(cas[0])) // 2 + cas[0][1]
        self.num_spin_orbs = num_spin_orbs
        self.num_orbs = num_spin_orbs // 2
        self._include_active_kappa = include_active_kappa
        self.num_active_elec = np.sum(cas[0])
        self.num_active_elec_alpha = cas[0][0]
        self.num_active_elec_beta = cas[0][1]
        self.num_active_spin_orbs = 0
        self.num_inactive_spin_orbs = 0
        self.num_virtual_spin_orbs = 0
        self._rdm1aa = None
        self._rdm1bb = None
        self._rdm2aaaa = None
        self._rdm2bbbb = None
        self._rdm2aabb = None
        self._rdm2bbaa = None
        self._haa_mo = None
        self._hbb_mo = None
        self._gaaaa_mo = None
        self._gbbbb_mo = None
        self._gaabb_mo = None
        self._gbbaa_mo = None
        self.ansatz_options = ansatz_options
        active_space = []
        orbital_counter = 0
        for i in range(
            2
            * min(
                self.num_elec_alpha - self.num_active_elec_alpha,
                self.num_elec_beta - self.num_active_elec_beta,
            ),
            2 * max(self.num_elec_alpha, self.num_elec_beta),
        ):
            active_space.append(i)
            orbital_counter += 1
        for i in range(
            2 * max(self.num_elec_alpha, self.num_elec_beta),
            2 * max(self.num_elec_alpha, self.num_elec_beta) + 2 * cas[1] - orbital_counter,
        ):
            active_space.append(i)
        for i in range(2 * max(self.num_elec_alpha, self.num_elec_beta)):
            if i in active_space:
                self.active_spin_idx.append(i)
                self.active_occ_spin_idx.append(i)
                self.num_active_spin_orbs += 1
            else:
                self.inactive_spin_idx.append(i)
                self.num_inactive_spin_orbs += 1
        for i in range(2 * max(self.num_elec_alpha, self.num_elec_beta), num_spin_orbs):
            if i in active_space:
                self.active_spin_idx.append(i)
                self.active_unocc_spin_idx.append(i)
                self.num_active_spin_orbs += 1
            else:
                self.virtual_spin_idx.append(i)
                self.num_virtual_spin_orbs += 1
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
        self.kappa_a = []
        self.kappa_b = []
        self.kappa_idx = []
        self.kappa_no_activeactive_idx = []
        self.kappa_no_activeactive_idx_dagger = []
        self.kappa_a_redundant = []
        self.kappa_b_redundant = []
        self.kappa_redundant_idx = []
        self._kappa_a_old = []
        self._kappa_b_old = []
        self._kappa_a_redundant_old = []
        self._kappa_b_redundant_old = []
        # kappa can be optimized in spatial basis
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                if p in self.inactive_idx and q in self.inactive_idx:
                    self.kappa_a_redundant.append(0.0)
                    self.kappa_b_redundant.append(0.0)
                    self._kappa_a_redundant_old.append(0.0)
                    self._kappa_b_redundant_old.append(0.0)
                    self.kappa_redundant_idx.append([p, q])
                    continue
                if p in self.virtual_idx and q in self.virtual_idx:
                    self.kappa_a_redundant.append(0.0)
                    self.kappa_b_redundant.append(0.0)
                    self._kappa_a_redundant_old.append(0.0)
                    self._kappa_b_redundant_old.append(0.0)
                    self.kappa_redundant_idx.append([p, q])
                    continue
                if not include_active_kappa:
                    if p in self.active_idx and q in self.active_idx:
                        self.kappa_a_redundant.append(0.0)
                        self.kappa_b_redundant.append(0.0)
                        self._kappa_a_redundant_old.append(0.0)
                        self._kappa_b_redundant_old.append(0.0)
                        self.kappa_redundant_idx.append([p, q])
                        continue
                if include_active_kappa:
                    if p in self.active_occ_idx and q in self.active_occ_idx:
                        self.kappa_a_redundant.append(0.0)
                        self.kappa_b_redundant.append(0.0)
                        self._kappa_a_redundant_old.append(0.0)
                        self._kappa_b_redundant_old.append(0.0)
                        self.kappa_redundant_idx.append([p, q])
                        continue
                    if p in self.active_unocc_idx and q in self.active_unocc_idx:
                        self.kappa_a_redundant.append(0.0)
                        self.kappa_b_redundant.append(0.0)
                        self._kappa_a_redundant_old.append(0.0)
                        self._kappa_b_redundant_old.append(0.0)
                        self.kappa_redundant_idx.append([p, q])
                        continue
                if not (p in self.active_idx and q in self.active_idx):
                    self.kappa_no_activeactive_idx.append([p, q])
                    self.kappa_no_activeactive_idx_dagger.append([q, p])
                self.kappa_a.append(0.0)
                self.kappa_b.append(0.0)
                self._kappa_a_old.append(0.0)
                self._kappa_b_old.append(0.0)
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
        self.csf_coeffs[self.det2idx[hf_det]] = 1
        self.ci_coeffs = np.copy(self.csf_coeffs)
        self.ups_layout = UpsStructure()
        if ansatz.lower() == "fuccsd":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ansatz_options["S"] = True
            self.ansatz_options["D"] = True
            self.ups_layout.create_fUCC(self.num_active_orbs, self.num_active_elec, self.ansatz_options)
        if ansatz.lower() == "fuccsdt":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ansatz_options["S"] = True
            self.ansatz_options["D"] = True
            self.ansatz_options["T"] = True
            self.ups_layout.create_fUCC(self.num_active_orbs, self.num_active_elec, self.ansatz_options)
        if ansatz.lower() == "fuccsdtq":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ansatz_options["S"] = True
            self.ansatz_options["D"] = True
            self.ansatz_options["T"] = True
            self.ansatz_options["Q"] = True
            self.ups_layout.create_fUCC(self.num_active_orbs, self.num_active_elec, self.ansatz_options)
        if ansatz.lower() == "fuccsdtq5":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ansatz_options["S"] = True
            self.ansatz_options["D"] = True
            self.ansatz_options["T"] = True
            self.ansatz_options["Q"] = True
            self.ansatz_options["5"] = True
            self.ups_layout.create_fUCC(self.num_active_orbs, self.num_active_elec, self.ansatz_options)
        if ansatz.lower() == "fuccsdtq56":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ansatz_options["S"] = True
            self.ansatz_options["D"] = True
            self.ansatz_options["T"] = True
            self.ansatz_options["Q"] = True
            self.ansatz_options["5"] = True
            self.ansatz_options["6"] = True
            self.ups_layout.create_fUCC(self.num_active_orbs, self.num_active_elec, self.ansatz_options)
        else:
            raise ValueError(f"Got unknown ansatz, {ansatz}")
        self._thetas = np.zeros(self.ups_layout.n_params).tolist()

    @property
    def c_a_orthonormal(self) -> np.ndarray:
        """Get orthonormalization coefficients (MO coefficients).

        Returns:
            Orthonormalization coefficients.
        """
        return self._c_a_orthonormal

    @property
    def c_b_orthonormal(self) -> np.ndarray:
        """Get orthonormalization coefficients (MO coefficients).

        Returns:
            Orthonormalization coefficients.
        """
        return self._c_b_orthonormal

    @property
    def c_orthonormal(self) -> tuple[np.ndarray, np.ndarray]:
        """Get orthonormalization coefficients (MO coefficients).

        Returns:
            Orthonormalization coefficients.
        """
        return (self.c_a_orthonormal, self.c_b_orthonormal)

    @c_orthonormal.setter
    def c_orthonormal(self, c: tuple[np.ndarray, np.ndarray]) -> None:
        """Set orthonormalization coefficients.

        Args:
            c: Orthonormalization coefficients.
        """
        self._energy_elec = None
        self._haa_mo = None
        self._hbb_mo = None
        self._gaaaa_mo = None
        self._gbbbb_mo = None
        self._gaabb_mo = None
        self._gbbaa_mo = None
        self._c_a_orthonormal = c[0]
        self._c_b_orthonormal = c[1]

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
        self._rdm1aa = None
        self._rdm1bb = None
        self._rdm2aaaa = None
        self._rdm2bbbb = None
        self._rdm2aabb = None
        self._rdm2bbaa = None
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
    def c_a_trans(self) -> np.ndarray:
        """Get orbital coefficients.

        Returns:
            Orbital coefficients.
        """
        kappa_a_mat = np.zeros_like(self.c_a_orthonormal)
        if len(self.kappa_a) != 0:
            if np.max(np.abs(self.kappa_a)) > 0.0:
                for kappa_a_val, (p, q) in zip(self.kappa_a, self.kappa_idx):
                    kappa_a_mat[p, q] = kappa_a_val
                    kappa_a_mat[q, p] = -kappa_a_val
        if len(self.kappa_a_redundant) != 0:
            if np.max(np.abs(self.kappa_a_redundant)) > 0.0:
                for kappa_a_val, (p, q) in zip(self.kappa_a_redundant, self.kappa_redundant_idx):
                    kappa_a_mat[p, q] = kappa_a_val
                    kappa_a_mat[q, p] = -kappa_a_val
        return np.matmul(self.c_a_orthonormal, scipy.linalg.expm(-kappa_a_mat))

    @property
    def c_b_trans(self) -> np.ndarray:
        """Get orbital coefficients.

        Returns:
            Orbital coefficients.
        """
        kappa_b_mat = np.zeros_like(self.c_b_orthonormal)
        if len(self.kappa_b) != 0:
            if np.max(np.abs(self.kappa_b)) > 0.0:
                for kappa_b_val, (p, q) in zip(self.kappa_b, self.kappa_idx):
                    kappa_b_mat[p, q] = kappa_b_val
                    kappa_b_mat[q, p] = -kappa_b_val
        if len(self.kappa_b_redundant) != 0:
            if np.max(np.abs(self.kappa_b_redundant)) > 0.0:
                for kappa_b_val, (p, q) in zip(self.kappa_b_redundant, self.kappa_redundant_idx):
                    kappa_b_mat[p, q] = kappa_b_val
                    kappa_b_mat[q, p] = -kappa_b_val
        return np.matmul(self.c_b_orthonormal, scipy.linalg.expm(-kappa_b_mat))

    @property
    def c_trans(self) -> tuple[np.ndarray, np.ndarray]:
        """Get orbital coefficients.

        Returns:
            Orbital coefficients.
        """
        return (self.c_a_trans, self.c_b_trans)

    @property
    def haa_mo(self) -> np.ndarray:
        """Get one-electron Hamiltonian integrals in MO basis.

        Returns:
            One-electron Hamiltonian integrals in MO basis.
        """
        if self._haa_mo is None:
            self._haa_mo = one_electron_integral_transform(self.c_a_trans, self.h_ao)
        return self._haa_mo

    @property
    def hbb_mo(self) -> np.ndarray:
        """Get one-electron Hamiltonian integrals in MO basis.

        Returns:
            One-electron Hamiltonian integrals in MO basis.
        """
        if self._hbb_mo is None:
            self._hbb_mo = one_electron_integral_transform(self.c_b_trans, self.h_ao)
        return self._hbb_mo

    @property
    def h_mo(self) -> tuple[np.ndarray, np.ndarray]:
        """Get one-electron Hamiltonian integrals in MO basis.

        Returns:
            One-electron Hamiltonian integrals in MO basis.
        """
        return (self.haa_mo, self.hbb_mo)

    @property
    def gaaaa_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._gaaaa_mo is None:
            self._gaaaa_mo = two_electron_integral_transform(self.c_a_trans, self.g_ao)
        return self._gaaaa_mo

    @property
    def gbbbb_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._gbbbb_mo is None:
            self._gbbbb_mo = two_electron_integral_transform(self.c_b_trans, self.g_ao)
        return self._gbbbb_mo

    @property
    def gaabb_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._gaabb_mo is None:
            self._gaabb_mo = two_electron_integral_transform_split(self.c_a_trans, self.c_b_trans, self.g_ao)
        return self._gaabb_mo

    @property
    def gbbaa_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._gbbaa_mo is None:
            self._gbbaa_mo = two_electron_integral_transform_split(self.c_b_trans, self.c_a_trans, self.g_ao)
        return self._gbbaa_mo

    @property
    def g_mo(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        return (self.gaaaa_mo, self.gbbbb_mo, self.gaabb_mo)

    def _calculate_rdm1(self, spin) -> np.ndarray:
        """Calcuate one-electron reduced density matrix.

        Returns:
            One-electron reduced density matrix.
        """
        # if self._calculate_rdm1 is None:
        self.calculate_rdm1 = np.zeros((self.num_active_orbs, self.num_active_orbs))
        for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
            p_idx = p - self.num_inactive_orbs
            for q in range(self.num_inactive_orbs, p + 1):
                q_idx = q - self.num_inactive_orbs
                val = expectation_value(
                    self.ci_coeffs,
                    [anni(p, spin, True) * anni(q, spin, False)],
                    self.ci_coeffs,
                    self.idx2det,
                    self.det2idx,
                    self.num_inactive_orbs,
                    self.num_active_orbs,
                    self.num_virtual_orbs,
                    self.num_active_elec_alpha,
                    self.num_active_elec_beta,
                    self.thetas,
                    self.ups_layout,
                )
                self.calculate_rdm1[p_idx, q_idx] = val
                self.calculate_rdm1[q_idx, p_idx] = val
        return self.calculate_rdm1

    @property
    def rdm1_C(self) -> np.ndarray:
        if self._rdm1aa is None:
            self._rdm1aa = self._calculate_rdm1("alpha")
        if self._rdm1bb is None:
            self._rdm1bb = self._calculate_rdm1("beta")
        return self._rdm1aa + self._rdm1bb

    # pink book 2.7.6

    @property
    def rdm1_S(self) -> np.ndarray:
        if self._rdm1aa is None:
            self._rdm1aa = self._calculate_rdm1("alpha")
        if self._rdm1bb is None:
            self._rdm1bb = self._calculate_rdm1("beta")
        return self._rdm1aa - self._rdm1bb

    # pink book 2.7.26. in the book there is a half in front? Dpq=1/2*(Dpa,qa - Dpb,qb)

    @property
    def rdm1aa(self) -> np.ndarray:
        if self._rdm1aa is None:
            self._rdm1aa = self._calculate_rdm1("alpha")
        return self._rdm1aa

    @property
    def rdm1bb(self) -> np.ndarray:
        if self._rdm1bb is None:
            self._rdm1bb = self._calculate_rdm1("beta")
        return self._rdm1bb

    def _calculate_rdm2(self, spin1, spin2) -> np.ndarray:
        """Calcuate two-electron unrestricted reduced density matrix.

        Returns:
            Two-electron unrestricted reduced density matrix.
        """

        self.calculate_rdm2 = np.zeros(
            (self.num_active_orbs, self.num_active_orbs, self.num_active_orbs, self.num_active_orbs)
        )
        for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
            p_idx = p - self.num_inactive_orbs
            for q in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                q_idx = q - self.num_inactive_orbs
                for r in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                    r_idx = r - self.num_inactive_orbs
                    for s in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                        s_idx = s - self.num_inactive_orbs
                        val = expectation_value(
                            self.ci_coeffs,
                            [
                                anni(p, spin1, True)
                                * anni(r, spin2, True)
                                * anni(s, spin2, False)
                                * anni(q, spin1, False)
                            ],
                            self.ci_coeffs,
                            self.idx2det,
                            self.det2idx,
                            self.num_inactive_orbs,
                            self.num_active_orbs,
                            self.num_virtual_orbs,
                            self.num_active_elec_alpha,
                            self.num_active_elec_beta,
                            self.thetas,
                            self.ups_layout,
                        )
                        self.calculate_rdm2[p_idx, q_idx, r_idx, s_idx] = val  # type: ignore
                        # self.calculate_rdm2[r_idx, s_idx, p_idx, q_idx] = val # type: ignore
                        # self.calculate_rdm2[q_idx, p_idx, s_idx, r_idx] = val # type: ignore
                        # self.calculate_rdm2[s_idx, r_idx, q_idx, p_idx] = val # type: ignore
        return self.calculate_rdm2

    @property
    def rdm2_C(self) -> np.ndarray:
        if self._rdm2aaaa is None:
            self._rdm2aaaa = self._calculate_rdm2("alpha", "alpha")
        if self._rdm2bbbb is None:
            self._rdm2bbbb = self._calculate_rdm2("beta", "beta")
        if self._rdm2aabb is None:
            self._rdm2aabb = self._calculate_rdm2("alpha", "beta")
        return self._rdm2aaaa + self._rdm2bbbb + 2 * self._rdm2aabb

    # 2*rdm2aabb = rdm2aabb*rdm2bbaa.T

    @property
    def rdm2aaaa(self) -> np.ndarray:
        if self._rdm2aaaa is None:
            self._rdm2aaaa = self._calculate_rdm2("alpha", "alpha")
        return self._rdm2aaaa

    @property
    def rdm2bbbb(self) -> np.ndarray:
        if self._rdm2bbbb is None:
            self._rdm2bbbb = self._calculate_rdm2("beta", "beta")
        return self._rdm2bbbb

    @property
    def rdm2aabb(self) -> np.ndarray:
        if self._rdm2aabb is None:
            self._rdm2aabb = self._calculate_rdm2("alpha", "beta")
        return self._rdm2aabb

    @property
    def rdm2bbaa(self) -> np.ndarray:
        if self._rdm2bbaa is None:
            self._rdm2bbaa = self._calculate_rdm2("beta", "alpha")
        return self._rdm2bbaa


    def manual_gradient(
        wf: UnrestrictedWaveFunctionUPS,
    ) -> np.ndarray:
        # lav en variable der samler alle parametre i expectation value
        h = unrestricted_hamiltonian_full_space(
            wf.haa_mo, wf.hbb_mo, wf.gaaaa_mo, wf.gbbbb_mo, wf.gaabb_mo, wf.gbbaa_mo, wf.num_orbs
        )
        gradient = np.zeros(2 * len(wf.kappa_idx))
        for idx, (m, n) in enumerate(wf.kappa_idx):
            for p in range(wf.num_inactive_orbs + wf.num_active_orbs):
                alpha = expectation_value(
                    wf.ci_coeffs,
                    [anni(m, "alpha", True) * anni(n, "alpha", False) * h],
                    wf.ci_coeffs,
                    wf.idx2det,
                    wf.det2idx,
                    wf.num_inactive_orbs,
                    wf.num_active_orbs,
                    wf.num_virtual_orbs,
                    wf.num_active_elec_alpha,
                    wf.num_active_elec_beta,
                    wf.thetas,
                    wf.ups_layout,
                )
                alpha -= expectation_value(
                    wf.ci_coeffs,
                    [anni(n, "alpha", True) * anni(m, "alpha", False) * h],
                    wf.ci_coeffs,
                    wf.idx2det,
                    wf.det2idx,
                    wf.num_inactive_orbs,
                    wf.num_active_orbs,
                    wf.num_virtual_orbs,
                    wf.num_active_elec_alpha,
                    wf.num_active_elec_beta,
                    wf.thetas,
                    wf.ups_layout,
                )

                alpha -= expectation_value(
                    wf.ci_coeffs,
                    [h * (anni(m, "alpha", True) * anni(n, "alpha", False))],
                    wf.ci_coeffs,
                    wf.idx2det,
                    wf.det2idx,
                    wf.num_inactive_orbs,
                    wf.num_active_orbs,
                    wf.num_virtual_orbs,
                    wf.num_active_elec_alpha,
                    wf.num_active_elec_beta,
                    wf.thetas,
                    wf.ups_layout,
                )
                alpha += expectation_value(
                    wf.ci_coeffs,
                    [h * (anni(n, "alpha", True) * anni(m, "alpha", False))],
                    wf.ci_coeffs,
                    wf.idx2det,
                    wf.det2idx,
                    wf.num_inactive_orbs,
                    wf.num_active_orbs,
                    wf.num_virtual_orbs,
                    wf.num_active_elec_alpha,
                    wf.num_active_elec_beta,
                    wf.thetas,
                    wf.ups_layout,
                )
                beta = expectation_value(
                    wf.ci_coeffs,
                    [anni(m, "beta", True) * anni(n, "beta", False) * h],
                    wf.ci_coeffs,
                    wf.idx2det,
                    wf.det2idx,
                    wf.num_inactive_orbs,
                    wf.num_active_orbs,
                    wf.num_virtual_orbs,
                    wf.num_active_elec_alpha,
                    wf.num_active_elec_beta,
                    wf.thetas,
                    wf.ups_layout,
                )
                beta -= expectation_value(
                    wf.ci_coeffs,
                    [anni(n, "beta", True) * anni(m, "beta", False) * h],
                    wf.ci_coeffs,
                    wf.idx2det,
                    wf.det2idx,
                    wf.num_inactive_orbs,
                    wf.num_active_orbs,
                    wf.num_virtual_orbs,
                    wf.num_active_elec_alpha,
                    wf.num_active_elec_beta,
                    wf.thetas,
                    wf.ups_layout,
                )
                beta -= expectation_value(
                    wf.ci_coeffs,
                    [h * (anni(m, "beta", True) * anni(n, "beta", False))],
                    wf.ci_coeffs,
                    wf.idx2det,
                    wf.det2idx,
                    wf.num_inactive_orbs,
                    wf.num_active_orbs,
                    wf.num_virtual_orbs,
                    wf.num_active_elec_alpha,
                    wf.num_active_elec_beta,
                    wf.thetas,
                    wf.ups_layout,
                )
                beta += expectation_value(
                    wf.ci_coeffs,
                    [h * (anni(n, "beta", True) * anni(m, "beta", False))],
                    wf.ci_coeffs,
                    wf.idx2det,
                    wf.det2idx,
                    wf.num_inactive_orbs,
                    wf.num_active_orbs,
                    wf.num_virtual_orbs,
                    wf.num_active_elec_alpha,
                    wf.num_active_elec_beta,
                    wf.thetas,
                    wf.ups_layout,
                )
                gradient[idx] = alpha
                gradient[idx + len(wf.kappa_idx)] = beta
        return gradient        
    
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
            parameters += self.kappa_a
            parameters += self.kappa_b
            num_kappa += len(self.kappa_a) + len(self.kappa_b)
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
        self.energy_elec = res["fun"]
        param_idx = 0
        if orbital_optimization:
            param_idx += len(self.kappa_a) + len(self.kappa_b)
            for i in range(len(self.kappa_a)):  # pylint: disable=consider-using-enumerate
                self.kappa_a[i] = 0
                self._kappa_a_old[i] = 0
                self.kappa_b[i] = 0
                self._kappa_b_old[i] = 0
            for i in range(len(self.kappa_a_redundant)):  # pylint: disable=consider-using-enumerate
                self.kappa_a_redundant[i] = 0
                self._kappa_a_redundant_old[i] = 0
                self.kappa_b_redundant[i] = 0
                self._kappa_b_redundant_old[i] = 0
        self.thetas = res["x"][param_idx : num_theta + param_idx].tolist()

    @property
    def energy_elec_RDM(self) -> float:
        rdms = UnrestrictedReducedDensityMatrix(
            self.num_inactive_orbs,
            self.num_active_orbs,
            self.num_virtual_orbs,
            rdm1aa=self.rdm1aa,
            rdm1bb=self.rdm1bb,
            rdm2aaaa=self.rdm2aaaa,
            rdm2bbbb=self.rdm2bbbb,
            rdm2aabb=self.rdm2aabb,
            rdm2bbaa=self.rdm2bbaa,
        )
        return get_electronic_energy_unrestricted(
            rdms,
            self.haa_mo,
            self.hbb_mo,
            self.gaaaa_mo,
            self.gbbbb_mo,
            self.gaabb_mo,
            self.gbbaa_mo,
            self.num_inactive_orbs,
            self.num_active_orbs,
        )

    @property
    def orbital_gradient_RDM(self) -> np.ndarray:
        rdms = UnrestrictedReducedDensityMatrix(
            self.num_inactive_orbs,
            self.num_active_orbs,
            self.num_virtual_orbs,
            rdm1aa=self.rdm1aa,
            rdm1bb=self.rdm1bb,
            rdm2aaaa=self.rdm2aaaa,
            rdm2bbbb=self.rdm2bbbb,
            rdm2aabb=self.rdm2aabb,
            rdm2bbaa=self.rdm2bbaa,
        )
        return get_orbital_gradient_unrestricted(
            rdms,
            self.haa_mo,
            self.hbb_mo,
            self.gaaaa_mo,
            self.gbbbb_mo,
            self.gaabb_mo,
            self.gbbaa_mo,
            self.kappa_idx,
            self.num_inactive_orbs,
            self.num_active_orbs,
        )


def energy_ups(
    parameters: Sequence[float],
    orbital_optimized: bool,
    wf: UnrestrictedWaveFunctionUPS,
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
    kappa_a = []
    kappa_b = []
    theta = []
    idx_counter = 0
    if orbital_optimized:
        for _ in range(len(wf.kappa_idx)):
            kappa_a.append(parameters[idx_counter])
            idx_counter += 1
        for _ in range(len(wf.kappa_idx)):
            kappa_b.append(parameters[idx_counter])
            idx_counter += 1
    for par in parameters[idx_counter:]:
        theta.append(par)
    assert len(parameters) == len(kappa_a) + len(kappa_b) + len(theta)

    kappa_a_mat = np.zeros_like(wf.c_a_orthonormal)
    kappa_b_mat = np.zeros_like(wf.c_b_orthonormal)
    if orbital_optimized:
        for kappa_a_val, kappa_b_val, (p, q) in zip(
            np.array(kappa_a) - np.array(wf._kappa_a_old),  # pylint: disable=protected-access
            np.array(kappa_b) - np.array(wf._kappa_b_old),  # pylint: disable=protected-access
            wf.kappa_idx,
        ):
            kappa_a_mat[p, q] = kappa_a_val
            kappa_a_mat[q, p] = -kappa_a_val
            kappa_b_mat[p, q] = kappa_b_val
            kappa_b_mat[q, p] = -kappa_b_val
    if len(wf.kappa_a_redundant) + len(wf.kappa_b_redundant) != 0:
        if np.max(np.abs(wf.kappa_a_redundant)) > 0.0 or np.max(np.abs(wf.kappa_b_redundant)) > 0.0:
            for kappa_a_val, kappa_b_val, (p, q) in zip(
                np.array(wf.kappa_a_redundant)
                - np.array(wf._kappa_a_redundant_old),  # pylint: disable=protected-access
                np.array(wf.kappa_b_redundant)
                - np.array(wf._kappa_b_redundant_old),  # pylint: disable=protected-access
                wf.kappa_redundant_idx,
            ):
                kappa_a_mat[p, q] = kappa_a_val
                kappa_a_mat[q, p] = -kappa_a_val
                kappa_b_mat[p, q] = kappa_b_val
                kappa_b_mat[q, p] = -kappa_b_val
    c_a_trans = np.matmul(wf.c_a_orthonormal, scipy.linalg.expm(-kappa_a_mat))
    c_b_trans = np.matmul(wf.c_b_orthonormal, scipy.linalg.expm(-kappa_b_mat))
    if orbital_optimized:
        wf._kappa_a_old = kappa_a.copy()  # pylint: disable=protected-access
        wf._kappa_b_old = kappa_b.copy()  # pylint: disable=protected-access
        wf._kappa_a_redundant_old = wf.kappa_a_redundant.copy()  # pylint: disable=protected-access
        wf._kappa_b_redundant_old = wf.kappa_b_redundant.copy()  # pylint: disable=protected-access
    # Moving expansion point of kappa
    wf.c_orthonormal = (c_a_trans, c_b_trans)
    # Add thetas
    wf.thetas = theta
    return expectation_value(
        wf.ci_coeffs,
        [
            unrestricted_hamiltonian_0i_0a(
                wf.haa_mo,
                wf.hbb_mo,
                wf.gaaaa_mo,
                wf.gbbbb_mo,
                wf.gaabb_mo,
                wf.gbbaa_mo,
                wf.num_inactive_orbs,
                wf.num_active_orbs,
            )
        ],
        wf.ci_coeffs,
        wf.idx2det,
        wf.det2idx,
        wf.num_inactive_orbs,
        wf.num_active_orbs,
        wf.num_virtual_orbs,
        wf.num_active_elec_alpha,
        wf.num_active_elec_beta,
        wf.thetas,
        wf.ups_layout,
    )


def gradient_ups(
    parameters: Sequence[float],
    orbital_optimized: bool,
    wf: UnrestrictedWaveFunctionUPS,
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
        number_kappas = len(wf.kappa_a) + len(wf.kappa_b)
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
    wf: UnrestrictedWaveFunctionUPS,
) -> np.ndarray:
    """Calcuate electronic gradient with respect to orbital rotations.

    Args:
        wf: Wave function object.

    Return:
        Electronic gradient with respect to orbital rotations.
    """
    rdms = UnrestrictedReducedDensityMatrix(
        wf.num_inactive_orbs,
        wf.num_active_orbs,
        wf.num_virtual_orbs,
        rdm1aa=wf.rdm1aa,
        rdm1bb=wf.rdm1bb,
        rdm2aaaa=wf.rdm2aaaa,
        rdm2bbbb=wf.rdm2bbbb,
        rdm2aabb=wf.rdm2aabb,
        rdm2bbaa=wf.rdm2bbaa,
    )
    gradient = get_orbital_gradient_unrestricted(
        rdms, wf.haa_mo, wf.hbb_mo, wf.gaaaa_mo, wf.gbbbb_mo, wf.gaabb_mo, wf.gbbaa_mo, wf.kappa_idx, wf.num_inactive_orbs, wf.num_active_orbs)

def active_space_parameter_gradient(
    wf: UnrestrictedWaveFunctionUPS,
) -> np.ndarray:
    """Calcuate electronic gradient with respect to active space parameters.

    Args:
        wf: Wave function object.

    Returns:
        Electronic gradient with respect to active spae parameters.
    """
    Hamiltonian = build_operator_matrix(
        unrestricted_hamiltonian_0i_0a(
            wf.haa_mo,
            wf.hbb_mo,
            wf.gaaaa_mo,
            wf.gbbbb_mo,
            wf.gaabb_mo,
            wf.gbbaa_mo,
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

