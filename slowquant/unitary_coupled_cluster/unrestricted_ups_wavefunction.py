from __future__ import annotations

import time
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
from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    construct_ups_state,
    expectation_value,
    get_grad_action,
    propagate_state,
    propagate_unitary,
)
from slowquant.unitary_coupled_cluster.operators import (
    G1,
    G2,
    a_op,
)
from slowquant.unitary_coupled_cluster.optimizers import Optimizers
from slowquant.unitary_coupled_cluster.unrestricted_density_matrix import (
    get_electronic_energy_unrestricted,
    get_orbital_gradient_unrestricted,
    get_orbital_response_hessian_block_unrestricted,
    get_orbital_response_metric_sigma_unrestricted,
)
from slowquant.unitary_coupled_cluster.unrestricted_operators import (
    unrestricted_hamiltonian_0i_0a,
    unrestricted_hamiltonian_full_space,
)
from slowquant.unitary_coupled_cluster.util import (
    UpsStructure,
    iterate_pair_t2,
    iterate_pair_t2_generalized,
    iterate_t1,
    iterate_t1_generalized,
    iterate_t2,
    iterate_t2_generalized,
)


class UnrestrictedWaveFunctionUPS:
    def __init__(
        self,
        num_elec: int,
        cas: tuple[tuple[int, int], int],
        mo_coeffs: np.ndarray,  # tuple[np.ndarray, np.ndarray] ?,
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
                 orbitals are counted in spatial basis. (det her skal rettes til unrestricted)
            mo_coeffs: Initial orbital coefficients. (der skal være to set, alpha og beta)
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
        self._c_a_mo = mo_coeffs[0]
        self._c_b_mo = mo_coeffs[1]
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
        self.num_spin_orbs = 2 * len(h_ao)
        self.num_orbs = len(h_ao)
        self._include_active_kappa = include_active_kappa
        self.num_active_elec_alpha = cas[0][0]
        self.num_active_elec_beta = cas[0][1]
        self.num_active_elec = self.num_active_elec_alpha + self.num_active_elec_beta
        self.num_active_spin_orbs = 2 * cas[1]
        self.num_inactive_spin_orbs = self.num_elec - self.num_active_elec
        self.num_virtual_spin_orbs = 2 * len(h_ao) - self.num_inactive_spin_orbs - self.num_active_spin_orbs
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
        self._kappa_a = []
        self._kappa_b = []
        self.kappa_idx = []
        self.kappa_no_activeactive_idx = []
        self.kappa_no_activeactive_idx_dagger = []
        self._kappa_a_redundant = []
        self._kappa_b_redundant = []
        self.kappa_redundant_idx = []
        self._kappa_a_old = []
        self._kappa_b_old = []
        self._kappa_a_redundant_old = []
        self._kappa_b_redundant_old = []
        # kappa can be optimized in spatial basis
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                if p in self.inactive_idx and q in self.inactive_idx:
                    self._kappa_a_redundant.append(0.0)
                    self._kappa_b_redundant.append(0.0)
                    self._kappa_a_redundant_old.append(0.0)
                    self._kappa_b_redundant_old.append(0.0)
                    self.kappa_redundant_idx.append((p, q))
                    continue
                if p in self.virtual_idx and q in self.virtual_idx:
                    self._kappa_a_redundant.append(0.0)
                    self._kappa_b_redundant.append(0.0)
                    self._kappa_a_redundant_old.append(0.0)
                    self._kappa_b_redundant_old.append(0.0)
                    self.kappa_redundant_idx.append((p, q))
                    continue
                if not include_active_kappa:
                    if p in self.active_idx and q in self.active_idx:
                        self._kappa_a_redundant.append(0.0)
                        self._kappa_b_redundant.append(0.0)
                        self._kappa_a_redundant_old.append(0.0)
                        self._kappa_b_redundant_old.append(0.0)
                        self.kappa_redundant_idx.append((p, q))
                        continue
                if include_active_kappa:
                    if p in self.active_occ_idx and q in self.active_occ_idx:
                        self._kappa_a_redundant.append(0.0)
                        self._kappa_b_redundant.append(0.0)
                        self._kappa_a_redundant_old.append(0.0)
                        self._kappa_b_redundant_old.append(0.0)
                        self.kappa_redundant_idx.append((p, q))
                        continue
                    if p in self.active_unocc_idx and q in self.active_unocc_idx:
                        self._kappa_a_redundant.append(0.0)
                        self._kappa_b_redundant.append(0.0)
                        self._kappa_a_redundant_old.append(0.0)
                        self._kappa_b_redundant_old.append(0.0)
                        self.kappa_redundant_idx.append((p, q))
                        continue
                if not (p in self.active_idx and q in self.active_idx):
                    self.kappa_no_activeactive_idx.append((p, q))
                    self.kappa_no_activeactive_idx_dagger.append((q, p))
                self._kappa_a.append(0.0)
                self._kappa_b.append(0.0)
                self._kappa_a_old.append(0.0)
                self._kappa_b_old.append(0.0)
                self.kappa_idx.append((p, q))
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
        self.ups_layout = UpsStructure()
        if ansatz.lower() == "utups":
            self.ansatz_options["do_utups"] = True
            self.ups_layout.create_tiled(self.num_active_orbs, self.ansatz_options)
        elif ansatz.lower() == "uqnp":
            self.ansatz_options["do_uqnp"] = True
            self.ups_layout.create_tiled(self.num_active_orbs, self.ansatz_options)
        elif ansatz.lower() == "fuccsd":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ansatz_options["S"] = True
            self.ansatz_options["D"] = True
            self.ups_layout.create_fUCC(
                self.active_occ_idx_shifted,
                self.active_unocc_idx_shifted,
                self.active_occ_spin_idx_shifted,
                self.active_unocc_spin_idx_shifted,
                self.num_active_orbs,
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
                self.active_occ_idx_shifted,
                self.active_unocc_idx_shifted,
                self.active_occ_spin_idx_shifted,
                self.active_unocc_spin_idx_shifted,
                self.num_active_orbs,
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
                self.active_occ_idx_shifted,
                self.active_unocc_idx_shifted,
                self.active_occ_spin_idx_shifted,
                self.active_unocc_spin_idx_shifted,
                self.num_active_orbs,
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
                self.active_occ_idx_shifted,
                self.active_unocc_idx_shifted,
                self.active_occ_spin_idx_shifted,
                self.active_unocc_spin_idx_shifted,
                self.num_active_orbs,
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
                self.active_occ_idx_shifted,
                self.active_unocc_idx_shifted,
                self.active_occ_spin_idx_shifted,
                self.active_unocc_spin_idx_shifted,
                self.num_active_orbs,
                self.ansatz_options,
            )
        elif ansatz.lower() == "adapt":
            None
        else:
            raise ValueError(f"Got unknown ansatz, {ansatz}")
        if self.ups_layout.n_params == 0:
            self._thetas = []
        else:
            self._thetas = np.zeros(self.ups_layout.n_params).tolist()

    @property
    def kappa_a(self) -> list[float]:
        """Get orbital rotation parameters."""
        return self._kappa_a.copy()

    @property
    def kappa_b(self) -> list[float]:
        """Get orbital rotation parameters."""
        return self._kappa_b.copy()

    @kappa_a.setter
    def kappa_a(self, k: list[float]) -> None:
        """Set orbital rotation parameters, and move current expansion point.

        Args:
            k: orbital rotation parameters.
        """
        self._haa_mo = None
        self._gaaaa_mo = None
        self._gaabb_mo = None
        self._gbbaa_mo = None
        self._energy_elec = None
        self._kappa_a = k.copy()
        if isinstance(self._kappa_a, np.ndarray):
            self._kappa_a = self._kappa_a.tolist()
        # Move current expansion point.
        self._c_a_mo = self.c_a_mo
        self._kappa_a_old = self.kappa_a

    @kappa_b.setter
    def kappa_b(self, k: list[float]) -> None:
        """Set orbital rotation parameters, and move current expansion point.

        Args:
            k: orbital rotation parameters.
        """
        self._hbb_mo = None
        self._gbbbb_mo = None
        self._gaabb_mo = None
        self._gbbaa_mo = None
        self._energy_elec = None
        self._kappa_b = k.copy()
        if isinstance(self._kappa_b, np.ndarray):
            self._kappa_b = self._kappa_b.tolist()
        # Move current expansion point.
        self._c_b_mo = self.c_b_mo
        self._kappa_b_old = self.kappa_b

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
        self._rdm1aa = None
        self._rdm1bb = None
        self._rdm2aaaa = None
        self._rdm2bbbb = None
        self._rdm2aabb = None
        self._rdm2bbaa = None
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
    def c_a_mo(self) -> np.ndarray:
        """Get molecular orbital coefficients.

        Returns:
            Molecular orbital coefficients.
        """
        # Construct anti-hermitian kappa matrix
        kappa_mat = np.zeros_like(self._c_a_mo)
        if len(self.kappa_a) != 0:
            # The MO transformation is calculated as a difference between current kappa and kappa old.
            # This is to make the moving of the expansion point to work with SciPy optimization algorithms.
            # Resetting kappa to zero would mess with any algorithm that has any memory f.x. BFGS.
            if np.max(np.abs(np.array(self.kappa_a) - np.array(self._kappa_a_old))) > 0.0:
                for kappa_val, kappa_old, (p, q) in zip(self.kappa_a, self._kappa_a_old, self.kappa_idx):
                    kappa_mat[p, q] = kappa_val - kappa_old
                    kappa_mat[q, p] = -(kappa_val - kappa_old)
        # Apply orbital rotation unitary to MO coefficients
        return np.matmul(self._c_a_mo, scipy.linalg.expm(-kappa_mat))

    @property
    def c_b_mo(self) -> np.ndarray:
        """Get molecular orbital coefficients.

        Returns:
            Molecular orbital coefficients.
        """
        # Construct anti-hermitian kappa matrix
        kappa_mat = np.zeros_like(self._c_b_mo)
        if len(self.kappa_b) != 0:
            # The MO transformation is calculated as a difference between current kappa and kappa old.
            # This is to make the moving of the expansion point to work with SciPy optimization algorithms.
            # Resetting kappa to zero would mess with any algorithm that has any memory f.x. BFGS.
            if np.max(np.abs(np.array(self.kappa_b) - np.array(self._kappa_b_old))) > 0.0:
                for kappa_val, kappa_old, (p, q) in zip(self.kappa_b, self._kappa_b_old, self.kappa_idx):
                    kappa_mat[p, q] = kappa_val - kappa_old
                    kappa_mat[q, p] = -(kappa_val - kappa_old)
        # Apply orbital rotation unitary to MO coefficients
        return np.matmul(self._c_b_mo, scipy.linalg.expm(-kappa_mat))

    @property
    def haa_mo(self) -> np.ndarray:
        """Get one-electron Hamiltonian integrals in MO basis.

        Returns:
            One-electron Hamiltonian integrals in MO basis.
        """
        if self._haa_mo is None:
            self._haa_mo = one_electron_integral_transform(self.c_a_mo, self.h_ao)
        return self._haa_mo

    @property
    def hbb_mo(self) -> np.ndarray:
        """Get one-electron Hamiltonian integrals in MO basis.

        Returns:
            One-electron Hamiltonian integrals in MO basis.
        """
        if self._hbb_mo is None:
            self._hbb_mo = one_electron_integral_transform(self.c_b_mo, self.h_ao)
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
            self._gaaaa_mo = two_electron_integral_transform(self.c_a_mo, self.g_ao)
        return self._gaaaa_mo

    @property
    def gbbbb_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._gbbbb_mo is None:
            self._gbbbb_mo = two_electron_integral_transform(self.c_b_mo, self.g_ao)
        return self._gbbbb_mo

    @property
    def gaabb_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._gaabb_mo is None:
            self._gaabb_mo = two_electron_integral_transform_split(self.c_a_mo, self.c_b_mo, self.g_ao)
        return self._gaabb_mo

    @property
    def gbbaa_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._gbbaa_mo is None:
            self._gbbaa_mo = two_electron_integral_transform_split(self.c_b_mo, self.c_a_mo, self.g_ao)
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
        calculate_rdm1 = np.zeros((self.num_active_orbs, self.num_active_orbs))
        for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
            p_idx = p - self.num_inactive_orbs
            for q in range(self.num_inactive_orbs, p + 1):
                q_idx = q - self.num_inactive_orbs
                val = expectation_value(
                    self.ci_coeffs,
                    [a_op(p, spin, True) * a_op(q, spin, False)],
                    self.ci_coeffs,
                    self.ci_info,
                )
                calculate_rdm1[p_idx, q_idx] = val
                calculate_rdm1[q_idx, p_idx] = val
        return calculate_rdm1

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
        calculate_rdm2 = np.zeros(
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
                                a_op(p, spin1, True)
                                * a_op(r, spin2, True)
                                * a_op(s, spin2, False)
                                * a_op(q, spin1, False)
                            ],
                            self.ci_coeffs,
                            self.ci_info,
                        )
                        calculate_rdm2[p_idx, q_idx, r_idx, s_idx] = val  # type: ignore
                        # calculate_rdm2[r_idx, s_idx, p_idx, q_idx] = val # type: ignore
                        # calculate_rdm2[q_idx, p_idx, s_idx, r_idx] = val # type: ignore
                        # calculate_rdm2[s_idx, r_idx, q_idx, p_idx] = val # type: ignore
        return calculate_rdm2

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
                    unrestricted_hamiltonian_0i_0a(
                        self.haa_mo,
                        self.hbb_mo,
                        self.gaaaa_mo,
                        self.gbbbb_mo,
                        self.gaabb_mo,
                        self.gbbaa_mo,
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
        print("### Parameters information:")
        if orbital_optimization:
            print(f"### Number kappa: {len(self.kappa_a) + len(self.kappa_b)}")
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

            if orbital_optimization and len(self.kappa_a) + len(self.kappa_b) != 0:
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
                self._old_opt_parameters = np.zeros(len(self.kappa_a) + len(self.kappa_b)) + 10**20
                self._E_opt_old = 0.0
                res = optimizer.minimize([0.0] * (len(self.kappa_a) + len(self.kappa_b)))
                for i in range(len(self.kappa_a)):
                    self._kappa_a[i] = 0.0
                    self._kappa_a_old[i] = 0.0
                for i in range(len(self.kappa_b)):
                    self._kappa_b[i] = 0.0
                    self._kappa_b_old[i] = 0.0
            else:
                # If there is no orbital optimization, then the algorithm is already converged.
                e_new = res.fun
                if orbital_optimization and len(self.kappa_a) + len(self.kappa_b) == 0:
                    print(
                        "WARNING: No orbital optimization performed, because there is no non-redundant orbital parameters."
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
                print(f"### Number kappa: {len(self.kappa_a) + len(self.kappa_b)}")
            print(f"### Number theta: {self.ups_layout.n_params}")
        if optimizer_name.lower() == "rotosolve":
            if orbital_optimization and len(self.kappa_a) + len(self.kappa_b) != 0:
                raise ValueError(
                    "Cannot use RotoSolve together with orbital optimization in the one-step solver."
                )

        if not is_silent:
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
                parameters = self.kappa_a + self.kappa_b + self.thetas
            else:
                parameters = self.kappa_a + self.kappa_b
        else:
            parameters = self.thetas
        optimizer = Optimizers(
            energy, optimizer_name, grad=gradient, maxiter=maxiter, tol=tol, is_silent=is_silent
        )
        self._old_opt_parameters = np.zeros_like(parameters) + 10**20
        self._E_opt_old = 0.0
        res = optimizer.minimize(
            parameters,
            extra_options={"R": self.ups_layout.grad_param_R, "param_names": self.ups_layout.param_names},
        )
        if orbital_optimization:
            self.thetas = res.x[len(self.kappa_a) + len(self.kappa_b) :].tolist()
            for i in range(len(self.kappa_a)):
                self._kappa_a[i] = 0.0
                self._kappa_a_old[i] = 0.0
            for i in range(len(self.kappa_b)):
                self._kappa_b[i] = 0.0
                self._kappa_b_old[i] = 0.0
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
        - GS, generalized singles.
        - GD, generalized doubles.
        - GpD, generalized pair doubles.

        Args:
            operator_pool: Which operators to include in the ADAPT.
            maxiter: Maximum iterations.
            grad_threshold: Convergence threshold based on gradient.
            orbital_optimization: Do orbital optimization.
        """
        excitation_pool: list[tuple[int, ...]] = []
        excitation_pool_type = []
        _operator_pool = [x.lower() for x in operator_pool]
        valid_operators = ("s", "d", "gs", "gd", "gpd", "pd")
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

        print(
            "Iteration # | Iteration time [s] | Electronic energy [Hartree] | max|grad| [Hartree] | Operator"
        )
        start = time.time()
        for iteration in range(maxiter):
            Hamiltonian = unrestricted_hamiltonian_0i_0a(
                self.haa_mo,
                self.hbb_mo,
                self.gaaaa_mo,
                self.gbbbb_mo,
                self.gaabb_mo,
                self.gbbaa_mo,
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
        num_kappa_a = 0
        num_kappa_b = 0
        if kappa_optimization:
            num_kappa_a = len(self.kappa_a)
            num_kappa_b = len(self.kappa_b)
            self.kappa_a = parameters[:num_kappa_a]
            self.kappa_b = parameters[num_kappa_a : num_kappa_a + num_kappa_b]
        if theta_optimization:
            self.thetas = parameters[num_kappa_a + num_kappa_b :]
        if kappa_optimization:
            # RDM is more expensive than evaluation of the Hamiltonian.
            # Thus only construct these if orbital-optimization is turned on,
            # since the RDMs will be reused in the oo gradient calculation.
            E = get_electronic_energy_unrestricted(
                self.haa_mo,
                self.hbb_mo,
                self.gaaaa_mo,
                self.gbbbb_mo,
                self.gaabb_mo,
                self.gbbaa_mo,
                self.num_inactive_orbs,
                self.num_active_orbs,
                self.rdm1aa,
                self.rdm1bb,
                self.rdm2aaaa,
                self.rdm2bbbb,
                self.rdm2aabb,
                self.rdm2bbaa,
            )
        else:
            E = expectation_value(
                self.ci_coeffs,
                [
                    unrestricted_hamiltonian_0i_0a(
                        self.haa_mo,
                        self.hbb_mo,
                        self.gaaaa_mo,
                        self.gbbbb_mo,
                        self.gaabb_mo,
                        self.gbbaa_mo,
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
        """Calculate electronic gradient.

        Args:
            parameters: Ansatz and orbital rotation parameters.
            theta_optimization: If used in theta optimization.
            kappa_optimization: If used in kappa optimization.

        Returns:
            Electronic gradient.
        """
        gradient = np.zeros(len(parameters))
        num_kappa_a = 0
        num_kappa_b = 0
        if kappa_optimization:
            num_kappa_a = len(self.kappa_a)
            num_kappa_b = len(self.kappa_b)
            self.kappa_a = parameters[:num_kappa_a]
            self.kappa_b = parameters[num_kappa_a : num_kappa_a + num_kappa_b]
        if theta_optimization:
            self.thetas = parameters[num_kappa_a + num_kappa_b :]
        if kappa_optimization:
            gradient[: num_kappa_a + num_kappa_b] = get_orbital_gradient_unrestricted(
                self.haa_mo,
                self.hbb_mo,
                self.gaaaa_mo,
                self.gbbbb_mo,
                self.gaabb_mo,
                self.gbbaa_mo,
                self.kappa_idx,
                self.num_inactive_orbs,
                self.num_active_orbs,
                self.rdm1aa,
                self.rdm1bb,
                self.rdm2aaaa,
                self.rdm2bbbb,
                self.rdm2aabb,
                self.rdm2bbaa,
            )
        if theta_optimization:
            Hamiltonian = unrestricted_hamiltonian_0i_0a(
                self.haa_mo,
                self.hbb_mo,
                self.gaaaa_mo,
                self.gbbbb_mo,
                self.gaabb_mo,
                self.gbbaa_mo,
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
                gradient[i + num_kappa_a + num_kappa_b] += 2 * np.matmul(bra_vec, ket_vec_tmp)
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

    @property
    def energy_elec_RDM(self) -> float:
        return get_electronic_energy_unrestricted(
            self.haa_mo,
            self.hbb_mo,
            self.gaaaa_mo,
            self.gbbbb_mo,
            self.gaabb_mo,
            self.gbbaa_mo,
            self.num_inactive_orbs,
            self.num_active_orbs,
            self.rdm1aa,
            self.rdm1bb,
            self.rdm2aaaa,
            self.rdm2bbbb,
            self.rdm2aabb,
            self.rdm2bbaa,
        )

    @property
    def orbital_gradient_RDM(self) -> np.ndarray:
        return get_orbital_gradient_unrestricted(
            self.haa_mo,
            self.hbb_mo,
            self.gaaaa_mo,
            self.gbbbb_mo,
            self.gaabb_mo,
            self.gbbaa_mo,
            self.kappa_idx,
            self.num_inactive_orbs,
            self.num_active_orbs,
            self.rdm1aa,
            self.rdm1bb,
            self.rdm2aaaa,
            self.rdm2bbbb,
            self.rdm2aabb,
            self.rdm2bbaa,
        )

    # print the unrestricted orbital response hessian block for test
    @property
    def orbital_hessian_unrestricted_A(self) -> np.ndarray:
        return get_orbital_response_hessian_block_unrestricted(
            self.haa_mo,
            self.hbb_mo,
            self.gaaaa_mo,
            self.gbbbb_mo,
            self.gaabb_mo,
            self.gbbaa_mo,
            self.kappa_no_activeactive_idx_dagger,
            self.kappa_no_activeactive_idx,
            self.num_inactive_orbs,
            self.num_active_orbs,
            self.rdm1aa,
            self.rdm1bb,
            self.rdm2aaaa,
            self.rdm2bbbb,
            self.rdm2aabb,
            self.rdm2bbaa,
        )

    @property
    def orbital_hessian_unrestricted_B(self) -> np.ndarray:
        return get_orbital_response_hessian_block_unrestricted(
            self.haa_mo,
            self.hbb_mo,
            self.gaaaa_mo,
            self.gbbbb_mo,
            self.gaabb_mo,
            self.gbbaa_mo,
            self.kappa_no_activeactive_idx_dagger,
            self.kappa_no_activeactive_idx_dagger,
            self.num_inactive_orbs,
            self.num_active_orbs,
            self.rdm1aa,
            self.rdm1bb,
            self.rdm2aaaa,
            self.rdm2bbbb,
            self.rdm2aabb,
            self.rdm2bbaa,
        )

    # Print response orbital metric sigma for test
    @property
    def orbital_response_metric_sigma_unrestricted(self) -> np.ndarray:
        return get_orbital_response_metric_sigma_unrestricted(
            self.kappa_no_activeactive_idx,
            self.num_inactive_orbs,
            self.num_active_orbs,
            self.rdm1aa,
            self.rdm1bb,
        )

    def manual_metric_sigma_unrestricted(
        wf: UnrestrictedWaveFunctionUPS,
    ) -> np.ndarray:
        sigma = np.zeros((2 * len(wf.kappa_no_activeactive_idx), 2 * len(wf.kappa_no_activeactive_idx)))
        for idx1, (q, p) in enumerate(wf.kappa_no_activeactive_idx):
            for idx2, (m, n) in enumerate(wf.kappa_no_activeactive_idx):
                q_qp_a = a_op(p, "alpha", True) * a_op(q, "alpha", False)
                q_mn_a = a_op(m, "alpha", True) * a_op(n, "alpha", False)
                q_qp_b = a_op(p, "beta", True) * a_op(q, "beta", False)
                q_mn_b = a_op(m, "beta", True) * a_op(n, "beta", False)
                aa = expectation_value(
                    wf.ci_coeffs,
                    [q_qp_a * q_mn_a],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                aa -= expectation_value(
                    wf.ci_coeffs,
                    [q_mn_a * q_qp_a],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                bb = expectation_value(
                    wf.ci_coeffs,
                    [q_qp_b * q_mn_b],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                bb -= expectation_value(
                    wf.ci_coeffs,
                    [q_mn_b * q_qp_b],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                sigma[idx1 * 2, idx2 * 2] = aa
                sigma[idx1 * 2 + 1, idx2 * 2 + 1] = bb
        return sigma

    def manual_hessian_unrestricted_A(
        wf: UnrestrictedWaveFunctionUPS,
    ) -> np.ndarray:
        h = unrestricted_hamiltonian_full_space(
            wf.haa_mo, wf.hbb_mo, wf.gaaaa_mo, wf.gbbbb_mo, wf.gaabb_mo, wf.gbbaa_mo, wf.num_orbs
        )
        A_block = np.zeros((2 * len(wf.kappa_no_activeactive_idx), 2 * len(wf.kappa_no_activeactive_idx)))
        for idx1, (u, t) in enumerate(wf.kappa_no_activeactive_idx):
            for idx2, (m, n) in enumerate(wf.kappa_no_activeactive_idx):
                E_tu_a = a_op(t, "alpha", True) * a_op(u, "alpha", False)
                E_mn_a = a_op(m, "alpha", True) * a_op(n, "alpha", False)
                E_tu_b = a_op(t, "beta", True) * a_op(u, "beta", False)
                E_mn_b = a_op(m, "beta", True) * a_op(n, "beta", False)
                aa = expectation_value(
                    wf.ci_coeffs,
                    [E_tu_a * h * E_mn_a],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                aa -= expectation_value(
                    wf.ci_coeffs,
                    [E_tu_a * E_mn_a * h],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                aa -= expectation_value(
                    wf.ci_coeffs,
                    [h * E_mn_a * E_tu_a],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                aa += expectation_value(
                    wf.ci_coeffs,
                    [E_mn_a * h * E_tu_a],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                ba = expectation_value(
                    wf.ci_coeffs,
                    [E_tu_b * h * E_mn_a],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                ba -= expectation_value(
                    wf.ci_coeffs,
                    [E_tu_b * E_mn_a * h],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                ba -= expectation_value(
                    wf.ci_coeffs,
                    [h * E_mn_a * E_tu_b],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                ba += expectation_value(
                    wf.ci_coeffs,
                    [E_mn_a * h * E_tu_b],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                ab = expectation_value(
                    wf.ci_coeffs,
                    [E_tu_a * h * E_mn_b],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                ab -= expectation_value(
                    wf.ci_coeffs,
                    [E_tu_a * E_mn_b * h],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                ab -= expectation_value(
                    wf.ci_coeffs,
                    [h * E_mn_b * E_tu_a],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                ab += expectation_value(
                    wf.ci_coeffs,
                    [E_mn_b * h * E_tu_a],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                bb = expectation_value(
                    wf.ci_coeffs,
                    [E_tu_b * h * E_mn_b],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                bb -= expectation_value(
                    wf.ci_coeffs,
                    [E_tu_b * E_mn_b * h],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                bb -= expectation_value(
                    wf.ci_coeffs,
                    [h * E_mn_b * E_tu_b],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                bb += expectation_value(
                    wf.ci_coeffs,
                    [E_mn_b * h * E_tu_b],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                A_block[idx1 * 2, idx2 * 2] = aa
                A_block[idx1 * 2, idx2 * 2 + 1] = ab
                A_block[idx1 * 2 + 1, idx2 * 2] = ba
                A_block[idx1 * 2 + 1, idx2 * 2 + 1] = bb
        return A_block

    def manual_hessian_unrestricted_B(
        wf: UnrestrictedWaveFunctionUPS,
    ) -> np.ndarray:
        h = unrestricted_hamiltonian_full_space(
            wf.haa_mo, wf.hbb_mo, wf.gaaaa_mo, wf.gbbbb_mo, wf.gaabb_mo, wf.gbbaa_mo, wf.num_orbs
        )
        B_block = np.zeros((2 * len(wf.kappa_no_activeactive_idx), 2 * len(wf.kappa_no_activeactive_idx)))
        for idx1, (u, t) in enumerate(wf.kappa_no_activeactive_idx):
            for idx2, (n, m) in enumerate(wf.kappa_no_activeactive_idx):
                E_tu_a = a_op(t, "alpha", True) * a_op(u, "alpha", False)
                E_mn_a = a_op(m, "alpha", True) * a_op(n, "alpha", False)
                E_tu_b = a_op(t, "beta", True) * a_op(u, "beta", False)
                E_mn_b = a_op(m, "beta", True) * a_op(n, "beta", False)
                aa = expectation_value(
                    wf.ci_coeffs,
                    [E_tu_a * h * E_mn_a],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                aa -= expectation_value(
                    wf.ci_coeffs,
                    [E_tu_a * E_mn_a * h],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                aa -= expectation_value(
                    wf.ci_coeffs,
                    [h * E_mn_a * E_tu_a],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                aa += expectation_value(
                    wf.ci_coeffs,
                    [E_mn_a * h * E_tu_a],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                ba = expectation_value(
                    wf.ci_coeffs,
                    [E_tu_b * h * E_mn_a],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                ba -= expectation_value(
                    wf.ci_coeffs,
                    [E_tu_b * E_mn_a * h],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                ba -= expectation_value(
                    wf.ci_coeffs,
                    [h * E_mn_a * E_tu_b],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                ba += expectation_value(
                    wf.ci_coeffs,
                    [E_mn_a * h * E_tu_b],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                ab = expectation_value(
                    wf.ci_coeffs,
                    [E_tu_a * h * E_mn_b],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                ab -= expectation_value(
                    wf.ci_coeffs,
                    [E_tu_a * E_mn_b * h],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                ab -= expectation_value(
                    wf.ci_coeffs,
                    [h * E_mn_b * E_tu_a],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                ab += expectation_value(
                    wf.ci_coeffs,
                    [E_mn_b * h * E_tu_a],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                bb = expectation_value(
                    wf.ci_coeffs,
                    [E_tu_b * h * E_mn_b],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                bb -= expectation_value(
                    wf.ci_coeffs,
                    [E_tu_b * E_mn_b * h],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                bb -= expectation_value(
                    wf.ci_coeffs,
                    [h * E_mn_b * E_tu_b],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                bb += expectation_value(
                    wf.ci_coeffs,
                    [E_mn_b * h * E_tu_b],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                B_block[idx1 * 2, idx2 * 2] = aa
                B_block[idx1 * 2, idx2 * 2 + 1] = ab
                B_block[idx1 * 2 + 1, idx2 * 2] = ba
                B_block[idx1 * 2 + 1, idx2 * 2 + 1] = bb
        return B_block

    def manual_gradient(
        wf: UnrestrictedWaveFunctionUPS,
    ) -> np.ndarray:
        h = unrestricted_hamiltonian_full_space(
            wf.haa_mo, wf.hbb_mo, wf.gaaaa_mo, wf.gbbbb_mo, wf.gaabb_mo, wf.gbbaa_mo, wf.num_orbs
        )
        gradient = np.zeros(2 * len(wf.kappa_idx))
        for idx, (m, n) in enumerate(wf.kappa_idx):
            for p in range(wf.num_inactive_orbs + wf.num_active_orbs):
                alpha = expectation_value(
                    wf.ci_coeffs,
                    [a_op(m, "alpha", True) * a_op(n, "alpha", False) * h],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                alpha -= expectation_value(
                    wf.ci_coeffs,
                    [a_op(n, "alpha", True) * a_op(m, "alpha", False) * h],
                    wf.ci_coeffs,
                    wf.ci_info,
                )

                alpha -= expectation_value(
                    wf.ci_coeffs,
                    [h * (a_op(m, "alpha", True) * a_op(n, "alpha", False))],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                alpha += expectation_value(
                    wf.ci_coeffs,
                    [h * (a_op(n, "alpha", True) * a_op(m, "alpha", False))],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                beta = expectation_value(
                    wf.ci_coeffs,
                    [a_op(m, "beta", True) * a_op(n, "beta", False) * h],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                beta -= expectation_value(
                    wf.ci_coeffs,
                    [a_op(n, "beta", True) * a_op(m, "beta", False) * h],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                beta -= expectation_value(
                    wf.ci_coeffs,
                    [h * (a_op(m, "beta", True) * a_op(n, "beta", False))],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                beta += expectation_value(
                    wf.ci_coeffs,
                    [h * (a_op(n, "beta", True) * a_op(m, "beta", False))],
                    wf.ci_coeffs,
                    wf.ci_info,
                )
                gradient[idx] = alpha
                gradient[idx + len(wf.kappa_idx)] = beta
        return gradient
