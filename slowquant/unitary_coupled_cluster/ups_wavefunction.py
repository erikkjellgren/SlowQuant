from __future__ import annotations

import time
from collections.abc import Sequence
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
from slowquant.unitary_coupled_cluster.operators import (
    Epq,
    epqrs,
    hamiltonian_wf_opt,
    hamiltonian_0i_0a,
)
from slowquant.unitary_coupled_cluster.optimizers import Optimizers
from slowquant.unitary_coupled_cluster.util import UpsStructure
from slowquant.unitary_coupled_cluster.fock_matrix import (
    build_fock_active,
    build_fock_inactive,
    build_fock_matrix,
    get_electronic_energy,
    get_orbital_gradient,
)
from slowquant.unitary_coupled_cluster.density_matrix import get_orbital_gradient as get_orbital_gradient_den

class WaveFunctionUPS:
    def __init__(
        self,
        cas: Sequence[int],
        mo_coeffs: np.ndarray,
        integral_generator: SlowQuant | pyscf.gto.mole.Mole,
        ansatz: str,
        ansatz_options: dict[str, Any] | None = None,
        include_active_kappa: bool = False,
    ) -> None:
        """Initialize for UPS wave function.

        Args:
            cas: CAS(num_active_elec, num_active_orbs),
                 orbitals are counted in spatial basis.
            mo_coeffs: Initial orbital coefficients.
            integral_generator: Integral generator object.
            ansatz: Name of ansatz.
            ansatz_options: Ansatz options.
            include_active_kappa: Include active-active orbital rotations.
        """
        if ansatz_options is None:
            ansatz_options = {}
        if len(cas) != 2:
            raise ValueError(f"cas must have two elements, got {len(cas)} elements.")
        # Init stuff
        self.int_gen = IntegralManager(integral_generator)
        self._c_mo = mo_coeffs
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
        self.num_spin_orbs = 2 * len(self.int_gen.kinetic_energy)
        self.num_orbs = len(self.int_gen.kinetic_energy)
        self.num_active_elec = 0
        self.num_active_spin_orbs = 0
        self.num_inactive_spin_orbs = 0
        self.num_virtual_spin_orbs = 0
        self._rdm1 = None
        self._rdm2 = None
        self._rdm3 = None
        self._rdm4 = None
        self._fock_mat = None
        self._fock_mat_inactive = None
        self._fock_mat_inactive_ao = None
        self._fock_mat_active = None
        self._fock_mat_active_ao = None
        self._DI_ao = None
        self._DA_ao = None
        self._h_ii = None
        self._h_vw = None
        self._g_iijj = None
        self._g_ijji = None
        self._g_iivw = None
        self._g_iviw = None
        self._g_vwxy = None
        self._g_Pvwx = None
        self._h_mo = None
        self._g_mo = None
        self._H_wf_opt = None
        self._energy_elec: float | None = None
        self.ansatz_options = ansatz_options
        self.num_energy_evals = 0
        # Construct spin orbital spaces and indices
        active_space = []
        orbital_counter = 0
        num_elec = self.int_gen.num_elec
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
        kappa_idx = []
        kappa_no_activeactive_idx = []
        kappa_no_activeactive_idx_dagger = []
        kappa_redundant_idx = []
        self._kappa_old = []
        # kappa can be optimized in spatial basis
        # Loop over all q>p orb combinations and find redundant kappas
        for p in range(0, self.num_orbs):
            for q in range(p + 1, self.num_orbs):
                # find redundant kappas
                if p in self.inactive_idx and q in self.inactive_idx:
                    kappa_redundant_idx.append((p, q))
                    continue
                if p in self.virtual_idx and q in self.virtual_idx:
                    kappa_redundant_idx.append((p, q))
                    continue
                if not include_active_kappa:
                    if p in self.active_idx and q in self.active_idx:
                        kappa_redundant_idx.append((p, q))
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
        self.kappa_redundant_idx = np.array(kappa_redundant_idx, dtype=int)
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
        if np.max(np.abs(np.array(self._kappa) - np.array(k))) > 10**-28:
            self._h_mo = None
            self._g_mo = None
            self._h_ii = None
            self._h_vw = None
            self._g_iijj = None
            self._g_ijji = None
            self._g_iivw = None
            self._g_iviw = None
            self._g_vwxy = None
            self._g_Pvwx = None
            self._fock_mat = None
            self._fock_mat_inactive = None
            self._fock_mat_inactive_ao = None
            self._fock_mat_active = None
            self._fock_mat_active_ao = None
            self._DI_ao = None
            self._DA_ao = None
            self._H_wf_opt = None
            self._energy_elec = None
            if isinstance(k, np.ndarray):
                k = k.tolist()
            self._kappa = k.copy()

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
        if np.max(np.abs(np.array(self._thetas) - np.array(theta_vals))) > 10**-28:
            self._rdm1 = None
            self._rdm2 = None
            self._rdm3 = None
            self._rdm4 = None
            self._fock_mat = None
            self._fock_mat_active = None
            self._fock_mat_active_ao = None
            self._DA_ao = None
            self._energy_elec = None
            if isinstance(theta_vals, np.ndarray):
                theta_vals = theta_vals.tolist()
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
            if np.max(np.abs(np.array(self.kappa) - np.array(self._kappa_old))) > 10**-28:
                for kappa_val, kappa_old, (p, q) in zip(self.kappa, self._kappa_old, self.kappa_idx):
                    kappa_mat[p, q] = kappa_val - kappa_old
                    kappa_mat[q, p] = -(kappa_val - kappa_old)
        # Apply orbital rotation unitary to MO coefficients
        return np.matmul(self._c_mo, scipy.linalg.expm(-kappa_mat))

    def _move_cep(self) -> None:
        self._c_mo = self.c_mo
        self._kappa_old = self.kappa

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
    def h_ii(self) -> np.ndarray:
        if self._h_ii is None:
            self._h_ii = np.einsum("Pi,Qi,PQ->i", self.c_mo[:, : self.num_inactive_orbs], self.c_mo[:, : self.num_inactive_orbs], self.int_gen.h_ao, optimize=True)
        return self._h_ii

    @property
    def h_vw(self) -> np.ndarray:
        if self._h_vw is None:
            self._h_vw = np.einsum("Pv,Qw,PQ->vw", self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs], self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs], self.int_gen.h_ao, optimize=True)
        return self._h_vw

    @property
    def g_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._g_mo is None:
            self._g_mo = two_electron_integral_transform(self.c_mo, self.int_gen.g_ao)
        return self._g_mo

    @property
    def g_iijj(self) -> np.ndarray:
        if self._g_iijj is None:
            self._g_iijj = np.einsum("Pi,Qi,Rj,Sj,PQRS->ij", self.c_mo[:, : self.num_inactive_orbs], self.c_mo[:, : self.num_inactive_orbs], self.c_mo[:, : self.num_inactive_orbs], self.c_mo[:, : self.num_inactive_orbs], self.int_gen.g_ao, optimize=True)
        return self._g_iijj

    @property
    def g_ijji(self) -> np.ndarray:
        if self._g_ijji is None:
            self._g_ijji = np.einsum("Pi,Qj,Rj,Si,PQRS->ij", self.c_mo[:, : self.num_inactive_orbs], self.c_mo[:, : self.num_inactive_orbs], self.c_mo[:, : self.num_inactive_orbs], self.c_mo[:, : self.num_inactive_orbs], self.int_gen.g_ao, optimize=True)
        return self._g_ijji

    @property
    def g_iivw(self) -> np.ndarray:
        if self._g_iivw is None:
            self._g_iivw = np.einsum("Pi,Qi,Rv,Sw,PQRS->ivw", self.c_mo[:, : self.num_inactive_orbs], self.c_mo[:, : self.num_inactive_orbs], self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs], self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs], self.int_gen.g_ao, optimize=True)
        return self._g_iivw

    @property
    def g_iviw(self) -> np.ndarray:
        if self._g_iviw is None:
            self._g_iviw = np.einsum("Pi,Qv,Ri,Sw,PQRS->ivw", self.c_mo[:, : self.num_inactive_orbs], self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs], self.c_mo[:, : self.num_inactive_orbs], self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs], self.int_gen.g_ao, optimize=True)
        return self._g_iviw

    @property
    def g_Pvwx(self) -> np.ndarray:
        if self._g_Pvwx is None:
            n_ao = self.num_orbs
            n_in = self.num_inactive_orbs
            n_act = self.num_active_orbs
            C_act = self.c_mo[:, n_in : n_in + n_act]
            # Transform S -> y (Index 3), make g_PQRy
            tmp = self.int_gen.g_ao.reshape(-1, n_ao) @ C_act  # (P*Q*R, y)
            tmp = tmp.reshape(n_ao, n_ao, n_ao, n_act)
            # Transform R -> x (Index 2), make g_PQyx
            tmp = tmp.transpose(0, 1, 3, 2).reshape(-1, n_ao) @ C_act  # (P*Q*y, x)
            tmp = tmp.reshape(n_ao, n_ao, n_act, n_act)
            # Transform Q -> w (Index 1), make g_Pyxw
            tmp = tmp.transpose(0, 2, 3, 1).reshape(-1, n_ao) @ C_act  # (P*x*y, w)
            tmp = tmp.reshape(n_ao, n_act, n_act, n_act)
            # Resulting mixed integral: g_Pwxy (renamed to g_Pvwx)
            self._g_Pvwx = tmp.transpose(0, 3, 2, 1)
            if self._g_vwxy is None:
                # Also making g_vwxy
                # Transform P -> v (Index 0), make g_yxwv
                tmp = tmp.transpose(1, 2, 3, 0).reshape(-1, n_ao) @ C_act  # (w*x*y, v)
                tmp = tmp.reshape(n_act, n_act, n_act, n_act)
                # Resulting integral: g_vwxy
                self._g_vwxy = tmp.reshape(n_act, n_act, n_act, n_act).transpose(3, 2, 1, 0)
        return self._g_Pvwx

    @property
    def g_vwxy(self) -> np.ndarray:
        if self._g_vwxy is None:
            n_ao = self.num_orbs
            n_in = self.num_inactive_orbs
            n_act = self.num_active_orbs
            C_act = self.c_mo[:, n_in : n_in + n_act]
            # Transform S -> y (Index 3), make g_PQRy
            tmp = self.int_gen.g_ao.reshape(-1, n_ao) @ C_act  # (P*Q*R, y)
            tmp = tmp.reshape(n_ao, n_ao, n_ao, n_act)
            # Transform R -> x (Index 2), make g_PQyx
            tmp = tmp.transpose(0, 1, 3, 2).reshape(-1, n_ao) @ C_act  # (P*Q*y, x)
            tmp = tmp.reshape(n_ao, n_ao, n_act, n_act)
            # Transform Q -> w (Index 1), make g_Pyxw
            tmp = tmp.transpose(0, 2, 3, 1).reshape(-1, n_ao) @ C_act  # (P*x*y, w)
            tmp = tmp.reshape(n_ao, n_act, n_act, n_act)
            # Resulting mixed integral: g_Pwxy (renamed to g_Pvwx)
            self._g_Pvwx = tmp.transpose(0, 3, 2, 1)
            # Also making g_vwxy
            # Transform P -> v (Index 0), make g_yxwv
            tmp = tmp.transpose(1, 2, 3, 0).reshape(-1, n_ao) @ C_act  # (w*x*y, v)
            tmp = tmp.reshape(n_act, n_act, n_act, n_act)
            # Resulting integral: g_vwxy
            self._g_vwxy = tmp.reshape(n_act, n_act, n_act, n_act).transpose(3, 2, 1, 0)
        return self._g_vwxy

    @property
    def fock_mat_inactive(self) -> np.ndarray:
        if self._fock_mat_inactive is None:
            self._fock_mat_inactive, self._fock_mat_inactive_ao = build_fock_inactive(
                self.int_gen.h_ao, self.int_gen.g_ao, self.c_mo, self.DI_ao
            )
        return self._fock_mat_inactive

    @property
    def fock_mat_inactive_ao(self) -> np.ndarray:
        if self._fock_mat_inactive_ao is None:
            self._fock_mat_inactive, self._fock_mat_inactive_ao = build_fock_inactive(
                self.int_gen.h_ao, self.int_gen.g_ao, self.c_mo, self.DI_ao
            )
        return self._fock_mat_inactive_ao

    @property
    def fock_mat_active(self) -> np.ndarray:
        if self._fock_mat_active is None:
            self._fock_mat_active, self._fock_mat_active_ao = build_fock_active(
                self.int_gen.g_ao,
                self.c_mo,
                self.DA_ao,
            )
        return self._fock_mat_active

    @property
    def fock_mat_active_ao(self) -> np.ndarray:
        if self._fock_mat_active_ao is None:
            self._fock_mat_active, self._fock_mat_active_ao = build_fock_active(
                self.int_gen.g_ao,
                self.c_mo,
                self.DA_ao,
            )
        return self._fock_mat_active_ao

    @property
    def fock_mat(self) -> np.ndarray:
        if self._fock_mat is None:
            self._fock_mat = build_fock_matrix(
                self.g_Pvwx,
                self.c_mo,
                self.fock_mat_inactive,
                self.fock_mat_active,
                self.rdm1,
                self.rdm2,
                self.num_inactive_orbs,
                self.num_active_orbs,
                self.num_virtual_orbs,
            )
        return self._fock_mat

    @property
    def DI_ao(self) -> np.ndarray:
        if self._DI_ao is None:
            self._DI_ao = 2 * np.einsum(
                "Pi,Qi->PQ", self.c_mo[:, : self.num_inactive_orbs], self.c_mo[:, : self.num_inactive_orbs]
            )
        return self._DI_ao

    @property
    def DA_ao(self) -> np.ndarray:
        if self._DA_ao is None:
            self._DA_ao = np.einsum(
                "vw,Pv,Qw->PQ",
                self.rdm1,
                self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs],
                self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs],
            )
        return self._DA_ao

    @property
    def H_wf_opt(self) -> FermionicOperator:
        if self._H_wf_opt is None:
            self._H_wf_opt = hamiltonian_wf_opt(
                        self.h_ii,
                        self.h_vw,
                        self.g_iijj,
                        self.g_ijji,
                        self.g_iivw,
                        self.g_iviw,
                        self.g_vwxy,
                        self.num_inactive_orbs,
                        self.num_active_orbs
            )
        return self._H_wf_opt

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
                [self.H_wf_opt],
                self.ci_coeffs,
                self.ci_info,
            )
        return self._energy_elec

    def _get_hamiltonian(self, qiskit_form: bool = False) -> FermionicOperator | dict[str, float]:
        """Return electronic Hamiltonian as FermionicOperator.

        Returns:
            FermionicOperator.
        """
        H = self.H_wf_opt
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
            hessp_theta = partial(
                self._calc_hessian_vector_product_optimization,
                theta_optimization=True,
                kappa_optimization=False,
            )
            optimizer = Optimizers(
                energy_theta,
                optimizer_name,
                grad=gradient_theta,
                hessp=hessp_theta,
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
                hessp_oo = partial(
                    self._calc_hessian_vector_product_optimization,
                    theta_optimization=False,
                    kappa_optimization=True,
                )
                optimizer = Optimizers(
                    energy_oo,
                    "l-bfgs-b",
                    grad=gradient_oo,
                    hessp=hessp_oo,
                    maxiter=maxiter,
                    tol=tol,
                    is_silent=is_silent_subiterations,
                    energy_eval_callback=lambda: self.num_energy_evals,
                    cep_move=self._move_cep,
                )
                self._old_opt_parameters = np.zeros(len(self.kappa_idx)) + 10**20
                self._E_opt_old = 0.0
                res = optimizer.minimize([0.0] * len(self.kappa_idx))
                self._move_cep()
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
                hessp = partial(
                    self._calc_hessian_vector_product_optimization,
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
                hessp = partial(
                    self._calc_hessian_vector_product_optimization,
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
            hessp = partial(
                self._calc_hessian_vector_product_optimization,
                theta_optimization=True,
                kappa_optimization=False,
            )
        if orbital_optimization:
            cep_move_fun = self._move_cep
            if len(self.thetas) > 0:
                parameters = self.kappa + self.thetas
            else:
                parameters = self.kappa
        else:
            cep_move_fun = None
            parameters = self.thetas
        optimizer = Optimizers(
            energy,
            optimizer_name,
            grad=gradient,
            hessp=hessp,
            maxiter=maxiter,
            tol=tol,
            energy_eval_callback=lambda: self.num_energy_evals,
            cep_move=cep_move_fun,
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
            self._move_cep()
            self.thetas = res.x[len(self.kappa) :].tolist()
            for i in range(len(self.kappa)):
                self._kappa[i] = 0.0
                self._kappa_old[i] = 0.0
        else:
            self.thetas = res.x.tolist()
        self._energy_elec = res.fun

    def _theta_gradient(self, operator: FermionicOperator) -> np.ndarray:
        gradient = np.zeros(len(self.thetas))
        # Reference bra state (no differentiations)
        bra_vec = propagate_state(
            [operator],
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
            gradient[i] += 2 * np.matmul(bra_vec, ket_vec_tmp)
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
            E = get_electronic_energy(
                self.int_gen.h_ao,
                self.g_vwxy,
                self.fock_mat_inactive_ao,
                self.fock_mat_inactive,
                self.DI_ao,
                self.rdm1,
                self.rdm2,
                self.num_inactive_orbs,
                self.num_active_orbs,
            )
        else:
            E = expectation_value(
                self.ci_coeffs,
                [self.H_wf_opt],
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
            gradient[:num_kappa] = get_orbital_gradient(self.kappa_idx, self.fock_mat)
        if theta_optimization:
            gradient[num_kappa:] = self._theta_gradient(self.H_wf_opt)
        return gradient

    def _calc_hessian_vector_product_optimization2(
        self,
        parameters: list[float],
        trial_vec: list[float],
        theta_optimization: bool,
        kappa_optimization: bool,
    ) -> np.ndarray:
        h = 10**-4
        parameters_plus = (np.array(parameters) + h * np.array(trial_vec)).tolist()
        g_plus = self._calc_gradient_optimization(parameters_plus, theta_optimization, kappa_optimization)
        parameters_minus = (np.array(parameters) - h * np.array(trial_vec)).tolist()
        g_minus = self._calc_gradient_optimization(parameters_minus, theta_optimization, kappa_optimization)
        num_kappa = 0
        if kappa_optimization:
            num_kappa = len(self.kappa_idx)
            self.kappa = parameters[:num_kappa]
        if theta_optimization:
            self.thetas = parameters[num_kappa:]
        return (g_plus - g_minus) / (2 * h)

    def _calc_hessian_vector_product_optimization(
        self,
        parameters: list[float],
        trial_vec: list[float],
        theta_optimization: bool,
        kappa_optimization: bool,
    ) -> np.ndarray:
        """Calculate Hessian vector product.

        Args:
            parameters: Ansatz and orbital rotation parameters.
            trial_vec: Trial vector.
            theta_optimization: If used in theta optimization.
            kappa_optimization: If used in kappa optimization.

        Returns:
            Hessian vector product.
        """
        hvp = np.zeros(len(parameters))
        num_kappa = 0
        if kappa_optimization:
            num_kappa = len(self.kappa_idx)
            self.kappa = parameters[:num_kappa]
        if theta_optimization:
            self.thetas = parameters[num_kappa:]
        if kappa_optimization:
            b_kappa = trial_vec[:num_kappa]
            kappa_mat = np.zeros_like(self._c_mo)
            for kappa_val, (p, q) in zip(b_kappa, self.kappa_idx):
                kappa_mat[p, q] = kappa_val
                kappa_mat[q, p] = -kappa_val
            h_k = np.einsum("po,oq->pq", kappa_mat, self.h_mo)
            h_k += np.einsum("qo,po->pq", kappa_mat, self.h_mo)
            g_k = np.einsum("po,oqrs->pqrs", kappa_mat, self.g_mo)
            g_k += np.einsum("qo,pors->pqrs", kappa_mat, self.g_mo)
            g_k += np.einsum("ro,pqos->pqrs", kappa_mat, self.g_mo)
            g_k += np.einsum("so,pqro->pqrs", kappa_mat, self.g_mo)
            grad_kappa = get_orbital_gradient(self.kappa_idx, self.fock_mat)
            grad_kappa_mat = np.zeros_like(self._c_mo)
            for grad, (p, q) in zip(grad_kappa, self.kappa_idx):
                grad_kappa_mat[p, q] = grad
                grad_kappa_mat[q, p] = -grad
            Ek_mat = np.matmul(grad_kappa_mat, kappa_mat) - np.matmul(kappa_mat, grad_kappa_mat)
            grad_k = get_orbital_gradient_den(
                h_k,
                g_k,
                self.kappa_idx,
                self.num_inactive_orbs,
                self.num_active_orbs,
                self.rdm1,
                self.rdm2,
            )
            # E_{kappa,kappa}b_kappa contribution to sigma_kappa
            for i, (p, q) in enumerate(self.kappa_idx):
                hvp[i] += grad_k[i] + Ek_mat[p, q]
            if theta_optimization:
                H_k = hamiltonian_0i_0a(
                    h_k,
                    g_k,
                    self.num_inactive_orbs,
                    self.num_active_orbs,
                )
                # E_{theta,kappa}b_kappa contribution to sigma_theta
                hvp[num_kappa:] += self._theta_gradient(H_k)
        if theta_optimization:
            # E_{theta,theta}b_theta contribution to sigma_theta
            b_theta = trial_vec[num_kappa:]
            psi_vec = np.copy(self.csf_coeffs)
            phi_vec = np.zeros_like(psi_vec)
            for i in range(len(self.thetas)):
                # $|\psi_i\rangle = U_i(\theta_i) |\psi_{i-1}\rangle$
                psi_vec = propagate_unitary(psi_vec, i, self.ci_info, self.thetas, self.ups_layout)
                # $$|\Phi_i\rangle = U_i(\theta_i) |\Phi_{i-1}\rangle + v_i \frac{\partial U_i}{\partial \theta_i} |\psi_{i-1}\rangle$$
                phi_vec = propagate_unitary(phi_vec, i, self.ci_info, self.thetas, self.ups_layout)
                # Remember gradient action does not apply U, but only the operator part, hence using psi_{i} instead of psi_{i-1}
                phi_vec += b_theta[i] * get_grad_action(psi_vec, i, self.ci_info, self.ups_layout)

            if kappa_optimization:
                # E_{kappa,theta}b_theta contribution to sigma_kappa
                tdm1 = np.zeros((self.num_active_orbs, self.num_active_orbs))
                for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                    p_idx = p - self.num_inactive_orbs
                    for q in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                        q_idx = q - self.num_inactive_orbs
                        op = Epq(p,q)
                        Ephi_vec = propagate_state([op], phi_vec, self.ci_info, wf_struct=self.ups_layout)
                        Epsi_vec = propagate_state([op], psi_vec, self.ci_info, wf_struct=self.ups_layout)
                        tdm1[p_idx, q_idx] = psi_vec@Ephi_vec + phi_vec@Epsi_vec
                        
                tdm2 = np.zeros((self.num_active_orbs, self.num_active_orbs, self.num_active_orbs, self.num_active_orbs))
                pairs = []
                for p in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                    for q in range(self.num_inactive_orbs, self.num_inactive_orbs + self.num_active_orbs):
                        pairs.append((p,q))
                for i in range(len(pairs)):
                    p,q = pairs[i]
                    p_idx = p - self.num_inactive_orbs
                    q_idx = q - self.num_inactive_orbs
                    for j in range(i, len(pairs)):
                        r,s = pairs[j]
                        r_idx = r - self.num_inactive_orbs
                        s_idx = s - self.num_inactive_orbs
                        op = epqrs(p,q,r,s)
                        ephi_vec = propagate_state([op], phi_vec, self.ci_info, self.thetas, self.ups_layout)
                        epsi_vec = propagate_state([op], psi_vec, self.ci_info, self.thetas, self.ups_layout)
                        val = psi_vec@ephi_vec + phi_vec@epsi_vec
                        tdm2[p_idx, q_idx, r_idx, s_idx] = val
                        tdm2[r_idx, s_idx, p_idx, q_idx] = val

                DA_ao_trans = np.einsum(
                    "vw,Pv,Qw->PQ",
                    tdm1,
                    self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs],
                    self.c_mo[:, self.num_inactive_orbs : self.num_inactive_orbs + self.num_active_orbs],
                )
                fock_mat_active_trans, _ = build_fock_active(
                    self.int_gen.g_ao,
                    self.c_mo,
                    DA_ao_trans,
                )
                fock_trans = build_fock_matrix(
                        self.g_Pvwx,
                        self.c_mo,
                        self.fock_mat_inactive,
                        fock_mat_active_trans,
                        tdm1,
                        tdm2,
                        self.num_inactive_orbs,
                        self.num_active_orbs,
                        self.num_virtual_orbs,
                        do_resp = True,
                )
                hvp[:num_kappa] += get_orbital_gradient(self.kappa_idx, fock_trans)

            lambda_vec = propagate_state([self.H_wf_opt], psi_vec, self.ci_info, self.thetas, self.ups_layout)
            mu_vec = propagate_state([self.H_wf_opt], phi_vec, self.ci_info, self.thetas, self.ups_layout)
            for i in range(len(self.thetas) - 1, -1, -1):
                # 1. Get gradient actions at the current state
                dpsi_vec = get_grad_action(psi_vec, i, self.ci_info, self.ups_layout)
                d2psi_vec = get_grad_action(dpsi_vec, i, self.ci_info, self.ups_layout)

                # 4. Calculate actions on the unwound Tangent Ket
                dphi_past = get_grad_action(phi_vec - b_theta[i] * dpsi_vec, i, self.ci_info, self.ups_layout)

                # 5. Assemble HvP
                term_overlap_and_future = np.vdot(mu_vec, dpsi_vec)
                term_past_curve = np.vdot(lambda_vec, dphi_past)
                term_diag_curve = b_theta[i] * np.vdot(lambda_vec, d2psi_vec)

                hvp[i + num_kappa] += 2 * np.real(term_overlap_and_future + term_past_curve + term_diag_curve)

                lambda_vec = propagate_unitary(
                    lambda_vec, i, self.ci_info, self.thetas, self.ups_layout, dagger=True
                )
                mu_vec = propagate_unitary(mu_vec, i, self.ci_info, self.thetas, self.ups_layout, dagger=True)
                psi_vec = propagate_unitary(
                    psi_vec, i, self.ci_info, self.thetas, self.ups_layout, dagger=True
                )
                phi_vec = propagate_unitary(
                    phi_vec, i, self.ci_info, self.thetas, self.ups_layout, dagger=True
                )

            self.num_energy_evals += (
                2 * np.sum(list(self.ups_layout.grad_param_R.values()))
            ) ** 2  # Count energy measurements for theta theta Hessian
        return hvp

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

        bra_vec = propagate_state_SA([self.H_wf_opt], state_vecs, self.ci_info, thetas_local, self.ups_layout)

        energies = []
        for bra, ket in zip(bra_vec, state_vecs):
            energies.append(bra @ ket)
        self.num_energy_evals += len(energies)

        return energies
