from __future__ import annotations

import time
from functools import partial
from typing import Any

import numpy as np
import scipy
import pyscf

from slowquant.SlowQuant import SlowQuant

from slowquant.unitary_coupled_cluster.DHF_integral_manager import IntegralManager
from slowquant.unitary_coupled_cluster.DHF_integral_direct import DHFDirect, DirectState, orbital_gradient
from slowquant.unitary_coupled_cluster.ci_sparse_DHF import ActiveCI

from slowquant.molecularintegrals.integralfunctions import (
    generalized_one_electron_transform,
    generalized_two_electron_transform,
    DHF_one_electron_transform,
    DHF_two_electron_transform,
    generalized_1e_transform_spin_separated,
)
from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing_generalized, get_indexing_DHF
from slowquant.unitary_coupled_cluster.generalized_density_matrix_DHF import (
    get_electronic_energy_generalized,
    get_orbital_gradient_expvalue_real_imag,
    get_orbital_gradient_generalized_real_imag,
    get_S2,
    get_Sz,
    get_nr_elec,
    get_S2_old,
    get_mu0,
    get_e0,
)
from slowquant.unitary_coupled_cluster.generalized_operators import (
    a_op_spin,
    DHF_hamiltonian_full_space,
)
from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import (
    generalized_construct_ups_state, generalized_construct_ups_state_test_anna,
    generalized_construct_ups_state_test_erik,
    generalized_expectation_value,
    generalized_expectation_value_energy,
    generalized_get_grad_action, generalized_get_grad_action_test_anna,
    generalized_propagate_state,
    generalized_propagate_unitary, generalized_propagate_unitary_test_anna,
)
from slowquant.unitary_coupled_cluster.operators import G1, G2
from slowquant.unitary_coupled_cluster.generalized_operators import DHF_hamiltonian_0i_0a
from slowquant.unitary_coupled_cluster.optimizers import Optimizers
from slowquant.unitary_coupled_cluster.util import ( 
    UpsStructure,
    iterate_t1,
    iterate_t1_generalized,
    iterate_t2,
    iterate_t2_generalized,
)

def Trev(C):
        """
        Time reversal for PySCF 4c ordering:
        [large 2c spinor block, small 2c spinor block]
        """

        nao, nmo = C.shape

        if nao % 2 != 0:
            raise ValueError("Expected even number of 4c AO functions")

        n2 = nao // 2

        out = np.zeros_like(C)

        # large component block
        L = C[:n2].conj()
        out[:n2:2, :] = -L[1::2, :]
        out[1:n2:2, :] =  L[0::2, :]

        # small component block
        S = C[n2:].conj()
        out[n2::2, :] = -S[1::2, :]
        out[n2+1::2, :] = S[0::2, :]

        return out



class GeneralizedWaveFunctionUPS:
    def __init__(
        self,
        cas: tuple[tuple[int, int], int],
        mo_coeffs: np.ndarray,
        integral_generator: SlowQuant | pyscf.gto.mole.Mole,
        kpairs: list[tuple],
        Kp: bool,
        ansatz: str,
        ansatz_options: dict[str, Any] | None = None,
        include_active_kappa: bool = False,
    ) -> None:
        """Initialize for UPS wave function.

        Args:
            num_elec: Number of electrons.
            cas: CAS(num_active_elec, num_active_orbs),
                 orbitals are counted in spin basis.
            mo_coeffs: Initial orbital coefficients.
            h_ao: One-electron integrals in AO for Hamiltonian.
            g_ao: Two-electron integrals in AO.
            ansatz: Name of ansatz.
            ansatz_options: Ansatz options.
            include_active_kappa: Include active-active orbital rotations.
        """
        self.int_gen: IntegralManager = IntegralManager(integral_generator)
        self._direct = DHFDirect(self.int_gen.int_obj)
        self._ds = None
        self._ds_ep = None
        self._ci_engine = None
        num_elec = self.int_gen.num_elec
        if ansatz_options is None:
            ansatz_options = {}
        if len(cas) != 2:
            raise ValueError(f"cas must have two elements, got {len(cas)} elements.")
        if len(cas[0]) != 2:
            raise ValueError(
                "Number of electrons in the active space must be specified as a tuple of (alpha, beta)."
            )
        if cas[1] > mo_coeffs.shape[1]:
            raise ValueError(
                f"More spatial active orbitls than total orbitals. Got {cas[1]} active orbitals, and {mo_coeffs.shape[1]} total orbitals."
            )
        if np.sum(cas[0]) > num_elec:
            raise ValueError(
                f"More active electrons than total electrons. Got {np.sum(cas[0])} active electrons, and {num_elec} total electrons."
            )
        # Init stuff
        self._c_mo_new = np.copy(mo_coeffs[int(mo_coeffs.shape[1]/2):]).astype(np.complex128) # DHF
        self._c_mo = np.copy(mo_coeffs).astype(np.complex128)
        self._rdm1 = None
        self._rdm2 = None
        self._h_mo = None
        self._g_mo = None
        self._energy_elec: float | None = None
        self.ansatz_options = ansatz_options
        self.num_energy_evals = 0
        # Construct spin orbital spaces and indices
        self.inactive_spin_idx = []
        self.virtual_spin_idx = []
        self.active_spin_idx = []
        self.active_occ_spin_idx = []
        self.active_unocc_spin_idx = []
        self.active_spin_idx_shifted = []
        self.active_occ_spin_idx_shifted = []
        self.active_unocc_spin_idx_shifted = []
        self.num_elec = num_elec
        self.num_elec_alpha = (num_elec - np.sum(cas[0])) // 2 + cas[0][0]
        self.num_elec_beta = (num_elec - np.sum(cas[0])) // 2 + cas[0][1]
        self.num_spin_orbs = mo_coeffs.shape[1]   # divide by 2??

        self.num_spin_orbs_NES = int(self.num_spin_orbs/2) #DHF 

        self._include_active_kappa = include_active_kappa
        self.num_active_elec_alpha = cas[0][0]
        self.num_active_elec_beta = cas[0][1]
        self.num_active_elec = self.num_active_elec_alpha + self.num_active_elec_beta
        self.num_active_spin_orbs = cas[1]
        self.num_inactive_spin_orbs = num_elec - self.num_active_elec
        self.num_virtual_spin_orbs = (
            self.num_spin_orbs - self.num_active_spin_orbs - self.num_inactive_spin_orbs - self.num_spin_orbs_NES
        )


        # New idx for DHF:
        self.inactive_spin_idx = [self.num_spin_orbs_NES + x for x in range(self.num_inactive_spin_orbs)]
        self.active_spin_idx = [self.num_spin_orbs_NES + x + self.num_inactive_spin_orbs for x in range(self.num_active_spin_orbs)]

        # Positronic states are all treated as virtual
        self.positronic_spin_idx = list(range(self.num_spin_orbs_NES))

        # Electronic virtual orbitals shifted past positronic block
        self.virtual_spin_idx = list(range(self.active_spin_idx[-1]+1, self.num_spin_orbs))

        self.active_occ_spin_idx = []
        for i in range(self.num_active_elec_alpha):
            self.active_occ_spin_idx.append(self.num_spin_orbs_NES + 2 * i + self.num_inactive_spin_orbs)
        for i in range(self.num_active_elec_beta):
            self.active_occ_spin_idx.append(self.num_spin_orbs_NES + 2 * i + 1 + self.num_inactive_spin_orbs)
        self.active_occ_spin_idx.sort()

        self.active_unocc_spin_idx = []
        for i in range(self.num_spin_orbs_NES + self.num_inactive_spin_orbs, self.num_spin_orbs_NES + self.num_inactive_spin_orbs + self.num_active_spin_orbs):
            if i not in self.active_occ_spin_idx:
                self.active_unocc_spin_idx.append(i)

        # Make shifted indices
        if len(self.active_spin_idx) != 0:
            active_shift = np.min(self.active_spin_idx)
            for active_idx in self.active_spin_idx:
                self.active_spin_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_occ_spin_idx:
                self.active_occ_spin_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_unocc_spin_idx:
                self.active_unocc_spin_idx_shifted.append(active_idx - active_shift)


        # Old idx

        # self.inactive_spin_idx = [x for x in range(self.num_inactive_spin_orbs)]
        # self.active_spin_idx = [x + self.num_inactive_spin_orbs for x in range(self.num_active_spin_orbs)]
        # self.virtual_spin_idx = [
        #     x + self.num_inactive_spin_orbs + self.num_active_spin_orbs
        #     for x in range(self.num_virtual_spin_orbs)
        # ]
        # self.active_occ_spin_idx = []
        # for i in range(self.num_active_elec_alpha):
        #     self.active_occ_spin_idx.append(2 * i + self.num_inactive_spin_orbs)
        # for i in range(self.num_active_elec_beta):
        #     self.active_occ_spin_idx.append(2 * i + 1 + self.num_inactive_spin_orbs)
        # self.active_occ_spin_idx.sort()
        # self.active_unocc_spin_idx = []
        # for i in range(self.num_inactive_spin_orbs, self.num_inactive_spin_orbs + self.num_active_spin_orbs):
        #     if i not in self.active_occ_spin_idx:
        #         self.active_unocc_spin_idx.append(i)
        # # Make shifted indices
        # if len(self.active_spin_idx) != 0:
        #     active_shift = np.min(self.active_spin_idx)
        #     for active_idx in self.active_spin_idx:
        #         self.active_spin_idx_shifted.append(active_idx - active_shift)
        #     for active_idx in self.active_occ_spin_idx:
        #         self.active_occ_spin_idx_shifted.append(active_idx - active_shift)
        #     for active_idx in self.active_unocc_spin_idx:
        #         self.active_unocc_spin_idx_shifted.append(active_idx - active_shift)

        self._kpairs = kpairs # Kramers

        self._kpartner = {}

        for p, q in self._kpairs: # Kramers
            self._kpartner[p] = q
            self._kpartner[q] = p

        #print(self._kpartner)

        self._Kp = Kp # Kramers
        self._S_int = self.int_gen.overlap



        # Find non-redundant kappas
        self._kappa_real = []
        self._kappa_imag = []
        self.kappa_spin_idx = []
        self.kappa_spin_idx_dagger = []

        self._kappa_real_ep = [] # Positronic
        self._kappa_imag_ep = [] # Positronic
        self.kappa_spin_idx_ep = [] # Positronic
        self.kappa_spin_idx_ep_dagger = [] # Positronic


        self.kappa_no_activeactive_spin_idx = []
        self.kappa_no_activeactive_spin_idx_dagger = []

        self.kappa_no_activeactive_spin_idx_resp = []
        self.kappa_no_activeactive_spin_idx_dagger_resp = []

        self.kappa_no_activeactive_spin_idx_ep = [] # Positronic
        self.kappa_no_activeactive_spin_idx_ep_dagger = [] # Positronic

        self._kappa_real_redundant = []
        self._kappa_imag_redundant = []
        self.kappa_redundant_spin_idx = []
        self._kappa_real_old = []
        self._kappa_imag_old = []

        self._kappa_real_old_ep = [] # Positronic
        self._kappa_imag_old_ep = [] # Positronic

        self._kappa_real_redundant_old = []
        self._kappa_imag_redundant_old = []
        # Annika has modified this, since non-redundant orbital rotations had been left out!
        inactive_set, virtual_set = set(self.inactive_spin_idx), set(self.virtual_spin_idx)
        active_set, positronic_set = set(self.active_spin_idx), set(self.positronic_spin_idx)
        for P in range(0, self.num_spin_orbs):
            for Q in range(P, self.num_spin_orbs): 
                if P < self.num_spin_orbs_NES and Q < self.num_spin_orbs_NES: #No p-p rotations
                    self._kappa_real_redundant.append(0.0)
                    self._kappa_imag_redundant.append(0.0)
                    self._kappa_real_redundant_old.append(0.0)
                    self._kappa_imag_redundant_old.append(0.0)
                    self.kappa_redundant_spin_idx.append((P, Q))
                    continue

                # if P < self.num_spin_orbs_NES or Q < self.num_spin_orbs_NES: #No e-p rotations
                #     self._kappa_real_redundant.append(0.0)
                #     self._kappa_imag_redundant.append(0.0)
                #     self._kappa_real_redundant_old.append(0.0)
                #     self._kappa_imag_redundant_old.append(0.0)
                #     self.kappa_redundant_spin_idx.append((P, Q))
                #     continue

                if P in inactive_set and Q in inactive_set:
                    self._kappa_real_redundant.append(0.0)
                    self._kappa_imag_redundant.append(0.0)
                    self._kappa_real_redundant_old.append(0.0)
                    self._kappa_imag_redundant_old.append(0.0)
                    self.kappa_redundant_spin_idx.append((P, Q))
                    continue
                if P in virtual_set and Q in virtual_set:
                    self._kappa_real_redundant.append(0.0)
                    self._kappa_imag_redundant.append(0.0)
                    self._kappa_real_redundant_old.append(0.0)
                    self._kappa_imag_redundant_old.append(0.0)
                    self.kappa_redundant_spin_idx.append((P, Q))
                    continue
                if not include_active_kappa:
                    if P in active_set and Q in active_set:
                        self._kappa_real_redundant.append(0.0)
                        self._kappa_imag_redundant.append(0.0)
                        self._kappa_real_redundant_old.append(0.0)
                        self._kappa_imag_redundant_old.append(0.0)
                        self.kappa_redundant_spin_idx.append((P, Q))
                        continue
                if include_active_kappa:
                    if P in active_set and Q in active_set:
                        if P==Q:
                            self._kappa_real_redundant.append(0.0)
                            self._kappa_imag_redundant.append(0.0)
                            self._kappa_real_redundant_old.append(0.0)
                            self._kappa_imag_redundant_old.append(0.0)
                            self.kappa_redundant_spin_idx.append((P, Q))
                            continue
                    # if P in self.active_occ_spin_idx and Q in self.active_occ_spin_idx:
                    #     self._kappa_real_redundant.append(0.0)
                    #     self._kappa_imag_redundant.append(0.0)
                    #     self._kappa_real_redundant_old.append(0.0)
                    #     self._kappa_imag_redundant_old.append(0.0)
                    #     self.kappa_redundant_spin_idx.append((P, Q))
                    #     continue
                    # if P in self.active_unocc_spin_idx and Q in self.active_unocc_spin_idx:
                    #     self._kappa_real_redundant.append(0.0)
                    #     self._kappa_imag_redundant.append(0.0)
                    #     self._kappa_real_redundant_old.append(0.0)
                    #     self._kappa_imag_redundant_old.append(0.0)
                    #     self.kappa_redundant_spin_idx.append((P, Q))
                    #     continue
                if not (P in active_set and Q in active_set):
                    if P not in positronic_set and Q not in positronic_set:
                        self.kappa_no_activeactive_spin_idx_resp.append((P, Q))
                        self.kappa_no_activeactive_spin_idx_dagger_resp.append((Q, P))

                        # if P in inactive_set and (Q in virtual_set or Q in self.active_unocc_spin_idx):
                        #     self.kappa_no_activeactive_spin_idx_resp.append((P, Q))
                        #     self.kappa_no_activeactive_spin_idx_dagger_resp.append((Q, P))
                        # if (P in virtual_set or Q in self.active_unocc_spin_idx) and Q in inactive_set:
                        #     self.kappa_no_activeactive_spin_idx_resp.append((P, Q))
                        #     self.kappa_no_activeactive_spin_idx_dagger_resp.append((Q, P))

                        # if P in self.active_occ_spin_idx and Q in virtual_set:
                        #     self.kappa_no_activeactive_spin_idx_resp.append((P, Q))
                        #     self.kappa_no_activeactive_spin_idx_dagger_resp.append((Q, P))
                        # if P in virtual_set and Q in self.active_occ_spin_idx: 

                        self.kappa_no_activeactive_spin_idx.append((P, Q))
                        self.kappa_no_activeactive_spin_idx_dagger.append((Q, P))

                    if P in positronic_set and Q not in positronic_set and Q not in virtual_set:
                    #if P in positronic_set and Q in self.active_occ_spin_idx:
                        self.kappa_no_activeactive_spin_idx_resp.append((Q, P)) # positronic included, Which one is dagger?
                        self.kappa_no_activeactive_spin_idx_dagger_resp.append((P, Q))

                        self.kappa_no_activeactive_spin_idx_ep.append((P, Q))
                        self.kappa_no_activeactive_spin_idx_ep_dagger.append((Q, P))

                    # if P in positronic_set and Q in inactive_set:
                    #     self.kappa_no_activeactive_spin_idx_resp.append((Q, P)) # positronic included, Which one is dagger?
                    #     self.kappa_no_activeactive_spin_idx_dagger_resp.append((P, Q))

                    #     self.kappa_no_activeactive_spin_idx_ep.append((P, Q))
                    #     self.kappa_no_activeactive_spin_idx_ep_dagger.append((Q, P))


                if P < self.num_spin_orbs_NES and Q >= self.num_spin_orbs_NES and Q not in virtual_set: # Positronic
                    if self._Kp:
                        if (self._kpartner[P], self._kpartner[Q]) in self.kappa_spin_idx or (self._kpartner[Q], self._kpartner[P]) in self.kappa_spin_idx_ep:
                            continue
                        self._kappa_real_ep.append(0.0) # Positronic
                        self._kappa_imag_ep.append(0.0) # Positronic 
                        self._kappa_real_old_ep.append(0.0) # Positronic
                        self._kappa_imag_old_ep.append(0.0) # Positronic 
                        self.kappa_spin_idx_ep.append((Q, P))
                        self.kappa_spin_idx_ep_dagger.append((P, Q)) # Positronic included, AND WHICH IS FIRST P OR Q?
                        continue
                    else:
                        self._kappa_real_ep.append(0.0) # Positronic
                        self._kappa_imag_ep.append(0.0) # Positronic 
                        self._kappa_real_old_ep.append(0.0) # Positronic
                        self._kappa_imag_old_ep.append(0.0) # Positronic 
                        self.kappa_spin_idx_ep.append((Q, P))
                        self.kappa_spin_idx_ep_dagger.append((P, Q)) # Positronic included, AND WHICH IS FIRST P OR Q?
                        continue

                if P < self.num_spin_orbs_NES and Q >= self.num_spin_orbs_NES and Q in virtual_set: # Positronic
                    self._kappa_real_redundant.append(0.0)
                    self._kappa_imag_redundant.append(0.0)
                    self._kappa_real_redundant_old.append(0.0)
                    self._kappa_imag_redundant_old.append(0.0)
                    self.kappa_redundant_spin_idx.append((P, Q))
                    continue

                if self._Kp:
                    if (self._kpartner[P], self._kpartner[Q]) in self.kappa_spin_idx or (self._kpartner[Q], self._kpartner[P]) in self.kappa_spin_idx:
                        continue
                    self._kappa_real.append(0.0)
                    self._kappa_imag.append(0.0)
                    self._kappa_real_old.append(0.0)
                    self._kappa_imag_old.append(0.0)
                    self.kappa_spin_idx.append((P, Q))
                    self.kappa_spin_idx_dagger.append((Q, P))
                else:
                    self._kappa_real.append(0.0)
                    self._kappa_imag.append(0.0)
                    self._kappa_real_old.append(0.0)
                    self._kappa_imag_old.append(0.0)
                    self.kappa_spin_idx.append((P, Q))
                    self.kappa_spin_idx_dagger.append((Q, P))






        # print('self.kappa_spin_idx', self.kappa_spin_idx) ###AE skal de krydse over (Juliane)

        # Construct determinant basis
        self.ci_info = get_indexing_DHF(
            self.num_inactive_spin_orbs,
            self.num_active_spin_orbs,
            self.num_virtual_spin_orbs,
            self.num_active_elec_alpha,
            self.num_active_elec_beta,
            self.num_spin_orbs_NES,
        )
        self.num_det = len(self.ci_info.idx2det)
        self.csf_coeffs = np.zeros(self.num_det, dtype=np.complex128)
        hf_string = ""
        for i in range(self.num_active_spin_orbs // 2):
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
        # Do the ansatz setup
        #
        # Note: Put here because I do not know here else to put it.
        #       The operator pool currectly does not have the diagonal elements.
        #       F.x. not the p^dagger p single excitation.
        #       These should be there? (At some point)
        #
        if ansatz.lower() == "fuccsd":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ansatz_options["S"] = True
            self.ansatz_options["D"] = True
            self.ups_layout.create_fUCC(
                [],
                [],
                self.active_occ_spin_idx_shifted,
                self.active_unocc_spin_idx_shifted,
                self.num_active_spin_orbs // 2,
                self.ansatz_options,
            )

        elif ansatz.lower() == "fuccd":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.ansatz_options["D"] = True
            self.ups_layout.create_fUCC(
                [],
                [],
                self.active_occ_spin_idx_shifted,
                self.active_unocc_spin_idx_shifted,
                self.num_active_spin_orbs // 2,
                self.ansatz_options,
            )
        elif ansatz.lower() == "adapt":
            None
        else:
            raise ValueError(f"Got unknown ansatz, {ansatz}")
        if self.ups_layout.n_params == 0:
            self._thetas_real = []
            self._thetas_imag = []
        else:
            self._thetas_real = np.zeros(self.ups_layout.n_params, dtype=float).tolist()
            self._thetas_imag = np.zeros(self.ups_layout.n_params, dtype=float).tolist()

    @property
    def kappa_real(self) -> list[float]:
        """Get real orbital rotation parameters."""
        return self._kappa_real.copy()
    
    @property
    def kappa_real_ep(self) -> list[float]:  # Positronic
        """Get real orbital rotation parameters."""
        return self._kappa_real_ep.copy()

    @property
    def kappa_imag(self) -> list[float]:
        """Get imaginary orbital rotation parameters."""
        return self._kappa_imag.copy()
    
    @property
    def kappa_imag_ep(self) -> list[float]:  # Positronic
        """Get imaginary orbital rotation parameters."""
        return self._kappa_imag_ep.copy()

    def set_kappa_cep(self, k_real: list[float], k_imag: list[float]) -> None:
        """Set orbital rotation parameters, and move current expansion point.

        Args:
            k: orbital rotation parameters.
        """
        self._h_mo = None
        self._g_mo = None
        self._energy_elec = None
        self._kappa_real = k_real.copy()
        self._kappa_imag = k_imag.copy()
        if isinstance(self._kappa_real, np.ndarray):
            self._kappa_real = self._kappa_real.tolist()
        if isinstance(self._kappa_imag, np.ndarray):
            self._kappa_imag = self._kappa_imag.tolist()
        # Move current expansion point.
        self._c_mo = self.c_mo
        self._kappa_real_old = self.kappa_real
        self._kappa_imag_old = self.kappa_imag

    def set_kappa_cep_ep(self, k_real_ep: list[float], k_imag_ep: list[float]) -> None: # Positronic
        """Set orbital rotation parameters, and move current expansion point.

        Args:
            k: orbital rotation parameters.
        """
        self._h_mo = None
        self._g_mo = None
        self._energy_elec = None
        self._kappa_real_ep = k_real_ep.copy()
        self._kappa_imag_ep = k_imag_ep.copy()
        if isinstance(self._kappa_real_ep, np.ndarray):
            self._kappa_real_ep = self._kappa_real_ep.tolist()
        if isinstance(self._kappa_imag_ep, np.ndarray):
            self._kappa_imag_ep = self._kappa_imag_ep.tolist()
        # Move current expansion point.
        self._c_mo = self.c_mo_ep
        self._kappa_real_old_ep = self.kappa_real_ep
        self._kappa_imag_old_ep = self.kappa_imag_ep

    @property
    def thetas_real(self) -> list[float]:
        """Get real theta values.

        Returns:
            theta values.
        """
        return self._thetas_real.copy()

    @property
    def thetas_imag(self) -> list[float]:
        """Get imaginary theta values.

        Returns:
            theta values.
        """
        return self._thetas_imag.copy()

    @property
    def thetas(self) -> list[complex]:
        """Get complex theta values.

        Returns:
            theta values.
        """
        return (np.array(self.thetas_real) + 1.0j * np.array(self.thetas_imag)).tolist()

    def set_thetas(self, theta_real: list[float], theta_imag: list[float]) -> None:
        """Set theta values.

        Args:
            theta_vals: theta values.
        """
        if len(theta_real) != len(self._thetas_real):
            raise ValueError(f"Expected {len(self._thetas_real)} real theta values got {len(theta_real)}")
        # Remove this warning for running with real valued thetas:
        if len(theta_imag) != len(self._thetas_imag):
            raise ValueError(
                f"Expected {len(self._thetas_imag)} imaginary theta values got {len(theta_imag)}"
            )
        self._rdm1 = None
        self._rdm2 = None
        self._energy_elec = None
        self._thetas_real = theta_real.copy()
        self._thetas_imag = theta_imag.copy()
        if isinstance(self._thetas_real, np.ndarray):
            self._thetas_real = self._thetas_real.tolist()
        if isinstance(self._thetas_imag, np.ndarray):
            self._thetas_imag = self._thetas_imag.tolist()
        
        self.ci_coeffs = generalized_construct_ups_state_test_erik(
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

        if self._kappa_real == self._kappa_real_old and self._kappa_imag == self._kappa_imag_old:
            return self._c_mo
        if not self._Kp:
            return self._rotate(self.kappa_spin_idx, self._kappa_real, self._kappa_real_old, self._kappa_imag, self._kappa_imag_old)

        # Calculate Kramers phases from current MO coefficients
        if self._Kp:
            M = self._c_mo.conj().T @ self._S_int @ Trev(self._c_mo)

            kphase = {}

            for p, q in self._kpairs:
                eta = M[p, q] / abs(M[p, q])

                kphase[p] = eta
                kphase[q] = -np.conj(eta)


        # Construct anti-hermitian kappa matrix
        kappa_mat = np.zeros_like(self._c_mo)

        if len(self.kappa_real) != 0:

            if np.max(np.abs(np.array(self.kappa_real) -
                            np.array(self._kappa_real_old))) > 0.0:

                for kappa_val, kappa_old, (p, q) in zip(
                    self.kappa_real,
                    self._kappa_real_old,
                    self.kappa_spin_idx
                ):

                    if p == q:
                        continue

                    delta = kappa_val - kappa_old

                    # Original rotation
                    kappa_mat[p, q] = delta
                    kappa_mat[q, p] = -delta


                    if self._Kp:

                        pb = self._kpartner[p]
                        qb = self._kpartner[q]

                        phase_factor = kphase[p] * np.conj(kphase[q])

                        partner_delta = phase_factor * np.conj(delta)

                        kappa_mat[pb, qb] = partner_delta
                        kappa_mat[qb, pb] = -np.conj(partner_delta)


            if np.max(np.abs(np.array(self.kappa_imag) -
                            np.array(self._kappa_imag_old))) > 0.0:

                for kappa_val, kappa_old, (p, q) in zip(
                    self.kappa_imag,
                    self._kappa_imag_old,
                    self.kappa_spin_idx
                ):

                    delta = 1.0j * (kappa_val - kappa_old)

                    # Original rotation
                    kappa_mat[p, q] += delta
                    kappa_mat[q, p] -= np.conj(delta)


                    if self._Kp:

                        pb = self._kpartner[p]
                        qb = self._kpartner[q]

                        phase_factor = kphase[p] * np.conj(kphase[q])

                        partner_delta = phase_factor * np.conj(delta)

                        kappa_mat[pb, qb] += partner_delta
                        kappa_mat[qb, pb] -= np.conj(partner_delta)

            return np.matmul(self._c_mo, scipy.linalg.expm(-kappa_mat))

        return self._c_mo


    def _rotate(self, idx, kr, kr_old, ki, ki_old) -> np.ndarray:
        P, Q = np.array(idx, dtype=int).reshape(-1, 2).T
        d = np.array(kr) - np.array(kr_old) + 1j * (np.array(ki) - np.array(ki_old))
        kappa_mat = np.zeros_like(self._c_mo)
        kappa_mat[P, Q] = d
        kappa_mat[Q, P] = -d.conj()
        return self._c_mo @ scipy.linalg.expm(-kappa_mat)

    @property
    def c_mo_ep(self) -> np.ndarray:  # Positronic
        """Get molecular orbital coefficients.

        Returns:
            Molecular orbital coefficients.
        """

        if self._kappa_real_ep == self._kappa_real_old_ep and self._kappa_imag_ep == self._kappa_imag_old_ep:
            return self._c_mo
        if not self._Kp:
            return self._rotate(self.kappa_spin_idx_ep, self._kappa_real_ep, self._kappa_real_old_ep, self._kappa_imag_ep, self._kappa_imag_old_ep)

        # Calculate Kramers phases from current positronic MOs
        if self._Kp:
            M = self._c_mo.conj().T @ self._S_int @ Trev(self._c_mo)

            kphase = {}

            for p, q in self._kpairs:
                eta = M[p, q] / abs(M[p, q])

                kphase[p] = eta
                kphase[q] = -np.conj(eta)


        # Construct anti-hermitian kappa matrix
        kappa_mat = np.zeros_like(self._c_mo)

        if len(self.kappa_real_ep) != 0:

            if np.max(
                np.abs(np.array(self.kappa_real_ep)
                    - np.array(self._kappa_real_old_ep))
            ) > 0.0:

                for kappa_val, kappa_old, (p, q) in zip(
                    self.kappa_real_ep,
                    self._kappa_real_old_ep,
                    self.kappa_spin_idx_ep
                ):

                    if p == q:
                        continue

                    delta = kappa_val - kappa_old

                    # Original rotation
                    kappa_mat[p, q] = delta
                    kappa_mat[q, p] = -delta


                    # Kramers partner rotation
                    if self._Kp:

                        pb = self._kpartner[p]
                        qb = self._kpartner[q]

                        phase_factor = kphase[p] * np.conj(kphase[q])

                        partner_delta = phase_factor * np.conj(delta)

                        kappa_mat[pb, qb] = partner_delta
                        kappa_mat[qb, pb] = -np.conj(partner_delta)

            if np.max(
                np.abs(np.array(self.kappa_imag_ep)
                    - np.array(self._kappa_imag_old_ep))
            ) > 0.0:

                for kappa_val, kappa_old, (p, q) in zip(
                    self.kappa_imag_ep,
                    self._kappa_imag_old_ep,
                    self.kappa_spin_idx_ep
                ):

                    delta = 1.0j * (kappa_val - kappa_old)

                    # Original rotation
                    kappa_mat[p, q] += delta
                    kappa_mat[q, p] -= np.conj(delta)


                    # Kramers partner rotation
                    if self._Kp:

                        pb = self._kpartner[p]
                        qb = self._kpartner[q]

                        phase_factor = kphase[p] * np.conj(kphase[q])

                        partner_delta = phase_factor * np.conj(delta)

                        kappa_mat[pb, qb] += partner_delta
                        kappa_mat[qb, pb] -= np.conj(partner_delta)

            return np.matmul(self._c_mo, scipy.linalg.expm(-kappa_mat))

        return self._c_mo
        


    @property
    def h_mo(self) -> np.ndarray:
        """Get one-electron Hamiltonian integrals in MO basis.

        Returns:
            One-electron Hamiltonian integrals in MO basis.
        """
        if self._h_mo is None:
            self._h_mo = DHF_one_electron_transform(self.c_mo, self.int_gen.h_ao) #DHF
        return self._h_mo

    @property
    def g_mo(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._g_mo is None:
            self._g_mo = DHF_two_electron_transform(self.c_mo, self.int_gen.electron_electron_repulsion) #DHF
        return self._g_mo

    def set_integral_backend(self, direct) -> None:
        """Use another integral backend (DHFDirect options or DHFCholesky) for OO and response."""
        self._direct = direct
        self._ds = self._ds_ep = None

    def _get_direct_state(self, C: np.ndarray, attr: str) -> DirectState:
        for ds in (self._ds, self._ds_ep):
            if ds is not None and np.array_equal(ds.C, C):
                setattr(self, attr, ds)
                return ds
        ds = DirectState(self._direct, C, self.inactive_spin_idx, self.active_spin_idx, self.rdm1)
        setattr(self, attr, ds)
        return ds

    @property
    def ds(self) -> DirectState:
        """Integral-direct Fock matrices and active integrals for c_mo."""
        return self._get_direct_state(self.c_mo, "_ds")

    @property
    def ds_ep(self) -> DirectState:
        """Integral-direct Fock matrices and active integrals for c_mo_ep."""
        return self._get_direct_state(self.c_mo_ep, "_ds_ep")

    @property
    def h_mo_ep(self) -> np.ndarray:
        """Get one-electron Hamiltonian integrals in MO basis.

        Returns:
            One-electron Hamiltonian integrals in MO basis.
        """
        if self._h_mo is None:
            self._h_mo = DHF_one_electron_transform(self.c_mo_ep, self.int_gen.h_ao) #DHF
        return self._h_mo

    @property
    def g_mo_ep(self) -> np.ndarray:
        """Get two-electron Hamiltonian integrals in MO basis.

        Returns:
            Two-electron Hamiltonian integrals in MO basis.
        """
        if self._g_mo is None:
            self._g_mo = DHF_two_electron_transform(self.c_mo_ep, self.int_gen.electron_electron_repulsion) #DHF
        return self._g_mo

    @property
    def ci_engine(self) -> ActiveCI:
        """Sparse active-space CI engine (E_tu matrices)."""
        if self._ci_engine is None:
            self._ci_engine = ActiveCI(self.ci_info)
        return self._ci_engine

    def _make_rdms(self) -> None:
        c = self.ci_coeffs
        _, d, G = self.ci_engine.transition_rdms(c[None], c)
        self._rdm1, self._rdm2 = d[0], G[0]

    @property
    def rdm1(self) -> np.ndarray:
        """Active-space one-electron reduced density matrix, rdm1[t, u] = <a^+_t a_u>."""
        if self._rdm1 is None:
            self._make_rdms()
        return self._rdm1

    @property
    def rdm2(self) -> np.ndarray:
        """Active-space two-electron reduced density matrix, rdm2[t, u, v, w] = <a^+_t a^+_v a_w a_u>."""
        if self._rdm2 is None:
            self._make_rdms()
        return self._rdm2

    @property
    def rdm1_FULL(self) -> np.ndarray:
        """Calculate one-electron reduced density matrix in the active space.

        Returns:
            One-electron reduced density matrix.
        """
        # Annika has added dtype=complex
        if self._rdm1 is None:
            self._rdm1 = np.zeros((self.num_active_spin_orbs, self.num_active_spin_orbs), dtype=np.complex128)
            for P in range(
                self.num_inactive_spin_orbs, self.num_inactive_spin_orbs + self.num_active_spin_orbs
            ):
                P_idx = P - self.num_inactive_spin_orbs
                for Q in range(
                    self.num_inactive_spin_orbs, self.num_inactive_spin_orbs + self.num_active_spin_orbs
                ):
                    Q_idx = Q - self.num_inactive_spin_orbs
                    val = generalized_expectation_value(
                        self.ci_coeffs,
                        [(a_op_spin(P, True) * a_op_spin(Q, False))],
                        self.ci_coeffs,
                        self.ci_info,
                    )
                    self._rdm1[P_idx, Q_idx] = val  # type: ignore
        return self._rdm1

    @property
    def rdm2_FULL(self) -> np.ndarray:
        """Calculate two-electron reduced density matrix in the active space.

        Returns:
            Two-electron reduced density matrix.
        """
        # Annika has added dtype=complex
        if self._rdm2 is None:
            self._rdm2 = np.zeros(
                (
                    self.num_active_spin_orbs,
                    self.num_active_spin_orbs,
                    self.num_active_spin_orbs,
                    self.num_active_spin_orbs,
                ),
                dtype=np.complex128,
            )
            for p in range(
                self.num_inactive_spin_orbs, self.num_inactive_spin_orbs + self.num_active_spin_orbs
            ):
                p_idx = p - self.num_inactive_spin_orbs
                for q in range(
                    self.num_inactive_spin_orbs, self.num_inactive_spin_orbs + self.num_active_spin_orbs
                ):
                    q_idx = q - self.num_inactive_spin_orbs
                    for r in range(
                        self.num_inactive_spin_orbs, self.num_inactive_spin_orbs + self.num_active_spin_orbs
                    ):
                        r_idx = r - self.num_inactive_spin_orbs
                        for s in range(
                            self.num_inactive_spin_orbs,
                            self.num_inactive_spin_orbs + self.num_active_spin_orbs,
                        ):
                            s_idx = s - self.num_inactive_spin_orbs
                            val = generalized_expectation_value(
                                self.ci_coeffs,
                                [
                                    (
                                        a_op_spin(p, True)
                                        * a_op_spin(r, True)
                                        * a_op_spin(s, False)
                                        * a_op_spin(q, False)
                                    )
                                ],
                                self.ci_coeffs,
                                self.ci_info,
                                do_folding=False
                            )
                            self._rdm2[p_idx, q_idx, r_idx, s_idx] = val  # type: ignore
        return self._rdm2
    
    @property
    def rdm2_new(self) -> np.ndarray:
        """Calculate two-electron reduced density matrix in the active space.

        Returns:
            Two-electron reduced density matrix.
        """
        # Annika has added dtype=complex
        if self._rdm2 is None:
            self._rdm2 = np.zeros(
                (
                    self.num_spin_orbs,
                    self.num_spin_orbs,
                    self.num_spin_orbs,
                    self.num_spin_orbs,
                ),
                dtype=np.complex128,
            )
            for p in range(
                self.num_spin_orbs
            ):
                p_idx = p 
                for q in range(
                    self.num_spin_orbs
                ):
                    q_idx = q
                    for r in range(
                        self.num_spin_orbs
                    ):
                        r_idx = r 
                        for s in range(
                            self.num_spin_orbs
                        ):
                            s_idx = s
                            val = generalized_expectation_value(
                                self.ci_coeffs,
                                [
                                    (
                                        a_op_spin(p, True)
                                        * a_op_spin(r, True)
                                        * a_op_spin(s, False)
                                        * a_op_spin(q, False)
                                    )
                                ],
                                self.ci_coeffs,
                                self.ci_info,
                                do_folding = True
                            )
                            self._rdm2[p_idx, q_idx, r_idx, s_idx] = val  # type: ignore
        return self._rdm2

    def check_orthonormality(self, overlap_integral: np.ndarray) -> None:
        r"""Check orthonormality of orbitals.

        .. math::
            \boldsymbol{I} = \boldsymbol{C}_\text{MO}\boldsymbol{S}\boldsymbol{C}_\text{MO}^T

        Args:
            overlap_integral: Overlap integral in AO basis.
        """
        S_ortho = generalized_one_electron_transform(self.c_mo, overlap_integral)
        one = np.identity(len(S_ortho))
        diff = np.abs(S_ortho - one)
        print("Max ortho-normal diff:", np.max(diff))

    @property
    def energy_elec(self) -> float:
        """Get the electronic energy.

        Returns:
            Electronic energy.
        """
        NES = self.num_spin_orbs_NES

        if self._energy_elec is None:
            c = self.ci_coeffs
            self._energy_elec = np.vdot(c, self.ds.ci_hamiltonian(self.ci_engine) @ c).real
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
            print(f"### Number kappa: {len(self.kappa_real)}")
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
            self._old_opt_parameters = np.zeros(2*len(self.thetas), dtype=np.float64) + 10**20
            self._E_opt_old = 0.0
            thetas = self.thetas_real + self.thetas_imag
            res = optimizer.minimize(
                thetas,
                extra_options={"R": self.ups_layout.grad_param_R, "param_names": self.ups_layout.param_names},
            )
            thetas_r = []
            thetas_i = []
            for i in range(len(self.thetas)):
                thetas_r.append(res.x[i])
                thetas_i.append(res.x[i + len(self.thetas)])
            self.set_thetas(thetas_r, thetas_i)

            if orbital_optimization and len(self.kappa_real) != 0:
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
                    "bfgs",
                    grad=gradient_oo,
                    maxiter=maxiter,
                    tol=tol,
                    is_silent=is_silent_subiterations,
                    energy_eval_callback=lambda: self.num_energy_evals,
                )
                self._old_opt_parameters = np.zeros(2 * len(self.kappa_spin_idx), dtype=np.float64) + 10**20
                self._E_opt_old = 0.0
                print("energy", self.energy_elec)
                res = optimizer.minimize([0.0] * 2 * len(self.kappa_spin_idx))
                print("energy", self.energy_elec)
                for i in range(len(self.kappa_real)):
                    self._kappa_real[i] = 0.0
                    self._kappa_imag[i] = 0.0
                    self._kappa_real_old[i] = 0.0
                    self._kappa_imag_old[i] = 0.0
            else:
                # If there is no orbital optimization, then the algorithm is already converged.
                e_new = res.fun
                if orbital_optimization and len(self.kappa_real) == 0:
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
    
    def run_wf_optimization_2step_DHF(
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
            print(f"### Number kappa: {len(self.kappa_real) + len(self.kappa_real_ep)}")
        print(f"### Number theta: {self.ups_layout.n_params}")
        e_old = 1e12
        print("Full optimization")
        print("Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #")
        for full_iter in range(0, int(maxiter)):
            full_start = time.time()
            if len(self.thetas) != 0:
                # Do ansatz optimization
                if not is_silent_subiterations:
                    print("--------Ansatz optimization")
                    print(
                        "--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #"
                    )
                energy_theta = partial(
                    self._calc_energy_optimization_DHF,
                    theta_optimization=True,
                    kappa_ee_optimization=False,
                    kappa_ep_optimization=False,
                )
                gradient_theta = partial(
                    self._calc_gradient_optimization_DHF,
                    theta_optimization=True,
                    kappa_ee_optimization=False,
                    kappa_ep_optimization=False,
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
                self._old_opt_parameters = np.zeros(2*len(self.thetas), dtype=np.float64) + 10**20
                self._E_opt_old = 0.0
                thetas = self.thetas_real + self.thetas_imag
                res = optimizer.minimize(
                    thetas,
                    extra_options={"R": self.ups_layout.grad_param_R, "param_names": self.ups_layout.param_names},
                )
                thetas_r = []
                thetas_i = []
                for i in range(len(self.thetas)):
                    thetas_r.append(res.x[i])
                    thetas_i.append(res.x[i + len(self.thetas)])
                self.set_thetas(thetas_r, thetas_i)

            # Electronic 
            if orbital_optimization and len(self.kappa_real) != 0:
                if not is_silent_subiterations:
                    print("--------Orbital optimization e-e")
                    print(
                        "--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #"
                    )
                energy_ee = partial(
                    self._calc_energy_optimization_DHF,
                    theta_optimization=False,
                    kappa_ee_optimization=True,
                    kappa_ep_optimization=False,
                )
                gradient_ee = partial(
                    self._calc_gradient_optimization_DHF,
                    theta_optimization=False,
                    kappa_ee_optimization=True,
                    kappa_ep_optimization=False,
                )

                optimizer = Optimizers(
                    energy_ee,
                    optimizer_name,
                    grad=gradient_ee,
                    maxiter=maxiter,
                    tol=tol,
                    is_silent=is_silent_subiterations,
                    energy_eval_callback=lambda: self.num_energy_evals,
                )

                self._old_opt_parameters = np.zeros(2 * len(self.kappa_spin_idx), dtype=np.float64) + 10**20
                self._E_opt_old = 0.0
                parameters = np.zeros(2 * len(self.kappa_real), dtype=float).tolist()
                res = optimizer.minimize(parameters)
                for i in range(len(self.kappa_real)):
                    self._kappa_real[i] = 0.0
                    self._kappa_imag[i] = 0.0
                    self._kappa_real_old[i] = 0.0
                    self._kappa_imag_old[i] = 0.0

                
                # Positronic
                if not is_silent_subiterations:
                    print("--------Orbital optimization e-p")
                    print(
                        "--------Iteration # | Iteration time [s] | Electronic energy [Hartree] | Energy measurement #"
                    )
                energy_ep = partial(
                    self._calc_energy_optimization_DHF,
                    theta_optimization=False,
                    kappa_ep_optimization=True,
                    kappa_ee_optimization=False,
                )
                gradient_ep = partial(
                    self._calc_gradient_optimization_DHF,
                    theta_optimization=False,
                    kappa_ep_optimization=True,
                    kappa_ee_optimization=False,
                )

                #angle_bound = np.pi / 500
                angle_bound = np.pi / 500
                bounds_ep = [(-angle_bound, angle_bound) for _ in range(2 * len(self.kappa_spin_idx_ep))]

                
                optimizer = Optimizers(
                    energy_ep,
                    optimizer_name,
                    grad=gradient_ep,
                    maxiter=maxiter,
                    tol=tol,
                    is_silent=is_silent_subiterations,
                    energy_eval_callback=lambda: self.num_energy_evals,
                )

                # optimizer = Optimizers(
                #     energy_ep,
                #     "trust-constr",
                #     #optimizer_name,
                #     grad=gradient_ep,
                #     maxiter=maxiter,
                #     tol=tol,
                #     is_silent=is_silent_subiterations,
                #     energy_eval_callback=lambda: self.num_energy_evals,
                #     maxls = 100,
                #     maxcor = 10,
                #     bounds=bounds_ep,
                #     gtol = 1e-6,
                #             )

                self._old_opt_parameters = np.zeros(2 * len(self.kappa_spin_idx_ep), dtype=np.float64) + 10**20
                self._E_opt_old = 0.0
                parameters = np.zeros(2 * len(self.kappa_real_ep), dtype=float).tolist()
                res = optimizer.minimize(parameters)
                for i in range(len(self.kappa_real_ep)):
                    self._kappa_real_ep[i] = 0.0
                    self._kappa_imag_ep[i] = 0.0
                    self._kappa_real_old_ep[i] = 0.0
                    self._kappa_imag_old_ep[i] = 0.0
            
            else:
                # If there is no orbital optimization, then the algorithm is already converged.
                e_new = res.fun
                if orbital_optimization and len(self.kappa_real) == 0:
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
        self._energy_elec = -e_new


    def _optimize_thetas(self, gtol: float, maxiter: int) -> float:
        """Theta optimization at fixed orbitals (L-BFGS-B, converged on the gradient)."""
        energy_theta = partial(
            self._calc_energy_optimization_DHF, theta_optimization=True, kappa_ee_optimization=False, kappa_ep_optimization=False
        )
        gradient_theta = partial(
            self._calc_gradient_optimization_DHF, theta_optimization=True, kappa_ee_optimization=False, kappa_ep_optimization=False
        )
        self._old_opt_parameters = np.zeros(2 * len(self.thetas), dtype=np.float64) + 10**20
        self._E_opt_old = 0.0
        res = scipy.optimize.minimize(
            energy_theta, np.array(self.thetas_real + self.thetas_imag), jac=gradient_theta, method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-16, "gtol": gtol, "maxcor": 30},
        )
        nt = len(self.thetas)
        self.set_thetas(list(res.x[:nt]), list(res.x[nt:]))
        return res.fun

    def _set_orbitals(self, C: np.ndarray) -> None:
        self._c_mo = C
        for lst in (
            self._kappa_real, self._kappa_imag, self._kappa_real_old, self._kappa_imag_old,
            self._kappa_real_ep, self._kappa_imag_ep, self._kappa_real_old_ep, self._kappa_imag_old_ep,
        ):
            lst[:] = [0.0] * len(lst)
        self._h_mo = None
        self._g_mo = None
        self._energy_elec = None

    def run_wf_optimization_direct_DHF(
        self,
        tol: float = 1e-5,
        maxiter: int = 100,
        theta_gtol: float = 1e-9,
        theta_maxiter: int = 1000,
        max_step: float = 1.0,
        h_min: float = 1e-3,
        memory: int = 10,
    ) -> None:
        r"""Integral-direct two-step OO optimization (e-e rotations minimized, e-p rotations maximized).

        Each macro iteration: full theta optimization at fixed orbitals (sparse CI Hamiltonian), then one
        orbital step C -> C exp(-K). e-p rotations: diagonal Newton step -g/h (h exact to high accuracy).
        e-e rotations: L-BFGS direction with the diagonal Hessian h = 2(D_M F_NN + D_N F_MM - F^gen_MM - F^gen_NN)
        as initial inverse Hessian, Armijo backtracking. One J/K pass (2 densities) and one (N a|a a)
        transformation per orbital point. Converged when max |orbital gradient| < tol.
        """
        idx = [tuple(i) for i in self.kappa_spin_idx] + [tuple(i) for i in self.kappa_spin_idx_ep]
        M, N_ = np.array(idx, dtype=int).reshape(-1, 2).T
        nk = len(idx)
        ep_set = set(tuple(i) for i in self.kappa_spin_idx_ep)
        ep = np.tile(np.array([i in ep_set for i in idx], dtype=bool), 2)
        ee = ~ep
        space = np.empty(self._c_mo.shape[1], dtype="<U1")
        space[self.positronic_spin_idx] = "p"
        space[self.inactive_spin_idx] = "i"
        space[self.active_spin_idx] = "a"
        space[self.virtual_spin_idx] = "v"
        print(f"### Integral-direct OO: {nk} orbital rotations (complex), {len(self.thetas)} thetas")
        print("Iteration | Time [s] | Electronic energy [Hartree] | max |orbital grad| (type) | step scale")

        def kappa(x):
            K = np.zeros_like(self._c_mo)
            K[M, N_] = x[:nk] + 1j * x[nk:]
            K[N_, M] = -(x[:nk] - 1j * x[nk:])
            return K

        def relax_thetas():
            if len(self.thetas) != 0:
                self._optimize_thetas(theta_gtol, theta_maxiter)
            return self.energy_elec

        def point():
            ds = self.ds
            ds.set_rdms(self.rdm1, self.rdm2)
            F = ds.gen_fock()
            D = np.zeros(F.shape[0])
            D[ds.I] = 1.0
            D[ds.A] = self.rdm1.diagonal().real
            Fd = (ds.FI + ds.FA).diagonal().real
            Fg = F.diagonal().real
            h = np.tile(2 * (D[M] * Fd[N_] + D[N_] * Fd[M] - Fg[M] - Fg[N_]), 2)
            return orbital_gradient(F, idx), np.where(ep, np.minimum(h, -h_min), np.maximum(h, h_min))

        e = relax_thetas()
        g, h = point()
        S, Y = [], []
        for it in range(maxiter):
            t0 = time.time()
            gmax = np.abs(g).max() if nk else 0.0
            if gmax < tol:
                print(f"{it + 1:^9} | {time.time() - t0:^8.2f} | {e:^27.12f} | {gmax:^18.3e}      | converged")
                break
            q = np.where(ee, g, 0.0)
            al = []
            for s_, y_ in zip(S[::-1], Y[::-1]):
                a_ = (s_ @ q) / (y_ @ s_)
                al.append(a_)
                q = q - a_ * y_
            r = np.where(ee, q / h, 0.0)
            for (s_, y_), a_ in zip(zip(S, Y), al[::-1]):
                r = r + s_ * (a_ - (y_ @ r) / (y_ @ s_))
            step = np.where(ee, -r, -g / h)
            if np.abs(step).max() > max_step:
                step *= max_step / np.abs(step).max()
            slope = g[ee] @ step[ee]
            if slope >= 0:
                S, Y = [], []
                step = -g / h
                slope = g[ee] @ step[ee]
            ep_change = abs(np.sum(np.where(ep, g * step + 0.5 * h * step**2, 0.0)))
            C, thetas = self._c_mo.copy(), (self.thetas_real, self.thetas_imag)
            a = 1.0
            for _ in range(15):
                self._set_orbitals(C @ scipy.linalg.expm(-kappa(a * step)))
                e_new = relax_thetas()
                if e_new - e <= 1e-4 * a * slope + ep_change + 1e-10:
                    break
                a *= 0.5
                self.set_thetas(*thetas)
            g_new, h = point()
            s_, y_ = np.where(ee, a * step, 0.0), np.where(ee, g_new - g, 0.0)
            if s_ @ y_ > 1e-12 * np.linalg.norm(s_) * np.linalg.norm(y_):
                S.append(s_)
                Y.append(y_)
                if len(S) > memory:
                    S.pop(0)
                    Y.pop(0)
            j = np.argmax(np.abs(g)) % nk
            print(f"{it + 1:^9} | {time.time() - t0:^8.2f} | {e_new:^27.12f} | {gmax:^11.3e} ({space[M[j]]}{space[N_[j]]}) | {a:.3f}")
            e, g = e_new, g_new
        else:
            print("WARNING: integral-direct OO did not converge")

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
                print(f"### Number kappa: {len(self.kappa_real)}")
            print(f"### Number theta: {self.ups_layout.n_params}")
        if optimizer_name.lower() == "rotosolve":
            if orbital_optimization and len(self.kappa_real) != 0:
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
                thetas = self.thetas_real + self.thetas_imag
                parameters = np.zeros(2 * len(self.kappa_real), dtype=float).tolist() + thetas
            else:
                parameters = np.zeros(2 * len(self.kappa_real), dtype=float).tolist()
        else:
            parameters = self.thetas_real + self.thetas_imag
        optimizer = Optimizers(
            energy,
            optimizer_name,
            grad=gradient,
            maxiter=maxiter,
            tol=tol,
            energy_eval_callback=lambda: self.num_energy_evals,
            is_silent=is_silent,
        )
        self._old_opt_parameters = np.zeros_like(parameters, dtype=np.float64) + 10**20
        self._E_opt_old = 0.0
        res = optimizer.minimize(
            parameters,
            extra_options={"R": self.ups_layout.grad_param_R, "param_names": self.ups_layout.param_names},
        )
        if orbital_optimization:
            if len(self.thetas) > 0:
                thetas_r = []
                thetas_i = []
                for i in range(len(self.thetas)):
                    thetas_r.append(res.x[i + 2 * len(self.kappa_real)])
                    thetas_i.append(res.x[i + 2 * len(self.kappa_real) + len(self.thetas)])
                self.set_thetas(thetas_r, thetas_i)
            for i in range(len(self.kappa_real)):
                self._kappa_real[i] = 0.0
                self._kappa_imag[i] = 0.0
                self._kappa_real_old[i] = 0.0
                self._kappa_imag_old[i] = 0.0
        else:
            thetas_r = []
            thetas_i = []
            for i in range(len(self.thetas)):
                thetas_r.append(res.x[i])
                thetas_i.append(res.x[i + len(self.thetas)])
            self.set_thetas(thetas_r, thetas_i)
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
        - GS, generalized singles.
        - GD, generalized doubles.

        Args:
            operator_pool: Which operators to include in the ADAPT.
            maxiter: Maximum iterations.
            grad_threshold: Convergence threshold based on gradient.
            orbital_optimization: Do orbital optimization.
        """
        excitation_pool: list[tuple[int, ...]] = []
        excitation_pool_type = []
        _operator_pool = [x.lower() for x in operator_pool]
        valid_operators = ("s", "d", "gs", "gd")
        for operator in _operator_pool:
            if operator not in valid_operators:
                raise ValueError(f"Got invalid operator for ADAPT, {operator}")
        if "s" in _operator_pool:
            for a, i in iterate_t1(
                self.active_occ_spin_idx_shifted, self.active_unocc_spin_idx_shifted, is_spin_conserving=False
            ):
                excitation_pool.append((i, a))
                excitation_pool_type.append("single")
        if "d" in _operator_pool:
            for a, i, b, j in iterate_t2(
                self.active_occ_spin_idx_shifted, self.active_unocc_spin_idx_shifted, is_spin_conserving=False
            ):
                excitation_pool.append((i, j, a, b))
                excitation_pool_type.append("double")
        if "gs" in _operator_pool:
            for a, i in iterate_t1_generalized(self.num_active_spin_orbs, is_spin_conserving=False):
                excitation_pool.append((i, a))
                excitation_pool_type.append("single")
        if "gd" in _operator_pool:
            for a, i, b, j in iterate_t2_generalized(self.num_active_spin_orbs, is_spin_conserving=False):
                excitation_pool.append((i, j, a, b))
                excitation_pool_type.append("double")

        print(
            "Iteration # | Iteration time [s] | Electronic energy [Hartree] | max|grad| [Hartree] | Operator"
        )
        start = time.time()
        for iteration in range(maxiter):
            # Hamiltonian = generalized_hamiltonian_0i_0a(
            #     self.h_mo,
            #     self.g_mo,
            #     self.num_inactive_spin_orbs,
            #     self.num_active_spin_orbs,
            # )
            Hamiltonian = DHF_hamiltonian_full_space( #AE rettet virker ikke (H0i_ai)
                self.h_mo,
                self.g_mo,
                self.num_inactive_spin_orbs,
                self.num_active_spin_orbs,
            )
            H_ket = generalized_propagate_state(
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
                gr = generalized_expectation_value(self.ci_coeffs, [T], H_ket, self.ci_info, do_folding=False)
                gr -= generalized_expectation_value(H_ket, [T], self.ci_coeffs, self.ci_info, do_folding=False)
                grad.append(gr)
            if np.max(np.abs(grad)) < grad_threshold:
                break
            max_arg = np.argmax(np.abs(grad))
            self.ups_layout.excitation_indices.append(excitation_pool[max_arg])
            self.ups_layout.excitation_operator_type.append(excitation_pool_type[max_arg])
            self.ups_layout.n_params += 1

            self._thetas_real.append(0.0)
            self._thetas_imag.append(0.0)
            # print("running 1step")
            self.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=orbital_optimization, is_silent=False)
            time_str = f"{time.time() - start:7.2f}"
            e_str = f"{self.energy_elec:3.12f}"
            grad_str = f"{np.abs(grad[max_arg]):3.12f}"
            print(
                f"{str(iteration + 1).center(11)} | {time_str.center(18)} | {e_str.center(27)} | {grad_str.center(19)} | {excitation_pool_type[max_arg]}{tuple([int(x) for x in excitation_pool[max_arg]])}"
            )

    def _calc_energy_optimization_DHF(
        self, parameters: list[float], theta_optimization: bool, kappa_ee_optimization: bool, kappa_ep_optimization : bool, 
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
        # print(self._old_opt_parameters, parameters)
        if np.max(np.abs(np.array(self._old_opt_parameters) - np.array(parameters))) < 10**-14:
            return self._E_opt_old
        num_kappa = 0
        if kappa_ee_optimization:
            num_kappa = 2 * len(self.kappa_spin_idx)
            kappa_r = []
            kappa_i = []
            for i in range(len(self.kappa_real)):
                kappa_r.append(parameters[i])
                kappa_i.append(parameters[i + len(self.kappa_real)])
            self.set_kappa_cep(kappa_r, kappa_i)
        if kappa_ep_optimization:
            num_kappa = 2 * len(self.kappa_spin_idx_ep)
            kappa_r = []
            kappa_i = []
            for i in range(len(self.kappa_real_ep)):
                kappa_r.append(parameters[i])
                kappa_i.append(parameters[i + len(self.kappa_real_ep)])
            self.set_kappa_cep_ep(kappa_r, kappa_i)
        if theta_optimization:
            thetas_r = []
            thetas_i = []
            for i in range(len(self.thetas)):
                thetas_r.append(parameters[i + num_kappa])
                thetas_i.append(parameters[i + num_kappa + len(self.thetas)])
            self.set_thetas(thetas_r, thetas_i)
        if kappa_ee_optimization:
            E = self.ds.energy(self.rdm1, self.rdm2)
        if kappa_ep_optimization:
            E = -self.ds_ep.energy(self.rdm1, self.rdm2)
        if theta_optimization:
            c = self.ci_coeffs
            E = np.vdot(c, self.ds.ci_hamiltonian(self.ci_engine) @ c)
        self._E_opt_old = E.real
        self._old_opt_parameters = np.copy(parameters)
        self.num_energy_evals += 1  # count one measurement
        return E.real

    def _calc_gradient_optimization_DHF(
        self, parameters: list[float], theta_optimization: bool, kappa_ee_optimization: bool, kappa_ep_optimization: bool,
    ) -> np.ndarray:
        """Calculate electronic gradient.

        Args:
            parameters: Ansatz and orbital rotation parameters.
            theta_optimization: If used in theta optimization.
            kappa_optimization: If used in kappa optimization.

        Returns:
            Electronic gradient.
        """
        gradient = np.zeros(len(parameters), dtype=np.float64)
        num_kappa = 0
        if kappa_ee_optimization:
            num_kappa = 2 * len(self.kappa_spin_idx)
            kappa_r = []
            kappa_i = []
            for i in range(len(self.kappa_real)):
                kappa_r.append(parameters[i])
                kappa_i.append(parameters[i + len(self.kappa_real)])
            self.set_kappa_cep(kappa_r, kappa_i)
        if kappa_ep_optimization:
            num_kappa = 2 * len(self.kappa_spin_idx_ep)
            kappa_r = []
            kappa_i = []
            for i in range(len(self.kappa_real_ep)):
                kappa_r.append(parameters[i])
                kappa_i.append(parameters[i + len(self.kappa_real_ep)])
            self.set_kappa_cep_ep(kappa_r, kappa_i)
        if theta_optimization:
            thetas_r = []
            thetas_i = []
            for i in range(len(self.thetas)):
                thetas_r.append(parameters[i + num_kappa])
                thetas_i.append(parameters[i + num_kappa + len(self.thetas)])
            self.set_thetas(thetas_r, thetas_i)
        if kappa_ee_optimization:
            ds = self.ds
            ds.set_rdms(self.rdm1, self.rdm2)
            gradient[:num_kappa] = orbital_gradient(ds.gen_fock(), self.kappa_spin_idx)
        if kappa_ep_optimization:
            ds = self.ds_ep
            ds.set_rdms(self.rdm1, self.rdm2)
            gradient[:num_kappa] = -orbital_gradient(ds.gen_fock(), self.kappa_spin_idx_ep)
            #print("Max e-p gradient:", np.max(np.abs(gradient[:num_kappa])))
        
            # for idx, (M, N) in enumerate(self.kappa_spin_idx):
            #     if M < self.num_spin_orbs_NES:
            #         print(gradient[idx])

            # print(gradient[:num_kappa])

            # gradient[:num_kappa] = get_orbital_gradient_expvalue_real_imag(
            #     self.ci_coeffs,
            #     self.ci_info,
            #     self.h_mo[size:,size:],
            #     self.g_mo[size:,size:,size:,size:],
            #     self.num_spin_orbs_NES,
            #     self.kappa_spin_idx,
            # )

        if theta_optimization:
            # Reference bra state (no differentiations)
            bra_vec = self.ds.ci_hamiltonian(self.ci_engine) @ self.ci_coeffs
            # Reference ket state (no differentiations)
            bra_vec = generalized_construct_ups_state_test_erik(
                bra_vec,
                self.ci_info,
                self.thetas,
                self.ups_layout,
                dagger=True,
            )
            # CSF reference state on ket
            ket_vec = np.copy(self.csf_coeffs)
            # Calculate analytical derivative w.r.t. each theta using gradient_action function
            for i in range(len(self.thetas)):
                # Remove the i-th U^dagger from bra_vec.
                # The effect of this unitary will be included from the derivative added to ket_vec.
                bra_vec = generalized_propagate_unitary_test_anna(
                    bra_vec,
                    i,
                    self.ci_info,
                    self.thetas,
                    self.ups_layout,
                )
                # Derivative action w.r.t. i-th theta on CSF ket
                ket_vec_tmp_R, ket_vec_tmp_I = generalized_get_grad_action(
                    ket_vec,
                    i,
                    self.ci_info,
                    self.thetas,
                    self.ups_layout,
                )
                gradient[i + num_kappa] += 2*np.matmul(bra_vec.conj(), ket_vec_tmp_R).real
                gradient[i + num_kappa + len(self.thetas)] += 2*np.matmul(bra_vec.conj(), ket_vec_tmp_I).real
               
                # Product rule implications on reference bra and CSF ket
                # See 10.48550/arXiv.2303.10825, Eq. 20 (appendix - v1)
                ket_vec = generalized_propagate_unitary_test_anna(
                    ket_vec,
                    i,
                    self.ci_info,
                    self.thetas,
                    self.ups_layout,
                )
            self.num_energy_evals += 2 * np.sum(
                list(self.ups_layout.grad_param_R.values())
            )  # Count energy measurements for all gradients
            #print(np.max(np.abs(gradient)))
            #print(gradient)
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
        # print(self._old_opt_parameters, parameters)
        if np.max(np.abs(np.array(self._old_opt_parameters) - np.array(parameters))) < 10**-14:
            return self._E_opt_old
        num_kappa = 0
        if kappa_optimization:
            num_kappa = 2 * len(self.kappa_spin_idx)
            kappa_r = []
            kappa_i = []
            for i in range(len(self.kappa_real)):
                kappa_r.append(parameters[i])
                kappa_i.append(parameters[i + len(self.kappa_real)])
            self.set_kappa_cep(kappa_r, kappa_i)
        if theta_optimization:
            thetas_r = []
            thetas_i = []
            for i in range(len(self.thetas)):
                thetas_r.append(parameters[i + num_kappa])
                thetas_i.append(parameters[i + num_kappa + len(self.thetas)])
            self.set_thetas(thetas_r, thetas_i)
        if kappa_optimization:
            # RDM is more expensive than evaluation of the Hamiltonian.
            # Thus only construct these if orbital-optimization is turned on,
            # since the RDMs will be reused in the oo gradient calculation.
            NES = self.num_spin_orbs_NES 

            E1 = get_electronic_energy_generalized(
                self.h_mo,
                self.g_mo,
                self.num_spin_orbs_NES,
                self.num_inactive_spin_orbs,
                self.num_active_spin_orbs,
                self.rdm1,
                self.rdm2,
            ) 

            # E2 = generalized_expectation_value_energy(
            #     self.ci_coeffs,
            #     # [generalized_hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_spin_orbs, self.num_active_spin_orbs)],
            #     [DHF_hamiltonian_full_space(self.h_mo[NES:,NES:], self.g_mo[NES:,NES:,NES:,NES:], self.num_spin_orbs_NES)],
            #     self.ci_coeffs,
            #     self.ci_info,
            # )
            E = E1
            #E = 2*E1-E2
            #E = E2
            #print(E)
            #print(E2)
        else:
            NES = self.num_spin_orbs_NES 
            E = generalized_expectation_value_energy(
                self.ci_coeffs,
                # [generalized_hamiltonian_0i_0a(self.h_mo, self.g_mo, self.num_inactive_spin_orbs, self.num_active_spin_orbs)],
                [DHF_hamiltonian_full_space(self.h_mo[NES:,NES:], self.g_mo[NES:,NES:,NES:,NES:], self.num_spin_orbs_NES)],
                self.ci_coeffs,
                self.ci_info,
            )
        self._E_opt_old = E.real
        self._old_opt_parameters = np.copy(parameters)
        self.num_energy_evals += 1  # count one measurement
        return E.real

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
        gradient = np.zeros(len(parameters), dtype=np.float64)
        num_kappa = 0
        if kappa_optimization:
            num_kappa = 2 * len(self.kappa_spin_idx)
            kappa_r = []
            kappa_i = []
            for i in range(len(self.kappa_real)):
                kappa_r.append(parameters[i])
                kappa_i.append(parameters[i + len(self.kappa_real)])
            self.set_kappa_cep(kappa_r, kappa_i)
        if theta_optimization:
            thetas_r = []
            thetas_i = []
            for i in range(len(self.thetas)):
                thetas_r.append(parameters[i + num_kappa])
                thetas_i.append(parameters[i + num_kappa + len(self.thetas)])
            self.set_thetas(thetas_r, thetas_i)
        if kappa_optimization:
            gradient[:num_kappa] = get_orbital_gradient_generalized_real_imag(
                self.h_mo,
                self.g_mo,
                self.kappa_spin_idx,
                self.num_spin_orbs_NES,
                self.num_inactive_spin_orbs,
                self.num_active_spin_orbs,
                self.rdm1,
                self.rdm2,
            )

            # for idx, (M, N) in enumerate(self.kappa_spin_idx):
            #     if M < self.num_spin_orbs_NES:
            #         print(gradient[idx])

            # print(gradient[:num_kappa])

            # gradient[:num_kappa] = get_orbital_gradient_expvalue_real_imag(
            #     self.ci_coeffs,
            #     self.ci_info,
            #     self.h_mo[size:,size:],
            #     self.g_mo[size:,size:,size:,size:],
            #     self.num_spin_orbs_NES,
            #     self.kappa_spin_idx,
            # )

        if theta_optimization:
            Hamiltonian = DHF_hamiltonian_0i_0a(
               self.h_mo,
               self.g_mo,
               self.num_inactive_spin_orbs,
               self.num_active_spin_orbs,
               self.num_spin_orbs_NES,
            )

            #size = self.num_spin_orbs_NES
            # Hamiltonian = DHF_hamiltonian_full_space(
            #     self.h_mo[size:,size:],
            #     self.g_mo[size:,size:,size:,size:],
            #     self.num_spin_orbs_NES,
            # )
            # Reference bra state (no differentiations)
            bra_vec = generalized_propagate_state(
                [Hamiltonian],
                self.ci_coeffs,
                self.ci_info,
            )
            # Reference ket state (no differentiations)
            bra_vec = generalized_construct_ups_state_test_erik(
                bra_vec,
                self.ci_info,
                self.thetas,
                self.ups_layout,
                dagger=True,
            )
            # CSF reference state on ket
            ket_vec = np.copy(self.csf_coeffs)
            # Calculate analytical derivative w.r.t. each theta using gradient_action function
            for i in range(len(self.thetas)):
                # Remove the i-th U^dagger from bra_vec.
                # The effect of this unitary will be included from the derivative added to ket_vec.
                bra_vec = generalized_propagate_unitary_test_anna(
                    bra_vec,
                    i,
                    self.ci_info,
                    self.thetas,
                    self.ups_layout,
                )
                # Derivative action w.r.t. i-th theta on CSF ket
                ket_vec_tmp_R, ket_vec_tmp_I = generalized_get_grad_action(
                    ket_vec,
                    i,
                    self.ci_info,
                    self.thetas,
                    self.ups_layout,
                )
                gradient[i + num_kappa] += 2*np.matmul(bra_vec.conj(), ket_vec_tmp_R).real
                gradient[i + num_kappa + len(self.thetas)] += 2*np.matmul(bra_vec.conj(), ket_vec_tmp_I).real
               
                # Product rule implications on reference bra and CSF ket
                # See 10.48550/arXiv.2303.10825, Eq. 20 (appendix - v1)
                ket_vec = generalized_propagate_unitary_test_anna(
                    ket_vec,
                    i,
                    self.ci_info,
                    self.thetas,
                    self.ups_layout,
                )
            self.num_energy_evals += 2 * np.sum(
                list(self.ups_layout.grad_param_R.values())
            )  # Count energy measurements for all gradients
            #print(np.max(np.abs(gradient)))
        return gradient

    def get_gradient_finite_diff_theta(self,
            parameters: list[complex], 
        ) -> np.ndarray:
            thetas= parameters
            gradient_R = np.zeros(len(thetas), dtype=np.complex128)
            gradient_I = np.zeros(len(thetas), dtype=np.complex128)

            step = 1e-8
            thetas = np.asarray(thetas, dtype=np.complex128)
            p0 = np.concatenate([thetas.real, thetas.imag]).astype(float)
            # print(p0)

            #Real
            for idx in range(len(thetas)):
                p_high = p0.copy()
                p_low  = p0.copy()
                p_high[idx] += step
                p_low[idx]  -= step

                E_high = self._calc_energy_optimization(p_high,
                        theta_optimization=True,
                        kappa_optimization=False)
                # print('E high',E_high)
                E_low  = self._calc_energy_optimization(p_low, theta_optimization=True,
                        kappa_optimization=False)
                # print('E low',E_low)

                gradient_R[idx] = (E_high - E_low) / (2.0 * step)
                
                # print("gr", gradient_R)

            #Imag
            for idx in range(len(thetas)):
                j = idx + len(thetas)
                p_high = p0.copy()
                p_low  = p0.copy()
                p_high[j] += step
                p_low[j]  -= step
                
                E_high = self._calc_energy_optimization(p_high,
                        theta_optimization=True,
                        kappa_optimization=False)
                # print('E high',E_high)
                E_low  = self._calc_energy_optimization(p_low, theta_optimization=True,
                        kappa_optimization=False)
                # print('E low',E_low)


                gradient_I[idx] = (E_high - E_low) / (2.0 * step)
                # print("gi", gradient_I)

            gradient_total = np.concatenate((gradient_R, gradient_I))
            
            print('Finite diff', gradient_total)

            return gradient_total

    @property
    def get_orbital_gradient_generalized_real_imag(self):
        return get_orbital_gradient_generalized_real_imag(
            self.h_mo,
            self.g_mo,
            self.kappa_spin_idx,
            self.num_inactive_spin_orbs,
            self.num_active_spin_orbs,
            self.rdm1,
            self.rdm2,
        )

    @property
    def get_orbital_gradient_generalized_expvalue_real_imag(self):
        return get_orbital_gradient_expvalue_real_imag(
            self.ci_coeffs, self.ci_info, self.h_mo, self.g_mo, self.num_spin_orbs, self.kappa_spin_idx
        )



    def calc_Sz(self, S_int):
            S_int_MO = generalized_1e_transform_spin_separated(self.c_mo[:,self.num_spin_orbs_NES:], S_int)
            return get_Sz(self.rdm1, self.num_inactive_spin_orbs, self.num_active_spin_orbs, S_int_MO)
        
    def calc_S2_old(self, S_int):
        S_int_MO = generalized_1e_transform_spin_separated(self.c_mo[:,self.num_spin_orbs_NES:], S_int)
        return get_S2_old(self.rdm1, self.rdm2, self.num_inactive_spin_orbs, self.num_active_spin_orbs, self.num_virtual_spin_orbs, S_int_MO) 
    
    def calc_S2(self, S_int):
        S_int_MO = generalized_1e_transform_spin_separated(self.c_mo[:,self.num_spin_orbs_NES:], S_int)
        return get_S2(self.rdm1, self.rdm2, self.num_inactive_spin_orbs, self.num_active_spin_orbs, S_int_MO) 
    
    def calc_multiplicity(self, S_int):
        S2 = self.calc_S2(S_int)

        # Protect against tiny negative values due to numerical noise
        if S2 < 0 and abs(S2) < 1e-10:
            S2 = 0.0
        elif S2 < 0:
            raise ValueError(f"Invalid <S^2> = {S2}")

        S = (-1 + np.sqrt(1 + 4*S2)) / 2

        return 2*S + 1
    
    def calc_S(self, S_int):
        S2 = self.calc_S2(S_int)

        # Protect against tiny negative values due to numerical noise
        if S2 < 0 and abs(S2) < 1e-10:
            S2 = 0.0
        elif S2 < 0:
            raise ValueError(f"Invalid <S^2> = {S2}")

        S = (-1 + np.sqrt(1 + 4*S2)) / 2

        return S
    
    def calc_nr_elec(self):
        N = get_nr_elec(self.num_inactive_spin_orbs, self.num_active_spin_orbs, self.rdm1)
        if np.abs(N.imag) > 1e-10:
            print("Warning! Complex Nr of electrons!!")
            return N
        else:
            return N.real


    def calc_mu0(self, S_int):
        S_int_MO = generalized_1e_transform_spin_separated(self.c_mo[:,self.num_spin_orbs_NES:], S_int)
        return get_mu0(self.rdm1, self.rdm2, self.num_inactive_spin_orbs, self.num_active_spin_orbs, S_int_MO) 
    
    def calc_e0(self, S_int):
        S_int_MO = generalized_1e_transform_spin_separated(self.c_mo[:,self.num_spin_orbs_NES:], S_int)
        return get_e0(self.rdm1, self.rdm2, self.num_inactive_spin_orbs, self.num_active_spin_orbs, S_int_MO) 


    def spin_analysis(self, S_int):
        S2 = self.calc_S2(S_int)
        # Protect against tiny negative values due to numerical noise
        if S2 < 0 and abs(S2) < 1e-10:
            S2 = 0.0
        elif S2 < 0:
            raise ValueError(f"Invalid <S^2> = {S2}")
        S = (-1 + np.sqrt(1 + 4*S2)) / 2

        Sz = self.calc_Sz(S_int)
        mu0 = self.calc_mu0(S_int)
        e0 = self.calc_e0(S_int)

        print()
        print("Full characterization of spin for a generalized wavefunction:")
        print("─" * 60)
        print("WARNING: A generalized WF breaks S\u00B2 and Sz symmetry!")
        print("The wavefunction should therefore be characterized by \u03F5\u2080 and \u03bc\u2080")
        print()

        # --- Results table ---
        labels = [
            ("Expectation value of S\u00B2", S2),
            ("Multiplicity (2S + 1)",       2*S + 1),
            ("Value of S",                  S),
            ("Expectation value of Sz",     Sz),
            ("\u03F5\u2080",                e0),
            ("\u03bc\u2080",                mu0),
        ]

        label_width = max(len(lbl) for lbl, _ in labels)

        print("┌" + "─"*(label_width+2) + "┬" + "─"*14 + "┐")
        for lbl, val in labels:
            print(f"│ {lbl:<{label_width}} │ {np.round(val,5):>12} │")
        print("└" + "─"*(label_width+2) + "┴" + "─"*14 + "┘")

        print()
        print("\u03F5\u2080 is the Euclidean norm of the vector (<Sx>, <Sy>, <Sz>)\u1d40,")
        print("and it determines unpaired spin in the direction of maximum unpaired spin.")
        print("If \u03F5\u2080 is an allowed value of Sz for the system and \u03bc\u2080 = 0, the solution is collinear.")
        print("If \u03bc\u2080 \u2260 0, the solution is non-collinear!")
        print()
