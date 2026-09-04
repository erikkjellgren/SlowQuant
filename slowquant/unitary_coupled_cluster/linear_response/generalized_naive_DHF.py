from collections.abc import Sequence

import numpy as np
from scipy.linalg import solve
import scipy.linalg as la
from pyscf.data import nist
from pyscf import lib

c = lib.param.LIGHT_SPEED

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform, generalized_one_electron_transform, DHF_one_electron_transform,
    RMB_GIAO_trans_1e, RMB_GIAO_trans_2e, DHF_two_electron_transform
)
from slowquant.unitary_coupled_cluster.generalized_density_matrix_DHF import (
    get_orbital_gradient_response, get_orbital_gradient_response_real_imag,
    get_orbital_response_hessian_block,
    get_orbital_response_metric_sigma,
    get_orbital_response_property_gradient_annika, get_orbital_response_property_gradient_real_imag, 
    get_orbital_response_metric_sigma_real_imag,  get_orbital_response_static_property_gradient_DHF,
    get_1e_exp_value, 
    get_orbital_gradient_generalized_real_imag, RDM1, get_orbital_response_static_property_gradient_DHF_RMB_GIAO, 
    get_exp_val_RMB_GIAO
)

from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.linear_response.generalized_lr_baseclass_DHF import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import (
    generalized_expectation_value,
    generalized_propagate_state,
)
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    expectation_value,
)
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction_DHF import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.generalized_operators import (
    DHF_one_elec_op_0i_0a, a_op_spin
)

from slowquant.unitary_coupled_cluster.generalized_operators import DHF_hamiltonian_full_space

class LinearResponse(LinearResponseBaseClass):
 import numpy as np

class LinearResponse(LinearResponseBaseClass):
    def __init__(
        self,
        wave_function: GeneralizedWaveFunctionUPS,
        excitations: str,
        screen: bool = True,
        thresh_A: float = 1e-6, 
        thresh_m: float = 1e-6,
    ) -> None:
        """Initialize linear response by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
        """
        super().__init__(wave_function, excitations, screen, thresh_A, thresh_m)

        idx_shift = len(self.q_ops)
        print("Gs", len(self.G_ops))
        print("qs", len(self.q_ops))


        if self.screen:
            # Pernille Screening
            # Screen for A_ii = 0 Pernille
            finite_excitations = []
            if len(self.q_ops) != 0:
                A = get_orbital_response_hessian_block(
                    self.wf.h_mo,
                    self.wf.g_mo,
                    self.wf.kappa_no_activeactive_spin_idx_dagger_resp,
                    self.wf.kappa_no_activeactive_spin_idx_resp,
                    self.wf.num_spin_orbs_NES,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.rdm1,
                    self.wf.rdm2,
                )
            # # Man behøver ikke regne hele A, men det er bare lige nemt at gøre for qq
            for i, q in enumerate(self.q_ops):
                if abs(A[i, i]) > self.thresh_A:  # whatever rimeligt threshold
                    finite_excitations.append(True)
                else:
                    finite_excitations.append(False)
            self.num_q_ops_finite = sum(bool(x) for x in finite_excitations)

            for i, G in enumerate(self.G_ops): 
                GI_ket = generalized_propagate_state([G], self.wf.ci_coeffs, *self.index_info)
                HGI_ket = generalized_propagate_state([self.H_0i_0a, G], self.wf.ci_coeffs, *self.index_info)
                # <0| GId H GJ |0>
                A = generalized_expectation_value(
                    GI_ket,
                    [],
                    HGI_ket,
                    *self.index_info,
                )
                if abs(A) > self.thresh_A:  # whatever rimeligt threshold
                    finite_excitations.append(True)
                else:
                    finite_excitations.append(False)
            self.num_G_ops_finite = sum(bool(x) for x in finite_excitations) - self.num_q_ops_finite

            self.finite_excitations_idx = finite_excitations

            self.q_ops_finite = finite_excitations[:self.num_q_ops_finite]
            self.G_ops_finite = finite_excitations[self.num_q_ops_finite:]

            # Removing operators:
            self.q_ops = [q for q, finite in zip(self.q_ops, self.q_ops_finite) if finite]
            self.G_ops = [G for G, finite in zip(self.G_ops, self.G_ops_finite) if finite]

            print("Gs after screening:", len(self.G_ops))
            print("qs after screening:", len(self.q_ops))

            self.wf.kappa_no_activeactive_spin_idx_resp = [q for q, finite in zip(self.wf.kappa_no_activeactive_spin_idx_resp, self.q_ops_finite) if finite]
            self.wf.kappa_no_activeactive_spin_idx_dagger_resp = [q for q, finite in zip(self.wf.kappa_no_activeactive_spin_idx_dagger_resp, self.q_ops_finite) if finite]

            # Reshaping the hessian and metric elements:
            size = self.num_q_ops_finite + self.num_G_ops_finite

            self.A = np.zeros((size, size), dtype=complex)
            self.B = np.zeros((size, size), dtype=complex)
            self.Sigma = np.zeros((size, size), dtype=complex)
            self.Delta = np.zeros((size, size), dtype=complex)
        

        # q gradient:
        if len(self.q_ops) != 0:
            grad = get_orbital_gradient_response_real_imag(
                self.wf.h_mo,
                self.wf.g_mo,
                self.wf.kappa_no_activeactive_spin_idx_resp,
                self.wf.num_spin_orbs_NES,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1,
                self.wf.rdm2,
            )
            print("idx, max(abs(grad orb)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**3:
                print("Large Gradient detected in q of ", np.max(np.abs(grad)))
                # raise ValueError("Large Gradient detected in q of ", np.max(np.abs(grad)))

        # G gradient:
        grad = np.zeros(2 * len(self.G_ops), dtype=complex) #AE complex
        H00_ket = generalized_propagate_state([self.H_0i_0a], self.wf.ci_coeffs, *self.index_info)
        for i, op in enumerate(self.G_ops):
            G_ket = generalized_propagate_state([op], self.wf.ci_coeffs, *self.index_info)
            Gd_ket = generalized_propagate_state([op.dagger], self.wf.ci_coeffs, *self.index_info)
            # <0 | H G |0>
            grad[i] = generalized_expectation_value(
                H00_ket,
                [],
                G_ket,
                *self.index_info,
            )
            # - <0| G H |0>
            grad[i] -= generalized_expectation_value(
                Gd_ket,
                [],
                H00_ket,
                *self.index_info,
            )
            # <0| Gd H |0>
            grad[i + len(self.G_ops)] = generalized_expectation_value(
                G_ket,
                [],
                H00_ket,
                *self.index_info,
            )
            # - <0| H Gd |0>
            grad[i + len(self.G_ops)] -= generalized_expectation_value(
                H00_ket,
                [],
                Gd_ket,
                *self.index_info,
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                print("Large Gradient detected in G of ", np.max(np.abs(grad)))
                # raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad))) #AE udkommenteret


        # qq Hessian:
        if len(self.q_ops) != 0:
            # Do orbital-orbital blocks
            self.A[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_hessian_block(
                self.wf.h_mo,
                self.wf.g_mo,
                self.wf.kappa_no_activeactive_spin_idx_dagger_resp,
                self.wf.kappa_no_activeactive_spin_idx_resp,
                self.wf.num_spin_orbs_NES, 
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1,
                self.wf.rdm2,
            )   
            self.B[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_hessian_block(
                self.wf.h_mo,
                self.wf.g_mo,
                self.wf.kappa_no_activeactive_spin_idx_dagger_resp,
                self.wf.kappa_no_activeactive_spin_idx_dagger_resp,
                self.wf.num_spin_orbs_NES, 
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1,
                self.wf.rdm2,
            )
            self.Sigma[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_metric_sigma(
                self.wf.kappa_no_activeactive_spin_idx_resp,
                self.wf.kappa_no_activeactive_spin_idx_resp,
                self.wf.num_spin_orbs_NES, 
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1,
            )

        # qG Hessian
        if len(self.q_ops) != 0:                          
            for j, qJ in enumerate(self.q_ops): #self.H_1i_1a
                Hq_ket = generalized_propagate_state([self.H_1i_1a * qJ], self.wf.ci_coeffs, *self.index_info) # do_unsafe=True
                qdH_ket = generalized_propagate_state([qJ.dagger * self.H_1i_1a], self.wf.ci_coeffs, *self.index_info)
                for i, GI in enumerate(self.G_ops):
                    G_ket = generalized_propagate_state([GI], self.wf.ci_coeffs, *self.index_info)
                    Gd_ket = generalized_propagate_state([GI.dagger], self.wf.ci_coeffs, *self.index_info)
                    # print("qG",i,j)
                    # # Make A
                    # <0| Gd H q |0>
                    val = generalized_expectation_value(
                        G_ket,
                        [],
                        Hq_ket,
                        *self.index_info,
                    )
                    # - 1/2<0| H q Gd |0>
                    val -= (
                        1
                        / 2
                        * generalized_expectation_value(
                            qdH_ket,
                            [],
                            Gd_ket,
                            *self.index_info,
                        )
                    )
                    # - 1/2<0| H Gd q |0>
                    val -= (
                        1
                        / 2
                        * generalized_expectation_value(
                            self.wf.ci_coeffs,
                            [self.H_1i_1a * GI.dagger * qJ],
                            self.wf.ci_coeffs,
                            *self.index_info,
                        )
                    )

                    self.A[i + idx_shift, j] = val
                    self.A[j, i + idx_shift] = val.conj()
                    
                    # Make B
                    # <0| qd H Gd |0>
                    val = generalized_expectation_value(
                        Hq_ket,
                        [],
                        Gd_ket,
                        *self.index_info,
                    )
                    # - 1/2*<0| Gd qd H |0>
                    val -= (
                        1
                        / 2
                        * generalized_expectation_value(
                            G_ket,
                            [],
                            qdH_ket,
                            *self.index_info,
                        )
                    )
                    # - 1/2*<0| qd Gd H |0>
                    val -= (
                        1
                        / 2
                        * generalized_expectation_value(
                            self.wf.ci_coeffs,
                            [qJ.dagger * GI.dagger * self.H_1i_1a],
                            self.wf.ci_coeffs,
                            *self.index_info,
                        )
                    )

                    self.B[i + idx_shift, j] = val
                    self.B[j, i + idx_shift] = val


        # GG Hessian
        for j, GJ in enumerate(self.G_ops): #self.H_0i_0a
            GJH_ket = generalized_propagate_state([GJ], H00_ket, *self.index_info)
            GJdH_ket = generalized_propagate_state([GJ.dagger], H00_ket, *self.index_info)
            HGJd_ket = generalized_propagate_state([self.H_0i_0a, GJ.dagger], self.wf.ci_coeffs, *self.index_info)
            HGJ_ket = generalized_propagate_state([self.H_0i_0a, GJ], self.wf.ci_coeffs, *self.index_info)
            GJ_ket = generalized_propagate_state([GJ], self.wf.ci_coeffs, *self.index_info)
            GJd_ket = generalized_propagate_state([GJ.dagger], self.wf.ci_coeffs, *self.index_info)
            for i, GI in enumerate(self.G_ops[j:], j):
                GI_ket = generalized_propagate_state([GI], self.wf.ci_coeffs, *self.index_info)
                GId_ket = generalized_propagate_state([GI.dagger], self.wf.ci_coeffs, *self.index_info)
                # print("GG",i,j)
                # Make A
                # <0| GId H GJ |0> #problemer med H0iai 
                val = generalized_expectation_value(
                    GI_ket,
                    [],
                    HGJ_ket,
                    *self.index_info,
                )

                # <0| GJ H GId |0>
                val += generalized_expectation_value(
                    HGJd_ket,
                    [],
                    GId_ket,
                    *self.index_info,
                )

                # - 1/2<0| GId GJ H |0>
                val -= (
                    1/2
                    * generalized_expectation_value(
                        GI_ket,
                        [],
                        GJH_ket,
                        *self.index_info,
                    )
                ) 

                # - 1/2*<0| H GJ GId |0>
                val -= (
                    1/2
                    * generalized_expectation_value(
                        GJdH_ket,
                        [],
                        GId_ket,
                        *self.index_info,
                    )
                )

                # - 1/2*<0| GJ GId H |0>
                val -= (
                    1
                    / 2
                    * generalized_expectation_value(
                        GJd_ket,
                        [GI.dagger],
                        H00_ket,
                        *self.index_info,
                    )
                )

                # - 1/2*<0| H GId GJ |0>
                val -= (
                    1
                    / 2
                    * generalized_expectation_value(
                        H00_ket,
                        [GI.dagger],
                        GJ_ket,
                        *self.index_info,
                    )
                )

                self.A[i + idx_shift, j + idx_shift] = val 
                self.A[j + idx_shift, i + idx_shift] = val.conj()
                    
                # Make B
                # <0| GId H GJd |0>
                val = generalized_expectation_value(
                    GI_ket,
                    [],
                    HGJd_ket,
                    *self.index_info,
                )
                # - <0| GId GJd H |0>
                val -= generalized_expectation_value(
                    GI_ket,
                    [],
                    GJdH_ket,
                    *self.index_info,
                )
                # - <0| H GJd GId |0>
                val -= generalized_expectation_value(
                    GJH_ket,
                    [],
                    GId_ket,
                    *self.index_info,
                )
                # <0| GJd H GId |0>
                val += generalized_expectation_value(
                    HGJ_ket,
                    [],
                    GId_ket,
                    *self.index_info,
                )

                self.B[i + idx_shift, j + idx_shift] = val 
                self.B[j + idx_shift, i + idx_shift] = val
                # Make Sigma
                # <0| GId GJ |0>
                val = generalized_expectation_value(
                    GI_ket,
                    [],
                    GJ_ket,
                    *self.index_info,
                )
                # - <0| GJ GId |0>
                val -= generalized_expectation_value(
                    GJd_ket,
                    [],
                    GId_ket,
                    *self.index_info,
                )

                # Does it also need to have the hermetian conjugate added??
                # Where is the Delta matrix? 

                self.Sigma[i + idx_shift, j + idx_shift] =  val   
                self.Sigma[j + idx_shift, i + idx_shift] =  val.conj()

        # Checking the Matrices       
        # Check hermiticity of the Hessian:
        size = len(self.A)
        E2 = np.zeros((size * 2, size * 2), dtype=complex) #AE complex
        E2[:size, :size] = self.A
        E2[:size, size:] = self.B
        E2[size:, :size] = self.B.conjugate() #AE added conjugtate 
        E2[size:, size:] = self.A.conjugate() #AE added conjugtate 

        S = np.zeros((size * 2, size * 2), dtype=complex) #AE complex
        S[:size, :size] = self.Sigma
        S[:size, size:] = self.Delta
        S[size:, :size] = -self.Delta.conjugate()
        S[size:, size:] = -self.Sigma.conjugate()

        print(f"Hermiticity check of the Hessian: max|E2 - E2†| = "
            f"{np.max(np.abs(E2 - E2.conj().T)):.2e}")  

        print(f"Hermiticity check of A: max|A - A†| = "
            f"{np.max(np.abs(self.A - self.A.conj().T)):.2e}")  

        print(f"Symmetry check of B: max|B - B.T| = "
            f"{np.max(np.abs(self.B - self.B.T)):.2e}")  

        # Check hermiticity of the Metric:
        print(f"Hermiticity check of the metric: max|S - S†| = "
            f"{np.max(np.abs(self.Sigma - self.Sigma.conj().T)):.2e}")

        self.hessian = E2
        self.metric = S


                                
    def get_transition_dipole(self, dipole_integrals: Sequence[np.ndarray]) -> np.ndarray:
        """Calculate transition dipole moment.

        Args:
            dipole_integrals: Dipole integrals ordered as (x,y,z).

        Returns:
            Transition dipole moment.
        """
        if len(dipole_integrals) != 3:
            raise ValueError(f"Expected 3 dipole integrals got {len(dipole_integrals)}")
        number_excitations = len(self.excitation_energies)
        mux = generalized_one_electron_transform(self.wf.c_mo, dipole_integrals[0])
        muy = generalized_one_electron_transform(self.wf.c_mo, dipole_integrals[1])
        muz = generalized_one_electron_transform(self.wf.c_mo, dipole_integrals[2])
        mux_op = DHF_one_elec_op_0i_0a(
            mux,
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
        )
        muy_op = DHF_one_elec_op_0i_0a(
            muy,
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
        )
        muz_op = DHF_one_elec_op_0i_0a(
            muz,
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
        )
        mux_ket = generalized_propagate_state([mux_op], self.wf.ci_coeffs, *self.index_info)
        muxd_ket = generalized_propagate_state([mux_op.dagger], self.wf.ci_coeffs, *self.index_info)
        muy_ket = generalized_propagate_state([muy_op], self.wf.ci_coeffs, *self.index_info)
        muyd_ket = generalized_propagate_state([muy_op.dagger], self.wf.ci_coeffs, *self.index_info)
        muz_ket = generalized_propagate_state([muz_op], self.wf.ci_coeffs, *self.index_info)
        muzd_ket = generalized_propagate_state([muz_op.dagger], self.wf.ci_coeffs, *self.index_info)
        transition_dipole_x = 0.0 + 0.0j
        transition_dipole_y = 0.0 + 0.0j
        transition_dipole_z = 0.0 + 0.0j
        transition_dipoles = np.zeros((number_excitations, 3), dtype=np.complex128)
        for state_number in range(number_excitations):
            transfer_op = FermionicOperator({})
            for i, G in enumerate(self.G_ops):
                transfer_op += (
                    self.Z_G_normed[i, state_number] * G.dagger + self.Y_G_normed[i, state_number] * G
                )
            q_part_x = 0.0
            q_part_y = 0.0
            q_part_z = 0.0
            if len(self.q_ops) != 0:
                q_part_x = get_orbital_response_property_gradient_annika(
                    mux,
                    self.wf.kappa_no_activeactive_spin_idx,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.rdm1,
                    self.normed_response_vectors,
                    state_number,
                    number_excitations,
                )
                q_part_y = get_orbital_response_property_gradient_annika(
                    muy,
                    self.wf.kappa_no_activeactive_spin_idx,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.rdm1,
                    self.normed_response_vectors,
                    state_number,
                    number_excitations,
                )
                q_part_z = get_orbital_response_property_gradient_annika(
                    muz,
                    self.wf.kappa_no_activeactive_spin_idx,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.rdm1,
                    self.normed_response_vectors,
                    state_number,
                    number_excitations,
                )
            transfer_ket = generalized_propagate_state([transfer_op], self.wf.ci_coeffs, *self.index_info)
            transferd_ket = generalized_propagate_state([transfer_op.dagger], self.wf.ci_coeffs, *self.index_info)
            # <0| mux T |0>
            transition_dipole_x = generalized_expectation_value(
                muxd_ket,
                [],
                transfer_ket,
                *self.index_info,
            )
            # - <0| T mux |0>
            transition_dipole_x -= generalized_expectation_value(
                transferd_ket,
                [],
                mux_ket,
                *self.index_info,
            )
            # <0| muy T |0>
            transition_dipole_y = generalized_expectation_value(
                muyd_ket,
                [],
                transfer_ket,
                *self.index_info,
            )
            # - <0| T muy |0>
            transition_dipole_y -= generalized_expectation_value(
                transferd_ket,
                [],
                muy_ket,
                *self.index_info,
            )
            # <0| muz T |0>
            transition_dipole_z = generalized_expectation_value(
                muzd_ket,
                [],
                transfer_ket,
                *self.index_info,
            )
            # - <0| T muz |0>
            transition_dipole_z -= generalized_expectation_value(
                transferd_ket,
                [],
                muz_ket,
                *self.index_info,
            )
            transition_dipoles[state_number, 0] = q_part_x + transition_dipole_x
            transition_dipoles[state_number, 1] = q_part_y + transition_dipole_y
            transition_dipoles[state_number, 2] = q_part_z + transition_dipole_z
        return transition_dipoles
    
    def get_oscillator_strengths(self, dipole_integrals):
        # Check if the excitation energies have been calculated:
        if not hasattr(self, "excitation_energies"):
            self.excitation_energies = self.calc_excitation_energies()
        # Calculate the transition dipole moments:
        tdm = self.get_transition_dipole(dipole_integrals)
        # Oscillator strengths:
        return np.round((2/3*np.multiply(self.excitation_energies,(np.square(tdm[:,0])+np.square(tdm[:,1])+np.square(tdm[:,2])))).real,8)



    # Property gradients: 
    def get_property_gradient_4comp(self, property_integrals: np.ndarray) -> np.ndarray:
        in_shape = property_integrals.shape[:-2]
        size_mo = self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs + self.wf.num_virtual_spin_orbs
        num_mo = len(property_integrals)
        mo = np.zeros((num_mo, size_mo, size_mo), dtype=complex)
        for i, ao in enumerate(property_integrals):
            mo[i, :, :] += DHF_one_electron_transform(self.wf.c_mo, ao)

        idx_shift_q = len(self.q_ops)
        V = np.zeros((len(self.q_ops + self.G_ops), num_mo), dtype=complex)

        if len(self.q_ops) != 0:
            V[:idx_shift_q, :] = get_orbital_response_static_property_gradient_DHF(
                mo,
                self.wf.kappa_no_activeactive_spin_idx_resp,
                self.wf.num_spin_orbs_NES,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1,
            )
        for idx, G in enumerate(self.G_ops):
            G_ket = generalized_propagate_state([G], self.wf.ci_coeffs, *self.index_info)
            Gd_ket = generalized_propagate_state([G.dagger], self.wf.ci_coeffs, *self.index_info)
            for i in range(self.wf.num_spin_orbs_NES, self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs):
                E_ket = generalized_propagate_state([a_op_spin(i,True), a_op_spin(i,False)], self.wf.ci_coeffs, *self.index_info)
                val = generalized_expectation_value(Gd_ket, [], E_ket, *self.index_info)
                val -= generalized_expectation_value(E_ket, [], G_ket, *self.index_info)
                V[idx + idx_shift_q, :] += mo[:, i, i] * val
            for p in range(self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs, self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs):
                for q in range(self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs, self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs):
                    E_ket = generalized_propagate_state([a_op_spin(p,True)*a_op_spin(q,False)], self.wf.ci_coeffs, *self.index_info)
                    Ed_ket = generalized_propagate_state([a_op_spin(q,True)*a_op_spin(p,False)], self.wf.ci_coeffs, *self.index_info)
                    val = generalized_expectation_value(Gd_ket, [], E_ket, *self.index_info)
                    val -= generalized_expectation_value(Ed_ket, [], G_ket, *self.index_info)
                    V[idx + idx_shift_q, :] += mo[:, p, q] * val

        # Determine hermiticity per component to set correct sign of lower block
        lower_V = np.zeros_like(V)
        for i in range(num_mo):
            if np.allclose(mo[i], mo[i].conj().T, atol=1e-10):
                # Hermitian operator: lower block is -V*
                lower_V[:, i] = -V[:, i].conj()
            else:
                # Anti-Hermitian operator: lower block is +V*
                lower_V[:, i] = V[:, i].conj()

        return np.vstack((V, lower_V)).reshape(-1, *in_shape)

    def get_property_gradient_4comp_RMB_GIAO(self, property_integrals: np.ndarray, property_integrals_2: np.ndarray, s_B: np.ndarray) -> np.ndarray:
        in_shape = property_integrals.shape[:-2]
        size_mo = self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs + self.wf.num_virtual_spin_orbs
        num_mo = len(property_integrals)

        # Hx AO -> MO:
        mo = np.zeros((num_mo, size_mo, size_mo), dtype=complex)
        for i, ao in enumerate(property_integrals):
            mo[i, :, :] += DHF_one_electron_transform(self.wf.c_mo, ao)

        # SB AO -> MO:
        mo_S = np.zeros((num_mo, size_mo, size_mo), dtype=complex)
        for i, ao in enumerate(s_B):
            mo_S[i, :, :] += DHF_one_electron_transform(self.wf.c_mo, ao)

        # g_B AO -> MO:
        mo2e = np.zeros((num_mo, size_mo, size_mo, size_mo, size_mo), dtype=complex)
        for i, ao in enumerate(property_integrals_2):
            mo2e[i, :, :, :, :] += DHF_two_electron_transform(self.wf.c_mo, ao)

        # Hx UMO -> OMO:
        mo_trans = []
        for i in range(3):
            mo_trans.append(RMB_GIAO_trans_1e(mo[i], self.wf.h_mo, mo_S[i]))

        # g_B UMO -> OMO:
        mo2e_trans = []
        for i in range(3):
            mo2e_trans.append(RMB_GIAO_trans_2e(mo2e[i], self.wf.g_mo, mo_S[i]))

        mo_trans = np.array(mo_trans)
        mo2e_trans = np.array(mo2e_trans)

        # q part of the property gradient:
        idx_shift_q = len(self.q_ops)
        V = np.zeros((len(self.q_ops + self.G_ops), num_mo), dtype=complex)

        if len(self.q_ops) != 0:
            V[:idx_shift_q, :] = get_orbital_response_static_property_gradient_DHF_RMB_GIAO(
                mo_trans,
                mo2e_trans,
                self.wf.kappa_no_activeactive_spin_idx_resp,
                self.wf.num_spin_orbs_NES,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1,
                self.wf.rdm2
            )

        # G part of the property gradient
        for idx, G in enumerate(self.G_ops):
            G_ket = generalized_propagate_state([G], self.wf.ci_coeffs, *self.index_info)
            Gd_ket = generalized_propagate_state([G.dagger], self.wf.ci_coeffs, *self.index_info)
            for i in range(self.wf.num_spin_orbs_NES, self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs):
                # 1e contribution to the G part of the property gradient:
                E_ket = generalized_propagate_state([a_op_spin(i,True), a_op_spin(i,False)], self.wf.ci_coeffs, *self.index_info)
                val = generalized_expectation_value(Gd_ket, [], E_ket, *self.index_info)
                val -= generalized_expectation_value(E_ket, [], G_ket, *self.index_info)
                V[idx + idx_shift_q, :] += mo_trans[:, i, i] * val

                for j in range(self.wf.num_spin_orbs_NES, self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs):
                    # 2e contribution to the G part of the property gradient:
                    E_ket = generalized_propagate_state([a_op_spin(i,True)*a_op_spin(j,True)*a_op_spin(j,False)*a_op_spin(i,False)], self.wf.ci_coeffs, *self.index_info)
                    Ed_ket = generalized_propagate_state([a_op_spin(i,True)*a_op_spin(j,True)*a_op_spin(j,False)*a_op_spin(i,False)], self.wf.ci_coeffs, *self.index_info)
                    val = generalized_expectation_value(Gd_ket, [], E_ket, *self.index_info)
                    val -= generalized_expectation_value(Ed_ket, [], G_ket, *self.index_info)
                    V[idx + idx_shift_q, :] += mo2e_trans[:, i, i, j, j] * val *.5
                    E_ket = generalized_propagate_state([a_op_spin(j,True)*a_op_spin(i,True)*a_op_spin(j,False)*a_op_spin(i,False)], self.wf.ci_coeffs, *self.index_info)
                    Ed_ket = generalized_propagate_state([a_op_spin(i,True)*a_op_spin(j,True)*a_op_spin(i,False)*a_op_spin(j,False)], self.wf.ci_coeffs, *self.index_info)
                    val = generalized_expectation_value(Gd_ket, [], E_ket, *self.index_info)
                    val -= generalized_expectation_value(Ed_ket, [], G_ket, *self.index_info)
                    V[idx + idx_shift_q, :] += mo2e_trans[:, i, j, j, i] * val * .5

                for p in range(self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs, self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs):
                    for q in range(self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs, self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs):
                        # 2e contribution to the G part of the property gradient:
                        E_ket = generalized_propagate_state([a_op_spin(i,True)*a_op_spin(p,True)*a_op_spin(q,False)*a_op_spin(i,False)], self.wf.ci_coeffs, *self.index_info)
                        Ed_ket = generalized_propagate_state([a_op_spin(i,True)*a_op_spin(q,True)*a_op_spin(p,False)*a_op_spin(i,False)], self.wf.ci_coeffs, *self.index_info)
                        val = generalized_expectation_value(Gd_ket, [], E_ket, *self.index_info)
                        val -= generalized_expectation_value(Ed_ket, [], G_ket, *self.index_info)
                        V[idx + idx_shift_q, :] += mo2e_trans[:, i, i, p, q] * val * .5
                        E_ket = generalized_propagate_state([a_op_spin(p,True)*a_op_spin(i,True)*a_op_spin(i,False)*a_op_spin(q,False)], self.wf.ci_coeffs, *self.index_info)
                        Ed_ket = generalized_propagate_state([a_op_spin(q,True)*a_op_spin(i,True)*a_op_spin(i,False)*a_op_spin(p,False)], self.wf.ci_coeffs, *self.index_info)
                        val = generalized_expectation_value(Gd_ket, [], E_ket, *self.index_info)
                        val -= generalized_expectation_value(Ed_ket, [], G_ket, *self.index_info)
                        V[idx + idx_shift_q, :] += mo2e_trans[:, p, q, i, i] * val * .5
                        E_ket = generalized_propagate_state([a_op_spin(p,True)*a_op_spin(i,True)*a_op_spin(q,False)*a_op_spin(i,False)], self.wf.ci_coeffs, *self.index_info)
                        Ed_ket = generalized_propagate_state([a_op_spin(i,True)*a_op_spin(q,True)*a_op_spin(i,False)*a_op_spin(p,False)], self.wf.ci_coeffs, *self.index_info)
                        val = generalized_expectation_value(Gd_ket, [], E_ket, *self.index_info)
                        val -= generalized_expectation_value(Ed_ket, [], G_ket, *self.index_info)
                        V[idx + idx_shift_q, :] += mo2e_trans[:, p, i, i, q] * val * .5
                        E_ket = generalized_propagate_state([a_op_spin(i,True)*a_op_spin(q,True)*a_op_spin(i,False)*a_op_spin(p,False)], self.wf.ci_coeffs, *self.index_info)
                        Ed_ket = generalized_propagate_state([a_op_spin(p,True)*a_op_spin(i,True)*a_op_spin(q,False)*a_op_spin(i,False)], self.wf.ci_coeffs, *self.index_info)
                        val = generalized_expectation_value(Gd_ket, [], E_ket, *self.index_info)
                        val -= generalized_expectation_value(Ed_ket, [], G_ket, *self.index_info)
                        V[idx + idx_shift_q, :] += mo2e_trans[:, i, p, q, i] * val * .5

            for p in range(self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs, self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs):
                for q in range(self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs, self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs):
                    # 1e contribution to the G part of the property gradient:
                    E_ket = generalized_propagate_state([a_op_spin(p,True)*a_op_spin(q,False)], self.wf.ci_coeffs, *self.index_info)
                    Ed_ket = generalized_propagate_state([a_op_spin(q,True)*a_op_spin(p,False)], self.wf.ci_coeffs, *self.index_info)
                    val = generalized_expectation_value(Gd_ket, [], E_ket, *self.index_info)
                    val -= generalized_expectation_value(Ed_ket, [], G_ket, *self.index_info)
                    V[idx + idx_shift_q, :] += mo_trans[:, p, q] * val

                    for r in range(self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs, self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs):
                        for s in range(self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs, self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs):
                            # 2e contribution to the G part of the property gradient:
                            E_ket = generalized_propagate_state([a_op_spin(p,True)*a_op_spin(r,True)*a_op_spin(s,False)*a_op_spin(q,False)], self.wf.ci_coeffs, *self.index_info)
                            Ed_ket = generalized_propagate_state([a_op_spin(q,True)*a_op_spin(s,True)*a_op_spin(r,False)*a_op_spin(p,False)], self.wf.ci_coeffs, *self.index_info)
                            val = generalized_expectation_value(Gd_ket, [], E_ket, *self.index_info)
                            val -= generalized_expectation_value(Ed_ket, [], G_ket, *self.index_info)
                            V[idx + idx_shift_q, :] += mo2e_trans[:, p, q, r, s] * val * .5

            # Naive
            # for p in range(self.wf.num_spin_orbs_NES, self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs):
            #     for q in range(self.wf.num_spin_orbs_NES, self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs):
            #         for r in range(self.wf.num_spin_orbs_NES, self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs):
            #             for s in range(self.wf.num_spin_orbs_NES, self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs):
            #                 pass
                            # 2e contribution to the G part of the property gradient:
                            # E_ket = generalized_propagate_state([a_op_spin(p,True)*a_op_spin(r,True)*a_op_spin(s,False)*a_op_spin(q,False)], self.wf.ci_coeffs, *self.index_info)
                            # Ed_ket = generalized_propagate_state([a_op_spin(q,True)*a_op_spin(s,True)*a_op_spin(r,False)*a_op_spin(p,False)], self.wf.ci_coeffs, *self.index_info)
                            # val = generalized_expectation_value(Gd_ket, [], E_ket, *self.index_info)
                            # val -= generalized_expectation_value(Ed_ket, [], G_ket, *self.index_info)
                            # V[idx + idx_shift_q, :] += mo2e_trans[:, p, q, r, s] * val * .5

        # Determine hermiticity per component to set correct sign of lower block
        lower_V = np.zeros_like(V)
        for i in range(num_mo):
            if np.allclose(mo_trans[i], mo_trans[i].conj().T, atol=1e-10):
                # Hermitian operator: lower block is -V*
                lower_V[:, i] = -V[:, i].conj()
            else:
                # Anti-Hermitian operator: lower block is +V*
                lower_V[:, i] = V[:, i].conj()

        return np.vstack((V, lower_V)).reshape(-1, *in_shape)



    # Working functions for shieldings and coupling constants: 
    def get_SSCC_4comp_iso(self, h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
        h1_int = np.zeros_like(h1)
        natm = h1.shape[0]

        for I in range(natm):
            for a in range(3):
                #h1_int[I, a] = DHF_one_electron_transform(self.wf.c_mo, h1[I, a])
                h1_int[I, a] = h1[I, a]

        test = True

        # Property gradients and responses for all nuclei
        # prop_grads = [self.get_property_gradient_4comp(h1_int[I]) for I in range(natm)]

        # responses = [
        #     np.linalg.pinv(E2, rcond=1e-6) @ prop_grads_eff[I]
        #     for I in range(natm)
        # ]

        # responses  = [solve(E2_mat, prop_grads[I]) for I in range(natm)]

        nuc_mag = 0.5 * (nist.E_MASS / nist.PROTON_MASS)
        au2Hz   = nist.HARTREE2J / nist.PLANCK

        if test == True:
            # AO -> MO transformation of the integrals needed for the diamagnetic contribution: 
            natm = h2.shape[0]

            size_mo = (
                self.wf.num_spin_orbs_NES +
                self.wf.num_inactive_spin_orbs +
                self.wf.num_active_spin_orbs +
                self.wf.num_virtual_spin_orbs
            )

            mo = np.zeros((natm, natm, 3, 3, size_mo, size_mo), dtype=complex)
            nuc_pair = []

            for I in range(natm):
                for J in range(I + 1, natm):
                    nuc_pair.append((I, J))
                    for a in range(3):
                        for b in range(3):
                            ao = h2[I, J, a, b]

                            mo[I, J, a, b] = mo[J, I, a, b] = DHF_one_electron_transform(
                                self.wf.c_mo,
                                ao
                            )

            # Making the diamagnetic contribution: 
            ssc_dia = np.zeros((natm, natm, 3, 3), dtype=complex)
   
            for k, (I, J) in enumerate(nuc_pair):
                for x in range(3):
                    for y in range(3):

                        val = 0.0 + 0.0j

                        for i in range(size_mo):
                            for j in range(size_mo):

                                val += (
                                    mo[I, J, x, y, i, j]
                                    *RDM1(i, j, self.wf.num_spin_orbs_NES, self.wf.num_inactive_spin_orbs, self.wf.num_active_spin_orbs, self.wf.rdm1)
                                )

                        ssc_dia[I, J, x, y] = ssc_dia[J, I, x, y] = val

            # Making the paramagnetic contribution:
            ssc_para = np.zeros((natm, natm, 3, 3), dtype=np.complex128)

            for k, (I, J) in enumerate(nuc_pair):
                for alpha in range(3):
                    for beta in range(3):
                        ssc_para[I, J, alpha, beta] = ssc_para[J, I, alpha, beta] = np.einsum('i,i->',
                                    -prop_grads_eff[I][:, alpha].conj(), responses[J][:, beta]).real
                        
            # Factors:
            ssc_dia *= nist.ALPHA**4
            ssc_para *= nist.ALPHA**4

            ktensor = np.zeros((natm, natm))

            for k, (I, J) in enumerate(nuc_pair):
                ktensor[I, J] = ktensor[J, I] = au2Hz * nuc_mag ** 2 * np.trace(ssc_para[I, J] + ssc_dia[I, J]).real / 3 
                #ktensor[I, J] = ktensor[J, I] = au2Hz * nuc_mag ** 2 * np.trace(ssc_para[I, J]).real / 3 

                print("Diamagnetic contribution:")
                print(np.round(ssc_dia[I,J].real, 10))
                print("Paramagnetic contribution:")
                print(np.round(ssc_para[I,J].real, 10))
            
        else:
            iso_ssc  = []
            nuc_pair = []

            for I in range(natm):
                for J in range(I+1, natm):
                    nuc_pair.append((I, J))

                    K_tensor = np.zeros((3, 3), dtype=np.complex128)     

                    for alpha in range(3):
                        for beta in range(3):
                            K_tensor[alpha, beta] = np.einsum(
                                'i,i->',
                                -prop_grads[I][:, alpha].conj(),
                                responses[J][:, beta]
                            ).real

                    K_tensor = (
                        K_tensor
                        * nist.ALPHA**4
                        * au2Hz
                        * nuc_mag**2
                    )

                    iso_K = np.trace(K_tensor).real / 3
                    iso_ssc.append(iso_K)

            ktensor = np.zeros((natm, natm))
            for k, (i, j) in enumerate(nuc_pair):
                ktensor[i, j] = ktensor[j, i] = iso_ssc[k]

        return ktensor  # reduced K (Hz), (natm, natm)
    
    def get_shieldings_4comp_iso(self, RMB_GIAO = True, output = False) -> np.ndarray:
        # Integrals:
        h_m = self.wf.int_gen.h_m
        if RMB_GIAO:
            h_Bm = self.wf.int_gen.h_Bm_RMB_GIAO
            h_B = self.wf.int_gen.h_B_RMB_GIAO
            g_B = self.wf.int_gen.g_B
            s_B = self.wf.int_gen.S_B
        else:
            h_B = self.wf.int_gen.h_B

        # Linear response contribution:
        # Property gradients:
        natm = h_m.shape[0]
        prop_grads_m =  [self.get_property_gradient_4comp(h_m[I]) for I in range(natm)]

        if RMB_GIAO:
            prop_grads_B  = self.get_property_gradient_4comp_RMB_GIAO(h_B, g_B, s_B)
        else:
            prop_grads_B =  self.get_property_gradient_4comp(h_B)

        # Also screening in the metric:
        def solve_lr_drop_sigma_null(H, sigma, prop_grads1=None, prop_grads2=None, cut=self.thresh_m):
            Hh = 0.5 * (H + H.conj().T)
            Sh = 0.5 * (sigma + sigma.conj().T)

            s, U = la.eigh(Sh)
            scale = np.max(np.abs(s))

            if scale == 0:
                raise ValueError("Metric sigma is identically zero.")

            Uk = U[:, np.abs(s) > cut * scale]

            Hk = Uk.conj().T @ Hh @ Uk
            Sk = Uk.conj().T @ Sh @ Uk

            prop_grads_k1 = None
            if prop_grads1 is not None:
                prop_grads_k1 = np.einsum(
                    "ji,ajk->aik",
                    Uk.conj(),
                    np.asarray(prop_grads1),
                    optimize=True,
                )

            prop_grads_k2 = None
            if prop_grads2 is not None:
                prop_grads_k2 = Uk.conj().T @ np.asarray(prop_grads2)

            return Hk, Sk, prop_grads_k1, prop_grads_k2

        #H_T, S_T, prop_grads_m_T, prop_grads_B_T = solve_lr_drop_sigma_null(self.hessian, self.metric, prop_grads_m, prop_grads_B)

        # Solving for responses:
        response_B  = solve(self.hessian, prop_grads_B)
        #response_B_T = solve(H_T, prop_grads_B_T)

        # Calculating linear response contribution:
        sc_resp = np.zeros((natm, 3, 3))

        for I in range(natm):
            for alpha in range(3):
                for beta in range(3):
                    sc_resp[I, alpha, beta] = np.einsum(
                        'i,i->',
                        -prop_grads_m[I][:, alpha].conj(),
                        response_B[:, beta]
                    ).real

        if RMB_GIAO:
            # Expectation value contribution:
            size_mo = (
                self.wf.num_spin_orbs_NES
                + self.wf.num_inactive_spin_orbs
                + self.wf.num_active_spin_orbs
                + self.wf.num_virtual_spin_orbs
            )

            # HBm AO -> MO:
            mo_exp = np.zeros((natm, 3, 3, size_mo, size_mo), dtype=complex)

            for I in range(natm):
                for a in range(3):
                    for b in range(3):
                        mo_exp[I, a, b, :, :] = DHF_one_electron_transform(
                            self.wf.c_mo,
                            h_Bm[I, a, b]
                        )

            # SB AO -> MO:
            mo_S = np.zeros((3, size_mo, size_mo), dtype=complex)
            for i, ao in enumerate(s_B):
                mo_S[i, :, :] += DHF_one_electron_transform(self.wf.c_mo, ao)

            # Hm AO -> MO:
            mo_m = np.zeros((natm, 3, size_mo, size_mo), dtype=complex)
            for I in range(natm):
                for i, ao in enumerate(h_m[I]):
                    mo_m[I, i, :, :] += DHF_one_electron_transform(self.wf.c_mo, ao)

            # HBm UMO -> OMO transformation:
            mo_exp_tilde = np.zeros((natm, 3, 3, size_mo, size_mo), dtype=complex)
            for I in range(natm):
                for a in range(3):
                    for b in range(3):
                        mo_exp_tilde[I, a, b] = RMB_GIAO_trans_1e(mo_exp[I, a, b], mo_m[I,b], mo_S[a], f=1)

            sc_exp = get_exp_val_RMB_GIAO(self.wf.rdm1, self.wf.num_spin_orbs_NES, self.wf.num_inactive_spin_orbs, self.wf.num_active_spin_orbs, mo_exp_tilde, natm).real

        # Units, returning and printing:
        unit_ppm = nist.ALPHA**2 * 1e6

        sc_resp *= unit_ppm
        if RMB_GIAO:
            sc_exp *= unit_ppm

        if RMB_GIAO:
            sigma_tot = sc_resp + sc_exp
        else:
            sigma_tot = sc_resp

        if output:
            with np.printoptions(precision=7, suppress=True, formatter={'float_kind': lambda x: f'{x:.7f}'}):
                print("Response:")
                print(sc_resp.real)
            if RMB_GIAO:
                with np.printoptions(precision=7, suppress=True, formatter={'float_kind': lambda x: f'{x:.7f}'}):
                    print("Expectation value:")
                    print(sc_exp.real)
        
        sigma_iso = np.trace(sigma_tot.real, axis1=1, axis2=2) / 3

        with np.printoptions(precision=7, suppress=True, formatter={'float_kind': lambda x: f'{x:.7f}'}):
            print("Shieldings:", sigma_iso)

        self.shieldings = sigma_iso

    

    

