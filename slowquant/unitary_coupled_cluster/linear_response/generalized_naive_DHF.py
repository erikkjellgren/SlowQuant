from collections.abc import Sequence

import numpy as np
from scipy.linalg import solve
import scipy.linalg as la
from pyscf.data import nist

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform, generalized_one_electron_transform, DHF_one_electron_transform,
)
from slowquant.unitary_coupled_cluster.generalized_density_matrix_DHF import (
    get_orbital_gradient_response, get_orbital_gradient_response_real_imag,
    get_orbital_response_hessian_block,
    get_orbital_response_metric_sigma,
    get_orbital_response_property_gradient_annika, get_orbital_response_property_gradient_real_imag, 
    get_orbital_response_metric_sigma_real_imag,  get_orbital_response_static_property_gradient_DHF,
    get_1e_exp_value, 
    get_orbital_gradient_generalized_real_imag, RDM1
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
    generalized_one_elec_op_0i_0a, a_op_spin
)

from slowquant.unitary_coupled_cluster.generalized_operators import generalized_hamiltonian_full_space

class LinearResponse(LinearResponseBaseClass):
 import numpy as np

class LinearResponse(LinearResponseBaseClass):
    def __init__(
        self,
        wave_function: GeneralizedWaveFunctionUPS,
        excitations: str,
    ) -> None:
        """Initialize linear response by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
        """
        super().__init__(wave_function, excitations)

        idx_shift = len(self.q_ops)
        print("Gs", len(self.G_ops))
        print("qs", len(self.q_ops))
        
        
        H = generalized_hamiltonian_full_space(self.wf.h_mo, self.wf.g_mo, self.wf.num_spin_orbs)

        
        """
        # # Screen for A_ii = 0 Pernille
        # finite_excitations = []
        # if len(self.q_ops) != 0:
        #     A = get_orbital_response_hessian_block(
        #         self.wf.h_mo,
        #         self.wf.g_mo,
        #         self.wf.kappa_no_activeactive_spin_idx_dagger,
        #         self.wf.kappa_no_activeactive_spin_idx,
        #         self.wf.num_inactive_spin_orbs,
        #         self.wf.num_active_spin_orbs,
        #         self.wf.rdm1,
        #         self.wf.rdm2,
        #     )
            
            
        # # # Man behøver ikke regne hele A, men det er bare lige nemt at gøre for qq
        # for i, q in enumerate(self.q_ops):
        #     if abs(A[i, i]) > 10**-6:  # whatever rimeligt threshold
        #         finite_excitations.append(True)
        #     else:
        #         finite_excitations.append(False)
        # self.q_ops_finite = sum(bool(x) for x in finite_excitations)
        # for i, G in enumerate(self.G_ops):
        #     GI_ket = generalized_propagate_state([G], self.wf.ci_coeffs, *self.index_info)
        #     HGI_ket = generalized_propagate_state([H, G], self.wf.ci_coeffs, *self.index_info)
        #     # <0| GId H GJ |0>
        #     A = generalized_expectation_value(
        #         GI_ket,
        #         [],
        #         HGI_ket,
        #         *self.index_info,
        #     )
        #     if abs(A) > 10**-6:  # whatever rimeligt threshold
        #         finite_excitations.append(True)
        #     else:
        #         finite_excitations.append(False)
        # self.G_ops_finite = sum(bool(x) for x in finite_excitations[len(self.q_ops):])
        # finite_excitations_idx = np.array(finite_excitations)"""
        
              
              
              
              
                
        if len(self.q_ops) != 0:
            grad = get_orbital_gradient_response_real_imag(
                self.wf.h_mo,
                self.wf.g_mo,
                self.wf.kappa_no_activeactive_spin_idx,
                self.wf.num_spin_orbs_NES,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1,
                self.wf.rdm2,
            )
            print("idx, max(abs(grad orb)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**3:
                raise ValueError("Large Gradient detected in q of ", np.max(np.abs(grad)))

        grad = np.zeros(2 * len(self.G_ops), dtype=complex) #AE complex
        # H00_ket = generalized_propagate_state([self.H_0i_0a], self.wf.ci_coeffs, *self.index_info)
        for i, op in enumerate(self.G_ops):
            #print(i)
            G_ket = generalized_propagate_state([op], self.wf.ci_coeffs, *self.index_info)
            Gd_ket = generalized_propagate_state([op.dagger], self.wf.ci_coeffs, *self.index_info)
            # <0 | H G |0>
            grad[i] = generalized_expectation_value(
                self.wf.ci_coeffs,
                [H*op],
                self.wf.ci_coeffs,
                *self.index_info,
            )
            # - <0| G H |0>
            grad[i] -= generalized_expectation_value(
                self.wf.ci_coeffs,
                [op*H],
                self.wf.ci_coeffs,
                *self.index_info,
            )
            # <0| Gd H |0>
            grad[i + len(self.G_ops)] = generalized_expectation_value(
                self.wf.ci_coeffs,
                [op.dagger*H],
                self.wf.ci_coeffs,
                *self.index_info,
            )
            # - <0| H Gd |0>
            grad[i + len(self.G_ops)] -= generalized_expectation_value(
                self.wf.ci_coeffs,
                [H*op.dagger],
                self.wf.ci_coeffs,
                *self.index_info,
            )
        
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                print("Large Gradient detected in G of ", np.max(np.abs(grad)))
                # raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad))) #AE udkommenteret
        if len(self.q_ops) != 0:
            # Do orbital-orbital blocks
            self.A[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_hessian_block(
                self.wf.h_mo,
                self.wf.g_mo,
                self.wf.kappa_no_activeactive_spin_idx_dagger,
                self.wf.kappa_no_activeactive_spin_idx,
                self.wf.num_spin_orbs_NES, 
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1,
                self.wf.rdm2,
            )

            self.B[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_hessian_block(
                self.wf.h_mo,
                self.wf.g_mo,
                self.wf.kappa_no_activeactive_spin_idx_dagger,
                self.wf.kappa_no_activeactive_spin_idx_dagger,
                self.wf.num_spin_orbs_NES, 
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1,
                self.wf.rdm2,
            )

            self.Sigma[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_metric_sigma(
                self.wf.kappa_no_activeactive_spin_idx,
                self.wf.num_spin_orbs_NES, 
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1,
            )
            """
        # qq block manual
            # for j, qJ in enumerate(self.q_ops):
            #     for i, qI in enumerate(self.q_ops):

            #         # Test Anna
            #         # Make A
            #         # <0| qJd H qI |0>
            #         val = generalized_expectation_value(
            #             self.wf.ci_coeffs,
            #             [qJ.dagger*H*qI],
            #             self.wf.ci_coeffs,
            #             *self.index_info,
            #         )
            #         # <0| qI H qJd |0>
            #         val += generalized_expectation_value(
            #             self.wf.ci_coeffs,
            #             [qI*H*qJ.dagger],
            #             self.wf.ci_coeffs,
            #             *self.index_info,
            #         )

            #         # - 1/2<0| qJd qI H |0>
            #         val -= (
            #             1
            #             / 2
            #             * generalized_expectation_value(
            #                 self.wf.ci_coeffs,
            #                 [qJ.dagger*qI*H],
            #                 self.wf.ci_coeffs,
            #                 *self.index_info,
            #             )
            #         )
            #         # - 1/2*<0| H qI qJd |0>
            #         val -= (
            #             1/2
            #             * generalized_expectation_value(
            #                 self.wf.ci_coeffs,
            #                 [H*qI*qJ.dagger],
            #                 self.wf.ci_coeffs,
            #                 *self.index_info,
            #             )
            #         )
            #         # - 1/2*<0| qI qJd H |0> # minus Pernille
            #         val -= (
            #             1
            #             / 2
            #             * generalized_expectation_value(
            #                 self.wf.ci_coeffs,
            #                 [qI*qJ.dagger*H],
            #                 self.wf.ci_coeffs,
            #                 *self.index_info,
            #             )
            #         )
            #         # - 1/2*<0| H qJd qI |0> # minus Pernille
            #         val -= (
            #             1
            #             / 2
            #             * generalized_expectation_value(
            #                 self.wf.ci_coeffs,
            #                 [H*qJ.dagger*qI],
            #                 self.wf.ci_coeffs,
            #                 *self.index_info,
            #             )
            #         )
            #         self.A[i, j] = val
                    
            # # print('qq blok regnet med expectation values',self.A)
            #         # Make B
            #         #<0| qJd H qId |0>
            #         val = generalized_expectation_value(
            #             self.wf.ci_coeffs,
            #             [qJ.dagger*H*qI.dagger],
            #             self.wf.ci_coeffs,
            #             *self.index_info,
            #         )
            #         # <0| qId H qJd |0>
            #         val += generalized_expectation_value(
            #             self.wf.ci_coeffs,
            #             [qI.dagger*H*qJ.dagger],
            #             self.wf.ci_coeffs,
            #             *self.index_info,
            #         )
            #         # - 1/2<0| qJd qId H |0>
            #         val -= (
            #             1
            #             / 2
            #             * generalized_expectation_value(
            #                 self.wf.ci_coeffs,
            #                 [qJ.dagger*qI.dagger*H],
            #                 self.wf.ci_coeffs,
            #                 *self.index_info,
            #             )
            #         )
            #         # - 1/2*<0| H qId qJd |0>
            #         val -= (
            #             1/2
            #             * generalized_expectation_value(
            #                 self.wf.ci_coeffs,
            #                 [H*qI.dagger*qJ.dagger],
            #                 self.wf.ci_coeffs,
            #                 *self.index_info,
            #             )
            #         )
            #         # - 1/2*<0| qId qJd H |0> # minus Pernille
            #         val -= (
            #             1
            #             / 2
            #             * generalized_expectation_value(
            #                 self.wf.ci_coeffs,
            #                 [qI.dagger*qJ.dagger*H],
            #                 self.wf.ci_coeffs,
            #                 *self.index_info,
            #             )
            #         )
            #         # - 1/2*<0| H qJd qId |0> # minus Pernille
            #         val -= (
            #             1
            #             / 2
            #             * generalized_expectation_value(
            #                 self.wf.ci_coeffs,
            #                 [H*qJ.dagger*qI.dagger],
            #                 self.wf.ci_coeffs,
            #                 *self.index_info,
            #             )
            #         )
            #         self.B[i, j] = val

            #         #Make Sigma ##fejl rettet her??
            #         val = generalized_expectation_value(
            #             self.wf.ci_coeffs,
            #             [qI.dagger*qJ],
            #             self.wf.ci_coeffs,
            #             *self.index_info
            #         )
            #         val -= generalized_expectation_value(
            #             self.wf.ci_coeffs,
            #             [qJ*qI.dagger],
            #             self.wf.ci_coeffs,
            #             *self.index_info
            #         )
            #         self.Sigma[i, j] =  val   

                    

            #manuel slut"""
        
        if len(self.q_ops) != 0:                          
            for j, qJ in enumerate(self.q_ops):
                Hq_ket = generalized_propagate_state([H * qJ], self.wf.ci_coeffs, *self.index_info) # do_unsafe=True
                qdH_ket = generalized_propagate_state([qJ.dagger * H], self.wf.ci_coeffs, *self.index_info)
                for i, GI in enumerate(self.G_ops):
                    G_ket = generalized_propagate_state([GI], self.wf.ci_coeffs, *self.index_info)
                    Gd_ket = generalized_propagate_state([GI.dagger], self.wf.ci_coeffs, *self.index_info)
                    # print("qG",i,j)
                    # # Make A
                    # <0| Gd H q |0>
                    val = generalized_expectation_value(
                        self.wf.ci_coeffs,
                        [GI.dagger*H*qJ],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    # - 1/2<0| H q Gd |0>
                    val -= (
                        1
                        / 2
                        * generalized_expectation_value(
                            self.wf.ci_coeffs,
                            [H*qJ*GI.dagger],
                            self.wf.ci_coeffs,
                            *self.index_info,
                        )
                    )
                    # - 1/2<0| H Gd q |0>
                    val -= (
                        1
                        / 2
                        * generalized_expectation_value(
                            self.wf.ci_coeffs,
                            [H * GI.dagger * qJ],
                            self.wf.ci_coeffs,
                            *self.index_info,
                        )
                    )

                    self.A[i + idx_shift, j] = val
                    self.A[j, i + idx_shift] = val.conj()
                    
                    # Make B
                    # <0| qd H Gd |0>
                    val = generalized_expectation_value(
                        self.wf.ci_coeffs,
                        [qJ.dagger*H*GI.dagger],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    # - 1/2*<0| Gd qd H |0>
                    val -= (
                        1
                        / 2
                        * generalized_expectation_value(
                            self.wf.ci_coeffs,
                            [GI.dagger*qJ.dagger*H],
                            self.wf.ci_coeffs,
                            *self.index_info,
                        )
                    )
                    # - 1/2*<0| qd Gd H |0>
                    val -= (
                        1
                        / 2
                        * generalized_expectation_value(
                            self.wf.ci_coeffs,
                            [qJ.dagger * GI.dagger * H],
                            self.wf.ci_coeffs,
                            *self.index_info,
                        )
                    )

                    self.B[i + idx_shift, j] = val
                    self.B[j, i + idx_shift] = val

        for j, GJ in enumerate(self.G_ops):
            # GJH_ket = generalized_propagate_state([GJ], H00_ket, *self.index_info)
            # GJdH_ket = generalized_propagate_state([GJ.dagger], H00_ket, *self.index_info)
            HGJd_ket = generalized_propagate_state([H, GJ.dagger], self.wf.ci_coeffs, *self.index_info)
            HGJ_ket = generalized_propagate_state([H, GJ], self.wf.ci_coeffs, *self.index_info)
            GJ_ket = generalized_propagate_state([GJ], self.wf.ci_coeffs, *self.index_info)
            GJd_ket = generalized_propagate_state([GJ.dagger], self.wf.ci_coeffs, *self.index_info)
            for i, GI in enumerate(self.G_ops[j:], j):
                GI_ket = generalized_propagate_state([GI], self.wf.ci_coeffs, *self.index_info)
                GId_ket = generalized_propagate_state([GI.dagger], self.wf.ci_coeffs, *self.index_info)
                # print("GG",i,j)
                # Make A
                # <0| GId H GJ |0> #problemer med H0iai 
                val = generalized_expectation_value(
                    self.wf.ci_coeffs,
                    [GI.dagger*H*GJ],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )

                # <0| GJ H GId |0>
                val += generalized_expectation_value(
                    self.wf.ci_coeffs,
                    [GJ*H*GI.dagger],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )

                # - 1/2<0| GId GJ H |0>
                val -= (
                    1/2
                    * generalized_expectation_value(
                        self.wf.ci_coeffs,
                        [GI.dagger*GJ*H],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                ) 

                # - 1/2*<0| H GJ GId |0>
                val -= (
                    1/2
                    * generalized_expectation_value(
                        self.wf.ci_coeffs,
                        [H*GJ*GI.dagger],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                )

                # - 1/2*<0| GJ GId H |0>
                val -= (
                    1
                    / 2
                    * generalized_expectation_value(
                        self.wf.ci_coeffs,
                        [GJ*GI.dagger*H],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                )

                # - 1/2*<0| H GId GJ |0>
                val -= (
                    1
                    / 2
                    * generalized_expectation_value(
                        self.wf.ci_coeffs,
                        [H*GI.dagger*GJ],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                )

                self.A[i + idx_shift, j + idx_shift] = val 
                self.A[j + idx_shift, i + idx_shift] = val.conj()

                self.A_GG[i, j] = val
                self.A_GG[j, i] = val.conj()
                 
                # Make B
                # <0| GId H GJd |0>
                val = generalized_expectation_value(
                    self.wf.ci_coeffs,
                    [GI.dagger*H*GJ.dagger],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                # - <0| GId GJd H |0>
                val -= generalized_expectation_value(
                    self.wf.ci_coeffs,
                    [GI.dagger*GJ.dagger*H],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                # - <0| H GJd GId |0>
                val -= generalized_expectation_value(
                    self.wf.ci_coeffs,
                    [H*GJ.dagger*GI.dagger],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                # <0| GJd H GId |0>
                val += generalized_expectation_value(
                    self.wf.ci_coeffs,
                    [GJ.dagger*H*GI.dagger],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )

                self.B[i + idx_shift, j + idx_shift] = val 
                self.B[j + idx_shift, i + idx_shift] = val

                self.B_GG[i, j] = val
                self.B_GG[j, i] = val
                # Make Sigma
                # <0| GId GJ |0>
                val = generalized_expectation_value(
                    self.wf.ci_coeffs,
                    [GI.dagger*GJ],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                # - <0| GJ GId |0>
                val -= generalized_expectation_value(
                    self.wf.ci_coeffs,
                    [GJ*GI.dagger],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )

                # Does it also need to have the hermetian conjugate added??
                # Where is the Delta matrix? 

                self.Sigma[i + idx_shift, j + idx_shift] =  val   
                self.Sigma[j + idx_shift, i + idx_shift] =  val.conj()


        # Check hermiticity of the Metric:
        print(f"Hermiticity check of the metric: max|S - S†| = "
            f"{np.max(np.abs(self.Sigma - self.Sigma.conj().T)):.2e}")
        
         
        
        # Check hermiticity of the Hessian:
        size = len(self.A)
        E2 = np.zeros((size * 2, size * 2), dtype=complex) #AE complex
        E2[:size, :size] = self.A
        E2[:size, size:] = self.B
        E2[size:, :size] = self.B.conjugate() #AE added conjugtate 
        E2[size:, size:] = self.A.conjugate() #AE added conjugtate 

        print(f"Hermiticity check of the Hessian: max|E2 - E2†| = "
            f"{np.max(np.abs(E2 - E2.conj().T)):.2e}")  

        print(f"Hermiticity check of A: max|A - A†| = "
            f"{np.max(np.abs(self.A - self.A.conj().T)):.2e}")  

        print(f"Symmetry check of B: max|B - B.T| = "
            f"{np.max(np.abs(self.B - self.B.T)):.2e}")  
        
        """                
        #print("H shape:", E2.shape)
        #print("sigma shape:", self.Sigma.shape)
        #print("H diagonal:", np.diag(E2).real)
        #print("sigma diagonal:", np.diag(self.Sigma).real)
        
        # print(f"Hermiticity check of A qG: max|A - A†| = "
        #     f"{np.max(np.abs(self.A[:idx_shift,idx_shift:] - self.A[idx_shift:,:idx_shift].conj().T)):.2e}") 

        # print(f"Symmetry check of B qq: max|B - B.T| = "
        #     f"{np.max(np.abs(self.B[:idx_shift,:idx_shift] - self.B[:idx_shift,:idx_shift].T)):.2e}") 

        # print(f"Symmetry check of B qG: max|B - B.T| = "
        #     f"{np.max(np.abs(self.B[:idx_shift,idx_shift:] - self.B[idx_shift:,:idx_shift].T)):.2e}") 
        
        #print(np.round(np.diag(Hessian_matrix),5))


                
        # for i in range(len(self.A)): 
        #     print('self A',self.A[i,i])
        
        # print('A', self.A)
        # print('B', self.B)
        # print('Sigma', self.Sigma)


# # #Pernille fjernelse af A
#         self.A = self.A[np.outer(finite_excitations_idx, finite_excitations_idx)].reshape(
#                     (np.sum(finite_excitations_idx), np.sum(finite_excitations_idx))
#                 )
#         self.B = self.B[np.outer(finite_excitations_idx, finite_excitations_idx)].reshape(
#                     (np.sum(finite_excitations_idx), np.sum(finite_excitations_idx))
#                 )
#         self.Sigma = self.Sigma[np.outer(finite_excitations_idx, finite_excitations_idx)].reshape(
#                     (np.sum(finite_excitations_idx), np.sum(finite_excitations_idx))
#                 )
                
#         self.Delta = np.zeros((len(self.Sigma), len(self.Sigma))
#         )  # Delta er defineret her fordi den ellers har forkert dimension i unrestricted_lr_baseclass.py 
"""
                

                                
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
        mux_op = generalized_one_elec_op_0i_0a(
            mux,
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
        )
        muy_op = generalized_one_elec_op_0i_0a(
            muy,
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
        )
        muz_op = generalized_one_elec_op_0i_0a(
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


    def get_property_gradient(self, property_integrals: np.ndarray) -> np.ndarray:
        """Calculate property gradient.

        Args:
            property_integrals: Integrals in AO basis.

        Returns:
            Property gradient.
        """
        in_shape = property_integrals.shape[:-2]
        size_mo = self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs + self.wf.num_virtual_spin_orbs
        # property_integrals = property_integrals.reshape(-1, size_mo, size_mo)
        num_mo = len(property_integrals)
        mo = np.zeros((num_mo, size_mo, size_mo), dtype=complex)
        for i, ao in enumerate(property_integrals):
            mo[i, :, :] += generalized_one_electron_transform(self.wf.c_mo, ao)

        idx_shift_q = len(self.q_ops)
        V = np.zeros((len(self.q_ops + self.G_ops), num_mo), dtype=complex)

        if len(self.q_ops) != 0:
            # Orbital response part
            V[:idx_shift_q, :] =  get_orbital_response_static_property_gradient(
                mo,
                self.wf.kappa_no_activeactive_spin_idx,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1,
            )
        for idx, G in enumerate(self.G_ops):
            G_ket = generalized_propagate_state([G], self.wf.ci_coeffs, *self.index_info)
            Gd_ket =generalized_propagate_state([G.dagger], self.wf.ci_coeffs, *self.index_info)
            # Inactive part
            for i in range(self.wf.num_inactive_spin_orbs):
                E_ket = generalized_propagate_state([a_op_spin(i,True), a_op_spin(i,False)], self.wf.ci_coeffs, *self.index_info) 
                # < 0 | G E | 0 >
                val = generalized_expectation_value(Gd_ket, [], E_ket, *self.index_info)
                # - < 0 | E G | 0 >
                val -= generalized_expectation_value(E_ket, [], G_ket, *self.index_info) # E_ket = Ed_ket for E(i,i)
                V[idx + idx_shift_q, :] += mo[:, i, i] * val
            # Active part
            for p in range(self.wf.num_inactive_spin_orbs, self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs):
                for q in range(
                    self.wf.num_inactive_spin_orbs, self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs
                ):
                    E_ket = generalized_propagate_state([a_op_spin(p,True)*a_op_spin(q,False)], self.wf.ci_coeffs, *self.index_info)
                    Ed_ket = generalized_propagate_state([a_op_spin(q,True)*a_op_spin(p,False)], self.wf.ci_coeffs, *self.index_info)
                    # < 0 | G E | 0 >
                    val = generalized_expectation_value(Gd_ket, [], E_ket, *self.index_info)
                    # - < 0 | E G | 0 >
                    val -= generalized_expectation_value(Ed_ket, [], G_ket, *self.index_info)
                    V[idx + idx_shift_q, :] += mo[:, p, q] * val
        if np.allclose(mo, mo.transpose(0, -1, -2)):
            return np.vstack((V, -1 * V)).reshape(-1, *in_shape)
        return np.vstack((V, V)).reshape(-1, *in_shape)
    

    def get_property_gradient_4comp_old(self, property_integrals: np.ndarray) -> np.ndarray:
        """Calculate property gradient.

        Args:
            property_integrals: Integrals in AO basis.

        Returns:
            Property gradient.
        """
        in_shape = property_integrals.shape[:-2]
        size_mo = self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs + self.wf.num_virtual_spin_orbs
        # property_integrals = property_integrals.reshape(-1, size_mo, size_mo)
        num_mo = len(property_integrals)
        mo = np.zeros((num_mo, size_mo, size_mo), dtype=complex)
        for i, ao in enumerate(property_integrals):
            mo[i, :, :] += DHF_one_electron_transform(self.wf.c_mo, ao)

        idx_shift_q = len(self.q_ops_resp)
        V = np.zeros((len(self.q_ops_resp + self.G_ops), num_mo), dtype=complex)

        if len(self.q_ops_resp) != 0:
            # Orbital response part
            V[:idx_shift_q, :] =  get_orbital_response_static_property_gradient_DHF(
                mo,
                #self.wf.kappa_no_activeactive_spin_idx,
                self.wf.kappa_no_activeactive_spin_idx_resp,
                self.wf.num_spin_orbs_NES,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1,
            )
        for idx, G in enumerate(self.G_ops):
            G_ket = generalized_propagate_state([G], self.wf.ci_coeffs, *self.index_info)
            Gd_ket =generalized_propagate_state([G.dagger], self.wf.ci_coeffs, *self.index_info)
            # Inactive part
            for i in range(self.wf.num_inactive_spin_orbs):
                E_ket = generalized_propagate_state([a_op_spin(i,True), a_op_spin(i,False)], self.wf.ci_coeffs, *self.index_info) 
                # < 0 | G E | 0 >
                val = generalized_expectation_value(Gd_ket, [], E_ket, *self.index_info)
                # - < 0 | E G | 0 >
                val -= generalized_expectation_value(E_ket, [], G_ket, *self.index_info) # E_ket = Ed_ket for E(i,i)
                V[idx + idx_shift_q, :] += mo[:, i, i] * val
            # Active part
            for p in range(self.wf.num_inactive_spin_orbs, self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs):
                for q in range(
                    self.wf.num_inactive_spin_orbs, self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs
                ):
                    E_ket = generalized_propagate_state([a_op_spin(p,True)*a_op_spin(q,False)], self.wf.ci_coeffs, *self.index_info)
                    Ed_ket = generalized_propagate_state([a_op_spin(q,True)*a_op_spin(p,False)], self.wf.ci_coeffs, *self.index_info)
                    # < 0 | G E | 0 >
                    val = generalized_expectation_value(Gd_ket, [], E_ket, *self.index_info)
                    # - < 0 | E G | 0 >
                    val -= generalized_expectation_value(Ed_ket, [], G_ket, *self.index_info)
                    V[idx + idx_shift_q, :] += mo[:, p, q] * val
        if np.allclose(mo, mo.transpose(0, -1, -2)):
            return np.vstack((V, -1 * V.conjugate())).reshape(-1, *in_shape)
        return np.vstack((V, V.conjugate())).reshape(-1, *in_shape)
        #return np.vstack((V, V.conjugate())).reshape(-1, *in_shape)


    def get_property_gradient_4comp(self, property_integrals: np.ndarray) -> np.ndarray:
        in_shape = property_integrals.shape[:-2]
        size_mo = self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs + self.wf.num_virtual_spin_orbs
        num_mo = len(property_integrals)
        mo = np.zeros((num_mo, size_mo, size_mo), dtype=complex)
        for i, ao in enumerate(property_integrals):
            mo[i, :, :] += DHF_one_electron_transform(self.wf.c_mo, ao)

        idx_shift_q = len(self.q_ops_resp)
        V = np.zeros((len(self.q_ops_resp + self.G_ops), num_mo), dtype=complex)

        if len(self.q_ops_resp) != 0:
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
            for i in range(self.wf.num_inactive_spin_orbs):
                E_ket = generalized_propagate_state([a_op_spin(i,True), a_op_spin(i,False)], self.wf.ci_coeffs, *self.index_info)
                val = generalized_expectation_value(Gd_ket, [], E_ket, *self.index_info)
                val -= generalized_expectation_value(E_ket, [], G_ket, *self.index_info)
                V[idx + idx_shift_q, :] += mo[:, i, i] * val
            for p in range(self.wf.num_inactive_spin_orbs, self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs):
                for q in range(self.wf.num_inactive_spin_orbs, self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs):
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
    
    def get_property_gradient_4comp_no_ep(self, property_integrals: np.ndarray) -> np.ndarray:
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
                self.wf.kappa_no_activeactive_spin_idx,
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
            for p in range(self.wf.num_spin_orbs_NES+self.wf.num_inactive_spin_orbs, self.wf.num_spin_orbs_NES +self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs):
                for q in range(self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs, self.wf.num_spin_orbs_NES + self.wf.num_inactive_spin_orbs + self.wf.num_active_spin_orbs):
                    E_ket = generalized_propagate_state([a_op_spin(p,True)*a_op_spin(q,False)], self.wf.ci_coeffs, *self.index_info)
                    Ed_ket = generalized_propagate_state([a_op_spin(q,True)*a_op_spin(p,False)], self.wf.ci_coeffs, *self.index_info)
                    val = generalized_expectation_value(Gd_ket, [], E_ket, *self.index_info)
                    val -= generalized_expectation_value(Ed_ket, [], G_ket, *self.index_info)
                    V[idx + idx_shift_q, :] += mo[:, p, q] * val

        # # Determine hermiticity per component to set correct sign of lower block
        # lower_V = np.zeros_like(V)
        # for i in range(num_mo):
        #     if np.allclose(mo[i], mo[i].conj().T, atol=1e-10):
        #         # Hermitian operator: lower block is -V*
        #         lower_V[:, i] = -V[:, i].conj()
        #     else:
        #         # Anti-Hermitian operator: lower block is +V*
        #         lower_V[:, i] = V[:, i].conj()

        # return np.vstack((V, lower_V)).reshape(-1, *in_shape)

        full_mask = np.concatenate([
                    self.finite_excitations_idx,
                    self.finite_excitations_idx,
                ])
        for i in range(num_mo):
            if np.allclose(mo, mo.conj().transpose(0, -1, -2)):
                V_stacked= np.vstack((V, -1 * V.conj())).reshape(-1, *in_shape)
                V_stacked = V_stacked[full_mask, :]
                return V_stacked
        else: 
            V_stacked =  np.vstack((V, V.conj())).reshape(-1, *in_shape)
            V_stacked=V_stacked[full_mask, :]
            return V_stacked


    def get_SSCC_4comp_old(self, h1_int, h2_int: np.ndarray) -> np.ndarray:
        prop_grad = self.get_property_gradient_4comp(h1_int)
        response = solve(self.hessian, prop_grad)
        para = np.einsum('ix,ix->x', prop_grad, response)

        dia = get_1e_exp_value(h2_int,self.wf.num_spin_orbs_NES,self.wf.num_inactive_spin_orbs,
                               self.wf.num_active_spin_orbs,self.wf.rdm1) * nist.ALPHA**4

        return para + dia
    
    def get_SSCC_4comp_tensor(self, h1_int, h2_int: np.ndarray) -> np.ndarray:
        """
        Compute SSC tensor for all atom pairs.
        
        h1_int: (natm, 3, n4c, n4c)  — paramagnetic perturbator at each nucleus
        h2_int: (natm, natm, 3, 3, n4c, n4c) — diamagnetic two-nucleus operator
        
        Returns: (natm, natm, 3, 3) SSC tensor
        """
        natm = h1_int.shape[0]
        
        # Precompute property gradients and responses for all atoms
        # prop_grads[I] has shape (3, ...) — gradient for nucleus I
        prop_grads = [self.get_property_gradient_4comp(h1_int[I]) for I in range(natm)]
        responses  = [solve(self.hessian, prop_grads[I]) for I in range(natm)]
        
        tensor = np.zeros((natm, natm, 3, 3))
        
        for I in range(natm):
            for J in range(I, natm):
                # Paramagnetic: bilinear contraction of gradient I with response to J
                para = np.einsum('ix,jx->ij', prop_grads[I], responses[J])  # (3, 3)
                
                # Diamagnetic: expectation value of two-nucleus operator
                dia = get_1e_exp_value(
                    h2_int[I, J],
                    self.wf.num_spin_orbs_NES,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.rdm1
                ) * nist.ALPHA**4  # (3, 3)
                
                tensor[I, J] = para + dia
                tensor[J, I] = para + dia  # symmetric

        return tensor 
        
    def get_SSCC_4comp_iso_old(self, h1_int: np.ndarray, h2_int: np.ndarray) -> np.ndarray:
        """
        Compute isotropic SSC constants for all atom pairs.
        
        h1_int: (natm, 3, n4c, n4c)
        h2_int: (natm, natm, 3, 3, n4c, n4c)
        
        Returns: (natm, natm) isotropic J-coupling constants in Hz
        """
        natm = h1_int.shape[0]
        
        prop_grads = [self.get_property_gradient_4comp(h1_int[I]) for I in range(natm)]
        responses  = [solve(self.hessian, prop_grads[I]) for I in range(natm)]

        nuc_mag = .5 * (nist.E_MASS / nist.PROTON_MASS)
        au2Hz   = nist.HARTREE2J / nist.PLANCK

        isotropic_J = np.zeros((natm, natm))

        for I in range(natm):
            for J in range(I+1, natm):

                #para = np.einsum('ix,jx->ij', prop_grads[I], responses[J])  # (3, 3)

                para = np.zeros((3,3), dtype=np.complex128)
                for alpha in range(3):
                    for beta in range(3):
                        para[alpha, beta] = np.einsum(
                            'i,i->', prop_grads[I][alpha, :], responses[J][beta, :])

                dia = np.zeros((3, 3), dtype=np.complex128)

                for alpha in range(3):
                    for beta in range(3):
                        dia[alpha, beta] = get_1e_exp_value(
                            h2_int[I, J, alpha, beta],  # (n4c, n4c)
                            self.wf.num_spin_orbs_NES,
                            self.wf.num_inactive_spin_orbs,
                            self.wf.num_active_spin_orbs,
                            self.wf.rdm1
                        )

                J_tensor = (para + dia) * nist.ALPHA**4 * au2Hz * nuc_mag**2

                iso = np.trace(J_tensor) / 3

                isotropic_J[I, J] = iso
                isotropic_J[J, I] = iso

        return isotropic_J  # (natm, natm)

    def get_SSCC_4comp_iso_nonreduced(self, h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
        """
        Compute isotropic SSC constants for all atom pairs.
        
        h1_int: (natm, 3, n4c, n4c)
        h2_int: (natm, natm, 3, 3, n4c, n4c)
        
        Returns: (natm, natm) isotropic J-coupling constants in Hz
        """
        h1_int = np.zeros_like(h1)
        h2_int = np.zeros_like(h2)


        natm = h1_int.shape[0]

        for I in range(natm):
            for a in range(3):
                h1[I,a] = DHF_one_electron_transform(self.wf.c_mo, h1[I,a])
            for J in range(I+1, natm):
                for alpha in range(3):
                    for beta in range(3):
                        h2_int[I,J,alpha,beta] = DHF_one_electron_transform(self.wf.c_mo, h2[I,J,alpha,beta])

        
        
        prop_grads = [self.get_property_gradient_4comp(h1_int[I]) for I in range(natm)]
        responses  = [solve(self.hessian, prop_grads[I]) for I in range(natm)]

        nuc_mag = 0.5 * (nist.E_MASS / nist.PROTON_MASS)
        au2Hz   = nist.HARTREE2J / nist.PLANCK

        # Compute SSC for each atom pair in self.nuc_pair
        iso_ssc = []
        nuc_pair = []
        for I in range(natm):
            for J in range(I+1, natm):
                nuc_pair.append((I,J))
                para = np.zeros((3,3), dtype=np.complex128)
                dia = np.zeros((3,3), dtype=np.complex128)

                for alpha in range(3):
                    for beta in range(3):
                        para[alpha,beta] = np.einsum('i,i->', prop_grads[I][alpha,:], responses[J][beta,:])
                        dia[alpha,beta] = get_1e_exp_value(
                            h2_int[I,J,alpha,beta],
                            self.wf.num_spin_orbs_NES,
                            self.wf.num_inactive_spin_orbs,
                            self.wf.num_active_spin_orbs,
                            self.wf.rdm1
                        )

                J_tensor = (para + dia) * nist.ALPHA**4 * au2Hz * nuc_mag**2
                iso = np.trace(J_tensor) / 3
                iso_ssc.append(iso.real)  # take real part to match PySCF
                print(iso.imag)

            # Map back to full (natm x natm) tensor
            ktensor = np.zeros((natm, natm))
        for k, (i,j) in enumerate(nuc_pair):
            ktensor[i,j] = ktensor[j,i] = iso_ssc[k]

        return ktensor  # isotropic J (Hz) in full (natm x natm) array
    
    def get_SSCC_4comp_iso_dia(self, h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
        """
        Compute isotropic reduced SSC constants K for all atom pairs.
        
        h1: (natm, 3, n4c, n4c)
        h2: (natm, natm, 3, 3, n4c, n4c)
        
        Returns: (natm, natm) reduced isotropic K-coupling constants in Hz
        """
        h1_int = np.zeros_like(h1)
        h2_int = np.zeros_like(h2)

        natm = h1.shape[0]

        for I in range(natm):
            for a in range(3):
                h1_int[I, a] = DHF_one_electron_transform(self.wf.c_mo, h1[I, a])
            for J in range(I+1, natm):
                for alpha in range(3):
                    for beta in range(3):
                        h2_int[I, J, alpha, beta] = DHF_one_electron_transform(
                            self.wf.c_mo, h2[I, J, alpha, beta]
                        )


        prop_grads = [self.get_property_gradient_4comp(h1_int[I]) for I in range(natm)]
        responses  = [solve(self.hessian, prop_grads[I]) for I in range(natm)]



        nuc_mag = 0.5 * (nist.E_MASS / nist.PROTON_MASS)  # e*hbar/2m
        au2Hz   = nist.HARTREE2J / nist.PLANCK

        iso_ssc  = []
        nuc_pair = []

        for I in range(natm):
            for J in range(I+1, natm):
                nuc_pair.append((I, J))

                para = np.zeros((3, 3), dtype=np.complex128)
                dia  = np.zeros((3, 3), dtype=np.complex128)

                for alpha in range(3):
                    for beta in range(3):
                        para[alpha, beta] = np.einsum(
                            'i,i->', prop_grads[I][alpha, :], responses[J][beta, :]
                        ) * 2  # factor of 2 from PySCF make_para
                        dia[alpha, beta] = get_1e_exp_value(
                            h2_int[I, J, alpha, beta],
                            self.wf.num_spin_orbs_NES,
                            self.wf.num_inactive_spin_orbs,
                            self.wf.num_active_spin_orbs,
                            self.wf.rdm1,
                        )

                # Reduced K: no gyromagnetic ratio — matching PySCF ktensor
                K_tensor = (para + dia) * nist.ALPHA**4 * au2Hz * nuc_mag**2
                iso_K = np.trace(K_tensor).real / 3
                iso_ssc.append(iso_K)

        ktensor = np.zeros((natm, natm))
        for k, (i, j) in enumerate(nuc_pair):
            ktensor[i, j] = ktensor[j, i] = iso_ssc[k]

        return ktensor  # reduced K (Hz), (natm, natm)
    
    def get_SSCC_4comp_iso(self, h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
        """
        Compute isotropic reduced SSC constants K for all atom pairs.
        Uses full Olsen-Jørgensen propagator including e-p rotations.
        No separate diamagnetic term needed.

        h1: (natm, 3, n4c, n4c) — AO basis perturbation operator kappa_M

        Returns: (natm, natm) reduced isotropic K-coupling constants in Hz
        """
        h1_int = np.zeros_like(h1)
        natm = h1.shape[0]

        for I in range(natm):
            for a in range(3):
                #h1_int[I, a] = DHF_one_electron_transform(self.wf.c_mo, h1[I, a])
                h1_int[I, a] = h1[I, a]

        test = True

        if test == True:
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

            
            def solve_lr_drop_sigma_null(H, sigma, cut=1e-10):
                # Hermitize (important for numerical stability)
                Hh = 0.5*(H + H.conj().T)
                Sh = 0.5*(sigma + sigma.conj().T)
                # 1) eigen-decompose metric
                s, U = la.eigh(Sh)
                # 2) keep only non-null directions
                keep = np.abs(s) > cut * np.max(np.abs(s))
                Uk = U[:, keep]
                # 3) project both matrices consistently
                Hk = Uk.conj().T @ Hh @ Uk
                Sk = Uk.conj().T @ Sh @ Uk
                return Hk, Sk

            #E2_red, S_red = solve_lr_drop_sigma_null(E2,S)

            # # Hermitize (important for numerical stability)
            # Hh = 0.5*(H + H.conj().T)
            # Sh = 0.5*(sigma + sigma.conj().T)
            # # 1) eigen-decompose metric
            # s, U = la.eigh(Sh)
            # # 2) keep only non-null directions
            # keep = np.abs(s) > cut * np.max(np.abs(s))
            # Uk = U[:, keep]
            # # 3) project both matrices consistently
            # Hk = Uk.conj().T @ Hh @ Uk
            # Sk = Uk.conj().T @ Sh @ Uk
        
        else:
            num_parameters = len(self.G_ops) + len(self.q_ops_resp)
            A_mat     = np.zeros((num_parameters, num_parameters), dtype=complex)
            B_mat     = np.zeros((num_parameters, num_parameters), dtype=complex)

            if len(self.q_ops_resp) != 0:
                """
            #     grad_test = get_orbital_gradient_response_real_imag(
            #         self.wf.h_mo,
            #         self.wf.g_mo,
            #         self.wf.kappa_no_activeactive_spin_idx_resp,
            #         self.wf.num_spin_orbs_NES,
            #         self.wf.num_inactive_spin_orbs,
            #         self.wf.num_active_spin_orbs,
            #         self.wf.rdm1,
            #         self.wf.rdm2,
            #     )
            #     grad2_test = get_orbital_gradient_generalized_real_imag(
            #         self.wf.h_mo,
            #         self.wf.g_mo,
            #         self.wf.kappa_no_activeactive_spin_idx_resp,
            #         self.wf.num_spin_orbs_NES,
            #         self.wf.num_inactive_spin_orbs,
            #         self.wf.num_active_spin_orbs,
            #         self.wf.rdm1,
            #         self.wf.rdm2,
            #     )
                #print("idx, max(abs(grad orb)):", np.argmax(np.abs(grad_test)), np.max(np.abs(grad_test)))

                #print("gradient", np.round(grad_test,5))


                #print("idx, max(abs(grad2 orb)):", np.argmax(np.abs(grad2_test)), np.max(np.abs(grad_test)))"""

                A_mat[:len(self.wf.kappa_no_activeactive_spin_idx_ep), :len(self.wf.kappa_no_activeactive_spin_idx_ep)] = get_orbital_response_hessian_block(
                    self.wf.h_mo, self.wf.g_mo,
                    #self.wf.kappa_no_activeactive_spin_idx_dagger_resp,
                    #self.wf.kappa_no_activeactive_spin_idx_resp,
                    self.wf.kappa_no_activeactive_spin_idx_ep_dagger,
                    self.wf.kappa_no_activeactive_spin_idx_ep,
                    self.wf.num_spin_orbs_NES,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.rdm1, self.wf.rdm2,
                )
                A_mat[: len(self.wf.kappa_no_activeactive_spin_idx_ep), len(self.wf.kappa_no_activeactive_spin_idx_ep) : len(self.q_ops_resp)] = get_orbital_response_hessian_block(
                    self.wf.h_mo, self.wf.g_mo,
                    self.wf.kappa_no_activeactive_spin_idx_ep_dagger,
                    self.wf.kappa_no_activeactive_spin_idx,
                    self.wf.num_spin_orbs_NES,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.rdm1, self.wf.rdm2,
                )
                A_mat[len(self.wf.kappa_no_activeactive_spin_idx_ep) : len(self.q_ops_resp), : len(self.wf.kappa_no_activeactive_spin_idx_ep)] = get_orbital_response_hessian_block(
                    self.wf.h_mo, self.wf.g_mo,
                    self.wf.kappa_no_activeactive_spin_idx_dagger,
                    self.wf.kappa_no_activeactive_spin_idx_ep,
                    self.wf.num_spin_orbs_NES,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.rdm1, self.wf.rdm2,
                )
                A_mat[len(self.wf.kappa_no_activeactive_spin_idx_ep) : len(self.q_ops_resp), len(self.wf.kappa_no_activeactive_spin_idx_ep) : len(self.q_ops_resp)] = get_orbital_response_hessian_block(
                    self.wf.h_mo, self.wf.g_mo,
                    self.wf.kappa_no_activeactive_spin_idx_dagger,
                    self.wf.kappa_no_activeactive_spin_idx,
                    self.wf.num_spin_orbs_NES,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.rdm1, self.wf.rdm2,
                )

                B_mat[: len(self.q_ops_resp), : len(self.q_ops_resp)] = get_orbital_response_hessian_block(
                    self.wf.h_mo, self.wf.g_mo,
                    self.wf.kappa_no_activeactive_spin_idx_dagger_resp,
                    self.wf.kappa_no_activeactive_spin_idx_dagger_resp,
                    self.wf.num_spin_orbs_NES,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.rdm1, self.wf.rdm2,
                )
            

            H = generalized_hamiltonian_full_space(self.wf.h_mo, self.wf.g_mo, self.wf.num_spin_orbs)

            idx_shift = len(self.q_ops_resp)

            if len(self.q_ops_resp) != 0:                          
                for j, qJ in enumerate(self.q_ops_resp):
                    Hq_ket = generalized_propagate_state([H * qJ], self.wf.ci_coeffs, *self.index_info) # do_unsafe=True
                    qdH_ket = generalized_propagate_state([qJ.dagger * H], self.wf.ci_coeffs, *self.index_info)
                    for i, GI in enumerate(self.G_ops):
                        G_ket = generalized_propagate_state([GI], self.wf.ci_coeffs, *self.index_info)
                        Gd_ket = generalized_propagate_state([GI.dagger], self.wf.ci_coeffs, *self.index_info)
                        # print("qG",i,j)
                        # # Make A
                        # <0| Gd H q |0>
                        val = generalized_expectation_value(
                            self.wf.ci_coeffs,
                            [GI.dagger*H*qJ],
                            self.wf.ci_coeffs,
                            *self.index_info,
                        )
                        # - 1/2<0| H q Gd |0>
                        val -= (
                            1
                            / 2
                            * generalized_expectation_value(
                                self.wf.ci_coeffs,
                                [H*qJ*GI.dagger],
                                self.wf.ci_coeffs,
                                *self.index_info,
                            )
                        )
                        # - 1/2<0| H Gd q |0>
                        val -= (
                            1
                            / 2
                            * generalized_expectation_value(
                                self.wf.ci_coeffs,
                                [H * GI.dagger * qJ],
                                self.wf.ci_coeffs,
                                *self.index_info,
                            )
                        )
                        if j < len(self.wf.kappa_no_activeactive_spin_idx_ep):
                            A_mat[i + idx_shift, j] = 0
                            A_mat[j, i + idx_shift] = 0
                        
                        else:
                            A_mat[i + idx_shift, j] = val
                            A_mat[j, i + idx_shift] = val.conj()
                        
                        # Make B
                        # <0| qd H Gd |0>
                        val = generalized_expectation_value(
                            self.wf.ci_coeffs,
                            [qJ.dagger*H*GI.dagger],
                            self.wf.ci_coeffs,
                            *self.index_info,
                        )
                        # - 1/2*<0| Gd qd H |0>
                        val -= (
                            1
                            / 2
                            * generalized_expectation_value(
                                self.wf.ci_coeffs,
                                [GI.dagger*qJ.dagger*H],
                                self.wf.ci_coeffs,
                                *self.index_info,
                            )
                        )
                        # - 1/2*<0| qd Gd H |0>
                        val -= (
                            1
                            / 2
                            * generalized_expectation_value(
                                self.wf.ci_coeffs,
                                [qJ.dagger * GI.dagger * H],
                                self.wf.ci_coeffs,
                                *self.index_info,
                            )
                        )

                        if j < len(self.wf.kappa_no_activeactive_spin_idx_ep):
                            B_mat[i + idx_shift, j] = 0
                            B_mat[j, i + idx_shift] = 0

                        else:
                            B_mat[i + idx_shift, j] = val
                            B_mat[j, i + idx_shift] = val



            A_mat[len(self.q_ops_resp):, len(self.q_ops_resp):] = self.A_GG
            B_mat[len(self.q_ops_resp):, len(self.q_ops_resp):] = self.B_GG

            size = len(A_mat)
            E2_mat = np.zeros((size * 2, size * 2), dtype=complex)
            E2_mat[:size, :size] = A_mat
            E2_mat[:size, size:] = B_mat
            E2_mat[size:, :size] = B_mat.conjugate()
            E2_mat[size:, size:] = A_mat.conjugate()



            print(f"Hermiticity check of the Hessian: max|E2 - E2†| = "
                f"{np.max(np.abs(E2_mat - E2_mat.conj().T)):.2e}") 

            print(f"Hermiticity check of A: max|A - A†| = "
                f"{np.max(np.abs(A_mat - A_mat.conj().T)):.2e}") 
            
            print(f"Symmetry check of B: max|B - B.T| = "
                f"{np.max(np.abs(B_mat - B_mat.T)):.2e}") 
            
            # print(f"Hermiticity check of A q: max|A - A†| = "
            #     f"{np.max(np.abs(A_mat[:len(self.q_ops_resp), :len(self.q_ops_resp)] - A_mat[:len(self.q_ops_resp), :len(self.q_ops_resp)].conj().T)):.2e}") 
            
            # print(f"Symmetry check of B q: max|B - B.T| = "
            #     f"{np.max(np.abs(B_mat[:len(self.q_ops_resp), :len(self.q_ops_resp)] - B_mat[:len(self.q_ops_resp), :len(self.q_ops_resp)].T)):.2e}") 
            
            # print(f"Hermiticity check of A G: max|A - A†| = "
            #     f"{np.max(np.abs(A_mat[len(self.q_ops_resp):, len(self.q_ops_resp):] - A_mat[len(self.q_ops_resp):, len(self.q_ops_resp):].conj().T)):.2e}") 
            
            # print(f"Symmetry check of B G: max|B - B.T| = "
            #     f"{np.max(np.abs(B_mat[len(self.q_ops_resp):, len(self.q_ops_resp):] - B_mat[len(self.q_ops_resp):, len(self.q_ops_resp):].T)):.2e}") 
            




        #s = A_mat.shape[0]//2

        #print(np.round((A_mat - A_mat.conj().T)[s:,:s],4),"\n\n")
        #print(np.round((A_mat - A_mat.conj().T)[:s,s:],4))

        # Property gradients and responses for all nuclei
        # prop_grads = [self.get_property_gradient_4comp(h1_int[I]) for I in range(natm)]
        # responses  = [solve(E2_mat, prop_grads[I]) for I in range(natm)]
        
        prop_grads = [self.get_property_gradient_4comp_no_ep(h1_int[I]) for I in range(natm)]
        # responses  = [solve(E2, prop_grads[I]) for I in range(natm)]

        # responses = [
        #     np.linalg.pinv(E2_mat, rcond=1e-10) @ prop_grads[I]
        #     for I in range(natm)
        # ]

        responses = [
            np.linalg.pinv(E2, rcond=1e-10) @ prop_grads[I]
            for I in range(natm)
        ]

        # prop_grads = [self.get_property_gradient_4comp_no_ep_select(h1_int[I]) for I in range(natm)]

        # responses = [
        #     np.linalg.pinv(E2_red, rcond=1e-10) @ prop_grads[I]
        #     for I in range(natm)
        # ]


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
                        ssc_para[I, J, alpha, beta] = ssc_para[J, I, alpha, beta] = np.einsum(
                            'i,i->',
                            -prop_grads[I][:, alpha].conj(),
                            responses[J][:, beta]
                        ).real

            # Factoring:
            ssc_dia *= nist.ALPHA**4
            ssc_para *= nist.ALPHA**4

            ktensor = np.zeros((natm, natm))

            for k, (I, J) in enumerate(nuc_pair):
                ktensor[I, J] = ktensor[J, I] = au2Hz * nuc_mag ** 2 * np.trace(ssc_para[I, J] + ssc_dia[I, J]).real / 3 

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

                    # for alpha in range(3):
                    #     for beta in range(3):
                    #         # eq. (389): <<A;B>> = -A^[1] . N^B  (minus sign)
                    #         K_tensor[alpha, beta] = -np.einsum(
                    #             'i,i->', prop_grads[I][alpha, :], responses[J][beta, :]
                    #         )


                    # for alpha in range(3):
                    #     for beta in range(3):
                    #         K_tensor[alpha, beta] = np.einsum(
                    #             'i,i->', prop_grads[I][:, alpha], responses[J][:, beta]
                    #         )      


                    for alpha in range(3):
                        for beta in range(3):
                            # K_tensor[alpha, beta] = np.einsum(
                            #     'i,i->',
                            #     -prop_grads[I][:, alpha],
                            #     responses[J][:, beta]
                            # )
                            K_tensor[alpha, beta] = np.einsum(
                                'i,i->',
                                -prop_grads[I][:, alpha].conj(),
                                responses[J][:, beta]
                            ).real

                    

                    #print(K_tensor)

                    # K_tensor = (
                    #     K_tensor
                    #     * nist.ALPHA**4
                    # )


                    K_tensor = (
                        K_tensor
                        * nist.ALPHA**4
                        * au2Hz
                        * nuc_mag**2
                    )

                    # Reduced K in Hz, eq. (37): K = e^2 Re<<kappa_M; kappa_N>>
                    #K_tensor = K_tensor * nist.ALPHA**4 * au2Hz * nuc_mag**2
                    #K_tensor = K_tensor * au2Hz * nuc_mag**2

                    iso_K = np.trace(K_tensor).real / 3
                    iso_ssc.append(iso_K)

            ktensor = np.zeros((natm, natm))
            for k, (i, j) in enumerate(nuc_pair):
                ktensor[i, j] = ktensor[j, i] = iso_ssc[k]

        # print("prop_grads[I] shape:", prop_grads[I].shape)
        # print("responses[J] shape:", responses[J].shape)
        # print("raw K_tensor before prefactor:", K_tensor)
        # print("ALPHA**4:", nist.ALPHA**4)
        # print("au2Hz:", au2Hz)
        # print("nuc_mag**2:", nuc_mag**2)
        # print("nuc_mag**2 * ALPHA**4 * au2Hz:", nuc_mag**2 * nist.ALPHA**4 * au2Hz)


        return ktensor  # reduced K (Hz), (natm, natm)
    


    
    def get_shieldings_4comp_iso(self, h1: np.ndarray, h2: np.ndarray, hb: np.ndarray,g_ssss: np.ndarray, g_lsss: np.ndarray) -> np.ndarray:
        # Hessian and metric:
        natm = h1.shape[0]

        size = len(self.A)

        E2 = np.zeros((size * 2, size * 2), dtype=complex) #AE complex
        E2[:size, :size] = self.A
        E2[:size, size:] = self.B
        E2[size:, :size] = self.B.conjugate() #AE added conjugtate 
        E2[size:, size:] = self.A.conjugate() #AE added conjugtate 

        
        num_parameters = len(self.G_ops) + len(self.q_ops_resp)
        A_mat     = np.zeros((num_parameters, num_parameters), dtype=complex)
        B_mat     = np.zeros((num_parameters, num_parameters), dtype=complex)
        Sigma_mat = np.zeros((num_parameters, num_parameters), dtype=complex)
        Delta_mat = np.zeros((num_parameters, num_parameters), dtype=complex)

        if len(self.q_ops_resp) != 0:
            A_mat[:len(self.wf.kappa_no_activeactive_spin_idx_ep), :len(self.wf.kappa_no_activeactive_spin_idx_ep)] = get_orbital_response_hessian_block(
                self.wf.h_mo, self.wf.g_mo,
                #self.wf.kappa_no_activeactive_spin_idx_dagger_resp,
                #self.wf.kappa_no_activeactive_spin_idx_resp,
                self.wf.kappa_no_activeactive_spin_idx_ep_dagger,
                self.wf.kappa_no_activeactive_spin_idx_ep,
                self.wf.num_spin_orbs_NES,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1, self.wf.rdm2,
            )
            A_mat[: len(self.wf.kappa_no_activeactive_spin_idx_ep), len(self.wf.kappa_no_activeactive_spin_idx_ep) : len(self.q_ops_resp)] = get_orbital_response_hessian_block(
                self.wf.h_mo, self.wf.g_mo,
                self.wf.kappa_no_activeactive_spin_idx_ep_dagger,
                self.wf.kappa_no_activeactive_spin_idx,
                self.wf.num_spin_orbs_NES,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1, self.wf.rdm2,
            )
            A_mat[len(self.wf.kappa_no_activeactive_spin_idx_ep) : len(self.q_ops_resp), : len(self.wf.kappa_no_activeactive_spin_idx_ep)] = get_orbital_response_hessian_block(
                self.wf.h_mo, self.wf.g_mo,
                self.wf.kappa_no_activeactive_spin_idx_dagger,
                self.wf.kappa_no_activeactive_spin_idx_ep,
                self.wf.num_spin_orbs_NES,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1, self.wf.rdm2,
            )
            A_mat[len(self.wf.kappa_no_activeactive_spin_idx_ep) : len(self.q_ops_resp), len(self.wf.kappa_no_activeactive_spin_idx_ep) : len(self.q_ops_resp)] = get_orbital_response_hessian_block(
                self.wf.h_mo, self.wf.g_mo,
                self.wf.kappa_no_activeactive_spin_idx_dagger,
                self.wf.kappa_no_activeactive_spin_idx,
                self.wf.num_spin_orbs_NES,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1, self.wf.rdm2,
            )

            B_mat[: len(self.q_ops_resp), : len(self.q_ops_resp)] = get_orbital_response_hessian_block(
                self.wf.h_mo, self.wf.g_mo,
                self.wf.kappa_no_activeactive_spin_idx_dagger_resp,
                self.wf.kappa_no_activeactive_spin_idx_dagger_resp,
                self.wf.num_spin_orbs_NES,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1, self.wf.rdm2,
            )

            Sigma_mat[: len(self.q_ops_resp), : len(self.q_ops_resp)] = get_orbital_response_metric_sigma(
                self.wf.kappa_no_activeactive_spin_idx_resp,
                self.wf.num_spin_orbs_NES, 
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1,
            )
        

        size = len(A_mat)
        E2_mat = np.zeros((size * 2, size * 2), dtype=complex)
        E2_mat[:size, :size] = A_mat
        E2_mat[:size, size:] = B_mat
        E2_mat[size:, :size] = B_mat.conjugate()
        E2_mat[size:, size:] = A_mat.conjugate()


        S_mat = np.zeros((size * 2, size * 2), dtype=complex) #AE complex
        S_mat[:size, :size] = Sigma_mat
        S_mat[:size, size:] = Delta_mat
        S_mat[size:, :size] = -Delta_mat.conjugate()
        S_mat[size:, size:] = -Sigma_mat.conjugate()

        '''# Transform perturbed 2e integrals to MO basis
        # g_ssss shape: (3, n2c, n2c, n2c, n2c) — need to embed in n4c first
        n4c = self.wf.c_mo.shape[0]
        n2c = n4c // 2

        # Embed in full 4c space and transform
        h_2e_correction = np.zeros((3, size_mo, size_mo), dtype=complex)

        for b in range(3):
            # Coulomb - exchange for each b direction
            # Contract over rs with RDM1: eff_pq = sum_rs (pq|rs) D_rs - (ps|rq) D_rs
            
            # Build effective AO matrix
            eff_ssss = np.zeros((n2c, n2c), dtype=complex)
            eff_lsss = np.zeros((n2c, n2c), dtype=complex)

            for i in range(size_mo):
                for j in range(size_mo):
                    rdm1_ij = RDM1(
                        i, j,
                        self.wf.num_spin_orbs_NES,
                        self.wf.num_inactive_spin_orbs,
                        self.wf.num_active_spin_orbs,
                        self.wf.rdm1,
                    )
                    # Coulomb: sum_rs g[b,p,q,r,s] * D[r,s]
                    eff_ssss += rdm1_ij * np.einsum(
                        'pqrs,r,s->pq',
                        g_ssss[b],
                        self.wf.c_mo[n2c:, i],
                        self.wf.c_mo[n2c:, j].conj()
                    )
                    eff_lsss += rdm1_ij * np.einsum(
                        'pqrs,r,s->pq',
                        g_lsss[b],
                        self.wf.c_mo[n2c:, i],
                        self.wf.c_mo[n2c:, j].conj()
                    )

            # Embed in full 4c AO matrix
            eff_ao = np.zeros((n4c, n4c), dtype=complex)
            eff_ao[n2c:, n2c:] = eff_ssss  # SS block
            eff_ao[:n2c, :n2c] = eff_lsss  # LL block

            # Transform to MO basis
            h_2e_correction[b] = DHF_one_electron_transform(self.wf.c_mo, eff_ao)'''


        # Paramagnetic:
        # Property gradients: shape (num_parameters, 3)
        prop_grads_h1 = [self.get_property_gradient_4comp_no_ep(h1[I]) for I in range(natm)]
        prop_grads_B  = [self.get_property_gradient_4comp_no_ep(hb[I]) for I in range(natm)]

        # prop_grads_h1 = [self.get_property_gradient_4comp(h1[I]) for I in range(natm)]
        # prop_grads_B  = [self.get_property_gradient_4comp(hb[I]) for I in range(natm)]

        # response_B = [
        #     np.linalg.pinv(E2_mat, rcond=1e-10) @ prop_grads_B[I]
        #     for I in range(natm)
        # ]



        def solve_lr_drop_sigma_null(H, sigma, prop_grads1=None, prop_grads2=None, cut=1e-10):
            """
            H: Hessian
            sigma: metric / overlap-like matrix
            prop_grads: optional list or array of RHS vectors (or dict of them)

            Returns:
                w, v, s, keep, (and projected prop_grads if given)
            """

            # 1) Hermitize
            Hh = 0.5 * (H + H.conj().T)
            Sh = 0.5 * (sigma + sigma.conj().T)

            # 2) eigen-decompose metric
            s, U = la.eigh(Sh)

            # 3) select non-null subspace
            keep = np.abs(s) > cut * np.max(np.abs(s))
            Uk = U[:, keep]

            # 4) project operators
            Hk = Uk.conj().T @ Hh @ Uk
            Sk = Uk.conj().T @ Sh @ Uk

            # 5) project property gradients (THIS WAS MISSING)
            prop_grads_k1 = None
            if prop_grads1 is not None:

                # case 1: single vector
                if isinstance(prop_grads1, np.ndarray) and prop_grads1.ndim == 1:
                    prop_grads_k1 = Uk.conj().T @ prop_grads1

                # case 2: list of vectors
                elif isinstance(prop_grads1, (list, tuple)):
                    prop_grads_k1 = [Uk.conj().T @ g for g in prop_grads1]

                # case 3: stacked array (n_props, n_dim)
                elif isinstance(prop_grads1, np.ndarray) and prop_grads1.ndim == 2:
                    prop_grads_k1 = np.array([Uk.conj().T @ g for g in prop_grads1])

                else:
                    raise ValueError("Unsupported prop_grads format")
                
            # 5) project property gradients (THIS WAS MISSING)
            prop_grads_k2 = None
            if prop_grads2 is not None:

                # case 1: single vector
                if isinstance(prop_grads2, np.ndarray) and prop_grads2.ndim == 1:
                    prop_grads_k2 = Uk.conj().T @ prop_grads2

                # case 2: list of vectors
                elif isinstance(prop_grads2, (list, tuple)):
                    prop_grads_k2 = [Uk.conj().T @ g for g in prop_grads2]

                # case 3: stacked array (n_props, n_dim)
                elif isinstance(prop_grads2, np.ndarray) and prop_grads2.ndim == 2:
                    prop_grads_k2 = np.array([Uk.conj().T @ g for g in prop_grads2])

                else:
                    raise ValueError("Unsupported prop_grads format")

            # 6) solve reduced generalized eigenproblem
            w, y = la.eig(Hk, Sk)

            # 7) backtransform eigenvectors
            v = Uk @ y

            return Hk, Sk, prop_grads_k1, prop_grads_k2


        #E2_mat_trans, S_mat_trans, prop_grads_h1_trans, prop_grads_B_trans = solve_lr_drop_sigma_null(E2_mat,S_mat,prop_grads_h1,prop_grads_B)


        
        '''# print(E2)
        (
            hess_eigval,
            _,
        ) = np.linalg.eig(E2_mat_trans)
        print(f"Smallest absolute value of Hessian eigenvalues: {np.min(np.abs(hess_eigval))}")
        if np.abs(np.min(hess_eigval)) < 10**-3:
            print("WARNING: Small eigenvalue in Hessian")
        for i in hess_eigval:
            if np.abs(i) < 1e-3:
                print("Small eigenvalue in Hessian:",i)
            #raise ValueError("Negative eigenvalue in Hessian.")
        
        # print(S)
        print(f"Smallest diagonal element in the metric: {np.min(np.abs(np.diagonal(S_mat_trans)))}")'''



        response_B  = [solve(E2, prop_grads_B[I]) for I in range(natm)]

        msc_para = np.zeros((natm, 3, 3))

        for I in range(natm):
            for alpha in range(3):
                for beta in range(3):
                    msc_para[I, alpha, beta] = np.einsum(
                        'i,i->',
                        -prop_grads_h1[I][:, alpha].conj(),
                        response_B[I][:, beta]
                    ).real


        # Diamagnetic:
        size_mo = (
            self.wf.num_spin_orbs_NES
            + self.wf.num_inactive_spin_orbs
            + self.wf.num_active_spin_orbs
            + self.wf.num_virtual_spin_orbs
        )

        mo2 = np.zeros((natm, 3, 3, size_mo, size_mo), dtype=complex)

        for I in range(natm):
            for a in range(3):
                for b in range(3):
                    mo2[I, a, b] = DHF_one_electron_transform(
                        self.wf.c_mo,
                        h2[I, a, b]
                    )

        msc_dia = np.zeros((natm, 3, 3))

        for I in range(natm):
            for a in range(3):
                for b in range(3):
                    for i in range(size_mo):
                        for j in range(size_mo):
                            rdm1_ij = RDM1(
                                i, j,
                                self.wf.num_spin_orbs_NES,
                                self.wf.num_inactive_spin_orbs,
                                self.wf.num_active_spin_orbs,
                                self.wf.rdm1,
                            )
                            msc_dia[I, a, b] += (mo2[I, a, b, i, j] * rdm1_ij).real

        # Units, returning and printing:
        unit_ppm = nist.ALPHA**2 * 1e6

        msc_para *= unit_ppm
        msc_dia *= unit_ppm

        sigma_total = msc_para + msc_dia

        print("para")
        print(msc_para)
        print("dia")
        print(msc_dia)

        return np.trace(sigma_total, axis1=1, axis2=2) / 3
    


