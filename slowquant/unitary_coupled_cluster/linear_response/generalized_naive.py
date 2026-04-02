from collections.abc import Sequence

import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform, generalized_one_electron_transform,
)
from slowquant.unitary_coupled_cluster.generalized_density_matrix import (
    get_orbital_gradient_response, get_orbital_gradient_response_real_imag,
    get_orbital_response_hessian_block,
    get_orbital_response_metric_sigma,
    get_orbital_response_property_gradient_annika, get_orbital_response_property_gradient_real_imag, 
    get_orbital_response_metric_sigma_real_imag,  get_orbital_response_static_property_gradient, 
)

from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.linear_response.generalized_lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import (
    generalized_expectation_value,
    generalized_propagate_state,
)
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    expectation_value,
)
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
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
        # finite_excitations_idx = np.array(finite_excitations)
        
              
              
              
              
                
        if len(self.q_ops) != 0:
            grad = get_orbital_gradient_response_real_imag(
                self.wf.h_mo,
                self.wf.g_mo,
                self.wf.kappa_no_activeactive_spin_idx,
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
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1,
                self.wf.rdm2,
            )

            self.Sigma[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_metric_sigma(
                self.wf.kappa_no_activeactive_spin_idx,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.rdm1,
            )
      
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

                    

            #manuel slut
        
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
    



    

