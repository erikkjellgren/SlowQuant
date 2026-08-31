import numpy as np

from slowquant.unitary_coupled_cluster.density_matrix import (
    get_orbital_gradient_response,
    get_orbital_response_hessian_block,
    get_triplet_orbital_response_hessian_block,
    get_orbital_response_metric_sigma,
    get_orbital_response_property_gradient_1e,
    get_orbital_response_property_gradient_2e,
)
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    expectation_value,
    propagate_state,
)
from slowquant.unitary_coupled_cluster.operators import Epq, Tpq, epqrs
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS


class LinearResponse(LinearResponseBaseClass):
    def __init__(
        self,
        wave_function: WaveFunctionUCC | WaveFunctionUPS,
        excitations: str,
        triplet: bool = False,
    ) -> None:
        """Initialize linear response by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
            triplet: If the linear response should be triplet spin-adapted.
        """
        super().__init__(wave_function, excitations, triplet)

        idx_shift = len(self.q_ops)
        print("Gs", len(self.G_ops))
        print("qs", len(self.q_ops))
        if len(self.q_ops) != 0:
            grad = get_orbital_gradient_response(
                self.wf.h_mo,
                self.wf.g_mo,
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1,
                self.wf.rdm2,
            )
            print("idx, max(abs(grad orb)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in q of ", np.max(np.abs(grad)))

        grad = np.zeros(2 * len(self.G_ops))
        H00_ket = propagate_state([self.H_0i_0a], self.wf.ci_coeffs, *self.index_info)
        for i, op in enumerate(self.G_ops):
            G_ket = propagate_state([op], self.wf.ci_coeffs, *self.index_info)
            Gd_ket = propagate_state([op.dagger], self.wf.ci_coeffs, *self.index_info)
            # <0 | H G |0>
            grad[i] = expectation_value(
                H00_ket,
                [],
                G_ket,
                *self.index_info,
            )
            # - <0| G H |0>
            grad[i] -= expectation_value(
                Gd_ket,
                [],
                H00_ket,
                *self.index_info,
            )
            # <0| Gd H |0>
            grad[i + len(self.G_ops)] = expectation_value(
                G_ket,
                [],
                H00_ket,
                *self.index_info,
            )
            # - <0| H Gd |0>
            grad[i + len(self.G_ops)] -= expectation_value(
                H00_ket,
                [],
                Gd_ket,
                *self.index_info,
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad)))
        if len(self.q_ops) != 0:
            # Do orbital-orbital blocks
            if not self.triplet:
                self.A[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_hessian_block(
                    self.wf.h_mo,
                    self.wf.g_mo,
                    self.wf.kappa_no_activeactive_idx_dagger,
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.wf.rdm2,
                )
                self.B[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_hessian_block(
                    self.wf.h_mo,
                    self.wf.g_mo,
                    self.wf.kappa_no_activeactive_idx_dagger,
                    self.wf.kappa_no_activeactive_idx_dagger,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.wf.rdm2,
                )
            else:
                self.A[: len(self.q_ops), : len(self.q_ops)] = get_triplet_orbital_response_hessian_block(
                    self.wf.h_mo,
                    self.wf.g_mo,
                    self.wf.kappa_no_activeactive_idx_dagger,
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.wf.rdm2,
                    self.wf.t_rdm2,
                )
                self.B[: len(self.q_ops), : len(self.q_ops)] = get_triplet_orbital_response_hessian_block(
                    self.wf.h_mo,
                    self.wf.g_mo,
                    self.wf.kappa_no_activeactive_idx_dagger,
                    self.wf.kappa_no_activeactive_idx_dagger,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.wf.rdm2,
                    self.wf.t_rdm2,
                )                
            self.Sigma[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_metric_sigma(
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1,
            )
        for j, qJ in enumerate(self.q_ops):
            Hq_ket = propagate_state([self.H_1i_1a * qJ], self.wf.ci_coeffs, *self.index_info)
            qdH_ket = propagate_state([qJ.dagger * self.H_1i_1a], self.wf.ci_coeffs, *self.index_info)
            for i, GI in enumerate(self.G_ops):
                G_ket = propagate_state([GI], self.wf.ci_coeffs, *self.index_info)
                Gd_ket = propagate_state([GI.dagger], self.wf.ci_coeffs, *self.index_info)
                # Make A
                # <0| Gd H q |0>
                val = expectation_value(
                    G_ket,
                    [],
                    Hq_ket,
                    *self.index_info,
                )
                # - <0| H q Gd |0>
                val -= expectation_value(
                        qdH_ket,
                        [],
                        Gd_ket,
                        *self.index_info,
                )
                self.A[i + idx_shift, j] = self.A[j, i + idx_shift] = val
                # Make B
                # <0| qd H Gd |0>
                val = expectation_value(
                    Hq_ket,
                    [],
                    Gd_ket,
                    *self.index_info,
                )
                # - <0| Gd qd H |0>
                val -= expectation_value(
                        G_ket,
                        [],
                        qdH_ket,
                        *self.index_info,
                )
                self.B[i + idx_shift, j] = self.B[j, i + idx_shift] = val
        for j, GJ in enumerate(self.G_ops):
            GJH_ket = propagate_state([GJ], H00_ket, *self.index_info)
            GJdH_ket = propagate_state([GJ.dagger], H00_ket, *self.index_info)
            HGJd_ket = propagate_state([self.H_0i_0a, GJ.dagger], self.wf.ci_coeffs, *self.index_info)
            HGJ_ket = propagate_state([self.H_0i_0a, GJ], self.wf.ci_coeffs, *self.index_info)
            GJ_ket = propagate_state([GJ], self.wf.ci_coeffs, *self.index_info)
            GJd_ket = propagate_state([GJ.dagger], self.wf.ci_coeffs, *self.index_info)
            for i, GI in enumerate(self.G_ops[j:], j):
                GI_ket = propagate_state([GI], self.wf.ci_coeffs, *self.index_info)
                GId_ket = propagate_state([GI.dagger], self.wf.ci_coeffs, *self.index_info)
                # Make A
                # <0| GId H GJ |0>
                val = expectation_value(
                    GI_ket,
                    [],
                    HGJ_ket,
                    *self.index_info,
                )
                # <0| GJ H GId |0>
                val += expectation_value(
                    HGJd_ket,
                    [],
                    GId_ket,
                    *self.index_info,
                )
                # - 1/2*<0| GId GJ H |0>
                val -= (
                    1
                    / 2
                    * expectation_value(
                        GI_ket,
                        [],
                        GJH_ket,
                        *self.index_info,
                    )
                )
                # - 1/2*<0| H GJ GId |0>
                val -= (
                    1
                    / 2
                    * expectation_value(
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
                    * expectation_value(
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
                    * expectation_value(
                        H00_ket,
                        [GI.dagger],
                        GJ_ket,
                        *self.index_info,
                    )
                )
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make B
                # <0| GId H GJd |0>
                val = expectation_value(
                    GI_ket,
                    [],
                    HGJd_ket,
                    *self.index_info,
                )
                # - <0| GId GJd H |0>
                val -= expectation_value(
                    GI_ket,
                    [],
                    GJdH_ket,
                    *self.index_info,
                )
                # - <0| H GJd GId |0>
                val -= expectation_value(
                    GJH_ket,
                    [],
                    GId_ket,
                    *self.index_info,
                )
                # <0| GJd H GId |0>
                val += expectation_value(
                    HGJ_ket,
                    [],
                    GId_ket,
                    *self.index_info,
                )
                self.B[i + idx_shift, j + idx_shift] = self.B[j + idx_shift, i + idx_shift] = val
                # Make Sigma
                # <0| GId GJ |0>
                val = expectation_value(
                    GI_ket,
                    [],
                    GJ_ket,
                    *self.index_info,
                )
                # - <0| GJ GId |0>
                val -= expectation_value(
                    GJd_ket,
                    [],
                    GId_ket,
                    *self.index_info,
                )
                self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[j + idx_shift, i + idx_shift] = val

    def get_property_gradient(self, int1e: np.ndarray, int2e: np.ndarray | None = None) -> np.ndarray:
        """Calculate property gradient.

        Args:
            prop_int1e: one-electron property integrals in MO basis.
            prop_int2e: two-electron property integrals in MO basis.

        Returns:
            Property gradient.
        """

        # Check if singlet or triplet response
        if not self.triplet:
            E = Epq
        else:
            if int2e is not None:
                raise ValueError("Got triplet=True and int2e is not None, cannot be done simultaneously.")
            E = Tpq

        # Check that int1e and int2e match
        if int2e is not None:
            if len(int1e) != len(int2e):
                raise ValueError(f"Cartesian components in int1e and int2e must match, got {len(int1e)} and {len(int2e)}")

        idx_shift_q = len(self.q_ops)
        V = np.zeros((len(self.q_ops + self.G_ops), len(int1e)))

        if len(self.q_ops) != 0:
            # Orbital response part
            V[:idx_shift_q, :] = get_orbital_response_property_gradient_1e(
                int1e,
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1,
            )

            if int2e is not None:
                V[:idx_shift_q, :] += get_orbital_response_property_gradient_2e(
                        int2e,
                        self.wf.kappa_no_activeactive_idx,
                        self.wf.num_inactive_orbs,
                        self.wf.num_active_orbs,
                        self.wf.rdm1,
                        self.wf.rdm2,
                    )

        for idx, G in enumerate(self.G_ops):
            G_ket = propagate_state([G], self.wf.ci_coeffs, *self.index_info)
            Gd_ket = propagate_state([G.dagger], self.wf.ci_coeffs, *self.index_info)
            # one-electron part
            # Inactive part
            for i in range(self.wf.num_inactive_orbs):
                E_ket = propagate_state([E(i, i)], self.wf.ci_coeffs, *self.index_info) 
                # < 0 | G E | 0 >
                val = expectation_value(
                    Gd_ket, 
                    [], 
                    E_ket, 
                    *self.index_info
                )
                # - < 0 | E G | 0 >
                val -= expectation_value(
                    E_ket, # E_ket = Ed_ket for E(i,i)
                    [], 
                    G_ket, 
                    *self.index_info
                ) 
                V[idx + idx_shift_q, :] += int1e[:, i, i] * val
            # Active part
            for v in range(self.wf.num_inactive_orbs, self.wf.num_inactive_orbs + self.wf.num_active_orbs):
                for w in range(
                    self.wf.num_inactive_orbs, self.wf.num_inactive_orbs + self.wf.num_active_orbs
                ):
                    E_ket = propagate_state([E(v, w)], self.wf.ci_coeffs, *self.index_info)
                    Ed_ket = propagate_state([E(w, v)], self.wf.ci_coeffs, *self.index_info)
                    # < 0 | G E | 0 >
                    val = expectation_value(
                        Gd_ket, 
                        [], 
                        E_ket, 
                        *self.index_info
                    )
                    # - < 0 | E G | 0 >
                    val -= expectation_value(
                        Ed_ket, 
                        [], 
                        G_ket, 
                        *self.index_info
                    )
                    V[idx + idx_shift_q, :] += int1e[:, v, w] * val

            # two-electron part
            if int2e is not None:  # seperate in inactive and active latter
                for p in range(self.wf.num_inactive_orbs + self.wf.num_active_orbs):
                    for q in range(self.wf.num_inactive_orbs + self.wf.num_active_orbs):
                        for r in range(self.wf.num_inactive_orbs + self.wf.num_active_orbs):
                            for s in range(self.wf.num_inactive_orbs + self.wf.num_active_orbs):
                                e_ket = propagate_state([epqrs(p, q, r, s)], self.wf.ci_coeffs, *self.index_info)
                                ed_ket = propagate_state([epqrs(s, r, q, p)], self.wf.ci_coeffs, *self.index_info)
                                # < 0 | G e | 0 >
                                val = expectation_value(
                                    Gd_ket, 
                                    [], 
                                    e_ket, 
                                    *self.index_info
                                )
                                # - < 0 | e G | 0 >
                                val -= expectation_value(
                                    ed_ket, 
                                    [], 
                                    G_ket, 
                                    *self.index_info
                                )
                                V[idx + idx_shift_q, :] += int2e[:, p, q, r, s] * val      
        
        if np.allclose(int1e, int1e.transpose(0, -1, -2)): # check if 2e are also imagniry, if one is and the other isn't throw and error
            return np.vstack((V, -1 * V))
        return np.vstack((V, V))
