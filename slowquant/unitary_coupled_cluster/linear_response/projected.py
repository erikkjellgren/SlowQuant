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
from slowquant.unitary_coupled_cluster.operators import one_elec_op_0i_0a, hamiltonian_0i_0a
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
            # <0| H G |0>
            grad[i] = expectation_value(
                H00_ket,
                [],
                G_ket,
                *self.index_info,
            )
            # - E * <0| G |0>
            grad[i] -= self.wf.energy_elec * expectation_value(self.wf.ci_coeffs, [], G_ket, *self.index_info)
            # <0| Gd H |0>
            grad[i + len(self.G_ops)] = expectation_value(
                G_ket,
                [],
                H00_ket,
                *self.index_info,
            )
            # - E * <0| Gd |0>
            grad[i + len(self.G_ops)] -= self.wf.energy_elec * expectation_value(
                G_ket, [], self.wf.ci_coeffs, *self.index_info
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
                # Make A
                # <0| Gd H q |0>
                val = expectation_value(
                    G_ket,
                    [],
                    Hq_ket,
                    *self.index_info,
                )
                self.A[j, i + idx_shift] = self.A[i + idx_shift, j] = val
                # Make B
                # - <0| Gd qd H |0>
                val = - expectation_value(
                        G_ket,
                        [],
                        qdH_ket,
                        *self.index_info,
                )
                self.B[j, i + idx_shift] = self.B[i + idx_shift, j] = val
        for j, GJ in enumerate(self.G_ops):
            GJ_ket = propagate_state([GJ], self.wf.ci_coeffs, *self.index_info)
            HGJ_ket = propagate_state([self.H_0i_0a], GJ_ket, *self.index_info)
            for i, GI in enumerate(self.G_ops[j:], j):
                GI_ket = propagate_state([GI], self.wf.ci_coeffs, *self.index_info)
                # Make A
                # <0| GId H GJ |0>
                val = expectation_value(
                    GI_ket,
                    [],
                    HGJ_ket,
                    *self.index_info,
                )
                # <0 | GId |0> * <0| GJ |0> * E
                val += (
                    expectation_value(
                        GI_ket,
                        [],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [],
                        GJ_ket,
                        *self.index_info,
                    )
                    * self.wf.energy_elec
                )
                # - <0| GId GJ |0> * E
                val -= (
                    expectation_value(
                        GI_ket,
                        [],
                        GJ_ket,
                        *self.index_info,
                    )
                    * self.wf.energy_elec
                )
                # - 1/2*<0| GId |0> * <0| H GJ |0>
                val -= (
                    1
                    / 2
                    * expectation_value(
                        GI_ket,
                        [],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [],
                        HGJ_ket,
                        *self.index_info,
                    )
                )
                # - 1/2*<0| GJ |0> * <0| GId H |0>
                val -= (
                    1
                    / 2
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [],
                        GJ_ket,
                        *self.index_info,
                    )
                    * expectation_value(
                        GI_ket,
                        [self.H_0i_0a],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                )
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make B
                # 1/2<0| GId H |0> * <0| GJd |0>
                val = (
                    1
                    / 2
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [GI.dagger, self.H_0i_0a],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [GJ.dagger],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                )
                # 1/2<0| GJd H |0> * <0| GId |0>
                val += (
                    1
                    / 2
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [GJ.dagger, self.H_0i_0a],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [GI.dagger],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                )
                # - <0| GId |0> * <0| GJd |0> * E
                val -= (
                    expectation_value(
                        GI_ket,
                        [],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    * expectation_value(
                        GJ_ket,
                        [],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    * self.wf.energy_elec
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
                # - <0| GId |0> * <0| GJ |0>
                val -= expectation_value(
                    GI_ket,
                    [],
                    self.wf.ci_coeffs,
                    *self.index_info,
                ) * expectation_value(
                    self.wf.ci_coeffs,
                    [],
                    GJ_ket,
                    *self.index_info,
                )
                self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[j + idx_shift, i + idx_shift] = val

    def get_property_gradient(self, int1e: np.ndarray, int2e: np.ndarray | None = None) -> np.ndarray:
        """Calculate property gradient.

        Args:
            int1e: one-electron property integrals in MO basis.
            int2e: two-electron property integrals in MO basis.

        Returns:
            Property gradient.
        """ 

        if np.allclose(int1e, int1e.transpose(0, -1, -2)):
            # real integral
            fac = -1
        elif np.allclose(int1e, -1 * int1e.transpose(0, -1, -2)):
            # imaginary integral
            fac = 1
        else:
            raise ValueError("Wrong symmetry: int1e must be symmetric or antisymmetric")

        if int2e is not None:
            if len(int1e) != len(int2e):
                raise ValueError(f"Mismatched arrays: int1e and int2e must have the same length, got {len(int1e)} and {len(int2e)}")
            if self.triplet:
                raise ValueError("Not implemented: triplet response and int2e cannot be used simutaniously.")
            if not np.allclose(int2e, -1 * fac * int2e.transpose(0,2,1,4,3)):
                raise ValueError("Mismatched symmetry: int1e and int2e must either both be symmetric or antisymmetric")

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

        for comp, op_int1e in enumerate(int1e):
            if int2e is None:
                op = one_elec_op_0i_0a(op_int1e, self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.triplet)
            else:
                op = hamiltonian_0i_0a(op_int1e, int2e[comp], self.wf.num_inactive_orbs, self.wf.num_active_orbs)
            op_ket = propagate_state([op], self.wf.ci_coeffs, *self.index_info)
            opd_ket = propagate_state([op.dagger], self.wf.ci_coeffs, *self.index_info)
            for idx, G in enumerate(self.G_ops):
                G_ket = propagate_state([G], self.wf.ci_coeffs, *self.index_info)
                # < 0 | op | 0 > * < 0 | G | 0 >
                V[idx + idx_shift_q, comp] += (
                    expectation_value(
                        self.wf.ci_coeffs,
                        [],
                        op_ket,
                        *self.index_info
                    )
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [],
                        G_ket,
                        *self.index_info
                    )
                )
                # - < 0 | op G | 0 >
                V[idx + idx_shift_q, comp] -= expectation_value(
                    opd_ket,
                    [],
                    G_ket,
                    *self.index_info
                )

        return np.vstack((V, fac * V))
