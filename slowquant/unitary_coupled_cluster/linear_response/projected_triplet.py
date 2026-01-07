import numpy as np

from slowquant.unitary_coupled_cluster.density_matrix import (
    ReducedDenstiyMatrix,
    get_orbital_gradient_response,
    get_triplet_orbital_response_hessian_block,
    get_orbital_response_metric_sigma,
)
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass_triplet import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    expectation_value,
    propagate_state,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS


class LinearResponseUCC(LinearResponseBaseClass):
    def __init__(
        self,
        wave_function: WaveFunctionUCC | WaveFunctionUPS,
        excitations: str,
    ) -> None:
        """Initialize linear response by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
        """
        super().__init__(wave_function, excitations)

        rdms = ReducedDenstiyMatrix(
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
            self.wf.rdm1,
            rdm2=self.wf.rdm2,
            t_rdm2=self.wf.t_rdm2,
        )
        idx_shift = len(self.q_ops)
        print("Gs", len(self.G_ops))
        print("qs", len(self.q_ops))
        grad = get_orbital_gradient_response(
            rdms,
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.kappa_no_activeactive_idx,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        if len(grad) != 0:
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
        # Do orbital-orbital blocks
        self.A[: len(self.q_ops), : len(self.q_ops)] = get_triplet_orbital_response_hessian_block(
            rdms,
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.kappa_no_activeactive_idx_dagger,
            self.wf.kappa_no_activeactive_idx,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        self.B[: len(self.q_ops), : len(self.q_ops)] = get_triplet_orbital_response_hessian_block(
            rdms,
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.kappa_no_activeactive_idx_dagger,
            self.wf.kappa_no_activeactive_idx_dagger,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        self.Sigma[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_metric_sigma(
            rdms, self.wf.kappa_no_activeactive_idx
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
                # - 1/2<0| Gd qd H |0>
                val = (
                    -1
                    / 2
                    * expectation_value(
                        G_ket,
                        [],
                        qdH_ket,
                        *self.index_info,
                    )
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

