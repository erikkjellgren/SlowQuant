import numpy as np

from slowquant.unitary_coupled_cluster.density_matrix import (
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


class LinearResponse(LinearResponseBaseClass):
    def __init__(
        self,
        wave_function: WaveFunctionUCC | WaveFunctionUPS,
        excitations: str,
        tda: bool = False,
    ) -> None:
        """Initialize linear response by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
            tda: Whether to use Tamm-Dancoff Approximation.
        """
        super().__init__(wave_function, excitations, tda)

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
        UdH00_ket = propagate_state(["Ud", self.H_0i_0a], self.wf.ci_coeffs, *self.index_info)
        for i, op in enumerate(self.G_ops):
            G_ket = propagate_state(
                [op],
                self.wf.csf_coeffs,
                *self.index_info,
            )
            # - <0| H U G |CSF>
            grad[i] = -expectation_value(
                UdH00_ket,
                [],
                G_ket,
                *self.index_info,
            )
            # <0| Gd Ud H |0>
            grad[i + len(self.G_ops)] = expectation_value(
                G_ket,
                [],
                UdH00_ket,
                *self.index_info,
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad)))
        if len(self.q_ops) != 0:
            # Do orbital-orbital blocks
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
            if not self.tda:
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
            UdHq_ket = propagate_state(["Ud", self.H_1i_1a * qJ], self.wf.ci_coeffs, *self.index_info)
            UdqdH_ket = propagate_state(["Ud", qJ.dagger * self.H_1i_1a], self.wf.ci_coeffs, *self.index_info)
            for i, GI in enumerate(self.G_ops):
                G_ket = propagate_state([GI], self.wf.csf_coeffs, *self.index_info)
                # Make A
                # <CSF| Gd Ud H q |0>
                val = expectation_value(
                    G_ket,
                    [],
                    UdHq_ket,
                    *self.index_info,
                )
                self.A[j, i + idx_shift] = self.A[i + idx_shift, j] = val
                if not self.tda:
                    # Make B
                    # - 1/2<CSF| Gd Ud qd H |0>
                    val = (
                        -1
                        / 2
                        * expectation_value(
                            G_ket,
                            [],
                            UdqdH_ket,
                            *self.index_info,
                        )
                    )
                    self.B[j, i + idx_shift] = self.B[i + idx_shift, j] = val
        for j, GJ in enumerate(self.G_ops):
            UdHUGJ = propagate_state(
                ["Ud", self.H_0i_0a, "U", GJ],
                self.wf.csf_coeffs,
                *self.index_info,
            )
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                # <CSF| GId Ud H U GJ | CSF>
                val = expectation_value(
                    self.wf.csf_coeffs,
                    [GI.dagger],
                    UdHUGJ,
                    *self.index_info,
                )
                if i == j:
                    val -= self.wf.energy_elec
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make Sigma
                if i == j:
                    self.Sigma[i + idx_shift, j + idx_shift] = 1
