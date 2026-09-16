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
                # Make B
                # - <CSF| Gd Ud qd H |0>
                val = - expectation_value(
                        G_ket,
                        [],
                        UdqdH_ket,
                        *self.index_info,
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
            Udopd_ket = propagate_state(["Ud", op.dagger], self.wf.ci_coeffs, *self.index_info)
            for idx, G in enumerate(self.G_ops):
                G_ket = propagate_state([G], self.wf.csf_coeffs, *self.index_info)
                # - < 0 | op U G | CSF >
                V[idx + idx_shift_q, comp] -= expectation_value(
                    Udopd_ket,
                    [],
                    G_ket,
                    *self.index_info
                )
        
        return np.vstack((V, fac * V))
