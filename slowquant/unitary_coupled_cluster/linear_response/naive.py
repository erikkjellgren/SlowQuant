import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.density_matrix import (
    get_orbital_gradient_response,
    get_orbital_response_hessian_block,
    get_orbital_response_metric_sigma,
    get_orbital_response_property_gradient,
)
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    expectation_value,
    propagate_state,
)
from slowquant.unitary_coupled_cluster.operators import one_elec_op_0i_0a
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS


class LinearResponse(LinearResponseBaseClass):
    def __init__(
        self,
        wave_function: WaveFunctionUCC | WaveFunctionUPS,
        excitations: list[str],
        do_orbital_response: bool = True,
    ) -> None:
        """Initialize linear response by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
        """
        super().__init__(wave_function, excitations, do_orbital_response)

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
                # - 1/2<0| H q Gd |0>
                val -= (
                    1
                    / 2
                    * expectation_value(
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
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [self.H_1i_1a * GI.dagger * qJ],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
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
                # - 1/2*<0| Gd qd H |0>
                val -= (
                    1
                    / 2
                    * expectation_value(
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
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [qJ.dagger * GI.dagger * self.H_1i_1a],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
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
                # - 1/2<0| GId GJ H |0>
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

    def _get_transition_property(self, property_integral: np.ndarray) -> np.ndarray:
        """Calculate transition dipole moment.

        Args:
            property_integral: Property integral.

        Returns:
            Transition property.
        """
        number_excitations = len(self.excitation_energies)
        prop = one_electron_integral_transform(self.wf.c_mo, property_integral)
        prop_op = one_elec_op_0i_0a(
            prop,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        prop_ket = propagate_state([prop_op], self.wf.ci_coeffs, *self.index_info)
        propd_ket = propagate_state([prop_op.dagger], self.wf.ci_coeffs, *self.index_info)
        transition_property = 0.0
        transition_properties = np.zeros(number_excitations, dtype=float)
        for state_number in range(number_excitations):
            transfer_op = FermionicOperator({})
            for i, G in enumerate(self.G_ops):
                transfer_op += (
                    self.Z_G_normed[i, state_number] * G.dagger + self.Y_G_normed[i, state_number] * G
                )
            q_part = 0.0
            if len(self.q_ops) != 0:
                q_part = get_orbital_response_property_gradient(
                    prop,
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.normed_response_vectors,
                    state_number,
                    number_excitations,
                )
            transfer_ket = propagate_state([transfer_op], self.wf.ci_coeffs, *self.index_info)
            transferd_ket = propagate_state([transfer_op.dagger], self.wf.ci_coeffs, *self.index_info)
            # <0| prop T |0>
            transition_property = expectation_value(
                propd_ket,
                [],
                transfer_ket,
                *self.index_info,
            )
            # - <0| T prop |0>
            transition_property -= expectation_value(
                transferd_ket,
                [],
                prop_ket,
                *self.index_info,
            )
            transition_properties[state_number] = q_part + transition_property
        return transition_properties
