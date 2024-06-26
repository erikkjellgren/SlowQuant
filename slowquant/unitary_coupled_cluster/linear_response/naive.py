from collections.abc import Sequence

import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.density_matrix import (
    ReducedDenstiyMatrix,
    get_orbital_gradient_response,
    get_orbital_response_hessian_block,
    get_orbital_response_metric_sigma,
    get_orbital_response_property_gradient,
)
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_matrix import (
    expectation_value,
    expectation_value_commutator,
    expectation_value_double_commutator,
)
from slowquant.unitary_coupled_cluster.operators import one_elec_op_0i_0a
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


class LinearResponseUCC(LinearResponseBaseClass):
    def __init__(
        self,
        wave_function: WaveFunctionUCC,
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
        )
        idx_shift = len(self.q_ops)
        print("Gs", len(self.G_ops))
        print("qs", len(self.q_ops))
        grad = get_orbital_gradient_response(
            rdms,
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.kappa_idx,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        if len(grad) != 0:
            print("idx, max(abs(grad orb)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in q of ", np.max(np.abs(grad)))

        grad = np.zeros(2 * len(self.G_ops))
        for i, op in enumerate(self.G_ops):
            grad[i] = expectation_value_commutator(
                self.wf.ci_coeffs,
                self.H_0i_0a,
                op,
                self.wf.ci_coeffs,
                self.wf.idx2det,
                self.wf.det2idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.num_virtual_orbs,
            )
            grad[i + len(self.G_ops)] = expectation_value_commutator(
                self.wf.ci_coeffs,
                op.dagger,
                self.H_0i_0a,
                self.wf.ci_coeffs,
                self.wf.idx2det,
                self.wf.det2idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.num_virtual_orbs,
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad)))
        # Do orbital-orbital blocks
        self.A[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_hessian_block(
            rdms,
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.kappa_idx_dagger,
            self.wf.kappa_idx,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        self.B[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_hessian_block(
            rdms,
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.kappa_idx_dagger,
            self.wf.kappa_idx_dagger,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        self.Sigma[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_metric_sigma(
            rdms, self.wf.kappa_idx
        )
        for j, qJ in enumerate(self.q_ops):
            for i, GI in enumerate(self.G_ops):
                # Make A
                val = expectation_value(
                    self.wf.ci_coeffs,
                    GI.dagger * self.H_1i_1a * qJ,
                    self.wf.ci_coeffs,
                    self.wf.idx2det,
                    self.wf.det2idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.num_virtual_orbs,
                )
                val += -expectation_value(
                    self.wf.ci_coeffs,
                    self.H_1i_1a * qJ * GI.dagger,
                    self.wf.ci_coeffs,
                    self.wf.idx2det,
                    self.wf.det2idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.num_virtual_orbs,
                )
                self.A[i + idx_shift, j] = self.A[j, i + idx_shift] = val
                # Make B
                val = expectation_value(
                    self.wf.ci_coeffs,
                    qJ.dagger * self.H_1i_1a * GI.dagger,
                    self.wf.ci_coeffs,
                    self.wf.idx2det,
                    self.wf.det2idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.num_virtual_orbs,
                )
                val += -expectation_value(
                    self.wf.ci_coeffs,
                    GI.dagger * qJ.dagger * self.H_1i_1a,
                    self.wf.ci_coeffs,
                    self.wf.idx2det,
                    self.wf.det2idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.num_virtual_orbs,
                )
                self.B[i + idx_shift, j] = self.B[j, i + idx_shift] = val
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = (
                    expectation_value_double_commutator(
                        self.wf.ci_coeffs,
                        GI.dagger,
                        self.H_0i_0a,
                        GJ,
                        self.wf.ci_coeffs,
                        self.wf.idx2det,
                        self.wf.det2idx,
                        self.wf.num_inactive_orbs,
                        self.wf.num_active_orbs,
                        self.wf.num_virtual_orbs,
                    )
                )
                # Make B
                self.B[i + idx_shift, j + idx_shift] = self.B[j + idx_shift, i + idx_shift] = (
                    expectation_value_double_commutator(
                        self.wf.ci_coeffs,
                        GI.dagger,
                        self.H_0i_0a,
                        GJ.dagger,
                        self.wf.ci_coeffs,
                        self.wf.idx2det,
                        self.wf.det2idx,
                        self.wf.num_inactive_orbs,
                        self.wf.num_active_orbs,
                        self.wf.num_virtual_orbs,
                    )
                )
                # Make Sigma
                self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[j + idx_shift, i + idx_shift] = (
                    expectation_value_commutator(
                        self.wf.ci_coeffs,
                        GI.dagger,
                        GJ,
                        self.wf.ci_coeffs,
                        self.wf.idx2det,
                        self.wf.det2idx,
                        self.wf.num_inactive_orbs,
                        self.wf.num_active_orbs,
                        self.wf.num_virtual_orbs,
                    )
                )

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
        rdms = ReducedDenstiyMatrix(
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
            self.wf.rdm1,
            rdm2=self.wf.rdm2,
        )
        mux = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[0])
        muy = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[1])
        muz = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[2])
        mux_op = one_elec_op_0i_0a(
            mux,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        muy_op = one_elec_op_0i_0a(
            muy,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        muz_op = one_elec_op_0i_0a(
            muz,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        transition_dipole_x = 0.0
        transition_dipole_y = 0.0
        transition_dipole_z = 0.0
        transition_dipoles = np.zeros((number_excitations, 3))
        for state_number in range(number_excitations):
            transfer_op = FermionicOperator({}, {})
            for i, G in enumerate(self.G_ops):
                transfer_op += (
                    self.Z_G_normed[i, state_number] * G.dagger + self.Y_G_normed[i, state_number] * G
                )
            q_part_x = get_orbital_response_property_gradient(
                rdms,
                mux,
                self.wf.kappa_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.normed_response_vectors,
                state_number,
                number_excitations,
            )
            q_part_y = get_orbital_response_property_gradient(
                rdms,
                muy,
                self.wf.kappa_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.normed_response_vectors,
                state_number,
                number_excitations,
            )
            q_part_z = get_orbital_response_property_gradient(
                rdms,
                muz,
                self.wf.kappa_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.normed_response_vectors,
                state_number,
                number_excitations,
            )
            if mux_op.operators != {}:
                transition_dipole_x = expectation_value_commutator(
                    self.wf.ci_coeffs,
                    mux_op,
                    transfer_op,
                    self.wf.ci_coeffs,
                    self.wf.idx2det,
                    self.wf.det2idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.num_virtual_orbs,
                )
            if muy_op.operators != {}:
                transition_dipole_y = expectation_value_commutator(
                    self.wf.ci_coeffs,
                    muy_op,
                    transfer_op,
                    self.wf.ci_coeffs,
                    self.wf.idx2det,
                    self.wf.det2idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.num_virtual_orbs,
                )
            if muz_op.operators != {}:
                transition_dipole_z = expectation_value_commutator(
                    self.wf.ci_coeffs,
                    muz_op,
                    transfer_op,
                    self.wf.ci_coeffs,
                    self.wf.idx2det,
                    self.wf.det2idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.num_virtual_orbs,
                )
            transition_dipoles[state_number, 0] = q_part_x + transition_dipole_x
            transition_dipoles[state_number, 1] = q_part_y + transition_dipole_y
            transition_dipoles[state_number, 2] = q_part_z + transition_dipole_z
        return transition_dipoles
