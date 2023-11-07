import copy
from collections.abc import Sequence

import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.density_matrix import (
    ReducedDenstiyMatrix,
    get_orbital_gradient_response,
    get_orbital_response_hessian_block,
    get_orbital_response_metric_sgima,
    get_orbital_response_property_gradient,
    get_orbital_response_vector_norm,
)
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_hybrid import (
    OperatorHybrid,
    convert_pauli_to_hybrid_form,
    expectation_value_hybrid_flow,
    expectation_value_hybrid_flow_commutator,
    expectation_value_hybrid_flow_double_commutator,
)
from slowquant.unitary_coupled_cluster.operator_pauli import OperatorPauli, epq_pauli
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


class LinearResponseUCC(LinearResponseBaseClass):
    def __init__(
        self,
        wave_function: WaveFunctionUCC,
        excitations: str,
        is_spin_conserving: bool = False,
    ) -> None:
        """Initialize linear response by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
            is_spin_conserving: Use spin-conseving operators.
        """
        super().__init__(wave_function, excitations, is_spin_conserving)

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
            grad[i] = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, self.H_1i_1a, op.operator, self.wf.state_vector
            )
            grad[i + len(self.G_ops)] = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, op.operator.dagger, self.H_1i_1a, self.wf.state_vector
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
        self.Sigma[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_metric_sgima(
            rdms, self.wf.kappa_idx
        )
        for j, opJ in enumerate(self.q_ops):
            qJ = opJ.operator
            for i, opI in enumerate(self.G_ops):
                GI = opI.operator
                # Make A
                val = expectation_value_hybrid_flow(
                    self.wf.state_vector, [GI.dagger, self.H_1i_1a, qJ], self.wf.state_vector
                ) - expectation_value_hybrid_flow(
                    self.wf.state_vector, [self.H_1i_1a, qJ, GI.dagger], self.wf.state_vector
                )
                self.A[i + idx_shift, j] = self.A[j, i + idx_shift] = val
                # Make B
                val = expectation_value_hybrid_flow(
                    self.wf.state_vector, [qJ.dagger, self.H_1i_1a, GI.dagger], self.wf.state_vector
                ) - expectation_value_hybrid_flow(
                    self.wf.state_vector, [GI.dagger, qJ.dagger, self.H_1i_1a], self.wf.state_vector
                )
                self.B[i + idx_shift, j] = self.B[j, i + idx_shift] = val
        for j, opJ in enumerate(self.G_ops):
            GJ = opJ.operator
            for i, opI in enumerate(self.G_ops):
                GI = opI.operator
                if i < j:
                    continue
                # Make A
                self.A[i + idx_shift, j + idx_shift] = self.A[
                    j + idx_shift, i + idx_shift
                ] = expectation_value_hybrid_flow_double_commutator(
                    self.wf.state_vector, GI.dagger, self.H_en, GJ, self.wf.state_vector
                )
                # Make B
                self.B[i + idx_shift, j + idx_shift] = self.B[
                    j + idx_shift, i + idx_shift
                ] = expectation_value_hybrid_flow_double_commutator(
                    self.wf.state_vector, GI.dagger, self.H_en, GJ.dagger, self.wf.state_vector
                )
                # Make Sigma
                self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[
                    j + idx_shift, i + idx_shift
                ] = expectation_value_hybrid_flow_commutator(
                    self.wf.state_vector, GI.dagger, GJ, self.wf.state_vector
                )

    def get_excited_state_norm(self) -> np.ndarray:
        """Calculate the norm of excited states.

        Returns:
            Norm of excited states.
        """
        rdms = ReducedDenstiyMatrix(
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
            self.wf.rdm1,
            rdm2=self.wf.rdm2,
        )
        norms = np.zeros(len(self.Z_G[0]))
        for state_number in range(len(self.Z_G[0])):
            q_part = get_orbital_response_vector_norm(
                rdms, self.wf.kappa_idx, self.response_vectors, state_number, len(self.excitation_energies)
            )
            transfer_op = OperatorHybrid({})
            for i, op in enumerate(self.G_ops):
                G = op.operator
                transfer_op += self.Z_G[i, state_number] * G.dagger + self.Y_G[i, state_number] * G
            norms[state_number] = q_part + expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, transfer_op, transfer_op.dagger, self.wf.state_vector
            )
        return norms

    def get_property_gradient(self, dipole_integrals: Sequence[np.ndarray]) -> np.ndarray:
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
        mux_op = OperatorPauli({})
        muy_op = OperatorPauli({})
        muz_op = OperatorPauli({})
        for p in range(self.wf.num_orbs):
            for q in range(self.wf.num_orbs):
                Epq_op = epq_pauli(p, q, self.wf.num_spin_orbs, self.wf.num_elec)
                if abs(mux[p, q]) > 10**-10:
                    mux_op += mux[p, q] * Epq_op
                if abs(muy[p, q]) > 10**-10:
                    muy_op += muy[p, q] * Epq_op
                if abs(muz[p, q]) > 10**-10:
                    muz_op += muz[p, q] * Epq_op
        mux_op = convert_pauli_to_hybrid_form(
            mux_op,
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
        )
        muy_op = convert_pauli_to_hybrid_form(
            muy_op,
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
        )
        muz_op = convert_pauli_to_hybrid_form(
            muz_op,
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
        )
        transition_dipole_x = 0.0
        transition_dipole_y = 0.0
        transition_dipole_z = 0.0
        transition_dipoles = np.zeros((len(self.Z_G_normed[0]), 3))
        for state_number in range(len(self.Z_G_normed[0])):
            transfer_op = OperatorHybrid({})
            for i, op in enumerate(self.G_ops):
                G = op.operator
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
                transition_dipole_x = expectation_value_hybrid_flow_commutator(
                    self.wf.state_vector, mux_op, transfer_op, self.wf.state_vector
                )
            if muy_op.operators != {}:
                transition_dipole_y = expectation_value_hybrid_flow_commutator(
                    self.wf.state_vector, muy_op, transfer_op, self.wf.state_vector
                )
            if muz_op.operators != {}:
                transition_dipole_z = expectation_value_hybrid_flow_commutator(
                    self.wf.state_vector, muz_op, transfer_op, self.wf.state_vector
                )
            transition_dipoles[state_number, 0] = q_part_x + transition_dipole_x
            transition_dipoles[state_number, 1] = q_part_y + transition_dipole_y
            transition_dipoles[state_number, 2] = q_part_z + transition_dipole_z
        return transition_dipoles
