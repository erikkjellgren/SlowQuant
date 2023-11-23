from collections.abc import Sequence

import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_hybrid import (
    OperatorHybrid,
    OperatorHybridData,
    convert_pauli_to_hybrid_form,
    expectation_value_hybrid,
    expectation_value_hybrid_flow_commutator,
    expectation_value_hybrid_flow_double_commutator,
    hamiltonian_hybrid_2i_2a,
    make_projection_operator,
    one_elec_op_hybrid_1i_1a,
)
from slowquant.unitary_coupled_cluster.operator_pauli import OperatorPauli, epq_pauli
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.util import ThetaPicker


class LinearResponseUCC(LinearResponseBaseClass):
    def __init__(
        self,
        wave_function: WaveFunctionUCC,
        excitations: str,
        operator_type: str,
        do_transform_orbital_rotations: bool = False,
        is_spin_conserving: bool = True,
    ) -> None:
        super().__init__(wave_function, excitations, is_spin_conserving)

        if operator_type.lower() not in ("naive", "projected", "selfconsistent", "statetransfer"):
            raise ValueError(f"Got unknown operator_type: {operator_type}")
        self.wf = wave_function
        self.theta_picker = ThetaPicker(
            self.wf.active_occ_spin_idx,
            self.wf.active_unocc_spin_idx,
            is_spin_conserving=is_spin_conserving,
        )

        if operator_type.lower() in ("projected", "statetransfer"):
            projection = make_projection_operator(self.wf.state_vector)
            self.projection = projection
        if operator_type.lower() in ("selfconsistent", "statetransfer"):
            inactive_str = "I" * self.wf.num_inactive_spin_orbs
            virtual_str = "I" * self.wf.num_virtual_spin_orbs
            U = OperatorHybrid(
                {inactive_str + virtual_str: OperatorHybridData(inactive_str, self.wf.u, virtual_str)}
            )
        if do_transform_orbital_rotations and operator_type.lower() in ("statetransfer", "selfconsistent"):
            valid_kappa_idx = self.wf.kappa_hf_like_idx
        else:
            valid_kappa_idx = self.wf.kappa_idx
        G_ops_tmp = self.G_ops.copy()
        q_ops_tmp = []
        for i, a in valid_kappa_idx:
            op_ = 2 ** (-1 / 2) * epq_pauli(a, i, self.wf.num_spin_orbs)
            op = convert_pauli_to_hybrid_form(
                op_,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
            )
            q_ops_tmp.append(op)
        self.G_ops = []
        self.q_ops = []
        for G in G_ops_tmp:
            if operator_type.lower() == "naive":
                self.G_ops.append(G)
            elif operator_type.lower() == "projected":
                G = G * projection
                fac = expectation_value_hybrid(self.wf.state_vector, G, self.wf.state_vector)
                G_diff_ = OperatorPauli({"I" * self.wf.num_spin_orbs: fac})
                G_diff = convert_pauli_to_hybrid_form(
                    G_diff_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                )
                self.G_ops.append(G - G_diff)
            elif operator_type.lower() == "selfconsistent":
                G = U * G * U.dagger
                self.G_ops.append(G)
            elif operator_type.lower() == "statetransfer":
                G = U * G * U.dagger
                G = G * projection
                self.G_ops.append(G)
        for q in q_ops_tmp:
            if do_transform_orbital_rotations:
                if operator_type.lower() == "naive":
                    self.q_ops.append(q)
                if operator_type.lower() == "projected":
                    q = q * projection
                    fac = expectation_value_hybrid(self.wf.state_vector, q, self.wf.state_vector)
                    q_diff_ = OperatorPauli({"I" * self.wf.num_spin_orbs: fac})
                    q_diff = convert_pauli_to_hybrid_form(
                        q_diff_,
                        self.wf.num_inactive_spin_orbs,
                        self.wf.num_active_spin_orbs,
                    )
                    self.q_ops.append(q - q_diff)
                elif operator_type.lower() == "selfconsistent":
                    q = U * q * U.dagger
                    self.q_ops.append(q)
                elif operator_type.lower() == "statetransfer":
                    q = U * q * U.dagger
                    q = q * projection
                    self.q_ops.append(q)
            else:
                self.q_ops.append(q)

        self.H_2i_2a = hamiltonian_hybrid_2i_2a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
        )
        idx_shift = len(self.q_ops)
        print("")
        print(f"Number active-space parameters: {len(self.G_ops)}")
        print(f"Number orbital-rotation parameters: {len(self.q_ops)}")
        grad = np.zeros(2 * len(self.q_ops))
        for i, op in enumerate(self.q_ops):
            grad[i] = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, op, self.H_1i_1a, self.wf.state_vector
            )
            grad[i + len(self.q_ops)] = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, op.dagger, self.H_1i_1a, self.wf.state_vector
            )
        if len(grad) != 0:
            print("idx, max(abs(grad orb)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in q of ", np.max(np.abs(grad)))
        grad = np.zeros(2 * len(self.G_ops))
        for i, op in enumerate(self.G_ops):
            grad[i] = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, op, self.H_0i_0a, self.wf.state_vector
            )
            grad[i + len(self.G_ops)] = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, op.dagger, self.H_0i_0a, self.wf.state_vector
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad)))

        for j, qJ in enumerate(self.q_ops):
            for i, qI in enumerate(self.q_ops[j:], j):
                # Make A
                self.A[i, j] = self.A[j, i] = expectation_value_hybrid_flow_double_commutator(
                    self.wf.state_vector,
                    qI.dagger,
                    self.H_2i_2a,
                    qJ,
                    self.wf.state_vector,
                )
                # Make B
                self.B[i, j] = self.B[j, i] = expectation_value_hybrid_flow_double_commutator(
                    self.wf.state_vector,
                    qI.dagger,
                    self.H_2i_2a,
                    qJ.dagger,
                    self.wf.state_vector,
                )
                # Make Sigma
                self.Sigma[i, j] = self.Sigma[j, i] = expectation_value_hybrid_flow_commutator(
                    self.wf.state_vector,
                    qI.dagger,
                    qJ,
                    self.wf.state_vector,
                )
                # Make Delta
                self.Delta[i, j] = expectation_value_hybrid_flow_commutator(
                    self.wf.state_vector,
                    qI.dagger,
                    qJ.dagger,
                    self.wf.state_vector,
                )
                self.Delta[j, i] = -self.Delta[i, j]
        for j, GJ in enumerate(self.G_ops):
            for i, qI in enumerate(self.q_ops):
                # Make A
                self.A[i, j + idx_shift] = self.A[
                    j + idx_shift, i
                ] = expectation_value_hybrid_flow_double_commutator(
                    self.wf.state_vector,
                    GJ,
                    self.H_1i_1a,
                    qI.dagger,
                    self.wf.state_vector,
                )
                # Make B
                self.B[i, j + idx_shift] = self.B[
                    j + idx_shift, i
                ] = expectation_value_hybrid_flow_double_commutator(
                    self.wf.state_vector,
                    GJ.dagger,
                    self.H_1i_1a,
                    qI.dagger,
                    self.wf.state_vector,
                )
                # Make Sigma
                self.Sigma[i, j + idx_shift] = self.Sigma[
                    j + idx_shift, i
                ] = expectation_value_hybrid_flow_commutator(
                    self.wf.state_vector, qI.dagger, GJ, self.wf.state_vector
                )
                # Make Delta
                self.Delta[i, j + idx_shift] = expectation_value_hybrid_flow_commutator(
                    self.wf.state_vector, qI.dagger, GJ.dagger, self.wf.state_vector
                )
                self.Delta[j + idx_shift, i] = -self.Delta[i, j + idx_shift]
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                self.A[i + idx_shift, j + idx_shift] = self.A[
                    j + idx_shift, i + idx_shift
                ] = expectation_value_hybrid_flow_double_commutator(
                    self.wf.state_vector,
                    GI.dagger,
                    self.H_0i_0a,
                    GJ,
                    self.wf.state_vector,
                )
                # Make B
                self.B[i + idx_shift, j + idx_shift] = self.B[
                    j + idx_shift, i + idx_shift
                ] = expectation_value_hybrid_flow_double_commutator(
                    self.wf.state_vector,
                    GI.dagger,
                    self.H_0i_0a,
                    GJ.dagger,
                    self.wf.state_vector,
                )
                # Make Sigma
                self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[
                    j + idx_shift, i + idx_shift
                ] = expectation_value_hybrid_flow_commutator(
                    self.wf.state_vector, GI.dagger, GJ, self.wf.state_vector
                )
                # Make Delta
                self.Delta[i + idx_shift, j + idx_shift] = expectation_value_hybrid_flow_commutator(
                    self.wf.state_vector, GI.dagger, GJ.dagger, self.wf.state_vector
                )
                self.Delta[j + idx_shift, i + idx_shift] = -self.Delta[i + idx_shift, j + idx_shift]

    def get_excited_state_norm(self) -> np.ndarray:
        """Calculate the norm of excited state.

        Returns:
            Norm of excited state.
        """
        number_excitations = len(self.excitation_energies)
        norms = np.zeros(len(self.response_vectors[0]))
        for state_number in range(len(self.response_vectors[0])):
            transfer_op = OperatorHybrid({})
            for i, G in enumerate(self.q_ops + self.G_ops):
                transfer_op += (
                    self.response_vectors[i, state_number] * G.dagger
                    + self.response_vectors[i + number_excitations, state_number] * G
                )
            norms[state_number] = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, transfer_op, transfer_op.dagger, self.wf.state_vector
            )
        return norms

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
        mux = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[0])
        muy = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[1])
        muz = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[2])
        mux_op = one_elec_op_hybrid_1i_1a(
            mux, self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.wf.num_virtual_orbs
        )
        muy_op = one_elec_op_hybrid_1i_1a(
            muy, self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.wf.num_virtual_orbs
        )
        muz_op = one_elec_op_hybrid_1i_1a(
            muz, self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.wf.num_virtual_orbs
        )
        transition_dipole_x = 0.0
        transition_dipole_y = 0.0
        transition_dipole_z = 0.0
        transition_dipoles = np.zeros((len(self.normed_response_vectors[0]), 3))
        for state_number in range(len(self.normed_response_vectors[0])):
            transfer_op = OperatorHybrid({})
            for i, G in enumerate(self.q_ops + self.G_ops):
                transfer_op += (
                    self.normed_response_vectors[i, state_number] * G.dagger
                    + self.normed_response_vectors[i + number_excitations, state_number] * G
                )
            transition_dipole_x = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, mux_op, transfer_op, self.wf.state_vector
            )
            transition_dipole_y = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, muy_op, transfer_op, self.wf.state_vector
            )
            transition_dipole_z = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, muz_op, transfer_op, self.wf.state_vector
            )
            transition_dipoles[state_number, 0] = transition_dipole_x
            transition_dipoles[state_number, 1] = transition_dipole_y
            transition_dipoles[state_number, 2] = transition_dipole_z
        return transition_dipoles
