import copy
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
    convert_pauli_to_hybrid_form,
    expectation_value_hybrid,
    expectation_value_hybrid_flow_commutator,
    expectation_value_hybrid_flow_double_commutator,
    make_projection_operator,
)
from slowquant.unitary_coupled_cluster.operator_pauli import (
    OperatorPauli,
    energy_hamiltonian_pauli,
    epq_pauli,
    hamiltonian_pauli_1i_1a,
    hamiltonian_pauli_2i_2a,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.util import ThetaPicker, construct_ucc_u


class LinearResponseUCC(LinearResponseBaseClass):
    def __init__(
        self,
        wave_function: WaveFunctionUCC,
        excitations: str,
        operator_type: str,
        do_transform_orbital_rotations: bool = False,
        use_matrix_symmetry: bool = True,
        is_spin_conserving: bool = True,
    ) -> None:
        if operator_type.lower() not in ("naive", "projected", "selfconsistent", "statetransfer"):
            raise ValueError(f"Got unknown operator_type: {operator_type}")
        self.wf = copy.deepcopy(wave_function)
        self.theta_picker = ThetaPicker(
            self.wf.active_occ_spin_idx,
            self.wf.active_unocc_spin_idx,
            is_spin_conserving=is_spin_conserving,
        )

        G_ops_tmp = []
        q_ops_tmp = []
        num_spin_orbs = self.wf.num_spin_orbs
        num_elec = self.wf.num_elec
        excitations = excitations.lower()
        U = construct_ucc_u(
            self.wf.num_active_spin_orbs,
            self.wf.num_active_elec,
            self.wf.theta1
            + self.wf.theta2
            + self.wf.theta3
            + self.wf.theta4
            + self.wf.theta5
            + self.wf.theta6,
            self.wf.singlet_excitation_operator_generator,
            "sdtq56",  # self.wf._excitations,
        )
        if operator_type.lower() in ("projected", "statetransfer"):
            projection = make_projection_operator(self.wf.state_vector)
            self.projection = projection
        if "s" in excitations:
            for _, _, _, op_ in self.theta_picker.get_t1_generator_sa(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                G_ops_tmp.append(op)
        if "d" in excitations:
            for _, _, _, _, _, op_ in self.theta_picker.get_t2_generator_sa(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                G_ops_tmp.append(op)
        if do_transform_orbital_rotations and operator_type.lower() in ("statetransfer", "selfconsistent"):
            valid_kappa_idx = self.wf.kappa_hf_like_idx
        else:
            valid_kappa_idx = self.wf.kappa_idx
        for i, a in valid_kappa_idx:
            op_ = 2 ** (-1 / 2) * epq_pauli(a, i, self.wf.num_spin_orbs, self.wf.num_elec)
            op = convert_pauli_to_hybrid_form(
                op_,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.num_virtual_spin_orbs,
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
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(G - G_diff)
            elif operator_type.lower() == "selfconsistent":
                G = G.apply_u_from_right(U.conj().transpose())
                G = G.apply_u_from_left(U)
                self.G_ops.append(G)
            elif operator_type.lower() == "statetransfer":
                G = G.apply_u_from_right(U.conj().transpose())
                G = G.apply_u_from_left(U)
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
                        self.wf.num_virtual_spin_orbs,
                    )
                    self.q_ops.append(q - q_diff)
                elif operator_type.lower() == "selfconsistent":
                    q = q.apply_u_from_right(U.conj().transpose())
                    q = q.apply_u_from_left(U)
                    self.q_ops.append(q)
                elif operator_type.lower() == "statetransfer":
                    q = q.apply_u_from_right(U.conj().transpose())
                    q = q.apply_u_from_left(U)
                    q = q * projection
                    self.q_ops.append(q)
            else:
                self.q_ops.append(q)

        num_parameters = len(self.G_ops) + len(self.q_ops)
        self.A = np.zeros((num_parameters, num_parameters))
        self.B = np.zeros((num_parameters, num_parameters))
        self.Sigma = np.zeros((num_parameters, num_parameters))
        self.Delta = np.zeros((num_parameters, num_parameters))
        H_1i_1a = convert_pauli_to_hybrid_form(
            hamiltonian_pauli_1i_1a(
                self.wf.h_ao,
                self.wf.g_ao,
                self.wf.c_trans,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.num_virtual_spin_orbs,
                num_elec,
            ),
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
        )
        H_2i_2a = convert_pauli_to_hybrid_form(
            hamiltonian_pauli_2i_2a(
                self.wf.h_ao,
                self.wf.g_ao,
                self.wf.c_trans,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.num_virtual_spin_orbs,
                num_elec,
            ),
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
        )
        H_en = convert_pauli_to_hybrid_form(
            energy_hamiltonian_pauli(
                self.wf.h_ao,
                self.wf.g_ao,
                self.wf.c_trans,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.num_virtual_spin_orbs,
                num_elec,
            ),
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
        )
        idx_shift = len(self.q_ops)
        print("")
        print(f"Number active-space parameters: {len(self.G_ops)}")
        print(f"Number orbital-rotation parameters: {len(self.q_ops)}")
        grad = np.zeros(len(self.q_ops))
        for i, op in enumerate(self.q_ops):
            grad[i] = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, op, H_1i_1a, self.wf.state_vector
            )
        if len(grad) != 0:
            print("idx, max(abs(grad orb)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in q of ", np.max(np.abs(grad)))
        grad = np.zeros(len(self.G_ops))
        for i, op in enumerate(self.G_ops):
            grad[i] = expectation_value_hybrid_flow_commutator(
                self.wf.state_vector, op, H_en, self.wf.state_vector
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad)))

        if use_matrix_symmetry:
            for j, qJ in enumerate(self.q_ops):
                for i, qI in enumerate(self.q_ops):
                    if i < j:
                        continue
                    # Make A
                    self.A[i, j] = self.A[j, i] = expectation_value_hybrid_flow_double_commutator(
                        self.wf.state_vector,
                        qI.dagger,
                        H_2i_2a,
                        qJ,
                        self.wf.state_vector,
                    )
                    # Make B
                    self.B[i, j] = self.B[j, i] = expectation_value_hybrid_flow_double_commutator(
                        self.wf.state_vector,
                        qI.dagger,
                        H_2i_2a,
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
                        H_1i_1a,
                        qI.dagger,
                        self.wf.state_vector,
                    )
                    # Make B
                    self.B[i, j + idx_shift] = self.B[
                        j + idx_shift, i
                    ] = expectation_value_hybrid_flow_double_commutator(
                        self.wf.state_vector,
                        GJ.dagger,
                        H_1i_1a,
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
                for i, GI in enumerate(self.G_ops):
                    if i < j:
                        continue
                    # Make A
                    self.A[i + idx_shift, j + idx_shift] = self.A[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_hybrid_flow_double_commutator(
                        self.wf.state_vector,
                        GI.dagger,
                        H_en,
                        GJ,
                        self.wf.state_vector,
                    )
                    # Make B
                    self.B[i + idx_shift, j + idx_shift] = self.B[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_hybrid_flow_double_commutator(
                        self.wf.state_vector,
                        GI.dagger,
                        H_en,
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
        else:
            for j, qJ in enumerate(self.q_ops):
                for i, qI in enumerate(self.q_ops):
                    # Make A
                    self.A[i, j] = expectation_value_hybrid_flow_double_commutator(
                        self.wf.state_vector,
                        qI.dagger,
                        H_2i_2a,
                        qJ,
                        self.wf.state_vector,
                    )
                    # Make B
                    self.B[i, j] = expectation_value_hybrid_flow_double_commutator(
                        self.wf.state_vector,
                        qI.dagger,
                        H_2i_2a,
                        qJ.dagger,
                        self.wf.state_vector,
                    )
                    # Make Sigma
                    self.Sigma[i, j] = expectation_value_hybrid_flow_commutator(
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
            for j, GJ in enumerate(self.G_ops):
                for i, qI in enumerate(self.q_ops):
                    # Make A
                    self.A[i, j + idx_shift] = expectation_value_hybrid_flow_double_commutator(
                        self.wf.state_vector,
                        GJ,
                        H_1i_1a,
                        qI.dagger,
                        self.wf.state_vector,
                    )
                    # Make B
                    self.B[i, j + idx_shift] = expectation_value_hybrid_flow_double_commutator(
                        self.wf.state_vector,
                        GJ.dagger,
                        H_1i_1a,
                        qI.dagger,
                        self.wf.state_vector,
                    )
                    # Make Sigma
                    self.Sigma[i, j + idx_shift] = expectation_value_hybrid_flow_commutator(
                        self.wf.state_vector, qI.dagger, GJ, self.wf.state_vector
                    )
                    # Make Delta
                    self.Delta[i, j + idx_shift] = expectation_value_hybrid_flow_commutator(
                        self.wf.state_vector, qI.dagger, GJ.dagger, self.wf.state_vector
                    )
            for j, qJ in enumerate(self.q_ops):
                for i, GI in enumerate(self.G_ops):
                    # Make A
                    self.A[i + idx_shift, j] = expectation_value_hybrid_flow_double_commutator(
                        self.wf.state_vector,
                        GI.dagger,
                        H_1i_1a,
                        qJ,
                        self.wf.state_vector,
                    )
                    # Make B
                    self.B[i + idx_shift, j] = expectation_value_hybrid_flow_double_commutator(
                        self.wf.state_vector,
                        GI.dagger,
                        H_1i_1a,
                        qJ.dagger,
                        self.wf.state_vector,
                    )
                    # Make Sigma
                    self.Sigma[i + idx_shift, j] = expectation_value_hybrid_flow_commutator(
                        self.wf.state_vector, GI.dagger, qJ, self.wf.state_vector
                    )
                    # Make Delta
                    self.Delta[i + idx_shift, j] = expectation_value_hybrid_flow_commutator(
                        self.wf.state_vector, GI.dagger, qJ.dagger, self.wf.state_vector
                    )
            for j, GJ in enumerate(self.G_ops):
                for i, GI in enumerate(self.G_ops):
                    # Make A
                    self.A[i + idx_shift, j + idx_shift] = expectation_value_hybrid_flow_double_commutator(
                        self.wf.state_vector,
                        GI.dagger,
                        H_en,
                        GJ,
                        self.wf.state_vector,
                    )
                    # Make B
                    self.B[i + idx_shift, j + idx_shift] = expectation_value_hybrid_flow_double_commutator(
                        self.wf.state_vector,
                        GI.dagger,
                        H_en,
                        GJ.dagger,
                        self.wf.state_vector,
                    )
                    # Make Sigma
                    self.Sigma[i + idx_shift, j + idx_shift] = expectation_value_hybrid_flow_commutator(
                        self.wf.state_vector, GI.dagger, GJ, self.wf.state_vector
                    )
                    # Make Delta
                    self.Delta[i + idx_shift, j + idx_shift] = expectation_value_hybrid_flow_commutator(
                        self.wf.state_vector, GI.dagger, GJ.dagger, self.wf.state_vector
                    )

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
        mux = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[0])
        muy = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[1])
        muz = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[2])
        mux_op = OperatorPauli({})
        muy_op = OperatorPauli({})
        muz_op = OperatorPauli({})
        for p in range(self.wf.num_spin_orbs // 2):
            for q in range(self.wf.num_spin_orbs // 2):
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
        transition_dipoles = np.zeros((len(self.normed_response_vectors[0]), 3))
        for state_number in range(len(self.normed_response_vectors[0])):
            transfer_op = OperatorHybrid({})
            for i, G in enumerate(self.q_ops + self.G_ops):
                transfer_op += (
                    self.normed_response_vectors[i, state_number] * G.dagger
                    + self.normed_response_vectors[i + number_excitations, state_number] * G
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
            transition_dipoles[state_number, 0] = transition_dipole_x
            transition_dipoles[state_number, 1] = transition_dipole_y
            transition_dipoles[state_number, 2] = transition_dipole_z
        return transition_dipoles
