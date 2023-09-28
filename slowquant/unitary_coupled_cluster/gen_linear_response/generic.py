import copy
from collections.abc import Sequence

import numpy as np
import scipy

import slowquant.unitary_coupled_cluster.linalg_wrapper as lw
from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.operator_contracted import (
    commutator_contract,
    double_commutator_contract,
    expectation_value_contracted,
)
from slowquant.unitary_coupled_cluster.operator_hybrid import (
    convert_pauli_to_hybrid_form,
    expectation_value_hybrid,
)
from slowquant.unitary_coupled_cluster.operator_pauli import (
    OperatorPauli,
    energy_hamiltonian_pauli,
    epq_pauli,
    hamiltonian_pauli,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.util import ThetaPicker, construct_ucc_u


class LinearResponseUCC:
    def __init__(
        self,
        wave_function: WaveFunctionUCC,
        excitations: str,
        operator_type: str,
        do_transform_orbital_rotations: bool = False,
        use_matrix_symmetry: bool = True,
        is_spin_conserving: bool = True,
    ) -> None:
        if operator_type.lower() not in ('naive', 'projected', 'selfconsistent', 'statetransfer'):
            raise ValueError(f'Got unknown operator_type: {operator_type}')
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
            self.wf.theta_picker_full,
            'sdtq56',  # self.wf._excitations,
        )
        if operator_type.lower() in ('projected', 'statetransfer'):
            if self.wf.num_active_spin_orbs >= 10:
                projection = lw.outer(
                    self.wf.state_vector.ket_active_csr, self.wf.state_vector.bra_active_csr
                )
            else:
                projection = lw.outer(self.wf.state_vector.ket_active, self.wf.state_vector.bra_active)
        if 's' in excitations:
            for _, _, _, op_ in self.theta_picker.get_t1_generator_sa(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                G_ops_tmp.append(op)
        if 'd' in excitations:
            for _, _, _, _, _, op_ in self.theta_picker.get_t2_generator_sa(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                G_ops_tmp.append(op)
        if do_transform_orbital_rotations and operator_type.lower() in ('statetransfer', 'selfconsistent'):
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
            if operator_type.lower() == 'naive':
                self.G_ops.append(G)
            elif operator_type.lower() == 'projected':
                G = G.apply_u_from_right(projection)
                fac = expectation_value_hybrid(self.wf.state_vector, G, self.wf.state_vector)
                G_diff_ = OperatorPauli({'I' * self.wf.num_spin_orbs: fac})
                G_diff = convert_pauli_to_hybrid_form(
                    G_diff_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(G - G_diff)
            elif operator_type.lower() == 'selfconsistent':
                G = G.apply_u_from_right(U.conj().transpose())
                G = G.apply_u_from_left(U)
                self.G_ops.append(G)
            elif operator_type.lower() == 'statetransfer':
                G = G.apply_u_from_right(U.conj().transpose())
                G = G.apply_u_from_left(U)
                G = G.apply_u_from_right(projection)
                self.G_ops.append(G)
        for q in q_ops_tmp:
            if do_transform_orbital_rotations:
                if operator_type.lower() == 'naive':
                    self.q_ops.append(q)
                if operator_type.lower() == 'projected':
                    q = q.apply_u_from_right(projection)
                    fac = expectation_value_hybrid(self.wf.state_vector, q, self.wf.state_vector)
                    q_diff_ = OperatorPauli({'I' * self.wf.num_spin_orbs: fac})
                    q_diff = convert_pauli_to_hybrid_form(
                        q_diff_,
                        self.wf.num_inactive_spin_orbs,
                        self.wf.num_active_spin_orbs,
                        self.wf.num_virtual_spin_orbs,
                    )
                    self.q_ops.append(q - q_diff)
                elif operator_type.lower() == 'selfconsistent':
                    q = q.apply_u_from_right(U.conj().transpose())
                    q = q.apply_u_from_left(U)
                    self.q_ops.append(q)
                elif operator_type.lower() == 'statetransfer':
                    q = q.apply_u_from_right(U.conj().transpose())
                    q = q.apply_u_from_left(U)
                    q = q.apply_u_from_right(projection)
                    self.q_ops.append(q)
            else:
                self.q_ops.append(q)

        num_parameters = len(self.G_ops) + len(self.q_ops)
        self.A = np.zeros((num_parameters, num_parameters))
        self.B = np.zeros((num_parameters, num_parameters))
        self.Sigma = np.zeros((num_parameters, num_parameters))
        self.Delta = np.zeros((num_parameters, num_parameters))
        H_pauli = hamiltonian_pauli(self.wf.h_core, self.wf.g_eri, self.wf.c_trans, num_spin_orbs, num_elec)
        H_1i_1a = convert_pauli_to_hybrid_form(
            H_pauli.screen_terms(1, 1, self.wf.num_inactive_spin_orbs, self.wf.num_virtual_spin_orbs),
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
        )
        H_2i_2a = convert_pauli_to_hybrid_form(
            H_pauli.screen_terms(2, 2, self.wf.num_inactive_spin_orbs, self.wf.num_virtual_spin_orbs),
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
        )
        H_en = convert_pauli_to_hybrid_form(
            energy_hamiltonian_pauli(
                self.wf.h_core,
                self.wf.g_eri,
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
        print('')
        print(f'Number active-space parameters: {len(self.G_ops)}')
        print(f'Number orbital-rotation parameters: {len(self.q_ops)}')
        grad = np.zeros(len(self.q_ops))
        for i, op in enumerate(self.q_ops):
            grad[i] = expectation_value_contracted(
                self.wf.state_vector, commutator_contract(op, H_1i_1a), self.wf.state_vector
            )
        if len(grad) != 0:
            print('idx, max(abs(grad orb)):', np.argmax(np.abs(grad)), np.max(np.abs(grad)))
        grad = np.zeros(len(self.G_ops))
        for i, op in enumerate(self.G_ops):
            grad[i] = expectation_value_contracted(
                self.wf.state_vector, commutator_contract(op, H_en), self.wf.state_vector
            )
        if len(grad) != 0:
            print('idx, max(abs(grad active)):', np.argmax(np.abs(grad)), np.max(np.abs(grad)))

        if use_matrix_symmetry:
            for j, qJ in enumerate(self.q_ops):
                for i, qI in enumerate(self.q_ops):
                    if i < j:
                        continue
                    # Make A
                    self.A[i, j] = self.A[j, i] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(qI.dagger, H_2i_2a, qJ),
                        self.wf.state_vector,
                    )
                    # Make B
                    self.B[i, j] = self.B[j, i] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(qI.dagger, H_2i_2a, qJ.dagger),
                        self.wf.state_vector,
                    )
                    # Make Sigma
                    self.Sigma[i, j] = self.Sigma[j, i] = expectation_value_contracted(
                        self.wf.state_vector,
                        commutator_contract(qI.dagger, qJ),
                        self.wf.state_vector,
                    )
                    # Make Delta
                    self.Delta[i, j] = expectation_value_contracted(
                        self.wf.state_vector,
                        commutator_contract(qI.dagger, qJ.dagger),
                        self.wf.state_vector,
                    )
                    self.Delta[j, i] = -self.Delta[i, j]
            # This one has been changed to reflect literature parametrization
            # If one would change back to the equations in the comments, one would obtain the results for an initial parametrization opposite literature, i.e. exp(s)exp(kappa)
            for j, GJ in enumerate(self.G_ops):
                for i, qI in enumerate(self.q_ops):
                    # Make A
                    self.A[i, j + idx_shift] = self.A[j + idx_shift, i] = expectation_value_contracted(
                        self.wf.state_vector,
                        # double_commutator_contract(qI.dagger, H_1i_1a, GJ),
                        double_commutator_contract(GJ, H_1i_1a, qI.dagger),
                        self.wf.state_vector,
                    )
                    # Make B
                    self.B[i, j + idx_shift] = self.B[j + idx_shift, i] = expectation_value_contracted(
                        self.wf.state_vector,
                        # double_commutator_contract(qI.dagger, H_1i_1a, GJ.dagger), #wrong parametrization
                        double_commutator_contract(GJ.dagger, H_1i_1a, qI.dagger),
                        self.wf.state_vector,
                    )
                    # Make Sigma
                    self.Sigma[i, j + idx_shift] = self.Sigma[
                        j + idx_shift, i
                    ] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(qI.dagger, GJ), self.wf.state_vector
                    )
                    # Make Delta
                    self.Delta[i, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(qI.dagger, GJ.dagger), self.wf.state_vector
                    )
                    self.Delta[j + idx_shift, i] = -self.Delta[i, j + idx_shift]
            for j, GJ in enumerate(self.G_ops):
                for i, GI in enumerate(self.G_ops):
                    if i < j:
                        continue
                    # Make A
                    self.A[i + idx_shift, j + idx_shift] = self.A[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(GI.dagger, H_en, GJ),
                        self.wf.state_vector,
                    )
                    # Make B
                    self.B[i + idx_shift, j + idx_shift] = self.B[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(GI.dagger, H_en, GJ.dagger),
                        self.wf.state_vector,
                    )
                    # Make Sigma
                    self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(GI.dagger, GJ), self.wf.state_vector
                    )
                    # Make Delta
                    self.Delta[i + idx_shift, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(GI.dagger, GJ.dagger), self.wf.state_vector
                    )
                    self.Delta[j + idx_shift, i + idx_shift] = -self.Delta[i + idx_shift, j + idx_shift]
        else:
            for j, qJ in enumerate(self.q_ops):
                for i, qI in enumerate(self.q_ops):
                    # Make A
                    self.A[i, j] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(qI.dagger, H_2i_2a, qJ),
                        self.wf.state_vector,
                    )
                    # Make B
                    self.B[i, j] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(qI.dagger, H_2i_2a, qJ.dagger),
                        self.wf.state_vector,
                    )
                    # Make Sigma
                    self.Sigma[i, j] = expectation_value_contracted(
                        self.wf.state_vector,
                        commutator_contract(qI.dagger, qJ),
                        self.wf.state_vector,
                    )
                    # Make Delta
                    self.Delta[i, j] = expectation_value_contracted(
                        self.wf.state_vector,
                        commutator_contract(qI.dagger, qJ.dagger),
                        self.wf.state_vector,
                    )
            # This one has been changed to reflect literature parametrization.
            # If one would change back to the equations in the comments, one would obtain the results for an initial parametrization opposite literature, i.e. exp(s)exp(kappa)
            for j, GJ in enumerate(self.G_ops):
                for i, qI in enumerate(self.q_ops):
                    # Make A
                    self.A[i, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector,
                        # double_commutator_contract(qI.dagger, H_1i_1a, GJ),
                        double_commutator_contract(GJ, H_1i_1a, qI.dagger),
                        self.wf.state_vector,
                    )
                    # Make B
                    self.B[i, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector,
                        # double_commutator_contract(qI.dagger, H_1i_1a, GJ.dagger),
                        double_commutator_contract(GJ.dagger, H_1i_1a, qI.dagger),
                        self.wf.state_vector,
                    )
                    # Make Sigma
                    self.Sigma[i, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(qI.dagger, GJ), self.wf.state_vector
                    )
                    # Make Delta
                    self.Delta[i, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(qI.dagger, GJ.dagger), self.wf.state_vector
                    )
            for j, qJ in enumerate(self.q_ops):
                for i, GI in enumerate(self.G_ops):
                    # Make A
                    self.A[i + idx_shift, j] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(GI.dagger, H_1i_1a, qJ),
                        self.wf.state_vector,
                    )
                    # Make B
                    self.B[i + idx_shift, j] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(GI.dagger, H_1i_1a, qJ.dagger),
                        self.wf.state_vector,
                    )
                    # Make Sigma
                    self.Sigma[i + idx_shift, j] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(GI.dagger, qJ), self.wf.state_vector
                    )
                    # Make Delta
                    self.Delta[i + idx_shift, j] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(GI.dagger, qJ.dagger), self.wf.state_vector
                    )
            for j, GJ in enumerate(self.G_ops):
                for i, GI in enumerate(self.G_ops):
                    # Make A
                    self.A[i + idx_shift, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(GI.dagger, H_en, GJ),
                        self.wf.state_vector,
                    )
                    # Make B
                    self.B[i + idx_shift, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(GI.dagger, H_en, GJ.dagger),
                        self.wf.state_vector,
                    )
                    # Make Sigma
                    self.Sigma[i + idx_shift, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(GI.dagger, GJ), self.wf.state_vector
                    )
                    # Make Delta
                    self.Delta[i + idx_shift, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(GI.dagger, GJ.dagger), self.wf.state_vector
                    )

    def calc_excitation_energies(self) -> None:
        """Calculate excitation energies."""
        size = len(self.A)
        E2 = np.zeros((size * 2, size * 2))
        E2[:size, :size] = self.A
        E2[:size, size:] = self.B
        E2[size:, :size] = self.B
        E2[size:, size:] = self.A
        (
            hess_eigval,
            _,
        ) = np.linalg.eig(E2)
        print(f'Smallest Hessian eigenvalue: {np.min(hess_eigval)}')

        S = np.zeros((size * 2, size * 2))
        S[:size, :size] = self.Sigma
        S[:size, size:] = self.Delta
        S[size:, :size] = -self.Delta
        S[size:, size:] = -self.Sigma
        print(f'Smallest diagonal element in the metric: {np.min(np.abs(np.diagonal(self.Sigma)))}')

        eigval, eigvec = scipy.linalg.eig(E2, S)
        sorting = np.argsort(eigval)
        self.excitation_energies = np.real(eigval[sorting][size:])
        self.response_vectors = np.real(eigvec[:, sorting][:, size:])
        self.normed_response_vectors = np.zeros_like(self.response_vectors)
        for state_number in range(size):
            norm = self.get_excited_state_norm(state_number)
            if norm < 10**-10:
                continue
            self.normed_response_vectors[:, state_number] = (
                self.response_vectors[:, state_number] * (1 / norm) ** 0.5
            )

    def analyse_response_vector(self, state_number: int, threshold: float = 0.05) -> str:
        """Analyse the response vector.

        Args:
            state_number: Excitation index counting from zero.
            threshold: Only print response vector elements that are larger than threshold*max(response vector element).

        Returns:
            Tabulized excitation assignment.
        """
        output = f'Response vector analysis for excitation {state_number+1}\n'
        output += (
            'Occupied idxs | Unoccupied idxs | Response vector element | Normalized response vector element\n'
        )
        excitations = len(self.q_ops) + len(self.G_ops)
        skip_threshold = threshold * np.max(np.abs(self.response_vectors[:, state_number]))
        for resp_val, normed_resp_val, operator in zip(
            self.response_vectors[:excitations, state_number],
            self.normed_response_vectors[:excitations, state_number],
            self.q_ops + self.G_ops,
        ):
            if abs(resp_val) < skip_threshold:
                continue
            resp_val_str = f'{resp_val:1.6f}'
            normed_resp_val_str = f'{normed_resp_val:1.6f}'
            output += f'{str(operator.occ_idx).center(13)} | {str(operator.unocc_idx).center(15)} | {resp_val_str.center(23)} | {normed_resp_val_str.center(34)}\n'
        for resp_val, normed_resp_val, operator in zip(
            self.response_vectors[excitations:, state_number],
            self.normed_response_vectors[excitations:, state_number],
            self.q_ops + self.G_ops,
        ):
            if abs(resp_val) < skip_threshold:
                continue
            resp_val_str = f'{resp_val:1.6f}'
            normed_resp_val_str = f'{normed_resp_val:1.6f}'
            output += f'{str(operator.unocc_idx).center(13)} | {str(operator.occ_idx).center(15)} | {resp_val_str.center(23)} | {normed_resp_val_str.center(34)}\n'
        return output

    def get_excited_state_norm(self, state_number: int) -> float:
        """Calculate the norm of excited state.

        Args:
            state_number: Which excited state, counting from zero.

        Returns:
            Norm of excited state.
        """
        number_excitations = len(self.excitation_energies)
        for i, op in enumerate(self.q_ops + self.G_ops):
            G = op
            if i == 0:
                transfer_op = (
                    self.response_vectors[i, state_number] * G.dagger
                    + self.response_vectors[i + number_excitations, state_number] * G
                )
            else:
                transfer_op += (
                    self.response_vectors[i, state_number] * G.dagger
                    + self.response_vectors[i + number_excitations, state_number] * G
                )
        return expectation_value_contracted(
            self.wf.state_vector, commutator_contract(transfer_op, transfer_op.dagger), self.wf.state_vector
        )

    def get_transition_dipole(
        self, state_number: int, dipole_integrals: Sequence[np.ndarray]
    ) -> tuple[float, float, float]:
        """Calculate transition dipole moment.

        Args:
            state_number: Which excited state, counting from zero.
            dipole_integrals: Dipole integrals ordered as (x,y,z).

        Returns:
            Transition dipole moment.
        """
        if len(dipole_integrals) != 3:
            raise ValueError(f'Expected 3 dipole integrals got {len(dipole_integrals)}')
        number_excitations = len(self.excitation_energies)
        for i, op in enumerate(self.q_ops + self.G_ops):
            G = op
            if i == 0:
                transfer_op = (
                    self.normed_response_vectors[i, state_number] * G.dagger
                    + self.normed_response_vectors[i + number_excitations, state_number] * G
                )
            else:
                transfer_op += (
                    self.normed_response_vectors[i, state_number] * G.dagger
                    + self.normed_response_vectors[i + number_excitations, state_number] * G
                )
        mux = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[0])
        muy = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[1])
        muz = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[2])
        counter = 0
        for p in range(self.wf.num_spin_orbs // 2):
            for q in range(self.wf.num_spin_orbs // 2):
                Epq_op = epq_pauli(p, q, self.wf.num_spin_orbs, self.wf.num_elec)
                if counter == 0:
                    mux_op = mux[p, q] * Epq_op
                    muy_op = muy[p, q] * Epq_op
                    muz_op = muz[p, q] * Epq_op
                    counter += 1
                else:
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
        if mux_op.operators != {}:
            transition_dipole_x = expectation_value_contracted(
                self.wf.state_vector, commutator_contract(mux_op, transfer_op), self.wf.state_vector
            )
        if muy_op.operators != {}:
            transition_dipole_y = expectation_value_contracted(
                self.wf.state_vector, commutator_contract(muy_op, transfer_op), self.wf.state_vector
            )
        if muz_op.operators != {}:
            transition_dipole_z = expectation_value_contracted(
                self.wf.state_vector, commutator_contract(muz_op, transfer_op), self.wf.state_vector
            )
        return transition_dipole_x, transition_dipole_y, transition_dipole_z

    def get_oscillator_strength(self, state_number: int, dipole_integrals: Sequence[np.ndarray]) -> float:
        r"""Calculate oscillator strength.

        .. math::
            f_n = \frac{2}{3}e_n\left|\left<0\left|\hat{\mu}\right|n\left>\right|^2

        Args:
            state_number: Target excited state (zero being the first excited state).
            dipole_integrals: Dipole integrals (x,y,z) in AO basis.

        Rerturns:
            Oscillator Strength.
        """
        transition_dipole_x, transition_dipole_y, transition_dipole_z = self.get_transition_dipole(
            state_number, dipole_integrals
        )
        excitation_energy = self.excitation_energies[state_number]
        return (
            2
            / 3
            * excitation_energy
            * (transition_dipole_x**2 + transition_dipole_y**2 + transition_dipole_z**2)
        )

    def get_nice_output(self, dipole_integrals: Sequence[np.ndarray]) -> str:
        """Create table of excitation energies and oscillator strengths.

        Args:
            dipole_integrals: Dipole integrals (x,y,z) in AO basis.

        Returns:
            Nicely formatted table.
        """
        output = (
            'Excitation # | Excitation energy [Hartree] | Excitation energy [eV] | Oscillator strengths\n'
        )
        for i, exc_energy in enumerate(self.excitation_energies):
            osc_strength = self.get_oscillator_strength(i, dipole_integrals)
            exc_str = f'{exc_energy:2.6f}'
            exc_str_ev = f'{exc_energy*27.2114079527:3.6f}'
            osc_str = f'{osc_strength:1.6f}'
            output += f'{str(i+1).center(12)} | {exc_str.center(27)} | {exc_str_ev.center(22)} | {osc_str.center(20)}\n'
        return output
