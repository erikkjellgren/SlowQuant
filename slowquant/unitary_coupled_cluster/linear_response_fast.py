import copy
from collections.abc import Sequence

import numpy as np
import scipy
import scipy.sparse as ss

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
    OperatorHybrid,
    OperatorHybridData,
    convert_pauli_to_hybrid_form,
    expectation_value_hybrid,
    expectation_value_hybrid_flow,
    make_projection_operator,
)
from slowquant.unitary_coupled_cluster.operator_pauli import (
    OperatorPauli,
    energy_hamiltonian_pauli,
    epq_pauli,
    hamiltonian_pauli,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.util import ThetaPicker, construct_ucc_u


class ResponseOperator:
    def __init__(self, occ_idx: Sequence[int], unocc_idx: Sequence[int], operator: OperatorHybrid) -> None:
        """Initialize response excitation operator.

        Args:
            occ_idx: Index of occupied orbitals.
            unocc_idx: Index of unoccupied orbitals.
            operator: Operator.
        """
        self.occ_idx = occ_idx
        self.unocc_idx = unocc_idx
        self.operator = operator


class LinearResponseUCC:
    def __init__(
        self,
        wave_function: WaveFunctionUCC,
        excitations: str,
        is_spin_conserving: bool = False,
        do_selfconsistent_operators: bool = True,
        do_projected_operators: bool = False,
        do_debugging: bool = False,
        do_statetransfer_operators: bool = False,
        do_hermitian_statetransfer_operators: bool = False,
        track_hermitian_statetransfer: bool = False,
    ) -> None:
        """Initialize linear response by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
            is_spin_conserving: Use spin-conseving operators.
            do_selfconsistent_operators: Use self-consistent active space excitation operators.
            do_projected_operators: Use projected active space excitation opreators.
        """
        self.do_debugging = do_debugging
        if sum([do_projected_operators, do_selfconsistent_operators, do_statetransfer_operators]) >= 2:
            raise ValueError('You set more than one method flag to True.')
        if self.do_debugging:
            if do_hermitian_statetransfer_operators:
                raise ValueError('Hermitian State-transfer operator is only implemented as work equations.')

        self.wf = copy.deepcopy(wave_function)
        self.theta_picker = ThetaPicker(
            self.wf.active_occ_spin_idx,
            self.wf.active_unocc_spin_idx,
            is_spin_conserving=is_spin_conserving,
        )

        self.G_ops: list[ResponseOperator] = []
        self.q_ops: list[ResponseOperator] = []
        self.do_debugging = do_debugging
        num_spin_orbs = self.wf.num_spin_orbs
        num_elec = self.wf.num_elec
        excitations = excitations.lower()
        # Construct unitary matrix, input from VQE
        U_matrix = construct_ucc_u(
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
        ### NEW
        inactive_str = 'I' * self.wf.num_inactive_spin_orbs
        virtual_str = 'I' * self.wf.num_virtual_spin_orbs
        U = OperatorHybrid(
            {inactive_str + virtual_str: OperatorHybridData(inactive_str, U_matrix, virtual_str)}
        )  # U is now an operator
        ### NEW

        #######
        ### Construct G / R based on excitation specified in excitations
        #######

        # Projection for statetransfer Ansatz.
        if do_projected_operators or do_statetransfer_operators:  # New projection
            projection = make_projection_operator(self.wf.state_vector)
            self.projection = projection
            # Old version with some bugs but much faster:
            # if self.wf.num_active_spin_orbs >= 10:
            #    projection = lw.outer(
            #        self.wf.state_vector.ket_active_csr, self.wf.state_vector.bra_active_csr
            #    )
            # else:
            #    projection = lw.outer(self.wf.state_vector.ket_active, self.wf.state_vector.bra_active)
        if 's' in excitations:
            for _, a, i, op_ in self.theta_picker.get_t1_generator_sa(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                if do_selfconsistent_operators and self.do_debugging:
                    op = op.apply_u_from_right(U_matrix.conj().transpose())
                    op = op.apply_u_from_left(U_matrix)
                if do_projected_operators and self.do_debugging:
                    op = op * projection
                    fac = expectation_value_hybrid(self.wf.state_vector, op, self.wf.state_vector)
                    op_diff_ = OperatorPauli({'I' * self.wf.num_spin_orbs: fac})
                    op_diff = convert_pauli_to_hybrid_form(
                        op_diff_,
                        self.wf.num_inactive_spin_orbs,
                        self.wf.num_active_spin_orbs,
                        self.wf.num_virtual_spin_orbs,
                    )
                    op = op - op_diff
                if do_statetransfer_operators and self.do_debugging:
                    op = op.apply_u_from_right(U_matrix.conj().transpose())
                    op = op.apply_u_from_left(U_matrix)
                    op = op * projection
                self.G_ops.append(ResponseOperator((i,), (a,), op))
        if 'd' in excitations:
            for _, a, i, b, j, op_ in self.theta_picker.get_t2_generator_sa(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                if do_selfconsistent_operators and self.do_debugging:
                    op = op.apply_u_from_right(U_matrix.conj().transpose())
                    op = op.apply_u_from_left(U_matrix)
                if do_projected_operators and self.do_debugging:
                    op = op * projection
                    fac = expectation_value_hybrid(self.wf.state_vector, op, self.wf.state_vector)
                    op_diff_ = OperatorPauli({'I' * self.wf.num_spin_orbs: fac})
                    op_diff = convert_pauli_to_hybrid_form(
                        op_diff_,
                        self.wf.num_inactive_spin_orbs,
                        self.wf.num_active_spin_orbs,
                        self.wf.num_virtual_spin_orbs,
                    )
                    op = op - op_diff
                if do_statetransfer_operators and self.do_debugging:
                    op = op.apply_u_from_right(U_matrix.conj().transpose())
                    op = op.apply_u_from_left(U_matrix)
                    op = op * projection
                self.G_ops.append(ResponseOperator((i, j), (a, b), op))
        if 't' in excitations:
            for _, a, i, b, j, c, k, op_ in self.theta_picker.get_t3_generator(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                if do_selfconsistent_operators and self.do_debugging:
                    op = op.apply_u_from_right(U_matrix.conj().transpose())
                    op = op.apply_u_from_left(U_matrix)
                if do_projected_operators and self.do_debugging:
                    op = op * projection
                    fac = expectation_value_hybrid(self.wf.state_vector, op, self.wf.state_vector)
                    op_diff_ = OperatorPauli({'I' * self.wf.num_spin_orbs: fac})
                    op_diff = convert_pauli_to_hybrid_form(
                        op_diff_,
                        self.wf.num_inactive_spin_orbs,
                        self.wf.num_active_spin_orbs,
                        self.wf.num_virtual_spin_orbs,
                    )
                    op = op - op_diff
                if do_statetransfer_operators and self.do_debugging:
                    op = op.apply_u_from_right(U_matrix.conj().transpose())
                    op = op.apply_u_from_left(U_matrix)
                    op = op * projection
                self.G_ops.append(ResponseOperator((i, j, k), (a, b, c), op))
        if 'q' in excitations:
            for _, a, i, b, j, c, k, d, l, op_ in self.theta_picker.get_t4_generator(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                if do_selfconsistent_operators and self.do_debugging:
                    op = op.apply_u_from_right(U_matrix.conj().transpose())
                    op = op.apply_u_from_left(U_matrix)
                if do_projected_operators and self.do_debugging:
                    op = op * projection
                    fac = expectation_value_hybrid(self.wf.state_vector, op, self.wf.state_vector)
                    op_diff_ = OperatorPauli({'I' * self.wf.num_spin_orbs: fac})
                    op_diff = convert_pauli_to_hybrid_form(
                        op_diff_,
                        self.wf.num_inactive_spin_orbs,
                        self.wf.num_active_spin_orbs,
                        self.wf.num_virtual_spin_orbs,
                    )
                    op = op - op_diff
                if do_statetransfer_operators and self.do_debugging:
                    op = op.apply_u_from_right(U_matrix.conj().transpose())
                    op = op.apply_u_from_left(U_matrix)
                    op = op * projection
                self.G_ops.append(ResponseOperator((i, j, k, l), (a, b, c, d), op))
        if '5' in excitations:
            for _, a, i, b, j, c, k, d, l, e, m, op_ in self.theta_picker.get_t5_generator(
                num_spin_orbs, num_elec
            ):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                if do_selfconsistent_operators and self.do_debugging:
                    op = op.apply_u_from_right(U_matrix.conj().transpose())
                    op = op.apply_u_from_left(U_matrix)
                if do_projected_operators and self.do_debugging:
                    op = op * projection
                    fac = expectation_value_hybrid(self.wf.state_vector, op, self.wf.state_vector)
                    op_diff_ = OperatorPauli({'I' * self.wf.num_spin_orbs: fac})
                    op_diff = convert_pauli_to_hybrid_form(
                        op_diff_,
                        self.wf.num_inactive_spin_orbs,
                        self.wf.num_active_spin_orbs,
                        self.wf.num_virtual_spin_orbs,
                    )
                    op = op - op_diff
                if do_statetransfer_operators and self.do_debugging:
                    op = op.apply_u_from_right(U_matrix.conj().transpose())
                    op = op.apply_u_from_left(U_matrix)
                    op = op * projection
                self.G_ops.append(ResponseOperator((i, j, k, l, m), (a, b, c, d, e), op))
        if '6' in excitations:
            for _, a, i, b, j, c, k, d, l, e, m, f, n, op_ in self.theta_picker.get_t6_generator(
                num_spin_orbs, num_elec
            ):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                if do_selfconsistent_operators and self.do_debugging:
                    op = op.apply_u_from_right(U_matrix.conj().transpose())
                    op = op.apply_u_from_left(U_matrix)
                if do_projected_operators and self.do_debugging:
                    op = op * projection
                    fac = expectation_value_hybrid(self.wf.state_vector, op, self.wf.state_vector)
                    op_diff_ = OperatorPauli({'I' * self.wf.num_spin_orbs: fac})
                    op_diff = convert_pauli_to_hybrid_form(
                        op_diff_,
                        self.wf.num_inactive_spin_orbs,
                        self.wf.num_active_spin_orbs,
                        self.wf.num_virtual_spin_orbs,
                    )
                    op = op - op_diff
                if do_statetransfer_operators and self.do_debugging:
                    op = op.apply_u_from_right(U_matrix.conj().transpose())
                    op = op.apply_u_from_left(U_matrix)
                    op = op * projection
                self.G_ops.append(ResponseOperator((i, j, k, l, m, n), (a, b, c, d, e, f), op))

        #######
        ### Construct q/Q
        #######

        for i, a in self.wf.kappa_idx:
            op_ = 2 ** (-1 / 2) * epq_pauli(a, i, self.wf.num_spin_orbs, self.wf.num_elec)
            op = convert_pauli_to_hybrid_form(
                op_,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.num_virtual_spin_orbs,
            )
            self.q_ops.append(ResponseOperator((i), (a), op))

        # Initiate matrices for linear response
        num_parameters = len(self.G_ops) + len(self.q_ops)
        self.M = np.zeros((num_parameters, num_parameters))
        self.Q = np.zeros((num_parameters, num_parameters))
        if track_hermitian_statetransfer:
            self.trackedQ = np.zeros((num_parameters, num_parameters))
        self.V = np.zeros((num_parameters, num_parameters))
        self.W = np.zeros((num_parameters, num_parameters))
        # Set up Hamiltonian(s)
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
        print('Gs', len(self.G_ops))
        print('qs', len(self.q_ops))
        grad = np.zeros(2 * len(self.q_ops))
        for i, op in enumerate(self.q_ops):
            grad[i] = expectation_value_contracted(
                self.wf.state_vector, commutator_contract(op.operator, H_1i_1a), self.wf.state_vector
            )
            grad[i + len(self.q_ops)] = expectation_value_contracted(
                self.wf.state_vector, commutator_contract(H_1i_1a, op.operator.dagger), self.wf.state_vector
            )
        if len(grad) != 0:
            print('idx, max(abs(grad orb)):', np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-4:
                raise ValueError('Large Hessian gradient detected.')
        grad = np.zeros(2 * len(self.G_ops))
        for i, op in enumerate(self.G_ops):
            grad[i] = expectation_value_contracted(
                self.wf.state_vector, commutator_contract(op.operator, H_en), self.wf.state_vector
            )
            grad[i + len(self.G_ops)] = expectation_value_contracted(
                self.wf.state_vector, commutator_contract(H_en, op.operator.dagger), self.wf.state_vector
            )
        if len(grad) != 0:
            print('idx, max(abs(grad active)):', np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-4:
                raise ValueError('Large Hessian gradient detected.')

        #######
        ### Construct matrices
        #######

        # Obtain |CSF> for naive implementation via work equations
        if do_selfconsistent_operators and not self.do_debugging:
            csf = copy.deepcopy(self.wf.state_vector)
            csf.active = csf._active
            csf.active_csr = ss.csr_matrix(csf._active)

        # Obtain |CSF> for naive implementation via work equations
        if do_statetransfer_operators and not self.do_debugging:
            csf = copy.deepcopy(self.wf.state_vector)
            csf.active = csf._active
            csf.active_csr = ss.csr_matrix(csf._active)

        # Obtain |CSF> for naive implementation via work equations
        if do_hermitian_statetransfer_operators and not self.do_debugging:
            csf = copy.deepcopy(self.wf.state_vector)
            csf.active = csf._active
            csf.active_csr = ss.csr_matrix(csf._active)

        # Work equation implementation
        if do_selfconsistent_operators:
            calculation_type = 'sc'
        elif do_projected_operators:  # not implemented yet
            calculation_type = 'proj'
        elif do_statetransfer_operators:
            calculation_type = 'st'
        elif do_hermitian_statetransfer_operators:
            if track_hermitian_statetransfer:
                calculation_type = 'tracked-hst'
            else:
                calculation_type = 'hst'
        else:
            calculation_type = 'naive'

        if self.do_debugging is True:  # no matter what operators
            calculation_type = 'generic'

        print('Calculation type: ', calculation_type)
        # QQ/qq
        for j, opJ in enumerate(self.q_ops):
            qJ = opJ.operator
            for i, opI in enumerate(self.q_ops):
                qI = opI.operator
                if i < j:
                    continue
                ### NEW
                if calculation_type in ('sc', 'naive', 'st', 'proj', 'hst', 'tracked-hst'):
                    # Make M
                    val = expectation_value_hybrid_flow(
                        self.wf.state_vector, [qI.dagger, H_2i_2a, qJ], self.wf.state_vector
                    ) - expectation_value_hybrid_flow(
                        self.wf.state_vector, [qI.dagger, qJ, H_2i_2a], self.wf.state_vector
                    )
                    self.M[i, j] = self.M[j, i] = val
                    # Make Q
                    self.Q[i, j] = self.Q[j, i] = -expectation_value_hybrid_flow(
                        self.wf.state_vector, [qI.dagger, qJ.dagger, H_2i_2a], self.wf.state_vector
                    )
                    # Make V
                    self.V[i, j] = self.V[j, i] = expectation_value_hybrid_flow(
                        self.wf.state_vector, [qI.dagger, qJ], self.wf.state_vector
                    )
                    # Make W = 0
                ### NEW End
                elif calculation_type == 'generic':
                    # Make M
                    self.M[i, j] = self.M[j, i] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(qI.dagger, H_2i_2a, qJ),
                        self.wf.state_vector,
                    )
                    # Make Q
                    self.Q[i, j] = self.Q[j, i] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(qI.dagger, H_2i_2a, qJ.dagger),
                        self.wf.state_vector,
                    )
                    # Make V
                    self.V[i, j] = self.V[j, i] = expectation_value_contracted(
                        self.wf.state_vector,
                        commutator_contract(qI.dagger, qJ),
                        self.wf.state_vector,
                    )
                    # Make W
                    self.W[i, j] = expectation_value_contracted(
                        self.wf.state_vector,
                        commutator_contract(qI.dagger, qJ.dagger),
                        self.wf.state_vector,
                    )
                    self.W[j, i] = -self.W[i, j]
                else:
                    raise NameError('Could not determine calculation_type got: {calculation_type}')
        # Gq/RQ and qG/QR
        # Remember: [G_i^d,[H,q_j]] = [[q_i^d,H],G_j] = [G_j,[H,q_i^d]]
        for j, opJ in enumerate(self.q_ops):
            qJ = opJ.operator
            for i, opI in enumerate(self.G_ops):
                GI = opI.operator
                ### NEW
                if calculation_type == 'sc':
                    # Make M
                    self.M[j, i + idx_shift] = self.M[i + idx_shift, j] = expectation_value_hybrid_flow(
                        csf, [GI.dagger, U.dagger, H_1i_1a, qJ], self.wf.state_vector
                    )
                    # Make Q
                    self.Q[j, i + idx_shift] = self.Q[i + idx_shift, j] = -expectation_value_hybrid_flow(
                        csf,
                        [GI.dagger, U.dagger, qJ.dagger, H_1i_1a],
                        self.wf.state_vector,
                    )
                    # Make V = 0
                    # Make W = 0
                elif calculation_type == 'naive':
                    # Make M
                    self.M[j, i + idx_shift] = self.M[i + idx_shift, j] = expectation_value_hybrid_flow(
                        self.wf.state_vector, [GI.dagger, H_1i_1a, qJ], self.wf.state_vector
                    ) - expectation_value_hybrid_flow(
                        self.wf.state_vector, [H_1i_1a, qJ, GI.dagger], self.wf.state_vector
                    )
                    # Make Q
                    self.Q[j, i + idx_shift] = self.Q[i + idx_shift, j] = expectation_value_hybrid_flow(
                        self.wf.state_vector, [qJ.dagger, H_1i_1a, GI.dagger], self.wf.state_vector
                    ) - expectation_value_hybrid_flow(
                        self.wf.state_vector, [GI.dagger, qJ.dagger, H_1i_1a], self.wf.state_vector
                    )
                    # Make V = 0
                    # Make W = 0
                elif calculation_type == 'proj':
                    # Make M
                    self.M[j, i + idx_shift] = self.M[i + idx_shift, j] = expectation_value_hybrid_flow(
                        self.wf.state_vector, [GI.dagger, H_1i_1a, qJ], self.wf.state_vector
                    )  # - (expectation_value_hybrid_flow(  # Just the gradient
                    #    self.wf.state_vector, [GI.dagger], self.wf.state_vector
                    # ) * expectation_value_hybrid_flow(
                    #    self.wf.state_vector, [H_1i_1a, qJ], self.wf.state_vector
                    # ))
                    # Make Q
                    self.Q[j, i + idx_shift] = self.Q[i + idx_shift, j] = -expectation_value_hybrid_flow(
                        self.wf.state_vector, [GI.dagger, qJ.dagger, H_1i_1a], self.wf.state_vector
                    )  # + (expectation_value_hybrid_flow(   # Just the gradient
                    #    self.wf.state_vector, [GI.dagger], self.wf.state_vector
                    # ) * expectation_value_hybrid_flow(
                    #    self.wf.state_vector, [qJ.dagger,H_1i_1a], self.wf.state_vector
                    # ))
                    # Make V = 0
                    # Make W = 0
                elif calculation_type == 'st':
                    # Make M
                    self.M[j, i + idx_shift] = self.M[i + idx_shift, j] = expectation_value_hybrid_flow(
                        csf, [GI.dagger, U.dagger, H_1i_1a, qJ], self.wf.state_vector
                    )
                    # Make Q
                    self.Q[j, i + idx_shift] = self.Q[i + idx_shift, j] = -expectation_value_hybrid_flow(
                        csf,
                        [GI.dagger, U.dagger, qJ.dagger, H_1i_1a],
                        self.wf.state_vector,
                    )
                    # Make V = 0
                    # Make W = 0
                elif calculation_type == 'hst':
                    # Make M
                    self.M[j, i + idx_shift] = self.M[i + idx_shift, j] = expectation_value_hybrid_flow(
                        csf, [GI.dagger, U.dagger, H_1i_1a, qJ], self.wf.state_vector
                    ) + expectation_value_hybrid_flow(
                        csf, [GI.dagger, U.dagger, qJ.dagger, H_1i_1a], self.wf.state_vector
                    )  # added an assumed zero (approximation)
                    # Make Q = 0 # assumed zero (approximation!)
                    # Make V = 0
                    # Make W = 0
                elif calculation_type == 'tracked-hst':
                    # Make M
                    self.M[j, i + idx_shift] = self.M[i + idx_shift, j] = expectation_value_hybrid_flow(
                        csf, [GI.dagger, U.dagger, H_1i_1a, qJ], self.wf.state_vector
                    ) + expectation_value_hybrid_flow(
                        csf, [GI.dagger, U.dagger, qJ.dagger, H_1i_1a], self.wf.state_vector
                    )  # added an assumed zero (approximation)
                    # Make Q for tracking error!
                    self.trackedQ[j, i + idx_shift] = self.trackedQ[
                        i + idx_shift, j
                    ] = -expectation_value_hybrid_flow(
                        csf,
                        [GI.dagger, U.dagger, qJ.dagger, H_1i_1a],
                        self.wf.state_vector,
                    )
                    # Make V = 0
                    # Make W = 0
                ### NEW End
                elif calculation_type == 'generic':
                    # Make M
                    self.M[j, i + idx_shift] = self.M[i + idx_shift, j] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(GI.dagger, H_1i_1a, qJ),
                        self.wf.state_vector,
                    )
                    # Make Q
                    self.Q[j, i + idx_shift] = self.Q[i + idx_shift, j] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(GI.dagger, H_1i_1a, qJ.dagger),
                        # double_commutator_contract(qJ.dagger, H_1i_1a, GI.dagger),  # wrong parametrization
                        self.wf.state_vector,
                    )
                    # Make V
                    self.V[j, i + idx_shift] = self.V[i + idx_shift, j] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(GI.dagger, qJ), self.wf.state_vector
                    )
                    # Make W
                    self.W[j, i + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(GI.dagger, qJ.dagger), self.wf.state_vector
                    )
                    self.W[i + idx_shift, j] = -self.W[j, i + idx_shift]
                else:
                    raise NameError('Could not determine calculation_type got: {calculation_type}')
        # GG/RR
        for j, opJ in enumerate(self.G_ops):
            GJ = opJ.operator
            for i, opI in enumerate(self.G_ops):
                GI = opI.operator
                if i < j:
                    continue
                ### NEW
                if calculation_type == 'sc':
                    # Make M
                    val = expectation_value_hybrid_flow(
                        csf, [GI.dagger, U.dagger, H_en, U, GJ], csf
                    ) - expectation_value_hybrid_flow(
                        csf, [GI.dagger, GJ, U.dagger, H_en], self.wf.state_vector
                    )
                    self.M[i + idx_shift, j + idx_shift] = self.M[j + idx_shift, i + idx_shift] = val
                    # Make Q
                    self.Q[i + idx_shift, j + idx_shift] = self.Q[
                        j + idx_shift, i + idx_shift
                    ] = -expectation_value_hybrid_flow(
                        csf, [GI.dagger, GJ.dagger, U.dagger, H_en], self.wf.state_vector
                    )
                    # Make V
                    if i == j:
                        self.V[i + idx_shift, j + idx_shift] = self.V[j + idx_shift, i + idx_shift] = 1
                    # Make W = 0
                elif calculation_type == 'naive':
                    # Make M
                    val = (
                        expectation_value_hybrid_flow(
                            self.wf.state_vector, [GI.dagger, H_en, GJ], self.wf.state_vector
                        )
                        - expectation_value_hybrid_flow(
                            self.wf.state_vector, [H_en, GJ, GI.dagger], self.wf.state_vector
                        )
                        - expectation_value_hybrid_flow(
                            self.wf.state_vector, [GI.dagger, GJ, H_en], self.wf.state_vector
                        )
                        + expectation_value_hybrid_flow(
                            self.wf.state_vector, [GJ, H_en, GI.dagger], self.wf.state_vector
                        )
                    )
                    self.M[i + idx_shift, j + idx_shift] = self.M[j + idx_shift, i + idx_shift] = val
                    # Make Q
                    val = (
                        expectation_value_hybrid_flow(
                            self.wf.state_vector, [GI.dagger, H_en, GJ.dagger], self.wf.state_vector
                        )
                        - expectation_value_hybrid_flow(
                            self.wf.state_vector, [H_en, GJ.dagger, GI.dagger], self.wf.state_vector
                        )
                        - expectation_value_hybrid_flow(
                            self.wf.state_vector, [GI.dagger, GJ.dagger, H_en], self.wf.state_vector
                        )
                        + expectation_value_hybrid_flow(
                            self.wf.state_vector, [GJ.dagger, H_en, GI.dagger], self.wf.state_vector
                        )
                    )
                    self.Q[i + idx_shift, j + idx_shift] = self.Q[j + idx_shift, i + idx_shift] = val
                    # Make V
                    self.V[i + idx_shift, j + idx_shift] = self.V[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_hybrid_flow(
                        self.wf.state_vector, [GI.dagger, GJ], self.wf.state_vector
                    ) - expectation_value_hybrid_flow(
                        self.wf.state_vector, [GJ, GI.dagger], self.wf.state_vector
                    )
                    # Make W = 0
                elif calculation_type == 'proj':
                    # Make M
                    val = (
                        expectation_value_hybrid_flow(
                            self.wf.state_vector, [GI.dagger, H_en, GJ], self.wf.state_vector
                        )
                        - (
                            expectation_value_hybrid_flow(
                                self.wf.state_vector, [GI.dagger, GJ], self.wf.state_vector
                            )
                            * self.wf.energy_elec
                        )
                        - (
                            expectation_value_hybrid_flow(
                                self.wf.state_vector, [GI.dagger], self.wf.state_vector
                            )
                            * expectation_value_hybrid_flow(
                                self.wf.state_vector, [H_en, GJ], self.wf.state_vector
                            )
                        )
                        + (
                            expectation_value_hybrid_flow(
                                self.wf.state_vector, [GI.dagger], self.wf.state_vector
                            )
                            * expectation_value_hybrid_flow(  # ToDo: this can be done more efficiently by calculating once and storing in WF object
                                self.wf.state_vector, [GJ], self.wf.state_vector
                            )
                            * self.wf.energy_elec
                        )
                    )
                    self.M[i + idx_shift, j + idx_shift] = self.M[j + idx_shift, i + idx_shift] = val
                    # Make Q
                    val = (
                        expectation_value_hybrid_flow(
                            self.wf.state_vector, [GI.dagger, H_en], self.wf.state_vector
                        )
                        * expectation_value_hybrid_flow(
                            self.wf.state_vector, [GJ.dagger], self.wf.state_vector
                        )
                    ) - (
                        expectation_value_hybrid_flow(self.wf.state_vector, [GI.dagger], self.wf.state_vector)
                        * expectation_value_hybrid_flow(
                            self.wf.state_vector, [GJ.dagger], self.wf.state_vector
                        )
                        * self.wf.energy_elec
                    )
                    self.Q[i + idx_shift, j + idx_shift] = self.Q[j + idx_shift, i + idx_shift] = val
                    # Make V
                    self.V[i + idx_shift, j + idx_shift] = self.V[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_hybrid_flow(
                        self.wf.state_vector, [GI.dagger, GJ], self.wf.state_vector
                    ) - (
                        expectation_value_hybrid_flow(self.wf.state_vector, [GI.dagger], self.wf.state_vector)
                        * expectation_value_hybrid_flow(self.wf.state_vector, [GJ], self.wf.state_vector)
                    )
                    # Make W = 0
                elif calculation_type in ('st', 'hst', 'tracked-hst'):
                    # Make M (A)
                    if i == j:
                        val = (
                            expectation_value_hybrid_flow(csf, [GI.dagger, U.dagger, H_en, U, GJ], csf)
                            - self.wf.energy_elec
                        )
                        self.M[i + idx_shift, j + idx_shift] = self.M[j + idx_shift, i + idx_shift] = val
                        self.V[i + idx_shift, j + idx_shift] = self.V[j + idx_shift, i + idx_shift] = 1
                    else:
                        val = expectation_value_hybrid_flow(csf, [GI.dagger, U.dagger, H_en, U, GJ], csf)
                        self.M[i + idx_shift, j + idx_shift] = self.M[j + idx_shift, i + idx_shift] = val
                    # Make Q (B)= 0
                    # Make V (\Sigma) = \delta_ij (see above)
                    # Make W (\Delta) = 0
                ### NEW End
                elif calculation_type == 'generic':
                    # Make M
                    self.M[i + idx_shift, j + idx_shift] = self.M[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(GI.dagger, H_en, GJ),
                        self.wf.state_vector,
                    )
                    # Make Q
                    self.Q[i + idx_shift, j + idx_shift] = self.Q[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(GI.dagger, H_en, GJ.dagger),
                        self.wf.state_vector,
                    )
                    # Make V
                    self.V[i + idx_shift, j + idx_shift] = self.V[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(GI.dagger, GJ), self.wf.state_vector
                    )
                    # Make W
                    self.W[i + idx_shift, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(GI.dagger, GJ.dagger), self.wf.state_vector
                    )
                    self.W[j + idx_shift, i + idx_shift] = -self.W[i + idx_shift, j + idx_shift]
                else:
                    raise NameError('Could not determine calculation_type got: {calculation_type}')

    def calc_excitation_energies(self) -> None:
        """Calculate excitation energies."""
        size = len(self.M)
        E2 = np.zeros((size * 2, size * 2))
        E2[:size, :size] = self.M
        E2[:size, size:] = self.Q
        E2[size:, :size] = np.conj(self.Q)
        E2[size:, size:] = np.conj(self.M)
        (
            hess_eigval,
            _,
        ) = np.linalg.eig(E2)
        print(f'Smallest Hessian eigenvalue: {np.min(hess_eigval)}')

        S = np.zeros((size * 2, size * 2))
        S[:size, :size] = self.V
        S[:size, size:] = self.W
        S[size:, :size] = -np.conj(self.W)
        S[size:, size:] = -np.conj(self.V)
        print(f'Smallest diagonal element in the metric: {np.min(np.abs(np.diagonal(self.V)))}')
        if np.min(np.abs(np.diagonal(self.V))) < 0:
            raise ValueError('This value is bad. Abort.')

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

    def get_excited_state_overlap(self, state_number: int) -> float:
        """Calculate overlap of excitated state with the ground state.

        Args:
            state_number: Which excited state, counting from zero.

        Returns:
            Overlap between ground state and excited state.
        """
        number_excitations = len(self.excitation_energies)
        print('WARNING: This function [get_excited_state_overlap] might not be working.')
        for i, op in enumerate(self.q_ops + self.G_ops):
            G = op.operator
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
        return expectation_value_hybrid(self.wf.state_vector, transfer_op, self.wf.state_vector)

    def get_excited_state_norm(self, state_number: int) -> float:
        """Calculate the norm of excited state.

        Args:
            state_number: Which excited state, counting from zero.

        Returns:
            Norm of excited state.
        """
        number_excitations = len(self.excitation_energies)
        for i, op in enumerate(self.q_ops + self.G_ops):
            G = op.operator
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
            G = op.operator
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
