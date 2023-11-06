import copy
from collections.abc import Sequence

import numpy as np
import scipy
import scipy.sparse as ss

from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_hybrid import (
    OperatorHybrid,
    OperatorHybridData,
    convert_pauli_to_hybrid_form,
    expectation_value_hybrid_flow,
)
from slowquant.unitary_coupled_cluster.operator_pauli import (
    energy_hamiltonian_pauli,
    epq_pauli,
    hamiltonian_pauli_1i_1a,
    hamiltonian_pauli_2i_2a,
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
        self.wf = copy.deepcopy(wave_function)
        self.theta_picker = ThetaPicker(
            self.wf.active_occ_spin_idx,
            self.wf.active_unocc_spin_idx,
            is_spin_conserving=is_spin_conserving,
        )

        self.G_ops: list[ResponseOperator] = []
        self.q_ops: list[ResponseOperator] = []
        num_spin_orbs = self.wf.num_spin_orbs
        num_elec = self.wf.num_elec
        excitations = excitations.lower()
        U_matrix = construct_ucc_u(
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
        inactive_str = "I" * self.wf.num_inactive_spin_orbs
        virtual_str = "I" * self.wf.num_virtual_spin_orbs
        self.U = OperatorHybrid(
            {inactive_str + virtual_str: OperatorHybridData(inactive_str, U_matrix, virtual_str)}
        )
        if "s" in excitations:
            for _, a, i, op_ in self.theta_picker.get_t1_generator_sa(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(ResponseOperator((i,), (a,), op))
        if "d" in excitations:
            for _, a, i, b, j, op_ in self.theta_picker.get_t2_generator_sa(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(ResponseOperator((i, j), (a, b), op))
        if "t" in excitations:
            for _, a, i, b, j, c, k, op_ in self.theta_picker.get_t3_generator(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(ResponseOperator((i, j, k), (a, b, c), op))
        if "q" in excitations:
            for _, a, i, b, j, c, k, d, l, op_ in self.theta_picker.get_t4_generator(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(ResponseOperator((i, j, k, l), (a, b, c, d), op))
        if "5" in excitations:
            for _, a, i, b, j, c, k, d, l, e, m, op_ in self.theta_picker.get_t5_generator(
                num_spin_orbs, num_elec
            ):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(ResponseOperator((i, j, k, l, m), (a, b, c, d, e), op))
        if "6" in excitations:
            for _, a, i, b, j, c, k, d, l, e, m, f, n, op_ in self.theta_picker.get_t6_generator(
                num_spin_orbs, num_elec
            ):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(ResponseOperator((i, j, k, l, m, n), (a, b, c, d, e, f), op))
        for i, a in self.wf.kappa_hf_like_idx:
            op_ = 2 ** (-1 / 2) * epq_pauli(a, i, self.wf.num_spin_orbs, self.wf.num_elec)
            op = convert_pauli_to_hybrid_form(
                op_,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.num_virtual_spin_orbs,
            )
            self.q_ops.append(ResponseOperator((i), (a), op))

        num_parameters = len(self.G_ops) + len(self.q_ops)
        self.A = np.zeros((num_parameters, num_parameters))
        self.B = np.zeros((num_parameters, num_parameters))
        self.Sigma = np.zeros((num_parameters, num_parameters))
        self.Delta = np.zeros((num_parameters, num_parameters))
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
        self.csf = copy.deepcopy(self.wf.state_vector)
        self.csf.active = self.csf._active
        self.csf.active_csr = ss.csr_matrix(self.csf._active)
        print("Gs", len(self.G_ops))
        print("qs", len(self.q_ops))
        grad = np.zeros(2 * len(self.q_ops))
        print("WARNING!")
        print("Gradient working equations not implemented for state transfer q operators")
        if len(grad) != 0:
            print("idx, max(abs(grad orb)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
        grad = np.zeros(2 * len(self.G_ops))
        for i, op in enumerate(self.G_ops):
            grad[i] = -expectation_value_hybrid_flow(
                self.wf.state_vector, [H_1i_1a, self.U, op.operator], self.csf
            )
            grad[i + len(self.G_ops)] = expectation_value_hybrid_flow(
                self.csf, [op.operator.dagger, self.U.dagger, H_1i_1a], self.wf.state_vector
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
        for j, opJ in enumerate(self.q_ops):
            qJ = opJ.operator
            for i, opI in enumerate(self.q_ops):
                qI = opI.operator
                if i < j:
                    continue
                # Make A
                val = expectation_value_hybrid_flow(
                    self.csf, [qI.dagger, self.U.dagger, H_2i_2a, self.U, qJ], self.csf
                )
                val -= expectation_value_hybrid_flow(
                    self.csf, [qI.dagger, qJ, self.U.dagger, H_2i_2a], self.wf.state_vector
                )
                self.A[i, j] = self.A[j, i] = val
                # Make B
                self.B[i, j] = self.B[j, i] = -expectation_value_hybrid_flow(
                    self.csf, [qI.dagger, qJ.dagger, self.U.dagger, H_2i_2a], self.wf.state_vector
                )
                # Make Sigma
                if i == j:
                    self.Sigma[i, j] = self.Sigma[j, i] = 1
        for j, opJ in enumerate(self.q_ops):
            qJ = opJ.operator
            for i, opI in enumerate(self.G_ops):
                GI = opI.operator
                # Make A
                val = expectation_value_hybrid_flow(
                    self.csf, [GI.dagger, self.U.dagger, H_1i_1a, self.U, qJ], self.csf
                )
                self.A[i + idx_shift, j] = self.A[j, i + idx_shift] = val
                # Make B
                self.B[i + idx_shift, j] = self.B[j, i + idx_shift] = -expectation_value_hybrid_flow(
                    self.csf,
                    [GI.dagger, qJ.dagger, self.U.dagger, H_1i_1a],
                    self.wf.state_vector,
                )
        for j, opJ in enumerate(self.G_ops):
            GJ = opJ.operator
            for i, opI in enumerate(self.G_ops):
                GI = opI.operator
                if i < j:
                    continue
                # Make A
                val = expectation_value_hybrid_flow(
                    self.csf, [GI.dagger, self.U.dagger, H_en, self.U, GJ], self.csf
                ) - expectation_value_hybrid_flow(
                    self.csf, [GI.dagger, GJ, self.U.dagger, H_en], self.wf.state_vector
                )
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make B
                self.B[i + idx_shift, j + idx_shift] = self.B[
                    j + idx_shift, i + idx_shift
                ] = -expectation_value_hybrid_flow(
                    self.csf, [GI.dagger, GJ.dagger, self.U.dagger, H_en], self.wf.state_vector
                )
                # Make Sigma
                if i == j:
                    self.Sigma[i + idx_shift, j + idx_shift] = 1

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
        print(f"Smallest Hessian eigenvalue: {np.min(hess_eigval)}")

        S = np.zeros((size * 2, size * 2))
        S[:size, :size] = self.Sigma
        S[:size, size:] = self.Delta
        S[size:, :size] = -self.Delta
        S[size:, size:] = -self.Sigma
        print(f"Smallest diagonal element in the metric: {np.min(np.abs(np.diagonal(self.Sigma)))}")

        eigval, eigvec = scipy.linalg.eig(E2, S)
        sorting = np.argsort(eigval)
        self.excitation_energies = np.real(eigval[sorting][size:])
        self.response_vectors = np.real(eigvec[:, sorting][:, size:])
