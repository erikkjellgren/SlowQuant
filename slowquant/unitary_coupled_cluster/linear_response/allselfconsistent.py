import copy

import numpy as np
import scipy
import scipy.sparse as ss

from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
    ResponseOperator,
)
from slowquant.unitary_coupled_cluster.operator_hybrid import (
    OperatorHybrid,
    OperatorHybridData,
    convert_pauli_to_hybrid_form,
    expectation_value_hybrid_flow,
)
from slowquant.unitary_coupled_cluster.operator_pauli import (
    epq_pauli,
    hamiltonian_pauli_2i_2a,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.util import construct_ucc_u


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

        # Overwrite Superclass
        self.q_ops: list[ResponseOperator] = []
        for i, a in self.wf.kappa_hf_like_idx:
            op_ = 2 ** (-1 / 2) * epq_pauli(a, i, self.wf.num_spin_orbs, self.wf.num_elec)
            op = convert_pauli_to_hybrid_form(
                op_,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
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
                self.wf.num_elec,
            ),
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
        )

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
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in q of ", np.max(np.abs(grad)))
        grad = np.zeros(2 * len(self.G_ops))
        for i, op in enumerate(self.G_ops):
            grad[i] = -expectation_value_hybrid_flow(
                self.wf.state_vector, [self.H_1i_1a, self.U, op.operator], self.csf
            )
            grad[i + len(self.G_ops)] = expectation_value_hybrid_flow(
                self.csf, [op.operator.dagger, self.U.dagger, self.H_1i_1a], self.wf.state_vector
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad)))
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
                    self.csf, [GI.dagger, self.U.dagger, self.H_1i_1a, self.U, qJ], self.csf
                )
                self.A[i + idx_shift, j] = self.A[j, i + idx_shift] = val
                # Make B
                self.B[i + idx_shift, j] = self.B[j, i + idx_shift] = -expectation_value_hybrid_flow(
                    self.csf,
                    [GI.dagger, qJ.dagger, self.U.dagger, self.H_1i_1a],
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
                    self.csf, [GI.dagger, self.U.dagger, self.H_en, self.U, GJ], self.csf
                ) - expectation_value_hybrid_flow(
                    self.csf, [GI.dagger, GJ, self.U.dagger, self.H_en], self.wf.state_vector
                )
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make B
                self.B[i + idx_shift, j + idx_shift] = self.B[
                    j + idx_shift, i + idx_shift
                ] = -expectation_value_hybrid_flow(
                    self.csf, [GI.dagger, GJ.dagger, self.U.dagger, self.H_en], self.wf.state_vector
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
