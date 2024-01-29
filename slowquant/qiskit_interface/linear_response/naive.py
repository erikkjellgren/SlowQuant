import numpy as np
import scipy
from dmdm.util import iterate_t1_sa, iterate_t2_sa  # temporary solution

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.qiskit_interface.base import FermionicOperator
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.qiskit_interface.linear_response.lr_baseclass import quantumLRBaseClass
from slowquant.qiskit_interface.operators import (
    G1,
    G2_1,
    G2_2,
    commutator,
    double_commutator,
    hamiltonian_pauli_0i_0a,
    hamiltonian_pauli_1i_1a,
    hamiltonian_pauli_2i_2a,
)
from slowquant.qiskit_interface.wavefunction import WaveFunction
from slowquant.unitary_coupled_cluster.density_matrix import (
    ReducedDenstiyMatrix,
    get_orbital_gradient_response,
    get_orbital_response_hessian_block,
    get_orbital_response_metric_sigma,
    get_orbital_response_property_gradient,
    get_orbital_response_vector_norm,
)


class quantumLR(quantumLRBaseClass):
    def run(
        self,
    ) -> None:
        """
        Run simulation of naive LR matrix elements
        """
        # RDMs
        rdms = ReducedDenstiyMatrix(
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
            self.wf.rdm1,
            rdm2=self.wf.rdm2,
        )

        idx_shift = self.num_q
        print("Gs", self.num_G)
        print("qs", self.num_q)

        # Check gradients
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
                print("WARNING: Large Gradient detected in q of ", np.max(np.abs(grad)))

        grad = np.zeros(2 * self.num_G)
        for i, op in enumerate(self.G_ops):
            grad[i] = self.wf.QI.quantum_expectation_value(
                commutator(self.H_0i_0a, op).get_folded_operator(*self.orbs)
            )
            grad[i + self.num_G] = self.wf.QI.quantum_expectation_value(
                commutator(op.dagger, self.H_0i_0a).get_folded_operator(*self.orbs)
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                print("WARNING: Large Gradient detected in G of ", np.max(np.abs(grad)))

        # qq
        self.A[: self.num_q, : self.num_q] = get_orbital_response_hessian_block(
            rdms,
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.kappa_idx_dagger,
            self.wf.kappa_idx,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        self.B[: self.num_q, : self.num_q] = get_orbital_response_hessian_block(
            rdms,
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.kappa_idx_dagger,
            self.wf.kappa_idx_dagger,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        self.Sigma[: self.num_q, : self.num_q] = get_orbital_response_metric_sigma(rdms, self.wf.kappa_idx)

        # Gq
        for j, qJ in enumerate(self.q_ops):
            for i, GI in enumerate(self.G_ops):
                # Make A
                val = self.wf.QI.quantum_expectation_value(
                    (GI.dagger * self.H_1i_1a * qJ).get_folded_operator(*self.orbs)
                ) - self.wf.QI.quantum_expectation_value(
                    (self.H_1i_1a * qJ * GI.dagger).get_folded_operator(*self.orbs)
                )
                self.A[i + idx_shift, j] = self.A[j, i + idx_shift] = val
                # Make B
                val = self.wf.QI.quantum_expectation_value(
                    (qJ.dagger * self.H_1i_1a * GI.dagger).get_folded_operator(*self.orbs)
                ) - self.wf.QI.quantum_expectation_value(
                    (GI.dagger * qJ.dagger * self.H_1i_1a).get_folded_operator(*self.orbs)
                )
                self.B[i + idx_shift, j] = self.B[j, i + idx_shift] = val

        # GG
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                self.A[i + idx_shift, j + idx_shift] = self.A[
                    j + idx_shift, i + idx_shift
                ] = self.wf.QI.quantum_expectation_value(
                    double_commutator(GI.dagger, self.H_0i_0a, GJ).get_folded_operator(*self.orbs)
                )
                # Make B
                self.B[i + idx_shift, j + idx_shift] = self.B[
                    j + idx_shift, i + idx_shift
                ] = self.wf.QI.quantum_expectation_value(
                    double_commutator(GI.dagger, self.H_0i_0a, GJ.dagger).get_folded_operator(*self.orbs)
                )
                # Make Sigma
                self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[
                    j + idx_shift, i + idx_shift
                ] = self.wf.QI.quantum_expectation_value(
                    commutator(GI.dagger, GJ).get_folded_operator(*self.orbs)
                )

    def _get_qbitmap(
        self,
    ) -> np.ndarray:
        idx_shift = self.num_q
        print("Gs", self.num_G)
        print("qs", self.num_q)

        ## qq block not possible (yet) as per RDMs

        A = [[0] * self.num_params for _ in range(self.num_params)]
        B = [[0] * self.num_params for _ in range(self.num_params)]
        Sigma = [[0] * self.num_params for _ in range(self.num_params)]

        # Gq
        for j, qJ in enumerate(self.q_ops):
            for i, GI in enumerate(self.G_ops):
                # Make A
                val = self.wf.QI.op_to_qbit(
                    (GI.dagger * self.H_1i_1a * qJ).get_folded_operator(*self.orbs)
                ) - self.wf.QI.op_to_qbit((self.H_1i_1a * qJ * GI.dagger).get_folded_operator(*self.orbs))
                A[i + idx_shift][j] = A[j][i + idx_shift] = val
                # Make B
                val = self.wf.QI.op_to_qbit(
                    (qJ.dagger * self.H_1i_1a * GI.dagger).get_folded_operator(*self.orbs)
                ) - self.wf.QI.op_to_qbit(
                    (GI.dagger * qJ.dagger * self.H_1i_1a).get_folded_operator(*self.orbs)
                )
                B[i + idx_shift][j] = B[j][i + idx_shift] = val

        # GG
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                A[i + idx_shift][j + idx_shift] = A[j + idx_shift][i + idx_shift] = self.wf.QI.op_to_qbit(
                    double_commutator(GI.dagger, self.H_1i_1a, GJ).get_folded_operator(*self.orbs)
                )
                # Make B
                B[i + idx_shift][j + idx_shift] = B[j + idx_shift][i + idx_shift] = self.wf.QI.op_to_qbit(
                    double_commutator(GI.dagger, self.H_1i_1a, GJ.dagger).get_folded_operator(*self.orbs)
                )
                # Make Sigma
                Sigma[i + idx_shift][j + idx_shift] = Sigma[j + idx_shift][
                    i + idx_shift
                ] = self.wf.QI.op_to_qbit(commutator(GI.dagger, GJ).get_folded_operator(*self.orbs))

        return A, B, Sigma
