import numpy as np
from dmdm.util import iterate_t1_sa, iterate_t2_sa  # temporary solution

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.qiskit_interface.base import FermionicOperator
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.qiskit_interface.linear_response.lr_baseclass import quantumLRBaseClass
from slowquant.qiskit_interface.operators import hamiltonian_pauli_2i_2a
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
        Run simulation of all projected LR matrix elements
        """
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

        # This hamiltonian is expensive and not needed in the naive orbitale rotation formalism
        self.H_2i_2a = hamiltonian_pauli_2i_2a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
        )

        # pre-calculate <0|G|0> and <0|HG|0>
        G_exp = []
        HG_exp = []
        for j, GJ in enumerate(self.G_ops):
            G_exp.append(self.wf.QI.quantum_expectation_value(GJ.get_folded_operator(*self.orbs)))
            HG_exp.append(
                self.wf.QI.quantum_expectation_value((self.H_0i_0a * GJ).get_folded_operator(*self.orbs))
            )

        # Check gradients
        grad = get_orbital_gradient_response(  # proj-q and naive-q lead to same working equations
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

        grad = np.zeros(self.num_G)  # G^\dagger is the same
        for i, op in enumerate(self.G_ops):
            grad[i] = HG_exp[i] - (self.wf.energy_elec * G_exp[i])
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                print("WARNING: Large Gradient detected in G of ", np.max(np.abs(grad)))

        # qq
        for j, qJ in enumerate(self.q_ops):
            for i, qI in enumerate(self.q_ops[j:], j):
                # Make A
                val = self.wf.QI.quantum_expectation_value(
                    (qI.dagger * self.H_2i_2a * qJ).get_folded_operator(*self.orbs)
                )
                qq_exp = self.wf.QI.quantum_expectation_value(
                    (qI.dagger * qJ).get_folded_operator(*self.orbs)
                )
                val -= qq_exp * self.wf.energy_elec
                self.A[i, j] = self.A[j, i] = val
                # Make Sigma
                self.Sigma[i, j] = self.Sigma[j, i] = qq_exp

        # Gq
        for j, qJ in enumerate(self.q_ops):
            for i, GI in enumerate(self.G_ops):
                # Make A
                self.A[j, i + idx_shift] = self.A[i + idx_shift, j] = self.wf.QI.quantum_expectation_value(
                    (GI.dagger * self.H_1i_1a * qJ).get_folded_operator(*self.orbs)
                )

        # Calculate Matrices
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                val = self.wf.QI.quantum_expectation_value(
                    (GI.dagger * self.H_0i_0a * GJ).get_folded_operator(*self.orbs)
                )
                GG_exp = self.wf.QI.quantum_expectation_value(
                    (GI.dagger * GJ).get_folded_operator(*self.orbs)
                )
                val -= GG_exp * self.wf.energy_elec
                val -= G_exp[i] * HG_exp[j]
                val += G_exp[i] * G_exp[j] * self.wf.energy_elec
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make B
                val = HG_exp[i] * G_exp[j]
                val -= G_exp[i] * G_exp[j] * self.wf.energy_elec
                self.B[i + idx_shift, j + idx_shift] = self.B[j + idx_shift, i + idx_shift] = val
                # Make Sigma
                self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[
                    j + idx_shift, i + idx_shift
                ] = GG_exp - (G_exp[i] * G_exp[j])
