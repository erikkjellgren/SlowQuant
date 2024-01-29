from collections.abc import Sequence

import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.qiskit_interface.base import FermionicOperator
from slowquant.qiskit_interface.linear_response.lr_baseclass import quantumLRBaseClass
from slowquant.qiskit_interface.operators import one_elec_op_pauli_0i_0a
from slowquant.unitary_coupled_cluster.density_matrix import (
    ReducedDenstiyMatrix,
    get_orbital_gradient_response,
    get_orbital_response_hessian_block,
    get_orbital_response_metric_sigma,
    get_orbital_response_property_gradient,
)


class quantumLR(quantumLRBaseClass):
    def run(
        self,
    ) -> None:
        """
        Run simulation of projected LR matrix elements
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

        # pre-calculate <0|G|0> and <0|HG|0>
        self.G_exp = []  # save and use for properties
        HG_exp = []
        for GJ in self.G_ops:
            self.G_exp.append(self.wf.QI.quantum_expectation_value(GJ.get_folded_operator(*self.orbs)))
            HG_exp.append(
                self.wf.QI.quantum_expectation_value((self.H_0i_0a * GJ).get_folded_operator(*self.orbs))
            )

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

        grad = np.zeros(self.num_G)  # G^\dagger is the same
        for i in range(self.num_G):
            grad[i] = HG_exp[i] - (self.wf.energy_elec * self.G_exp[i])
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
                self.A[j, i + idx_shift] = self.A[i + idx_shift, j] = self.wf.QI.quantum_expectation_value(
                    (GI.dagger * self.H_1i_1a * qJ).get_folded_operator(*self.orbs)
                )
                # Make B
                self.B[j, i + idx_shift] = self.B[i + idx_shift, j] = -self.wf.QI.quantum_expectation_value(
                    (GI.dagger * qJ.dagger * self.H_1i_1a).get_folded_operator(*self.orbs)
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
                val -= self.G_exp[i] * HG_exp[j]
                val += self.G_exp[i] * self.G_exp[j] * self.wf.energy_elec
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make B
                val = HG_exp[i] * self.G_exp[j]
                val -= self.G_exp[i] * self.G_exp[j] * self.wf.energy_elec
                self.B[i + idx_shift, j + idx_shift] = self.B[j + idx_shift, i + idx_shift] = val
                # Make Sigma
                self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[
                    j + idx_shift, i + idx_shift
                ] = GG_exp - (self.G_exp[i] * self.G_exp[j])

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
        mux_op = one_elec_op_pauli_0i_0a(mux, self.wf.num_inactive_orbs, self.wf.num_active_orbs)
        muy_op = one_elec_op_pauli_0i_0a(muy, self.wf.num_inactive_orbs, self.wf.num_active_orbs)
        muz_op = one_elec_op_pauli_0i_0a(muz, self.wf.num_inactive_orbs, self.wf.num_active_orbs)

        transition_dipoles = np.zeros((number_excitations, 3))
        for state_number in range(number_excitations):
            q_part_x = get_orbital_response_property_gradient(
                rdms,
                mux,
                self.wf.kappa_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.normed_excitation_vectors,
                state_number,
                number_excitations,
            )
            q_part_y = get_orbital_response_property_gradient(
                rdms,
                muy,
                self.wf.kappa_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.normed_excitation_vectors,
                state_number,
                number_excitations,
            )
            q_part_z = get_orbital_response_property_gradient(
                rdms,
                muz,
                self.wf.kappa_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.normed_excitation_vectors,
                state_number,
                number_excitations,
            )
            g_part_x = 0.0
            g_part_y = 0.0
            g_part_z = 0.0
            exp_mux = self.wf.QI.quantum_expectation_value(mux_op.get_folded_operator(*self.orbs))
            exp_muy = self.wf.QI.quantum_expectation_value(muy_op.get_folded_operator(*self.orbs))
            exp_muz = self.wf.QI.quantum_expectation_value(muz_op.get_folded_operator(*self.orbs))
            for i, G in enumerate(self.G_ops):
                exp_G = self.G_exp[i]
                exp_Gmux = self.wf.QI.quantum_expectation_value(
                    (G.dagger * mux_op).get_folded_operator(*self.orbs)
                )
                exp_Gmuy = self.wf.QI.quantum_expectation_value(
                    (G.dagger * muy_op).get_folded_operator(*self.orbs)
                )
                exp_Gmuz = self.wf.QI.quantum_expectation_value(
                    (G.dagger * muz_op).get_folded_operator(*self.orbs)
                )

                g_part_x += self._Z_G_normed[i, state_number] * exp_G * exp_mux
                g_part_x -= self._Z_G_normed[i, state_number] * exp_Gmux
                g_part_x -= self._Y_G_normed[i, state_number] * exp_G * exp_mux
                g_part_x += self._Y_G_normed[i, state_number] * exp_Gmux
                g_part_y += self._Z_G_normed[i, state_number] * exp_G * exp_muy
                g_part_y -= self._Z_G_normed[i, state_number] * exp_Gmuy
                g_part_y -= self._Y_G_normed[i, state_number] * exp_G * exp_muy
                g_part_y += self._Y_G_normed[i, state_number] * exp_Gmuy
                g_part_z += self._Z_G_normed[i, state_number] * exp_G * exp_muz
                g_part_z -= self._Z_G_normed[i, state_number] * exp_Gmuz
                g_part_z -= self._Y_G_normed[i, state_number] * exp_G * exp_muz
                g_part_z += self._Y_G_normed[i, state_number] * exp_Gmuz

            transition_dipoles[state_number, 0] = q_part_x + g_part_x
            transition_dipoles[state_number, 1] = q_part_y + g_part_y
            transition_dipoles[state_number, 2] = q_part_z + g_part_z
        return transition_dipoles
