from collections.abc import Sequence

import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.qiskit_interface.interface import make_cliques
from slowquant.qiskit_interface.linear_response.lr_baseclass import (
    get_num_CBS_elements,
    get_num_nonCBS,
    quantumLRBaseClass,
)
from slowquant.qiskit_interface.operators import one_elec_op_0i_0a


class quantumLR(quantumLRBaseClass):
    def run(
        self,
    ) -> None:
        """Run simulation of projected LR matrix elements."""
        print("Gs", self.num_G)

        # pre-calculate <0|G|0> and <0|HG|0>
        self.G_exp = []  # save and use for properties
        HG_exp = []
        for GJ in self.G_ops:
            self.G_exp.append(self.wf.QI.quantum_expectation_value(GJ.get_folded_operator(*self.orbs)))
            HG_exp.append(
                self.wf.QI.quantum_expectation_value((self.H_0i_0a * GJ).get_folded_operator(*self.orbs))
            )

        # Check gradients
        grad = np.zeros(self.num_G)  # G^\dagger is the same
        for i in range(self.num_G):
            grad[i] = HG_exp[i] - (self.wf.energy_elec * self.G_exp[i])
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                print("WARNING: Large Gradient detected in G of ", np.max(np.abs(grad)))

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
                self.A[i, j] = self.A[j, i] = val
                # Make B
                val = HG_exp[i] * self.G_exp[j]
                val -= self.G_exp[i] * self.G_exp[j] * self.wf.energy_elec
                self.B[i, j] = self.B[j, i] = val
                # Make Sigma
                self.Sigma[i, j] = self.Sigma[j, i] = (
                    GG_exp - (self.G_exp[i] * self.G_exp[j])
                )

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
        mux_op = one_elec_op_0i_0a(mux, self.wf.num_inactive_orbs, self.wf.num_active_orbs)
        muy_op = one_elec_op_0i_0a(muy, self.wf.num_inactive_orbs, self.wf.num_active_orbs)
        muz_op = one_elec_op_0i_0a(muz, self.wf.num_inactive_orbs, self.wf.num_active_orbs)

        transition_dipoles = np.zeros((number_excitations, 3))
        for state_number in range(number_excitations):
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

            transition_dipoles[state_number, 0] = g_part_x
            transition_dipoles[state_number, 1] = g_part_y
            transition_dipoles[state_number, 2] = g_part_z
        return transition_dipoles
