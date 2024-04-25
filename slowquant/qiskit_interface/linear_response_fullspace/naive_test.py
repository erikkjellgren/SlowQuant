from collections.abc import Sequence

import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.qiskit_interface.base import FermionicOperator
from slowquant.qiskit_interface.interface import make_cliques
from slowquant.qiskit_interface.linear_response.lr_baseclass import (
    get_num_CBS_elements,
    get_num_nonCBS,
    quantumLRBaseClass,
)
from slowquant.qiskit_interface.operators import (
    commutator,
    double_commutator,
    one_elec_op_0i_0a,
)


class quantumLR(quantumLRBaseClass):
    def run(
        self,
    ) -> None:
        """Run simulation of naive LR matrix elements."""
        idx_shift = self.num_q
        print("Gs", self.num_G)

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

        # GG
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = (
                    2 * self.wf.energy_elec * self.wf.QI.quantum_expectation_value((GI.dagger * self.H_0i_0a * GJ - GI.dagger * GJ - GJ * GI.dagger + GJ * self.H_0i_0a * GI.dagger).get_folded_operators(*self.orbs))
                )
                # Make B
                self.B[i + idx_shift, j + idx_shift] = self.B[j + idx_shift, i + idx_shift] = (
                    2 * self.wf.energy_elec * self.wf.QI.quantum_expectation_value((GI.dagger * self.H_0i_0a * GJ.dagger - GI.dagger * GJ.dagger - GJ.dagger * GI.dagger + GJ.dagger * self.H_0i_0a * GI.dagger).get_folded_operators(*self.orbs))
                    )
                # Make Sigma
                self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[j + idx_shift, i + idx_shift] = (
                    self.wf.QI.quantum_expectation_value(
                        commutator(GI.dagger, GJ).get_folded_operator(*self.orbs)
                    )
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
        transition_dipole_x = 0.0
        transition_dipole_y = 0.0
        transition_dipole_z = 0.0
        transition_dipoles = np.zeros((number_excitations, 3))
        for state_number in range(number_excitations):
            transfer_op = FermionicOperator({}, {})
            for i, G in enumerate(self.G_ops):
                transfer_op += (
                    self._Z_G_normed[i, state_number] * G.dagger + self._Y_G_normed[i, state_number] * G
                )

            if self.num_G != 0:
                transition_dipole_x = self.wf.QI.quantum_expectation_value(
                    commutator(mux_op, transfer_op).get_folded_operator(*self.orbs)
                )
                transition_dipole_y = self.wf.QI.quantum_expectation_value(
                    commutator(muy_op, transfer_op).get_folded_operator(*self.orbs)
                )
                transition_dipole_z = self.wf.QI.quantum_expectation_value(
                    commutator(muz_op, transfer_op).get_folded_operator(*self.orbs)
                )
            transition_dipoles[state_number, 0] = q_part_x + transition_dipole_x
            transition_dipoles[state_number, 1] = q_part_y + transition_dipole_y
            transition_dipoles[state_number, 2] = q_part_z + transition_dipole_z

        return transition_dipoles
