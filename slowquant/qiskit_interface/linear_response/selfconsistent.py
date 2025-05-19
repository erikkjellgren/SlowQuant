from collections.abc import Sequence

import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.qiskit_interface.linear_response.lr_baseclass import quantumLRBaseClass
from slowquant.unitary_coupled_cluster.fermionic_operator import (
    get_determinant_expansion_from_operator_on_HF,
)
from slowquant.unitary_coupled_cluster.operators import one_elec_op_0i_0a


class quantumLR(quantumLRBaseClass):
    def run(
        self,
        skip_orbital_rotations: bool = False,
        do_gradients: bool = True,
    ) -> None:
        """Run simulation of naive LR matrix elements.

        Args:
            skip_orbital_rotations: Skip orbital rotations in the linear response equations.
            do_gradients: Calculate gradients w.r.t. orbital rotations and active space excitations.
        """
        print("Gs", self.num_G)
        self.A = np.zeros((self.num_G, self.num_G))
        self.B = np.zeros((self.num_G, self.num_G))
        self.Sigma = np.zeros((self.num_G, self.num_G))
        self.states = {}
        hf_det = ""
        for i in range(2 * self.wf.num_active_orbs):
            if i % 2 == 0 and i // 2 < self.wf.num_active_elec_alpha:
                hf_det += "1"
                continue
            if i % 2 == 1 and i // 2 < self.wf.num_active_elec_beta:
                hf_det += "1"
                continue
            hf_det += "0"
        self.states = {"HF": ([1.0], [hf_det])}
        for i, G in enumerate(self.G_ops):
            coeffs, dets = get_determinant_expansion_from_operator_on_HF(
                G.get_folded_operator(*self.orbs),
                self.wf.num_active_orbs,
                self.wf.num_active_elec_alpha,
                self.wf.num_active_elec_beta,
            )
            self.states[f"G{i}"] = (coeffs, dets)
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                coeffs, dets = get_determinant_expansion_from_operator_on_HF(
                    GJ.get_folded_operator(*self.orbs) * GI.get_folded_operator(*self.orbs),
                    self.wf.num_active_orbs,
                    self.wf.num_active_elec_alpha,
                    self.wf.num_active_elec_beta,
                )
                self.states[f"G{j}G{i}"] = (coeffs, dets)
                coeffs, dets = get_determinant_expansion_from_operator_on_HF(
                    (GJ.dagger).get_folded_operator(*self.orbs) * GI.get_folded_operator(*self.orbs),
                    self.wf.num_active_orbs,
                    self.wf.num_active_elec_alpha,
                    self.wf.num_active_elec_beta,
                )
                self.states[f"Gd{j}G{i}"] = (coeffs, dets)

        if self.num_q != 0 and not skip_orbital_rotations:
            raise NotImplementedError(
                "Found orbital rotations and skip_orbital_rotations is set to False. Self-consistent is only implemented for active space parameters."
            )
        H_active = self.H_0i_0a.get_folded_operator(*self.orbs)
        if do_gradients:
            grad = np.zeros(2 * self.num_G)
            for i in range(self.num_G):
                # <CSF| Ud H U G |CSF>
                grad[i] = self.wf.QI.quantum_expectation_value_csfs(
                    self.states["HF"], H_active, self.states[f"G{i}"]
                )
                # <CSF| Gd Ud H U |HF>
                grad[i + self.num_G] = self.wf.QI.quantum_expectation_value_csfs(
                    self.states[f"G{i}"], H_active, self.states["HF"]
                )
            if len(grad) != 0:
                print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
                if np.max(np.abs(grad)) > 10**-3:
                    print("WARNING: Large Gradient detected in G of ", np.max(np.abs(grad)))

        # GG
        for j in range(self.num_G):
            for i in range(j, self.num_G):
                # Make A
                # <CSF| GId Ud H U GJ |CSF>
                val = self.wf.QI.quantum_expectation_value_csfs(
                    self.states[f"G{i}"], H_active, self.states[f"G{j}"]
                )
                # - <CSF| GId GJ Ud H U |CSF>
                if self.states[f"Gd{j}G{i}"] != ([], []):
                    val -= self.wf.QI.quantum_expectation_value_csfs(
                        self.states[f"Gd{j}G{i}"], H_active, self.states["HF"]
                    )
                self.A[i, j] = self.A[j, i] = val
                # Make B
                # - <CSF| GId GJd Ud H U |CSF>
                val = 0.0
                if self.states[f"G{j}G{i}"] != ([], []):
                    val -= self.wf.QI.quantum_expectation_value_csfs(
                        self.states[f"G{j}G{i}"], H_active, self.states["HF"]
                    )
                self.B[i, j] = self.B[j, i] = val
                # Make Sigma
                if i == j:
                    self.Sigma[i, j] = 1

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
        mux_active = mux_op.get_folded_operator(*self.orbs)
        muy_active = muy_op.get_folded_operator(*self.orbs)
        muz_active = muz_op.get_folded_operator(*self.orbs)
        transition_dipoles = np.zeros((number_excitations, 3))
        for state_number in range(number_excitations):
            g_part_x = 0.0
            g_part_y = 0.0
            g_part_z = 0.0
            for i in range(self.num_G):
                # -Z * <CSF| Ud mux U G | CSF>
                g_part_x -= self._Z_G_normed[i, state_number] * self.wf.QI.quantum_expectation_value_csfs(
                    self.states["HF"], mux_active, self.states[f"G{i}"]
                )
                # Y * <CSF| Gd Ud mux U | CSF>
                g_part_x += self._Y_G_normed[i, state_number] * self.wf.QI.quantum_expectation_value_csfs(
                    self.states[f"G{i}"], mux_active, self.states["HF"]
                )
                # -Z * <CSF| Ud muy U G | CSF>
                g_part_y -= self._Z_G_normed[i, state_number] * self.wf.QI.quantum_expectation_value_csfs(
                    self.states["HF"], muy_active, self.states[f"G{i}"]
                )
                # Y * <CSF| Gd Ud muy U | CSF>
                g_part_y += self._Y_G_normed[i, state_number] * self.wf.QI.quantum_expectation_value_csfs(
                    self.states[f"G{i}"], muy_active, self.states["HF"]
                )
                # -Z * <CSF| Ud muz U G | CSF>
                g_part_z -= self._Z_G_normed[i, state_number] * self.wf.QI.quantum_expectation_value_csfs(
                    self.states["HF"], muz_active, self.states[f"G{i}"]
                )
                # Y * <CSF| Gd Ud muz U | CSF>
                g_part_z += self._Y_G_normed[i, state_number] * self.wf.QI.quantum_expectation_value_csfs(
                    self.states[f"G{i}"], muz_active, self.states["HF"]
                )
            transition_dipoles[state_number, 0] = g_part_x
            transition_dipoles[state_number, 1] = g_part_y
            transition_dipoles[state_number, 2] = g_part_z
        return transition_dipoles
