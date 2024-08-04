from collections.abc import Sequence

import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.density_matrix import (
    ReducedDenstiyMatrix,
    get_orbital_gradient_response,
    get_orbital_response_property_gradient,
)
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_matrix import expectation_value
from slowquant.unitary_coupled_cluster.operators import (
    hamiltonian_2i_2a,
    one_elec_op_0i_0a,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


class LinearResponseUCC(LinearResponseBaseClass):
    def __init__(
        self,
        wave_function: WaveFunctionUCC,
        excitations: str,
    ) -> None:
        """Initialize linear response by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
        """
        super().__init__(wave_function, excitations)

        H_2i_2a = hamiltonian_2i_2a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
        )

        rdms = ReducedDenstiyMatrix(
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
            self.wf.rdm1,
            rdm2=self.wf.rdm2,
        )
        idx_shift = len(self.q_ops)
        print("Gs", len(self.G_ops))
        print("qs", len(self.q_ops))
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
                raise ValueError("Large Gradient detected in q of ", np.max(np.abs(grad)))
        grad = np.zeros(2 * len(self.G_ops))
        for i, op in enumerate(self.G_ops):
            grad[i] = expectation_value(
                self.wf.ci_coeffs,
                [self.H_0i_0a, op],
                self.wf.ci_coeffs,
                *self.index_info,
            )
            grad[i] += -self.wf.energy_elec * expectation_value(
                self.wf.ci_coeffs,
                [op],
                self.wf.ci_coeffs,
                *self.index_info,
            )
            grad[i + len(self.G_ops)] = expectation_value(
                self.wf.ci_coeffs,
                [op.dagger, self.H_0i_0a],
                self.wf.ci_coeffs,
                *self.index_info,
            )
            grad[i + len(self.G_ops)] += -self.wf.energy_elec * expectation_value(
                self.wf.ci_coeffs,
                [op.dagger],
                self.wf.ci_coeffs,
                *self.index_info,
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad)))
        for j, qJ in enumerate(self.q_ops):
            for i, qI in enumerate(self.q_ops[j:], j):
                # Make A
                val = expectation_value(
                    self.wf.ci_coeffs,
                    [qI.dagger * H_2i_2a * qJ],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                val -= (
                    expectation_value(
                        self.wf.ci_coeffs,
                        [qI.dagger * qJ],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    * self.wf.energy_elec
                )
                self.A[i, j] = self.A[j, i] = val
                # Make Sigma
                self.Sigma[i, j] = self.Sigma[j, i] = expectation_value(
                    self.wf.ci_coeffs,
                    [qI.dagger * qJ],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
        for j, qJ in enumerate(self.q_ops):
            for i, GI in enumerate(self.G_ops):
                # Make A
                self.A[j, i + idx_shift] = self.A[i + idx_shift, j] = expectation_value(
                    self.wf.ci_coeffs,
                    [GI.dagger, self.H_1i_1a * qJ],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                val = expectation_value(
                    self.wf.ci_coeffs,
                    [GI.dagger, self.H_0i_0a, GJ],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                val -= (
                    expectation_value(
                        self.wf.ci_coeffs,
                        [GI.dagger, GJ],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    * self.wf.energy_elec
                )
                val -= expectation_value(
                    self.wf.ci_coeffs,
                    [GI.dagger],
                    self.wf.ci_coeffs,
                    *self.index_info,
                ) * expectation_value(
                    self.wf.ci_coeffs,
                    [self.H_0i_0a, GJ],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                val += (
                    expectation_value(
                        self.wf.ci_coeffs,
                        [GI.dagger],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [GJ],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    * self.wf.energy_elec
                )
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make B
                val = expectation_value(
                    self.wf.ci_coeffs,
                    [GI.dagger, self.H_0i_0a],
                    self.wf.ci_coeffs,
                    *self.index_info,
                ) * expectation_value(
                    self.wf.ci_coeffs,
                    [GJ.dagger],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                val -= (
                    expectation_value(
                        self.wf.ci_coeffs,
                        [GI.dagger],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [GJ.dagger],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    * self.wf.energy_elec
                )
                self.B[i + idx_shift, j + idx_shift] = self.B[j + idx_shift, i + idx_shift] = val
                # Make Sigma
                self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[
                    j + idx_shift, i + idx_shift
                ] = expectation_value(
                    self.wf.ci_coeffs,
                    [GI.dagger, GJ],
                    self.wf.ci_coeffs,
                    *self.index_info,
                ) - (
                    expectation_value(
                        self.wf.ci_coeffs,
                        [GI.dagger],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [GJ],
                        self.wf.ci_coeffs,
                        *self.index_info,
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
        mux_op = one_elec_op_0i_0a(
            mux,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        muy_op = one_elec_op_0i_0a(
            muy,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        muz_op = one_elec_op_0i_0a(
            muz,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        transition_dipoles = np.zeros((number_excitations, 3))
        for state_number in range(number_excitations):
            q_part_x = get_orbital_response_property_gradient(
                rdms,
                mux,
                self.wf.kappa_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.normed_response_vectors,
                state_number,
                number_excitations,
            )
            q_part_y = get_orbital_response_property_gradient(
                rdms,
                muy,
                self.wf.kappa_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.normed_response_vectors,
                state_number,
                number_excitations,
            )
            q_part_z = get_orbital_response_property_gradient(
                rdms,
                muz,
                self.wf.kappa_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.normed_response_vectors,
                state_number,
                number_excitations,
            )
            g_part_x = 0.0
            g_part_y = 0.0
            g_part_z = 0.0
            for i, G in enumerate(self.G_ops):
                exp_G = expectation_value(
                    self.wf.ci_coeffs,
                    [G],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                exp_G_dagger = expectation_value(
                    self.wf.ci_coeffs,
                    [G.dagger],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                g_part_x += (
                    self.Z_G_normed[i, state_number]
                    * exp_G_dagger
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [mux_op],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                )
                g_part_x -= self.Z_G_normed[i, state_number] * expectation_value(
                    self.wf.ci_coeffs,
                    [G.dagger, mux_op],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                g_part_x -= (
                    self.Y_G_normed[i, state_number]
                    * exp_G
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [mux_op],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                )
                g_part_x += self.Y_G_normed[i, state_number] * expectation_value(
                    self.wf.ci_coeffs,
                    [mux_op, G],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                g_part_y += (
                    self.Z_G_normed[i, state_number]
                    * exp_G_dagger
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [muy_op],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                )
                g_part_y -= self.Z_G_normed[i, state_number] * expectation_value(
                    self.wf.ci_coeffs,
                    [G.dagger, muy_op],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                g_part_y -= (
                    self.Y_G_normed[i, state_number]
                    * exp_G
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [muy_op],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                )
                g_part_y += self.Y_G_normed[i, state_number] * expectation_value(
                    self.wf.ci_coeffs,
                    [muy_op, G],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                g_part_z += (
                    self.Z_G_normed[i, state_number]
                    * exp_G_dagger
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [muz_op],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                )
                g_part_z -= self.Z_G_normed[i, state_number] * expectation_value(
                    self.wf.ci_coeffs,
                    [G.dagger, muz_op],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
                g_part_z -= (
                    self.Y_G_normed[i, state_number]
                    * exp_G
                    * expectation_value(
                        self.wf.ci_coeffs,
                        [muz_op],
                        self.wf.ci_coeffs,
                        *self.index_info,
                    )
                )
                g_part_z += self.Y_G_normed[i, state_number] * expectation_value(
                    self.wf.ci_coeffs,
                    [muz_op, G],
                    self.wf.ci_coeffs,
                    *self.index_info,
                )
            transition_dipoles[state_number, 0] = q_part_x + g_part_x
            transition_dipoles[state_number, 1] = q_part_y + g_part_y
            transition_dipoles[state_number, 2] = q_part_z + g_part_z
        return transition_dipoles
