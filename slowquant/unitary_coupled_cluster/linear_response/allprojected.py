import copy
from collections.abc import Sequence

import numpy as np
import scipy.sparse as ss

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.density_matrix import (
    ReducedDenstiyMatrix,
    get_orbital_gradient_response,
    get_orbital_response_property_gradient,
    get_orbital_response_vector_norm,
)
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_hybrid import (
    OperatorHybrid,
    expectation_value_hybrid_flow,
    hamiltonian_hybrid_2i_2a,
    one_elec_op_hybrid_0i_0a,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


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

        H_2i_2a = hamiltonian_hybrid_2i_2a(
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
        self.csf = copy.deepcopy(self.wf.state_vector)
        self.csf.active = self.csf._active
        self.csf.active_csr = ss.csr_matrix(self.csf._active)
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
            grad[i] = expectation_value_hybrid_flow(
                self.wf.state_vector, [self.H_0i_0a, op], self.wf.state_vector
            ) - (
                self.wf.energy_elec
                * expectation_value_hybrid_flow(self.wf.state_vector, [op], self.wf.state_vector)
            )
            grad[i + len(self.G_ops)] = expectation_value_hybrid_flow(
                self.wf.state_vector, [op.dagger, self.H_0i_0a], self.wf.state_vector
            ) - (
                self.wf.energy_elec
                * expectation_value_hybrid_flow(self.wf.state_vector, [op.dagger], self.wf.state_vector)
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad)))
        for j, qJ in enumerate(self.q_ops):
            for i, qI in enumerate(self.q_ops[j:], j):
                # Make A
                val = expectation_value_hybrid_flow(
                    self.wf.state_vector, [qI.dagger, H_2i_2a, qJ], self.wf.state_vector
                )
                val -= (
                    expectation_value_hybrid_flow(self.wf.state_vector, [qI.dagger, qJ], self.wf.state_vector)
                    * self.wf.energy_elec
                )
                self.A[i, j] = self.A[j, i] = val
                # Make Sigma
                self.Sigma[i, j] = self.Sigma[j, i] = expectation_value_hybrid_flow(
                    self.wf.state_vector, [qI.dagger, qJ], self.wf.state_vector
                )
        for j, qJ in enumerate(self.q_ops):
            for i, GI in enumerate(self.G_ops):
                # Make A
                self.A[j, i + idx_shift] = self.A[i + idx_shift, j] = expectation_value_hybrid_flow(
                    self.wf.state_vector, [GI.dagger, self.H_1i_1a, qJ], self.wf.state_vector
                )
                # Make B
                self.B[j, i + idx_shift] = self.B[i + idx_shift, j] = -expectation_value_hybrid_flow(
                    self.wf.state_vector, [GI.dagger, qJ.dagger, self.H_1i_1a], self.wf.state_vector
                )
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                val = expectation_value_hybrid_flow(
                    self.wf.state_vector, [GI.dagger, self.H_0i_0a, GJ], self.wf.state_vector
                )
                val -= (
                    expectation_value_hybrid_flow(self.wf.state_vector, [GI.dagger, GJ], self.wf.state_vector)
                    * self.wf.energy_elec
                )
                val -= expectation_value_hybrid_flow(
                    self.wf.state_vector, [GI.dagger], self.wf.state_vector
                ) * expectation_value_hybrid_flow(
                    self.wf.state_vector, [self.H_0i_0a, GJ], self.wf.state_vector
                )
                val += (
                    expectation_value_hybrid_flow(self.wf.state_vector, [GI.dagger], self.wf.state_vector)
                    * expectation_value_hybrid_flow(self.wf.state_vector, [GJ], self.wf.state_vector)
                    * self.wf.energy_elec
                )
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make B
                val = expectation_value_hybrid_flow(
                    self.wf.state_vector, [GI.dagger, self.H_0i_0a], self.wf.state_vector
                ) * expectation_value_hybrid_flow(self.wf.state_vector, [GJ.dagger], self.wf.state_vector)
                val -= (
                    expectation_value_hybrid_flow(self.wf.state_vector, [GI.dagger], self.wf.state_vector)
                    * expectation_value_hybrid_flow(self.wf.state_vector, [GJ.dagger], self.wf.state_vector)
                    * self.wf.energy_elec
                )
                self.B[i + idx_shift, j + idx_shift] = self.B[j + idx_shift, i + idx_shift] = val
                # Make Sigma
                self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[
                    j + idx_shift, i + idx_shift
                ] = expectation_value_hybrid_flow(
                    self.wf.state_vector, [GI.dagger, GJ], self.wf.state_vector
                ) - (
                    expectation_value_hybrid_flow(self.wf.state_vector, [GI.dagger], self.wf.state_vector)
                    * expectation_value_hybrid_flow(self.wf.state_vector, [GJ], self.wf.state_vector)
                )

    def get_excited_state_norm(self) -> np.ndarray:
        """Calculate the norm of excited state.

        Returns:
            Norm of excited state.
        """
        number_excitations = len(self.excitation_energies)
        rdms = ReducedDenstiyMatrix(
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
            self.wf.rdm1,
            rdm2=self.wf.rdm2,
        )
        norms = np.zeros(len(self.response_vectors[0]))
        for state_number in range(len(self.response_vectors[0])):
            q_part = get_orbital_response_vector_norm(
                rdms, self.wf.kappa_idx, self.response_vectors, state_number, number_excitations
            )
            g_part = 0
            for i, GI in enumerate(self.G_ops):
                for j, GJ in enumerate(self.G_ops):
                    g_part += (
                        self.Z_G[i, state_number]
                        * self.Z_G[j, state_number]
                        * expectation_value_hybrid_flow(
                            self.wf.state_vector, [GI.dagger, GJ], self.wf.state_vector
                        )
                    )
                    g_part -= (
                        self.Y_G[i, state_number]
                        * self.Y_G[j, state_number]
                        * expectation_value_hybrid_flow(
                            self.wf.state_vector, [GJ.dagger, GI], self.wf.state_vector
                        )
                    )
                    g_part -= (
                        self.Z_G[i, state_number]
                        * self.Z_G[j, state_number]
                        * expectation_value_hybrid_flow(self.wf.state_vector, [GJ], self.wf.state_vector)
                        * expectation_value_hybrid_flow(
                            self.wf.state_vector, [GI.dagger], self.wf.state_vector
                        )
                    )
                    g_part += (
                        self.Y_G[i, state_number]
                        * self.Y_G[j, state_number]
                        * expectation_value_hybrid_flow(
                            self.wf.state_vector, [GJ.dagger], self.wf.state_vector
                        )
                        * expectation_value_hybrid_flow(self.wf.state_vector, [GI], self.wf.state_vector)
                    )

            norms[state_number] = q_part + g_part
        return norms

    def get_excited_state_overlap(self) -> np.ndarray:
        """Calculate the overlap of excited states with the ground state

        Returns:
            Overlap of excited states with ground state
        """

        overlaps = np.zeros(len(self.Z_G_normed[0]))
        for state_number in range(len(self.Z_G_normed[0])):
            transfer_op = OperatorHybrid({})
            for i, G in enumerate(self.G_ops):
                transfer_op += (
                    self.Z_G_normed[i, state_number] * G.dagger + self.Y_G_normed[i, state_number] * G
                )
            overlaps[state_number] = expectation_value_hybrid_flow(
                self.wf.state_vector, [transfer_op], self.wf.state_vector
            )
        return overlaps

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
        mux_op = one_elec_op_hybrid_0i_0a(
            mux, self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.wf.num_virtual_orbs
        )
        muy_op = one_elec_op_hybrid_0i_0a(
            muy, self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.wf.num_virtual_orbs
        )
        muz_op = one_elec_op_hybrid_0i_0a(
            muz, self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.wf.num_virtual_orbs
        )
        transition_dipoles = np.zeros((len(self.normed_response_vectors[0]), 3))
        for state_number in range(len(self.normed_response_vectors[0])):
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
                exp_G = expectation_value_hybrid_flow(self.wf.state_vector, [G], self.wf.state_vector)
                exp_G_dagger = expectation_value_hybrid_flow(
                    self.wf.state_vector, [G.dagger], self.wf.state_vector
                )
                g_part_x += (
                    self.Z_G_normed[i, state_number]
                    * exp_G_dagger
                    * expectation_value_hybrid_flow(self.wf.state_vector, [mux_op], self.wf.state_vector)
                )
                g_part_x -= self.Z_G_normed[i, state_number] * expectation_value_hybrid_flow(
                    self.wf.state_vector, [G.dagger, mux_op], self.wf.state_vector
                )
                g_part_x -= (
                    self.Y_G_normed[i, state_number]
                    * exp_G
                    * expectation_value_hybrid_flow(self.wf.state_vector, [mux_op], self.wf.state_vector)
                )
                g_part_x += self.Y_G_normed[i, state_number] * expectation_value_hybrid_flow(
                    self.wf.state_vector, [mux_op, G], self.wf.state_vector
                )
                g_part_y += (
                    self.Z_G_normed[i, state_number]
                    * exp_G_dagger
                    * expectation_value_hybrid_flow(self.wf.state_vector, [muy_op], self.wf.state_vector)
                )
                g_part_y -= self.Z_G_normed[i, state_number] * expectation_value_hybrid_flow(
                    self.wf.state_vector, [G.dagger, muy_op], self.wf.state_vector
                )
                g_part_y -= (
                    self.Y_G_normed[i, state_number]
                    * exp_G
                    * expectation_value_hybrid_flow(self.wf.state_vector, [muy_op], self.wf.state_vector)
                )
                g_part_y += self.Y_G_normed[i, state_number] * expectation_value_hybrid_flow(
                    self.wf.state_vector, [muy_op, G], self.wf.state_vector
                )
                g_part_z += (
                    self.Z_G_normed[i, state_number]
                    * exp_G_dagger
                    * expectation_value_hybrid_flow(self.wf.state_vector, [muz_op], self.wf.state_vector)
                )
                g_part_z -= self.Z_G_normed[i, state_number] * expectation_value_hybrid_flow(
                    self.wf.state_vector, [G.dagger, muz_op], self.wf.state_vector
                )
                g_part_z -= (
                    self.Y_G_normed[i, state_number]
                    * exp_G
                    * expectation_value_hybrid_flow(self.wf.state_vector, [muz_op], self.wf.state_vector)
                )
                g_part_z += self.Y_G_normed[i, state_number] * expectation_value_hybrid_flow(
                    self.wf.state_vector, [muz_op, G], self.wf.state_vector
                )
            transition_dipoles[state_number, 0] = q_part_x + g_part_x
            transition_dipoles[state_number, 1] = q_part_y + g_part_y
            transition_dipoles[state_number, 2] = q_part_z + g_part_z
        return transition_dipoles
