import copy
from collections.abc import Sequence

import numpy as np
import scipy.sparse as ss

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_hybrid import (
    OperatorHybrid,
    OperatorHybridData,
    convert_pauli_to_hybrid_form,
    expectation_value_hybrid_flow,
    hamiltonian_hybrid_2i_2a,
    one_elec_op_hybrid_0i_0a,
    one_elec_op_hybrid_1i_1a,
)
from slowquant.unitary_coupled_cluster.operator_pauli import epq_pauli
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

        # Overwrite Superclass
        self.q_ops: list[OperatorHybrid] = []
        for i, a in self.wf.kappa_hf_like_idx:
            op_ = 2 ** (-1 / 2) * epq_pauli(a, i, self.wf.num_spin_orbs)
            op = convert_pauli_to_hybrid_form(
                op_,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
            )
            self.q_ops.append(op)

        num_parameters = len(self.G_ops) + len(self.q_ops)
        self.A = np.zeros((num_parameters, num_parameters))
        self.B = np.zeros((num_parameters, num_parameters))
        self.Sigma = np.zeros((num_parameters, num_parameters))
        self.Delta = np.zeros((num_parameters, num_parameters))

        H_2i_2a = hamiltonian_hybrid_2i_2a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
        )

        inactive_str = "I" * self.wf.num_inactive_spin_orbs
        virtual_str = "I" * self.wf.num_virtual_spin_orbs
        self.U = OperatorHybrid(
            {inactive_str + virtual_str: OperatorHybridData(inactive_str, self.wf.u, virtual_str)}
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
                self.wf.state_vector, [self.H_0i_0a, self.U, op], self.csf
            )
            grad[i + len(self.G_ops)] = expectation_value_hybrid_flow(
                self.csf, [op.dagger, self.U.dagger, self.H_0i_0a], self.wf.state_vector
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad)))
        for j, qJ in enumerate(self.q_ops):
            for i, qI in enumerate(self.q_ops[j:], j):
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
        for j, qJ in enumerate(self.q_ops):
            for i, GI in enumerate(self.G_ops):
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
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                val = expectation_value_hybrid_flow(
                    self.csf, [GI.dagger, self.U.dagger, self.H_0i_0a, self.U, GJ], self.csf
                ) - expectation_value_hybrid_flow(
                    self.csf, [GI.dagger, GJ, self.U.dagger, self.H_0i_0a], self.wf.state_vector
                )
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make B
                self.B[i + idx_shift, j + idx_shift] = self.B[j + idx_shift, i + idx_shift] = (
                    -expectation_value_hybrid_flow(
                        self.csf, [GI.dagger, GJ.dagger, self.U.dagger, self.H_0i_0a], self.wf.state_vector
                    )
                )
                # Make Sigma
                if i == j:
                    self.Sigma[i + idx_shift, j + idx_shift] = 1

    def get_transition_dipole(self, dipole_integrals: Sequence[np.ndarray]) -> np.ndarray:
        """Calculate transition dipole moment.

        Args:
            dipole_integrals: Dipole integrals ordered as (x,y,z).

        Returns:
            Transition dipole moment.
        """
        if len(dipole_integrals) != 3:
            raise ValueError(f"Expected 3 dipole integrals got {len(dipole_integrals)}")

        mux = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[0])
        muy = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[1])
        muz = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[2])
        mux_op_G = one_elec_op_hybrid_0i_0a(
            mux, self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.wf.num_virtual_orbs
        )
        muy_op_G = one_elec_op_hybrid_0i_0a(
            muy, self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.wf.num_virtual_orbs
        )
        muz_op_G = one_elec_op_hybrid_0i_0a(
            muz, self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.wf.num_virtual_orbs
        )
        mux_op_q = one_elec_op_hybrid_1i_1a(
            mux, self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.wf.num_virtual_orbs
        )
        muy_op_q = one_elec_op_hybrid_1i_1a(
            muy, self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.wf.num_virtual_orbs
        )
        muz_op_q = one_elec_op_hybrid_1i_1a(
            muz, self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.wf.num_virtual_orbs
        )
        transition_dipoles = np.zeros((len(self.normed_response_vectors[0]), 3))
        for state_number in range(len(self.normed_response_vectors[0])):
            q_part_x = 0.0
            q_part_y = 0.0
            q_part_z = 0.0
            for i, q in enumerate(self.q_ops):
                q_part_x -= self.Z_q_normed[i, state_number] * expectation_value_hybrid_flow(
                    self.wf.state_vector, [mux_op_q, self.U, q], self.csf
                )
                q_part_x += self.Y_q_normed[i, state_number] * expectation_value_hybrid_flow(
                    self.csf, [q.dagger, self.U.dagger, mux_op_q], self.wf.state_vector
                )
                q_part_y -= self.Z_q_normed[i, state_number] * expectation_value_hybrid_flow(
                    self.wf.state_vector, [muy_op_q, self.U, q], self.csf
                )
                q_part_y += self.Y_q_normed[i, state_number] * expectation_value_hybrid_flow(
                    self.csf, [q.dagger, self.U.dagger, muy_op_q], self.wf.state_vector
                )
                q_part_z -= self.Z_q_normed[i, state_number] * expectation_value_hybrid_flow(
                    self.wf.state_vector, [muz_op_q, self.U, q], self.csf
                )
                q_part_z += self.Y_q_normed[i, state_number] * expectation_value_hybrid_flow(
                    self.csf, [q.dagger, self.U.dagger, muz_op_q], self.wf.state_vector
                )
            g_part_x = 0.0
            g_part_y = 0.0
            g_part_z = 0.0
            for i, G in enumerate(self.G_ops):
                g_part_x -= self.Z_G_normed[i, state_number] * expectation_value_hybrid_flow(
                    self.wf.state_vector, [mux_op_G, self.U, G], self.csf
                )
                g_part_x += self.Y_G_normed[i, state_number] * expectation_value_hybrid_flow(
                    self.csf, [G.dagger, self.U.dagger, mux_op_G], self.wf.state_vector
                )
                g_part_y -= self.Z_G_normed[i, state_number] * expectation_value_hybrid_flow(
                    self.wf.state_vector, [muy_op_G, self.U, G], self.csf
                )
                g_part_y += self.Y_G_normed[i, state_number] * expectation_value_hybrid_flow(
                    self.csf, [G.dagger, self.U.dagger, muy_op_G], self.wf.state_vector
                )
                g_part_z -= self.Z_G_normed[i, state_number] * expectation_value_hybrid_flow(
                    self.wf.state_vector, [muz_op_G, self.U, G], self.csf
                )
                g_part_z += self.Y_G_normed[i, state_number] * expectation_value_hybrid_flow(
                    self.csf, [G.dagger, self.U.dagger, muz_op_G], self.wf.state_vector
                )
            transition_dipoles[state_number, 0] = q_part_x + g_part_x
            transition_dipoles[state_number, 1] = q_part_y + g_part_y
            transition_dipoles[state_number, 2] = q_part_z + g_part_z
        return transition_dipoles
