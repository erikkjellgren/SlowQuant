from collections.abc import Sequence

import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_extended import (
    construct_ucc_u_extended,
    expectation_value_propagate_extended,
    get_indexing_extended,
)
from slowquant.unitary_coupled_cluster.operators import (
    Epq,
    hamiltonian_2i_2a,
    one_elec_op_0i_0a,
    one_elec_op_1i_1a,
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
        # Overwrite Superclass
        idx2det, det2idx = get_indexing_extended(
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
            self.wf.num_active_elec_alpha,
            self.wf.num_active_elec_beta,
            2,
        )
        self.index_info_extended = (
            idx2det,
            det2idx,
            self.wf.num_orbs,
        )
        thetas = []
        if "s" in self.wf._excitations:
            thetas += self.wf.theta1
        if "d" in self.wf._excitations:
            thetas += self.wf.theta2
        if "t" in self.wf._excitations:
            thetas += self.wf.theta3
        if "q" in self.wf._excitations:
            thetas += self.wf.theta4
        if "5" in self.wf._excitations:
            thetas += self.wf.theta5
        if "6" in self.wf._excitations:
            thetas += self.wf.theta6
        self.u = construct_ucc_u_extended(
            len(idx2det),
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
            self.wf.num_active_elec_alpha,
            self.wf.num_active_elec_beta,
            thetas,
            self.wf.singlet_excitation_operator_generator,
            self.wf._excitations,
            2,
        )
        num_det = len(idx2det)
        self.csf_coeffs = np.zeros(num_det)
        hf_det = int("1" * self.wf.num_elec + "0" * (self.wf.num_spin_orbs - self.wf.num_elec), 2)
        self.csf_coeffs[det2idx[hf_det]] = 1
        self.ci_coeffs = np.matmul(self.u, self.csf_coeffs)
        self.q_ops: list[FermionicOperator] = []
        for i, a in self.wf.kappa_hf_like_idx:
            op = 2 ** (-1 / 2) * Epq(a, i)
            self.q_ops.append(op)

        num_parameters = len(self.G_ops) + len(self.q_ops)
        self.A = np.zeros((num_parameters, num_parameters))
        self.B = np.zeros((num_parameters, num_parameters))
        self.Sigma = np.zeros((num_parameters, num_parameters))
        self.Delta = np.zeros((num_parameters, num_parameters))

        H_2i_2a = hamiltonian_2i_2a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
        )

        idx_shift = len(self.q_ops)
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
            grad[i] = -expectation_value_propagate_extended(
                self.ci_coeffs,
                [self.H_0i_0a, self.u, op],
                self.csf_coeffs,
                *self.index_info_extended,
            )
            grad[i + len(self.G_ops)] = expectation_value_propagate_extended(
                self.csf_coeffs,
                [op.dagger, self.u.conjugate().transpose(), self.H_0i_0a],
                self.ci_coeffs,
                *self.index_info_extended,
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad)))
        for j, qJ in enumerate(self.q_ops):
            for i, qI in enumerate(self.q_ops[j:], j):
                # Make A
                val = expectation_value_propagate_extended(
                    self.csf_coeffs,
                    [qI.dagger, self.u.conjugate().transpose(), H_2i_2a, self.u, qJ],
                    self.csf_coeffs,
                    *self.index_info_extended,
                )
                val -= expectation_value_propagate_extended(
                    self.csf_coeffs,
                    [qI.dagger, qJ, self.u.conjugate().transpose(), H_2i_2a],
                    self.ci_coeffs,
                    *self.index_info_extended,
                )
                self.A[i, j] = self.A[j, i] = val
                # Make B
                self.B[i, j] = self.B[j, i] = -expectation_value_propagate_extended(
                    self.csf_coeffs,
                    [qI.dagger, qJ.dagger, self.u.conjugate().transpose(), H_2i_2a],
                    self.ci_coeffs,
                    *self.index_info_extended,
                )
                # Make Sigma
                if i == j:
                    self.Sigma[i, j] = self.Sigma[j, i] = 1
        for j, qJ in enumerate(self.q_ops):
            for i, GI in enumerate(self.G_ops):
                # Make A
                val = expectation_value_propagate_extended(
                    self.csf_coeffs,
                    [GI.dagger, self.u.conjugate().transpose(), self.H_1i_1a, self.u, qJ],
                    self.csf_coeffs,
                    *self.index_info_extended,
                )
                self.A[i + idx_shift, j] = self.A[j, i + idx_shift] = val
                # Make B
                self.B[i + idx_shift, j] = self.B[j, i + idx_shift] = -expectation_value_propagate_extended(
                    self.csf_coeffs,
                    [GI.dagger, qJ.dagger, self.u.conjugate().transpose(), self.H_1i_1a],
                    self.ci_coeffs,
                    *self.index_info_extended,
                )
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                val = expectation_value_propagate_extended(
                    self.csf_coeffs,
                    [GI.dagger, self.u.conjugate().transpose(), self.H_0i_0a, self.u, GJ],
                    self.csf_coeffs,
                    *self.index_info_extended,
                ) - expectation_value_propagate_extended(
                    self.csf_coeffs,
                    [GI.dagger, GJ, self.u.conjugate().transpose(), self.H_0i_0a],
                    self.ci_coeffs,
                    *self.index_info_extended,
                )
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make B
                self.B[i + idx_shift, j + idx_shift] = self.B[j + idx_shift, i + idx_shift] = (
                    -expectation_value_propagate_extended(
                        self.csf_coeffs,
                        [GI.dagger, GJ.dagger, self.u.conjugate().transpose(), self.H_0i_0a],
                        self.ci_coeffs,
                        *self.index_info_extended,
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
        mux_op_G = one_elec_op_0i_0a(
            mux,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        muy_op_G = one_elec_op_0i_0a(
            muy,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        muz_op_G = one_elec_op_0i_0a(
            muz,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        mux_op_q = one_elec_op_1i_1a(
            mux, self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.wf.num_virtual_orbs
        )
        muy_op_q = one_elec_op_1i_1a(
            muy, self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.wf.num_virtual_orbs
        )
        muz_op_q = one_elec_op_1i_1a(
            muz, self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.wf.num_virtual_orbs
        )
        transition_dipoles = np.zeros((len(self.normed_response_vectors[0]), 3))
        for state_number in range(len(self.normed_response_vectors[0])):
            q_part_x = 0.0
            q_part_y = 0.0
            q_part_z = 0.0
            for i, q in enumerate(self.q_ops):
                q_part_x -= self.Z_q_normed[i, state_number] * expectation_value_propagate_extended(
                    self.ci_coeffs,
                    [mux_op_q, self.u, q],
                    self.csf_coeffs,
                    *self.index_info_extended,
                )
                q_part_x += self.Y_q_normed[i, state_number] * expectation_value_propagate_extended(
                    self.csf_coeffs,
                    [q.dagger, self.u.conjugate().transpose(), mux_op_q],
                    self.ci_coeffs,
                    *self.index_info_extended,
                )
                q_part_y -= self.Z_q_normed[i, state_number] * expectation_value_propagate_extended(
                    self.ci_coeffs,
                    [muy_op_q, self.u, q],
                    self.csf_coeffs,
                    *self.index_info_extended,
                )
                q_part_y += self.Y_q_normed[i, state_number] * expectation_value_propagate_extended(
                    self.csf_coeffs,
                    [q.dagger, self.u.conjugate().transpose(), muy_op_q],
                    self.ci_coeffs,
                    *self.index_info_extended,
                )
                q_part_z -= self.Z_q_normed[i, state_number] * expectation_value_propagate_extended(
                    self.ci_coeffs,
                    [muz_op_q, self.u, q],
                    self.csf_coeffs,
                    *self.index_info_extended,
                )
                q_part_z += self.Y_q_normed[i, state_number] * expectation_value_propagate_extended(
                    self.csf_coeffs,
                    [q.dagger, self.u.conjugate().transpose(), muz_op_q],
                    self.ci_coeffs,
                    *self.index_info_extended,
                )
            g_part_x = 0.0
            g_part_y = 0.0
            g_part_z = 0.0
            for i, G in enumerate(self.G_ops):
                g_part_x -= self.Z_G_normed[i, state_number] * expectation_value_propagate_extended(
                    self.ci_coeffs,
                    [mux_op_G, self.u, G],
                    self.csf_coeffs,
                    *self.index_info_extended,
                )
                g_part_x += self.Y_G_normed[i, state_number] * expectation_value_propagate_extended(
                    self.csf_coeffs,
                    [G.dagger, self.u.conjugate().transpose(), mux_op_G],
                    self.ci_coeffs,
                    *self.index_info_extended,
                )
                g_part_y -= self.Z_G_normed[i, state_number] * expectation_value_propagate_extended(
                    self.ci_coeffs,
                    [muy_op_G, self.u, G],
                    self.csf_coeffs,
                    *self.index_info_extended,
                )
                g_part_y += self.Y_G_normed[i, state_number] * expectation_value_propagate_extended(
                    self.csf_coeffs,
                    [G.dagger, self.u.conjugate().transpose(), muy_op_G],
                    self.ci_coeffs,
                    *self.index_info_extended,
                )
                g_part_z -= self.Z_G_normed[i, state_number] * expectation_value_propagate_extended(
                    self.ci_coeffs,
                    [muz_op_G, self.u, G],
                    self.csf_coeffs,
                    *self.index_info_extended,
                )
                g_part_z += self.Y_G_normed[i, state_number] * expectation_value_propagate_extended(
                    self.csf_coeffs,
                    [G.dagger, self.u.conjugate().transpose(), muz_op_G],
                    self.ci_coeffs,
                    *self.index_info_extended,
                )
            transition_dipoles[state_number, 0] = q_part_x + g_part_x
            transition_dipoles[state_number, 1] = q_part_y + g_part_y
            transition_dipoles[state_number, 2] = q_part_z + g_part_z
        return transition_dipoles
