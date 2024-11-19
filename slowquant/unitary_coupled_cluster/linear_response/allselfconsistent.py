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
    expectation_value_extended,
    get_indexing_extended,
    propagate_state_extended,
)
from slowquant.unitary_coupled_cluster.operators import (
    Epq,
    hamiltonian_2i_2a,
    one_elec_op_0i_0a,
    one_elec_op_1i_1a,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.util import UccStructure, UpsStructure


class LinearResponseUCC(LinearResponseBaseClass):
    index_info_extended: (
        tuple[list[int], dict[int, int], int, int, int, int, int, list[float], UpsStructure, int]
        | tuple[list[int], dict[int, int], int, int, int, int, int, list[float], UccStructure, int]
    )

    def __init__(
        self,
        wave_function: WaveFunctionUCC | WaveFunctionUPS,
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
        if isinstance(self.wf, WaveFunctionUCC):
            self.index_info_extended = (
                idx2det,
                det2idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.num_virtual_orbs,
                self.wf.num_active_elec_alpha,
                self.wf.num_active_elec_beta,
                self.wf.thetas,
                self.wf.ucc_layout,
                2,
            )
        elif isinstance(self.wf, WaveFunctionUPS):
            self.index_info_extended = (
                idx2det,
                det2idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.num_virtual_orbs,
                self.wf.num_active_elec_alpha,
                self.wf.num_active_elec_beta,
                self.wf.thetas,
                self.wf.ups_layout,
                2,
            )
        else:
            raise ValueError(f"Got incompatible wave function type, {type(self.wf)}")
        num_det = len(idx2det)
        self.csf_coeffs = np.zeros(num_det)
        hf_det = int("1" * self.wf.num_elec + "0" * (self.wf.num_spin_orbs - self.wf.num_elec), 2)
        self.csf_coeffs[det2idx[hf_det]] = 1
        self.ci_coeffs = propagate_state_extended(["U"], self.csf_coeffs, *self.index_info_extended)
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
        UdH00_ket = propagate_state_extended(["Ud", self.H_0i_0a], self.ci_coeffs, *self.index_info_extended)
        for i, op in enumerate(self.G_ops):
            G_ket = propagate_state_extended([op], self.csf_coeffs, *self.index_info_extended)
            # - <0| H U G |CSF>
            grad[i] = -expectation_value_extended(
                UdH00_ket,
                [],
                G_ket,
                *self.index_info_extended,
            )
            # <CSF| Gd Ud H |0>
            grad[i + len(self.G_ops)] = expectation_value_extended(
                G_ket,
                [],
                UdH00_ket,
                *self.index_info_extended,
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad)))
        H_ket = propagate_state_extended([H_2i_2a], self.ci_coeffs, *self.index_info_extended)
        for j, qJ in enumerate(self.q_ops):
            UdHUqJ_ket = propagate_state_extended(
                ["Ud", H_2i_2a, "U", qJ], self.csf_coeffs, *self.index_info_extended
            )
            UdH_ket = propagate_state_extended(["Ud"], H_ket, *self.index_info_extended)
            qJUdH_ket = propagate_state_extended([qJ], UdH_ket, *self.index_info_extended)
            qJdUdH_ket = propagate_state_extended([qJ.dagger], UdH_ket, *self.index_info_extended)
            for i, qI in enumerate(self.q_ops[j:], j):
                qI_ket = propagate_state_extended([qI], self.csf_coeffs, *self.index_info_extended)
                # Make A
                # <CSF| qId Ud H U qJ |CSF>
                val = expectation_value_extended(
                    qI_ket,
                    [],
                    UdHUqJ_ket,
                    *self.index_info_extended,
                )
                # - <CSF| qId qJ Ud H |0>
                val -= expectation_value_extended(
                    qI_ket,
                    [],
                    qJUdH_ket,
                    *self.index_info_extended,
                )
                self.A[i, j] = self.A[j, i] = val
                # Make B
                # - <CSF| qId qJd Ud H |0>
                val = -expectation_value_extended(
                    qI_ket,
                    [],
                    qJdUdH_ket,
                    *self.index_info_extended,
                )
                self.B[i, j] = self.B[j, i] = val
                # Make Sigma
                if i == j:
                    self.Sigma[i, j] = self.Sigma[j, i] = 1
        for j, qJ in enumerate(self.q_ops):
            UdHUq_ket = propagate_state_extended(
                ["Ud", self.H_1i_1a, "U", qJ], self.csf_coeffs, *self.index_info_extended
            )
            qdUdH_ket = propagate_state_extended(
                [qJ.dagger, "Ud", self.H_1i_1a], self.ci_coeffs, *self.index_info_extended
            )
            for i, GI in enumerate(self.G_ops):
                G_ket = propagate_state_extended([GI], self.csf_coeffs, *self.index_info_extended)
                # Make A
                # <CSF| Gd Ud H U q |CSF>
                val = expectation_value_extended(
                    G_ket,
                    [],
                    UdHUq_ket,
                    *self.index_info_extended,
                )
                self.A[i + idx_shift, j] = self.A[j, i + idx_shift] = val
                # Make B
                # - <CSF| Gd qd Ud H |0>
                val = -expectation_value_extended(
                    G_ket,
                    [],
                    qdUdH_ket,
                    *self.index_info_extended,
                )
                self.B[i + idx_shift, j] = self.B[j, i + idx_shift] = val
        for j, GJ in enumerate(self.G_ops):
            UdHUGJ_ket = propagate_state_extended(
                ["Ud", self.H_0i_0a, "U", GJ], self.csf_coeffs, *self.index_info_extended
            )
            GJUdH_ket = propagate_state_extended([GJ], UdH00_ket, *self.index_info_extended)
            GJdUdH_ket = propagate_state_extended([GJ.dagger], UdH00_ket, *self.index_info_extended)
            for i, GI in enumerate(self.G_ops[j:], j):
                GI_ket = propagate_state_extended([GI], self.csf_coeffs, *self.index_info_extended)
                # Make A
                # <CSF| GId Ud H U GJ |CSF>
                val = expectation_value_extended(
                    GI_ket,
                    [],
                    UdHUGJ_ket,
                    *self.index_info_extended,
                )
                # - <CSF| GId GJ Ud H |0>
                val -= expectation_value_extended(
                    GI_ket,
                    [],
                    GJUdH_ket,
                    *self.index_info_extended,
                )
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make B
                # - <CSF| GId GJd Ud H |0>
                val = -expectation_value_extended(
                    GI_ket,
                    [],
                    GJdUdH_ket,
                    *self.index_info_extended,
                )
                self.B[i + idx_shift, j + idx_shift] = self.B[j + idx_shift, i + idx_shift] = val
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

        mux = one_electron_integral_transform(self.wf.c_mo, dipole_integrals[0])
        muy = one_electron_integral_transform(self.wf.c_mo, dipole_integrals[1])
        muz = one_electron_integral_transform(self.wf.c_mo, dipole_integrals[2])
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
                q_part_x -= self.Z_q_normed[i, state_number] * expectation_value_extended(
                    self.ci_coeffs,
                    [mux_op_q, "U", q],
                    self.csf_coeffs,
                    *self.index_info_extended,
                )
                q_part_x += self.Y_q_normed[i, state_number] * expectation_value_extended(
                    self.csf_coeffs,
                    [q.dagger, "Ud", mux_op_q],
                    self.ci_coeffs,
                    *self.index_info_extended,
                )
                q_part_y -= self.Z_q_normed[i, state_number] * expectation_value_extended(
                    self.ci_coeffs,
                    [muy_op_q, "U", q],
                    self.csf_coeffs,
                    *self.index_info_extended,
                )
                q_part_y += self.Y_q_normed[i, state_number] * expectation_value_extended(
                    self.csf_coeffs,
                    [q.dagger, "Ud", muy_op_q],
                    self.ci_coeffs,
                    *self.index_info_extended,
                )
                q_part_z -= self.Z_q_normed[i, state_number] * expectation_value_extended(
                    self.ci_coeffs,
                    [muz_op_q, "U", q],
                    self.csf_coeffs,
                    *self.index_info_extended,
                )
                q_part_z += self.Y_q_normed[i, state_number] * expectation_value_extended(
                    self.csf_coeffs,
                    [q.dagger, "Ud", muz_op_q],
                    self.ci_coeffs,
                    *self.index_info_extended,
                )
            g_part_x = 0.0
            g_part_y = 0.0
            g_part_z = 0.0
            for i, G in enumerate(self.G_ops):
                g_part_x -= self.Z_G_normed[i, state_number] * expectation_value_extended(
                    self.ci_coeffs,
                    [mux_op_G, "U", G],
                    self.csf_coeffs,
                    *self.index_info_extended,
                )
                g_part_x += self.Y_G_normed[i, state_number] * expectation_value_extended(
                    self.csf_coeffs,
                    [G.dagger, "Ud", mux_op_G],
                    self.ci_coeffs,
                    *self.index_info_extended,
                )
                g_part_y -= self.Z_G_normed[i, state_number] * expectation_value_extended(
                    self.ci_coeffs,
                    [muy_op_G, "U", G],
                    self.csf_coeffs,
                    *self.index_info_extended,
                )
                g_part_y += self.Y_G_normed[i, state_number] * expectation_value_extended(
                    self.csf_coeffs,
                    [G.dagger, "Ud", muy_op_G],
                    self.ci_coeffs,
                    *self.index_info_extended,
                )
                g_part_z -= self.Z_G_normed[i, state_number] * expectation_value_extended(
                    self.ci_coeffs,
                    [muz_op_G, "U", G],
                    self.csf_coeffs,
                    *self.index_info_extended,
                )
                g_part_z += self.Y_G_normed[i, state_number] * expectation_value_extended(
                    self.csf_coeffs,
                    [G.dagger, "Ud", muz_op_G],
                    self.ci_coeffs,
                    *self.index_info_extended,
                )
            transition_dipoles[state_number, 0] = q_part_x + g_part_x
            transition_dipoles[state_number, 1] = q_part_y + g_part_y
            transition_dipoles[state_number, 2] = q_part_z + g_part_z
        return transition_dipoles
