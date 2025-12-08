from collections.abc import Sequence

import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.density_matrix import (
    get_orbital_gradient_response,
    get_orbital_response_property_gradient,
)
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_state_algebra import (
    expectation_value,
    propagate_state,
)
from slowquant.unitary_coupled_cluster.operators import (
    hamiltonian_2i_2a,
    one_elec_op_0i_0a,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS


class LinearResponse(LinearResponseBaseClass):
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
        if len(self.q_ops) != 0:
            grad = get_orbital_gradient_response(  # proj-q and naive-q lead to same working equations
                self.wf.h_mo,
                self.wf.g_mo,
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.wf.rdm1,
                self.wf.rdm2,
            )
            print("idx, max(abs(grad orb)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in q of ", np.max(np.abs(grad)))
        grad = np.zeros(2 * len(self.G_ops))
        UdH00_ket = propagate_state(["Ud", self.H_0i_0a], self.wf.ci_coeffs, *self.index_info)
        for i, op in enumerate(self.G_ops):
            G_ket = propagate_state(
                [op],
                self.wf.csf_coeffs,
                *self.index_info,
            )
            # <0| H U G |CSF>
            grad[i] = -expectation_value(
                UdH00_ket,
                [],
                G_ket,
                *self.index_info,
            )
            # <CSF| Gd Ud H |0>
            grad[i + len(self.G_ops)] = expectation_value(
                G_ket,
                [],
                UdH00_ket,
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
                        self.wf.ci_coeffs, [qI.dagger * qJ], self.wf.ci_coeffs, *self.index_info
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
            UdHq_ket = propagate_state(["Ud", self.H_1i_1a * qJ], self.wf.ci_coeffs, *self.index_info)
            for i, GI in enumerate(self.G_ops):
                G_ket = propagate_state([GI], self.wf.csf_coeffs, *self.index_info)
                # Make A
                # <CSF| Gd Ud H q |0>
                val = expectation_value(
                    G_ket,
                    [],
                    UdHq_ket,
                    *self.index_info,
                )
                self.A[j, i + idx_shift] = self.A[i + idx_shift, j] = val
        for j, GJ in enumerate(self.G_ops):
            UdHUGJ_ket = propagate_state(
                ["Ud", self.H_0i_0a, "U", GJ],
                self.wf.csf_coeffs,
                *self.index_info,
            )
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                # <CSF| GId Ud H U GJ |CSF>
                val = expectation_value(
                    self.wf.csf_coeffs,
                    [GI.dagger],
                    UdHUGJ_ket,
                    *self.index_info,
                )
                if i == j:
                    val -= self.wf.energy_elec
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
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
        number_excitations = len(self.excitation_energies)
        mux = one_electron_integral_transform(self.wf.c_mo, dipole_integrals[0])
        muy = one_electron_integral_transform(self.wf.c_mo, dipole_integrals[1])
        muz = one_electron_integral_transform(self.wf.c_mo, dipole_integrals[2])
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
        Udmuxd_ket = propagate_state(["Ud", mux_op.dagger], self.wf.ci_coeffs, *self.index_info)
        Udmuyd_ket = propagate_state(["Ud", muy_op.dagger], self.wf.ci_coeffs, *self.index_info)
        Udmuzd_ket = propagate_state(["Ud", muz_op.dagger], self.wf.ci_coeffs, *self.index_info)
        Udmux_ket = propagate_state(["Ud", mux_op], self.wf.ci_coeffs, *self.index_info)
        Udmuy_ket = propagate_state(["Ud", muy_op], self.wf.ci_coeffs, *self.index_info)
        Udmuz_ket = propagate_state(["Ud", muz_op], self.wf.ci_coeffs, *self.index_info)
        transition_dipoles = np.zeros((len(self.normed_response_vectors[0]), 3))
        for state_number in range(len(self.normed_response_vectors[0])):
            q_part_x = 0.0
            q_part_y = 0.0
            q_part_z = 0.0
            if len(self.q_ops) != 0:
                q_part_x = get_orbital_response_property_gradient(
                    mux,
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.normed_response_vectors,
                    state_number,
                    number_excitations,
                )
                q_part_y = get_orbital_response_property_gradient(
                    muy,
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.normed_response_vectors,
                    state_number,
                    number_excitations,
                )
                q_part_z = get_orbital_response_property_gradient(
                    muz,
                    self.wf.kappa_no_activeactive_idx,
                    self.wf.num_inactive_orbs,
                    self.wf.num_active_orbs,
                    self.wf.rdm1,
                    self.normed_response_vectors,
                    state_number,
                    number_excitations,
                )
            g_part_x = 0.0
            g_part_y = 0.0
            g_part_z = 0.0
            for i, G in enumerate(self.G_ops):
                G_ket = propagate_state(
                    [G],
                    self.wf.csf_coeffs,
                    *self.index_info,
                )
                # -Z * <0| mux U G | CSF>
                g_part_x -= self.Z_G_normed[i, state_number] * expectation_value(
                    Udmuxd_ket,
                    [],
                    G_ket,
                    *self.index_info,
                )
                # Y * <0| Gd Ud mux | CSF>
                g_part_x += self.Y_G_normed[i, state_number] * expectation_value(
                    G_ket,
                    [],
                    Udmux_ket,
                    *self.index_info,
                )
                # -Z * <0| muy U G | CSF>
                g_part_y -= self.Z_G_normed[i, state_number] * expectation_value(
                    Udmuyd_ket,
                    [],
                    G_ket,
                    *self.index_info,
                )
                # Y * <0| Gd Ud muy | CSF>
                g_part_y += self.Y_G_normed[i, state_number] * expectation_value(
                    G_ket,
                    [],
                    Udmuy_ket,
                    *self.index_info,
                )
                # -Z * <0| muz U G | CSF>
                g_part_z -= self.Z_G_normed[i, state_number] * expectation_value(
                    Udmuzd_ket,
                    [],
                    G_ket,
                    *self.index_info,
                )
                # Y * <0| Gd Ud muz | CSF>
                g_part_z += self.Y_G_normed[i, state_number] * expectation_value(
                    G_ket,
                    [],
                    Udmuz_ket,
                    *self.index_info,
                )
            transition_dipoles[state_number, 0] = q_part_x + g_part_x
            transition_dipoles[state_number, 1] = q_part_y + g_part_y
            transition_dipoles[state_number, 2] = q_part_z + g_part_z
        return transition_dipoles
