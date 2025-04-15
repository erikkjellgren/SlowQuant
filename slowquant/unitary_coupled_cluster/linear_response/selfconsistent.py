from collections.abc import Sequence

import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.density_matrix import (
    ReducedDenstiyMatrix,
    get_orbital_gradient_response,
    get_orbital_response_hessian_block,
    get_orbital_response_metric_sigma,
    get_orbital_response_property_gradient,
)
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_extended import (
    expectation_value_extended,
    get_indexing_extended,
    propagate_state_extended,
)
from slowquant.unitary_coupled_cluster.operators import (
    one_elec_op_0i_0a,
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
            1,
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
                1,
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
                1,
            )
        else:
            raise ValueError(f"Got incompatible wave function type, {type(self.wf)}")
        num_det = len(idx2det)
        self.csf_coeffs = np.zeros(num_det)
        hf_det = int("1" * self.wf.num_elec + "0" * (self.wf.num_spin_orbs - self.wf.num_elec), 2)
        self.csf_coeffs[det2idx[hf_det]] = 1
        self.ci_coeffs = propagate_state_extended(["U"], self.csf_coeffs, *self.index_info_extended)

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
        grad = get_orbital_gradient_response(
            rdms,
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.kappa_no_activeactive_idx,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        if len(grad) != 0:
            print("idx, max(abs(grad orb)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in q of ", np.max(np.abs(grad)))
        grad = np.zeros(2 * len(self.G_ops))
        UdH00_ket = propagate_state_extended(["Ud", self.H_0i_0a], self.ci_coeffs, *self.index_info_extended)
        for i, op in enumerate(self.G_ops):
            G_ket = propagate_state_extended(
                [op],
                self.csf_coeffs,
                *self.index_info_extended,
            )
            # <0| H U G |CSF>
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
        # Do orbital-orbital blocks
        self.A[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_hessian_block(
            rdms,
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.kappa_no_activeactive_idx_dagger,
            self.wf.kappa_no_activeactive_idx,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        self.B[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_hessian_block(
            rdms,
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.kappa_no_activeactive_idx_dagger,
            self.wf.kappa_no_activeactive_idx_dagger,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        self.Sigma[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_metric_sigma(
            rdms, self.wf.kappa_no_activeactive_idx
        )
        for j, qJ in enumerate(self.q_ops):
            UdHq_ket = propagate_state_extended(
                ["Ud", self.H_1i_1a, qJ], self.ci_coeffs, *self.index_info_extended
            )
            UdqdH_ket = propagate_state_extended(
                ["Ud", qJ.dagger, self.H_1i_1a], self.ci_coeffs, *self.index_info_extended
            )
            for i, GI in enumerate(self.G_ops):
                G_ket = propagate_state_extended(
                    [GI],
                    self.csf_coeffs,
                    *self.index_info_extended,
                )
                # Make A
                # <CSF| Gd Ud H q |0>
                val = expectation_value_extended(
                    G_ket,
                    [],
                    UdHq_ket,
                    *self.index_info_extended,
                )
                # -1/2<0| H U Gd Ud q |0>
                val -= (
                    1
                    / 2
                    * expectation_value_extended(
                        self.ci_coeffs,
                        [self.H_1i_1a, "U", GI.dagger, "Ud", qJ],
                        self.ci_coeffs,
                        *self.index_info_extended,
                    )
                )
                self.A[i + idx_shift, j] = self.A[j, i + idx_shift] = val
                # Make B
                # - 1/2<CSF| Gd Ud qd H |0>
                val = (
                    -1
                    / 2
                    * expectation_value_extended(
                        G_ket,
                        [],
                        UdqdH_ket,
                        *self.index_info_extended,
                    )
                )
                # - 1/2<0| qd U Gd Ud H |0>
                val -= (
                    1
                    / 2
                    * expectation_value_extended(
                        self.ci_coeffs,
                        [qJ.dagger, "U", GI.dagger, "Ud", self.H_1i_1a],
                        self.ci_coeffs,
                        *self.index_info_extended,
                    )
                )
                self.B[i + idx_shift, j] = self.B[j, i + idx_shift] = val
        for j, GJ in enumerate(self.G_ops):
            UdHUGJ_ket = propagate_state_extended(
                ["Ud", self.H_0i_0a, "U", GJ],
                self.csf_coeffs,
                *self.index_info_extended,
            )
            GJUdH_ket = propagate_state_extended(
                [GJ],
                UdH00_ket,
                *self.index_info_extended,
            )
            GJdUdH_ket = propagate_state_extended(
                [GJ.dagger],
                UdH00_ket,
                *self.index_info_extended,
            )
            for i, GI in enumerate(self.G_ops[j:], j):
                GI_ket = propagate_state_extended(
                    [GI],
                    self.csf_coeffs,
                    *self.index_info_extended,
                )
                # Make A
                # <CSF| GId Ud H U GJ |CSF>
                val = expectation_value_extended(
                    GI_ket,
                    [],
                    UdHUGJ_ket,
                    *self.index_info_extended,
                )
                # - 1/2<CSF| GId GJ Ud H |0>
                val -= (
                    1
                    / 2
                    * expectation_value_extended(
                        GI_ket,
                        [],
                        GJUdH_ket,
                        *self.index_info_extended,
                    )
                )
                # - 1/2<0| H U GId GJ |CSF>
                val -= (
                    1
                    / 2
                    * expectation_value_extended(
                        UdH00_ket,
                        [GI.dagger, GJ],
                        self.csf_coeffs,
                        *self.index_info_extended,
                    )
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
        number_excitations = len(self.excitation_energies)
        rdms = ReducedDenstiyMatrix(
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
            self.wf.rdm1,
            rdm2=self.wf.rdm2,
        )
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
        Udmuxd_ket = propagate_state_extended(
            ["Ud", mux_op.dagger], self.ci_coeffs, *self.index_info_extended
        )
        Udmuyd_ket = propagate_state_extended(
            ["Ud", muy_op.dagger], self.ci_coeffs, *self.index_info_extended
        )
        Udmuzd_ket = propagate_state_extended(
            ["Ud", muz_op.dagger], self.ci_coeffs, *self.index_info_extended
        )
        Udmux_ket = propagate_state_extended(["Ud", mux_op], self.ci_coeffs, *self.index_info_extended)
        Udmuy_ket = propagate_state_extended(["Ud", muy_op], self.ci_coeffs, *self.index_info_extended)
        Udmuz_ket = propagate_state_extended(["Ud", muz_op], self.ci_coeffs, *self.index_info_extended)
        transition_dipoles = np.zeros((len(self.normed_response_vectors[0]), 3))
        for state_number in range(len(self.normed_response_vectors[0])):
            q_part_x = get_orbital_response_property_gradient(
                rdms,
                mux,
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.normed_response_vectors,
                state_number,
                number_excitations,
            )
            q_part_y = get_orbital_response_property_gradient(
                rdms,
                muy,
                self.wf.kappa_no_activeactive_idx,
                self.wf.num_inactive_orbs,
                self.wf.num_active_orbs,
                self.normed_response_vectors,
                state_number,
                number_excitations,
            )
            q_part_z = get_orbital_response_property_gradient(
                rdms,
                muz,
                self.wf.kappa_no_activeactive_idx,
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
                G_ket = propagate_state_extended(
                    [G],
                    self.csf_coeffs,
                    *self.index_info_extended,
                )
                # -Z * <0| mux U G | CSF>
                g_part_x -= self.Z_G_normed[i, state_number] * expectation_value_extended(
                    Udmuxd_ket,
                    [],
                    G_ket,
                    *self.index_info_extended,
                )
                # Y * <0| Gd Ud mux | CSF>
                g_part_x += self.Y_G_normed[i, state_number] * expectation_value_extended(
                    G_ket,
                    [],
                    Udmux_ket,
                    *self.index_info_extended,
                )
                # -Z * <0| muy U G | CSF>
                g_part_y -= self.Z_G_normed[i, state_number] * expectation_value_extended(
                    Udmuyd_ket,
                    [],
                    G_ket,
                    *self.index_info_extended,
                )
                # Y * <0| Gd Ud muy | CSF>
                g_part_y += self.Y_G_normed[i, state_number] * expectation_value_extended(
                    G_ket,
                    [],
                    Udmuy_ket,
                    *self.index_info_extended,
                )
                # -Z * <0| muz U G | CSF>
                g_part_z -= self.Z_G_normed[i, state_number] * expectation_value_extended(
                    Udmuzd_ket,
                    [],
                    G_ket,
                    *self.index_info_extended,
                )
                # Y * <0| Gd Ud muz | CSF>
                g_part_z += self.Y_G_normed[i, state_number] * expectation_value_extended(
                    G_ket,
                    [],
                    Udmuz_ket,
                    *self.index_info_extended,
                )
            transition_dipoles[state_number, 0] = q_part_x + g_part_x
            transition_dipoles[state_number, 1] = q_part_y + g_part_y
            transition_dipoles[state_number, 2] = q_part_z + g_part_z
        return transition_dipoles
