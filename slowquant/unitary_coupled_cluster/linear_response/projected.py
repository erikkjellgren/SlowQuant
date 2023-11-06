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
    get_orbital_response_hessian_block,
    get_orbital_response_metric_sgima,
    get_orbital_response_property_gradient,
    get_orbital_response_vector_norm,
)
from slowquant.unitary_coupled_cluster.linear_response.lr_baseclass import (
    LinearResponseBaseClass,
)
from slowquant.unitary_coupled_cluster.operator_hybrid import (
    OperatorHybrid,
    OperatorHybridData,
    convert_pauli_to_hybrid_form,
    expectation_value_hybrid_flow,
)
from slowquant.unitary_coupled_cluster.operator_pauli import (
    OperatorPauli,
    energy_hamiltonian_pauli,
    epq_pauli,
    hamiltonian_pauli_1i_1a,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.util import ThetaPicker, construct_ucc_u


class ResponseOperator:
    def __init__(self, occ_idx: Sequence[int], unocc_idx: Sequence[int], operator: OperatorHybrid) -> None:
        """Initialize response excitation operator.

        Args:
            occ_idx: Index of occupied orbitals.
            unocc_idx: Index of unoccupied orbitals.
            operator: Operator.
        """
        self.occ_idx = occ_idx
        self.unocc_idx = unocc_idx
        self.operator = operator


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
        self.wf = copy.deepcopy(wave_function)
        self.theta_picker = ThetaPicker(
            self.wf.active_occ_spin_idx,
            self.wf.active_unocc_spin_idx,
            is_spin_conserving=is_spin_conserving,
        )

        self.G_ops: list[ResponseOperator] = []
        self.q_ops: list[ResponseOperator] = []
        num_spin_orbs = self.wf.num_spin_orbs
        num_elec = self.wf.num_elec
        excitations = excitations.lower()
        U_matrix = construct_ucc_u(
            self.wf.num_active_spin_orbs,
            self.wf.num_active_elec,
            self.wf.theta1
            + self.wf.theta2
            + self.wf.theta3
            + self.wf.theta4
            + self.wf.theta5
            + self.wf.theta6,
            self.wf.singlet_excitation_operator_generator,
            "sdtq56",  # self.wf._excitations,
        )
        inactive_str = "I" * self.wf.num_inactive_spin_orbs
        virtual_str = "I" * self.wf.num_virtual_spin_orbs
        self.U = OperatorHybrid(
            {inactive_str + virtual_str: OperatorHybridData(inactive_str, U_matrix, virtual_str)}
        )
        if "s" in excitations:
            for _, a, i, op_ in self.theta_picker.get_t1_generator_sa(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(ResponseOperator((i,), (a,), op))
        if "d" in excitations:
            for _, a, i, b, j, op_ in self.theta_picker.get_t2_generator_sa(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(ResponseOperator((i, j), (a, b), op))
        if "t" in excitations:
            for _, a, i, b, j, c, k, op_ in self.theta_picker.get_t3_generator(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(ResponseOperator((i, j, k), (a, b, c), op))
        if "q" in excitations:
            for _, a, i, b, j, c, k, d, l, op_ in self.theta_picker.get_t4_generator(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(ResponseOperator((i, j, k, l), (a, b, c, d), op))
        if "5" in excitations:
            for _, a, i, b, j, c, k, d, l, e, m, op_ in self.theta_picker.get_t5_generator(
                num_spin_orbs, num_elec
            ):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(ResponseOperator((i, j, k, l, m), (a, b, c, d, e), op))
        if "6" in excitations:
            for _, a, i, b, j, c, k, d, l, e, m, f, n, op_ in self.theta_picker.get_t6_generator(
                num_spin_orbs, num_elec
            ):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                self.G_ops.append(ResponseOperator((i, j, k, l, m, n), (a, b, c, d, e, f), op))
        for i, a in self.wf.kappa_idx:
            op_ = 2 ** (-1 / 2) * epq_pauli(a, i, self.wf.num_spin_orbs, self.wf.num_elec)
            op = convert_pauli_to_hybrid_form(
                op_,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.num_virtual_spin_orbs,
            )
            self.q_ops.append(ResponseOperator((i), (a), op))

        num_parameters = len(self.G_ops) + len(self.q_ops)
        self.A = np.zeros((num_parameters, num_parameters))
        self.B = np.zeros((num_parameters, num_parameters))
        self.Sigma = np.zeros((num_parameters, num_parameters))
        self.Delta = np.zeros((num_parameters, num_parameters))
        H_1i_1a = convert_pauli_to_hybrid_form(
            hamiltonian_pauli_1i_1a(
                self.wf.h_ao,
                self.wf.g_ao,
                self.wf.c_trans,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.num_virtual_spin_orbs,
                num_elec,
            ),
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
        )
        H_en = convert_pauli_to_hybrid_form(
            energy_hamiltonian_pauli(
                self.wf.h_ao,
                self.wf.g_ao,
                self.wf.c_trans,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.num_virtual_spin_orbs,
                num_elec,
            ),
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
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
        grad = get_orbital_gradient_response(
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
            print("WARNING!")
            print("Gradient working equations not implemented for projected G operators")
            break
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                raise ValueError("Large Gradient detected in G of ", np.max(np.abs(grad)))
        # Do orbital-orbital blocks
        self.A[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_hessian_block(
            rdms,
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.kappa_idx_dagger,
            self.wf.kappa_idx,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        self.B[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_hessian_block(
            rdms,
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.kappa_idx_dagger,
            self.wf.kappa_idx_dagger,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )
        self.Sigma[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_metric_sgima(
            rdms, self.wf.kappa_idx
        )
        for j, opJ in enumerate(self.q_ops):
            qJ = opJ.operator
            for i, opI in enumerate(self.G_ops):
                GI = opI.operator
                # Make A
                self.A[j, i + idx_shift] = self.A[i + idx_shift, j] = expectation_value_hybrid_flow(
                    self.wf.state_vector, [GI.dagger, H_1i_1a, qJ], self.wf.state_vector
                )
                # Make B
                self.B[j, i + idx_shift] = self.B[i + idx_shift, j] = -expectation_value_hybrid_flow(
                    self.wf.state_vector, [GI.dagger, qJ.dagger, H_1i_1a], self.wf.state_vector
                )
        for j, opJ in enumerate(self.G_ops):
            GJ = opJ.operator
            for i, opI in enumerate(self.G_ops):
                GI = opI.operator
                if i < j:
                    continue
                # Make A
                val = expectation_value_hybrid_flow(
                    self.wf.state_vector, [GI.dagger, H_en, GJ], self.wf.state_vector
                )
                val -= (
                    expectation_value_hybrid_flow(self.wf.state_vector, [GI.dagger, GJ], self.wf.state_vector)
                    * self.wf.energy_elec
                )
                val -= expectation_value_hybrid_flow(
                    self.wf.state_vector, [GI.dagger], self.wf.state_vector
                ) * expectation_value_hybrid_flow(self.wf.state_vector, [H_en, GJ], self.wf.state_vector)
                val += (
                    expectation_value_hybrid_flow(self.wf.state_vector, [GI.dagger], self.wf.state_vector)
                    * expectation_value_hybrid_flow(self.wf.state_vector, [GJ], self.wf.state_vector)
                    * self.wf.energy_elec
                )
                self.A[i + idx_shift, j + idx_shift] = self.A[j + idx_shift, i + idx_shift] = val
                # Make B
                val = expectation_value_hybrid_flow(
                    self.wf.state_vector, [GI.dagger, H_en], self.wf.state_vector
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
            for i, op in enumerate(self.G_ops):
                GI = op.operator
                for j, op2 in enumerate(self.G_ops):
                    GJ = op2.operator

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
        mux_op = OperatorPauli({})
        muy_op = OperatorPauli({})
        muz_op = OperatorPauli({})
        for p in range(self.wf.num_spin_orbs // 2):
            for q in range(self.wf.num_spin_orbs // 2):
                Epq_op = epq_pauli(p, q, self.wf.num_spin_orbs, self.wf.num_elec)
                if abs(mux[p, q]) > 10**-10:
                    mux_op += mux[p, q] * Epq_op
                if abs(muy[p, q]) > 10**-10:
                    muy_op += muy[p, q] * Epq_op
                if abs(muz[p, q]) > 10**-10:
                    muz_op += muz[p, q] * Epq_op
        mux_op = convert_pauli_to_hybrid_form(
            mux_op,
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
        )
        muy_op = convert_pauli_to_hybrid_form(
            muy_op,
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
        )
        muz_op = convert_pauli_to_hybrid_form(
            muz_op,
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
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
            for i, op in enumerate(self.G_ops):
                G = op.operator
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
