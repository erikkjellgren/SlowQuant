import copy
from collections.abc import Sequence

import numpy as np
import scipy

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.operator_contracted import (
    commutator_contract,
    double_commutator_contract,
    expectation_value_contracted,
    operatormul3_contract,
    operatormul_contract,
)
from slowquant.unitary_coupled_cluster.operator_hybrid import (
    convert_pauli_to_hybrid_form,
    expectation_value_hybrid,
)
from slowquant.unitary_coupled_cluster.operator_pauli import (
    Epq,
    Hamiltonian,
    Hamiltonian_energy_only,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.util import ThetaPicker, construct_UCC_U


class LinearResponseUCC:
    def __init__(
        self,
        wave_function: WaveFunctionUCC,
        excitations: str,
        is_spin_conserving: bool = False,
        do_selfconsistent_operators: bool = True,
    ) -> None:
        self.wf = copy.deepcopy(wave_function)
        self.theta_picker = ThetaPicker(
            self.wf.active_occ,
            self.wf.active_unocc,
            is_spin_conserving=is_spin_conserving,
            deshift=self.wf.num_inactive_spin_orbs,
        )

        self.G_ops = []
        self.q_ops = []
        num_spin_orbs = self.wf.num_spin_orbs
        num_elec = self.wf.num_elec
        excitations = excitations.lower()
        U = construct_UCC_U(
            self.wf.num_active_spin_orbs,
            self.wf.num_active_elec,
            self.wf.theta1 + self.wf.theta2 + self.wf.theta3 + self.wf.theta4,
            self.wf.theta_picker_full,
            "sdtq",  # self.wf._excitations,
        )
        if "s" in excitations:
            for _, a, i, op_ in self.theta_picker.get_T1_generator_SA(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                if do_selfconsistent_operators:
                    op = op.apply_U_from_right(U.conj().transpose())
                    op = op.apply_U_from_left(U)
                self.G_ops.append([{i}, {a}, op])
        if "d" in excitations:
            for _, a, i, b, j, op_ in self.theta_picker.get_T2_generator_SA(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                if do_selfconsistent_operators:
                    op = op.apply_U_from_right(U.conj().transpose())
                    op = op.apply_U_from_left(U)
                self.G_ops.append([{i, j}, {a, b}, op])
        if "t" in excitations:
            for _, _, _, _, _, _, _, op_ in self.theta_picker.get_T3_generator(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                if do_selfconsistent_operators:
                    op = op.apply_U_from_right(U.conj().transpose())
                    op = op.apply_U_from_left(U)
                self.G_ops.append(op)
        if "q" in excitations:
            for _, _, _, _, _, _, _, _, _, op_ in self.theta_picker.get_T4_generator(num_spin_orbs, num_elec):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                    self.wf.num_virtual_spin_orbs,
                )
                if do_selfconsistent_operators:
                    op = op.apply_U_from_right(U.conj().transpose())
                    op = op.apply_U_from_left(U)
                self.G_ops.append(op)
        for i, a in self.wf.kappa_idx:
            op_ = 2 ** (-1 / 2) * Epq(a, i, self.wf.num_spin_orbs, self.wf.num_elec)
            op = convert_pauli_to_hybrid_form(
                op_,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
                self.wf.num_virtual_spin_orbs,
            )
            self.q_ops.append([{i}, {a}, op])

        num_parameters = len(self.G_ops) + len(self.q_ops)
        self.M = np.zeros((num_parameters, num_parameters))
        self.Q = np.zeros((num_parameters, num_parameters))
        self.V = np.zeros((num_parameters, num_parameters))
        self.W = np.zeros((num_parameters, num_parameters))
        H = convert_pauli_to_hybrid_form(
            Hamiltonian(self.wf.h_core, self.wf.g_eri, self.wf.c_trans, num_spin_orbs, num_elec),
            self.wf.num_inactive_spin_orbs,
            self.wf.num_active_spin_orbs,
            self.wf.num_virtual_spin_orbs,
        )
        H_en = convert_pauli_to_hybrid_form(
            Hamiltonian_energy_only(
                self.wf.h_core,
                self.wf.g_eri,
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
        idx_shift = len(self.q_ops)
        print("Gs", len(self.G_ops))
        print("qs", len(self.q_ops))
        if do_selfconsistent_operators:
            calculation_type = "selfconsistent"
        else:
            calculation_type = "naive"
        # calculation_type = "generic"
        for j, (occ_idxs_j, unocc_idxs_i, qJ) in enumerate(self.q_ops):
            for i, (occ_idx_i, unocc_idx_i, qI) in enumerate(self.q_ops):
                if i < j:
                    continue
                if calculation_type == "selfconsistent" or calculation_type == "naive":
                    # Make M
                    operator = operatormul3_contract(qI.dagger, H, qJ) - operatormul3_contract(
                        qI.dagger, qJ, H
                    )
                    self.M[i, j] = self.M[j, i] = expectation_value_contracted(
                        self.wf.state_vector, operator, self.wf.state_vector
                    )
                    # Make Q
                    self.Q[i, j] = self.Q[j, i] = -expectation_value_contracted(
                        self.wf.state_vector,
                        operatormul3_contract(qI.dagger, qJ.dagger, H),
                        self.wf.state_vector,
                    )
                    # Make V
                    if i == j:
                        self.V[i, j] = self.V[j, i] = expectation_value_contracted(
                            self.wf.state_vector,
                            operatormul_contract(qI.dagger, qJ),
                            self.wf.state_vector,
                        )
                    # Make W
                elif calculation_type == "generic":
                    # Make M
                    self.M[i, j] = self.M[j, i] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(qI.dagger, H, qJ),
                        self.wf.state_vector,
                    )
                    # Make Q
                    self.Q[i, j] = self.Q[j, i] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(qI.dagger, H, qJ.dagger),
                        self.wf.state_vector,
                    )
                    # Make V
                    self.V[i, j] = self.V[j, i] = expectation_value_contracted(
                        self.wf.state_vector,
                        commutator_contract(qI.dagger, qJ),
                        self.wf.state_vector,
                    )
                    # Make W
                    self.W[i, j] = self.W[j, i] = expectation_value_contracted(
                        self.wf.state_vector,
                        commutator_contract(qI.dagger, qJ.dagger),
                        self.wf.state_vector,
                    )
                else:
                    raise NameError("Could not determine calculation_type got: {calculation_type}")
        for j, (occ_idxs_j, unocc_idxs_j, GJ) in enumerate(self.G_ops):
            for i, (occ_idxs_i, unocc_idx_j, qI) in enumerate(self.q_ops):
                if calculation_type == "selfconsistent":
                    # Make M
                    operator = operatormul3_contract(qI.dagger, H, GJ) - operatormul3_contract(
                        qI.dagger, GJ, H
                    )
                    self.M[i, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector, operator, self.wf.state_vector
                    )
                    # Make Q
                    self.Q[i, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector,
                        operatormul3_contract(qI.dagger, GJ.dagger, H),
                        self.wf.state_vector,
                    )
                    # Make V
                    # Make W
                elif calculation_type == "naive":
                    # Make M
                    operator = operatormul3_contract(qI.dagger, H, GJ) - operatormul3_contract(
                        qI.dagger, GJ, H
                    )
                    self.M[i, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector, operator, self.wf.state_vector
                    )
                    # Make Q
                    operator = operatormul3_contract(qI.dagger, H, GJ.dagger) - operatormul3_contract(
                        qI.dagger, GJ.dagger, H
                    )
                    self.Q[i, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector,
                        operator,
                        self.wf.state_vector,
                    )
                    # Make V
                    # Make W
                elif calculation_type == "generic":
                    # Make M
                    self.M[i, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(qI.dagger, H, GJ),
                        self.wf.state_vector,
                    )
                    # Make Q
                    self.Q[i, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(qI.dagger, H, GJ.dagger),
                        self.wf.state_vector,
                    )
                    # Make V
                    self.V[i, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(qI.dagger, GJ), self.wf.state_vector
                    )
                    # Make W
                    self.W[i, j + idx_shift] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(qI.dagger, GJ.dagger), self.wf.state_vector
                    )
                else:
                    raise NameError("Could not determine calculation_type got: {calculation_type}")
        for j, (occ_idxs_j, unocc_idx_j, qJ) in enumerate(self.q_ops):
            for i, (occ_idxs_i, unocc_idxs_j, GI) in enumerate(self.G_ops):
                if calculation_type == "selfconsistent":
                    # Make M
                    operator = operatormul3_contract(GI.dagger, H, qJ) - operatormul3_contract(
                        GI.dagger, qJ, H
                    )
                    self.M[i + idx_shift, j] = expectation_value_contracted(
                        self.wf.state_vector, operator, self.wf.state_vector
                    )
                    # Make Q
                    self.Q[i + idx_shift, j] = -expectation_value_contracted(
                        self.wf.state_vector,
                        operatormul3_contract(GI.dagger, qJ.dagger, H),
                        self.wf.state_vector,
                    )
                    # Make V
                    # Make W
                elif calculation_type == "naive":
                    # Make M
                    operator = (
                        operatormul3_contract(GI.dagger, H, qJ)
                        - operatormul3_contract(GI.dagger, qJ, H)
                        - operatormul3_contract(H, qJ, GI.dagger)
                    )
                    self.M[i + idx_shift, j] = expectation_value_contracted(
                        self.wf.state_vector, operator, self.wf.state_vector
                    )
                    # Make Q
                    operator = operatormul3_contract(qJ.dagger, H, GI.dagger) - operatormul3_contract(
                        GI.dagger, qJ.dagger, H
                    )
                    self.Q[i + idx_shift, j] = expectation_value_contracted(
                        self.wf.state_vector, operator, self.wf.state_vector
                    )
                    # Make V
                    # Make W
                elif calculation_type == "generic":
                    # Make M
                    self.M[i + idx_shift, j] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(GI.dagger, H, qJ),
                        self.wf.state_vector,
                    )
                    # Make Q
                    self.Q[i + idx_shift, j] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(GI.dagger, H, qJ.dagger),
                        self.wf.state_vector,
                    )
                    # Make V
                    self.V[i + idx_shift, j] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(GI.dagger, qJ), self.wf.state_vector
                    )
                    # Make W
                    self.W[i + idx_shift, j] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(GI.dagger, qJ.dagger), self.wf.state_vector
                    )
                else:
                    raise NameError("Could not determine calculation_type got: {calculation_type}")
        for j, (occ_idxs_j, unocc_idxs_j, GJ) in enumerate(self.G_ops):
            for i, (occ_idxs_i, unocc_idxs_i, GI) in enumerate(self.G_ops):
                if i < j:
                    continue
                if calculation_type == "selfconsistent":
                    # Make M
                    operator = operatormul3_contract(GI.dagger, H_en, GJ)
                    # Need to check this is valid for double excitations also
                    if len(occ_idxs_j) == len(occ_idxs_i) == 1:
                        if occ_idxs_j == occ_idxs_i and occ_idxs_j == occ_idxs_i:
                            operator -= operatormul3_contract(GI.dagger, GJ, H_en)
                    else:
                        operator -= operatormul3_contract(GI.dagger, GJ, H_en)
                    self.M[i + idx_shift, j + idx_shift] = self.M[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_contracted(self.wf.state_vector, operator, self.wf.state_vector)
                    # Make Q
                    self.Q[i + idx_shift, j + idx_shift] = self.Q[
                        j + idx_shift, i + idx_shift
                    ] = -expectation_value_contracted(
                        self.wf.state_vector,
                        operatormul3_contract(GI.dagger, GJ.dagger, H_en),
                        self.wf.state_vector,
                    )
                    # Make V
                    if i == j:
                        self.V[i + idx_shift, j + idx_shift] = self.V[j + idx_shift, i + idx_shift] = 1
                    # Make W
                elif calculation_type == "naive":
                    # Make M
                    self.M[i + idx_shift, j + idx_shift] = self.M[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(GI.dagger, H_en, GJ),
                        self.wf.state_vector,
                    )
                    # Make Q
                    self.Q[i + idx_shift, j + idx_shift] = self.Q[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(GI.dagger, H_en, GJ.dagger),
                        self.wf.state_vector,
                    )
                    # Make V
                    self.V[i + idx_shift, j + idx_shift] = self.V[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(GI.dagger, GJ), self.wf.state_vector
                    )
                    # Make W
                elif calculation_type == "generic":
                    # Make M
                    self.M[i + idx_shift, j + idx_shift] = self.M[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(GI.dagger, H_en, GJ),
                        self.wf.state_vector,
                    )
                    # Make Q
                    self.Q[i + idx_shift, j + idx_shift] = self.Q[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_contracted(
                        self.wf.state_vector,
                        double_commutator_contract(GI.dagger, H_en, GJ.dagger),
                        self.wf.state_vector,
                    )
                    # Make V
                    self.V[i + idx_shift, j + idx_shift] = self.V[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(GI.dagger, GJ), self.wf.state_vector
                    )
                    # Make W
                    self.W[i + idx_shift, j + idx_shift] = self.W[
                        j + idx_shift, i + idx_shift
                    ] = expectation_value_contracted(
                        self.wf.state_vector, commutator_contract(GI.dagger, GJ.dagger), self.wf.state_vector
                    )
                else:
                    raise NameError("Could not determine calculation_type got: {calculation_type}")

    def calc_excitation_energies(self) -> None:
        size = len(self.M)
        E2 = np.zeros((size * 2, size * 2))
        E2[:size, :size] = self.M
        E2[:size, size:] = self.Q
        E2[size:, :size] = np.conj(self.Q)
        E2[size:, size:] = np.conj(self.M)

        S = np.zeros((size * 2, size * 2))
        S[:size, :size] = self.V
        S[:size, size:] = self.W
        S[size:, :size] = -np.conj(self.W)
        S[size:, size:] = -np.conj(self.V)

        eigval, eigvec = scipy.linalg.eig(E2, S)
        sorting = np.argsort(eigval)
        self.excitation_energies = np.real(eigval[sorting][size:])
        self.response_vectors = np.real(eigvec[:, sorting][:, size:])
        self.normed_response_vectors = np.zeros_like(self.response_vectors)
        for state_number in range(size):
            norm = self.get_excited_state_norm(state_number)
            if norm < 10**-10:
                continue
            self.normed_response_vectors[:, state_number] = (
                self.response_vectors[:, state_number] * (1 / norm) ** 0.5
            )

    def get_excited_state_overlap(self, state_number: int) -> float:
        number_excitations = len(self.excitation_energies)
        print("WARNING: This function [get_excited_state_overlap] might not be working.")
        for i, (_, _, G) in enumerate(self.q_ops + self.G_ops):
            if i == 0:
                transfer_op = (
                    self.normed_response_vectors[i, state_number] * G.dagger
                    + self.normed_response_vectors[i + number_excitations, state_number] * G
                )
            else:
                transfer_op += (
                    self.normed_response_vectors[i, state_number] * G.dagger
                    + self.normed_response_vectors[i + number_excitations, state_number] * G
                )
        return expectation_value_hybrid(self.wf.state_vector, transfer_op, self.wf.state_vector)

    def get_excited_state_norm(self, state_number: int) -> float:
        number_excitations = len(self.excitation_energies)
        for i, (_, _, G) in enumerate(self.q_ops + self.G_ops):
            if i == 0:
                transfer_op = (
                    self.response_vectors[i, state_number] * G.dagger
                    + self.response_vectors[i + number_excitations, state_number] * G
                )
            else:
                transfer_op += (
                    self.response_vectors[i, state_number] * G.dagger
                    + self.response_vectors[i + number_excitations, state_number] * G
                )
        return expectation_value_contracted(
            self.wf.state_vector, commutator_contract(transfer_op, transfer_op.dagger), self.wf.state_vector
        )

    def get_transition_dipole(
        self, state_number: int, dipole_integrals: Sequence[np.ndarray]
    ) -> tuple[float, float, float]:
        if len(dipole_integrals) != 3:
            raise ValueError(f"Expected 3 dipole integrals got {len(dipole_integrals)}")
        number_excitations = len(self.excitation_energies)
        for i, (_, _, G) in enumerate(self.q_ops + self.G_ops):
            if i == 0:
                transfer_op = (
                    self.normed_response_vectors[i, state_number] * G.dagger
                    + self.normed_response_vectors[i + number_excitations, state_number] * G
                )
            else:
                transfer_op += (
                    self.normed_response_vectors[i, state_number] * G.dagger
                    + self.normed_response_vectors[i + number_excitations, state_number] * G
                )
        mux = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[0])
        muy = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[1])
        muz = one_electron_integral_transform(self.wf.c_trans, dipole_integrals[2])
        counter = 0
        for p in range(self.wf.num_spin_orbs // 2):
            for q in range(self.wf.num_spin_orbs // 2):
                Epq_op = Epq(p, q, self.wf.num_spin_orbs, self.wf.num_elec)
                if counter == 0:
                    mux_op = mux[p, q] * Epq_op
                    muy_op = muy[p, q] * Epq_op
                    muz_op = muz[p, q] * Epq_op
                    counter += 1
                else:
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
        transition_dipole_x = 0.0
        transition_dipole_y = 0.0
        transition_dipole_z = 0.0
        if mux_op.operators != {}:
            transition_dipole_x = expectation_value_contracted(
                self.wf.state_vector, commutator_contract(mux_op, transfer_op), self.wf.state_vector
            )
        if muy_op.operators != {}:
            transition_dipole_y = expectation_value_contracted(
                self.wf.state_vector, commutator_contract(muy_op, transfer_op), self.wf.state_vector
            )
        if muz_op.operators != {}:
            transition_dipole_z = expectation_value_contracted(
                self.wf.state_vector, commutator_contract(muz_op, transfer_op), self.wf.state_vector
            )
        return transition_dipole_x, transition_dipole_y, transition_dipole_z

    def get_oscillator_strength(self, state_number: int, dipole_integrals: Sequence[np.ndarray]) -> float:
        r"""Calculate oscillator strength.

        .. math::
            f_n = \frac{2}{3}e_n\left|\left<0\left|\hat{\mu}\right|n\left>\right|^2

        Args:
            state_number: Target excited state (zero being the first excited state).
            dipole_integrals: Dipole integrals (x,y,z) in AO basis.

        Rerturns:
            Oscillator Strength.
        """
        transition_dipole_x, transition_dipole_y, transition_dipole_z = self.get_transition_dipole(
            state_number, dipole_integrals
        )
        excitation_energy = self.excitation_energies[state_number]
        return (
            2
            / 3
            * excitation_energy
            * (transition_dipole_x**2 + transition_dipole_y**2 + transition_dipole_z**2)
        )

    def get_nice_output(self, dipole_integrals: np.ndarray) -> str:
        """Create table of excitation energies and oscillator strengths.

        Args:
            dipole_integrals: Dipole integrals (x,y,z) in AO basis.

        Returns:
            Nicely formatted table.
        """
        output = (
            "Excitation # | Excitation energy [Hartree] | Excitation energy [eV] | Oscillator strengths\n"
        )
        for i, exc_energy in enumerate(self.excitation_energies):
            osc_strength = self.get_oscillator_strength(i, dipole_integrals)
            exc_str = f"{exc_energy:2.6f}"
            exc_str_ev = f"{exc_energy*27.2114079527:3.6f}"
            osc_str = f"{osc_strength:1.6f}"
            output += f"{str(i+1).center(12)} | {exc_str.center(27)} | {exc_str_ev.center(22)} | {osc_str.center(20)}\n"
        return output
