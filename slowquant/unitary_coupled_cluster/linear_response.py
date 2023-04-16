import copy

import numpy as np
import scipy

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
from slowquant.unitary_coupled_cluster.base import (
    Epq,
    Hamiltonian,
    PauliOperator,
    a_op_spin,
    commutator,
    expectation_value,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.util import iterate_T1, iterate_T2, ThetaPicker, iterate_T3, iterate_T4


class LinearResponseUCC:
    def __init__(self, wave_function: WaveFunctionUCC, excitations: str, is_spin_conserving: bool = False, use_TDA: bool = False) -> None:
        self.wf = copy.deepcopy(wave_function)
        self.theta_picker = ThetaPicker(self.wf.active_occ, self.wf.active_unocc, is_spin_conserving=is_spin_conserving)

        self.G_ops = []
        num_spin_orbs = self.wf.num_spin_orbs
        num_elec = self.wf.num_elec
        excitations = excitations.lower()
        if 's' in excitations:
            for (_, a, i) in self.theta_picker.get_T1_generator():
                self.G_ops.append(
                    PauliOperator(a_op_spin(a, True, num_spin_orbs, num_elec))
                    * PauliOperator(a_op_spin(i, False, num_spin_orbs, num_elec))
                )
        if 'd' in excitations:
            for (_, a, i, b, j) in self.theta_picker.get_T2_generator():
                tmp = PauliOperator(a_op_spin(a, True, num_spin_orbs, num_elec)) * PauliOperator(
                    a_op_spin(b, True, num_spin_orbs, num_elec)
                )
                tmp = tmp * PauliOperator(a_op_spin(j, False, num_spin_orbs, num_elec))
                tmp = tmp * PauliOperator(a_op_spin(i, False, num_spin_orbs, num_elec))
                self.G_ops.append(tmp)
        if 't' in excitations:
            for (_, a, i, b, j, c, k) in self.theta_picker.get_T3_generator():
                tmp = PauliOperator(a_op_spin(a, True, num_spin_orbs, num_elec)) * PauliOperator(
                    a_op_spin(b, True, num_spin_orbs, num_elec)
                )
                tmp = tmp * PauliOperator(a_op_spin(c, True, num_spin_orbs, num_elec))
                tmp = tmp * PauliOperator(a_op_spin(k, False, num_spin_orbs, num_elec))
                tmp = tmp * PauliOperator(a_op_spin(j, False, num_spin_orbs, num_elec))
                tmp = tmp * PauliOperator(a_op_spin(i, False, num_spin_orbs, num_elec))
                self.G_ops.append(tmp)
        if 'q' in excitations:
            for (_, a, i, b, j, c, k, d, l) in self.theta_picker.get_T4_generator():
                tmp = PauliOperator(a_op_spin(a, True, num_spin_orbs, num_elec)) * PauliOperator(
                    a_op_spin(b, True, num_spin_orbs, num_elec)
                )
                tmp = tmp * PauliOperator(a_op_spin(c, True, num_spin_orbs, num_elec))
                tmp = tmp * PauliOperator(a_op_spin(d, True, num_spin_orbs, num_elec))
                tmp = tmp * PauliOperator(a_op_spin(l, False, num_spin_orbs, num_elec))
                tmp = tmp * PauliOperator(a_op_spin(k, False, num_spin_orbs, num_elec))
                tmp = tmp * PauliOperator(a_op_spin(j, False, num_spin_orbs, num_elec))
                tmp = tmp * PauliOperator(a_op_spin(i, False, num_spin_orbs, num_elec))
                self.G_ops.append(tmp)

        num_parameters = len(self.G_ops)
        H = Hamiltonian(self.wf.h_core, self.wf.g_eri, self.wf.c_trans, num_spin_orbs, num_elec)
        self.M = np.zeros((num_parameters, num_parameters))
        self.V = np.zeros((num_parameters, num_parameters))
        self.Q = np.zeros((num_parameters, num_parameters))
        self.W = np.zeros((num_parameters, num_parameters))
        for j, G2 in enumerate(self.G_ops):
            H_G2 = commutator(H, G2)
            if not use_TDA:
                H_G2_dagger = commutator(H, G2.dagger)
            for i, G1 in enumerate(self.G_ops):
                # Make M
                operator = commutator(G1.dagger, H_G2)
                self.M[i, j] = expectation_value(self.wf.state_vector, operator, self.wf.state_vector)
                # Make V
                operator = commutator(G1.dagger, G2)
                self.V[i, j] = expectation_value(self.wf.state_vector, operator, self.wf.state_vector)
                if not use_TDA:
                    # Make Q
                    operator = commutator(G1.dagger, H_G2_dagger)
                    self.Q[i, j] = -expectation_value(self.wf.state_vector, operator, self.wf.state_vector)
                    # Make W
                    operator = commutator(G1.dagger, G2.dagger)
                    self.W[i, j] = -expectation_value(self.wf.state_vector, operator, self.wf.state_vector)

    def calc_excitation_energies(self) -> None:
        size = len(self.M)
        self.E2 = np.zeros((size * 2, size * 2))
        self.E2[:size, :size] = self.M
        self.E2[:size, size:] = self.Q
        self.E2[size:, :size] = np.conj(self.Q)
        self.E2[size:, size:] = np.conj(self.M)

        self.S = np.zeros((size * 2, size * 2))
        self.S[:size, :size] = self.V
        self.S[:size, size:] = self.W
        self.S[size:, :size] = -np.conj(self.W)
        self.S[size:, size:] = -np.conj(self.V)

        eigval, eigvec = scipy.linalg.eig(self.E2, self.S)
        sorting = np.argsort(eigval)
        self.excitation_energies = np.real(eigval[sorting][size:])
        self.response_vectors = np.real(eigvec[:, sorting][:, size:])

    def get_excited_state_overlap(self, state_number: int) -> float:
        number_excitations = len(self.excitation_energies)
        for i, G in enumerate(self.G_ops):
            if i == 0:
                transfer_op = (
                    self.response_vectors[i, state_number] * G
                    + self.response_vectors[i + number_excitations, state_number] * G.dagger
                )
            else:
                transfer_op += (
                    self.response_vectors[i, state_number] * G
                    + self.response_vectors[i + number_excitations, state_number] * G.dagger
                )
        return expectation_value(self.wf.state_vector, transfer_op, self.wf.state_vector)

    def get_excited_state_norm(self, state_number: int) -> float:
        number_excitations = len(self.excitation_energies)
        for i, G in enumerate(self.G_ops):
            if i == 0:
                transfer_op = (
                    self.response_vectors[i, state_number] * G
                    + self.response_vectors[i + number_excitations, state_number] * G.dagger
                )
            else:
                transfer_op += (
                    self.response_vectors[i, state_number] * G
                    + self.response_vectors[i + number_excitations, state_number] * G.dagger
                )
        return expectation_value(self.wf.state_vector, transfer_op.dagger * transfer_op, self.wf.state_vector)

    def get_transition_dipole(self, state_number: int, multipole_integral: np.ndarray) -> float:
        number_excitations = len(self.excitation_energies)
        for i, G in enumerate(self.G_ops):
            if i == 0:
                transfer_op = (
                    self.response_vectors[i, state_number] * G
                    + self.response_vectors[i + number_excitations, state_number] * G.dagger
                )
            else:
                transfer_op += (
                    self.response_vectors[i, state_number] * G
                    + self.response_vectors[i + number_excitations, state_number] * G.dagger
                )
        muz = one_electron_integral_transform(self.wf.c_trans, multipole_integral)
        counter = 0
        for p in range(self.wf.num_spin_orbs // 2):
            for q in range(self.wf.num_spin_orbs // 2):
                if counter == 0:
                    muz_op = muz[p, q] * Epq(p, q, self.wf.num_spin_orbs, self.wf.num_elec)
                    counter += 1
                else:
                    muz_op += muz[p, q] * Epq(p, q, self.wf.num_spin_orbs, self.wf.num_elec)
        return expectation_value(self.wf.state_vector, muz_op * transfer_op, self.wf.state_vector)
