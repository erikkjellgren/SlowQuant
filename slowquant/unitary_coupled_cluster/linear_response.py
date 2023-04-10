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
from slowquant.unitary_coupled_cluster.util import iterate_T1, iterate_T2


class LinearResponseUCC:
    def __init__(self, wave_function: WaveFunctionUCC, is_spin_conserving: bool = False) -> None:
        self.wf = copy.deepcopy(wave_function)

        self.G_ops = []
        num_spin_orbs = self.wf.num_spin_orbs
        num_elec = self.wf.num_elec
        for (_, a, i) in iterate_T1(
            self.wf.active_occ, self.wf.active_unocc, is_spin_conserving=is_spin_conserving
        ):
            self.G_ops.append(
                PauliOperator(a_op_spin(a, True, num_spin_orbs, num_elec))
                * PauliOperator(a_op_spin(i, False, num_spin_orbs, num_elec))
            )
        for (_, a, i, b, j) in iterate_T2(
            self.wf.active_occ, self.wf.active_unocc, is_spin_conserving=is_spin_conserving
        ):
            tmp = PauliOperator(a_op_spin(a, True, num_spin_orbs, num_elec)) * PauliOperator(
                a_op_spin(b, True, num_spin_orbs, num_elec)
            )
            tmp = tmp * PauliOperator(a_op_spin(j, False, num_spin_orbs, num_elec))
            tmp = tmp * PauliOperator(a_op_spin(i, False, num_spin_orbs, num_elec))
            self.G_ops.append(tmp)

        num_parameters = len(self.G_ops)
        H = Hamiltonian(self.wf.h_core, self.wf.g_eri, self.wf.c_trans, num_spin_orbs, num_elec)
        self.M = np.zeros((num_parameters, num_parameters))
        self.V = np.zeros((num_parameters, num_parameters))
        self.Q = np.zeros((num_parameters, num_parameters))
        self.W = np.zeros((num_parameters, num_parameters))
        for i, G1 in enumerate(self.G_ops):
            for j, G2 in enumerate(self.G_ops):
                # Make M
                operator = commutator(H, G2)
                operator = commutator(G1.dagger, operator)
                self.M[i, j] = expectation_value(self.wf.state_vector, operator, self.wf.state_vector)
                # Make V
                operator = commutator(G1.dagger, G2)
                self.V[i, j] = expectation_value(self.wf.state_vector, operator, self.wf.state_vector)
                # Make Q
                operator = commutator(H, G2.dagger)
                operator = commutator(G1.dagger, operator)
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
