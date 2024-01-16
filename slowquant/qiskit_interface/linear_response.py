import numpy as np
import scipy

from slowquant.qiskit_interface.base import FermionicOperator
from slowquant.qiskit_interface.interface import QuantumInterface
from slowquant.qiskit_interface.operators import (
    G1,
    G2_1,
    G2_2,
    commutator,
    double_commutator,
    hamiltonian_full_space,
)
from slowquant.qiskit_interface.wavefunction import WaveFunction


class quantumLR:
    def __init__(
        self,
        WF: WaveFunction,
    ) -> None:
        """
        Initialize linear response by calculating the needed matrices.
        """
        # create out of WF obj
        # TBD

        self.H = H
        self.G_ops = G_ops

        num_parameters = len(G_ops)
        self.A = np.zeros((num_parameters, num_parameters))
        self.B = np.zeros((num_parameters, num_parameters))
        self.Sigma = np.zeros((num_parameters, num_parameters))
        self.Delta = np.zeros((num_parameters, num_parameters))

    def run_naive(
        self,
        QI: QuantumInterface,
    ) -> None:
        # Calculate Matrices
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                self.A[i, j] = self.A[j, i] = QI.quantum_expectation_value(
                    double_commutator(GI.dagger, self.H, GJ)
                )
                # Make B
                self.B[i, j] = self.B[j, i] = QI.quantum_expectation_value(
                    double_commutator(GI.dagger, self.H, GJ.dagger)
                )
                # Make Sigma
                self.Sigma[i, j] = self.Sigma[j, i] = QI.quantum_expectation_value(commutator(GI.dagger, GJ))

    def run_projected(
        self,
        QI: QuantumInterface,
    ) -> None:
        # Calculate Matrices
        # pre-calculate <0|G|0> and <0|HG|0>
        G_exp = []
        HG_exp = []
        for j, GJ in enumerate(self.G_ops):
            G_exp.append(QI.quantum_expectation_value(GJ))
            HG_exp.append(QI.quantum_expectation_value(self.H * GJ))
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                val = QI.quantum_expectation_value(GI.dagger * self.H * GJ)
                GG_exp = QI.quantum_expectation_value(GI.dagger * GJ)
                val -= GG_exp * QI.vqe.electronic_energies[0]  # WF
                val -= G_exp[i] * HG_exp[j]
                val += G_exp[i] * G_exp[j] * QI.vqe.electronic_energies[0]
                self.A[i, j] = self.A[j, i] = val
                # Make B
                val = HG_exp[i] * G_exp[j]
                val -= G_exp[i] * G_exp[j] * QI.vqe.electronic_energies[0]
                self.B[i, j] = self.B[j, i] = val
                # Make Sigma
                self.Sigma[i, j] = self.Sigma[j, i] = GG_exp - (G_exp[i] * G_exp[j])

    def get_excitation_energies(self):
        """
        Solve LR eigenvalue problem
        """

        # Build Hessian and Metric
        size = len(self.A)
        self.Delta = np.zeros_like(self.Sigma)
        self.E2 = np.zeros((size * 2, size * 2))
        self.E2[:size, :size] = self.A
        self.E2[:size, size:] = self.B
        self.E2[size:, :size] = self.B
        self.E2[size:, size:] = self.A
        self.S = np.zeros((size * 2, size * 2))
        self.S[:size, :size] = self.Sigma
        self.S[:size, size:] = self.Delta
        self.S[size:, :size] = -self.Delta
        self.S[size:, size:] = -self.Sigma

        # Get eigenvalues
        eigval, eigvec = scipy.linalg.eig(self.E2, self.S)
        sorting = np.argsort(eigval)
        self.excitation_energies = np.real(eigval[sorting][size:])

        return self.excitation_energies
