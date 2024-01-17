import numpy as np
import scipy
from dmdm.util import iterate_t1_sa, iterate_t2_sa  # temporary solution

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

        self.WF = WF
        # Create operators
        self.H = hamiltonian_full_space(WF.h_mo, WF.g_mo, WF.num_orbs)

        self.G_ops = []
        # G1
        for a, i, _, _, _ in iterate_t1_sa(WF.active_occ_idx, WF.active_unocc_idx):
            self.G_ops.append(G1(a, i))
        # G2
        for a, i, b, j, _, _, id in iterate_t2_sa(WF.active_occ_idx, WF.active_unocc_idx):
            if id > 0:
                # G2_1
                self.G_ops.append(G2_1(a, b, i, j))
            else:
                # G2_2
                self.G_ops.append(G2_2(a, b, i, j))

        # q
        self.q_ops = []
        for p in range(0, WF.num_orbs):
            for q in range(p + 1, WF.num_orbs):
                if p in WF.inactive_idx and q in WF.inactive_idx:
                    continue
                if p in WF.virtual_idx and q in WF.virtual_idx:
                    continue
                if p in WF.active_idx and q in WF.active_idx:
                    continue
                self.q_ops.append(G1(p, q))

        num_parameters = len(self.q_ops) + len(self.G_ops)
        self.A = np.zeros((num_parameters, num_parameters))
        self.B = np.zeros((num_parameters, num_parameters))
        self.Sigma = np.zeros((num_parameters, num_parameters))
        self.Delta = np.zeros((num_parameters, num_parameters))

        self.orbs = [self.WF.num_inactive_orbs, self.WF.num_active_orbs, self.WF.num_virtual_orbs]

    def run_naive(
        self,
    ) -> None:
        # Calculate Matrices
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                self.A[i, j] = self.A[j, i] = self.WF.QI.quantum_expectation_value(
                    double_commutator(GI.dagger, self.H, GJ).get_folded_operator(*self.orbs)
                )
                # Make B
                self.B[i, j] = self.B[j, i] = self.WF.QIquantum_expectation_value(
                    double_commutator(GI.dagger, self.H, GJ.dagger).get_folded_operator(*self.orbs)
                )
                # Make Sigma
                self.Sigma[i, j] = self.Sigma[j, i] = self.WF.QIquantum_expectation_value(
                    commutator(GI.dagger, GJ).get_folded_operator(*self.orbs)
                )

    def run_projected(
        self,
    ) -> None:
        # Calculate Matrices
        # pre-calculate <0|G|0> and <0|HG|0>
        G_exp = []
        HG_exp = []
        for j, GJ in enumerate(self.G_ops):
            G_exp.append(self.WF.QIquantum_expectation_value(GJ.get_folded_operator(*self.orbs)))
            HG_exp.append(self.WF.QIquantum_expectation_value((self.H * GJ).get_folded_operator(*self.orbs)))
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                val = self.WF.QIquantum_expectation_value(
                    (GI.dagger * self.H * GJ).get_folded_operator(*self.orbs)
                )
                GG_exp = self.WF.QIquantum_expectation_value((GI.dagger * GJ).get_folded_operator(*self.orbs))
                val -= GG_exp * self.WF.energy_elec
                val -= G_exp[i] * HG_exp[j]
                val += G_exp[i] * G_exp[j] * self.WF.energy_elec
                self.A[i, j] = self.A[j, i] = val
                # Make B
                val = HG_exp[i] * G_exp[j]
                val -= G_exp[i] * G_exp[j] * self.WF.energy_elec
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
