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
from slowquant.unitary_coupled_cluster.density_matrix import (
    ReducedDenstiyMatrix,
    get_orbital_gradient_response,
    get_orbital_response_hessian_block,
    get_orbital_response_metric_sigma,
    get_orbital_response_property_gradient,
    get_orbital_response_vector_norm,
)


class quantumLR:
    def __init__(
        self,
        wf: WaveFunction,
    ) -> None:
        """
        Initialize linear response by calculating the needed matrices.
        """

        self.wf = wf
        # Create operators
        self.H = hamiltonian_full_space(wf.h_mo, wf.g_mo, wf.num_orbs)

        self.G_ops = []
        # G1
        for a, i, _, _, _ in iterate_t1_sa(wf.active_occ_idx, wf.active_unocc_idx):
            self.G_ops.append(G1(a, i))
        # G2
        for a, i, b, j, _, _, id in iterate_t2_sa(wf.active_occ_idx, wf.active_unocc_idx):
            if id > 0:
                # G2_1
                self.G_ops.append(G2_1(a, b, i, j))
            else:
                # G2_2
                self.G_ops.append(G2_2(a, b, i, j))

        # q
        self.q_ops = []
        for p in range(0, wf.num_orbs):
            for q in range(p + 1, wf.num_orbs):
                if p in wf.inactive_idx and q in wf.inactive_idx:
                    continue
                if p in wf.virtual_idx and q in wf.virtual_idx:
                    continue
                if p in wf.active_idx and q in wf.active_idx:
                    continue
                self.q_ops.append(G1(p, q))

        num_parameters = len(self.q_ops) + len(self.G_ops)
        self.A = np.zeros((num_parameters, num_parameters))
        self.B = np.zeros((num_parameters, num_parameters))
        self.Sigma = np.zeros((num_parameters, num_parameters))
        self.Delta = np.zeros((num_parameters, num_parameters))

        self.orbs = [self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.wf.num_virtual_orbs]

    def run_naive(
        self,
    ) -> None:
        # RDMs
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

        # Check gradients
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
                print("WARNING: Large Gradient detected in q of ", np.max(np.abs(grad)))

        grad = np.zeros(2 * len(self.G_ops))
        for i, op in enumerate(self.G_ops):
            grad[i] = self.wf.QI.quantum_expectation_value((self.H * op).get_folded_operator(*self.orbs))
            grad[i + len(self.G_ops)] = self.wf.QI.quantum_expectation_value(
                (op.dagger * self.H).get_folded_operator(*self.orbs)
            )
        if len(grad) != 0:
            print("idx, max(abs(grad active)):", np.argmax(np.abs(grad)), np.max(np.abs(grad)))
            if np.max(np.abs(grad)) > 10**-3:
                print("WARNING: Large Gradient detected in G of ", np.max(np.abs(grad)))

        # qq
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
        self.Sigma[: len(self.q_ops), : len(self.q_ops)] = get_orbital_response_metric_sigma(
            rdms, self.wf.kappa_idx
        )

        # Gq
        for j, qJ in enumerate(self.q_ops):
            for i, GI in enumerate(self.G_ops):
                # Make A
                val = self.wf.QI.quantum_expectation_value(
                    (GI.dagger * self.H * qJ).get_folded_operator(*self.orbs)
                ) - self.wf.QI.quantum_expectation_value(
                    (self.H * qJ * GI.dagger).get_folded_operator(*self.orbs)
                )
                self.A[i + idx_shift, j] = self.A[j, i + idx_shift] = val
                # Make B
                val = self.wf.QI.quantum_expectation_value(
                    (qJ.dagger * self.H * GI.dagger).get_folded_operator(*self.orbs)
                ) - self.wf.QI.quantum_expectation_value(
                    (GI.dagger * qJ.dagger * self.H).get_folded_operator(*self.orbs)
                )
                self.B[i + idx_shift, j] = self.B[j, i + idx_shift] = val

        # GG
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                self.A[i + idx_shift, j + idx_shift] = self.A[
                    j + idx_shift, i + idx_shift
                ] = self.wf.QI.quantum_expectation_value(
                    double_commutator(GI.dagger, self.H, GJ).get_folded_operator(*self.orbs)
                )
                # Make B
                self.B[i + idx_shift, j + idx_shift] = self.B[
                    j + idx_shift, i + idx_shift
                ] = self.wf.QI.quantum_expectation_value(
                    double_commutator(GI.dagger, self.H, GJ.dagger).get_folded_operator(*self.orbs)
                )
                # Make Sigma
                self.Sigma[i + idx_shift, j + idx_shift] = self.Sigma[
                    j + idx_shift, i + idx_shift
                ] = self.wf.QI.quantum_expectation_value(
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
            G_exp.append(self.wf.QI.quantum_expectation_value(GJ.get_folded_operator(*self.orbs)))
            HG_exp.append(self.wf.QI.quantum_expectation_value((self.H * GJ).get_folded_operator(*self.orbs)))
        for j, GJ in enumerate(self.G_ops):
            for i, GI in enumerate(self.G_ops[j:], j):
                # Make A
                val = self.wf.QI.quantum_expectation_value(
                    (GI.dagger * self.H * GJ).get_folded_operator(*self.orbs)
                )
                GG_exp = self.wf.QI.quantum_expectation_value(
                    (GI.dagger * GJ).get_folded_operator(*self.orbs)
                )
                val -= GG_exp * self.wf.energy_elec
                val -= G_exp[i] * HG_exp[j]
                val += G_exp[i] * G_exp[j] * self.wf.energy_elec
                self.A[i, j] = self.A[j, i] = val
                # Make B
                val = HG_exp[i] * G_exp[j]
                val -= G_exp[i] * G_exp[j] * self.wf.energy_elec
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
