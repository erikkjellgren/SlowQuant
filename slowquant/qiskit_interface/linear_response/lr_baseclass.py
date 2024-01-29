import numpy as np
import scipy
from dmdm.util import iterate_t1_sa, iterate_t2_sa  # temporary solution

from slowquant.qiskit_interface.operators import (
    G1,
    G2_1,
    G2_2,
    hamiltonian_pauli_0i_0a,
    hamiltonian_pauli_1i_1a,
    hamiltonian_pauli_2i_2a,
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


class quantumLRBaseClass:
    def __init__(
        self,
        wf: WaveFunction,
    ) -> None:
        """
        Initialize linear response by calculating the needed matrices.
        """

        self.wf = wf
        # Create operators
        self.H_0i_0a = hamiltonian_pauli_0i_0a(wf.h_mo, wf.g_mo, wf.num_inactive_orbs, wf.num_active_orbs)
        self.H_1i_1a = hamiltonian_pauli_1i_1a(
            wf.h_mo, wf.g_mo, wf.num_inactive_orbs, wf.num_active_orbs, wf.num_virtual_orbs
        )

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
        for p, q in wf.kappa_idx:
            self.q_ops.append(G1(p, q))

        num_parameters = len(self.q_ops) + len(self.G_ops)
        self.num_params = num_parameters
        self.num_q = len(self.q_ops)
        self.num_G = len(self.G_ops)
        self.A = np.zeros((num_parameters, num_parameters))
        self.B = np.zeros((num_parameters, num_parameters))
        self.Sigma = np.zeros((num_parameters, num_parameters))
        self.Delta = np.zeros((num_parameters, num_parameters))

        self.orbs = [self.wf.num_inactive_orbs, self.wf.num_active_orbs, self.wf.num_virtual_orbs]

    def run(self) -> None:
        raise NotImplementedError

    def _get_qbitmap(self) -> np.ndarray:
        raise NotImplementedError

    def get_excitation_energies(self) -> np.ndarray:
        """
        Solve LR eigenvalue problem
        """

        # Build Hessian and Metric
        size = len(self.A)
        self.Delta = np.zeros_like(self.Sigma)
        self.hessian = np.zeros((size * 2, size * 2))
        self.hessian[:size, :size] = self.A
        self.hessian[:size, size:] = self.B
        self.hessian[size:, :size] = self.B
        self.hessian[size:, size:] = self.A
        self.metric = np.zeros((size * 2, size * 2))
        self.metric[:size, :size] = self.Sigma
        self.metric[:size, size:] = self.Delta
        self.metric[size:, :size] = -self.Delta
        self.metric[size:, size:] = -self.Sigma

        # Solve eigenvalue equation
        eigval, eigvec = scipy.linalg.eig(self.hessian, self.metric)
        sorting = np.argsort(eigval)
        self.excitation_energies = np.real(eigval[sorting][size:])
        self.excitation_vectors = np.real(eigvec[:, sorting][:, size:])

        return self.excitation_energies

    def get_normed_excitation_vectors(self) -> None:
        """
        Get normed excitation vectors via excitated state norm
        """

        self.normed_excitation_vectors = np.zeros_like(self.excitation_vectors)
        self._Z_q = self.excitation_vectors[: self.num_q, :]
        self._Z_G = self.excitation_vectors[self.num_q : self.num_q + self.num_G, :]
        self._Y_q = self.excitation_vectors[self.num_q + self.num_G : 2 * self.num_q + self.num_G]
        self._Y_G = self.excitation_vectors[2 * self.num_q + self.num_G :]
        self._Z_q_normed = np.zeros_like(self._Z_q)
        self._Z_G_normed = np.zeros_like(self._Z_G)
        self._Y_q_normed = np.zeros_like(self._Y_q)
        self._Y_G_normed = np.zeros_like(self._Y_G)

        norms = self._get_excited_state_norm()
        for state_number, norm in enumerate(norms):
            if norm < 10**-10:
                print(f"WARNING: State number {state_number} could not be normalized. Norm of {norm}.")
                continue
            self._Z_q_normed[:, state_number] = self._Z_q[:, state_number] * (1 / norm) ** 0.5
            self._Z_G_normed[:, state_number] = self._Z_G[:, state_number] * (1 / norm) ** 0.5
            self._Y_q_normed[:, state_number] = self._Y_q[:, state_number] * (1 / norm) ** 0.5
            self._Y_G_normed[:, state_number] = self._Y_G[:, state_number] * (1 / norm) ** 0.5
            self.normed_excitation_vectors[:, state_number] = (
                self.excitation_vectors[:, state_number] * (1 / norm) ** 0.5
            )

    def _get_excited_state_norm(self) -> np.ndarray:
        """Calculate the norm of excited states.

        Returns:
            Norm of excited states.
        """

        norms = np.zeros(len(self._Z_G[0]))
        for state_number in range(len(self._Z_G[0])):
            # Get Z_q Z_G Y_q and Y_G matrices
            ZZq = np.outer(self._Z_q[:, state_number], self._Z_q[:, state_number].transpose())
            YYq = np.outer(self._Y_q[:, state_number], self._Y_q[:, state_number].transpose())
            ZZG = np.outer(self._Z_G[:, state_number], self._Z_G[:, state_number].transpose())
            YYG = np.outer(self._Y_G[:, state_number], self._Y_G[:, state_number].transpose())

            norms[state_number] = np.sum(self.metric[: self.num_q, : self.num_q] * (ZZq - YYq)) + np.sum(
                self.metric[self.num_q : self.num_q + self.num_G, self.num_q : self.num_q + self.num_G]
                * (ZZG - YYG)
            )

        return norms
