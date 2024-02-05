from collections.abc import Sequence

import numpy as np
import scipy
from dmdm.util import iterate_t1_sa, iterate_t2_sa  # temporary solution

from slowquant.qiskit_interface.operators import (
    G1,
    G2_1,
    G2_2,
    hamiltonian_pauli_0i_0a,
    hamiltonian_pauli_1i_1a,
)
from slowquant.qiskit_interface.wavefunction import WaveFunction


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

    def get_transition_dipole(self, dipole_integrals: Sequence[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def get_oscillator_strength(self, dipole_integrals: Sequence[np.ndarray]) -> np.ndarray:
        r"""Calculate oscillator strength.

        .. math::
            f_n = \frac{2}{3}e_n\left|\left<0\left|\hat{\mu}\right|n\left>\right|^2

        Args:
            dipole_integrals: Dipole integrals (x,y,z) in AO basis.

        Rerturns:
            Oscillator Strength.
        """
        transition_dipoles = self.get_transition_dipole(dipole_integrals)
        osc_strs = np.zeros(len(transition_dipoles))
        for idx, (excitation_energy, transition_dipole) in enumerate(
            zip(self.excitation_energies, transition_dipoles)
        ):
            osc_strs[idx] = (
                2
                / 3
                * excitation_energy
                * (transition_dipole[0] ** 2 + transition_dipole[1] ** 2 + transition_dipole[2] ** 2)
            )
        self.oscillator_strengths = osc_strs
        return osc_strs

    def get_formatted_oscillator_strength(self) -> str:
        """Create table of excitation energies and oscillator strengths.

        Returns:
            Nicely formatted table.
        """
        if not hasattr(self, "oscillator_strengths"):
            raise ValueError(
                "Oscillator strengths have not been calculated. Run get_oscillator_strength() first."
            )

        output = (
            "Excitation # | Excitation energy [Hartree] | Excitation energy [eV] | Oscillator strengths\n"
        )

        for i, (exc_energy, osc_strength) in enumerate(
            zip(self.excitation_energies, self.oscillator_strengths)
        ):
            exc_str = f"{exc_energy:2.6f}"
            exc_str_ev = f"{exc_energy*27.2114079527:3.6f}"
            osc_str = f"{osc_strength:1.6f}"
            output += f"{str(i+1).center(12)} | {exc_str.center(27)} | {exc_str_ev.center(22)} | {osc_str.center(20)}\n"
        return output


def get_num_nonCBS(matrix):
    count = 0
    dim = len(matrix[0])
    for i in range(dim):
        for j in range(i, dim):
            for paulis in matrix[i][j]:
                if any(letter in paulis for letter in ("X", "Y")):
                    count += 1
    return count


def get_num_CBS_elements(matrix):
    count_CBS = 0
    count_nCBS = 0
    dim = len(matrix[0])
    for i in range(dim):
        for j in range(i, dim):
            count_IM = 0
            for paulis in matrix[i][j]:
                if any(letter in paulis for letter in ("X", "Y")):
                    count_IM += 1
            if count_IM == 0 and not matrix[i][j] == "":
                count_CBS += 1
            elif count_IM > 0 and not matrix[i][j] == "":
                count_nCBS += 1
    return count_CBS, count_nCBS
