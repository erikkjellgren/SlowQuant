from collections.abc import Sequence

import numpy as np
import scipy


class LinearResponseBaseClass:
    A: np.ndarray
    B: np.ndarray
    Sigma: np.ndarray
    Delta: np.ndarray

    def __init__(self) -> None:
        raise NotImplementedError

    def calc_excitation_energies(self) -> None:
        """Calculate excitation energies."""
        size = len(self.A)
        E2 = np.zeros((size * 2, size * 2))
        E2[:size, :size] = self.A
        E2[:size, size:] = self.B
        E2[size:, :size] = self.B
        E2[size:, size:] = self.A
        (
            hess_eigval,
            _,
        ) = np.linalg.eig(E2)
        print(f"Smallest Hessian eigenvalue: {np.min(hess_eigval)}")

        S = np.zeros((size * 2, size * 2))
        S[:size, :size] = self.Sigma
        S[:size, size:] = self.Delta
        S[size:, :size] = -self.Delta
        S[size:, size:] = -self.Sigma
        print(f"Smallest diagonal element in the metric: {np.min(np.abs(np.diagonal(self.Sigma)))}")

        eigval, eigvec = scipy.linalg.eig(E2, S)
        sorting = np.argsort(eigval)
        self.excitation_energies = np.real(eigval[sorting][size:])
        self.response_vectors = np.real(eigvec[:, sorting][:, size:])
        self.normed_response_vectors = np.zeros_like(self.response_vectors)
        norms = self.get_excited_state_norm()
        for state_number, norm in enumerate(norms):
            if norm < 10**-10:
                continue
            self.normed_response_vectors[:, state_number] = (
                self.response_vectors[:, state_number] * (1 / norm) ** 0.5
            )

    def get_excited_state_norm(self) -> np.ndarray:
        raise NotImplementedError

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
        return osc_strs

    def get_nice_output(self, dipole_integrals: Sequence[np.ndarray]) -> str:
        """Create table of excitation energies and oscillator strengths.

        Args:
            dipole_integrals: Dipole integrals (x,y,z) in AO basis.

        Returns:
            Nicely formatted table.
        """
        output = (
            "Excitation # | Excitation energy [Hartree] | Excitation energy [eV] | Oscillator strengths\n"
        )
        osc_strs = self.get_oscillator_strength(dipole_integrals)
        for i, (exc_energy, osc_strength) in enumerate(zip(self.excitation_energies, osc_strs)):
            exc_str = f"{exc_energy:2.6f}"
            exc_str_ev = f"{exc_energy*27.2114079527:3.6f}"
            osc_str = f"{osc_strength:1.6f}"
            output += f"{str(i+1).center(12)} | {exc_str.center(27)} | {exc_str_ev.center(22)} | {osc_str.center(20)}\n"
        return output
