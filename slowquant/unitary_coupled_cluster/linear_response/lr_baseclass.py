from collections.abc import Sequence

import numpy as np
import scipy

from slowquant.unitary_coupled_cluster.operator_hybrid import (
    OperatorHybrid,
    convert_pauli_to_hybrid_form,
    hamiltonian_hybrid_0i_0a,
    hamiltonian_hybrid_1i_1a,
)
from slowquant.unitary_coupled_cluster.operator_pauli import epq_pauli
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.util import ThetaPicker


class LinearResponseBaseClass:
    A: np.ndarray
    B: np.ndarray
    Sigma: np.ndarray
    Delta: np.ndarray
    wf: WaveFunctionUCC

    def __init__(
        self,
        wave_function: WaveFunctionUCC,
        excitations: str,
        is_spin_conserving: bool = True,
    ) -> None:
        """Initialize linear response by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
            is_spin_conserving: Use spin-conseving operators.
        """
        self.wf = wave_function
        self.theta_picker = ThetaPicker(
            self.wf.active_occ_spin_idx,
            self.wf.active_unocc_spin_idx,
            is_spin_conserving=is_spin_conserving,
        )

        self.G_ops: list[OperatorHybrid] = []
        self.q_ops: list[OperatorHybrid] = []
        num_spin_orbs = self.wf.num_spin_orbs
        excitations = excitations.lower()

        if "s" in excitations:
            for _, _, _, op_ in self.theta_picker.get_t1_generator_sa(num_spin_orbs):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                )
                self.G_ops.append(op)
        if "d" in excitations:
            for _, _, _, _, _, op_ in self.theta_picker.get_t2_generator_sa(num_spin_orbs):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                )
                self.G_ops.append(op)
        if "t" in excitations:
            for _, _, _, _, _, _, _, op_ in self.theta_picker.get_t3_generator(num_spin_orbs):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                )
                self.G_ops.append(op)
        if "q" in excitations:
            for _, _, _, _, _, _, _, _, _, op_ in self.theta_picker.get_t4_generator(num_spin_orbs):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                )
                self.G_ops.append(op)
        if "5" in excitations:
            for _, _, _, _, _, _, _, _, _, _, _, op_ in self.theta_picker.get_t5_generator(num_spin_orbs):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                )
                self.G_ops.append(op)
        if "6" in excitations:
            for _, _, _, _, _, _, _, _, _, _, _, _, _, op_ in self.theta_picker.get_t6_generator(
                num_spin_orbs
            ):
                op = convert_pauli_to_hybrid_form(
                    op_,
                    self.wf.num_inactive_spin_orbs,
                    self.wf.num_active_spin_orbs,
                )
                self.G_ops.append(op)
        for i, a in self.wf.kappa_idx:
            op_ = 2 ** (-1 / 2) * epq_pauli(a, i, self.wf.num_spin_orbs)
            op = convert_pauli_to_hybrid_form(
                op_,
                self.wf.num_inactive_spin_orbs,
                self.wf.num_active_spin_orbs,
            )
            self.q_ops.append(op)

        num_parameters = len(self.G_ops) + len(self.q_ops)
        self.A = np.zeros((num_parameters, num_parameters))
        self.B = np.zeros((num_parameters, num_parameters))
        self.Sigma = np.zeros((num_parameters, num_parameters))
        self.Delta = np.zeros((num_parameters, num_parameters))
        self.H_1i_1a = hamiltonian_hybrid_1i_1a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
        )
        self.H_0i_0a = hamiltonian_hybrid_0i_0a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
        )

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
        if np.min(hess_eigval) < 0:
            raise ValueError("Negative eigenvalue in Hessian.")

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
        num_q = len(self.q_ops)
        num_G = size - num_q
        self.Z_q = self.response_vectors[:num_q, :]
        self.Z_G = self.response_vectors[num_q : num_q + num_G, :]
        self.Y_q = self.response_vectors[num_q + num_G : 2 * num_q + num_G]
        self.Y_G = self.response_vectors[2 * num_q + num_G :]
        self.Z_q_normed = np.zeros_like(self.Z_q)
        self.Z_G_normed = np.zeros_like(self.Z_G)
        self.Y_q_normed = np.zeros_like(self.Y_q)
        self.Y_G_normed = np.zeros_like(self.Y_G)
        norms = self.get_excited_state_norm()
        print(norms)
        for state_number, norm in enumerate(norms):
            if norm < 10**-10:
                print(f"WARNING: State number {state_number} could not be normalized. Norm of {norm}.")
                continue
            self.Z_q_normed[:, state_number] = self.Z_q[:, state_number] * (1 / norm) ** 0.5
            self.Z_G_normed[:, state_number] = self.Z_G[:, state_number] * (1 / norm) ** 0.5
            self.Y_q_normed[:, state_number] = self.Y_q[:, state_number] * (1 / norm) ** 0.5
            self.Y_G_normed[:, state_number] = self.Y_G[:, state_number] * (1 / norm) ** 0.5
            self.normed_response_vectors[:, state_number] = (
                self.response_vectors[:, state_number] * (1 / norm) ** 0.5
            )

    def get_excited_state_norm(self) -> np.ndarray:
        raise NotImplementedError

    def get_excited_state_overlap(self) -> np.ndarray:
        raise NotImplementedError

    def get_property_gradient(
        self, property_integrals: Sequence[np.ndarray], is_imag_op: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def get_oscillator_strengths(self, dipole_integrals: Sequence[np.ndarray]) -> np.ndarray:
        r"""Calculate oscillator strengths.

        .. math::
            f_n = \frac{2}{3}e_n\left|\left<0\left|\hat{\mu}\right|n\left>\right|^2

        Args:
            dipole_integrals: Dipole integrals (x,y,z) in AO basis.

        Rerturns:
            Oscillator strengths.
        """
        transition_dipoles = self.get_property_gradient(dipole_integrals)
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

    def get_rotational_strengths(
        self, dipole_integrals: Sequence[np.ndarray], magnetic_moment_integrals: Sequence[np.ndarray]
    ) -> np.ndarray:
        r"""Calculate rotational strengths.

        .. math::
            R_n = e_n\left<0\left|\hat{\mu}\right|n\left>\cdot\left<n\left|\hat{m}\right|0\left>

        Args:
            dipole_integrals: Dipole integrals (x,y,z) in AO basis.
            magnetic_moment_integrals: Magnetic moment integrals (x,y,z) in AO basis.

        Rerturns:
            Rotational strengths.
        """
        transition_electric_dipoles = self.get_property_gradient(dipole_integrals)
        transition_magnetic_dipoles = self.get_property_gradient(magnetic_moment_integrals, is_imag_op=True)
        rot_strengths = np.zeros(len(transition_electric_dipoles))
        for idx, (excitation_energy, transition_electric_dipole, transition_magnetic_dipole) in enumerate(
            zip(self.excitation_energies, transition_electric_dipoles, transition_magnetic_dipoles)
        ):
            rot_strengths[idx] = (
                transition_electric_dipole[0] * transition_magnetic_dipole[0]
                + transition_electric_dipole[1] * transition_magnetic_dipole[1]
                + transition_electric_dipole[2] * transition_magnetic_dipole[2]
            )
        print("rot str")
        with np.printoptions(precision=4, suppress=True):
            print(0.5 * transition_magnetic_dipoles)
        print("osc str")
        with np.printoptions(precision=4, suppress=True):
            print(transition_electric_dipoles)

        return rot_strengths

    def get_uvvis_output(self, dipole_integrals: Sequence[np.ndarray]) -> str:
        """Create table of excitation energies and oscillator strengths.

        Args:
            dipole_integrals: Dipole integrals (x,y,z) in AO basis.

        Returns:
            Nicely formatted table.
        """
        output = (
            "Excitation # | Excitation energy [Hartree] | Excitation energy [eV] | Oscillator strengths\n"
        )
        osc_strengths = self.get_oscillator_strengths(dipole_integrals)
        for i, (exc_energy, osc_strength) in enumerate(zip(self.excitation_energies, osc_strengths)):
            exc_str = f"{exc_energy:2.6f}"
            exc_str_ev = f"{exc_energy*27.2114079527:3.6f}"
            osc_str = f"{osc_strength:1.6f}"
            output += f"{str(i+1).center(12)} | {exc_str.center(27)} | {exc_str_ev.center(22)} | {osc_str.center(20)}\n"
        return output

    def get_ecd_output(
        self, dipole_integrals: Sequence[np.ndarray], magnetic_moment_integrals: Sequence[np.ndarray]
    ) -> str:
        """Create table of excitation energies and rotational strengths.

        Args:
            dipole_integrals: Dipole integrals (x,y,z) in AO basis.
            magnetic_moment_integrals: Magnetic moment integrals (x,y,z) in AO basis.

        Returns:
            Nicely formatted table.
        """
        output = (
            "Excitation # | Excitation energy [Hartree] | Excitation energy [eV] | Rotational strengths\n"
        )
        rot_strengths = self.get_rotational_strengths(dipole_integrals, magnetic_moment_integrals)
        for i, (exc_energy, rot_strength) in enumerate(zip(self.excitation_energies, rot_strengths)):
            exc_str = f"{exc_energy:2.6f}"
            exc_str_ev = f"{exc_energy*27.2114079527:3.6f}"
            rot_str = f"{rot_strength:1.6f}"
            output += f"{str(i+1).center(12)} | {exc_str.center(27)} | {exc_str_ev.center(22)} | {rot_str.center(20)}\n"
        return output
