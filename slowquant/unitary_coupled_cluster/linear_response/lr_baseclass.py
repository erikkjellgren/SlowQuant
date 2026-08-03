import numpy as np
import scipy

from slowquant.unitary_coupled_cluster.ci_spaces import CI_Info
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import (
    G3,
    G4,
    G5,
    G6,
    G1_sa,
    G2_sa,
    G1,
    G2,
    hamiltonian_0i_0a,
    hamiltonian_1i_1a,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.util import (
    UccStructure,
    UpsStructure,
    iterate_t1_sa,
    iterate_t2_sa,
    iterate_t3,
    iterate_t4,
    iterate_t5,
    iterate_t6,
    iterate_t1,
    iterate_pair_t2,
    iterate_t2,
    iterate_t1_generalized,
    iterate_t1_sa_generalized,
    iterate_t2_generalized,
    iterate_t2_sa_generalized,
    iterate_pair_t2_generalized,
)


class LinearResponseBaseClass:
    index_info: tuple[CI_Info, list[float], UpsStructure] | tuple[CI_Info, list[float], UccStructure]

    def __init__(
        self,
        wave_function: WaveFunctionUCC | WaveFunctionUPS,
        excitations: list[str],
        do_orbital_response: bool,
    ) -> None:
        """Initialize linear response by calculating the needed matrices.

        Args:
            wave_function: Wave function object.
            excitations: Which excitation orders to include in response.
        """
        self.wf = wave_function
        if isinstance(self.wf, WaveFunctionUCC):
            self.index_info = (
                self.wf.ci_info,
                self.wf.thetas,
                self.wf.ucc_layout,
            )
        elif isinstance(self.wf, WaveFunctionUPS):
            self.index_info = (
                self.wf.ci_info,
                self.wf.thetas,
                self.wf.ups_layout,
            )
        else:
            raise ValueError(f"Got incompatible wave function type, {type(self.wf)}")

        self.G_ops: list[FermionicOperator] = []
        self.q_ops: list[FermionicOperator] = []
        excitations = [x.lower() for x in excitations]
        self._excitation_energies = None
        self._oscillator_strengths = None

        if "sas" in excitations:
            for a, i, _ in iterate_t1_sa(self.wf.active_occ_idx, self.wf.active_unocc_idx):
                self.G_ops.append(G1_sa(i, a))
        if "sags" in excitations:
            for a, i, _ in iterate_t1_sa_generalized(self.wf.num_orbs):
                self.G_ops.append(G1_sa(i, a))
        if "sad" in excitations:
            for a, i, b, j, _, op_type in iterate_t2_sa(self.wf.active_occ_idx, self.wf.active_unocc_idx):
                self.G_ops.append(G2_sa(i, j, a, b, op_type))
        if "sagd" in excitations:
            for a, i, b, j, _, op_type in iterate_t2_sa_generalized(self.wf.num_orbs):
                self.G_ops.append(G2_sa(i, j, a, b, op_type))
        if "s" in excitations:
            for a, i in iterate_t1(self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx):
                self.G_ops.append(G1(i, a))
        if "gs" in excitations:
            for a, i in iterate_t1_generalized(self.wf.num_spin_orbs):
                self.G_ops.append(G1(i, a))
        if "d" in excitations:
            for a, i, b, j in iterate_t2(self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx):
                self.G_ops.append(G2(i, j, a, b))
        if "gd" in excitations:
            for a, i, b, j in iterate_t2_generalized(self.wf.num_spin_orbs):
                self.G_ops.append(G2(i, j, a, b))
        if "pd" in excitations:
            for a, i, b, j in iterate_pair_t2(self.wf.active_occ_idx, self.wf.active_unocc_idx):
                self.G_ops.append(G2(i, j, a, b))
        if "gpd" in excitations:
            for a, i, b, j in iterate_pair_t2_generalized(self.wf.num_orbs):
                self.G_ops.append(G2(i, j, a, b))
        if "t" in excitations:
            for a, i, b, j, c, k in iterate_t3(self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx):
                self.G_ops.append(G3(i, j, k, a, b, c))
        if "q" in excitations:
            for a, i, b, j, c, k, d, l in iterate_t4(
                self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx
            ):
                self.G_ops.append(G4(i, j, k, l, a, b, c, d))
        if "5" in excitations:
            for a, i, b, j, c, k, d, l, e, m in iterate_t5(
                self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx
            ):
                self.G_ops.append(G5(i, j, k, l, m, a, b, c, d, e))
        if "6" in excitations:
            for a, i, b, j, c, k, d, l, e, m, f, n in iterate_t6(
                self.wf.active_occ_spin_idx, self.wf.active_unocc_spin_idx
            ):
                self.G_ops.append(G6(i, j, k, l, m, n, a, b, c, d, e, f))
        if do_orbital_response:
            for p, q in self.wf.kappa_no_activeactive_idx:
                self.q_ops.append(G1_sa(p, q))

        num_parameters = len(self.G_ops) + len(self.q_ops)
        self.A = np.zeros((num_parameters, num_parameters))
        self.B = np.zeros((num_parameters, num_parameters))
        self._B_is_zero = False
        self.Sigma = np.zeros((num_parameters, num_parameters))
        self.H_1i_1a = hamiltonian_1i_1a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
        )
        self.H_0i_0a = hamiltonian_0i_0a(
            self.wf.h_mo,
            self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
        )

    def _solve_linear_response(self) -> None:
        """Calculate excitation energies."""
        if self._B_is_zero:
            self.E2 = self.A
            hess_eigval, _ = np.linalg.eig(self.E2)
            print(f"Smallest Hessian eigenvalue: {np.min(hess_eigval)}")
            if np.abs(np.min(hess_eigval)) < 10**-8:
                print("WARNING: Small eigenvalue in Hessian")
            elif np.min(hess_eigval) < 0:
                raise ValueError("Negative eigenvalue in Hessian.")

            self.S2 = self.Sigma
            print(f"Smallest diagonal element in the metric: {np.min(np.abs(np.diagonal(self.Sigma)))}")

            eigval, eigvec = scipy.linalg.eig(self.E2, self.S2)
            sorting = np.argsort(eigval)
            self._eigenvalues = np.real(eigval[sorting])
            self.response_vectors = np.real(eigvec[:, sorting])
            self.normed_response_vectors = np.zeros_like(self.response_vectors)
            self.num_q = len(self.q_ops)
            self.num_G = len(self.A) - self.num_q
            self.Z_q = self.response_vectors[: self.num_q, :]
            self.Z_G = self.response_vectors[self.num_q : self.num_q + self.num_G, :]
            self.Z_q_normed = np.zeros_like(self.Z_q)
            self.Z_G_normed = np.zeros_like(self.Z_G)
            norms = self._get_excited_state_norm()
            for state_number, norm in enumerate(norms):
                if norm < 10**-10:
                    print(f"WARNING: State number {state_number} could not be normalized. Norm of {norm}.")
                    continue
                self.Z_q_normed[:, state_number] = self.Z_q[:, state_number] * (1 / norm) ** 0.5
                self.Z_G_normed[:, state_number] = self.Z_G[:, state_number] * (1 / norm) ** 0.5
                self.normed_response_vectors[:, state_number] = (
                    self.response_vectors[:, state_number] * (1 / norm) ** 0.5
                )

        else:
            size = len(self.A)
            self.E2 = np.zeros((size * 2, size * 2), dtype=float)
            self.E2[:size, :size] = self.A
            self.E2[:size, size:] = self.B
            self.E2[size:, :size] = self.B
            self.E2[size:, size:] = self.A
            hess_eigval, _ = np.linalg.eig(self.E2)
            print(f"Smallest Hessian eigenvalue: {np.min(hess_eigval)}")
            if np.abs(np.min(hess_eigval)) < 10**-8:
                print("WARNING: Small eigenvalue in Hessian")
            elif np.min(hess_eigval) < 0:
                raise ValueError("Negative eigenvalue in Hessian.")

            self.S2 = np.zeros((size * 2, size * 2), dtype=float)
            self.S2[:size, :size] = self.Sigma
            self.S2[size:, size:] = -self.Sigma
            print(f"Smallest diagonal element in the metric: {np.min(np.abs(np.diagonal(self.Sigma)))}")

            eigval, eigvec = scipy.linalg.eig(self.E2, self.S2)
            sorting = np.argsort(eigval)
            self._eigenvalues = np.real(eigval[sorting][size:])
            self.response_vectors = np.real(eigvec[:, sorting][:, size:])
            self.normed_response_vectors = np.zeros_like(self.response_vectors, dtype=float)
            self.num_q = len(self.q_ops)
            self.num_G = size - self.num_q
            self.Z_q = self.response_vectors[: self.num_q, :]
            self.Z_G = self.response_vectors[self.num_q : self.num_q + self.num_G, :]
            self.Y_q = self.response_vectors[self.num_q + self.num_G : 2 * self.num_q + self.num_G]
            self.Y_G = self.response_vectors[2 * self.num_q + self.num_G :]
            self.Z_q_normed = np.zeros_like(self.Z_q)
            self.Z_G_normed = np.zeros_like(self.Z_G)
            self.Y_q_normed = np.zeros_like(self.Y_q)
            self.Y_G_normed = np.zeros_like(self.Y_G)
            norms = self._get_excited_state_norm()
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

    def _get_excited_state_norm(self) -> np.ndarray:
        """Calculate the norm of excited states.

        Returns:
            Norm of excited states.
        """
        norms = np.zeros(len(self.response_vectors[0]))
        for state_number in range(len(self.response_vectors[0])):
            if self._B_is_zero:
                # Get Z_q and Z_G matrices
                ZZq = np.outer(self.Z_q[:, state_number], self.Z_q[:, state_number].transpose())
                ZZG = np.outer(self.Z_G[:, state_number], self.Z_G[:, state_number].transpose())
                norms[state_number] = np.sum(self.S2[: self.num_q, : self.num_q] * ZZq) + np.sum(
                    self.S2[self.num_q : self.num_q + self.num_G, self.num_q : self.num_q + self.num_G]
                    * ZZG
                )
            else:
                # Get Z_q Z_G Y_q and Y_G matrices
                ZZq = np.outer(self.Z_q[:, state_number], self.Z_q[:, state_number].transpose())
                YYq = np.outer(self.Y_q[:, state_number], self.Y_q[:, state_number].transpose())
                ZZG = np.outer(self.Z_G[:, state_number], self.Z_G[:, state_number].transpose())
                YYG = np.outer(self.Y_G[:, state_number], self.Y_G[:, state_number].transpose())
                norms[state_number] = np.sum(self.S2[: self.num_q, : self.num_q] * (ZZq - YYq)) + np.sum(
                    self.S2[self.num_q : self.num_q + self.num_G, self.num_q : self.num_q + self.num_G]
                    * (ZZG - YYG)
                )
        return norms

    def _get_transitation_property(self, property_integral: np.ndarray) -> np.ndarray:
        """Function to be overwritten in LR subclasses."""
        raise NotImplementedError

    def get_transition_property(self, property_integral: np.ndarray) -> np.ndarray:
        """Calculate transition property.

        Args:
            property_integral: Property integral.

        Returns:
            Transition property.
        """
        if not hasattr(self, "_eigenvalues"):
            # Solve linear response if not solved yet.
            self._solve_linear_response()
        return self._get_transitation_property(property_integral)

    @property
    def excitation_energies(self) -> np.ndarray:
        """Calculate excitation energies.

        Returns:
            Excitation energies.
        """
        if self._excitation_energies is None:
            self._solve_linear_response()
            self._excitation_energies = self._eigenvalues
        return self._excitation_energies

    @property
    def oscillator_strengths(self) -> np.ndarray:
        r"""Calculate oscillator strength.

        .. math::
            f_n = \frac{2}{3}e_n\left|\left<0\left|\hat{\mu}\right|n\right>\right|^2

        Returns:
            Oscillator Strength.
        """
        if self._oscillator_strengths is None:
            dipole_integrals = self.wf.int_gen.electric_dipole
            transition_dipoles = np.zeros((len(self.excitation_energies), 3), dtype=float) 
            transition_dipoles[:,0] = self.get_transition_property(dipole_integrals[0])
            transition_dipoles[:,1] = self.get_transition_property(dipole_integrals[1])
            transition_dipoles[:,2] = self.get_transition_property(dipole_integrals[2])
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
            self._oscillator_strengths = osc_strs
        return self._oscillator_strengths

    @property
    def get_formatted_oscillator_strength(self) -> str:
        """Create table of excitation energies and oscillator strengths.

        Returns:
            Nicely formatted table.
        """
        output = (
            "Excitation # | Excitation energy [Hartree] | Excitation energy [eV] | Oscillator strengths\n"
        )

        for i, (exc_energy, osc_strength) in enumerate(
            zip(self.excitation_energies, self.oscillator_strengths)
        ):
            exc_str = f"{exc_energy:2.6f}"
            exc_str_ev = f"{exc_energy * 27.2114079527:3.6f}"
            osc_str = f"{osc_strength:1.6f}"
            output += f"{str(i + 1).center(12)} | {exc_str.center(27)} | {exc_str_ev.center(22)} | {osc_str.center(20)}\n"
        return output


