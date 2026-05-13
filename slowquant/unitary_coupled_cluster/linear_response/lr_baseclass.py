import numpy as np
import scipy

from slowquant.molecularintegrals.integralfunctions import one_electron_integral_transform
from slowquant.unitary_coupled_cluster.ci_spaces import CI_Info
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.operators import (
    G3,
    G4,
    G5,
    G6,
    G1_sa,
    G2_sa,
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
)
from slowquant.unitary_coupled_cluster.linear_response.solvers import PairedDavidson

class LinearResponseBaseClass:
    index_info: tuple[CI_Info, list[float], UpsStructure] | tuple[CI_Info, list[float], UccStructure]

    def __init__(
        self,
        wave_function: WaveFunctionUCC | WaveFunctionUPS,
        excitations: str,
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
        excitations = excitations.lower()

        if "s" in excitations:
            for a, i, _ in iterate_t1_sa(self.wf.active_occ_idx, self.wf.active_unocc_idx):
                self.G_ops.append(G1_sa(i, a))
        if "d" in excitations:
            for a, i, b, j, _, op_type in iterate_t2_sa(self.wf.active_occ_idx, self.wf.active_unocc_idx):
                self.G_ops.append(G2_sa(i, j, a, b, op_type))
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
        for p, q in self.wf.kappa_no_activeactive_idx:
            self.q_ops.append(G1_sa(p, q))

        num_parameters = len(self.G_ops) + len(self.q_ops)
        self.A = np.zeros((num_parameters, num_parameters))
        self.B = np.zeros((num_parameters, num_parameters))
        self.Sigma = np.zeros((num_parameters, num_parameters))
        self.Delta = np.zeros((num_parameters, num_parameters))
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

    def _construct_hessian_metric_blocks(self):
        """Construct Hessian and metric blocks."""
        raise NotImplementedError

    def _compute_preconditioner(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute preconditioner for Davidson solver."""
        raise NotImplementedError

    def _right_transform(self, trial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Right transform for Davidson solver.

        Args:
            trial: Trial vectors.
        """
        raise NotImplementedError

    def linear_response_function(self, frequency: float, lr_property: str, solver_settings: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Calculate linear response function at given frequency.

        Args:
            frequency: Frequency to calculate response function at.
            lr_property: Property for which to calculate the response function.
                Dipole Polarizability can be calculated by setting lr_property to "dipole polarizability" or "dp".
            solver_settings: Settings for the Davidson solver:
                max_iteration: Maximum number of iterations. Default is 100.
                tolerance: Convergence tolerance. Default is 1e-8.
                max_reduced_space: Maximum size of the reduced space. Default is 8 times number of roots for the lr_property.
                is_silent: Whether to print convergence information. Default is False.
                _start_guess: Optional starting guess for the response vector. Default is None.

        Returns:
            Linear response function at given frequency.
        """
        if solver_settings is None:
            solver_settings = {}
        if lr_property.lower() in ("dipole polarizability", "dp"):
            title_string = f"Calculating dipole polarizability at frequency {frequency} a.u."
            integrals = self.wf.int_gen.electric_dipole
            property_gradient_x = self.property_gradient(one_electron_integral_transform(self.wf.c_mo, integrals[0]))
            property_gradient_y = self.property_gradient(one_electron_integral_transform(self.wf.c_mo, integrals[1]))
            property_gradient_z = self.property_gradient(one_electron_integral_transform(self.wf.c_mo, integrals[2]))
            property_gradient = np.hstack([property_gradient_x, property_gradient_y, property_gradient_z])
            full_gradient = np.vstack((
                property_gradient.reshape(len(property_gradient), -1),
                -property_gradient.reshape(len(property_gradient), -1)
            ))
        else:
            raise ValueError(f"Unknown property {lr_property} for linear response function.")

        print()
        print(title_string)

        n_roots = property_gradient.shape[-1]
        preconditioner = self._compute_preconditioner()
        solver = PairedDavidson()
        _, response_vectors = (
            solver.solve(
                self._right_transform,
                preconditioner,
                max_iteration=solver_settings.get("max_iteration", 100),
                tolerance=solver_settings.get("tolerance", 1e-8),
                n_roots=n_roots,
                max_reduced_space=solver_settings.get("max_reduced_space", n_roots * 8),
                frequency=frequency,
                property_gradient=property_gradient,
                is_silent=solver_settings.get("is_silent", False),
                _start_guess=solver_settings.get("_start_guess", None),
            )
        )

        return response_vectors, full_gradient

    def property_gradient(self, integral: np.ndarray) -> np.ndarray:
        """Calculate gradient of property.

        Args:
            trial: Trial vectors.
            integral: Integral matrix.

        Returns:
            Gradient of property.
        """
        raise NotImplementedError

    def calc_excitation_energies(self, n_roots: int = 0, solver_settings: dict | None = None) -> None:
        """Calculate excitation energies.

        Args:
            n_roots: Number of roots to calculate. If 0, calculate all roots.
            solver_settings: Settings for the Davidson solver:
                max_iteration: Maximum number of iterations. Default is 100.
                tolerance: Convergence tolerance. Default is 1e-8.
                max_reduced_space: Maximum size of the reduced space. Default is 8*n_roots.
                is_silent: Whether to print convergence information. Default is False.
        """
        if n_roots <= 0:
            self._all_excitation_energies()
        else:
            solver = PairedDavidson()
            if solver_settings is None:
                solver_settings = {}

            preconditioner = self._compute_preconditioner()

            self.excitation_energies, self.normed_response_vectors = (
                solver.solve(
                    self._right_transform,
                    preconditioner,
                    max_iteration=solver_settings.get("max_iteration", 100),
                    tolerance=solver_settings.get("tolerance", 1e-8),
                    n_roots=n_roots,
                    max_reduced_space=solver_settings.get("max_reduced_space", 8*n_roots),
                    is_silent=solver_settings.get("is_silent", False),
                )
            )
            self.Z_q_normed = self.normed_response_vectors[: len(self.q_ops), :]
            self.Z_G_normed = self.normed_response_vectors[len(self.q_ops) : len(self.q_ops) + len(self.G_ops), :]
            self.Y_q_normed = self.normed_response_vectors[len(self.q_ops) + len(self.G_ops) : 2 * len(self.q_ops) + len(self.G_ops), :]
            self.Y_G_normed = self.normed_response_vectors[2 * len(self.q_ops) + len(self.G_ops) :, :]

    def _all_excitation_energies(self):
        self._construct_hessian_metric_blocks()

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
        if np.abs(np.min(hess_eigval)) < 10**-8:
            print("WARNING: Small eigenvalue in Hessian")
        elif np.min(hess_eigval) < 0:
            raise ValueError("Negative eigenvalue in Hessian.")

        S = np.zeros((size * 2, size * 2))
        S[:size, :size] = self.Sigma
        S[:size, size:] = self.Delta
        S[size:, :size] = -self.Delta
        S[size:, size:] = -self.Sigma
        print(f"Smallest diagonal element in the metric: {np.min(np.abs(np.diagonal(self.Sigma)))}")

        self.hessian = E2
        self.metric = S

        eigval, eigvec = scipy.linalg.eig(self.hessian, self.metric)
        sorting = np.argsort(eigval)
        self.excitation_energies = np.real(eigval[sorting][size:])
        self.response_vectors = np.real(eigvec[:, sorting][:, size:])
        self.normed_response_vectors = np.zeros_like(self.response_vectors)
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
        norms = self.get_excited_state_norm()
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
        """Calculate the norm of excited states.

        Returns:
            Norm of excited states.
        """
        norms = np.zeros(len(self.response_vectors[0]))
        for state_number in range(len(self.response_vectors[0])):
            # Get Z_q Z_G Y_q and Y_G matrices
            ZZq = np.outer(self.Z_q[:, state_number], self.Z_q[:, state_number].transpose())
            YYq = np.outer(self.Y_q[:, state_number], self.Y_q[:, state_number].transpose())
            ZZG = np.outer(self.Z_G[:, state_number], self.Z_G[:, state_number].transpose())
            YYG = np.outer(self.Y_G[:, state_number], self.Y_G[:, state_number].transpose())

            norms[state_number] = np.sum(self.metric[: self.num_q, : self.num_q] * (ZZq - YYq)) + np.sum(
                self.metric[self.num_q : self.num_q + self.num_G, self.num_q : self.num_q + self.num_G]
                * (ZZG - YYG)
            )

        return norms

    def get_transition_dipole(self) -> np.ndarray:
        """Calculate transition dipole moment.

        Returns:
            Transition dipole moment.
        """
        raise NotImplementedError

    def get_oscillator_strength(self) -> np.ndarray:
        r"""Calculate oscillator strength.

        .. math::
            f_n = \frac{2}{3}e_n\left|\left<0\left|\hat{\mu}\right|n\right>\right|^2

        Returns:
            Oscillator Strength.
        """
        transition_dipoles = self.get_transition_dipole()
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
            exc_str_ev = f"{exc_energy * 27.2114079527:3.6f}"
            osc_str = f"{osc_strength:1.6f}"
            output += f"{str(i + 1).center(12)} | {exc_str.center(27)} | {exc_str_ev.center(22)} | {osc_str.center(20)}\n"
        return output
