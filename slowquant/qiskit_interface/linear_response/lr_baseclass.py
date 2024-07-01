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
        """Initialize linear response by calculating the needed matrices.

        Args:
            wf: Wavefunction object.
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
        for a, i, b, j, _, _, op_id in iterate_t2_sa(wf.active_occ_idx, wf.active_unocc_idx):
            if op_id > 0:
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

    def get_operator_info(self) -> None:
        """Information about operators."""
        # q operators
        print(f"{'Annihilation'.center(12)} | {'Creation'.center(12)} | {'Coefficient'.center(12)}\n")
        if self.num_q > 0:
            print("Orbital rotation operators:")
            for nr, op in enumerate(self.q_ops):
                annihilation, creation, coefficients = op.get_info()
                print("q" + str(nr) + ":")
                for a, c, coeff in zip(annihilation, creation, coefficients):
                    if a[0] in self.wf.active_spin_idx:
                        exc_type = "active -> virtual"
                    elif c[0] in self.wf.active_spin_idx:
                        exc_type = "inactive -> active"
                    else:
                        exc_type = "inactive -> virtual"
                    print(
                        str(a).center(12)
                        + " | "
                        + str(c).center(12)
                        + " | "
                        + f"{coeff:.3f}".center(12)
                        + " | "
                        + exc_type
                    )

        if self.num_G > 0:
            print("Active-space excitation operators:")
            for nr, op in enumerate(self.G_ops):
                annihilation, creation, coefficients = op.get_info()
                print("G" + str(nr) + ":")
                for a, c, coeff in zip(annihilation, creation, coefficients):
                    print(str(a).center(12) + " | " + str(c).center(12) + " | " + f"{coeff:.3f}".center(12))

    def run(self) -> None:
        """Run linear response."""
        raise NotImplementedError

    def _get_qbitmap(self) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
        """Get qbitmapping of operators."""
        raise NotImplementedError

    def run_std(
        self,
        no_coeffs: bool = False,
        verbose: bool = True,
        cv: bool = True,
        save: bool = False,
    ) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
        """Get standard deviation in matrix elements of LR equation.

        Args:
            no_coeffs:  Boolean to no include coefficiants
            verbose:    Boolean to print more info
            cv:         Boolean to calculate coefficient of variance
            save:       Boolean to save operator-specific standard deviations

        Returns:
            Array of standard deviations for A, B and Sigma
        """
        raise NotImplementedError

    def _analyze_std(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Sigma: np.ndarray,
        max_values: int = 4,
        verbose: bool = True,
        cv: bool = True,
        save: bool = False,
    ) -> None:
        """Analyze standard deviation in matrix elements of LR equation."""
        print("\n Condition numbers:\n")
        print(f"Hessian: {np.linalg.cond(self.hessian)}")
        print(f"A      : {np.linalg.cond(self.A)}")
        print(f"B      : {np.linalg.cond(self.B)}")
        print(f"Metric : {np.linalg.cond(self.metric)}")
        print(f"Sigma  : {np.linalg.cond(self.Sigma)}")
        print(f"S-1E   : {np.linalg.cond(np.linalg.inv(self.metric)@self.hessian)}")

        print("\n Quantum variance analysis:")
        matrix_name = ["A", "B", "Sigma"]
        for nr, matrix in enumerate([np.abs(A), np.abs(B), np.abs(Sigma)]):
            print(f"\nAnalysis of {matrix_name[nr]}")
            print(f"The average standard deviation is {(np.sum(matrix) / (self.num_params**2))}")
            print(f"Maximum standard deviations are of value {np.sort(matrix.flatten())[::-1][:max_values]}")
            indices = np.unravel_index(np.argsort(matrix.flatten())[::-1][:max_values], matrix.shape)
            print("These maximum values are in:")
            for i in range(max_values):
                area = ""
                if indices[0][i] < self.num_q:
                    area += "q"
                else:
                    area += "G"
                if indices[1][i] < self.num_q:
                    area += "q"
                else:
                    area += "G"
                print(f"Indices {indices[0][i],indices[1][i]}. Part of matrix block {area}")
        if verbose:
            A_row = np.sum(A, axis=1) / self.num_params
            B_row = np.sum(B, axis=1) / self.num_params
            Sigma_row = np.sum(Sigma, axis=1) / self.num_params
            if save:
                self._std_A_row = A_row
                self._std_B_row = B_row
                self._std_Sigma_row = Sigma_row
            else:
                print("\nStandard deviation in each operator row for E | A | B | Sigma")
                for nr, i in enumerate(range(self.num_params)):
                    if nr < self.num_q:
                        print(
                            f"q{str(nr):<{3}}:"
                            + f"{(A_row[nr]+B_row[nr])/2:3.6f}".center(10)
                            + " | "
                            + f"{A_row[nr]:3.6f}".center(10)
                            + " | "
                            + f"{B_row[nr]:3.6f}".center(10)
                            + " | "
                            f"{Sigma_row[nr]:3.6f}".center(10)
                        )
                    else:
                        print(
                            f"G{str(nr-self.num_q):<{3}}:"
                            + f"{(A_row[nr]+B_row[nr])/2:3.6f}".center(10)
                            + " | "
                            + f"{A_row[nr]:3.6f}".center(10)
                            + " | "
                            + f"{B_row[nr]:3.6f}".center(10)
                            + " | "
                            f"{Sigma_row[nr]:3.6f}".center(10)
                        )

        if cv:
            print("\n Coefficient of variation:")
            if np.all(self.A == 0):
                print("Expectation values are needed for coefficient of variation. Running qLR")
                self.run()
            A_cv = np.abs(A / self.A)
            B_cv = np.abs(B / self.B)
            Sigma_cv = np.abs(Sigma / self.Sigma)
            A_cv[np.isnan(A_cv)] = 0
            B_cv[np.isnan(B_cv)] = 0
            Sigma_cv[np.isnan(Sigma_cv)] = 0
            # disregard values smaller 10**-10
            mask = [np.abs(self.A) >= 10**-10, np.abs(self.B) >= 10**-10, np.abs(self.Sigma) >= 10**-10]
            for nr, matrix in enumerate([A_cv, B_cv, Sigma_cv]):
                print(f"\nAnalysis of {matrix_name[nr]}")
                print(f"The average CV is {(np.sum(matrix[mask[nr]]) / (np.sum(mask[nr]))):3.6f}")
                if np.all(mask[nr]):
                    print(f"Maximum CV are of value {np.sort(matrix.flatten())[::-1][:max_values]}")
                    indices = np.unravel_index(np.argsort(matrix.flatten())[::-1][:max_values], matrix.shape)
                    print("These maximum values are in:")
                    for i in range(max_values):
                        area = ""
                        if indices[0][i] < self.num_q:
                            area += "q"
                        else:
                            area += "G"
                        if indices[1][i] < self.num_q:
                            area += "q"
                        else:
                            area += "G"
                        print(f"Indices {indices[0][i],indices[1][i]}. Part of matrix block {area}")

            if verbose:
                if np.all(mask):
                    A_row = np.sum(A_cv, axis=1) / self.num_params
                    B_row = np.sum(B_cv, axis=1) / self.num_params
                    Sigma_row = np.sum(Sigma_cv, axis=1) / self.num_params
                elif (
                    np.sum(mask[0]) < self.num_params
                    or np.sum(mask[1]) < self.num_params
                    or np.sum(mask[2]) < self.num_params
                ):
                    print("CV per operator analysis not possible.")
                    return
                else:
                    print(mask)
                    A_row = A_cv[mask[0]]
                    B_row = B_cv[mask[1]]
                    Sigma_row = Sigma_cv[mask[2]]
                    # Check if it is still a matrix
                    if len(np.shape(A_row)) >= 2:
                        A_row = np.sum(A_row, axis=1) / np.sum(mask[0], axis=1)
                    else:
                        A_row = A_row / np.sum(mask[0]) * 2
                    if len(np.shape(B_row)) >= 2:
                        B_row = np.sum(B_row, axis=1) / np.sum(mask[1], axis=1)
                    else:
                        B_row = B_row / np.sum(mask[1]) * 2
                    if len(np.shape(Sigma_row)) >= 2:
                        Sigma_row = np.sum(Sigma_row, axis=1) / np.sum(mask[2], axis=1)
                    else:
                        Sigma_row = Sigma_row / np.sum(mask[2]) * 2
                if save:
                    self._CV_A_row = A_row
                    self._CV_B_row = B_row
                    self._CV_Sigma_row = Sigma_row
                else:
                    print("\nCV in each operator row for E | A | B | Sigma")
                    for nr, i in enumerate(range(self.num_params)):
                        if nr < self.num_q:
                            print(
                                f"q{str(nr):<{3}}:"
                                + f"{(A_row[nr]+B_row[nr])/2:3.6f}".center(10)
                                + " | "
                                + f"{A_row[nr]:3.6f}".center(10)
                                + " | "
                                + f"{B_row[nr]:3.6f}".center(10)
                                + " | "
                                f"{Sigma_row[nr]:3.6f}".center(10)
                            )
                        else:
                            print(
                                f"G{str(nr-self.num_q):<{3}}:"
                                + f"{(A_row[nr]+B_row[nr])/2:3.6f}".center(10)
                                + " | "
                                + f"{A_row[nr]:3.6f}".center(10)
                                + " | "
                                + f"{B_row[nr]:3.6f}".center(10)
                                + " | "
                                f"{Sigma_row[nr]:3.6f}".center(10)
                            )

    def get_excitation_energies(self) -> np.ndarray:
        """Solve LR eigenvalue problem."""
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

        # Check eigenvalues of Hessian/Metric
        (
            hess_eigval,
            _,
        ) = scipy.linalg.eig(self.hessian)
        print(f"Smallest Hessian eigenvalue: {np.min(hess_eigval)}")
        if np.min(hess_eigval) < 0:
            print("WARNING: Negative eigenvalue in Hessian.")
        print(f"Smallest diagonal element in the metric: {np.min(np.abs(np.diagonal(self.metric)))}")

        # Solve eigenvalue equation
        eigval, eigvec = scipy.linalg.eig(self.hessian, self.metric)
        sorting = np.argsort(eigval)
        self.excitation_energies = np.real(eigval[sorting][size:])
        self.excitation_vectors = np.real(eigvec[:, sorting][:, size:])

        return self.excitation_energies

    def get_normed_excitation_vectors(self) -> None:
        """Get normed excitation vectors via excitated state norm."""
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
        """Calculate transtition dipole moment.

        Args:
            dipole_integrals: Dipole integrals (x,y,z) in AO basis.

        Returns:
            Transition dipole moments.
        """
        raise NotImplementedError

    def get_oscillator_strength(self, dipole_integrals: Sequence[np.ndarray]) -> np.ndarray:
        r"""Calculate oscillator strength.

        .. math::
            f_n = \frac{2}{3}e_n\left|\left<0\left|\hat{\mu}\right|n\right>\right|^2

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
            exc_str_ev = f"{exc_energy * 27.2114079527:3.6f}"
            osc_str = f"{osc_strength:1.6f}"
            output += f"{str(i + 1).center(12)} | {exc_str.center(27)} | {exc_str_ev.center(22)} | {osc_str.center(20)}\n"
        return output

    def get_excited_state_contributions(self, num_contr: int | None = None, cutoff: float = 10**-2) -> None:
        """Create table of contributions to each excitation vector.

        Returns:
            Nicely formatted table.
        """
        if not hasattr(self, "excitation_vectors"):
            raise ValueError("Excitation vectors have not been calculated.")

        do_osc = hasattr(self, "oscillator_strengths")

        if num_contr is None:
            num_contr = self.num_params * 2

        print(f"{'Value'.center(12)} | {'Position'.center(12)} | {'Operator'.center(12)}\n")

        for state, vec in enumerate(self.excitation_vectors.T):

            sorted_indices = np.argsort(np.abs(vec))[::-1]
            sorted_vec = np.abs(vec[sorted_indices]) ** 2

            if do_osc:
                print(
                    f"Excited state: {state:3}: {self.excitation_energies[state]:2.3f} Ha f = {self.oscillator_strengths[state]:2.3f}"
                )
            else:
                print(f"Excited state: {state:3}: {self.excitation_energies[state]:2.3f} Ha")
            for i in range(num_contr):
                if sorted_vec[i] < cutoff:
                    continue
                element = f"{sorted_vec[i]:.3f}"
                if sorted_indices[i] < self.num_params:
                    if sorted_indices[i] < self.num_q:
                        operator_index = "q" + str(sorted_indices[i])
                    else:
                        operator_index = "G" + str(sorted_indices[i] - self.num_q)
                else:
                    if sorted_indices[i] - self.num_params < self.num_q:
                        operator_index = "q" + str(sorted_indices[i] - self.num_params) + "^d"
                    else:
                        operator_index = "G" + str(sorted_indices[i] - self.num_params - self.num_q) + "^d"

                print(
                    f"{element.center(12)} | {str(sorted_indices[i]).center(12)} | {operator_index.center(12)}"
                )

    def _get_std_trend(
        self, re_calc: bool = False
    ) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
        """Analyze standard deviation trend across excited states."""
        # result vectors
        A_trend = np.zeros(self.num_params)
        B_trend = np.zeros(self.num_params)
        Sigma_trend = np.zeros(self.num_params)

        # get variance in operators
        if not hasattr(self, "_std_A_row") or re_calc:
            self.run_std(save=True, cv=False)

        # get excitation vector contributions combining (de-)excitation
        for state, vec in enumerate(np.abs(self.excitation_vectors.T) ** 2):
            exc_vec = vec[: self.num_params] + vec[self.num_params :]

            A_trend[state] = np.sum(exc_vec * self._std_A_row)
            B_trend[state] = np.sum(exc_vec * self._std_B_row)
            Sigma_trend[state] = np.sum(exc_vec * self._std_Sigma_row)

        return A_trend, B_trend, Sigma_trend

    def _get_CV_trend(
        self, re_calc: bool = False
    ) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
        """Analyze CV trend across excited states."""
        # result vectors
        A_trend = np.zeros(self.num_params)
        B_trend = np.zeros(self.num_params)
        Sigma_trend = np.zeros(self.num_params)

        # get variance in operators
        if not hasattr(self, "_CV_A_row") or re_calc:
            self.run_std(save=True, cv=True)

        # get excitation vector contributions combining (de-)excitation
        for state, vec in enumerate(np.abs(self.excitation_vectors.T) ** 2):
            exc_vec = vec[: self.num_params] + vec[self.num_params :]

            A_trend[state] = np.sum(exc_vec * self._CV_A_row)
            B_trend[state] = np.sum(exc_vec * self._CV_B_row)
            Sigma_trend[state] = np.sum(exc_vec * self._CV_Sigma_row)

        return A_trend, B_trend, Sigma_trend


def get_num_nonCBS(matrix: list[list[str]]) -> int:
    """Count number of non computational basis measurements in operator matrix.

    Args:
        matrix: Operator matrix.

    Returns:
        Number of non computational basis measurements.
    """
    count = 0
    dim = len(matrix[0])
    for i in range(dim):
        for j in range(i, dim):
            for paulis in matrix[i][j]:
                if any(letter in paulis for letter in ("X", "Y")):
                    count += 1
    return count


def get_num_CBS_elements(matrix: list[list[str]]) -> tuple[int, int]:
    """Count how many individual elements in matrix require only measurement in computational basis and how many do not.

    Args:
        matrix: Operator matrix.

    Returns:
        Number of elements only requiring computational basis measurements and how many do not.
    """
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
