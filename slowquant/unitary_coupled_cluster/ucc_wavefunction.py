import time
from collections.abc import Sequence
from functools import partial

import numpy as np
import scipy
import scipy.optimize

from slowquant.unitary_coupled_cluster.base import StateVector
from slowquant.unitary_coupled_cluster.operator_hybrid import (
    convert_pauli_to_hybrid_form,
    expectation_value_hybrid,
)
from slowquant.unitary_coupled_cluster.operator_pauli import (
    energy_hamiltonian_pauli,
    expectation_value_pauli,
)
from slowquant.unitary_coupled_cluster.util import (
    ThetaPicker,
    construct_integral_trans_mat,
    construct_ucc_u,
)


class WaveFunctionUCC:
    def __init__(
        self,
        num_spin_orbs: int,
        num_elec: int,
        cas: Sequence[int],
        c_orthonormal: np.ndarray,
        h_core: np.ndarray,
        g_eri: np.ndarray,
        is_generalized: bool = False,
        include_active_kappa: bool = False,
    ) -> None:
        """Initialize for UCC wave function.

        Args:
            num_spin_orbs: Number of spin orbitals.
            num_elec: Number of electrons.
            cas: CAS(num_active_elec, num_active_orbs),
                 orbitals are counted in spatial basis.
            c_orthonormal: Initial orbital coefficients.
            h_core: One-electron integrals in AO for Hamiltonian.
            g_eri: Two-electron integrals in AO.
            is_generalized: Do generalized UCC.
            include_active_kappa: Include active-active orbital rotations.
        """
        if len(cas) != 2:
            raise ValueError(f'cas must have two elements, got {len(cas)} elements.')
        o = np.array([0, 1])
        z = np.array([1, 0])
        self.c_orthonormal = c_orthonormal
        self.h_core = h_core
        self.g_eri = g_eri
        self.inactive_spin_idx = []
        self.virtual_spin_idx = []
        self.active_spin_idx = []
        self.active_occ_spin_idx = []
        self.active_unocc_spin_idx = []
        self.active_spin_idx_shifted = []
        self.active_occ_spin_idx_shifted = []
        self.active_unocc_spin_idx_shifted = []
        self.num_elec = num_elec
        self.num_spin_orbs = num_spin_orbs
        inactive_on_vector = []
        active_on_vector = []
        virtual_on_vector = []
        self.num_active_elec = 0
        self.num_active_spin_orbs = 0
        self.num_inactive_spin_orbs = 0
        self.num_virtual_spin_orbs = 0
        active_space = []
        orbital_counter = 0
        for i in range(num_elec - cas[0], num_elec):
            active_space.append(i)
            orbital_counter += 1
        for i in range(num_elec, num_elec + 2 * cas[1] - orbital_counter):
            active_space.append(i)
        for i in range(num_elec):
            if i in active_space:
                self.active_spin_idx.append(i)
                self.active_occ_spin_idx.append(i)
                active_on_vector.append(o)
                self.num_active_spin_orbs += 1
                self.num_active_elec += 1
            else:
                self.inactive_spin_idx.append(i)
                inactive_on_vector.append(o)
                self.num_inactive_spin_orbs += 1
        for i in range(num_elec, num_spin_orbs):
            if i in active_space:
                self.active_spin_idx.append(i)
                self.active_unocc_spin_idx.append(i)
                active_on_vector.append(z)
                self.num_active_spin_orbs += 1
            else:
                self.virtual_spin_idx.append(i)
                virtual_on_vector.append(z)
                self.num_virtual_spin_orbs += 1
        if len(self.active_spin_idx) != 0:
            active_shift = np.min(self.active_spin_idx)
            for active_idx in self.active_spin_idx:
                self.active_spin_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_occ_spin_idx:
                self.active_occ_spin_idx_shifted.append(active_idx - active_shift)
            for active_idx in self.active_unocc_spin_idx:
                self.active_unocc_spin_idx_shifted.append(active_idx - active_shift)
        self.state_vector = StateVector(inactive_on_vector, active_on_vector, virtual_on_vector)
        # Contruct spatial idx
        self.inactive_idx: list[int] = []
        self.virtual_idx: list[int] = []
        self.active_idx: list[int] = []
        self.active_occ_idx: list[int] = []
        self.active_unocc_idx: list[int] = []
        for idx in self.inactive_spin_idx:
            if idx // 2 not in self.inactive_idx:
                self.inactive_idx.append(idx // 2)
        for idx in self.active_spin_idx:
            if idx // 2 not in self.active_idx:
                self.active_idx.append(idx // 2)
        for idx in self.virtual_spin_idx:
            if idx // 2 not in self.virtual_idx:
                self.virtual_idx.append(idx // 2)
        for idx in self.active_occ_spin_idx:
            if idx // 2 not in self.active_occ_idx:
                self.active_occ_idx.append(idx // 2)
        for idx in self.active_unocc_spin_idx:
            if idx // 2 not in self.active_unocc_idx:
                self.active_unocc_idx.append(idx // 2)
        # Find non-redundant kappas
        self.kappa = []
        self.kappa_idx = []
        self.kappa_redundant = []
        self.kappa_redundant_idx = []
        # kappa can be optimized in spatial basis
        for p in range(0, self.num_spin_orbs // 2):
            for q in range(p + 1, self.num_spin_orbs // 2):
                if p in self.inactive_idx and q in self.inactive_idx:
                    self.kappa_redundant.append(0)
                    self.kappa_redundant_idx.append([p, q])
                    continue
                if p in self.virtual_idx and q in self.virtual_idx:
                    self.kappa_redundant.append(0)
                    self.kappa_redundant_idx.append([p, q])
                    continue
                if not include_active_kappa:
                    if p in self.active_idx and q in self.active_idx:
                        self.kappa_redundant.append(0)
                        self.kappa_redundant_idx.append([p, q])
                        continue
                if include_active_kappa:
                    if p in self.active_occ_idx and q in self.active_occ_idx:
                        self.kappa_redundant.append(0)
                        self.kappa_redundant_idx.append([p, q])
                        continue
                    if p in self.active_unocc_idx and q in self.active_unocc_idx:
                        self.kappa_redundant.append(0)
                        self.kappa_redundant_idx.append([p, q])
                        continue
                self.kappa.append(0)
                self.kappa_idx.append([p, q])
        # Construct theta1
        self.theta_picker = ThetaPicker(
            self.active_occ_spin_idx_shifted,
            self.active_unocc_spin_idx_shifted,
            is_spin_conserving=True,
            is_generalized=is_generalized,
        )
        self.theta_picker_full = ThetaPicker(
            self.active_occ_spin_idx_shifted,
            self.active_unocc_spin_idx_shifted,
            is_spin_conserving=False,
            is_generalized=is_generalized,
        )
        self.theta1 = []
        for _ in self.theta_picker_full.get_t1_generator_sa(0, 0):
            self.theta1.append(0)
        # Construct theta2
        self.theta2 = []
        for _ in self.theta_picker_full.get_t2_generator_sa(0, 0):
            self.theta2.append(0)
        # Construct theta3
        self.theta3 = []
        for _ in self.theta_picker_full.get_t3_generator(0, 0):
            self.theta3.append(0)
        # Construct theta4
        self.theta4 = []
        for _ in self.theta_picker_full.get_t4_generator(0, 0):
            self.theta4.append(0)

    @property
    def c_trans(self) -> np.ndarray:
        """Get orbital coefficients.

        Returns:
            Orbital coefficients.
        """
        kappa_mat = np.zeros_like(self.c_orthonormal)
        if len(self.kappa) != 0:
            if np.max(np.abs(self.kappa)) > 0.0:
                for kappa_val, (p, q) in zip(self.kappa, self.kappa_idx):
                    kappa_mat[p, q] = kappa_val
                    kappa_mat[q, p] = -kappa_val
        if len(self.kappa_redundant) != 0:
            if np.max(np.abs(self.kappa_redundant)) > 0.0:
                for kappa_val, (p, q) in zip(self.kappa_redundant, self.kappa_redundant_idx):
                    kappa_mat[p, q] = kappa_val
                    kappa_mat[q, p] = -kappa_val
        return np.matmul(self.c_orthonormal, scipy.linalg.expm(-kappa_mat))

    def run_ucc(
        self,
        excitations: str,
        orbital_optimization: bool = False,
        is_silent: bool = False,
        convergence_threshold: float = 10**-8,
    ) -> None:
        """Run optimization of UCC wave function.

        Args:
            excitations: Excitation orders to include.
            orbital_optimization: Do orbital optimization.
        """
        excitations = excitations.lower()
        self._excitations = excitations
        if orbital_optimization:
            e_tot = partial(
                energy_ucc,
                num_inactive_spin_orbs=self.num_inactive_spin_orbs,
                num_active_spin_orbs=self.num_active_spin_orbs,
                num_virtual_spin_orbs=self.num_virtual_spin_orbs,
                num_elec=self.num_elec,
                num_active_elec=self.num_active_elec,
                state_vector=self.state_vector,
                c_orthonormal=self.c_orthonormal,
                h_core=self.h_core,
                g_eri=self.g_eri,
                theta_picker=self.theta_picker,
                excitations=excitations,
                orbital_optimized=True,
                kappa_idx=self.kappa_idx,
                kappa_redundant=self.kappa_redundant,
                kappa_redundant_idx=self.kappa_redundant_idx,
            )
            parameter_gradient = partial(
                gradient_ucc,
                num_inactive_spin_orbs=self.num_inactive_spin_orbs,
                num_active_spin_orbs=self.num_active_spin_orbs,
                num_virtual_spin_orbs=self.num_virtual_spin_orbs,
                num_elec=self.num_elec,
                num_active_elec=self.num_active_elec,
                state_vector=self.state_vector,
                c_orthonormal=self.c_orthonormal,
                h_core=self.h_core,
                g_eri=self.g_eri,
                theta_picker=self.theta_picker,
                excitations=excitations,
                orbital_optimized=True,
                kappa_idx=self.kappa_idx,
                kappa_redundant=self.kappa_redundant,
                kappa_redundant_idx=self.kappa_redundant_idx,
            )
        else:
            e_tot = partial(
                energy_ucc,
                num_inactive_spin_orbs=self.num_inactive_spin_orbs,
                num_active_spin_orbs=self.num_active_spin_orbs,
                num_virtual_spin_orbs=self.num_virtual_spin_orbs,
                num_elec=self.num_elec,
                num_active_elec=self.num_active_elec,
                state_vector=self.state_vector,
                c_orthonormal=construct_integral_trans_mat(self.c_orthonormal, self.kappa, self.kappa_idx),
                h_core=self.h_core,
                g_eri=self.g_eri,
                theta_picker=self.theta_picker,
                excitations=excitations,
                orbital_optimized=False,
                kappa_idx=[],
                kappa_redundant=self.kappa_redundant,
                kappa_redundant_idx=self.kappa_redundant_idx,
            )
            parameter_gradient = partial(
                gradient_ucc,
                num_inactive_spin_orbs=self.num_inactive_spin_orbs,
                num_active_spin_orbs=self.num_active_spin_orbs,
                num_virtual_spin_orbs=self.num_virtual_spin_orbs,
                num_elec=self.num_elec,
                num_active_elec=self.num_active_elec,
                state_vector=self.state_vector,
                c_orthonormal=construct_integral_trans_mat(self.c_orthonormal, self.kappa, self.kappa_idx),
                h_core=self.h_core,
                g_eri=self.g_eri,
                theta_picker=self.theta_picker,
                excitations=excitations,
                orbital_optimized=False,
                kappa_idx=[],
                kappa_redundant=self.kappa_redundant,
                kappa_redundant_idx=self.kappa_redundant_idx,
            )
        global iteration
        global start
        iteration = 0  # type: ignore
        start = time.time()  # type: ignore

        def print_progress(X: Sequence[float]) -> None:
            """Print progress during energy minimization of wave function.

            Args:
                X: Wave function parameters.
            """
            global iteration
            global start
            time_str = f'{time.time() - start:7.2f}'  # type: ignore
            e_str = f'{e_tot(X):3.6f}'
            print(f'{str(iteration+1).center(11)} | {time_str.center(18)} | {e_str.center(27)}')  # type: ignore
            iteration += 1  # type: ignore
            if iteration > 100:
                raise ValueError('Did not converge in 100 iterations in energy minimization.')
            start = time.time()  # type: ignore

        def silent_progress(X: Sequence[float]) -> None:
            """Print progress during energy minimization of wave function.

            Args:
                X: Wave function parameters.
            """
            global iteration
            iteration += 1  # type: ignore
            if iteration > 100:
                raise ValueError('Did not converge in 100 iterations in energy minimization.')

        parameters = []
        num_kappa = 0
        num_theta1 = 0
        num_theta2 = 0
        num_theta3 = 0
        num_theta4 = 0
        if orbital_optimization:
            parameters += self.kappa
            num_kappa += len(self.kappa)
        if 's' in excitations:
            for idx, _, _, _ in self.theta_picker.get_t1_generator_sa(0, 0):
                parameters += [self.theta1[idx]]
                num_theta1 += 1
        if 'd' in excitations:
            for idx, _, _, _, _, _ in self.theta_picker.get_t2_generator_sa(0, 0):
                parameters += [self.theta2[idx]]
                num_theta2 += 1
        if 't' in excitations:
            for idx, _, _, _, _, _, _, _ in self.theta_picker.get_t3_generator(0, 0):
                parameters += [self.theta3[idx]]
                num_theta3 += 1
        if 'q' in excitations:
            for idx, _, _, _, _, _, _, _, _, _ in self.theta_picker.get_t4_generator(0, 0):
                parameters += [self.theta4[idx]]
                num_theta4 += 1
        if is_silent:
            res = scipy.optimize.minimize(
                e_tot,
                parameters,
                tol=convergence_threshold,
                callback=silent_progress,
                method='SLSQP',
                #jac=parameter_gradient,
            )
        else:
            print('### Parameters information:')
            print(f'### Number kappa: {num_kappa}')
            print(f'### Number theta1: {num_theta1}')
            print(f'### Number theta2: {num_theta2}')
            print(f'### Number theta3: {num_theta3}')
            print(f'### Number theta4: {num_theta4}')
            print(f'### Total parameters: {num_kappa+num_theta1+num_theta2+num_theta3+num_theta4}\n')
            print('Iteration # | Iteration time [s] | Electronic energy [Hartree]')
            res = scipy.optimize.minimize(
                e_tot,
                parameters,
                tol=convergence_threshold,
                callback=print_progress,
                method='SLSQP',
                #jac=parameter_gradient,
            )
        self.energy_elec = res['fun']
        param_idx = 0
        if orbital_optimization:
            self.kappa = res['x'][param_idx : len(self.kappa) + param_idx].tolist()
            param_idx += len(self.kappa)
        if 's' in excitations:
            thetas = res['x'][param_idx : num_theta1 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for idx, _, _, _ in self.theta_picker.get_t1_generator_sa(0, 0):
                self.theta1[idx] = thetas[counter]
                counter += 1
        if 'd' in excitations:
            thetas = res['x'][param_idx : num_theta2 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for idx, _, _, _, _, _ in self.theta_picker.get_t2_generator_sa(0, 0):
                self.theta2[idx] = thetas[counter]
                counter += 1
        if 't' in excitations:
            thetas = res['x'][param_idx : num_theta3 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for idx, _, _, _, _, _, _, _ in self.theta_picker.get_t3_generator(0, 0):
                self.theta3[idx] = thetas[counter]
                counter += 1
        if 'q' in excitations:
            thetas = res['x'][param_idx : num_theta4 + param_idx].tolist()
            param_idx += len(thetas)
            counter = 0
            for idx, _, _, _, _, _, _, _, _, _ in self.theta_picker.get_t4_generator(0, 0):
                self.theta4[idx] = thetas[counter]
                counter += 1


def run_compactify_wf(wf: WaveFunctionUCC, excitations: str) -> None:
    global iteration2
    global start2
    iteration2 = 0  # type: ignore
    start2 = time.time()  # type: ignore

    def print_progress2(X: Sequence[float]) -> None:
        """Print progress during energy minimization of wave function.

        Args:
            X: Wave function parameters.
        """
        global iteration2
        global start2
        time_str = f'{time.time() - start2:7.2f}'  # type: ignore
        e_str = f'{entropy_tot(X):1.8f}'
        print(f'{str(iteration2+1).center(11)} | {time_str.center(18)} | {e_str.center(13)}')  # type: ignore
        iteration2 += 1  # type: ignore
        start2 = time.time()  # type: ignore

    print(wf.theta1)
    print(wf.theta2)
    print(f'Initial cost function value: {entropy_ucc(wf.kappa_redundant, wf=wf, excitations=excitations)}')

    print('Iteration # | Iteration time [s] | Cost function')
    entropy_tot = partial(entropy_ucc, wf=wf, excitations=excitations)
    res = scipy.optimize.minimize(
        entropy_tot,
        np.zeros(len(wf.kappa_redundant)),
        tol=1e-8,
        callback=print_progress2,
        method='BFGS',
        options={'eps': 10**-6},
    )
    print(wf.theta1)
    print(wf.theta2)
    print(res)


def entropy_ucc(parameters: Sequence[float], wf: WaveFunctionUCC, excitations: str) -> float:
    for i, kappa in enumerate(parameters):
        if abs(kappa) > 2 * np.pi:
            raise ValueError(f'kappa {i} is uncontrolled, value of {kappa}')
        wf.kappa_redundant[i] = kappa
    wf.run_ucc(excitations, False, is_silent=True, convergence_threshold=10**-12)
    entropy = 0
    for state in wf.theta1 + wf.theta2:
        entropy += np.log(1 / (1e-16 + state**2))
    return -entropy


def energy_ucc(
    parameters: Sequence[float],
    num_inactive_spin_orbs: int,
    num_active_spin_orbs: int,
    num_virtual_spin_orbs: int,
    num_elec: int,
    num_active_elec: int,
    state_vector: StateVector,
    c_orthonormal: np.ndarray,
    h_core: np.ndarray,
    g_eri: np.ndarray,
    theta_picker: ThetaPicker,
    excitations: str,
    orbital_optimized: bool,
    kappa_idx: Sequence[Sequence[int]],
    kappa_redundant: Sequence[float],
    kappa_redundant_idx: Sequence[Sequence[int]],
) -> float:
    r"""Calculate electronic energy of UCC wave function.

    .. math::
        E = \left<0\left|\hat{H}\right|0\right>

    Args:
        parameters: Sequence of all parameters.
                    Ordered as orbital rotations, active-space singles, active-space doubles, ...
        num_inactive_spin_orbs: Number of inactive spin orbitals.
        num_active_spin_orbs: Number of active spin orbitals.
        num_virtual_spin_orbs: Number of virtual spin orbitals.
        num_elec: Number of electrons.
        num_active_elec: Number of electrons in active-space.
        state_vector: State vector object.
        c_othonormal: Orbital coefficients.
        h_core: Core Hamiltonian integrals in AO.
        g_eri: Two-electron integrals in AO.
        theta_picker: Cluster operator generator object.
        excitations: Excitation orders to consider.
        orbital_optimized: Do orbital optimization.
        kappa_idx: Indicies of non-redundant orbital rotations.

    Returns:
        Electronic energy.
    """
    kappa = []
    theta1 = []
    theta2 = []
    theta3 = []
    theta4 = []
    idx_counter = 0
    for _ in range(len(kappa_idx)):
        kappa.append(parameters[idx_counter])
        idx_counter += 1
    if 's' in excitations:
        for _ in theta_picker.get_t1_generator_sa(0, 0):
            theta1.append(parameters[idx_counter])
            idx_counter += 1
    if 'd' in excitations:
        for _ in theta_picker.get_t2_generator_sa(
            num_inactive_spin_orbs + num_active_spin_orbs + num_virtual_spin_orbs, num_elec
        ):
            theta2.append(parameters[idx_counter])
            idx_counter += 1
    if 't' in excitations:
        for _ in theta_picker.get_t3_generator(0, 0):
            theta3.append(parameters[idx_counter])
            idx_counter += 1
    if 'q' in excitations:
        for _ in theta_picker.get_t4_generator(0, 0):
            theta4.append(parameters[idx_counter])
            idx_counter += 1
    assert len(parameters) == len(kappa) + len(theta1) + len(theta2) + len(theta3) + len(theta4)

    kappa_mat = np.zeros_like(c_orthonormal)
    if orbital_optimized:
        for kappa_val, (p, q) in zip(kappa, kappa_idx):
            kappa_mat[p, q] = kappa_val
            kappa_mat[q, p] = -kappa_val
    if len(kappa_redundant) != 0:
        if np.max(np.abs(kappa_redundant)) > 0.0:
            for kappa_val, (p, q) in zip(kappa_redundant, kappa_redundant_idx):
                kappa_mat[p, q] = kappa_val
                kappa_mat[q, p] = -kappa_val
    c_trans = np.matmul(c_orthonormal, scipy.linalg.expm(-kappa_mat))

    U = construct_ucc_u(
        num_active_spin_orbs,
        num_active_elec,
        theta1 + theta2 + theta3 + theta4,
        theta_picker,
        excitations,
        allowed_states=state_vector.allowed_active_states_number_spin_conserving,
    )
    state_vector.new_u(U, allowed_states=state_vector.allowed_active_states_number_spin_conserving)
    return expectation_value_pauli(
        state_vector,
        energy_hamiltonian_pauli(
            h_core,
            g_eri,
            c_trans,
            num_inactive_spin_orbs,
            num_active_spin_orbs,
            num_virtual_spin_orbs,
            num_elec,
        ),
        state_vector,
    )


def gradient_ucc(
    parameters: Sequence[float],
    num_inactive_spin_orbs: int,
    num_active_spin_orbs: int,
    num_virtual_spin_orbs: int,
    num_elec: int,
    num_active_elec: int,
    state_vector: StateVector,
    c_orthonormal: np.ndarray,
    h_core: np.ndarray,
    g_eri: np.ndarray,
    theta_picker: ThetaPicker,
    excitations: str,
    orbital_optimized: bool,
    kappa_idx: Sequence[Sequence[int]],
    kappa_redundant: Sequence[float],
    kappa_redundant_idx: Sequence[Sequence[int]],
) -> np.ndarray:
    """ """
    number_kappas = 0
    if orbital_optimized:
        number_kappas = len(kappa_idx)
    gradient = np.zeros_like(parameters)
    if orbital_optimized:
        gradient[:number_kappas] = orbital_rotation_gradient(
            parameters,
            num_inactive_spin_orbs,
            num_active_spin_orbs,
            num_virtual_spin_orbs,
            num_elec,
            num_active_elec,
            state_vector,
            c_orthonormal,
            h_core,
            g_eri,
            theta_picker,
            excitations,
            orbital_optimized,
            kappa_idx,
            kappa_redundant,
            kappa_redundant_idx,
        )
    gradient[number_kappas:] = active_space_parameter_gradient(
        parameters,
        num_inactive_spin_orbs,
        num_active_spin_orbs,
        num_virtual_spin_orbs,
        num_elec,
        num_active_elec,
        state_vector,
        c_orthonormal,
        h_core,
        g_eri,
        theta_picker,
        excitations,
        kappa_idx,
        kappa_redundant,
        kappa_redundant_idx,
    )
    return gradient


def orbital_rotation_gradient(
    parameters: Sequence[float],
    num_inactive_spin_orbs: int,
    num_active_spin_orbs: int,
    num_virtual_spin_orbs: int,
    num_elec: int,
    num_active_elec: int,
    state_vector: StateVector,
    c_orthonormal: np.ndarray,
    h_core: np.ndarray,
    g_eri: np.ndarray,
    theta_picker: ThetaPicker,
    excitations: str,
    orbital_optimized: bool,
    kappa_idx: Sequence[Sequence[int]],
    kappa_redundant: Sequence[float],
    kappa_redundant_idx: Sequence[Sequence[int]],
) -> np.ndarray:
    """ """
    kappa = []
    theta1 = []
    theta2 = []
    theta3 = []
    theta4 = []
    idx_counter = 0
    for _ in range(len(kappa_idx)):
        kappa.append(parameters[idx_counter])
        idx_counter += 1
    if 's' in excitations:
        for _ in theta_picker.get_t1_generator_sa(0, 0):
            theta1.append(parameters[idx_counter])
            idx_counter += 1
    if 'd' in excitations:
        for _ in theta_picker.get_t2_generator_sa(
            num_inactive_spin_orbs + num_active_spin_orbs + num_virtual_spin_orbs, num_elec
        ):
            theta2.append(parameters[idx_counter])
            idx_counter += 1
    if 't' in excitations:
        for _ in theta_picker.get_t3_generator(0, 0):
            theta3.append(parameters[idx_counter])
            idx_counter += 1
    if 'q' in excitations:
        for _ in theta_picker.get_t4_generator(0, 0):
            theta4.append(parameters[idx_counter])
            idx_counter += 1
    assert len(parameters) == len(kappa) + len(theta1) + len(theta2) + len(theta3) + len(theta4)

    U = construct_ucc_u(
        num_active_spin_orbs,
        num_active_elec,
        theta1 + theta2 + theta3 + theta4,
        theta_picker,
        excitations,
        allowed_states=state_vector.allowed_active_states_number_spin_conserving,
    )
    state_vector.new_u(U, allowed_states=state_vector.allowed_active_states_number_spin_conserving)

    step_size = 10**-8
    gradient_kappa = np.zeros_like(kappa)
    for i, _ in enumerate(kappa):
        kappa[i] += step_size
        kappa_mat = np.zeros_like(c_orthonormal)
        if orbital_optimized:
            for kappa_val, (p, q) in zip(kappa, kappa_idx):
                kappa_mat[p, q] = kappa_val
                kappa_mat[q, p] = -kappa_val
        if len(kappa_redundant) != 0:
            if np.max(np.abs(kappa_redundant)) > 0.0:
                for kappa_val, (p, q) in zip(kappa_redundant, kappa_redundant_idx):
                    kappa_mat[p, q] = kappa_val
                    kappa_mat[q, p] = -kappa_val
        c_trans = np.matmul(c_orthonormal, scipy.linalg.expm(-kappa_mat))
        E_plus = expectation_value_pauli(
            state_vector,
            energy_hamiltonian_pauli(
                h_core,
                g_eri,
                c_trans,
                num_inactive_spin_orbs,
                num_active_spin_orbs,
                num_virtual_spin_orbs,
                num_elec,
            ),
            state_vector,
        )
        kappa[i] -= step_size
        kappa[i] -= step_size
        kappa_mat = np.zeros_like(c_orthonormal)
        if len(kappa) != 0:
            if np.max(np.abs(kappa)) > 0.0:
                for kappa_val, (p, q) in zip(kappa, kappa_idx):
                    kappa_mat[p, q] = kappa_val
                    kappa_mat[q, p] = -kappa_val
        if len(kappa_redundant) != 0:
            if np.max(np.abs(kappa_redundant)) > 0.0:
                for kappa_val, (p, q) in zip(kappa_redundant, kappa_redundant_idx):
                    kappa_mat[p, q] = kappa_val
                    kappa_mat[q, p] = -kappa_val
        c_trans = np.matmul(c_orthonormal, scipy.linalg.expm(-kappa_mat))
        E_minus = expectation_value_pauli(
            state_vector,
            energy_hamiltonian_pauli(
                h_core,
                g_eri,
                c_trans,
                num_inactive_spin_orbs,
                num_active_spin_orbs,
                num_virtual_spin_orbs,
                num_elec,
            ),
            state_vector,
        )
        kappa[i] += step_size
        gradient_kappa[i] = (E_plus - E_minus) / (2 * step_size)
    return gradient_kappa


def active_space_parameter_gradient(
    parameters: Sequence[float],
    num_inactive_spin_orbs: int,
    num_active_spin_orbs: int,
    num_virtual_spin_orbs: int,
    num_elec: int,
    num_active_elec: int,
    state_vector: StateVector,
    c_orthonormal: np.ndarray,
    h_core: np.ndarray,
    g_eri: np.ndarray,
    theta_picker: ThetaPicker,
    excitations: str,
    kappa_idx: Sequence[Sequence[int]],
    kappa_redundant: Sequence[float],
    kappa_redundant_idx: Sequence[Sequence[int]],
) -> np.ndarray:
    """ """
    kappa = []
    theta1 = []
    theta2 = []
    theta3 = []
    theta4 = []
    idx_counter = 0
    for _ in range(len(kappa_idx)):
        kappa.append(parameters[idx_counter])
        idx_counter += 1
    if 's' in excitations:
        for _ in theta_picker.get_t1_generator_sa(0, 0):
            theta1.append(parameters[idx_counter])
            idx_counter += 1
    if 'd' in excitations:
        for _ in theta_picker.get_t2_generator_sa(
            num_inactive_spin_orbs + num_active_spin_orbs + num_virtual_spin_orbs, num_elec
        ):
            theta2.append(parameters[idx_counter])
            idx_counter += 1
    if 't' in excitations:
        for _ in theta_picker.get_t3_generator(0, 0):
            theta3.append(parameters[idx_counter])
            idx_counter += 1
    if 'q' in excitations:
        for _ in theta_picker.get_t4_generator(0, 0):
            theta4.append(parameters[idx_counter])
            idx_counter += 1
    assert len(parameters) == len(kappa) + len(theta1) + len(theta2) + len(theta3) + len(theta4)

    kappa_mat = np.zeros_like(c_orthonormal)
    if len(kappa) != 0:
        if np.max(np.abs(kappa)) > 0.0:
            for kappa_val, (p, q) in zip(kappa, kappa_idx):
                kappa_mat[p, q] = kappa_val
                kappa_mat[q, p] = -kappa_val
    if len(kappa_redundant) != 0:
        if np.max(np.abs(kappa_redundant)) > 0.0:
            for kappa_val, (p, q) in zip(kappa_redundant, kappa_redundant_idx):
                kappa_mat[p, q] = kappa_val
                kappa_mat[q, p] = -kappa_val

    c_trans = np.matmul(c_orthonormal, scipy.linalg.expm(-kappa_mat))
    Hamiltonian = convert_pauli_to_hybrid_form(
        energy_hamiltonian_pauli(
            h_core,
            g_eri,
            c_trans,
            num_inactive_spin_orbs,
            num_active_spin_orbs,
            num_virtual_spin_orbs,
            num_elec,
        ),
        num_inactive_spin_orbs,
        num_active_spin_orbs,
        num_virtual_spin_orbs,
    )

    theta_params = theta1 + theta2 + theta3 + theta4
    gradient_theta = np.zeros_like(theta_params)
    step_size = 10**-8
    for i, _ in enumerate(theta_params):
        theta_params[i] += step_size
        U = construct_ucc_u(
            num_active_spin_orbs,
            num_active_elec,
            theta_params,
            theta_picker,
            excitations,
            allowed_states=state_vector.allowed_active_states_number_spin_conserving,
        )
        state_vector.new_u(U, allowed_states=state_vector.allowed_active_states_number_spin_conserving)
        E_plus = expectation_value_hybrid(state_vector, Hamiltonian, state_vector)
        theta_params[i] -= step_size
        theta_params[i] -= step_size
        U = construct_ucc_u(
            num_active_spin_orbs,
            num_active_elec,
            theta1 + theta2 + theta3 + theta4,
            theta_picker,
            excitations,
            allowed_states=state_vector.allowed_active_states_number_spin_conserving,
        )
        state_vector.new_u(U, allowed_states=state_vector.allowed_active_states_number_spin_conserving)
        theta_params[i] += step_size
        E_minus = expectation_value_hybrid(state_vector, Hamiltonian, state_vector)
        gradient_theta[i] = (E_plus - E_minus) / (2 * step_size)
    return gradient_theta
