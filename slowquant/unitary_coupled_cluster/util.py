from collections.abc import Generator, Sequence

import numpy as np
import scipy.linalg

from slowquant.unitary_coupled_cluster.operator_matrix import (
    G1_sa_matrix,
    G2_1_sa_matrix,
    G2_2_sa_matrix,
)


def construct_integral_trans_mat(
    c_orthonormal: np.ndarray, kappa: Sequence[float], kappa_idx: Sequence[Sequence[int]]
) -> np.ndarray:
    """Contruct orbital transformation matrix.

    Args:
        c_orthonormal: Initial orbital coefficients.
        kappa: Orbital rotation parameters.
        kappa_idx: Non-redundant orbital rotation parameters indices.

    Returns:
        Orbital transformation matrix.
    """
    kappa_mat = np.zeros_like(c_orthonormal)
    for kappa_val, (p, q) in zip(kappa, kappa_idx):
        kappa_mat[p, q] = kappa_val
        kappa_mat[q, p] = -kappa_val
    c_trans = np.matmul(c_orthonormal, scipy.linalg.expm(-kappa_mat))
    return c_trans


class ThetaPicker:
    def __init__(
        self,
        active_occ_spin_idx: Sequence[int],
        active_unocc_spin_idx: Sequence[int],
    ) -> None:
        """Initialize helper class to iterate over active space parameters.

        Args:
            active_occ_spin_idx: Spin index of strongly occupied orbitals.
            active_unocc_spin_idx: Spin index of weakly occupied orbitals.
        """
        self.active_occ_spin_idx: list[int] = []
        self.active_unocc_spin_idx: list[int] = []
        self.active_occ_idx: list[int] = []
        self.active_unocc_idx: list[int] = []
        for idx in active_occ_spin_idx:
            self.active_occ_spin_idx.append(idx)
            if idx // 2 not in self.active_occ_idx:
                self.active_occ_idx.append(idx // 2)
        for idx in active_unocc_spin_idx:
            self.active_unocc_spin_idx.append(idx)
            if idx // 2 not in self.active_unocc_idx:
                self.active_unocc_idx.append(idx // 2)

    def get_t1_generator_sa(
        self,
    ) -> Generator[tuple[int, int, int, float], None, None]:
        """Get generate over T1 spin-adapted operators.

        Returns:
            T1 operator generator.
        """
        return iterate_t1_sa(self.active_occ_idx, self.active_unocc_idx)

    def get_t2_generator_sa(
        self,
    ) -> Generator[tuple[int, int, int, int, int, float, int], None, None]:
        """Get generate over T2 spin-adapted operators.

        Returns:
            T2 operator generator.
        """
        return iterate_t2_sa(self.active_occ_idx, self.active_unocc_idx)


def iterate_t1_sa(
    active_occ_idx: Sequence[int],
    active_unocc_idx: Sequence[int],
) -> Generator[tuple[int, int, int, float], None, None]:
    """Iterate over T1 spin-adapted operators.

    Args:
        active_occ_idx: Indices of strongly occupied orbitals.
        active_unocc_idx: Indices of weakly occupied orbitals.
        num_spin_orbs: Number of spin orbitals.

    Returns:
        T1 operator iteration.
    """
    theta_idx = -1
    for i in active_occ_idx:
        for a in active_unocc_idx:
            theta_idx += 1
            fac = 2 ** (-1 / 2)
            yield theta_idx, a, i, fac


def iterate_t2_sa(
    active_occ_idx: Sequence[int],
    active_unocc_idx: Sequence[int],
) -> Generator[tuple[int, int, int, int, int, float, int], None, None]:
    """Iterate over T2 spin-adapted operators.

    Args:
        active_occ_idx: Indices of strongly occupied orbitals.
        active_unocc_idx: Indices of weakly occupied orbitals.
        num_spin_orbs: Number of spin orbitals.

    Returns:
        T2 operator iteration.
    """
    theta_idx = -1
    for idx_i, i in enumerate(active_occ_idx):
        for j in active_occ_idx[idx_i:]:
            for idx_a, a in enumerate(active_unocc_idx):
                for b in active_unocc_idx[idx_a:]:
                    theta_idx += 1
                    fac = 1
                    if a == b:
                        fac *= 2
                    if i == j:
                        fac *= 2
                    fac = 1 / 2 * (fac) ** (-1 / 2)
                    yield theta_idx, a, i, b, j, fac, 1
                    if i == j or a == b:
                        continue
                    theta_idx += 1
                    fac = 1 / (2 * 3 ** (1 / 2))
                    yield theta_idx, a, i, b, j, fac, 2


def construct_ucc_u(
    num_det: int,
    num_active_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
    theta: Sequence[float],
    theta_picker: ThetaPicker,
    excitations: str,
) -> np.ndarray:
    """Contruct unitary transformation matrix.

    Args:
       num_spin_orbs: Number of spin orbitals.
       theta: Active-space parameters.
              Ordered as (S, D, T, ...).
       theta_picker: Helper class to pick the parameters in the right order.
       excitations: Excitation orders to include.

    Returns:
        Unitary transformation matrix.
    """
    t = np.zeros((num_det, num_det))
    counter = 0
    if "s" in excitations:
        for _, a, i, _ in theta_picker.get_t1_generator_sa():
            if theta[counter] != 0.0:
                t += (
                    theta[counter]
                    * G1_sa_matrix(i, a, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
                )
            counter += 1
    if "d" in excitations:
        for _, a, i, b, j, _, type_idx in theta_picker.get_t2_generator_sa():
            if theta[counter] != 0.0:
                if type_idx == 1:
                    t += (
                        theta[counter]
                        * G2_1_sa_matrix(i, j, a, b, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
                    )
                elif type_idx == 2:
                    t += (
                        theta[counter]
                        * G2_2_sa_matrix(i, j, a, b, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
                    )
                else:
                    raise ValueError(f"Expected type_idx to be in (1,2) got {type_idx}")
            counter += 1
    assert counter == len(theta)
    T = t - t.conjugate().transpose()
    A = scipy.linalg.expm(T)
    return A


class UpsStructure:
    def __init__(self) -> None:
        self.excitation_indicies = []
        self.excitation_operator_type = []
        self.n_params = 0

    def create_tups(self, n_layers: int, num_active_orbs: int) -> None:
        for _ in range(n_layers):
            for p in range(0, num_active_orbs - 1, 2):
                # First single
                self.excitation_operator_type.append(1)
                self.excitation_indicies.append((p + 1, p))
                # Double
                self.excitation_operator_type.append(2)
                self.excitation_indicies.append((p + 1, p + 1, p, p))
                # Second single
                self.excitation_operator_type.append(1)
                self.excitation_indicies.append((p + 1, p))
                self.n_params += 3
            for p in range(1, num_active_orbs - 2, 2):
                # First single
                self.excitation_operator_type.append(1)
                self.excitation_indicies.append((p + 1, p))
                # Double
                self.excitation_operator_type.append(2)
                self.excitation_indicies.append((p + 1, p + 1, p, p))
                # Second single
                self.excitation_operator_type.append(1)
                self.excitation_indicies.append((p + 1, p))
                self.n_params += 3


def construct_ups_u(
    num_det: int,
    num_active_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
    thetas: Sequence[float],
    ups_struct: UpsStructure,
) -> np.ndarray:
    A = np.eye(num_det)
    for exc_type, exc_indices, theta in zip(
        ups_struct.excitation_operator_type, ups_struct.excitation_indicies, thetas
    ):
        if exc_type == 1:
            i, a = exc_indices
            t = theta * G1_sa_matrix(i, a, num_active_orbs, num_elec_alpha, num_elec_beta)
        elif exc_type == 2:
            i, j, a, b = exc_indices
            t = theta * G2_1_sa_matrix(i, j, a, b, num_active_orbs, num_elec_alpha, num_elec_beta)
        elif exc_type == 3:
            i, j, a, b = exc_indices
            t = theta * G2_2_sa_matrix(i, j, a, b, num_active_orbs, num_elec_alpha, num_elec_beta)
        else:
            raise ValueError(f"Got unknown excitation type, {exc_type}")
        B = scipy.sparse.linalg.expm(t - t.conjugate().transpose()).todense()
        A = np.matmul(B, A)
    return A
