from collections.abc import Generator, Sequence

import numpy as np
import scipy.linalg

from slowquant.unitary_coupled_cluster.operator_matrix import (
    T1_sa_matrix,
    T2_1_sa_matrix,
    T2_2_sa_matrix,
    T3_matrix,
    T4_matrix,
    T5_matrix,
    T6_matrix,
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
    ) -> Generator[tuple[int, int, float], None, None]:
        """Get generate over T1 spin-adapted operators.

        Returns:
            T1 operator generator.
        """
        return iterate_t1_sa(self.active_occ_idx, self.active_unocc_idx)

    def get_t2_generator_sa(
        self,
    ) -> Generator[tuple[int, int, int, int, float, int], None, None]:
        """Get generate over T2 spin-adapted operators.

        Returns:
            T2 operator generator.
        """
        return iterate_t2_sa(self.active_occ_idx, self.active_unocc_idx)

    def get_t1_generator(self) -> Generator[tuple[int, int], None, None]:
        """Get generate over T1 spin-conserving operators.

        Returns:
            T1 operator generator.
        """
        return iterate_t1(self.active_occ_spin_idx, self.active_unocc_spin_idx)

    def get_t2_generator(self) -> Generator[tuple[int, int, int, int], None, None]:
        """Get generate over T2 spin-conserving operators.

        Returns:
            T2 operator generator.
        """
        return iterate_t2(self.active_occ_spin_idx, self.active_unocc_spin_idx)

    def get_t3_generator(self) -> Generator[tuple[int, int, int, int, int, int], None, None]:
        """Get generate over T3 spin-conserving operators.

        Returns:
            T3 operator generator.
        """
        return iterate_t3(self.active_occ_spin_idx, self.active_unocc_spin_idx)

    def get_t4_generator(self) -> Generator[tuple[int, int, int, int, int, int, int, int], None, None]:
        """Get generate over T4 spin-conserving operators.

        Returns:
            T4 operator generator.
        """
        return iterate_t4(self.active_occ_spin_idx, self.active_unocc_spin_idx)

    def get_t5_generator(
        self,
    ) -> Generator[tuple[int, int, int, int, int, int, int, int, int, int], None, None]:
        """Get generate over T5 spin-conserving operators.

        Returns:
            T5 operator generator.
        """
        return iterate_t5(self.active_occ_spin_idx, self.active_unocc_spin_idx)

    def get_t6_generator(
        self,
    ) -> Generator[tuple[int, int, int, int, int, int, int, int, int, int, int, int], None, None]:
        """Get generate over T6 spin-conserving operators.

        Returns:
            T6 operator generator.
        """
        return iterate_t6(self.active_occ_spin_idx, self.active_unocc_spin_idx)


def iterate_t1_sa(
    active_occ_idx: Sequence[int],
    active_unocc_idx: Sequence[int],
) -> Generator[tuple[int, int, float], None, None]:
    """Iterate over T1 spin-adapted operators.

    Args:
        active_occ_idx: Indices of strongly occupied orbitals.
        active_unocc_idx: Indices of weakly occupied orbitals.

    Returns:
        T1 operator iteration.
    """
    for i in active_occ_idx:
        for a in active_unocc_idx:
            fac = 2 ** (-1 / 2)
            yield a, i, fac


def iterate_t2_sa(
    active_occ_idx: Sequence[int],
    active_unocc_idx: Sequence[int],
) -> Generator[tuple[int, int, int, int, float, int], None, None]:
    """Iterate over T2 spin-adapted operators.

    Args:
        active_occ_idx: Indices of strongly occupied orbitals.
        active_unocc_idx: Indices of weakly occupied orbitals.

    Returns:
        T2 operator iteration.
    """
    for idx_i, i in enumerate(active_occ_idx):
        for j in active_occ_idx[idx_i:]:
            for idx_a, a in enumerate(active_unocc_idx):
                for b in active_unocc_idx[idx_a:]:
                    fac = 1.0
                    if a == b:
                        fac *= 2.0
                    if i == j:
                        fac *= 2.0
                    fac = 1 / 2 * (fac) ** (-1 / 2)
                    yield a, i, b, j, fac, 1
                    if i == j or a == b:
                        continue
                    fac = 1 / (2 * 3 ** (1 / 2))
                    yield a, i, b, j, fac, 2


def iterate_t1(
    active_occ_spin_idx: list[int],
    active_unocc_spin_idx: list[int],
) -> Generator[tuple[int, int], None, None]:
    """Iterate over T1 spin-conserving operators.

    Args:
        active_occ_idx: Spin indices of strongly occupied orbitals.
        active_unocc_idx: Spin indices of weakly occupied orbitals.

    Returns:
        T1 operator iteration.
    """
    for a in active_unocc_spin_idx:
        for i in active_occ_spin_idx:
            num_alpha = 0
            num_beta = 0
            if a % 2 == 0:
                num_alpha += 1
            else:
                num_beta += 1
            if i % 2 == 0:
                num_alpha += 1
            else:
                num_beta += 1
            if num_alpha % 2 != 0 or num_beta % 2 != 0:
                continue
            yield a, i


def iterate_t2(
    active_occ_spin_idx: list[int],
    active_unocc_spin_idx: list[int],
) -> Generator[tuple[int, int, int, int], None, None]:
    """Iterate over T2 spin-conserving operators.

    Args:
        active_occ_idx: Spin indices of strongly occupied orbitals.
        active_unocc_idx: Spin indices of weakly occupied orbitals.

    Returns:
        T2 operator iteration.
    """
    for idx_a, a in enumerate(active_unocc_spin_idx):
        for b in active_unocc_spin_idx[idx_a + 1 :]:
            for idx_i, i in enumerate(active_occ_spin_idx):
                for j in active_occ_spin_idx[idx_i + 1 :]:
                    num_alpha = 0
                    num_beta = 0
                    if a % 2 == 0:
                        num_alpha += 1
                    else:
                        num_beta += 1
                    if b % 2 == 0:
                        num_alpha += 1
                    else:
                        num_beta += 1
                    if i % 2 == 0:
                        num_alpha += 1
                    else:
                        num_beta += 1
                    if j % 2 == 0:
                        num_alpha += 1
                    else:
                        num_beta += 1
                    if num_alpha % 2 != 0 or num_beta % 2 != 0:
                        continue
                    yield a, i, b, j


def iterate_t3(
    active_occ_spin_idx: list[int],
    active_unocc_spin_idx: list[int],
) -> Generator[tuple[int, int, int, int, int, int], None, None]:
    """Iterate over T3 spin-conserving operators.

    Args:
        active_occ_idx: Spin indices of strongly occupied orbitals.
        active_unocc_idx: Spin indices of weakly occupied orbitals.

    Returns:
        T3 operator iteration.
    """
    for idx_a, a in enumerate(active_unocc_spin_idx):
        for idx_b, b in enumerate(active_unocc_spin_idx[idx_a + 1 :], idx_a + 1):
            for c in active_unocc_spin_idx[idx_b + 1 :]:
                for idx_i, i in enumerate(active_occ_spin_idx):
                    for idx_j, j in enumerate(active_occ_spin_idx[idx_i + 1 :], idx_i + 1):
                        for k in active_occ_spin_idx[idx_j + 1 :]:
                            num_alpha = 0
                            num_beta = 0
                            if a % 2 == 0:
                                num_alpha += 1
                            else:
                                num_beta += 1
                            if b % 2 == 0:
                                num_alpha += 1
                            else:
                                num_beta += 1
                            if c % 2 == 0:
                                num_alpha += 1
                            else:
                                num_beta += 1
                            if i % 2 == 0:
                                num_alpha += 1
                            else:
                                num_beta += 1
                            if j % 2 == 0:
                                num_alpha += 1
                            else:
                                num_beta += 1
                            if k % 2 == 0:
                                num_alpha += 1
                            else:
                                num_beta += 1
                            if num_alpha % 2 != 0 or num_beta % 2 != 0:
                                continue
                            yield a, i, b, j, c, k


def iterate_t4(
    active_occ_spin_idx: list[int],
    active_unocc_spin_idx: list[int],
) -> Generator[tuple[int, int, int, int, int, int, int, int], None, None]:
    """Iterate over T4 spin-conserving operators.

    Args:
        active_occ_idx: Spin indices of strongly occupied orbitals.
        active_unocc_idx: Spin indices of weakly occupied orbitals.

    Returns:
        T4 operator iteration.
    """
    for idx_a, a in enumerate(active_unocc_spin_idx):
        for idx_b, b in enumerate(active_unocc_spin_idx[idx_a + 1 :], idx_a + 1):
            for idx_c, c in enumerate(active_unocc_spin_idx[idx_b + 1 :], idx_b + 1):
                for d in active_unocc_spin_idx[idx_c + 1 :]:
                    for idx_i, i in enumerate(active_occ_spin_idx):
                        for idx_j, j in enumerate(active_occ_spin_idx[idx_i + 1 :], idx_i + 1):
                            for idx_k, k in enumerate(active_occ_spin_idx[idx_j + 1 :], idx_j + 1):
                                for l in active_occ_spin_idx[idx_k + 1 :]:
                                    num_alpha = 0
                                    num_beta = 0
                                    if a % 2 == 0:
                                        num_alpha += 1
                                    else:
                                        num_beta += 1
                                    if b % 2 == 0:
                                        num_alpha += 1
                                    else:
                                        num_beta += 1
                                    if c % 2 == 0:
                                        num_alpha += 1
                                    else:
                                        num_beta += 1
                                    if d % 2 == 0:
                                        num_alpha += 1
                                    else:
                                        num_beta += 1
                                    if i % 2 == 0:
                                        num_alpha += 1
                                    else:
                                        num_beta += 1
                                    if j % 2 == 0:
                                        num_alpha += 1
                                    else:
                                        num_beta += 1
                                    if k % 2 == 0:
                                        num_alpha += 1
                                    else:
                                        num_beta += 1
                                    if l % 2 == 0:
                                        num_alpha += 1
                                    else:
                                        num_beta += 1
                                    if num_alpha % 2 != 0 or num_beta % 2 != 0:
                                        continue
                                    yield a, i, b, j, c, k, d, l


def iterate_t5(
    active_occ_spin_idx: list[int],
    active_unocc_spin_idx: list[int],
) -> Generator[tuple[int, int, int, int, int, int, int, int, int, int], None, None]:
    """Iterate over T5 spin-conserving operators.

    Args:
        active_occ_idx: Spin indices of strongly occupied orbitals.
        active_unocc_idx: Spin indices of weakly occupied orbitals.

    Returns:
        T5 operator iteration.
    """
    for idx_a, a in enumerate(active_unocc_spin_idx):
        for idx_b, b in enumerate(active_unocc_spin_idx[idx_a + 1 :], idx_a + 1):
            for idx_c, c in enumerate(active_unocc_spin_idx[idx_b + 1 :], idx_b + 1):
                for idx_d, d in enumerate(active_unocc_spin_idx[idx_c + 1 :], idx_c + 1):
                    for e in active_unocc_spin_idx[idx_d + 1 :]:
                        for idx_i, i in enumerate(active_occ_spin_idx):
                            for idx_j, j in enumerate(active_occ_spin_idx[idx_i + 1 :], idx_i + 1):
                                for idx_k, k in enumerate(active_occ_spin_idx[idx_j + 1 :], idx_j + 1):
                                    for idx_l, l in enumerate(active_occ_spin_idx[idx_k + 1 :], idx_k + 1):
                                        for m in active_occ_spin_idx[idx_l + 1 :]:
                                            num_alpha = 0
                                            num_beta = 0
                                            if a % 2 == 0:
                                                num_alpha += 1
                                            else:
                                                num_beta += 1
                                            if b % 2 == 0:
                                                num_alpha += 1
                                            else:
                                                num_beta += 1
                                            if c % 2 == 0:
                                                num_alpha += 1
                                            else:
                                                num_beta += 1
                                            if d % 2 == 0:
                                                num_alpha += 1
                                            else:
                                                num_beta += 1
                                            if e % 2 == 0:
                                                num_alpha += 1
                                            else:
                                                num_beta += 1
                                            if i % 2 == 0:
                                                num_alpha += 1
                                            else:
                                                num_beta += 1
                                            if j % 2 == 0:
                                                num_alpha += 1
                                            else:
                                                num_beta += 1
                                            if k % 2 == 0:
                                                num_alpha += 1
                                            else:
                                                num_beta += 1
                                            if l % 2 == 0:
                                                num_alpha += 1
                                            else:
                                                num_beta += 1
                                            if m % 2 == 0:
                                                num_alpha += 1
                                            else:
                                                num_beta += 1
                                            if num_alpha % 2 != 0 or num_beta % 2 != 0:
                                                continue
                                            yield a, i, b, j, c, k, d, l, e, m


def iterate_t6(
    active_occ_spin_idx: list[int],
    active_unocc_spin_idx: list[int],
) -> Generator[tuple[int, int, int, int, int, int, int, int, int, int, int, int], None, None]:
    """Iterate over T6 spin-conserving operators.

    Args:
        active_occ_idx: Spin indices of strongly occupied orbitals.
        active_unocc_idx: Spin indices of weakly occupied orbitals.

    Returns:
        T6 operator iteration.
    """
    for idx_a, a in enumerate(active_unocc_spin_idx):
        for idx_b, b in enumerate(active_unocc_spin_idx[idx_a + 1 :], idx_a + 1):
            for idx_c, c in enumerate(active_unocc_spin_idx[idx_b + 1 :], idx_b + 1):
                for idx_d, d in enumerate(active_unocc_spin_idx[idx_c + 1 :], idx_c + 1):
                    for idx_e, e in enumerate(active_unocc_spin_idx[idx_d + 1 :], idx_d + 1):
                        for f in active_unocc_spin_idx[idx_e + 1 :]:
                            for idx_i, i in enumerate(active_occ_spin_idx):
                                for idx_j, j in enumerate(active_occ_spin_idx[idx_i + 1 :], idx_i + 1):
                                    for idx_k, k in enumerate(active_occ_spin_idx[idx_j + 1 :], idx_j + 1):
                                        for idx_l, l in enumerate(
                                            active_occ_spin_idx[idx_k + 1 :], idx_k + 1
                                        ):
                                            for idx_m, m in enumerate(
                                                active_occ_spin_idx[idx_l + 1 :], idx_l + 1
                                            ):
                                                for n in active_occ_spin_idx[idx_m + 1 :]:
                                                    num_alpha = 0
                                                    num_beta = 0
                                                    if a % 2 == 0:
                                                        num_alpha += 1
                                                    else:
                                                        num_beta += 1
                                                    if b % 2 == 0:
                                                        num_alpha += 1
                                                    else:
                                                        num_beta += 1
                                                    if c % 2 == 0:
                                                        num_alpha += 1
                                                    else:
                                                        num_beta += 1
                                                    if d % 2 == 0:
                                                        num_alpha += 1
                                                    else:
                                                        num_beta += 1
                                                    if e % 2 == 0:
                                                        num_alpha += 1
                                                    else:
                                                        num_beta += 1
                                                    if f % 2 == 0:
                                                        num_alpha += 1
                                                    else:
                                                        num_beta += 1
                                                    if i % 2 == 0:
                                                        num_alpha += 1
                                                    else:
                                                        num_beta += 1
                                                    if j % 2 == 0:
                                                        num_alpha += 1
                                                    else:
                                                        num_beta += 1
                                                    if k % 2 == 0:
                                                        num_alpha += 1
                                                    else:
                                                        num_beta += 1
                                                    if l % 2 == 0:
                                                        num_alpha += 1
                                                    else:
                                                        num_beta += 1
                                                    if m % 2 == 0:
                                                        num_alpha += 1
                                                    else:
                                                        num_beta += 1
                                                    if n % 2 == 0:
                                                        num_alpha += 1
                                                    else:
                                                        num_beta += 1
                                                    if num_alpha % 2 != 0 or num_beta % 2 != 0:
                                                        continue
                                                    yield a, i, b, j, c, k, d, l, e, m, f, n


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
        num_det: Number of determinants.
        num_active_orbs: Number of active spatial orbitals.
        num_elec_alpha: Number of active alpha electrons.
        num_elec_beta: Number of active beta electrons.
        theta: Active-space parameters.
               Ordered as (S, D, T, ...).
        theta_picker: Helper class to pick the parameters in the right order.
        excitations: Excitation orders to include.

    Returns:
        Unitary transformation matrix.
    """
    T = np.zeros((num_det, num_det))
    counter = 0
    if "s" in excitations:
        for a, i, _ in theta_picker.get_t1_generator_sa():
            if theta[counter] != 0.0:
                T += (
                    theta[counter]
                    * T1_sa_matrix(i, a, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
                )
            counter += 1
    if "d" in excitations:
        for a, i, b, j, _, type_idx in theta_picker.get_t2_generator_sa():
            if theta[counter] != 0.0:
                if type_idx == 1:
                    T += (
                        theta[counter]
                        * T2_1_sa_matrix(i, j, a, b, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
                    )
                elif type_idx == 2:
                    T += (
                        theta[counter]
                        * T2_2_sa_matrix(i, j, a, b, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
                    )
                else:
                    raise ValueError(f"Expected type_idx to be in (1,2) got {type_idx}")
            counter += 1
    if "t" in excitations:
        for a, i, b, j, c, k in theta_picker.get_t3_generator():
            if theta[counter] != 0.0:
                T += (
                    theta[counter]
                    * T3_matrix(i, j, k, a, b, c, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
                )
            counter += 1
    if "q" in excitations:
        for a, i, b, j, c, k, d, l in theta_picker.get_t4_generator():
            if theta[counter] != 0.0:
                T += (
                    theta[counter]
                    * T4_matrix(
                        i, j, k, l, a, b, c, d, num_active_orbs, num_elec_alpha, num_elec_beta
                    ).todense()
                )
            counter += 1
    if "5" in excitations:
        for a, i, b, j, c, k, d, l, e, m in theta_picker.get_t5_generator():
            if theta[counter] != 0.0:
                T += (
                    theta[counter]
                    * T5_matrix(
                        i, j, k, l, m, a, b, c, d, e, num_active_orbs, num_elec_alpha, num_elec_beta
                    ).todense()
                )
            counter += 1
    if "6" in excitations:
        for a, i, b, j, c, k, d, l, e, m, f, n in theta_picker.get_t6_generator():
            if theta[counter] != 0.0:
                T += (
                    theta[counter]
                    * T6_matrix(
                        i, j, k, l, m, n, a, b, c, d, e, f, num_active_orbs, num_elec_alpha, num_elec_beta
                    ).todense()
                )
            counter += 1
    assert counter == len(theta)
    A = scipy.linalg.expm(T)
    return A
