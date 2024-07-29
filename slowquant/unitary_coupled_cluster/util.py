from collections.abc import Generator, Sequence
from typing import Any

import numpy as np
import scipy.linalg

from slowquant.unitary_coupled_cluster.operator_matrix import (
    T1_matrix,
    T1_sa_matrix,
    T2_1_sa_matrix,
    T2_2_sa_matrix,
    T2_matrix,
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


def iterate_t1_sa_generalized(
    num_orbs: int,
) -> Generator[tuple[int, int, float], None, None]:
    """Iterate over T1 spin-adapted operators.

    Args:
        num_orbs: Number of active spatial orbitals.

    Returns:
        T1 operator iteration.
    """
    for i in range(num_orbs):
        for a in range(i + 1, num_orbs):
            fac = 2 ** (-1 / 2)
            yield a, i, fac


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
                num_alpha -= 1
            else:
                num_beta -= 1
            if num_alpha != 0 or num_beta != 0:
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
                        num_alpha -= 1
                    else:
                        num_beta -= 1
                    if j % 2 == 0:
                        num_alpha -= 1
                    else:
                        num_beta -= 1
                    if num_alpha != 0 or num_beta != 0:
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
                                num_alpha -= 1
                            else:
                                num_beta -= 1
                            if j % 2 == 0:
                                num_alpha -= 1
                            else:
                                num_beta -= 1
                            if k % 2 == 0:
                                num_alpha -= 1
                            else:
                                num_beta -= 1
                            if num_alpha != 0 or num_beta != 0:
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
                                        num_alpha -= 1
                                    else:
                                        num_beta -= 1
                                    if j % 2 == 0:
                                        num_alpha -= 1
                                    else:
                                        num_beta -= 1
                                    if k % 2 == 0:
                                        num_alpha -= 1
                                    else:
                                        num_beta -= 1
                                    if l % 2 == 0:
                                        num_alpha -= 1
                                    else:
                                        num_beta -= 1
                                    if num_alpha != 0 or num_beta != 0:
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
                                                num_alpha -= 1
                                            else:
                                                num_beta -= 1
                                            if j % 2 == 0:
                                                num_alpha -= 1
                                            else:
                                                num_beta -= 1
                                            if k % 2 == 0:
                                                num_alpha -= 1
                                            else:
                                                num_beta -= 1
                                            if l % 2 == 0:
                                                num_alpha -= 1
                                            else:
                                                num_beta -= 1
                                            if m % 2 == 0:
                                                num_alpha -= 1
                                            else:
                                                num_beta -= 1
                                            if num_alpha != 0 or num_beta != 0:
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
                                                        num_alpha -= 1
                                                    else:
                                                        num_beta -= 1
                                                    if j % 2 == 0:
                                                        num_alpha -= 1
                                                    else:
                                                        num_beta -= 1
                                                    if k % 2 == 0:
                                                        num_alpha -= 1
                                                    else:
                                                        num_beta -= 1
                                                    if l % 2 == 0:
                                                        num_alpha -= 1
                                                    else:
                                                        num_beta -= 1
                                                    if m % 2 == 0:
                                                        num_alpha -= 1
                                                    else:
                                                        num_beta -= 1
                                                    if n % 2 == 0:
                                                        num_alpha -= 1
                                                    else:
                                                        num_beta -= 1
                                                    if num_alpha != 0 or num_beta != 0:
                                                        continue
                                                    yield a, i, b, j, c, k, d, l, e, m, f, n


def iterate_pair_t2(
    active_occ_idx: Sequence[int],
    active_unocc_idx: Sequence[int],
) -> Generator[tuple[int, int, int, int], None, None]:
    """Iterate over pair T2 operators.

    Args:
        active_occ_idx: Indices of strongly occupied orbitals.
        active_unocc_idx: Indices of weakly occupied orbitals.

    Returns:
        T2 operator iteration.
    """
    for i in active_occ_idx:
        for a in active_unocc_idx:
            yield 2 * a, 2 * i, 2 * a + 1, 2 * i + 1


def iterate_pair_t2_generalized(
    num_orbs: int,
) -> Generator[tuple[int, int, int, int], None, None]:
    """Iterate over pair T2 operators.

    Args:
        num_orbs: Number of active spatial orbitals.

    Returns:
        T2 operator iteration.
    """
    for i in range(num_orbs):
        for a in range(i + 1, num_orbs):
            yield 2 * a, 2 * i, 2 * a + 1, 2 * i + 1


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
                        i,
                        j,
                        k,
                        l,
                        a,
                        b,
                        c,
                        d,
                        num_active_orbs,
                        num_elec_alpha,
                        num_elec_beta,
                    ).todense()
                )
            counter += 1
    if "5" in excitations:
        for a, i, b, j, c, k, d, l, e, m in theta_picker.get_t5_generator():
            if theta[counter] != 0.0:
                T += (
                    theta[counter]
                    * T5_matrix(
                        i,
                        j,
                        k,
                        l,
                        m,
                        a,
                        b,
                        c,
                        d,
                        e,
                        num_active_orbs,
                        num_elec_alpha,
                        num_elec_beta,
                    ).todense()
                )
            counter += 1
    if "6" in excitations:
        for a, i, b, j, c, k, d, l, e, m, f, n in theta_picker.get_t6_generator():
            if theta[counter] != 0.0:
                T += (
                    theta[counter]
                    * T6_matrix(
                        i,
                        j,
                        k,
                        l,
                        m,
                        n,
                        a,
                        b,
                        c,
                        d,
                        e,
                        f,
                        num_active_orbs,
                        num_elec_alpha,
                        num_elec_beta,
                    ).todense()
                )
            counter += 1
    assert counter == len(theta)
    A = scipy.linalg.expm(T)
    return A


class UpsStructure:
    def __init__(self) -> None:
        self.excitation_indicies: list[tuple[int]] = []
        self.excitation_operator_type: list[str] = []
        self.n_params = 0

    def create_tups(self, num_active_orbs: int, ansatz_options: dict[str, Any]) -> None:
        """tUPS ansatz.

        #. 10.1103/PhysRevResearch.6.023300
        #. 10.1088/1367-2630/ac2cb3

        Ansatz Options:
            * n_layers [int]: Number of layers.
            * do_qnp [bool]: Do QNP tiling.

        Args:
            num_active_orbs: Number of spatial active orbitals.
            ansatz_options: Ansatz options.

        Returns:
            tUPS ansatz.
        """
        valid_options = ("n_layers", "do_qnp")
        for option in ansatz_options:
            if option not in valid_options:
                raise ValueError(f"Got unknown option for tUPS, {option}. Valid options are: {valid_options}")
        if "n_layers" not in ansatz_options.keys():
            raise ValueError("tUPS require the option 'n_layers'")
        n_layers = ansatz_options["n_layers"]
        if "do_qnp" in ansatz_options.keys():
            do_qnp = ansatz_options["do_qnp"]
        else:
            do_qnp = False
        for _ in range(n_layers):
            for p in range(0, num_active_orbs - 1, 2):
                if not do_qnp:
                    # First single
                    self.excitation_operator_type.append("tups_single")
                    self.excitation_indicies.append((p,))
                    self.n_params += 1
                # Double
                self.excitation_operator_type.append("tups_double")
                self.excitation_indicies.append((p,))
                self.n_params += 1
                # Second single
                self.excitation_operator_type.append("tups_single")
                self.excitation_indicies.append((p,))
                self.n_params += 1
            for p in range(1, num_active_orbs - 2, 2):
                if not do_qnp:
                    # First single
                    self.excitation_operator_type.append("tups_single")
                    self.excitation_indicies.append((p,))
                    self.n_params += 1
                # Double
                self.excitation_operator_type.append("tups_double")
                self.excitation_indicies.append((p,))
                self.n_params += 1
                # Second single
                self.excitation_operator_type.append("tups_single")
                self.excitation_indicies.append((p,))
                self.n_params += 1

    def create_fUCCSD(self, states: list[list[str]], ansatz_options: dict[str, Any]) -> None:
        valid_options = ()
        for option in ansatz_options:
            if option not in valid_options:
                raise ValueError(f"Got unknown option for fUCC, {option}. Valid options are: {valid_options}")
        occupied = []
        unoccupied = []
        for state in states:
            for det in state:
                occ_tmp = []
                unocc_tmp = []
                for i, occ in enumerate(det):
                    if occ == "1":
                        occ_tmp.append(i)
                    else:
                        unocc_tmp.append(i)
                occupied.append(occ_tmp)
                unoccupied.append(unocc_tmp)
        for occ, unocc in zip(occupied, unoccupied):
            for a, i in iterate_t1(occ, unocc):
                if a < i:
                    i, a = a, i
                if (i, a) not in self.excitation_indicies:
                    self.excitation_operator_type.append("single")
                    self.excitation_indicies.append((i, a))
                    self.n_params += 1
        for occ, unocc in zip(occupied, unoccupied):
            for a, i, b, j in iterate_t2(occ, unocc):
                if i % 2 == j % 2 == a % 2 == b % 2:
                    i, j, a, b = np.sort([i, j, a, b])
                elif i % 2 == a % 2:
                    if a < i:
                        i, a = a, i
                    if b < j:
                        j, b = b, j
                else:
                    if a < j:
                        j, a = a, j
                    if b < i:
                        i, b = b, i
                if (i, j, a, b) not in self.excitation_indicies:
                    self.excitation_operator_type.append("double")
                    self.excitation_indicies.append((i, j, a, b))
                    self.n_params += 1

    def create_safUCCSpD(
        self, occ: list[int], unocc: list[int], num_orbs: int, ansatz_options: dict[str, Any]
    ) -> None:
        valid_options = ("do_generalized",)
        for option in ansatz_options:
            if option not in valid_options:
                raise ValueError(
                    f"Got unknown option for safUCCSpD, {option}. Valid options are: {valid_options}"
                )
        if "do_generalized" in ansatz_options.keys():
            do_generalized = ansatz_options["do_generalized"]
        else:
            do_generalized = False
        if do_generalized:
            for a, i, _ in iterate_t1_sa_generalized(num_orbs):
                if (i, a) not in self.excitation_indicies:
                    self.excitation_operator_type.append("sa_single")
                    self.excitation_indicies.append((i, a))
                    self.n_params += 1
            for a, i, b, j in iterate_pair_t2_generalized(num_orbs):
                if (i, j, a, b) not in self.excitation_indicies:
                    self.excitation_operator_type.append("double")
                    self.excitation_indicies.append((i, j, a, b))
                    self.n_params += 1
        else:
            for a, i, _ in iterate_t1_sa(occ, unocc):
                if (i, a) not in self.excitation_indicies:
                    self.excitation_operator_type.append("sa_single")
                    self.excitation_indicies.append((i, a))
                    self.n_params += 1
            for a, i, b, j in iterate_pair_t2(occ, unocc):
                if (i, j, a, b) not in self.excitation_indicies:
                    self.excitation_operator_type.append("double")
                    self.excitation_indicies.append((i, j, a, b))
                    self.n_params += 1


def construct_ups_state(
    state: np.ndarray,
    num_active_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
    thetas: Sequence[float],
    ups_struct: UpsStructure,
    dagger: bool = False,
) -> np.ndarray:
    r"""

    #. 10.48550/arXiv.2303.10825, Eq. 15
    """
    tmp = state.copy()
    order = 1
    if dagger:
        order = -1
    for exc_type, exc_indices, theta in zip(
        ups_struct.excitation_operator_type[::order], ups_struct.excitation_indicies[::order], thetas[::order]
    ):
        if abs(theta) < 10**-14:
            continue
        if dagger:
            theta = -theta
        if exc_type in ("tups_single", "sa_single"):
            if exc_type == "tups_single":
                (p,) = exc_indices
                Ta = T1_matrix(p * 2, (p + 1) * 2, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
                Tb = T1_matrix(
                    p * 2 + 1, (p + 1) * 2 + 1, num_active_orbs, num_elec_alpha, num_elec_beta
                ).todense()
            elif exc_type == "sa_single":
                (i, a) = exc_indices
                Ta = T1_matrix(i * 2, a * 2, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
                Tb = T1_matrix(i * 2 + 1, a * 2 + 1, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
            tmp = (
                tmp
                + np.sin(2 ** (-1 / 2) * theta) * np.matmul(Ta, tmp)
                + (1 - np.cos(2 ** (-1 / 2) * theta)) * np.matmul(Ta, np.matmul(Ta, tmp))
            )
            tmp = (
                tmp
                + np.sin(2 ** (-1 / 2) * theta) * np.matmul(Tb, tmp)
                + (1 - np.cos(2 ** (-1 / 2) * theta)) * np.matmul(Tb, np.matmul(Tb, tmp))
            )
        elif exc_type in ("tups_double", "single", "double"):
            if exc_type == "tups_double":
                (p,) = exc_indices
                T = T2_1_sa_matrix(
                    p, p, p + 1, p + 1, num_active_orbs, num_elec_alpha, num_elec_beta
                ).todense()
            elif exc_type == "single":
                (i, a) = exc_indices
                T = T1_matrix(i, a, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
            elif exc_type == "double":
                (i, j, a, b) = exc_indices
                T = T2_matrix(i, j, a, b, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
            tmp = (
                tmp
                + np.sin(theta) * np.matmul(T, tmp)
                + (1 - np.cos(theta)) * np.matmul(T, np.matmul(T, tmp))
            )
        else:
            raise ValueError(f"Got unknown excitation type, {exc_type}")
    return tmp


def propagate_unitary(
    state: np.ndarray,
    idx: int,
    num_active_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
    thetas: Sequence[float],
    ups_struct: UpsStructure,
) -> np.ndarray:
    exc_type = ups_struct.excitation_operator_type[idx]
    exc_indices = ups_struct.excitation_indicies[idx]
    theta = thetas[idx]
    if abs(theta) < 10**-14:
        return np.copy(state)
    if exc_type in ("tups_single", "sa_single"):
        if exc_type == "tups_single":
            (p,) = exc_indices
            Ta = T1_matrix(p * 2, (p + 1) * 2, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
            Tb = T1_matrix(
                p * 2 + 1, (p + 1) * 2 + 1, num_active_orbs, num_elec_alpha, num_elec_beta
            ).todense()
        elif exc_type == "sa_single":
            (i, a) = exc_indices
            Ta = T1_matrix(i * 2, a * 2, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
            Tb = T1_matrix(i * 2 + 1, a * 2 + 1, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
        A = 2 ** (-1 / 2)
        tmp = (
            state
            + np.sin(A * theta) * np.matmul(Ta, state)
            + (1 - np.cos(A * theta)) * np.matmul(Ta, np.matmul(Ta, state))
        )
        tmp = (
            tmp
            + np.sin(A * theta) * np.matmul(Tb, tmp)
            + (1 - np.cos(A * theta)) * np.matmul(Tb, np.matmul(Tb, tmp))
        )
    elif exc_type in ("tups_double", "single", "double"):
        if exc_type == "tups_double":
            (p,) = exc_indices
            T = T2_1_sa_matrix(p, p, p + 1, p + 1, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
        elif exc_type == "single":
            (i, a) = exc_indices
            T = T1_matrix(i, a, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
        elif exc_type == "double":
            (i, j, a, b) = exc_indices
            T = T2_matrix(i, j, a, b, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
        tmp = (
            state
            + np.sin(theta) * np.matmul(T, state)
            + (1 - np.cos(theta)) * np.matmul(T, np.matmul(T, state))
        )
    else:
        raise ValueError(f"Got unknown excitation type, {exc_type}")
    return tmp


def get_grad_action(
    state: np.ndarray,
    idx: int,
    num_active_orbs: int,
    num_elec_alpha: int,
    num_elec_beta: int,
    ups_struct: UpsStructure,
) -> np.ndarray:
    exc_type = ups_struct.excitation_operator_type[idx]
    exc_indices = ups_struct.excitation_indicies[idx]
    if exc_type in ("tups_single", "sa_single"):
        if exc_type == "tups_single":
            (p,) = exc_indices
            Ta = T1_matrix(p * 2, (p + 1) * 2, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
            Tb = T1_matrix(
                p * 2 + 1, (p + 1) * 2 + 1, num_active_orbs, num_elec_alpha, num_elec_beta
            ).todense()
        elif exc_type == "sa_single":
            (i, a) = exc_indices
            Ta = T1_matrix(i * 2, a * 2, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
            Tb = T1_matrix(i * 2 + 1, a * 2 + 1, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
        A = 2 ** (-1 / 2)
        tmp = np.matmul(A * (Ta + Tb), state)
    elif exc_type in ("tups_double", "single", "double"):
        if exc_type == "tups_double":
            (p,) = exc_indices
            T = T2_1_sa_matrix(p, p, p + 1, p + 1, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
        elif exc_type == "single":
            (i, a) = exc_indices
            T = T1_matrix(i, a, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
        elif exc_type == "double":
            (i, j, a, b) = exc_indices
            T = T2_matrix(i, j, a, b, num_active_orbs, num_elec_alpha, num_elec_beta).todense()
        tmp = np.matmul(T, state)
    else:
        raise ValueError(f"Got unknown excitation type, {exc_type}")
    return tmp
