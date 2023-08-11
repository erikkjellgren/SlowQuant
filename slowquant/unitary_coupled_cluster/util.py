from collections.abc import Generator, Sequence

import numpy as np
import scipy.linalg
import scipy.sparse as ss

import slowquant.unitary_coupled_cluster.linalg_wrapper as lw
from slowquant.unitary_coupled_cluster.operator_matrix import (
    a_op_spin_matrix,
    epq_matrix,
)
from slowquant.unitary_coupled_cluster.operator_pauli import (
    OperatorPauli,
    a_spin_pauli,
    epq_pauli,
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
        is_spin_conserving: bool = False,
        is_generalized: bool = False,
    ) -> None:
        """Initialize helper class to iterate over active space parameters.

        Args:
            active_occ_spin_idx: Spin index of strongly occupied orbitals.
            active_unocc_spin_idx: Spin index of weakly occupied orbitals.
            is_spin_conserving: Generate spin conserving operators.
            is_generalized: Generate generalized operators.
        """
        self.active_occ_spin_idx: list[int] = []
        self.active_unocc_spin_idx: list[int] = []
        self.active_occ_idx: list[int] = []
        self.active_unocc_idx: list[int] = []
        self.is_spin_conserving = is_spin_conserving
        self.is_generalized = is_generalized
        for idx in active_occ_spin_idx:
            self.active_occ_spin_idx.append(idx)
            if idx // 2 not in self.active_occ_idx:
                self.active_occ_idx.append(idx // 2)
        for idx in active_unocc_spin_idx:
            self.active_unocc_spin_idx.append(idx)
            if idx // 2 not in self.active_unocc_idx:
                self.active_unocc_idx.append(idx // 2)

    def get_t1_generator(
        self, num_spin_orbs: int, num_elec: int
    ) -> Generator[tuple[int, int, int, OperatorPauli], None, None]:
        """Get generate over T1 operators.

        Args:
            num_spin_orbs: Number of spin orbitals.
            num_elec: Number of electrons.

        Returns:
            T1 operator generator.
        """
        return iterate_t1(
            self.active_occ_spin_idx,
            self.active_unocc_spin_idx,
            num_spin_orbs,
            num_elec,
            self.is_spin_conserving,
            self.is_generalized,
        )

    def get_t2_generator(
        self, num_spin_orbs: int, num_elec: int
    ) -> Generator[tuple[int, int, int, int, int, OperatorPauli], None, None]:
        """Get generate over T2 operators.

        Args:
            num_spin_orbs: Number of spin orbitals.
            num_elec: Number of electrons.

        Returns:
            T2 operator generator.
        """
        return iterate_t2(
            self.active_occ_spin_idx,
            self.active_unocc_spin_idx,
            num_spin_orbs,
            num_elec,
            self.is_spin_conserving,
            self.is_generalized,
        )

    def get_t3_generator(
        self, num_spin_orbs: int, num_elec: int
    ) -> Generator[tuple[int, int, int, int, int, int, int, OperatorPauli], None, None]:
        """Get generate over T3 operators.

        Args:
            num_spin_orbs: Number of spin orbitals.
            num_elec: Number of electrons.

        Returns:
            T3 operator generator.
        """
        return iterate_t3(
            self.active_occ_spin_idx,
            self.active_unocc_spin_idx,
            num_spin_orbs,
            num_elec,
            self.is_spin_conserving,
            self.is_generalized,
        )

    def get_t4_generator(
        self, num_spin_orbs: int, num_elec: int
    ) -> Generator[tuple[int, int, int, int, int, int, int, int, int, OperatorPauli], None, None]:
        """Get generate over T4 operators.

        Args:
            num_spin_orbs: Number of spin orbitals.
            num_elec: Number of electrons.

        Returns:
            T4 operator generator.
        """
        return iterate_t4(
            self.active_occ_spin_idx,
            self.active_unocc_spin_idx,
            num_spin_orbs,
            num_elec,
            self.is_spin_conserving,
            self.is_generalized,
        )

    def get_t5_generator(
        self, num_spin_orbs: int, num_elec: int
    ) -> Generator[tuple[int, int, int, int, int, int, int, int, int, int, int, OperatorPauli], None, None]:
        """Get generate over T5 operators.

        Args:
            num_spin_orbs: Number of spin orbitals.
            num_elec: Number of electrons.

        Returns:
            T5 operator generator.
        """
        return iterate_t5(
            self.active_occ_spin_idx,
            self.active_unocc_spin_idx,
            num_spin_orbs,
            num_elec,
            self.is_spin_conserving,
            self.is_generalized,
        )

    def get_t6_generator(
        self, num_spin_orbs: int, num_elec: int
    ) -> Generator[
        tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, OperatorPauli], None, None
    ]:
        """Get generate over T6 operators.

        Args:
            num_spin_orbs: Number of spin orbitals.
            num_elec: Number of electrons.

        Returns:
            T6 operator generator.
        """
        return iterate_t6(
            self.active_occ_spin_idx,
            self.active_unocc_spin_idx,
            num_spin_orbs,
            num_elec,
            self.is_spin_conserving,
            self.is_generalized,
        )

    def get_t1_generator_sa(
        self, num_spin_orbs: int, num_elec: int
    ) -> Generator[tuple[int, int, int, OperatorPauli], None, None]:
        """Get generate over T1 spin-adapted operators.

        Args:
            num_spin_orbs: Number of spin orbitals.
            num_elec: Number of electrons.

        Returns:
            T1 operator generator.
        """
        if self.is_generalized:
            raise ValueError('Spin-adapted not implemented for generelized operators')
        return iterate_t1_sa(self.active_occ_idx, self.active_unocc_idx, num_spin_orbs, num_elec)

    def get_t2_generator_sa(
        self, num_spin_orbs: int, num_elec: int
    ) -> Generator[tuple[int, int, int, int, int, OperatorPauli], None, None]:
        """Get generate over T2 spin-adapted operators.

        Args:
            num_spin_orbs: Number of spin orbitals.
            num_elec: Number of electrons.

        Returns:
            T2 operator generator.
        """
        if self.is_generalized:
            raise ValueError('Spin-adapted not implemented for generelized operators')
        return iterate_t2_sa(self.active_occ_idx, self.active_unocc_idx, num_spin_orbs, num_elec)

    def get_t1_generator_sa_matrix(
        self,
        num_spin_orbs: int,
        num_elec: int,
        use_csr: int = 10,
    ) -> Generator[tuple[int, int, int, OperatorPauli], None, None]:
        """Get generate over T1 spin-adapted operators in matrix form.

        Args:
            num_spin_orbs: Number of spin orbitals.
            num_elec: Number of electrons.
            use_csr: Orbital limit for which sparse matrices will be used.

        Returns:
            T1 operator generator.
        """
        if self.is_generalized:
            raise ValueError('Spin-adapted not implemented for generelized operators')
        return iterate_t1_sa_matrix(
            self.active_occ_idx, self.active_unocc_idx, num_spin_orbs, num_elec, use_csr=use_csr
        )

    def get_t2_generator_sa_matrix(
        self,
        num_spin_orbs: int,
        num_elec: int,
        use_csr: int = 10,
    ) -> Generator[tuple[int, int, int, int, int, OperatorPauli], None, None]:
        """Get generate over T2 spin-adapted operators in matrix form.

        Args:
            num_spin_orbs: Number of spin orbitals.
            num_elec: Number of electrons.
            use_csr: Orbital limit for which sparse matrices will be used.

        Returns:
            T2 operator generator.
        """
        if self.is_generalized:
            raise ValueError('Spin-adapted not implemented for generelized operators')
        return iterate_t2_sa_matrix(
            self.active_occ_idx, self.active_unocc_idx, num_spin_orbs, num_elec, use_csr=use_csr
        )


def iterate_t1_sa(
    active_occ_idx: Sequence[int],
    active_unocc_idx: Sequence[int],
    num_spin_orbs: int,
    num_elec: int,
) -> Generator[tuple[int, int, int, OperatorPauli], None, None]:
    """Iterate over T1 spin-adapted operators.

    Args:
        active_occ_idx: Indices of strongly occupied orbitals.
        active_unocc_idx: Indices of weakly occupied orbitals.
        num_spin_orbs: Number of spin orbitals.
        num_elec: Number of electrons.

    Returns:
        T1 operator iteration.
    """
    theta_idx = -1
    for i in active_occ_idx:
        for a in active_unocc_idx:
            theta_idx += 1
            operator = 2 ** (-1 / 2) * epq_pauli(a, i, num_spin_orbs, num_elec)
            yield theta_idx, i, a, operator


def iterate_t1_sa_matrix(
    active_occ_idx: Sequence[int],
    active_unocc_idx: Sequence[int],
    num_spin_orbs: int,
    num_elec: int,
    use_csr: int,
) -> Generator[tuple[int, int, int, OperatorPauli], None, None]:
    """Iterate over T1 spin-adapted operators in matrix form.

    Args:
        active_occ_idx: Indices of strongly occupied orbitals.
        active_unocc_idx: Indices of weakly occupied orbitals.
        num_spin_orbs: Number of spin orbitals.
        num_elec: Number of electrons.
        use_csr: Use sparse matrices.

    Returns:
        T1 operator iteration.
    """
    theta_idx = -1
    for i in active_occ_idx:
        for a in active_unocc_idx:
            theta_idx += 1
            operator = 2 ** (-1 / 2) * epq_matrix(a, i, num_spin_orbs, num_elec, use_csr=use_csr)
            yield theta_idx, a, i, operator


def iterate_t2_sa(
    active_occ_idx: Sequence[int],
    active_unocc_idx: Sequence[int],
    num_spin_orbs: int,
    num_elec: int,
) -> Generator[tuple[int, int, int, int, int, OperatorPauli], None, None]:
    """Iterate over T2 spin-adapted operators.

    Args:
        active_occ_idx: Indices of strongly occupied orbitals.
        active_unocc_idx: Indices of weakly occupied orbitals.
        num_spin_orbs: Number of spin orbitals.
        num_elec: Number of electrons.

    Returns:
        T2 operator iteration.
    """
    theta_idx = -1
    for i in active_occ_idx:
        for j in active_occ_idx:
            if i > j:
                continue
            for a in active_unocc_idx:
                for b in active_unocc_idx:
                    if a > b:
                        continue
                    theta_idx += 1
                    fac = 1
                    if a == b:
                        fac *= 2
                    if i == j:
                        fac *= 2
                    operator = (
                        1
                        / 2
                        * (fac) ** (-1 / 2)
                        * (
                            epq_pauli(a, i, num_spin_orbs, num_elec)
                            * epq_pauli(b, j, num_spin_orbs, num_elec)
                            + epq_pauli(a, j, num_spin_orbs, num_elec)
                            * epq_pauli(b, i, num_spin_orbs, num_elec)
                        )
                    )
                    yield theta_idx, a, i, b, j, operator
                    if i == j or a == b:
                        continue
                    theta_idx += 1
                    operator = (
                        1
                        / (2 * 3 ** (1 / 2))
                        * (
                            epq_pauli(a, i, num_spin_orbs, num_elec)
                            * epq_pauli(b, j, num_spin_orbs, num_elec)
                            - epq_pauli(a, j, num_spin_orbs, num_elec)
                            * epq_pauli(b, i, num_spin_orbs, num_elec)
                        )
                    )
                    yield theta_idx, a, i, b, j, operator


def iterate_t2_sa_matrix(
    active_occ_idx: Sequence[int],
    active_unocc_idx: Sequence[int],
    num_spin_orbs: int,
    num_elec: int,
    use_csr: int,
) -> Generator[tuple[int, int, int, int, int, OperatorPauli], None, None]:
    """Iterate over T2 spin-adapted operators in matrix form.

    Args:
        active_occ_idx: Indices of strongly occupied orbitals.
        active_unocc_idx: Indices of weakly occupied orbitals.
        num_spin_orbs: Number of spin orbitals.
        num_elec: Number of electrons.
        use_csr: Use sparse matrices.

    Returns:
        T2 operator iteration.
    """
    theta_idx = -1
    for i in active_occ_idx:
        for j in active_occ_idx:
            if i > j:
                continue
            for a in active_unocc_idx:
                for b in active_unocc_idx:
                    if a > b:
                        continue
                    theta_idx += 1
                    fac = 1
                    if a == b:
                        fac *= 2
                    if i == j:
                        fac *= 2
                    operator = (
                        1
                        / 2
                        * (fac) ** (-1 / 2)
                        * (
                            lw.matmul(
                                epq_matrix(a, i, num_spin_orbs, num_elec, use_csr=use_csr),
                                epq_matrix(b, j, num_spin_orbs, num_elec, use_csr=use_csr),
                            )
                            + lw.matmul(
                                epq_matrix(a, j, num_spin_orbs, num_elec, use_csr=use_csr),
                                epq_matrix(b, i, num_spin_orbs, num_elec, use_csr=use_csr),
                            )
                        )
                    )
                    yield theta_idx, a, i, b, j, operator
                    if i == j or a == b:
                        continue
                    theta_idx += 1
                    operator = (
                        1
                        / (2 * 3 ** (1 / 2))
                        * (
                            lw.matmul(
                                epq_matrix(a, i, num_spin_orbs, num_elec, use_csr=use_csr),
                                epq_matrix(b, j, num_spin_orbs, num_elec, use_csr=use_csr),
                            )
                            - lw.matmul(
                                epq_matrix(a, j, num_spin_orbs, num_elec, use_csr=use_csr),
                                epq_matrix(b, i, num_spin_orbs, num_elec, use_csr=use_csr),
                            )
                        )
                    )
                    yield theta_idx, a, i, b, j, operator


def iterate_t1(
    active_occ_spin_idx: list[int],
    active_unocc_spin_idx: list[int],
    num_spin_orbs: int,
    num_elec: int,
    is_spin_conserving: bool,
    is_generalized: bool,
) -> Generator[tuple[int, int, int, OperatorPauli], None, None]:
    """Iterate over T1 operators.

    Args:
        active_occ_idx: Spin indices of strongly occupied orbitals.
        active_unocc_idx: Spin indices of weakly occupied orbitals.
        num_spin_orbs: Number of spin orbitals.
        num_elec: Number of electrons.
        is_spin_conserving: Make spin conserving operators.
        is_generalized: Make generalized operators.

    Returns:
        T1 operator iteration.
    """
    theta_idx = -1
    active_spin_idx = active_occ_spin_idx + active_unocc_spin_idx
    for a in active_spin_idx:
        for i in active_spin_idx:
            if not is_generalized:
                if a in active_occ_spin_idx:
                    continue
                if i in active_unocc_spin_idx:
                    continue
            theta_idx += 1
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
            if (num_alpha % 2 != 0 or num_beta % 2 != 0) and is_spin_conserving:
                continue
            operator = a_spin_pauli(a, True, num_spin_orbs, num_elec)
            operator *= a_spin_pauli(i, False, num_spin_orbs, num_elec)
            yield theta_idx, a, i, operator


def iterate_t2(
    active_occ_spin_idx: list[int],
    active_unocc_spin_idx: list[int],
    num_spin_orbs: int,
    num_elec: int,
    is_spin_conserving: bool,
    is_generalized: bool,
) -> Generator[tuple[int, int, int, int, int, OperatorPauli], None, None]:
    """Iterate over T2 operators.

    Args:
        active_occ_idx: Spin indices of strongly occupied orbitals.
        active_unocc_idx: Spin indices of weakly occupied orbitals.
        num_spin_orbs: Number of spin orbitals.
        num_elec: Number of electrons.
        is_spin_conserving: Make spin conserving operators.
        is_generalized: Make generalized operators.

    Returns:
        T2 operator iteration.
    """
    theta_idx = -1
    active_spin_idx = active_occ_spin_idx + active_unocc_spin_idx
    for a in active_spin_idx:
        for b in active_spin_idx:
            if a >= b:
                continue
            for i in active_spin_idx:
                for j in active_spin_idx:
                    if i >= j:
                        continue
                    if not is_generalized:
                        if a in active_occ_spin_idx:
                            continue
                        if b in active_occ_spin_idx:
                            continue
                        if i in active_unocc_spin_idx:
                            continue
                        if j in active_unocc_spin_idx:
                            continue
                    theta_idx += 1
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
                    if (num_alpha % 2 != 0 or num_beta % 2 != 0) and is_spin_conserving:
                        continue
                    operator = a_spin_pauli(a, True, num_spin_orbs, num_elec)
                    operator = operator * a_spin_pauli(b, True, num_spin_orbs, num_elec)
                    operator = operator * a_spin_pauli(j, False, num_spin_orbs, num_elec)
                    operator = operator * a_spin_pauli(i, False, num_spin_orbs, num_elec)
                    yield theta_idx, a, i, b, j, operator


def iterate_t3(
    active_occ_spin_idx: list[int],
    active_unocc_spin_idx: list[int],
    num_spin_orbs: int,
    num_elec: int,
    is_spin_conserving: bool,
    is_generalized: bool,
) -> Generator[tuple[int, int, int, int, int, int, int, OperatorPauli], None, None]:
    """Iterate over T3 operators.

    Args:
        active_occ_idx: Spin indices of strongly occupied orbitals.
        active_unocc_idx: Spin indices of weakly occupied orbitals.
        num_spin_orbs: Number of spin orbitals.
        num_elec: Number of electrons.
        is_spin_conserving: Make spin conserving operators.
        is_generalized: Make generalized operators.

    Returns:
        T3 operator iteration.
    """
    theta_idx = -1
    active_spin_idx = active_occ_spin_idx + active_unocc_spin_idx
    for a in active_spin_idx:
        for b in active_spin_idx:
            if a >= b:
                continue
            for c in active_spin_idx:
                if b >= c:
                    continue
                for i in active_spin_idx:
                    for j in active_spin_idx:
                        if i >= j:
                            continue
                        for k in active_spin_idx:
                            if j >= k:
                                continue
                            if not is_generalized:
                                if a in active_occ_spin_idx:
                                    continue
                                if b in active_occ_spin_idx:
                                    continue
                                if c in active_occ_spin_idx:
                                    continue
                                if i in active_unocc_spin_idx:
                                    continue
                                if j in active_unocc_spin_idx:
                                    continue
                                if k in active_unocc_spin_idx:
                                    continue
                            theta_idx += 1
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
                            if (num_alpha % 2 != 0 or num_beta % 2 != 0) and is_spin_conserving:
                                continue
                            operator = a_spin_pauli(a, True, num_spin_orbs, num_elec) * a_spin_pauli(
                                b, True, num_spin_orbs, num_elec
                            )
                            operator = operator * a_spin_pauli(c, True, num_spin_orbs, num_elec)
                            operator = operator * a_spin_pauli(k, False, num_spin_orbs, num_elec)
                            operator = operator * a_spin_pauli(j, False, num_spin_orbs, num_elec)
                            operator = operator * a_spin_pauli(i, False, num_spin_orbs, num_elec)
                            yield theta_idx, a, i, b, j, c, k, operator


def iterate_t4(
    active_occ_spin_idx: list[int],
    active_unocc_spin_idx: list[int],
    num_spin_orbs: int,
    num_elec: int,
    is_spin_conserving: bool,
    is_generalized: bool,
) -> Generator[tuple[int, int, int, int, int, int, int, int, int, OperatorPauli], None, None]:
    """Iterate over T4 operators.

    Args:
        active_occ_idx: Spin indices of strongly occupied orbitals.
        active_unocc_idx: Spin indices of weakly occupied orbitals.
        num_spin_orbs: Number of spin orbitals.
        num_elec: Number of electrons.
        is_spin_conserving: Make spin conserving operators.
        is_generalized: Make generalized operators.

    Returns:
        T4 operator iteration.
    """
    theta_idx = -1
    active_spin_idx = active_occ_spin_idx + active_unocc_spin_idx
    for a in active_spin_idx:
        for b in active_spin_idx:
            if a >= b:
                continue
            for c in active_spin_idx:
                if b >= c:
                    continue
                for d in active_spin_idx:
                    if c >= d:
                        continue
                    for i in active_spin_idx:
                        for j in active_spin_idx:
                            if i >= j:
                                continue
                            for k in active_spin_idx:
                                if j >= k:
                                    continue
                                for l in active_spin_idx:
                                    if k >= l:
                                        continue
                                    if not is_generalized:
                                        if a in active_occ_spin_idx:
                                            continue
                                        if b in active_occ_spin_idx:
                                            continue
                                        if c in active_occ_spin_idx:
                                            continue
                                        if d in active_occ_spin_idx:
                                            continue
                                        if i in active_unocc_spin_idx:
                                            continue
                                        if j in active_unocc_spin_idx:
                                            continue
                                        if k in active_unocc_spin_idx:
                                            continue
                                        if l in active_unocc_spin_idx:
                                            continue
                                    theta_idx += 1
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
                                    if (num_alpha % 2 != 0 or num_beta % 2 != 0) and is_spin_conserving:
                                        continue
                                    operator = a_spin_pauli(a, True, num_spin_orbs, num_elec) * a_spin_pauli(
                                        b, True, num_spin_orbs, num_elec
                                    )
                                    operator = operator * a_spin_pauli(c, True, num_spin_orbs, num_elec)
                                    operator = operator * a_spin_pauli(d, True, num_spin_orbs, num_elec)
                                    operator = operator * a_spin_pauli(l, False, num_spin_orbs, num_elec)
                                    operator = operator * a_spin_pauli(k, False, num_spin_orbs, num_elec)
                                    operator = operator * a_spin_pauli(j, False, num_spin_orbs, num_elec)
                                    operator = operator * a_spin_pauli(i, False, num_spin_orbs, num_elec)
                                    yield theta_idx, a, i, b, j, c, k, d, l, operator


def iterate_t5(
    active_occ_spin_idx: list[int],
    active_unocc_spin_idx: list[int],
    num_spin_orbs: int,
    num_elec: int,
    is_spin_conserving: bool,
    is_generalized: bool,
) -> Generator[tuple[int, int, int, int, int, int, int, int, int, int, int, OperatorPauli], None, None]:
    """Iterate over T5 operators.

    Args:
        active_occ_idx: Spin indices of strongly occupied orbitals.
        active_unocc_idx: Spin indices of weakly occupied orbitals.
        num_spin_orbs: Number of spin orbitals.
        num_elec: Number of electrons.
        is_spin_conserving: Make spin conserving operators.
        is_generalized: Make generalized operators.

    Returns:
        T5 operator iteration.
    """
    theta_idx = -1
    active_spin_idx = active_occ_spin_idx + active_unocc_spin_idx
    for a in active_spin_idx:
        for b in active_spin_idx:
            if a >= b:
                continue
            for c in active_spin_idx:
                if b >= c:
                    continue
                for d in active_spin_idx:
                    if c >= d:
                        continue
                    for e in active_spin_idx:
                        if d >= e:
                            continue
                        for i in active_spin_idx:
                            for j in active_spin_idx:
                                if i >= j:
                                    continue
                                for k in active_spin_idx:
                                    if j >= k:
                                        continue
                                    for l in active_spin_idx:
                                        if k >= l:
                                            continue
                                        for m in active_spin_idx:
                                            if l >= m:
                                                continue
                                            if not is_generalized:
                                                if a in active_occ_spin_idx:
                                                    continue
                                                if b in active_occ_spin_idx:
                                                    continue
                                                if c in active_occ_spin_idx:
                                                    continue
                                                if d in active_occ_spin_idx:
                                                    continue
                                                if e in active_occ_spin_idx:
                                                    continue
                                                if i in active_unocc_spin_idx:
                                                    continue
                                                if j in active_unocc_spin_idx:
                                                    continue
                                                if k in active_unocc_spin_idx:
                                                    continue
                                                if l in active_unocc_spin_idx:
                                                    continue
                                                if m in active_unocc_spin_idx:
                                                    continue
                                            theta_idx += 1
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
                                            if (
                                                num_alpha % 2 != 0 or num_beta % 2 != 0
                                            ) and is_spin_conserving:
                                                continue
                                            operator = a_spin_pauli(
                                                a, True, num_spin_orbs, num_elec
                                            ) * a_spin_pauli(b, True, num_spin_orbs, num_elec)
                                            operator = operator * a_spin_pauli(
                                                c, True, num_spin_orbs, num_elec
                                            )
                                            operator = operator * a_spin_pauli(
                                                d, True, num_spin_orbs, num_elec
                                            )
                                            operator = operator * a_spin_pauli(
                                                e, True, num_spin_orbs, num_elec
                                            )
                                            operator = operator * a_spin_pauli(
                                                l, False, num_spin_orbs, num_elec
                                            )
                                            operator = operator * a_spin_pauli(
                                                k, False, num_spin_orbs, num_elec
                                            )
                                            operator = operator * a_spin_pauli(
                                                j, False, num_spin_orbs, num_elec
                                            )
                                            operator = operator * a_spin_pauli(
                                                i, False, num_spin_orbs, num_elec
                                            )
                                            operator = operator * a_spin_pauli(
                                                m, False, num_spin_orbs, num_elec
                                            )
                                            yield theta_idx, a, i, b, j, c, k, d, l, e, m, operator


def iterate_t6(
    active_occ_spin_idx: list[int],
    active_unocc_spin_idx: list[int],
    num_spin_orbs: int,
    num_elec: int,
    is_spin_conserving: bool,
    is_generalized: bool,
) -> Generator[
    tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, OperatorPauli], None, None
]:
    """Iterate over T6 operators.

    Args:
        active_occ_idx: Spin indices of strongly occupied orbitals.
        active_unocc_idx: Spin indices of weakly occupied orbitals.
        num_spin_orbs: Number of spin orbitals.
        num_elec: Number of electrons.
        is_spin_conserving: Make spin conserving operators.
        is_generalized: Make generalized operators.

    Returns:
        T6 operator iteration.
    """
    theta_idx = -1
    active_spin_idx = active_occ_spin_idx + active_unocc_spin_idx
    for a in active_spin_idx:
        for b in active_spin_idx:
            if a >= b:
                continue
            for c in active_spin_idx:
                if b >= c:
                    continue
                for d in active_spin_idx:
                    if c >= d:
                        continue
                    for e in active_spin_idx:
                        if d >= e:
                            continue
                        for f in active_spin_idx:
                            if e >= f:
                                continue
                            for i in active_spin_idx:
                                for j in active_spin_idx:
                                    if i >= j:
                                        continue
                                    for k in active_spin_idx:
                                        if j >= k:
                                            continue
                                        for l in active_spin_idx:
                                            if k >= l:
                                                continue
                                            for m in active_spin_idx:
                                                if l >= m:
                                                    continue
                                                for n in active_spin_idx:
                                                    if m >= n:
                                                        continue
                                                    if not is_generalized:
                                                        if a in active_occ_spin_idx:
                                                            continue
                                                        if b in active_occ_spin_idx:
                                                            continue
                                                        if c in active_occ_spin_idx:
                                                            continue
                                                        if d in active_occ_spin_idx:
                                                            continue
                                                        if e in active_occ_spin_idx:
                                                            continue
                                                        if f in active_occ_spin_idx:
                                                            continue
                                                        if i in active_unocc_spin_idx:
                                                            continue
                                                        if j in active_unocc_spin_idx:
                                                            continue
                                                        if k in active_unocc_spin_idx:
                                                            continue
                                                        if l in active_unocc_spin_idx:
                                                            continue
                                                        if m in active_unocc_spin_idx:
                                                            continue
                                                        if n in active_unocc_spin_idx:
                                                            continue
                                                    theta_idx += 1
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
                                                    if (
                                                        num_alpha % 2 != 0 or num_beta % 2 != 0
                                                    ) and is_spin_conserving:
                                                        continue
                                                    operator = a_spin_pauli(
                                                        a, True, num_spin_orbs, num_elec
                                                    ) * a_spin_pauli(b, True, num_spin_orbs, num_elec)
                                                    operator = operator * a_spin_pauli(
                                                        c, True, num_spin_orbs, num_elec
                                                    )
                                                    operator = operator * a_spin_pauli(
                                                        d, True, num_spin_orbs, num_elec
                                                    )
                                                    operator = operator * a_spin_pauli(
                                                        e, True, num_spin_orbs, num_elec
                                                    )
                                                    operator = operator * a_spin_pauli(
                                                        f, True, num_spin_orbs, num_elec
                                                    )
                                                    operator = operator * a_spin_pauli(
                                                        l, False, num_spin_orbs, num_elec
                                                    )
                                                    operator = operator * a_spin_pauli(
                                                        k, False, num_spin_orbs, num_elec
                                                    )
                                                    operator = operator * a_spin_pauli(
                                                        j, False, num_spin_orbs, num_elec
                                                    )
                                                    operator = operator * a_spin_pauli(
                                                        i, False, num_spin_orbs, num_elec
                                                    )
                                                    operator = operator * a_spin_pauli(
                                                        m, False, num_spin_orbs, num_elec
                                                    )
                                                    operator = operator * a_spin_pauli(
                                                        n, False, num_spin_orbs, num_elec
                                                    )
                                                    yield theta_idx, a, i, b, j, c, k, d, l, e, m, f, n, operator


def construct_ucc_u(
    num_spin_orbs: int,
    num_elec: int,
    theta: Sequence[float],
    theta_picker: ThetaPicker,
    excitations: str,
    allowed_states: np.ndarray | None = None,
    use_csr: int = 10,
) -> np.ndarray:
    """Contruct unitary transformation matrix.

    Args:
       num_spin_orbs: Number of spin orbitals.
       num_elec: Number of electrons.
       theta: Active-space parameters.
              Ordered as (S, D, T, ...).
       theta_picker: Helper class to pick the parameters in the right order.
       excitations: Excitation orders to include.
       allowed_states: Allowed states to consider in the state-vector.
       use_csr: Use sparse matrices after n spin orbitals.

    Returns:
        Unitary transformation matrix.
    """
    if num_spin_orbs >= use_csr:
        t = ss.csr_matrix((2**num_spin_orbs, 2**num_spin_orbs))
    else:
        t = np.zeros((2**num_spin_orbs, 2**num_spin_orbs))
    counter = 0
    if 's' in excitations:
        for _, a, i, operator in theta_picker.get_t1_generator_sa_matrix(
            num_spin_orbs, num_elec, use_csr=use_csr
        ):
            if theta[counter] != 0.0:
                t += theta[counter] * operator
            counter += 1

    if 'd' in excitations:
        for _, a, i, b, j, operator in theta_picker.get_t2_generator_sa_matrix(
            num_spin_orbs, num_elec, use_csr=use_csr
        ):
            if theta[counter] != 0.0:
                t += theta[counter] * operator
            counter += 1

    if 't' in excitations:
        for _, a, i, b, j, c, k, _ in theta_picker.get_t3_generator(0, 0):
            if theta[counter] != 0.0:
                tmp = a_op_spin_matrix(a, True, num_spin_orbs, use_csr=use_csr).dot(
                    a_op_spin_matrix(b, True, num_spin_orbs, use_csr=use_csr)
                )
                tmp = tmp.dot(a_op_spin_matrix(c, True, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(k, False, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(j, False, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(i, False, num_spin_orbs, use_csr=use_csr))
                t += theta[counter] * tmp
            counter += 1

    if 'q' in excitations:
        for _, a, i, b, j, c, k, d, l, _ in theta_picker.get_t4_generator(0, 0):
            if theta[counter] != 0.0:
                tmp = a_op_spin_matrix(a, True, num_spin_orbs, use_csr=use_csr).dot(
                    a_op_spin_matrix(b, True, num_spin_orbs, use_csr=use_csr)
                )
                tmp = tmp.dot(a_op_spin_matrix(c, True, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(d, True, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(l, False, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(k, False, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(j, False, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(i, False, num_spin_orbs, use_csr=use_csr))
                t += theta[counter] * tmp
            counter += 1
    if '5' in excitations:
        for _, a, i, b, j, c, k, d, l, e, m, _ in theta_picker.get_t5_generator(0, 0):
            if theta[counter] != 0.0:
                tmp = a_op_spin_matrix(a, True, num_spin_orbs, use_csr=use_csr).dot(
                    a_op_spin_matrix(b, True, num_spin_orbs, use_csr=use_csr)
                )
                tmp = tmp.dot(a_op_spin_matrix(c, True, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(d, True, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(e, True, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(l, False, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(k, False, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(j, False, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(i, False, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(m, False, num_spin_orbs, use_csr=use_csr))
                t += theta[counter] * tmp
            counter += 1
    if '6' in excitations:
        for _, a, i, b, j, c, k, d, l, e, m, f, n, _ in theta_picker.get_t6_generator(0, 0):
            if theta[counter] != 0.0:
                tmp = a_op_spin_matrix(a, True, num_spin_orbs, use_csr=use_csr).dot(
                    a_op_spin_matrix(b, True, num_spin_orbs, use_csr=use_csr)
                )
                tmp = tmp.dot(a_op_spin_matrix(c, True, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(d, True, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(e, True, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(f, True, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(l, False, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(k, False, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(j, False, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(i, False, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(m, False, num_spin_orbs, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(n, False, num_spin_orbs, use_csr=use_csr))
                t += theta[counter] * tmp
            counter += 1
    assert counter == len(theta)

    T = t - t.conjugate().transpose()
    if allowed_states is not None:
        T = T[allowed_states, :]
        T = T[:, allowed_states]
    A = lw.expm(T)
    return A
