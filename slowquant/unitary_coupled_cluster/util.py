import time
from typing import Generator, Sequence

import numpy as np
import scipy.linalg

import slowquant.unitary_coupled_cluster.linalg_wrapper as lw
from slowquant.unitary_coupled_cluster.base import (
    PauliOperator,
    a_op_spin,
    a_op_spin_matrix,
)


def construct_integral_trans_mat(
    c_orthonormal: np.ndarray, kappa: Sequence[float], kappa_idx: Sequence[Sequence[int]]
) -> np.ndarray:
    kappa_mat = np.zeros_like(c_orthonormal)
    for kappa_val, (p, q) in zip(kappa, kappa_idx):
        kappa_mat[p, q] = kappa_val
        kappa_mat[q, p] = -kappa_val
    c_trans = np.matmul(c_orthonormal, scipy.linalg.expm(-kappa_mat))
    return c_trans


class ThetaPicker:
    def __init__(
        self,
        active_occ: list[int],
        active_unocc: list[int],
        is_spin_conserving: bool = False,
        is_generalized: bool = False,
        deshift: int = None,
    ) -> None:
        self.active_occ = active_occ.copy()
        self.active_unocc = active_unocc.copy()
        self.is_spin_conserving = is_spin_conserving
        self.is_generalized = is_generalized
        if deshift is not None:
            for i in range(len(self.active_occ)):
                self.active_occ[i] += deshift
            for i in range(len(self.active_unocc)):
                self.active_unocc[i] += deshift

    def get_T1_generator(
        self, num_spin_orbs: int, num_elec: int
    ) -> Generator[tuple[int, int, int, PauliOperator], None, None]:
        return iterate_T1(
            self.active_occ,
            self.active_unocc,
            num_spin_orbs,
            num_elec,
            self.is_spin_conserving,
            self.is_generalized,
        )

    def get_T2_generator(
        self, num_spin_orbs: int, num_elec: int
    ) -> Generator[tuple[int, int, int, int, int, PauliOperator], None, None]:
        return iterate_T2(
            self.active_occ,
            self.active_unocc,
            num_spin_orbs,
            num_elec,
            self.is_spin_conserving,
            self.is_generalized,
        )

    def get_T3_generator(
        self, num_spin_orbs: int, num_elec: int
    ) -> Generator[tuple[int, int, int, int, int, int, int, PauliOperator], None, None]:
        return iterate_T3(
            self.active_occ,
            self.active_unocc,
            num_spin_orbs,
            num_elec,
            self.is_spin_conserving,
            self.is_generalized,
        )

    def get_T4_generator(
        self, num_spin_orbs: int, num_elec: int
    ) -> Generator[tuple[int, int, int, int, int, int, int, int, int, PauliOperator], None, None]:
        return iterate_T4(
            self.active_occ,
            self.active_unocc,
            num_spin_orbs,
            num_elec,
            self.is_spin_conserving,
            self.is_generalized,
        )


def iterate_T1(
    active_occ: list[int],
    active_unocc: list[int],
    num_spin_orbs: int,
    num_elec: int,
    is_spin_conserving: bool,
    is_generalized: bool,
) -> tuple[int]:
    theta_idx = -1
    for a in active_occ + active_unocc:
        for i in active_occ + active_unocc:
            if not is_generalized:
                if a in active_occ:
                    continue
                if i in active_unocc:
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
            operator = PauliOperator(a_op_spin(a, True, num_spin_orbs, num_elec))
            operator *= PauliOperator(a_op_spin(i, False, num_spin_orbs, num_elec))
            yield theta_idx, a, i, operator


def iterate_T2(
    active_occ: list[int],
    active_unocc: list[int],
    num_spin_orbs: int,
    num_elec: int,
    is_spin_conserving: bool,
    is_generalized: bool,
) -> tuple[int]:
    theta_idx = -1
    for a in active_occ + active_unocc:
        for b in active_occ + active_unocc:
            if a >= b:
                continue
            for i in active_occ + active_unocc:
                for j in active_occ + active_unocc:
                    if i >= j:
                        continue
                    if not is_generalized:
                        if a in active_occ:
                            continue
                        if b in active_occ:
                            continue
                        if i in active_unocc:
                            continue
                        if j in active_unocc:
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
                    operator = PauliOperator(a_op_spin(a, True, num_spin_orbs, num_elec))
                    operator = operator * PauliOperator(a_op_spin(b, True, num_spin_orbs, num_elec))
                    operator = operator * PauliOperator(a_op_spin(j, False, num_spin_orbs, num_elec))
                    operator = operator * PauliOperator(a_op_spin(i, False, num_spin_orbs, num_elec))
                    yield theta_idx, a, i, b, j, operator


def iterate_T3(
    active_occ: list[int],
    active_unocc: list[int],
    num_spin_orbs: int,
    num_elec: int,
    is_spin_conserving: bool,
    is_generalized: bool,
) -> tuple[int]:
    theta_idx = -1
    for a in active_occ + active_unocc:
        for b in active_occ + active_unocc:
            if a >= b:
                continue
            for c in active_occ + active_unocc:
                if b >= c:
                    continue
                for i in active_occ + active_unocc:
                    for j in active_occ + active_unocc:
                        if i >= j:
                            continue
                        for k in active_occ + active_unocc:
                            if j >= k:
                                continue
                            if not is_generalized:
                                if a in active_occ:
                                    continue
                                if b in active_occ:
                                    continue
                                if c in active_occ:
                                    continue
                                if i in active_unocc:
                                    continue
                                if j in active_unocc:
                                    continue
                                if k in active_unocc:
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
                            operator = PauliOperator(
                                a_op_spin(a, True, num_spin_orbs, num_elec)
                            ) * PauliOperator(a_op_spin(b, True, num_spin_orbs, num_elec))
                            operator = operator * PauliOperator(a_op_spin(c, True, num_spin_orbs, num_elec))
                            operator = operator * PauliOperator(a_op_spin(k, False, num_spin_orbs, num_elec))
                            operator = operator * PauliOperator(a_op_spin(j, False, num_spin_orbs, num_elec))
                            operator = operator * PauliOperator(a_op_spin(i, False, num_spin_orbs, num_elec))
                            yield theta_idx, a, i, b, j, c, k, operator


def iterate_T4(
    active_occ: list[int],
    active_unocc: list[int],
    num_spin_orbs: int,
    num_elec: int,
    is_spin_conserving: bool,
    is_generalized: bool,
) -> tuple[int]:
    theta_idx = -1
    for a in active_occ + active_unocc:
        for b in active_occ + active_unocc:
            if a >= b:
                continue
            for c in active_occ + active_unocc:
                if b >= c:
                    continue
                for d in active_occ + active_unocc:
                    if c >= d:
                        continue
                    for i in active_occ + active_unocc:
                        for j in active_occ + active_unocc:
                            if i >= j:
                                continue
                            for k in active_occ + active_unocc:
                                if j >= k:
                                    continue
                                for l in active_occ + active_unocc:
                                    if k >= l:
                                        continue
                                    if not is_generalized:
                                        if a in active_occ:
                                            continue
                                        if b in active_occ:
                                            continue
                                        if c in active_occ:
                                            continue
                                        if d in active_occ:
                                            continue
                                        if i in active_unocc:
                                            continue
                                        if j in active_unocc:
                                            continue
                                        if k in active_unocc:
                                            continue
                                        if l in active_unocc:
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
                                    operator = PauliOperator(
                                        a_op_spin(a, True, num_spin_orbs, num_elec)
                                    ) * PauliOperator(a_op_spin(b, True, num_spin_orbs, num_elec))
                                    operator = operator * PauliOperator(
                                        a_op_spin(c, True, num_spin_orbs, num_elec)
                                    )
                                    operator = operator * PauliOperator(
                                        a_op_spin(d, True, num_spin_orbs, num_elec)
                                    )
                                    operator = operator * PauliOperator(
                                        a_op_spin(l, False, num_spin_orbs, num_elec)
                                    )
                                    operator = operator * PauliOperator(
                                        a_op_spin(k, False, num_spin_orbs, num_elec)
                                    )
                                    operator = operator * PauliOperator(
                                        a_op_spin(j, False, num_spin_orbs, num_elec)
                                    )
                                    operator = operator * PauliOperator(
                                        a_op_spin(i, False, num_spin_orbs, num_elec)
                                    )
                                    yield theta_idx, a, i, b, j, c, k, d, l, operator


def construct_UCC_U(
    num_spin_orbs: int,
    num_elec: int,
    theta: Sequence[float],
    theta_picker: ThetaPicker,
    excitations: str,
    allowed_states: np.ndarray | None = None,
    use_csr: int = 10,
) -> np.ndarray:
    t = np.zeros((2**num_spin_orbs, 2**num_spin_orbs))
    counter = 0
    start = time.time()
    if "s" in excitations:
        for _, a, i, _ in theta_picker.get_T1_generator(0, 0):
            if theta[counter] != 0.0:
                tmp = a_op_spin_matrix(a, True, num_spin_orbs, num_elec, use_csr=use_csr).dot(
                    a_op_spin_matrix(i, False, num_spin_orbs, num_elec, use_csr=use_csr)
                )
                t += theta[counter] * tmp
            counter += 1

    if "d" in excitations:
        for _, a, i, b, j, _ in theta_picker.get_T2_generator(0, 0):
            if theta[counter] != 0.0:
                tmp = a_op_spin_matrix(a, True, num_spin_orbs, num_elec, use_csr=use_csr).dot(
                    a_op_spin_matrix(b, True, num_spin_orbs, num_elec, use_csr=use_csr)
                )
                tmp = tmp.dot(a_op_spin_matrix(j, False, num_spin_orbs, num_elec, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(i, False, num_spin_orbs, num_elec, use_csr=use_csr))
                t += theta[counter] * tmp
            counter += 1

    if "t" in excitations:
        for _, a, i, b, j, c, k, _ in theta_picker.get_T3_generator(0, 0):
            if theta[counter] != 0.0:
                tmp = a_op_spin_matrix(a, True, num_spin_orbs, num_elec, use_csr=use_csr).dot(
                    a_op_spin_matrix(b, True, num_spin_orbs, num_elec, use_csr=use_csr)
                )
                tmp = tmp.dot(a_op_spin_matrix(c, True, num_spin_orbs, num_elec, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(k, False, num_spin_orbs, num_elec, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(j, False, num_spin_orbs, num_elec, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(i, False, num_spin_orbs, num_elec, use_csr=use_csr))
                t += theta[counter] * tmp
            counter += 1

    if "q" in excitations:
        for _, a, i, b, j, c, k, d, l, _ in theta_picker.get_T4_generator(0, 0):
            if theta[counter] != 0.0:
                tmp = a_op_spin_matrix(a, True, num_spin_orbs, num_elec, use_csr=use_csr).dot(
                    a_op_spin_matrix(b, True, num_spin_orbs, num_elec, use_csr=use_csr)
                )
                tmp = tmp.dot(a_op_spin_matrix(c, True, num_spin_orbs, num_elec, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(d, True, num_spin_orbs, num_elec, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(l, False, num_spin_orbs, num_elec, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(k, False, num_spin_orbs, num_elec, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(j, False, num_spin_orbs, num_elec, use_csr=use_csr))
                tmp = tmp.dot(a_op_spin_matrix(i, False, num_spin_orbs, num_elec, use_csr=use_csr))
                t += theta[counter] * tmp
            counter += 1
    assert counter == len(theta)

    T = t - t.conjugate().transpose()
    if allowed_states is not None:
        T = T[allowed_states, :]
        T = T[:, allowed_states]
    A = lw.expm(T)
    return A