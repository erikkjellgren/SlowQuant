import time
from typing import Generator

import numpy as np
import scipy.linalg

from slowquant.unitary_coupled_cluster.base import a_op_spin, PauliOperator


def construct_integral_trans_mat(
    c_orthonormal: np.ndarray, kappa: list[float], kappa_idx: list[list[int]]
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
    ) -> None:
        self.active_occ = active_occ
        self.active_unocc = active_unocc
        self.is_spin_conserving = is_spin_conserving
        self.is_generalized = is_generalized

    def get_T1_generator(self, num_spin_orbs: int, num_elec: int) -> Generator[tuple[int, int, int, PauliOperator], None, None]:
        return iterate_T1(self.active_occ, self.active_unocc, num_spin_orbs, num_elec, self.is_spin_conserving, self.is_generalized)

    def get_T2_generator(self, num_spin_orbs: int, num_elec: int) -> Generator[tuple[int, int, int, int, int, PauliOperator], None, None]:
        return iterate_T2(self.active_occ, self.active_unocc, num_spin_orbs, num_elec, self.is_spin_conserving, self.is_generalized)

    def get_T3_generator(self, num_spin_orbs: int, num_elec: int) -> Generator[tuple[int, int, int, int, int, int, int, PauliOperator], None, None]:
        return iterate_T3(self.active_occ, self.active_unocc, num_spin_orbs, num_elec, self.is_spin_conserving, self.is_generalized)

    def get_T4_generator(self, num_spin_orbs: int, num_elec: int) -> Generator[tuple[int, int, int, int, int, int, int, int, int, PauliOperator], None, None]:
        return iterate_T4(self.active_occ, self.active_unocc, num_spin_orbs, num_elec, self.is_spin_conserving, self.is_generalized)


def iterate_T1(
    active_occ: list[int], active_unocc: list[int], num_spin_orbs: int, num_elec: int, is_spin_conserving: bool, is_generalized: bool
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
            operator = PauliOperator(a_op_spin(a, True, num_spin_orbs, num_elec)) * PauliOperator(a_op_spin(i, False, num_spin_orbs, num_elec))
            yield theta_idx, a, i, operator


def iterate_T2(
    active_occ: list[int], active_unocc: list[int], num_spin_orbs: int, num_elec: int, is_spin_conserving: bool, is_generalized: bool
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
    active_occ: list[int], active_unocc: list[int], num_spin_orbs: int, num_elec: int, is_spin_conserving: bool, is_generalized: bool
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
                            operator = PauliOperator(a_op_spin(a, True, num_spin_orbs, num_elec)) * PauliOperator(
                                a_op_spin(b, True, num_spin_orbs, num_elec)
                            )
                            operator = operator * PauliOperator(a_op_spin(c, True, num_spin_orbs, num_elec))
                            operator = operator * PauliOperator(a_op_spin(k, False, num_spin_orbs, num_elec))
                            operator = operator * PauliOperator(a_op_spin(j, False, num_spin_orbs, num_elec))
                            operator = operator * PauliOperator(a_op_spin(i, False, num_spin_orbs, num_elec))
                            yield theta_idx, a, i, b, j, c, k, operator


def iterate_T4(
    active_occ: list[int], active_unocc: list[int], num_spin_orbs: int, num_elec: int, is_spin_conserving: bool, is_generalized: bool
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
                                    operator = PauliOperator(a_op_spin(a, True, num_spin_orbs, num_elec)) * PauliOperator(
                                        a_op_spin(b, True, num_spin_orbs, num_elec)
                                    )
                                    operator = operator * PauliOperator(a_op_spin(c, True, num_spin_orbs, num_elec))
                                    operator = operator * PauliOperator(a_op_spin(d, True, num_spin_orbs, num_elec))
                                    operator = operator * PauliOperator(a_op_spin(l, False, num_spin_orbs, num_elec))
                                    operator = operator * PauliOperator(a_op_spin(k, False, num_spin_orbs, num_elec))
                                    operator = operator * PauliOperator(a_op_spin(j, False, num_spin_orbs, num_elec))
                                    operator = operator* PauliOperator(a_op_spin(i, False, num_spin_orbs, num_elec))
                                    yield theta_idx, a, i, b, j, c, k, d, l, operator


def construct_UCC_U(
    num_spin_orbs: int,
    num_elec: int,
    theta: list[float],
    theta_picker: ThetaPicker,
    excitations: str,
    allowed_states: np.ndarray = None,
    use_csr: int = 8,
) -> np.ndarray:
    counter = 0
    t = PauliOperator({"I"*num_spin_orbs: 0})
    if "s" in excitations:
        for (_, _, _, op) in theta_picker.get_T1_generator(num_spin_orbs, num_elec):
            if theta[counter] != 0.0:
                t += theta[counter] * op
            counter += 1

    if "d" in excitations:
        for (_, _, _, _, _, op) in theta_picker.get_T2_generator(num_spin_orbs, num_elec):
            if theta[counter] != 0.0:
                t += theta[counter] * op
            counter += 1

    if "t" in excitations:
        for (_, _, _, _, _, _, _, op) in theta_picker.get_T3_generator(num_spin_orbs, num_elec):
            if theta[counter] != 0.0:
                t += theta[counter] * op
            counter += 1

    if "q" in excitations:
        for (_, _, _, _, _, _, _, _, _, op) in theta_picker.get_T4_generator(num_spin_orbs, num_elec):
            if theta[counter] != 0.0:
                t += theta[counter] * op
            counter += 1
    assert counter == len(theta)

    T = (t - t.dagger).matrix_form(use_csr=use_csr, is_real=True)
    if num_spin_orbs >= use_csr:
        if allowed_states is not None:
            T = T[allowed_states, :]
            T = T[:, allowed_states]
        A = scipy.sparse.linalg.expm(T)
    else:
        if allowed_states is not None:
            T = T[allowed_states, :]
            T = T[:, allowed_states]
        A = scipy.linalg.expm(T)
    return A
