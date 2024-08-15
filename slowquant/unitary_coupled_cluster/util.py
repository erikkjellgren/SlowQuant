from collections.abc import Generator, Sequence
from typing import Any

import numpy as np
import scipy.linalg


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
    active_occ_spin_idx: Sequence[int],
    active_unocc_spin_idx: Sequence[int],
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
    active_occ_spin_idx: Sequence[int],
    active_unocc_spin_idx: Sequence[int],
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
    active_occ_spin_idx: Sequence[int],
    active_unocc_spin_idx: Sequence[int],
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
    active_occ_spin_idx: Sequence[int],
    active_unocc_spin_idx: Sequence[int],
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
    active_occ_spin_idx: Sequence[int],
    active_unocc_spin_idx: Sequence[int],
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
    active_occ_spin_idx: Sequence[int],
    active_unocc_spin_idx: Sequence[int],
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


class UccStructure:
    def __init__(self) -> None:
        """Intialize the unitary coupled cluster ansatz structure."""
        self.excitation_indicies: list[tuple[int, ...]] = []
        self.excitation_operator_type: list[str] = []
        self.n_params = 0

    def add_sa_singles(self, active_occ_idx: Sequence[int], active_unocc_idx: Sequence[int]) -> None:
        """Add spin-adapted singles.

        Args:
            active_occ_idx: Active strongly occupied spatial orbital indices.
            active_unocc_idx: Active weakly occupied spatial orbital indices.
        """
        for a, i, _ in iterate_t1_sa(active_occ_idx, active_unocc_idx):
            self.excitation_indicies.append((i, a))
            self.excitation_operator_type.append("sa_single")
            self.n_params += 1

    def add_sa_doubles(self, active_occ_idx: Sequence[int], active_unocc_idx: Sequence[int]) -> None:
        """Add spin-adapted doubles.

        Args:
            active_occ_idx: Active strongly occupied spatial orbital indices.
            active_unocc_idx: Active weakly occupied spatial orbital indices.
        """
        for a, i, b, j, _, op_type in iterate_t2_sa(active_occ_idx, active_unocc_idx):
            self.excitation_indicies.append((i, j, a, b))
            if op_type == 1:
                self.excitation_operator_type.append("sa_double_1")
            elif op_type == 2:
                self.excitation_operator_type.append("sa_double_2")
            self.n_params += 1

    def add_triples(self, active_occ_spin_idx: Sequence[int], active_unocc_spin_idx: Sequence[int]) -> None:
        """Add alpha-number and beta-number conserving triples.

        Args:
            active_occ_spin_idx: Active strongly occupied spin orbital indices.
            active_unocc_spin_idx: Active weakly occupied spin orbital indices.
        """
        for a, i, b, j, c, k in iterate_t3(active_occ_spin_idx, active_unocc_spin_idx):
            self.excitation_indicies.append((i, j, k, a, b, c))
            self.excitation_operator_type.append("triple")
            self.n_params += 1

    def add_quadruples(
        self, active_occ_spin_idx: Sequence[int], active_unocc_spin_idx: Sequence[int]
    ) -> None:
        """Add alpha-number and beta-number conserving quadruples.

        Args:
            active_occ_spin_idx: Active strongly occupied spin orbital indices.
            active_unocc_spin_idx: Active weakly occupied spin orbital indices.
        """
        for a, i, b, j, c, k, d, l in iterate_t4(active_occ_spin_idx, active_unocc_spin_idx):
            self.excitation_indicies.append((i, j, k, l, a, b, c, d))
            self.excitation_operator_type.append("quadruple")
            self.n_params += 1

    def add_quintuples(
        self, active_occ_spin_idx: Sequence[int], active_unocc_spin_idx: Sequence[int]
    ) -> None:
        """Add alpha-number and beta-number conserving quintuples.

        Args:
            active_occ_spin_idx: Active strongly occupied spin orbital indices.
            active_unocc_spin_idx: Active weakly occupied spin orbital indices.
        """
        for a, i, b, j, c, k, d, l, e, m in iterate_t5(active_occ_spin_idx, active_unocc_spin_idx):
            self.excitation_indicies.append((i, j, k, l, m, a, b, c, d, e))
            self.excitation_operator_type.append("quintuple")
            self.n_params += 1

    def add_sextuples(self, active_occ_spin_idx: Sequence[int], active_unocc_spin_idx: Sequence[int]) -> None:
        """Add alpha-number and beta-number conserving sextuples.

        Args:
            active_occ_spin_idx: Active strongly occupied spin orbital indices.
            active_unocc_spin_idx: Active weakly occupied spin orbital indices.
        """
        for a, i, b, j, c, k, d, l, e, m, f, n in iterate_t6(active_occ_spin_idx, active_unocc_spin_idx):
            self.excitation_indicies.append((i, j, k, l, m, n, a, b, c, d, e, f))
            self.excitation_operator_type.append("sextuple")
            self.n_params += 1


class UpsStructure:
    def __init__(self) -> None:
        """Intialize the unitary product state ansatz structure."""
        self.excitation_indicies: list[tuple[int, ...]] = []
        self.excitation_operator_type: list[str] = []
        self.n_params = 0

    def create_tups(self, num_active_orbs: int, ansatz_options: dict[str, Any]) -> None:
        """Create tUPS ansatz.

        #. 10.1103/PhysRevResearch.6.023300
        #. 10.1088/1367-2630/ac2cb3 (QNP)

        Ansatz Options:
            * n_layers [int]: Number of layers.
            * do_qnp [bool]: Do QNP tiling. (default: False)
            * skip_last_singles [bool]: Skip last layer of singles operators. (default: False)

        Args:
            num_active_orbs: Number of spatial active orbitals.
            ansatz_options: Ansatz options.

        Returns:
            tUPS ansatz.
        """
        valid_options = ("n_layers", "do_qnp", "skip_last_singles")
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
        if "skip_last_singles" in ansatz_options.keys():
            skip_last_singles = ansatz_options["skip_last_singles"]
        else:
            skip_last_singles = False
        for n in range(n_layers):
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
                if n + 1 == n_layers and skip_last_singles and num_active_orbs == 2:
                    # Special case for two orbital.
                    # Here the layer is only one block, thus,
                    # the last single excitation is earlier than expected.
                    continue
                self.excitation_operator_type.append("tups_single")
                self.excitation_indicies.append((p,))
                self.n_params += 1
            for p in range(1, num_active_orbs - 1, 2):
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
                if n + 1 == n_layers and skip_last_singles:
                    continue
                self.excitation_operator_type.append("tups_single")
                self.excitation_indicies.append((p,))
                self.n_params += 1

    def create_fUCCSD(self, states: list[list[str]], ansatz_options: dict[str, Any]) -> None:
        """Create factorized UCCSD ansatz.

        If used with a state-averaged wave function, the operator pool will be the union of all
        possible singles and doubles from the determinants included in the states in the state-averaged wave function.

        Ansatz Options:
            * None

        Args:
            states: States to create excitation operators with respect to.
            ansatz_options: Ansatz options.

        Returns:
            Factorized UCCSD ansatz.
        """
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
                for i, occ_str in enumerate(det):
                    if occ_str == "1":
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

    def create_kSAfUpCCGSD(self, num_orbs: int, ansatz_options: dict[str, Any]) -> None:
        """Create modified k-UpCCGSD ansatz.

        The ansatz have been modifed to use spin-adapted singet single excitation operators.

        #. 10.1021/acs.jctc.8b01004

        Ansatz Options:
            * n_layers [int]: Number of layers.

        Args:
            num_active_orbs: Number of spatial active orbitals.
            ansatz_options: Ansatz options.

        Returns:
            Modified k-UpCCGSD ansatz.
        """
        valid_options = "n_layers"
        for option in ansatz_options:
            if option not in valid_options:
                raise ValueError(
                    f"Got unknown option for kSAfUpCCGSD, {option}. Valid options are: {valid_options}"
                )
        if "n_layers" not in ansatz_options.keys():
            raise ValueError("kSAfUpCCGSD require the option 'n_layers'")
        n_layers = ansatz_options["n_layers"]
        for _ in range(n_layers):
            for a, i, _ in iterate_t1_sa_generalized(num_orbs):
                self.excitation_operator_type.append("sa_single")
                self.excitation_indicies.append((i, a))
                self.n_params += 1
            for a, i, b, j in iterate_pair_t2_generalized(num_orbs):
                self.excitation_operator_type.append("double")
                self.excitation_indicies.append((i, j, a, b))
                self.n_params += 1
