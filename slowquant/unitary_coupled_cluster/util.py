from collections.abc import Generator, Sequence
from typing import Any

import numpy as np


def iterate_t1_sa(
    active_occ_idx: Sequence[int],
    active_unocc_idx: Sequence[int],
) -> Generator[tuple[int, int, float], None, None]:
    """Iterate over T1 spin-adapted operators.

    Args:
        active_occ_idx: Indices of strongly occupied orbitals.
        active_unocc_idx: Indices of weakly occupied orbitals.

    Returns:
        Spin-adapted T1 operator iteration.
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
        Spin-adapted T2 operator iteration.
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
                    if i == j and a == b:
                        yield a, i, b, j, fac, 1
                    elif i == j:
                        yield a, i, b, j, fac, 2
                    elif a == b:
                        yield a, i, b, j, fac, 3
                    else:
                        yield a, i, b, j, fac, 4
                    if i == j or a == b:
                        continue
                    fac = 1 / (2 * 3 ** (1 / 2))
                    yield a, i, b, j, fac, 5


def iterate_t1_sa_generalized(
    num_orbs: int,
) -> Generator[tuple[int, int, float], None, None]:
    """Iterate over T1 spin-adapted operators.

    Args:
        num_orbs: Number of active spatial orbitals.

    Returns:
        Generalized spin-adapted T1 operator iteration.
    """
    for i in range(num_orbs):
        for a in range(i + 1, num_orbs):
            fac = 2 ** (-1 / 2)
            yield a, i, fac


def iterate_t1(
    active_occ_spin_idx: Sequence[int],
    active_unocc_spin_idx: Sequence[int],
    is_spin_conserving: bool = True,
) -> Generator[tuple[int, int], None, None]:
    """Iterate over T1 spin-conserving operators.

    Args:
        active_occ_spin_idx: Spin indices of strongly occupied orbitals.
        active_unocc_spin_idx: Spin indices of weakly occupied orbitals.
        is_spin_conserving: If the operators are spin-conserving.

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
            if (num_alpha != 0 or num_beta != 0) and is_spin_conserving:
                continue
            yield a, i


def iterate_t2(
    active_occ_spin_idx: Sequence[int],
    active_unocc_spin_idx: Sequence[int],
    is_spin_conserving: bool = True,
) -> Generator[tuple[int, int, int, int], None, None]:
    """Iterate over T2 spin-conserving operators.

    Args:
        active_occ_spin_idx: Spin indices of strongly occupied orbitals.
        active_unocc_spin_idx: Spin indices of weakly occupied orbitals.
        is_spin_conserving: If the operators are spin-conserving.

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
                    if (num_alpha != 0 or num_beta != 0) and is_spin_conserving:
                        continue
                    yield a, i, b, j


def iterate_t3(
    active_occ_spin_idx: Sequence[int],
    active_unocc_spin_idx: Sequence[int],
    is_spin_conserving: bool = True,
) -> Generator[tuple[int, int, int, int, int, int], None, None]:
    """Iterate over T3 spin-conserving operators.

    Args:
        active_occ_spin_idx: Spin indices of strongly occupied orbitals.
        active_unocc_spin_idx: Spin indices of weakly occupied orbitals.
        is_spin_conserving: If the operators are spin-conserving.

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
                            if (num_alpha != 0 or num_beta != 0) and is_spin_conserving:
                                continue
                            yield a, i, b, j, c, k


def iterate_t4(
    active_occ_spin_idx: Sequence[int],
    active_unocc_spin_idx: Sequence[int],
    is_spin_conserving: bool = True,
) -> Generator[tuple[int, int, int, int, int, int, int, int], None, None]:
    """Iterate over T4 spin-conserving operators.

    Args:
        active_occ_spin_idx: Spin indices of strongly occupied orbitals.
        active_unocc_spin_idx: Spin indices of weakly occupied orbitals.
        is_spin_conserving: If the operators are spin-conserving.

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
                                    if (num_alpha != 0 or num_beta != 0) and is_spin_conserving:
                                        continue
                                    yield a, i, b, j, c, k, d, l


def iterate_t5(
    active_occ_spin_idx: Sequence[int],
    active_unocc_spin_idx: Sequence[int],
    is_spin_conserving: bool = True,
) -> Generator[tuple[int, int, int, int, int, int, int, int, int, int], None, None]:
    """Iterate over T5 spin-conserving operators.

    Args:
        active_occ_spin_idx: Spin indices of strongly occupied orbitals.
        active_unocc_spin_idx: Spin indices of weakly occupied orbitals.
        is_spin_conserving: If the operators are spin-conserving.

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
                                            if (num_alpha != 0 or num_beta != 0) and is_spin_conserving:
                                                continue
                                            yield a, i, b, j, c, k, d, l, e, m


def iterate_t6(
    active_occ_spin_idx: Sequence[int],
    active_unocc_spin_idx: Sequence[int],
    is_spin_conserving: bool = True,
) -> Generator[tuple[int, int, int, int, int, int, int, int, int, int, int, int], None, None]:
    """Iterate over T6 spin-conserving operators.

    Args:
        active_occ_spin_idx: Spin indices of strongly occupied orbitals.
        active_unocc_spin_idx: Spin indices of weakly occupied orbitals.
        is_spin_conserving: If the operators are spin-conserving.

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
                                                    if (num_alpha != 0 or num_beta != 0) and is_spin_conserving:
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
        Pair T2 operator iteration.
    """
    for i in active_occ_idx:
        for a in active_unocc_idx:
            yield 2 * a, 2 * i, 2 * a + 1, 2 * i + 1


def iterate_pair_t2_generalized(
    num_orbs: int,
) -> Generator[tuple[int, int, int, int], None, None]:
    """Iterate over generalized pair T2 operators.

    Args:
        num_orbs: Number of active spatial orbitals.

    Returns:
        Generlaized pair T2 operator iteration.
    """
    for i in range(num_orbs):
        for a in range(i + 1, num_orbs):
            yield 2 * a, 2 * i, 2 * a + 1, 2 * i + 1


class UccStructure:
    __slots__ = ("excitation_indices", "excitation_operator_type", "n_params")

    def __init__(self) -> None:
        """Intialize the unitary coupled cluster ansatz structure."""
        self.excitation_indices: list[tuple[int, ...]] = []
        self.excitation_operator_type: list[str] = []
        self.n_params = 0

    def add_sa_singles(self, active_occ_idx: Sequence[int], active_unocc_idx: Sequence[int]) -> None:
        """Add spin-adapted singles.

        Args:
            active_occ_idx: Active strongly occupied spatial orbital indices.
            active_unocc_idx: Active weakly occupied spatial orbital indices.
        """
        for a, i, _ in iterate_t1_sa(active_occ_idx, active_unocc_idx):
            self.excitation_indices.append((i, a))
            self.excitation_operator_type.append("sa_single")
            self.n_params += 1

    def add_sa_doubles(self, active_occ_idx: Sequence[int], active_unocc_idx: Sequence[int]) -> None:
        """Add spin-adapted doubles.

        Args:
            active_occ_idx: Active strongly occupied spatial orbital indices.
            active_unocc_idx: Active weakly occupied spatial orbital indices.
        """
        for a, i, b, j, _, op_type in iterate_t2_sa(active_occ_idx, active_unocc_idx):
            self.excitation_indices.append((i, j, a, b))
            if op_type == 1:
                self.excitation_operator_type.append("sa_double_1")
            elif op_type == 2:
                self.excitation_operator_type.append("sa_double_2")
            elif op_type == 3:
                self.excitation_operator_type.append("sa_double_3")
            elif op_type == 4:
                self.excitation_operator_type.append("sa_double_4")
            elif op_type == 5:
                self.excitation_operator_type.append("sa_double_5")
            self.n_params += 1

    def add_triples(self, active_occ_spin_idx: Sequence[int], active_unocc_spin_idx: Sequence[int]) -> None:
        """Add alpha-number and beta-number conserving triples.

        Args:
            active_occ_spin_idx: Active strongly occupied spin orbital indices.
            active_unocc_spin_idx: Active weakly occupied spin orbital indices.
        """
        for a, i, b, j, c, k in iterate_t3(active_occ_spin_idx, active_unocc_spin_idx):
            self.excitation_indices.append((i, j, k, a, b, c))
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
            self.excitation_indices.append((i, j, k, l, a, b, c, d))
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
            self.excitation_indices.append((i, j, k, l, m, a, b, c, d, e))
            self.excitation_operator_type.append("quintuple")
            self.n_params += 1

    def add_sextuples(self, active_occ_spin_idx: Sequence[int], active_unocc_spin_idx: Sequence[int]) -> None:
        """Add alpha-number and beta-number conserving sextuples.

        Args:
            active_occ_spin_idx: Active strongly occupied spin orbital indices.
            active_unocc_spin_idx: Active weakly occupied spin orbital indices.
        """
        for a, i, b, j, c, k, d, l, e, m, f, n in iterate_t6(active_occ_spin_idx, active_unocc_spin_idx):
            self.excitation_indices.append((i, j, k, l, m, n, a, b, c, d, e, f))
            self.excitation_operator_type.append("sextuple")
            self.n_params += 1


class UpsStructure:
    __slots__ = ("excitation_indices", "excitation_operator_type", "grad_param_R", "n_params", "param_names")

    def __init__(self) -> None:
        """Initialize the unitary product state ansatz structure."""
        self.excitation_indices: list[tuple[int, ...]] = []
        self.excitation_operator_type: list[str] = []
        self.n_params: int = 0
        self.grad_param_R: dict[str, int] = {}
        self.param_names: list[str] = []

    def create_tups(self, num_active_orbs: int, ansatz_options: dict[str, Any]) -> None:
        """Create tUPS ansatz.

        #. 10.1103/PhysRevResearch.6.023300 (tUPS)
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
        # Options
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
        # Layer loop
        for n in range(n_layers):
            for p in range(0, num_active_orbs - 1, 2):  # first column of brick-wall
                if not do_qnp:
                    # First single
                    self.excitation_operator_type.append("sa_single")
                    self.excitation_indices.append((p, p + 1))
                    self.grad_param_R[f"p{self.n_params:09d}"] = 4
                    self.param_names.append(f"p{self.n_params:09d}")
                    self.n_params += 1
                # Double
                self.excitation_operator_type.append("double")
                self.excitation_indices.append((2 * p, 2 * p + 1, 2 * p + 2, 2 * p + 3))
                self.grad_param_R[f"p{self.n_params:09d}"] = 2
                self.param_names.append(f"p{self.n_params:09d}")
                self.n_params += 1
                # Second single
                if n + 1 == n_layers and skip_last_singles and num_active_orbs == 2:
                    # Special case for two orbital.
                    # Here the layer is only one block, thus,
                    # the last single excitation is earlier than expected.
                    continue
                self.excitation_operator_type.append("sa_single")
                self.excitation_indices.append((p, p + 1))
                self.grad_param_R[f"p{self.n_params:09d}"] = 4
                self.param_names.append(f"p{self.n_params:09d}")
                self.n_params += 1
            for p in range(1, num_active_orbs - 1, 2):  # second column of brick-wall
                if not do_qnp:
                    # First single
                    self.excitation_operator_type.append("sa_single")
                    self.excitation_indices.append((p, p + 1))
                    self.grad_param_R[f"p{self.n_params:09d}"] = 4
                    self.param_names.append(f"p{self.n_params:09d}")
                    self.n_params += 1
                # Double
                self.excitation_operator_type.append("double")
                self.excitation_indices.append((2 * p, 2 * p + 1, 2 * p + 2, 2 * p + 3))
                self.grad_param_R[f"p{self.n_params:09d}"] = 2
                self.param_names.append(f"p{self.n_params:09d}")
                self.n_params += 1
                # Second single
                if n + 1 == n_layers and skip_last_singles:
                    continue
                self.excitation_operator_type.append("sa_single")
                self.excitation_indices.append((p, p + 1))
                self.grad_param_R[f"p{self.n_params:09d}"] = 4
                self.param_names.append(f"p{self.n_params:09d}")
                self.n_params += 1

    def create_fUCC(self, num_orbs: int, num_elec: int, ansatz_options: dict[str, Any]) -> None:
        """Create factorized UCC ansatz.

        #. 10.1021/acs.jctc.8b01004 (k-UpCCGSD)

        Ansatz Options:
            * n_layers [int]: Number of layers.
            * S [bool]: Add single excitations.
            * SAS [bool]: Add spin-adapted single excitations.
            * SAGS [bool]: Add generalized spin-adapted single excitations.
            * D [bool]: Add double excitations.
            * pD [bool]: Add pair double excitations.
            * GpD [bool]: Add generalized pair double excitations.

        Args:
            num_orbs: Number of active spatial orbitals.
            num_elec: Number of active electrons.
            ansatz_options: Ansatz options.

        Returns:
            Factorized UCC ansatz.
        """
        # Options
        valid_options = ("n_layers", "S", "D", "SAGS", "pD", "GpD", "SAS", "SAD")
        for option in ansatz_options:
            if option not in valid_options:
                raise ValueError(f"Got unknown option for fUCC, {option}. Valid options are: {valid_options}")
        if "n_layers" not in ansatz_options.keys():
            raise ValueError("fUCC require the option 'n_layers'")
        do_S = False
        do_SAS = False
        do_SAGS = False
        do_D = False
        do_pD = False
        do_GpD = False
        do_SAD = False
        if "S" in ansatz_options.keys():
            if ansatz_options["S"]:
                do_S = True
        if "SAS" in ansatz_options.keys():
            if ansatz_options["SAS"]:
                do_SAS = True
        if "SAGS" in ansatz_options.keys():
            if ansatz_options["SAGS"]:
                do_SAGS = True
        if "D" in ansatz_options.keys():
            if ansatz_options["D"]:
                do_D = True
        if "pD" in ansatz_options.keys():
            if ansatz_options["pD"]:
                do_pD = True
        if "GpD" in ansatz_options.keys():
            if ansatz_options["GpD"]:
                do_GpD = True
        if "SAD" in ansatz_options.keys():
            if ansatz_options["SAD"]:
                do_SAD = True
        if True not in (do_S, do_SAS, do_SAGS, do_D, do_pD, do_GpD, do_SAD):
            raise ValueError("fUCC requires some excitations got none.")
        n_layers = ansatz_options["n_layers"]
        num_spin_orbs = 2 * num_orbs
        occ_spin = []
        unocc_spin = []
        occ = []
        unocc = []
        idx = 0
        for _ in range(np.sum(num_elec)):
            occ_spin.append(idx)
            if idx % 2 == 0:
                occ.append(idx // 2)
            idx += 1
        for _ in range(num_spin_orbs - np.sum(num_elec)):
            unocc_spin.append(idx)
            if idx % 2 == 0:
                unocc.append(idx // 2)
            idx += 1
        # Layer loop
        for _ in range(n_layers):
            if do_S:
                for a, i in iterate_t1(occ_spin, unocc_spin):
                    self.excitation_operator_type.append("single")
                    self.excitation_indices.append((i, a))
                    self.grad_param_R[f"p{self.n_params:09d}"] = 2
                    self.param_names.append(f"p{self.n_params:09d}")
                    self.n_params += 1
            if do_SAS:
                for a, i, _ in iterate_t1_sa(occ, unocc):
                    self.excitation_operator_type.append("sa_single")
                    self.excitation_indices.append((i, a))
                    self.grad_param_R[f"p{self.n_params:09d}"] = 4
                    self.param_names.append(f"p{self.n_params:09d}")
                    self.n_params += 1
            if do_SAGS:
                for a, i, _ in iterate_t1_sa_generalized(num_orbs):
                    self.excitation_operator_type.append("sa_single")
                    self.excitation_indices.append((i, a))
                    self.grad_param_R[f"p{self.n_params:09d}"] = 4
                    self.param_names.append(f"p{self.n_params:09d}")
                    self.n_params += 1
            if do_D:
                for a, i, b, j in iterate_t2(occ_spin, unocc_spin):
                    self.excitation_operator_type.append("double")
                    self.excitation_indices.append((i, j, a, b))
                    self.grad_param_R[f"p{self.n_params:09d}"] = 2
                    self.param_names.append(f"p{self.n_params:09d}")
                    self.n_params += 1
            if do_pD:
                for a, i, b, j in iterate_pair_t2(occ, unocc):
                    self.excitation_operator_type.append("double")
                    self.excitation_indices.append((i, j, a, b))
                    self.grad_param_R[f"p{self.n_params:09d}"] = 2
                    self.param_names.append(f"p{self.n_params:09d}")
                    self.n_params += 1
            if do_GpD:
                for a, i, b, j in iterate_pair_t2_generalized(num_orbs):
                    self.excitation_operator_type.append("double")
                    self.excitation_indices.append((i, j, a, b))
                    self.grad_param_R[f"p{self.n_params:09d}"] = 2
                    self.param_names.append(f"p{self.n_params:09d}")
                    self.n_params += 1
            if do_SAD:
                for a, i, b, j, _, op_case in iterate_t2_sa(occ, unocc):
                    self.excitation_operator_type.append(f"sa_double_{op_case}")
                    self.excitation_indices.append((i, j, a, b))
                    # Rotosolve not implemented for SA doubles
                    # self.grad_param_R[f"p{self.n_params:09d}"] = None
                    self.param_names.append(f"p{self.n_params:09d}")
                    self.n_params += 1

    def create_SDSfUCC(self, num_orbs: int, num_elec: int, ansatz_options: dict[str, Any]) -> None:
        r"""Create SDS ordered factorized UCC.

        The operator ordering of this implementation is,

        .. math::
            \boldsymbol{U}\left|\text{CSF}\right> = \prod_{ijab}\exp\left(\theta_{jb}\left(\hat{T}_{jb}-\hat{T}_{jb}^\dagger\right)\right)
            \exp\left(\theta_{ijab}\left(\hat{T}_{ijab}-\hat{T}_{ijab}^\dagger\right)\right)
            \exp\left(\theta_{ia}\left(\hat{T}_{ia}-\hat{T}_{ia}^\dagger\right)\right)\left|\text{CSF}\right>

        #. 10.1063/1.5133059, Eq. 25, Eq. 35 (SDS)
        #. 10.1021/acs.jctc.8b01004 (k-UpCCGSD)

        Ansatz Options:
            * n_layers [int]: Number of layers.
            * D [bool]: Add double excitations.
            * pD [bool]: Add pair double excitations.
            * GpD [bool]: Add generalized pair double excitations.

        Args:
            num_orbs: Number of active spatial orbitals.
            num_elec: Number of active electrons.
            ansatz_options: Ansatz options.

        Returns:
            SDS ordered fUCC ansatz.
        """
        # Options
        valid_options = ("n_layers", "D", "pD", "GpD")
        for option in ansatz_options:
            if option not in valid_options:
                raise ValueError(
                    f"Got unknown option for SDSfUCC, {option}. Valid options are: {valid_options}"
                )
        if "n_layers" not in ansatz_options.keys():
            raise ValueError("SDSfUCC require the option 'n_layers'")
        do_D = False
        do_pD = False
        do_GpD = False
        if "D" in ansatz_options.keys():
            if ansatz_options["D"]:
                do_D = True
        if "pD" in ansatz_options.keys():
            if ansatz_options["pD"]:
                do_pD = True
        if "GpD" in ansatz_options.keys():
            if ansatz_options["GpD"]:
                do_GpD = True
        if True not in (do_D, do_pD, do_GpD):
            raise ValueError("SDSfUCC requires some excitations got none.")
        n_layers = ansatz_options["n_layers"]
        num_spin_orbs = 2 * num_orbs
        occ = []
        unocc = []
        idx = 0
        for _ in range(np.sum(num_elec)):
            occ.append(idx)
            idx += 1
        for _ in range(num_spin_orbs - np.sum(num_elec)):
            unocc.append(idx)
            idx += 1
        # Layer loop
        for _ in range(n_layers):
            # Kind of D excitation determines indices for complete SDS block
            if do_D:
                for a, i, b, j in iterate_t2(occ, unocc):
                    if i % 2 == a % 2:
                        self.excitation_indices.append((i, a))
                    else:
                        self.excitation_indices.append((i, b))
                    self.excitation_operator_type.append("single")
                    self.grad_param_R[f"p{self.n_params:09d}"] = 2
                    self.param_names.append(f"p{self.n_params:09d}")
                    self.n_params += 1
                    self.excitation_operator_type.append("double")
                    self.excitation_indices.append((i, j, a, b))
                    self.grad_param_R[f"p{self.n_params:09d}"] = 2
                    self.param_names.append(f"p{self.n_params:09d}")
                    self.n_params += 1
                    if i % 2 == a % 2:
                        self.excitation_indices.append((j, b))
                    else:
                        self.excitation_indices.append((j, a))
                    self.excitation_operator_type.append("single")
                    self.grad_param_R[f"p{self.n_params:09d}"] = 2
                    self.param_names.append(f"p{self.n_params:09d}")
                    self.n_params += 1
            if do_pD:
                for a, i, b, j in iterate_pair_t2(occ, unocc):
                    self.excitation_operator_type.append("sa_single")
                    self.excitation_indices.append((i // 2, a // 2))
                    self.grad_param_R[f"p{self.n_params:09d}"] = 4
                    self.param_names.append(f"p{self.n_params:09d}")
                    self.n_params += 1
                    self.excitation_operator_type.append("double")
                    self.excitation_indices.append((i, j, a, b))
                    self.grad_param_R[f"p{self.n_params:09d}"] = 2
                    self.param_names.append(f"p{self.n_params:09d}")
                    self.n_params += 1
                    self.excitation_operator_type.append("sa_single")
                    self.excitation_indices.append((i // 2, a // 2))
                    self.grad_param_R[f"p{self.n_params:09d}"] = 4
                    self.param_names.append(f"p{self.n_params:09d}")
                    self.n_params += 1
            if do_GpD:
                for a, i, b, j in iterate_pair_t2_generalized(num_orbs):
                    self.excitation_operator_type.append("sa_single")
                    self.excitation_indices.append((i // 2, a // 2))
                    self.grad_param_R[f"p{self.n_params:09d}"] = 4
                    self.param_names.append(f"p{self.n_params:09d}")
                    self.n_params += 1
                    self.excitation_operator_type.append("double")
                    self.excitation_indices.append((i, j, a, b))
                    self.grad_param_R[f"p{self.n_params:09d}"] = 2
                    self.param_names.append(f"p{self.n_params:09d}")
                    self.n_params += 1
                    self.excitation_operator_type.append("sa_single")
                    self.excitation_indices.append((i // 2, a // 2))
                    self.grad_param_R[f"p{self.n_params:09d}"] = 4
                    self.param_names.append(f"p{self.n_params:09d}")
                    self.n_params += 1
