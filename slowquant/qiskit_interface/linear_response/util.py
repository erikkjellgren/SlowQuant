from collections.abc import Generator, Sequence


def iterate_t1_sa(
    active_occ_idx: Sequence[int],
    active_unocc_idx: Sequence[int],
) -> Generator[tuple[int, int, int, int, float], None, None]:
    """Iterate over T1 spin-adapted operators.

    Args:
        active_occ_idx: Indices of strongly occupied orbitals.
        active_unocc_idx: Indices of weakly occupied orbitals.

    Returns:
        T1 operator iteration.
    """
    for a in active_occ_idx:
        for i in active_unocc_idx:
            yield a, i, 1, 2 ** (-1 / 2), 1


def iterate_t2_sa(
    active_occ_idx: Sequence[int],
    active_unocc_idx: Sequence[int],
) -> Generator[tuple[int, int, int, int, int, int, float], None, None]:
    """Iterate over T2 spin-adapted operators.

    Args:
        active_occ_idx: Indices of strongly occupied orbitals.
        active_unocc_idx: Indices of weakly occupied orbitals.

    Returns:
        T2 operator iteration.
    """
    for idx_a, a in enumerate(active_occ_idx):
        for b in active_occ_idx[idx_a:]:
            for idx_i, i in enumerate(active_unocc_idx):
                for j in active_unocc_idx[idx_i:]:
                    fac: float = 1
                    if i == j:
                        fac *= 2
                    if a == b:
                        fac *= 2
                    fac = 1 / 2 * (fac) ** (-1 / 2)
                    yield a, i, b, j, 2, fac, 1
                    if i == j or a == b:
                        continue
                    fac = 1 / (2 * 3 ** (1 / 2))
                    yield a, i, b, j, 2, fac, -1
