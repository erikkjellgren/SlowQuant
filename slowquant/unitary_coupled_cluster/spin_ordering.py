r"""Spin-orbital index ordering conventions.

SlowQuant orders spin orbitals in an :math:`\alpha`/:math:`\beta`-blocked fashion, matching
Qiskit Nature,

.. math::
    \left|0_\alpha 1_\alpha ... N_\alpha 0_\beta 1_\beta ... N_\beta\right>

so the spin orbital of spatial orbital :math:`p` is :math:`p` for :math:`\alpha` and
:math:`p + N` for :math:`\beta`, with :math:`N` the number of spatial orbitals of the space
the index lives in.

Determinants are stored as integers where spin-orbital index :math:`i` sits at bit
:math:`2N-1-i`, i.e. index zero is the most significant bit. With blocked ordering the
:math:`\alpha` string therefore occupies the high half of the integer and the :math:`\beta`
string the low half,

.. math::
    \text{alpha string} = \text{det} \gg N, \qquad
    \text{beta string} = \text{det} \wedge \left(2^N - 1\right)

The interleaved convention,

.. math::
    \left|0_\alpha 0_\beta 1_\alpha 1_\beta ... N_\alpha N_\beta\right>

is retained only as a human-readable input format, for example when specifying reference
determinants. The converters below translate it to the internal blocked convention.

These helpers are used when constructing operators and determinant spaces, not inside the
Numba kernels, so they are kept as plain Python.
"""


def alpha_idx(p: int, num_orbs: int) -> int:
    """Get spin-orbital index of an alpha spin orbital.

    Args:
        p: Spatial orbital index.
        num_orbs: Number of spatial orbitals in the space the index lives in.

    Returns:
        Spin-orbital index.
    """
    if not 0 <= p < num_orbs:
        raise ValueError(f"Spatial orbital index {p} is outside a space of {num_orbs} orbitals.")
    return p


def beta_idx(p: int, num_orbs: int) -> int:
    """Get spin-orbital index of a beta spin orbital.

    Args:
        p: Spatial orbital index.
        num_orbs: Number of spatial orbitals in the space the index lives in.

    Returns:
        Spin-orbital index.
    """
    if not 0 <= p < num_orbs:
        raise ValueError(f"Spatial orbital index {p} is outside a space of {num_orbs} orbitals.")
    return p + num_orbs


def spin_orb_idx(p: int, spin: str, num_orbs: int) -> int:
    """Get spin-orbital index of a spatial orbital with a given spin.

    Args:
        p: Spatial orbital index.
        spin: Alpha or beta spin.
        num_orbs: Number of spatial orbitals in the space the index lives in.

    Returns:
        Spin-orbital index.
    """
    if spin not in ("alpha", "beta"):
        raise ValueError(f'spin must be "alpha" or "beta" got {spin}')
    if spin == "alpha":
        return alpha_idx(p, num_orbs)
    return beta_idx(p, num_orbs)


def spatial_idx(i: int, num_orbs: int) -> int:
    """Get spatial orbital index of a spin orbital.

    Args:
        i: Spin-orbital index.
        num_orbs: Number of spatial orbitals in the space the index lives in.

    Returns:
        Spatial orbital index.
    """
    if not 0 <= i < 2 * num_orbs:
        raise ValueError(f"Spin-orbital index {i} is outside a space of {2 * num_orbs} spin orbitals.")
    return i % num_orbs


def is_alpha(i: int, num_orbs: int) -> bool:
    """Check if a spin orbital is of alpha spin.

    Args:
        i: Spin-orbital index.
        num_orbs: Number of spatial orbitals in the space the index lives in.

    Returns:
        True if the spin orbital is alpha.
    """
    if not 0 <= i < 2 * num_orbs:
        raise ValueError(f"Spin-orbital index {i} is outside a space of {2 * num_orbs} spin orbitals.")
    return i < num_orbs


def interleaved_to_blocked(i: int, num_orbs: int) -> int:
    """Convert an interleaved spin-orbital index to a blocked one.

    Args:
        i: Spin-orbital index in interleaved ordering.
        num_orbs: Number of spatial orbitals in the space the index lives in.

    Returns:
        Spin-orbital index in blocked ordering.
    """
    if not 0 <= i < 2 * num_orbs:
        raise ValueError(f"Spin-orbital index {i} is outside a space of {2 * num_orbs} spin orbitals.")
    if i % 2 == 0:
        return i // 2
    return i // 2 + num_orbs


def blocked_to_interleaved(i: int, num_orbs: int) -> int:
    """Convert a blocked spin-orbital index to an interleaved one.

    Args:
        i: Spin-orbital index in blocked ordering.
        num_orbs: Number of spatial orbitals in the space the index lives in.

    Returns:
        Spin-orbital index in interleaved ordering.
    """
    if not 0 <= i < 2 * num_orbs:
        raise ValueError(f"Spin-orbital index {i} is outside a space of {2 * num_orbs} spin orbitals.")
    if i < num_orbs:
        return 2 * i
    return 2 * (i - num_orbs) + 1


def det_interleaved_to_blocked(det: str) -> str:
    """Convert a determinant string from interleaved to blocked ordering.

    Note that this only reorders the occupations. A determinant is an ordered product of
    creation operators, so re-expressing it in the blocked ordering also carries a phase,
    see get_reordering_sign.

    Args:
        det: Determinant in interleaved ordering.

    Returns:
        Determinant in blocked ordering.
    """
    if len(det) % 2 != 0:
        raise ValueError(f"Determinant must have an even number of spin orbitals, got {len(det)}.")
    return det[0::2] + det[1::2]


def det_blocked_to_interleaved(det: str) -> str:
    """Convert a determinant string from blocked to interleaved ordering.

    Args:
        det: Determinant in blocked ordering.

    Returns:
        Determinant in interleaved ordering.
    """
    if len(det) % 2 != 0:
        raise ValueError(f"Determinant must have an even number of spin orbitals, got {len(det)}.")
    num_orbs = len(det) // 2
    return "".join(a + b for a, b in zip(det[:num_orbs], det[num_orbs:]))


def get_reordering_sign(det: str) -> int:
    r"""Get sign from reordering a determinant from interleaved to blocked ordering.

    A determinant is an ordered product of creation operators, so moving all
    :math:`\alpha` operators in front of all :math:`\beta` operators is a permutation
    whose sign is the parity of the number of occupied
    :math:`\left(\beta,\alpha\right)` inversions, i.e. the number of pairs where an
    occupied :math:`\beta` spin orbital precedes an occupied :math:`\alpha` one.

    Args:
        det: Determinant in interleaved ordering.

    Returns:
        Phase factor from the reordering.
    """
    if len(det) % 2 != 0:
        raise ValueError(f"Determinant must have an even number of spin orbitals, got {len(det)}.")
    sign = 1
    num_alpha = 0
    # Walk from the highest spin-orbital index downwards, counting the alpha operators that
    # every occupied beta operator has to be moved past.
    for i, occ in enumerate(det[::-1]):
        if occ != "1":
            continue
        # In the reversed string the alpha spin orbitals are the odd entries.
        if i % 2 == 1:
            num_alpha += 1
        elif num_alpha % 2 == 1:
            sign *= -1
    return sign
