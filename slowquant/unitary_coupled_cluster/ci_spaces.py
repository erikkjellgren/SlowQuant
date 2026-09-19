import itertools
from collections.abc import Generator, Sequence

import numba as nb
import numba.typed as nbt
import numpy as np


@nb.jit(nopython=True, inline="always")
def bitcount(x: int) -> int:
    """Count number of ones in binary representation of an integer.

    Implementaion of Brian Kernighan algorithm,
    https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan

    Args:
        x: Integer.

    Returns:
        Number of ones in the binary.
    """
    b = 0
    while x > 0:
        x &= x - 1
        b += 1
    return b


class CI_Info:
    __slots__ = (
        "_det2idx",
        "_idx2det",
        "alpha_str2idx",
        "alpha_str2idx_nb",
        "beta_str2idx",
        "beta_str2idx_nb",
        "block_layouts",
        "idx2alpha_str",
        "idx2beta_str",
        "num_active_elec_alpha",
        "num_active_elec_beta",
        "num_active_orbs",
        "num_alpha_strings",
        "num_beta_strings",
        "num_dets",
        "num_inactive_orbs",
        "num_virtual_orbs",
        "rotation_layouts",
        "space_extension_offset",
        "spin_arena",
        "spin_arena_length",
        "spin_arena_packed",
        "spin_block_layouts",
        "spin_op_cache",
        "string_rotation_layouts",
    )

    def __init__(
        self,
        num_inactive_orbs: int,
        num_active_orbs: int,
        num_virtual_orbs: int,
        num_active_elec_alpha: int,
        num_active_elec_beta: int,
        idx2det: np.ndarray | None = None,
        det2idx: dict[int, int] | None = None,
        alpha_str2idx: dict[int, int] | None = None,
        beta_str2idx: dict[int, int] | None = None,
    ) -> None:
        r"""Initialize configuration expansion information object.

        The spin-orbital ordering is alpha/beta-blocked, so a determinant integer splits into an
        alpha string in the high half and a beta string in the low half, see spin_ordering.

        When the expansion is a product of an alpha and a beta string space, which is the case for
        get_indexing but not for get_indexing_extended, the per-spin index maps relate the two,

        .. math::
            I = I_\alpha N_\beta + I_\beta

        The product structure is what spin_factorized_algebra is built on. For an expansion that
        is not a spin product the maps are empty, num_alpha_strings and num_beta_strings are zero,
        and is_spin_product is False, which routes the algebra back to the general kernels.

        Args:
            num_inactive_orbs: Number of inactive spatial orbitals.
            num_active_orbs: Number of active spatial orbitals.
            num_virtual_orbs: Number of virtual orbitals.
            num_active_elec_alpha: Number of active alpha electrons.
            num_active_elec_beta: Number of active beta electrons.
            idx2det: Index to determinant mapping, or None to derive it from the per-spin maps.
            det2idx: Determinant to index mapping, or None to derive it from the per-spin maps.
            alpha_str2idx: Alpha string to alpha index mapping, if the space is a spin product.
            beta_str2idx: Beta string to beta index mapping, if the space is a spin product.
        """
        self.num_inactive_orbs = num_inactive_orbs
        self.num_active_orbs = num_active_orbs
        self.num_virtual_orbs = num_virtual_orbs
        self.num_active_elec_alpha = num_active_elec_alpha
        self.num_active_elec_beta = num_active_elec_beta
        self.space_extension_offset = 0
        self.alpha_str2idx = {} if alpha_str2idx is None else alpha_str2idx
        self.beta_str2idx = {} if beta_str2idx is None else beta_str2idx
        self.num_alpha_strings = len(self.alpha_str2idx)
        self.num_beta_strings = len(self.beta_str2idx)
        # Over a spin product both determinant maps are implied by the per-spin ones, and they
        # are the two largest structures in the program: one entry per determinant each, against
        # one per spin string for the maps that replace them. So they are only materialized if
        # something actually asks, which over a spin product nothing on the fast path does.
        self._idx2det = idx2det
        self._det2idx: dict[int, int] | None = None
        if det2idx is not None:
            # Unfortunately, Numba needs a little bit of typing help.
            nb_dict = nbt.Dict.empty(key_type=nb.int64, value_type=nb.int64)
            for k, v in det2idx.items():
                nb_dict[k] = v
            self._det2idx = nb_dict
        if idx2det is not None:
            self.num_dets = len(idx2det)
        else:
            self.num_dets = self.num_alpha_strings * self.num_beta_strings
        # Array form of the per-spin maps, which is what the Numba kernels of the
        # spin-factorized algebra can consume. Empty when the space is not a spin product.
        self.idx2alpha_str = np.zeros(self.num_alpha_strings, dtype=int)
        for spin_str, spin_idx in self.alpha_str2idx.items():
            self.idx2alpha_str[spin_idx] = spin_str
        self.idx2beta_str = np.zeros(self.num_beta_strings, dtype=int)
        for spin_str, spin_idx in self.beta_str2idx.items():
            self.idx2beta_str[spin_idx] = spin_str
        # The same maps again for the Numba kernels, which cannot read a Python dict.
        self.alpha_str2idx_nb = nbt.Dict.empty(key_type=nb.int64, value_type=nb.int64)
        for spin_str, spin_idx in self.alpha_str2idx.items():
            self.alpha_str2idx_nb[spin_str] = spin_idx
        self.beta_str2idx_nb = nbt.Dict.empty(key_type=nb.int64, value_type=nb.int64)
        for spin_str, spin_idx in self.beta_str2idx.items():
            self.beta_str2idx_nb[spin_str] = spin_idx
        # Excitation maps of the spin sub-strings seen so far, one arena per spin, laid out back
        # to back. A map says which spin string each one goes to and with what phase, and depends
        # only on the CI space, so it is built once and reused by every later operator.
        # spin_op_cache gives a sub-string its slice of the arena, spin_arena_length is where the
        # next one starts, and spin_arena_packed is the concatenated form, rebuilt only when a
        # new sub-string is added.
        self.spin_op_cache: dict[tuple[bool, tuple[int, ...], tuple[int, ...]], tuple[int, int]] = {}
        self.spin_arena: dict[bool, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
            True: [],
            False: [],
        }
        self.spin_arena_length: dict[bool, int] = {True: 0, False: 0}
        self.spin_arena_packed: dict[bool, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {
            True: None,
            False: None,
        }
        # Determinant pairs rotated by each ansatz generator, see
        # operator_state_algebra.build_rotation_layout. Keyed by excitation type and indices,
        # holding None for a generator that is not a pairing. Depends only on the CI space, so
        # it survives every change of the ansatz parameters.
        self.rotation_layouts: dict[
            tuple[str, tuple[int, ...]], tuple[np.ndarray, np.ndarray, np.ndarray] | None
        ] = {}
        # The same for generators that connect more than two determinants at a time, see
        # operator_state_algebra.build_generator_blocks.
        self.block_layouts: dict[tuple[str, tuple[int, ...]], tuple[np.ndarray, ...] | None] = {}
        # The same rotations again, but held as pairs of spin strings rather than pairs of
        # determinants, which is what a spin product allows and what keeps the ansatz's memory
        # independent of the size of the CI space. See
        # spin_factorized_algebra.build_string_rotation_layout.
        # Generators that are not pairings, the spin-adapted doubles above all, block over pairs
        # of spin-string cells instead. See spin_factorized_algebra.build_spin_block_layout.
        self.spin_block_layouts: dict[tuple[str, tuple[int, ...]], tuple | None] = {}
        self.string_rotation_layouts: dict[
            tuple[str, tuple[int, ...]], tuple[int, tuple[np.ndarray, ...], tuple[np.ndarray, ...]] | None
        ] = {}

    @property
    def idx2det(self) -> np.ndarray:
        r"""Determinant of every index, built on first use over a spin product.

        .. math::
            \text{det}\left(I_\alpha N_\beta + I_\beta\right)
                = \left(\text{str}_\alpha \ll N\right) | \text{str}_\beta

        Costs eight bytes per determinant, so over a spin product it is left unbuilt until
        something asks for it, which only the general kernels do.

        Returns:
            Index to determinant mapping.
        """
        if self._idx2det is None:
            self._idx2det = (
                (self.idx2alpha_str[:, np.newaxis] << self.num_active_orbs) | self.idx2beta_str[np.newaxis, :]
            ).ravel()
        return self._idx2det

    @property
    def det2idx(self) -> dict[int, int]:
        """Index of every determinant, built on first use over a spin product.

        This is the single largest structure in the program, tens of bytes per determinant in a
        Numba typed dictionary, and over a spin product it is redundant with the two per-spin
        maps. Use index_of_determinant for a single lookup rather than forcing it to be built.

        Returns:
            Determinant to index mapping.
        """
        if self._det2idx is None:
            nb_dict = nbt.Dict.empty(key_type=nb.int64, value_type=nb.int64)
            for idx, det in enumerate(self.idx2det):
                nb_dict[int(det)] = idx
            self._det2idx = nb_dict
        return self._det2idx

    def index_of_determinant(self, det: int) -> int:
        r"""Look one determinant up without building the whole determinant map.

        Over a spin product the index follows from the two per-spin maps,

        .. math::
            I = I_\alpha N_\beta + I_\beta

        so a handful of lookups, as the wave function classes do to place their reference
        determinant, need not pay for a map over every determinant in the space.

        Args:
            det: Determinant as an integer.

        Returns:
            Index of the determinant.
        """
        if self._det2idx is None and self.is_spin_product:
            alpha_str = det >> self.num_active_orbs
            beta_str = det & ((1 << self.num_active_orbs) - 1)
            return self.alpha_str2idx[alpha_str] * self.num_beta_strings + self.beta_str2idx[beta_str]
        return self.det2idx[det]

    @property
    def is_spin_product(self) -> bool:
        r"""Check if the determinant expansion is a product of an alpha and a beta string space.

        True for get_indexing and False for get_indexing_extended. Only a spin product can be
        acted on with the spin-factorized algebra, and only for a spin product does

        .. math::
            I = I_\alpha N_\beta + I_\beta

        hold.

        Returns:
            True if the expansion is a spin product.
        """
        return self.num_alpha_strings != 0


def generate_spin_strings(num_orbs: int, num_elec: int) -> Generator[list[int], None, None]:
    """Generate all unique length-n lists of 0s and 1s with exactly k 1s.

    Args:
        num_orbs: Number of spatial orbitals.
        num_elec: Number of electrons.

    Returns:
        Determinants with k electrons in n orbitals.
    """
    if num_elec < 0:
        # Nothing to iterate
        return
    for indices in itertools.combinations(range(num_orbs), num_elec):
        string = [0] * num_orbs
        for idx in indices:
            string[idx] = 1
        yield string


def spin_string_to_int(occupation: Sequence[int]) -> int:
    """Convert an occupation list of a single spin to an integer.

    The first orbital is the most significant bit.

    Args:
        occupation: Occupation of each spatial orbital for one spin.

    Returns:
        Spin string as an integer.
    """
    spin_string = 0
    for occ in occupation:
        spin_string = (spin_string << 1) | occ
    return spin_string


def det_from_spin_strings(alpha_string: int, beta_string: int, num_orbs: int) -> int:
    r"""Combine an alpha and a beta spin string into a determinant.

    With alpha/beta-blocked ordering the alpha string occupies the high half of the determinant
    integer and the beta string the low half,

    .. math::
        \left|\text{det}\right> = \left|\alpha\right>\otimes\left|\beta\right>

    Args:
        alpha_string: Alpha spin string as an integer.
        beta_string: Beta spin string as an integer.
        num_orbs: Number of spatial orbitals in the space.

    Returns:
        Determinant as an integer.
    """
    return (alpha_string << num_orbs) | beta_string


def get_indexing(
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
) -> CI_Info:
    """Get relation between index and determinant.

    Args:
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_beta: Number of active beta electrons.

    Returns:
        CI_Info object.
    """
    # Only the per-spin maps are built. A determinant is an alpha string in the high half and a
    # beta string in the low half, and the index of one is idx_alpha*num_beta_strings +
    # idx_beta, so the maps over every determinant follow from these two and are left to
    # CI_Info to derive if anything ever asks. Alpha is still the outer space and beta the
    # inner one; that convention is what makes the index formula hold, so do not swap them.
    alpha_str2idx = {
        spin_string_to_int(occupation): idx_alpha
        for idx_alpha, occupation in enumerate(generate_spin_strings(num_active_orbs, num_active_elec_alpha))
    }
    beta_str2idx = {
        spin_string_to_int(occupation): idx_beta
        for idx_beta, occupation in enumerate(generate_spin_strings(num_active_orbs, num_active_elec_beta))
    }
    return CI_Info(
        num_inactive_orbs,
        num_active_orbs,
        num_virtual_orbs,
        num_active_elec_alpha,
        num_active_elec_beta,
        alpha_str2idx=alpha_str2idx,
        beta_str2idx=beta_str2idx,
    )


def get_indexing_extended(
    num_inactive_orbs: int,
    num_active_orbs: int,
    num_virtual_orbs: int,
    num_active_elec_alpha: int,
    num_active_elec_beta: int,
    order: int,
) -> CI_Info:
    r"""Get indexing between index and determinant, extended to include complete active-space on-top of a full space singles or full space singles and doubles.

    Needed for full-space operators (e.g. orbital rotations between spaces) that act on the reference before the unitary ansatz is applied (e.g. $Uq\left|CSF\right>$) .
    This leads to a change in particle number in the active space and precludes the standard indexing formalism that is based on operator folding into the active space.
    Now the determinant basis spans a larger portion of the Fock space made of the complete active space and singles and doubles in virtual and occupied space.

    Args:
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals.
        num_active_elec_alpha: Number of active alpha electrons.
        num_active_elec_beta: Number of active beta electrons.
        order: Excitation order the space will be extended with.

    Returns:
        CI_Info object.
    """
    if order > 2:
        raise ValueError("Excitation order needs to be <= 2")
    # The extended space spans all orbitals, so determinants are built over the full space.
    num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs
    # Obtain additional determinants from single excitations that break active space particle symmetry
    inactive_singles = []
    virtual_singles = []
    for inactive, virtual in generate_singles(num_inactive_orbs, num_virtual_orbs):
        inactive_singles.append(inactive)
        virtual_singles.append(virtual)
    # Obtain additional determinants from double excitations that break active space particle symmetry
    inactive_doubles = []
    virtual_doubles = []
    if order == 2:
        for inactive, virtual in generate_doubles(num_inactive_orbs, num_virtual_orbs):
            inactive_doubles.append(inactive)
            virtual_doubles.append(virtual)
    idx = 0
    idx2det = []
    det2idx = {}
    # Generate 0th space
    # Particle and spin conserving determinants in active space. No excitation in occ and virtual orbs.
    for alpha_string in generate_spin_strings(num_active_orbs, num_active_elec_alpha):
        for beta_string in generate_spin_strings(num_active_orbs, num_active_elec_beta):
            det = det_from_spin_strings(
                spin_string_to_int([1] * num_inactive_orbs + alpha_string + [0] * num_virtual_orbs),
                spin_string_to_int([1] * num_inactive_orbs + beta_string + [0] * num_virtual_orbs),
                num_orbs,
            )
            if det in idx2det:
                continue
            idx2det.append(det)
            det2idx[det] = idx
            idx += 1
    # Generate 1,2 exc alpha space
    # Loop over occ and virtual particle number breaking determinants (single and double exc) for alpha electrons
    # Beta electrons stay particle number conserving in active space
    for alpha_inactive, alpha_virtual in zip(
        inactive_singles + inactive_doubles, virtual_singles + virtual_doubles
    ):  # singles/doubles inactive and virtual determinants in alpha
        active_alpha_elec = int(
            num_active_elec_alpha - np.sum(alpha_virtual) + num_inactive_orbs - np.sum(alpha_inactive)
        )
        for alpha_string in generate_spin_strings(num_active_orbs, active_alpha_elec):
            for beta_string in generate_spin_strings(num_active_orbs, num_active_elec_beta):
                det = det_from_spin_strings(
                    spin_string_to_int(alpha_inactive + alpha_string + alpha_virtual),
                    spin_string_to_int([1] * num_inactive_orbs + beta_string + [0] * num_virtual_orbs),
                    num_orbs,
                )
                if det in idx2det:
                    continue
                idx2det.append(det)
                det2idx[det] = idx
                idx += 1
    # Generate 1,2 exc beta space
    # Loop over occ and virtual particle number breaking determinants (single and double exc) for beta orbs
    # Alpha orbs stay particle number conserving in active space
    for beta_inactive, beta_virtual in zip(
        inactive_singles + inactive_doubles, virtual_singles + virtual_doubles
    ):  # singles/doubles inactive and virtual determinants in beta
        active_beta_elec = int(
            num_active_elec_beta - np.sum(beta_virtual) + num_inactive_orbs - np.sum(beta_inactive)
        )
        for alpha_string in generate_spin_strings(num_active_orbs, num_active_elec_alpha):
            for beta_string in generate_spin_strings(num_active_orbs, active_beta_elec):
                det = det_from_spin_strings(
                    spin_string_to_int([1] * num_inactive_orbs + alpha_string + [0] * num_virtual_orbs),
                    spin_string_to_int(beta_inactive + beta_string + beta_virtual),
                    num_orbs,
                )
                if det in idx2det:
                    continue
                idx2det.append(det)
                det2idx[det] = idx
                idx += 1
    # Generate 1 exc alpha 1 exc beta space
    # Loop over occ and virtual particle number breaking determinants (single excitation) for alpha and beta orbs
    if order == 2:
        for alpha_inactive, alpha_virtual in zip(inactive_singles, virtual_singles):
            active_alpha_elec = int(
                num_active_elec_alpha - np.sum(alpha_virtual) + num_inactive_orbs - np.sum(alpha_inactive)
            )  # singles inactive and virtual determinants in alpha
            for beta_inactive, beta_virtual in zip(inactive_singles, virtual_singles):
                active_beta_elec = int(
                    num_active_elec_beta - np.sum(beta_virtual) + num_inactive_orbs - np.sum(beta_inactive)
                )  # singles inactive and virtual determinants in beta
                for alpha_string in generate_spin_strings(num_active_orbs, active_alpha_elec):
                    for beta_string in generate_spin_strings(num_active_orbs, active_beta_elec):
                        det = det_from_spin_strings(
                            spin_string_to_int(alpha_inactive + alpha_string + alpha_virtual),
                            spin_string_to_int(beta_inactive + beta_string + beta_virtual),
                            num_orbs,
                        )
                        if det in idx2det:
                            continue
                        idx2det.append(det)
                        det2idx[det] = idx
                        idx += 1
    ci_info = CI_Info(
        0,
        num_inactive_orbs + num_active_orbs + num_virtual_orbs,
        0,
        num_active_elec_alpha + num_inactive_orbs,
        num_active_elec_beta + num_inactive_orbs,
        np.array(idx2det, dtype=int),
        det2idx,
    )
    ci_info.space_extension_offset = num_inactive_orbs
    return ci_info


def generate_singles(
    num_inactive_orbs: int, num_virtual_orbs: int
) -> Generator[tuple[list[int], list[int]], None, None]:
    """Generate single excited determinant in the inactive and virtual space.

    These are generated via single excitation between all three spaces and thus are only particle conserving in the full space.
    It includes single excitations: inactive -> virtual, inactive -> active (no change in virtual), active -> virtual (no change in occ)
    The reference is also included.

    Args:
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals.

    Returns:
        Single excited determinants.
    """
    inactive = [1] * num_inactive_orbs
    virtual = [0] * num_virtual_orbs
    # loop over excitations out of all inactive orbs
    # add loop index for not changing inactive orb
    for i in range(num_inactive_orbs + 1):
        if i != num_inactive_orbs:  # excite out
            inactive[i] = 0
        # loop over excitations into virtual orbs
        # add loop index for not changing virtual orb
        for j in range(num_virtual_orbs + 1):
            if j != num_virtual_orbs:  # excite in
                virtual[j] = 1
            yield inactive.copy(), virtual.copy()
            if j != num_virtual_orbs:  # reset
                virtual[j] = 0
        if i != num_inactive_orbs:  # reset
            inactive[i] = 1


def generate_doubles(
    num_inactive_orbs: int, num_virtual_orbs: int
) -> Generator[tuple[list[int], list[int]], None, None]:
    """Generate double excited determinant in the inactive and virtual space.

    These are generated via double excitation between all three spaces and thus are only particle conserving in the full space.
    It includes double excitations: inactive -> virtual, inactive -> active (no change in virtual), active -> virtual (no change in occ)
    The reference is also included.

    Args:
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_virtual_orbs: Number of virtual spatial orbitals.

    Returns:
        Double excited determinants.
    """
    inactive = [1] * num_inactive_orbs
    virtual = [0] * num_virtual_orbs
    # loop over excitations out of all inactive orbs
    # add loop index for not changing inactive orb
    for i in range(num_inactive_orbs + 1):
        if i != num_inactive_orbs:
            inactive[i] = 0
        # second orb for excitation out of
        for i2 in range(min(i + 1, num_inactive_orbs), num_inactive_orbs + 1):
            if i2 != num_inactive_orbs:
                inactive[i2] = 0
            # loop over excitations into virtual orbs
            # add loop index for not changing virtual orb
            for j in range(num_virtual_orbs + 1):
                if j != num_virtual_orbs:
                    virtual[j] = 1
                # second orb for excitation into
                for j2 in range(min(j + 1, num_virtual_orbs), num_virtual_orbs + 1):
                    if j2 != num_virtual_orbs:
                        virtual[j2] = 1
                    yield inactive.copy(), virtual.copy()
                    if j2 != num_virtual_orbs:
                        virtual[j2] = 0
                if j != num_virtual_orbs:
                    virtual[j] = 0
            if i2 != num_inactive_orbs:
                inactive[i2] = 1
        if i != num_inactive_orbs:
            inactive[i] = 1
