import numpy as np
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper

from slowquant.qiskit_interface.util import CliqueHead, find_swaps, swap_indices


class Mitigation:
    def __init__(self, postselection: bool, M: str) -> None:
        """Initialize error mitigation class.

        Args:
            postselection: Boolean to do postselection.
            M: Confusion matrix for read-out (and gate) error.
        """
        M_options = ["None", "M", "M0"]
        if M not in M_options:
            raise ValueError("Specified M option does not exist. Choose from ", M_options)
        self.postselection = postselection
        self.M = M
        self._Minv = None  # what about custom M?

    # Stuff to take care of changing mitigation!
    # Custom M problem?

    def is_none(self) -> bool:
        if not self.postselection and self.M == "None":
            return True
        return False

    def apply_mitigation_to_clique(self, clique_head: CliqueHead) -> list[dict[int, float]]:
        """Apply EM"""
        return self.apply_mitigation_to_dist(clique_head.distr, clique_head.head)
        # saver?

    def apply_mitigation_to_dist(self, dist: list[dict[int, float]], pauli: str) -> list[dict[int, float]]:
        """Apply EM"""
        dist_corr = dist.copy()
        # Do M

        # Do postselection

        return dist_corr


# Move them all in the class?
def correct_distribution(dist: dict[int, float], M: np.ndarray) -> dict[int, float]:
    r"""Corrects a quasi-distribution of bitstrings based on a correlation matrix in statevector notation.

    Args:
        dist: Quasi-distribution.
        M: Correlation matrix (inverse).

    Returns:
        Quasi-distribution corrected by correlation matrix.
    """
    C = np.zeros(np.shape(M)[0])
    # Convert bitstring distribution to columnvector of probabilities
    for bitint, prob in dist.items():
        C[bitint] = prob
    # Apply M error mitigation matrix
    C_new = M @ C
    # Convert columnvector of probabilities to bitstring distribution
    for bitint, prob in dist.items():
        dist[bitint] = C_new[bitint]
    return dist


def correct_distribution_with_layout_v2(
    dist: dict[int, float], M: np.ndarray, ref_layout: list[int], new_layout: list[int]
) -> dict[int, float]:
    """Corrects a quasi-distribution of bitstrings based on a correlation matrix in statevector notation.

    Uses layout correction via distribution mapping.

    Args:
        dist: Quasi-distribution.
        M: Correlation matrix (inverse).
        ref_layout: Reference layout of M measurement.
        new_layout: Layout of current to be corrected circuit measurement.

    Returns:
        Quasi-distribution corrected by correlation matrix with corrected layout.
    """
    # Find swaps that map new layout to reference layout
    # Layout indices need to be inverted due to qiskit saving layout indices q0->qN and distribtions qN->q0.
    swaps = find_swaps(new_layout[::-1], ref_layout[::-1])
    num_qubits = len(ref_layout)

    C = np.zeros(np.shape(M)[0])
    # Convert bitstring distribution to columnvector of probabilities
    for bitint, prob in dist.items():
        C[bitint] = prob
    # Forward correction of layout
    for swap in swaps:
        C = C[swap_indices(num_qubits, swap)]
    # Apply M error mitigation matrix
    C_new = M @ C
    # Backward correction of layout
    for swap in swaps[::-1]:
        C_new = C_new[swap_indices(num_qubits, swap)]
    # Convert columnvector of probabilities to bitstring distribution
    for bitint, prob in enumerate(C_new):
        dist[bitint] = prob
    return dist


def correct_distribution_with_layout(
    dist: dict[int, float], M_in: np.ndarray, ref_layout: list[int], new_layout: list[int]
) -> dict[int, float]:
    """Corrects a quasi-distribution of bitstrings based on a correlation matrix in statevector notation.

    Uses layout correction via M mapping.

    Args:
        dist: Quasi-distribution.
        M: Correlation matrix (not inverse).
        ref_layout: Reference layout of M measurement.
        new_layout: Layout of current to be corrected circuit measurement.

    Returns:
        Quasi-distribution corrected by correlation matrix with corrected layout.
    """
    # Find swaps that map new layout to reference layout
    # Layout indices need to be inverted due to qiskit saving layout indices q0->qN and distribtions qN->q0.
    swaps = find_swaps(new_layout[::-1], ref_layout[::-1])
    num_qubits = len(ref_layout)

    # Create new M
    M = M_in.copy()
    for swap in swaps:
        idx = np.array(swap_indices(num_qubits, swap))
        print(idx)
        M = M[idx, :][:, idx]
    M_inv = np.linalg.inv(M)

    C = np.zeros(np.shape(M)[0])
    # Convert bitstring distribution to columnvector of probabilities
    for bitint, prob in dist.items():
        C[bitint] = prob
    # Apply M error mitigation matrix
    C_new = M_inv @ C
    # Convert columnvector of probabilities to bitstring distribution
    for bitint, prob in enumerate(C_new):
        dist[bitint] = prob
    return dist


def postselection(
    dist: dict[int, float],
    mapper: FermionicMapper,
    num_elec: tuple[int, int],
    num_qubits: int,
) -> dict[int, float]:
    r"""Perform post-selection on distribution in computational basis.

    For the Jordan-Wigner mapper the post-selection ensure that,

    .. math::
        \text{sum}\left(\left|\alpha\right>\right) = N_\alpha

    and,

    .. math::
        \text{sum}\left(\left|\beta\right>\right) = N_\beta

    For the Parity mapper it is counted how many times bitstring changes between 0 and 1.
    For the bitstring :math:`\left|01\right>` the counting is done by padding the string before counting.
    I.e.

    .. math::
        \left|01\right> \rightarrow 0\left|01\right>p

    Where :math:`p` is zero for even number of electrons and one for odd number of electrons.
    This counting is done independtly for the :math:`\alpha` part and :math:`\beta` part.

    Args:
        dist: Measured quasi-distribution.
        mapper: Fermionic to qubit mapper.
        num_elec: Number of electrons (alpha, beta).
        num_qubits: Number of qubits.

    Returns:
        Post-selected distribution.
    """
    new_dist = {}
    prob_sum = 0.0
    if isinstance(mapper, JordanWignerMapper):
        for bitint, val in dist.items():
            bitstr = format(bitint, f"0{num_qubits}b")
            num_a = len(bitstr) // 2
            # Remember that in Qiskit notation you read |0101> from right to left.
            bitstr_a = bitstr[num_a:]
            bitstr_b = bitstr[:num_a]
            if bitstr_a.count("1") == num_elec[0] and bitstr_b.count("1") == num_elec[1]:
                new_dist[int(bitstr, 2)] = val
                prob_sum += val
    elif isinstance(mapper, ParityMapper):
        for bitint, val in dist.items():
            bitstr = format(bitint, f"0{num_qubits}b")
            num_a = len(bitstr) // 2
            bitstr_a = bitstr[num_a:]
            bitstr_b = bitstr[:num_a]
            current_parity = "0"
            change_counter = 0
            for bit in bitstr_b:
                if bit != current_parity:
                    current_parity = bit
                    change_counter += 1
            if current_parity == "1" and num_elec[1] % 2 == 0:
                change_counter += 1
                current_parity = "0"
            elif current_parity == "0" and num_elec[1] % 2 == 1:
                change_counter += 1
                current_parity = "1"
            if change_counter != num_elec[0]:
                continue
            change_counter = 0
            for bit in bitstr_a:
                if bit != current_parity:
                    current_parity = bit
                    change_counter += 1
            if current_parity == "1" and (num_elec[0] + num_elec[1]) % 2 == 0:
                change_counter += 1
            elif current_parity == "0" and (num_elec[0] + num_elec[1]) % 2 == 1:
                change_counter += 1
            if change_counter != num_elec[1]:
                continue
            new_dist[int(bitstr, 2)] = val
            prob_sum += val
    else:
        raise ValueError(f"Post-selection only supported for JW and parity mapper got, {type(mapper)}")
    # Renormalize distribution
    for bitint, val in new_dist.items():
        new_dist[bitint] = val / prob_sum
    return new_dist
