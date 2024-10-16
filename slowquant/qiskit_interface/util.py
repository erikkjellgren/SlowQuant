import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.transpiler import PassManager
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper


def to_CBS_measurement(op: str, transpiled: None | list[QuantumCircuit] = None) -> QuantumCircuit:
    r"""Convert a Pauli string to Pauli measurement circuit.

    This is achived by the following transformation:

    .. math::
        \begin{align}
        I &\rightarrow I\\
        Z &\rightarrow Z\\
        X &\rightarrow XH\\
        Y &\rightarrow YS^{\dagger}H
        \end{align}

    Args:
        op: Pauli string.
        transpiled: List of transpiled X and Y gate.

    Returns:
        Pauli measuremnt quantum circuit.
    """
    if transpiled is None:
        num_qubits = len(op)
        qc = QuantumCircuit(num_qubits)
        for i, pauli in enumerate(op[::-1]):  # turn order to q0,q1,...,qN
            if pauli == "X":
                qc.append(Pauli("X"), [i])
                qc.h(i)
            elif pauli == "Y":
                qc.append(Pauli("Y"), [i])
                qc.sdg(i)
                qc.h(i)
    else:
        num_qubits = len(op)
        qc = QuantumCircuit(num_qubits)
        for i, pauli in enumerate(op[::-1]):
            if pauli == "X":
                qc.compose(transpiled[0], [i], inplace=True)
            elif pauli == "Y":
                qc.compose(transpiled[1], [i], inplace=True)

    return qc


def get_bitstring_sign(op: str, binary: int) -> int:
    r"""Convert Pauli string and bit-string measurement to expectation value.

    Takes Pauli String and a state in binary form and returns the sign based on the expectation value of the Pauli string with each single qubit state.

    This is achieved by using the following evaluations:

    .. math::
        \begin{align}
        \left<0\left|I\right|0\right> &= 1\\
        \left<1\left|I\right|1\right> &= 1\\
        \left<0\left|Z\right|0\right> &= 1\\
        \left<1\left|Z\right|1\right> &= -1\\
        \left<0\left|HXH\right|0\right> &= 1\\
        \left<1\left|HXH\right|1\right> &= -1\\
        \left<0\left|HSYS^{\dagger}H\right|0\right> &= 1\\
        \left<1\left|HSYS^{\dagger}H\right|1\right> &= -1
        \end{align}

    The total expectation value is then evaulated as:

    .. math::
        E = \prod_i^N\left<b_i\left|P_{i,T}\right|b_i\right>

    With :math:`b_i` being the :math:`i` th bit and :math:`P_{i,T}` being the :math:`i` th properly transformed Pauli operator.

    Args:
        op: Pauli string operator.
        binary: Measured bit-string.

    Returns:
        Expectation value of Pauli string.
    """
    # The sign will never change if the letter is I, thus represent all I's as 0.
    # The rest is represented by 1.
    opbit = int(op.replace("I", "0").replace("Z", "1").replace("X", "1").replace("Y", "1"), 2)
    # There can only be sign change when the binary-string is 1.
    # Now a binary-and can be performed to calculate number of sign changes.
    count = (opbit & binary).bit_count()
    if count % 2 == 1:
        return -1
    return 1


def swap_indices(num_qubits: int, swap: tuple[int, int]) -> list[int]:
    """Find new bit ordering based on a bit swap.

    Args:
        num_qubits: Number of qubits
        swap: Swap to be performed

    Returns:
        List of new ordering in qubit basis based on swap.
    """
    number = 2**num_qubits
    pos1, pos2 = swap

    # original list in decimals
    decimal_list = np.arange(number)

    # Generate the binary representation matrix
    binary_matrix = ((decimal_list[:, None] & (1 << np.arange(num_qubits)[::-1])) > 0).astype(int)

    # Swap the bits at pos1 and pos2 for each row (each binary number)
    swapped_binary_matrix = binary_matrix.copy()
    swapped_binary_matrix[:, [pos1, pos2]] = swapped_binary_matrix[:, [pos2, pos1]]

    # Convert the new swapped binary matrix back to decimal
    powers_of_two = 2 ** np.arange(num_qubits)[::-1]  # [2^(num_bits-1), ..., 2^0]
    swapped_decimal_list = swapped_binary_matrix.dot(powers_of_two)

    return np.argsort(swapped_decimal_list)


def find_swaps(new: list[int], ref: list[int]) -> list[tuple[int, int]]:
    """Find swaps to turn new into ref.

    Args:
        new: List to be changed
        ref: Reference list

    Returns:
        Swaps to turn list into ref
    """
    swaps = []
    list_in = new.copy()

    for i in range(len(list_in)):
        if list_in[i] != ref[i]:
            # Find where the element from ref[i] is in list_in and swap it
            swap_idx = list_in.index(ref[i], i)
            swaps.append((i, swap_idx))
            # Perform the swap in list_in
            list_in[i], list_in[swap_idx] = list_in[swap_idx], list_in[i]

    return swaps


class CliqueHead:
    def __init__(self, head: str, distr: dict[int, float] | None) -> None:
        """Initialize clique head dataclass.

        Args:
            head: Clique head.
            distr: Sample state distribution.
        """
        self.head = head
        self.distr = distr


class Clique:
    """Clique class.

    #. 10.1109/TQE.2020.3035814, Sec. IV. A, IV. B, and VIII.
    """

    def __init__(self) -> None:
        """Initialize clique class."""
        self.cliques: list[CliqueHead] = []

    def add_paulis(self, paulis: list[str]) -> list[str]:
        """Add list of Pauli strings to cliques and return clique heads to be simulated.

        Args:
            paulis: Paulis to be added to cliques.

        Returns:
            List of clique heads to be calculated.
        """
        # The special case of computational basis
        # should always be the first clique.
        if len(self.cliques) == 0:
            self.cliques.append(CliqueHead("Z" * len(paulis[0]), None))

        # Loop over Pauli strings (passed via observable) in reverse sorted order
        for pauli in sorted(paulis, reverse=True):
            # Loop over Clique heads simulated so far
            for clique_head in self.cliques:
                # Check if Pauli string belongs to any already simulated Clique head.
                do_fit, head_fit = fit_in_clique(pauli, clique_head.head)
                if do_fit:
                    if head_fit != clique_head.head:
                        # Update Clique head by setting distr to None (= to be simulated)
                        clique_head.distr = None
                    clique_head.head = head_fit
                    break
            else:  # no break
                # Pauli String does not fit any simulated Clique head and has to be simulated
                self.cliques.append(CliqueHead(pauli, None))

        # Find new Paulis that need to be measured
        new_heads = []
        for clique_head in self.cliques:
            if clique_head.distr is None:
                new_heads.append(clique_head.head)
        return new_heads

    def update_distr(self, new_heads: list[str], new_distr: list[dict[int, float]]) -> None:
        """Update sample state distributions of clique heads.

        Args:
            new_heads: List of clique heads.
            new_distr: List of sample state distributions.
        """
        for head, distr in zip(new_heads, new_distr):
            for clique_head in self.cliques:
                if head == clique_head.head:
                    if clique_head.distr is not None:
                        raise ValueError(
                            f"Trying to update head distr that is not None. Head; {clique_head.head}"
                        )
                    clique_head.distr = distr

        # Check that all heads have a distr
        for clique_head in self.cliques:
            if clique_head.distr is None:
                raise ValueError(f"Head, {clique_head.head}, has a distr that is None")

    def get_distr(self, pauli: str) -> dict[int, float]:
        """Get sample state distribution for a Pauli string.

        Args:
            pauli: Pauli string.

        Returns:
            Sample state distribution.
        """
        for clique_head in self.cliques:
            do_fit, head_fit = fit_in_clique(pauli, clique_head.head)
            if do_fit:
                if clique_head.head != head_fit:
                    raise ValueError(
                        f"Found matching clique, but head will be mutated. Head; {clique_head.head}, Pauli; {pauli}"
                    )
                if clique_head.distr is None:
                    raise ValueError(f"Head, {clique_head.head}, has a distr that is None")
                return clique_head.distr
        raise ValueError(f"Could not find matching clique for Pauli, {pauli}")


def fit_in_clique(pauli: str, head: str) -> tuple[bool, str]:
    """Check if a Pauli fits in a given clique.

    Args:
        pauli: Pauli string.
        head: Clique head.

    Returns:
        If commuting and new clique head.
    """
    is_commuting = True
    new_head = ""
    # Check commuting
    for p_clique, p_op in zip(head, pauli):
        if p_clique == "I" or p_op == "I":
            continue
        if p_clique != p_op:
            is_commuting = False
            break
    # Check common Clique head
    if is_commuting:
        for p_clique, p_op in zip(head, pauli):
            if p_clique != "I":
                new_head += p_clique
            else:
                new_head += p_op
    return is_commuting, new_head


def correct_distribution(dist: dict[int, float], M: np.ndarray) -> dict[int, float]:
    r"""Corrects a quasi-distribution of bitstrings based on a correlation matrix in statevector notation.

    Args:
        dist: Quasi-distribution.
        M: Correlation martix (inverse).

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
    r"""Corrects a quasi-distribution of bitstrings based on a correlation matrix in statevector notation.

    Uses layout correction via distribution mapping.

    Args:
        dist: Quasi-distribution.
        M: Correlation martix (inverse).
        ref_layout: Reference layout of M measurement.
        new_layout: Layout of current to be corrected circuit measurement.

    Returns:
        Quasi-distribution corrected by correlation matrix with corrected layout
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
    for bitint, prob in dist.items():  # is this missing sth? Can I have new numbers?
        dist[bitint] = C_new[bitint]
    return dist


def correct_distribution_with_layout(
    dist: dict[int, float], M_in: np.ndarray, ref_layout: list[int], new_layout: list[int]
) -> dict[int, float]:
    r"""Corrects a quasi-distribution of bitstrings based on a correlation matrix in statevector notation.

    Uses layout correction via M mapping.

    Args:
        dist: Quasi-distribution.
        M: Correlation martix (not inverse).
        ref_layout: Reference layout of M measurement.
        new_layout: Layout of current to be corrected circuit measurement.

    Returns:
        Quasi-distribution corrected by correlation matrix with corrected layout
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
    for bitint, prob in dist.items():  # is this missing sth? Can I have new numbers?
        dist[bitint] = C_new[bitint]
    return dist


def layout_conserving_compose(
    ansatz: QuantumCircuit, state: QuantumCircuit, pm: PassManager, optimization: bool = False
) -> QuantumCircuit:
    """Composing an un-transpiled state circuit to the front of a transpiled Ansatz circuit.

    Args:
        ansatz: Transpiled Ansatz circuit
        state: Un-transpiled state circuit
        pm: PassManager that produces Ansatz's initial layout indices
        optimization: Boolean for optimizing composed circuit.
            Note that optimization can lead to change in Ansatz's gates and CX count.
            This can be problematic together with M_Ansatz0.

    Returns:
        Composed QuantumCircuit.
    """
    state_tmp = pm.translation.run(pm.layout.run(state))
    state_tmp._layout = ansatz.layout  # pylint: disable=protected-access
    state_tmp = pm.optimization.run(state_tmp)

    composed = ansatz.compose(state_tmp, front=True)

    if composed.layout.initial_index_layout(filter_ancillas=True) != ansatz.layout.initial_index_layout(
        filter_ancillas=True
    ):
        raise ValueError("Something went wrong with layout conserving composing. Initial layout changed.")
    if composed.layout.final_index_layout != ansatz.layout.final_index_layout:
        raise ValueError("Something went wrong with layout conserving composing. Final layout changed.")

    if optimization:
        composed_opt = pm.optimization.run(composed)
        composed_opt._layout = ansatz.layout  # pylint: disable=protected-access

        if composed_opt.layout.initial_index_layout(
            filter_ancillas=True
        ) != ansatz.layout.initial_index_layout(filter_ancillas=True):
            raise ValueError(
                "Something went wrong with layout conserving composing. Initial layout changed in optimization."
            )
        if composed_opt.layout.final_index_layout != ansatz.layout.final_index_layout:
            raise ValueError(
                "Something went wrong with layout conserving composing. Final layout changed in optimization."
            )

        return composed_opt
    return composed


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


def f2q(i: int, num_orbs: int) -> int:
    r"""Convert fermionic index to qubit index.

    The fermionic index is assumed to follow the convention,

    .. math::
        \left|0_\alpha 0_\beta 1_\alpha 1_\beta ... N_\alpha N_\beta\right>

    The qubit index follows,

    .. math::
       \left|0_\alpha 1_\alpha ... N_\alpha 0_\beta 1_\beta ... N_\beta\right>

    This function assumes Jordan-Wigner mapping.

    Args:
        i: Fermionic index.
        num_orbs: Number of spatial orbitals.

    Returns:
        Qubit index.
    """
    if i % 2 == 0:
        return i // 2
    return i // 2 + num_orbs


def get_determinant_superposition_reference(
    det1: str, det2: str, num_orbs: int, mapper: JordanWignerMapper
) -> QuantumCircuit:
    """Erik could have described the function."""
    if not isinstance(mapper, JordanWignerMapper):
        raise TypeError("Only implemented for JordanWignerMapper. Got: {type(mapper)}")
    qc = QuantumCircuit(2 * num_orbs)
    for i, occ in enumerate(det1):
        idx = f2q(i, num_orbs)
        if occ == "1":
            qc.x(idx)
    for i, (occ1, occ2) in enumerate(zip(det1, det2)):
        idx = f2q(i, num_orbs)
        if occ1 == "0" and occ2 == "1":
            hadamard_idx = idx
            qc.h(idx)
            break
    else:  # No break
        raise ValueError("Failed to find idx for Hadamard gate")
    for i, (occ1, occ2) in enumerate(zip(det1, det2)):
        idx = f2q(i, num_orbs)
        if occ1 == occ2 or idx == hadamard_idx:
            continue
        if occ1 == "1" or occ2 == "1":
            qc.cx(hadamard_idx, idx)
    return qc


def get_determinant_reference(det, num_orbs, mapper) -> QuantumCircuit:
    """Erik could have described the function."""
    if not isinstance(mapper, JordanWignerMapper):
        raise TypeError("Only implemented for JordanWignerMapper. Got: {type(mapper)}")
    qc = QuantumCircuit(2 * num_orbs)
    for i, occ in enumerate(det):
        idx = f2q(i, num_orbs)
        if occ == "1":
            qc.x(idx)
    return qc


def get_reordering_sign(det) -> int:
    """Erik could have described the function."""
    sign = 1
    alphas = 0
    for i, occ in enumerate(det[::-1]):
        # Doing reverse thus alpha are the uneven
        if i % 2 == 1 and occ == "1":
            alphas += 1
        # Doing the reverse thus beta are the even
        elif i % 2 == 0 and occ == "1":
            if alphas % 2 == 1:
                sign *= -1
    return sign
