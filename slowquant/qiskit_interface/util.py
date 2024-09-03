import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_nature.second_q.circuit.library import HartreeFock
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
        for i, pauli in enumerate(op[::-1]):
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
        M: Correlation martix.

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
            for bit in bitstr_a:
                if bit != current_parity:
                    current_parity = bit
                    change_counter += 1
            if current_parity == "1" and num_elec[0] % 2 == 0:
                change_counter += 1
            elif current_parity == "0" and num_elec[0] % 2 == 1:
                change_counter += 1
            if change_counter != num_elec[0]:
                break
            current_parity = "0"
            change_counter = 0
            for bit in bitstr_b:
                if bit != current_parity:
                    current_parity = bit
                    change_counter += 1
            if current_parity == "1" and num_elec[1] % 2 == 0:
                change_counter += 1
            elif current_parity == "0" and num_elec[1] % 2 == 1:
                change_counter += 1
            if change_counter != num_elec[1]:
                break
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


def get_csf_reference(
    csf: list[str], num_orbs: int, num_elec: tuple[int, int], mapper: JordanWignerMapper
) -> QuantumCircuit:
    if not isinstance(mapper, JordanWignerMapper):
        raise TypeError("Only implemented for JordanWignerMapper. Got: {type(mapper)}")
    qc = HartreeFock(num_orbs, num_elec, mapper)
    return qc


def get_determinant_superposition_reference(
    det1: str, det2: str, num_orbs: int, num_elec: tuple[int, int], mapper: JordanWignerMapper
) -> QuantumCircuit:
    if not isinstance(mapper, JordanWignerMapper):
        raise TypeError("Only implemented for JordanWignerMapper. Got: {type(mapper)}")
    qc = HartreeFock(num_orbs, num_elec, mapper)
    return qc
