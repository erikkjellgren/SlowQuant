import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, PauliList


def to_CBS_measurement(op: PauliList) -> QuantumCircuit:
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
        op: Pauli string operator.

    Returns:
        Pauli measuremnt quantum circuit.
    """
    num_qubits = len(op)
    qc = QuantumCircuit(num_qubits)
    for i, pauli in enumerate(op):
        if pauli == Pauli("X"):
            qc.append(pauli, [i])
            qc.h(i)
        elif pauli == Pauli("Y"):
            qc.append(pauli, [i])
            qc.sdg(i)
            qc.h(i)
    return qc


def get_bitstring_sign(op: Pauli, binary: str) -> int:
    r"""Convert Pauli string and bit-string measurement to expectation value.

    Takes Pauli String and a state in binary form and returns the sign based on the expectation value of the Pauli string with each single quibit state.

    This is achived by using the following evaluations:

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

    With :math:`b_i` being the :math:`i` th bit and :math:`P_{i,T}` being the :math:`i` th proberly transformed Pauli operator.

    Args:
        op: Pauli string operator.
        binary: Measured bit-string.

    Returns:
        Expectation value of Pauli string.
    """
    sign = 1
    for i, pauli in enumerate(op.to_label()):
        if not pauli == "I":
            if binary[i] == "1":
                sign = sign * (-1)
    return sign


class CliqueHead:
    def __init__(self, head: str, distr: dict[str, float] | None) -> None:
        """Initialize clique head dataclass.

        Args:
            head: Clique head.
            distr: Sample state distribution.
        """
        self.head = head
        self.distr = distr


class Clique:
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
        # Is that needed? Alreday in make_cliques?
        if len(self.cliques) == 0:
            self.cliques.append(CliqueHead("Z" * len(paulis[0]), None))

        # Loop over Pauli strings (passed via observable)
        for pauli in paulis:
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

    def update_distr(self, new_heads: list[str], new_distr: list[dict[str, float]]) -> None:
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

    def get_distr(self, pauli: str) -> dict[str, float]:
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
                        f"Found matching clique, but head will be mutate. Head; {clique_head.head}, Pauli; {pauli}"
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


def make_cliques(paulis: PauliList) -> dict[str, list[str]]:
    """Partition Pauli strings into simultaniously measurable cliques.

    The Pauli strings are put into cliques accourding to Qubit-Wise Commutativity (QWC).

    #. https://arxiv.org/pdf/1907.13623.pdf, Sec. 4.1, 4.2, and 7.0
    """
    cliques: dict[str, list[str]] = {"Z" * len(paulis[0]): []}
    for pauli in paulis:
        pauli_str = str(pauli)
        if "X" not in pauli_str and "Y" not in pauli_str:
            cliques["Z" * len(paulis[0])].append(pauli_str)
        else:
            for clique in cliques:
                is_commuting = True
                for p_clique, p_op in zip(clique, pauli_str):
                    if p_clique == "I" or p_op == "I":
                        continue
                    if p_clique != p_op:
                        is_commuting = False
                        break
                if is_commuting:
                    commuting_clique = clique
                    break
            if is_commuting:
                new_clique_pauli = ""
                for p_clique, p_op in zip(commuting_clique, pauli_str):
                    if p_clique != "I":
                        new_clique_pauli += p_clique
                    else:
                        new_clique_pauli += p_op
                if new_clique_pauli != commuting_clique:
                    cliques[new_clique_pauli] = cliques[commuting_clique]
                    del cliques[commuting_clique]
                cliques[new_clique_pauli].append(pauli_str)
            else:
                cliques[pauli_str] = [pauli_str]
    return cliques


def correct_distribution(dist: dict[str, float], M: np.ndarray) -> dict[str, float]:
    r"""Corrects a quasi-distribution of bitstrings based on a correlation matrix in statevector notation.

    Args:
        dist: Quasi-distribution.
        M:    Correlation martix.

    Returns:
        Quasi-distribution corrected by correlation matrix.
    """
    C = np.zeros(np.shape(M)[0])
    # Convert bitstring distribution to columnvector of probabilities
    for bitstring, prob in dist.items():
        idx = int(bitstring[::-1], 2)
        C[idx] = prob
    # Apply M error mitigation matrix
    C_new = M @ C
    # Convert columnvector of probabilities to bitstring distribution
    for bitstring, prob in dist.items():
        idx = int(bitstring[::-1], 2)
        dist[bitstring] = C_new[idx]
    return dist
