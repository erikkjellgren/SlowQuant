import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper


def to_CBS_measurement(op: str, transpiled: None | list[QuantumCircuit] = None) -> QuantumCircuit:
    r"""Convert a Pauli string to Pauli measurement circuit.

    This is achieved by the following transformation:

    .. math::
        \begin{align}
        I &\rightarrow I\\
        Z &\rightarrow I\\
        X &\rightarrow H\\
        Y &\rightarrow S^{\dagger}H
        \end{align}

    #. https://learn.microsoft.com/en-us/azure/quantum/concepts-pauli-measurements, 14/11-2025

    Args:
        op: Pauli string.
        transpiled: List of transpiled X and Y gate.

    Returns:
        Pauli measurement quantum circuit.
    """
    if transpiled is None:
        num_qubits = len(op)
        qc = QuantumCircuit(num_qubits)
        for i, pauli in enumerate(op[::-1]):  # turn order to q0,q1,...,qN
            if pauli == "X":
                qc.h(i)
            elif pauli == "Y":
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

    The total expectation value is then evaluated as:

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
        num_qubits: Number of qubits.
        swap: Swap to be performed.

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
        new: List to be changed.
        ref: Reference list.

    Returns:
        Swaps to turn list into ref.
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


class MitigationFlags:
    """Class to handle mitigation flags."""

    def __init__(
        self,
        do_M_mitigation: bool = False,
        do_M_ansatz0: bool = False,
        do_M_ansatz0_plus: bool = False,
        do_postselection: bool = False,
    ):
        """Initialize mitigation flags.

        Args:
            do_M_mitigation: Whether to perform error mitigation.
            do_M_ansatz0: Whether to perform M0 mitigation.
            do_M_ansatz0_plus: Whether to perform M0 mitigation for each initial superposition state.
            do_postselection: Whether to perform post-selection.
        """
        if do_M_ansatz0 is True:
            do_M_mitigation = True
        if do_M_ansatz0_plus is True:
            do_M_mitigation = True
            do_M_ansatz0 = True
        self.do_M_mitigation = do_M_mitigation
        self.do_M_ansatz0 = do_M_ansatz0
        self.do_M_ansatz0_plus = do_M_ansatz0_plus
        self.do_postselection = do_postselection
        # Tuple of flag names in the order they should be processed.
        self._flag_order = ("do_M_mitigation", "do_M_ansatz0", "do_M_ansatz0_plus", "do_postselection")

        print("You selected the following mitigation flags:\n" + self.status_report())

    def _validate_flags(self, flag_dict) -> None:
        """Validate the provided flags against the defined flag_order.

        Args:
            flag_dict: Dictionary of flags to validate.
        """
        unknown_flags = set(flag_dict) - set(self._flag_order)
        if unknown_flags:
            raise ValueError(
                f"Unknown flag(s): {', '.join(unknown_flags)}\nAccepted flags: {self._flag_order}"
            )
        for flag in flag_dict.items():
            if not isinstance(flag[1], bool):
                raise TypeError(f"{flag[0]} must be a boolean")

    def update_flags(self, **kwargs) -> None:
        """Update mitigation flags with provided keyword arguments.

        Args:
            **kwargs: Flags to be updated. If a flag is not provided, it remains unchanged.
        """
        self._validate_flags(kwargs)
        for flag, value in kwargs.items():
            setattr(self, flag, value)
        if self.do_M_ansatz0 is True:
            self.do_M_mitigation = True
        if self.do_M_ansatz0_plus is True:
            self.do_M_mitigation = True
            self.do_M_ansatz0 = True
        print("You selected the following mitigation flags:\n" + self.status_report())

    def all_to_false(self) -> None:
        """Set all mitigation flags to False."""
        for flag in self._flag_order:
            setattr(self, flag, False)
        print("All mitigation flags set to False.")

    def to_int(self) -> int:
        """Convert mitigation flags to an integer representation.

        Returns:
            Integer representation of the mitigation flags, where each bit corresponds to a flag.
        """
        result = 0
        for i, flag in enumerate(self._flag_order):
            if getattr(self, flag):
                result |= 1 << i
        return result

    def status_report(self) -> str:
        """Generate a status report of the mitigation flags.

        Returns:
            A string report indicating the status of each mitigation flag.
        """
        lines = []
        for flag in self._flag_order:
            value = getattr(self, flag)
            status = "ON" if value else "OFF"
            lines.append(f"{flag}: {status}")
        return "\n".join(lines)

    def is_enabled(self) -> bool:
        """Check if any mitigation is enabled.

        Returns:
            True if any mitigation flag is set to True, False otherwise.
        """
        if self.to_int() == 0:
            return False
        return True

    def __repr__(self):
        """Representation of the MitigationFlags object."""
        flags = {f: getattr(self, f) for f in self._flag_order}
        return f"<MitigationFlags int={self.to_int()} bin={bin(self.to_int())} flags={flags}>"


class CliqueSaver:
    """Class to handle saving of cliques.

    This class is used to save a clique heads distributions based on the used mitigation scheme.
    """

    def __init__(self) -> None:
        """Initialize CliqueSaver."""
        self.data: dict[int, dict[int, float]] = {0: {}}  # initialize raw results saver

    def reset(self) -> None:
        """Reset CliqueSaver."""
        self.data = {0: {}}  # reset to empty raw results saver

    def is_empty(self, mitigation_flags: MitigationFlags | None = None) -> bool:
        """Check if saver is empty.

        Args:
            mitigation_flags: Mitigation flags object, default is None.

        Returns:
            True if empty, False otherwise.
        """
        mitigation_int = mitigation_flags.to_int() if mitigation_flags is not None else 0
        if mitigation_int not in self.data:
            return True  # it is empty if it does not exist.
        if self.data[mitigation_int]:
            return False  # not empty
        # check if all are empty
        if mitigation_int == 0 and len(self.data) > 1:
            if not all(len(inner) == 0 for inner in self.data.values()):
                raise ValueError("Empty data not consistent accross mitigation saver.")
        return True  # empty

    def add_distr(self, distr: dict[int, float], mitigation_flags: MitigationFlags | None = None) -> None:
        """Add distribution to saver.

        Args:
            distr: Distribution to be added.
            mitigation_flags: Mitigation flags object, default is None.
        """
        mitigation_int = mitigation_flags.to_int() if mitigation_flags is not None else 0
        self.data[mitigation_int] = distr

    def get_distr(self, mitigation_flags: MitigationFlags | None = None) -> dict[int, float]:
        """Return distribution.

        Args:
            mitigation_flags: Mitigation flags object, default is None.

        Returns:
            Distribution for the given mitigation integer.
        """
        mitigation_int = mitigation_flags.to_int() if mitigation_flags is not None else 0
        return self.data[mitigation_int]


class CliqueHead:
    def __init__(self, head: str) -> None:
        """Initialize clique head dataclass.

        Args:
            head: Clique head.
            distr: Sample state distribution.
        """
        self.head = head
        self.distr = CliqueSaver()


class Clique:
    """Clique class.

    #. 10.1109/TQE.2020.3035814, Sec. IV. A, IV. B, and VIII.
    """

    def __init__(self, csfs_option: int = 1) -> None:
        """Initialize clique class."""
        self.cliques: list[CliqueHead] = []
        self.csfs_option = csfs_option

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
            self.cliques.append(CliqueHead("Z" * len(paulis[0])))

        # Loop over Pauli strings (passed via observable) in reverse sorted order
        for pauli in sorted(paulis, reverse=True):
            # Loop over Clique heads simulated so far
            for clique_head in self.cliques:
                # Check if Pauli string belongs to any already simulated Clique head.
                do_fit, head_fit = fit_in_clique(pauli, clique_head.head)
                if do_fit:
                    if head_fit != clique_head.head:
                        # Update Clique head by resetting distr (= to be simulated)
                        clique_head.distr.reset()
                    clique_head.head = head_fit
                    break
            else:  # no break
                # Pauli String does not fit any simulated Clique head and has to be simulated
                self.cliques.append(CliqueHead(pauli))

        # Find new Paulis that need to be measured
        new_heads = []
        for clique_head in self.cliques:
            if clique_head.distr.is_empty():
                new_heads.append(clique_head.head)
        return new_heads

    def update_distr(
        self,
        new_heads: list[str],
        new_distr: list[dict[int, float]],
        mitigation_flags: MitigationFlags | None = None,
    ) -> None:
        """Update sample state distributions of clique heads.

        Args:
            new_heads: List of clique heads.
            new_distr: List of sample state distributions.
            mitigation_flags: Mitigation flags object, default is None.
        """
        for head, distr in zip(new_heads, new_distr):
            for clique_head in self.cliques:
                if head == clique_head.head:
                    if not clique_head.distr.is_empty(mitigation_flags):
                        raise ValueError(
                            f"Trying to update head distr that is not None. Head; {clique_head.head}; mitigation; {mitigation_flags}"
                        )
                    clique_head.distr.add_distr(distr, mitigation_flags)

        # Check that all heads have a distr
        for clique_head in self.cliques:
            if clique_head.distr.is_empty(mitigation_flags):
                raise ValueError(
                    f"Head, {clique_head.head}, has not been allocated for mitigation; {mitigation_flags}"
                )

    def get_distr(self, pauli: str, mitigation_flags: MitigationFlags | None = None) -> dict[int, float]:
        """Get sample state distribution for a Pauli string.

        Args:
            pauli: Pauli string.
            mitigation_flags: Mitigation flags object, default is None.

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
                if clique_head.distr.is_empty(mitigation_flags):
                    raise ValueError(
                        f"Head, {clique_head.head}, has no distribution for mitigation; {mitigation_flags}"
                    )
                return clique_head.distr.get_distr(mitigation_flags)
        raise ValueError(f"Could not find matching clique for Pauli, {pauli}")

    def get_empty_heads(self, mitigation_int) -> list[str]:
        """Return all heads that do not have any data for a specific mitigation int.

        Args:
            mitigation_int: Mitigation integer, default is 0.

        Returns:
            List of heads that have no data for the given mitigation integer.
        """
        empties = []
        for clique_head in self.cliques:
            if clique_head.distr.is_empty(mitigation_int):
                empties.append(clique_head.head)
        return empties

    def get_distr_heads(self, heads: list[str], mitigation_int: int = 0) -> list[dict[int, float]]:
        """Return the distribution data for a list of heads and for a given mitigation int (default 0).

        This function looks only for the specific heads given, not trying to find pauli strings in commuting heads.
        The latter would be the function get_distr.

        Args:
            heads: List of clique heads.
            mitigation_int: Mitigation integer, default is 0.

        Returns:
            List of distributions for the given heads and mitigation integer.
        """
        distr = []
        # This needs looping many times over all cliques.
        # To avoid this one would need to re-write the whole clique structure into dictionaries.
        # But this might make the other clique operations slower.
        for head in heads:
            for clique_heads in self.cliques:
                if clique_heads.head == head:
                    distr.append(clique_heads.distr.data[mitigation_int])
                    break

        return distr

    def get_empty_heads_distr(
        self, mitigation_flags: MitigationFlags
    ) -> tuple[list[str], list[dict[int, float]]]:
        """Return all heads that have empty data for a given mitigation_int plus return the corresponding raw data.

        Args:
            mitigation_flags: Mitigation flags object.

        Returns:
            Tuple of lists: List of heads that have no data for the given mitigation integer,
            and list of corresponding raw data distributions.
        """
        heads = []
        distr = []
        for clique_head in self.cliques:
            if clique_head.distr.is_empty(mitigation_flags):
                heads.append(clique_head.head)
                distr.append(clique_head.distr.data[0].copy())  # raw data
        return heads, distr

    def print_cliques(self) -> None:
        """Print all cliques and their distributions."""
        print("Clique heads and their distributions:")
        for clique_head in self.cliques:
            print(f"Head: {clique_head.head}, Distribution: {clique_head.distr.data}")

    def print_clique_heads(self) -> None:
        """Print all clique heads."""
        print("Clique heads:")
        heads = []
        for clique_head in self.cliques:
            heads.append(clique_head.head)
        print(heads)


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
        M_in: Correlation matrix (not inverse).
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


def find_best_path(coupling_map: CouplingMap, start: int, target: int) -> list[int]:
    """Find the best path between two qubits using the coupling map.

    Args:
        coupling_map: The coupling map defining valid connections.
        start: Starting qubit.
        target: Target qubit.

    Returns:
        List of qubits representing the path from start to target.
    """
    # Convert the coupling map to a NetworkX graph
    graph = nx.Graph()
    graph.add_edges_from(coupling_map.get_edges())

    # Find the shortest path between start and target
    try:
        path = nx.shortest_path(graph, source=start, target=target)
    except nx.NetworkXNoPath as e:
        raise ValueError(f"No valid path between qubit {start} and qubit {target}") from e

    return path


def add_permutation_gate(circuit: QuantumCircuit, permutation: list[int], coupling_map: CouplingMap) -> None:
    """Add a permutation gate to a circuit, adhering to a given coupling map.

    Args:
        circuit: The quantum circuit to modify.
        permutation: A list defining the new positions of qubits.
                            E.g., [2, 0, 1] means qubit 0 goes to 2, 1 goes to 0, and 2 goes to 1.
        coupling_map: The coupling map that defines valid qubit connections.
    """
    # Get the number of qubits
    num_qubits = len(permutation)

    # Validate input
    if num_qubits != circuit.num_qubits:
        raise ValueError("Permutation size must match the number of qubits in the circuit.")

    # Current positions of qubits
    current_positions = list(range(num_qubits))

    # Self-consistent approach to finding all swaps
    while current_positions != permutation:
        # Generate SWAP gates to achieve the permutation
        # Loop over all positions in array (num_qubits)
        for target_position in range(num_qubits):
            # Find the current qubit that needs to be swapped to the target position
            correct_qubit = permutation[target_position]  # qubit meant to be in target_position
            current_qubit_position = current_positions.index(correct_qubit)  # position of correct_qubit

            if current_qubit_position != target_position:
                # Find the full path from the current position to the target position
                path = find_best_path(coupling_map, current_qubit_position, target_position)

                # Apply SWAP gates along the path
                for i in range(len(path) - 1):
                    circuit.swap(path[i], path[i + 1])

                    # Update current positions after each SWAP
                    current_positions[path[i]], current_positions[path[i + 1]] = (
                        current_positions[path[i + 1]],
                        current_positions[path[i]],
                    )


def layout_conserving_compose(
    ansatz: QuantumCircuit,
    state: QuantumCircuit,
    pm: PassManager,
    coupling_map: CouplingMap,
    optimization: bool = False,
    M_circuit: bool = False,
) -> QuantumCircuit | tuple[QuantumCircuit, QuantumCircuit]:
    """Composing an un-transpiled state circuit to the front of a transpiled Ansatz circuit.

    Args:
        ansatz: Transpiled Ansatz circuit.
        state: Un-transpiled state circuit.
        pm: PassManager that produces Ansatz's initial layout indices.
        coupling_map: Coupling map for device layout.
        optimization: Boolean for optimizing composed circuit.
            Note that optimization can lead to changes in Ansatz's gates and CX count.
            This can be problematic together with M_Ansatz0.
        M_circuit: Depricated?

    Returns:
        Composed QuantumCircuit.
    """
    print("\nLayout-conserving circuit composing requested.")
    print("Superposition state contains ", state.num_nonlocal_gates(), "non-local gates")
    if ansatz.layout is not None:  # or just use ISA_option 1
        state_tmp = pm.run(state)
        nlg_state = state_tmp.num_nonlocal_gates()
        print("Transpiled superposition state contains ", nlg_state, "non-local gates")

        if (
            state_tmp.layout.initial_index_layout(filter_ancillas=True)
            != state_tmp.layout.final_index_layout()
        ):
            # Qiskit wants to change the layout? Not with me. Change it back!

            if coupling_map is None:
                raise ValueError("No coupling map defined for layout conserving circuit composing.")

            print("Starting self-consistent permutation circuit search...")
            add_permutation_gate(state_tmp, state_tmp.layout.routing_permutation(), coupling_map)
            state_tmp = pm.translation.run(state_tmp)
            print(
                "Layout correction added ",
                state_tmp.num_nonlocal_gates() - nlg_state,
                " non-local gates (without swap optimization)",
            )
            state_tmp = pm.optimization.run(state_tmp)
            print(
                "Layout correction added ",
                state_tmp.num_nonlocal_gates() - nlg_state,
                " non-local gates (with swap optimization)",
            )

        composed = ansatz.compose(state_tmp, front=True)

        if optimization:
            nlg_composed = composed.num_nonlocal_gates()
            composed = pm.optimization.run(composed)
            composed._layout = ansatz.layout
            print(
                "Composed circuit optimization eliminated ",
                nlg_composed - composed.num_nonlocal_gates(),
                "non-local gates",
            )
    else:
        state_tmp = pm.run(state)
        nlg_state = state_tmp.num_nonlocal_gates()
        print("Transpiled superposition state contains ", nlg_state, "non-local gates")
        composed = ansatz.compose(state_tmp, front=True)
        if optimization:
            nlg_composed = composed.num_nonlocal_gates()
            composed = pm.optimization.run(composed)
            print(
                "Optimization eliminated ",
                nlg_composed - composed.num_nonlocal_gates(),
                "non-local gates",
            )

    if M_circuit:  # deprecated?
        state_corr = state.copy()
        # Get state circuit without non-local gates
        for idx, instruction in reversed(list(enumerate(state_corr.data))):
            if instruction.is_controlled_gate():
                del state_corr.data[idx]
        # Translate and optimize
        state_corr = pm.optimization.run(pm.translation.run(state_corr))
        # Negate
        if ansatz.layout is not None:
            composed_M = composed.compose(
                state_corr, front=True, qubits=composed.layout.initial_index_layout(filter_ancillas=True)
            )
        else:
            composed_M = composed.compose(state_corr, front=True)

        return (composed, composed_M)

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
    This counting is done independently for the :math:`\alpha` part and :math:`\beta` part.

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
    """Get quantum circuit for superposition of two determinants.

    Args:
        det1: First determinant.
        det2: Second determinant.
        num_orbs: Number of spatial orbitals.
        mapper: Fermionic to qubit mapper.

    Returns:
        Quantum circuit for superposition of two determinant.
    """
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


def get_determinant_superposition_reference_MAnsatz0(
    det1: str, det2: str, num_orbs: int, mapper: JordanWignerMapper
) -> QuantumCircuit:
    """Get superposition state for MAnsatz_0.

    Args:
        det1: determinant string 1.
        det2: determinant string 2.
        num_orbs: Number of spin orbitals.
        mapper: Fermionic to qubit mapper.

    Returns:
        Quantum circuit.
    """
    if not isinstance(mapper, JordanWignerMapper):
        raise TypeError("Only implemented for JordanWignerMapper. Got: {type(mapper)}")
    qc = QuantumCircuit(2 * num_orbs)
    for i, (occ1, occ2) in enumerate(zip(det1, det2)):
        idx = f2q(i, num_orbs)
        if occ1 == "0" and occ2 == "1":
            hadamard_idx = idx
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


def get_determinant_reference(det: str, num_orbs: int, mapper: FermionicMapper) -> QuantumCircuit:
    """Get quantum circuit for a determinant.

    Args:
        det: Determinant.
        num_orbs: Number of spatial orbitals.
        mapper: Fermionic to qubit mapper.

    Returns:
        Quantum circuit for determinant.
    """
    if not isinstance(mapper, JordanWignerMapper):
        raise TypeError("Only implemented for JordanWignerMapper. Got: {type(mapper)}")
    qc = QuantumCircuit(2 * num_orbs)
    for i, occ in enumerate(det):
        idx = f2q(i, num_orbs)
        if occ == "1":
            qc.x(idx)
    return qc


def get_reordering_sign(det: str) -> int:
    """Get sign from reordering determinant.

    The reordering is done from spin-paired to spin-blocked.

    Args:
        det: Determinant.

    Returns:
        Phase factor from the reordering.
    """
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
