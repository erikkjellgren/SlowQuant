import itertools

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimator, BaseSampler
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
from qiskit_nature.second_q.circuit.library import PUCCD, UCC, UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.operators import FermionicOp

from slowquant.qiskit_interface.base import FermionicOperator
from slowquant.qiskit_interface.custom_ansatz import (
    ErikD_JW,
    ErikD_Parity,
    ErikSD_JW,
    ErikSD_Parity,
)


class QuantumInterface:
    """Quantum interface class.

    This class handles the interface with qiskit and the communication with quantum hardware.
    """

    def __init__(
        self,
        primitive: BaseEstimator | BaseSampler,
        ansatz: str,
        mapper: FermionicMapper,
        do_M_mitigation: bool = False,
        do_M_iqa: bool = False,
        do_M_ansatz0: bool = False,
    ) -> None:
        """Interface to Qiskit to use IBM quantum hardware or simulator.

        Args:
            primitive: Qiskit Estimator or Sampler object.
            ansatz: Name of ansatz to be used.
            mapper: Qiskit mapper object.
            do_M_mitigation: Do error mitigation via read-out correlation matrix.
            do_M_iqa: Use independent qubit approximation when constructing the read-out correlation matrix.
            do_M_ansatz0: Use the ansatz with theta=0 when constructing the read-out correlation matrix
        """
        allowed_ansatz = ("UCCSD", "PUCCD", "UCCD", "ErikD", "ErikSD")
        if ansatz not in allowed_ansatz:
            raise ValueError("The chosen Ansatz is not availbale. Choose from: ", allowed_ansatz)
        self.ansatz = ansatz
        self._primitive = primitive
        self.mapper = mapper
        self._do_M_mitigation = do_M_mitigation
        self._do_M_iqa = do_M_iqa
        self._do_M_ansatz0 = do_M_ansatz0

    def construct_circuit(self, num_orbs: int, num_elec: tuple[int, int]) -> None:
        """Construct qiskit circuit.

        Args:
            num_orbs: Number of orbitals in spatial basis.
            num_elec: Number of electrons (alpha, beta).
        """
        self.num_orbs = num_orbs
        self.num_spin_orbs = 2 * num_orbs
        self.num_elec = num_elec

        if self.ansatz == "UCCSD":
            self.circuit = UCCSD(
                num_orbs,
                num_elec,
                self.mapper,
                initial_state=HartreeFock(
                    num_orbs,
                    num_elec,
                    self.mapper,
                ),
            )
        elif self.ansatz == "PUCCD":
            self.circuit = PUCCD(
                num_orbs,
                num_elec,
                self.mapper,
                initial_state=HartreeFock(
                    num_orbs,
                    num_elec,
                    self.mapper,
                ),
            )
        elif self.ansatz == "UCCD":
            self.circuit = UCC(
                num_orbs,
                num_elec,
                "d",
                self.mapper,
                initial_state=HartreeFock(
                    num_orbs,
                    num_elec,
                    self.mapper,
                ),
            )
        elif self.ansatz == "ErikD":
            if num_orbs != 2 or list(num_elec) != [1, 1]:
                raise ValueError(f"Chosen ansatz, {self.ansatz}, only works for (2,2)")
            if isinstance(self.mapper, JordanWignerMapper):
                self.circuit = ErikD_JW()
            elif isinstance(self.mapper, ParityMapper):
                self.circuit = ErikD_Parity()
            else:
                raise ValueError(f"Unsupported mapper, {type(self.mapper)}, for ansatz {self.ansatz}")
        elif self.ansatz == "ErikSD":
            if num_orbs != 2 or list(num_elec) != [1, 1]:
                raise ValueError(f"Chosen ansatz, {self.ansatz}, only works for (2,2)")
            if isinstance(self.mapper, JordanWignerMapper):
                self.circuit = ErikSD_JW()
            elif isinstance(self.mapper, ParityMapper):
                self.circuit = ErikSD_Parity()
            else:
                raise ValueError(f"Unsupported mapper, {type(self.mapper)}, for ansatz {self.ansatz}")

        self.num_qubits = self.circuit.num_qubits
        # Set parameter to HarteeFock
        self._parameters = [0.0] * self.circuit.num_parameters

    @property
    def parameters(self) -> list[float]:
        """Get ansatz parameters.

        Returns:
            Ansatz parameters.
        """
        return self._parameters

    @parameters.setter
    def parameters(
        self,
        parameters: list[float],
    ) -> None:
        """Set ansatz parameters.

        Args:
            parameters: List of ansatz parameters.
        """
        if len(parameters) != self.circuit.num_parameters:
            raise ValueError(
                "The length of the parameter list does not fit the chosen circuit for the Ansatz ",
                self.ansatz,
            )
        self._parameters = parameters.copy()

    def op_to_qbit(self, op: FermionicOperator) -> SparsePauliOp:
        """Fermionic operator to qbit rep.

        Args:
            op: Operator as SlowQuant's FermionicOperator object

        Returns:
            Qubit representation of operator.
        """
        return self.mapper.map(FermionicOp(op.get_qiskit_form(self.num_orbs), self.num_spin_orbs))

    def quantum_expectation_value(
        self, op: FermionicOperator, custom_parameters: list[float] | None = None
    ) -> float:
        """Calculate expectation value of circuit and observables.

        Args:
            op: Operator as SlowQuant's FermionicOperator object.

        Returns:
            Expectation value of fermionic operator.
        """
        if custom_parameters is None:
            run_parameters = self.parameters
        else:
            run_parameters = custom_parameters

        # Check if estimator or sampler
        if isinstance(self._primitive, BaseEstimator):
            return self._estimator_quantum_expectation_value(op, run_parameters)
        if isinstance(self._primitive, BaseSampler):
            return self._sampler_quantum_expectation_value(op, run_parameters)
        raise ValueError(
            "The Quantum Interface was initiated with an unknown Qiskit primitive, {type(self._primitive)}"
        )

    def _estimator_quantum_expectation_value(
        self, op: FermionicOperator, run_parameters: list[float]
    ) -> float:
        """Calculate expectation value of circuit and observables via Estimator.

        Args:
            op: SlowQuant fermionic operator.
            run_parameters: Circuit parameters.

        Returns:
            Expectation value of operator.
        """
        job = self._primitive.run(
            circuits=self.circuit,
            parameter_values=run_parameters,
            observables=self.op_to_qbit(op),
        )
        result = job.result()
        values = result.values[0]

        if isinstance(values, complex):
            if abs(values.imag) > 10**-2:
                print("Warning: Complex number detected with Im = ", values.imag)

        return values.real

    def _sampler_quantum_expectation_value(self, op: FermionicOperator, run_parameters: list[float]) -> float:
        r"""Calculate expectation value of circuit and observables via Sampler.

        The expectation value over a fermionic operator is calcuated as:

        .. math::
            E = \sum_i^N c_i\left<0\left|P_i\right|0\right>

        With :math:`c_i` being the :math:`i` the coefficient and :math:`P_i` the :math:`i` the Pauli string.

        Args:
            op: SlowQuant fermionic operator.
            run_parameters: Circuit parameters.

        Returns:
            Expectation value of operator.
        """
        values = 0.0
        observables = self.op_to_qbit(op)

        # Loop over all clique Paulies
        cliques = make_cliques(observables.paulis)
        distributions = {}
        for clique_pauli, clique in cliques.items():
            dist = self._sampler_distributions(Pauli(clique_pauli), run_parameters)
            if self._do_M_mitigation:
                if not hasattr(self, "_Minv"):
                    self._make_Minv()
                C = np.zeros(2**self.num_qubits)
                for bitstring, prob in dist.items():
                    idx = int(bitstring[::-1], 2)
                    C[idx] = prob
                C_new = self._Minv @ C
                for bitstring, prob in dist.items():
                    idx = int(bitstring[::-1], 2)
                    dist[bitstring] = C_new[idx]
            # It is wasteful to store the distribution per Pauli instead of per Clique,
            # but it help unpack it later.
            for pauli in clique:
                distributions[pauli] = dist

        # Loop over all qubit-mapped Paul strings and get Sampler distributions
        for pauli, coeff in zip(observables.paulis, observables.coeffs):
            result = 0.0
            for key, value in distributions[str(pauli)].items():
                # Here we could check if we want a given key (bitstring) in the result distribution
                result += value * get_bitstring_sign(pauli, key)
            values += result * coeff

        if isinstance(values, complex):
            if abs(values.imag) > 10**-2:
                print("Warning: Complex number detected with Im = ", values.imag)

        return values.real

    def _sampler_distributions(
        self, pauli: PauliList, run_parameters: list[float], custom_circ: None | QuantumCircuit = None
    ) -> dict[str, float]:
        r"""Get results from a sampler distribution for one given Pauli string.

        The expectation value of a Pauli string is calcuated as:

        .. math::
            E = \sum_i^N p_i\left<b_i\left|P\right|b_i\right>

        With :math:`p_i` being the :math:`i` th probability and :math:`b_i` being the `i` th bit-string.

        Args:
            pauli: Pauli string to measure.
            run_paramters: Parameters of circuit.

        Returns:
            Probability weighted Pauli string.
        """
        # Create QuantumCircuit
        if custom_circ is None:
            ansatz_w_obs = self.circuit.compose(to_CBS_measurement(pauli))
        else:
            ansatz_w_obs = custom_circ.compose(to_CBS_measurement(pauli))
        ansatz_w_obs.measure_all()

        # Run sampler
        job = self._primitive.run(ansatz_w_obs, parameter_values=run_parameters)

        # Get quasi-distribution in binary probabilities
        distr = job.result().quasi_dists[0].binary_probabilities()
        return distr

    def _make_Minv(self) -> None:
        r"""Make inverse of read-out correlation matrix.

        The read-out correlation matrix is of the form (for two qubits):

        .. math::
            M = \begin{pmatrix}
                P(00|00) & P(00|10) & P(00|01) & P(00|11)\\
                P(10|00) & P(10|10) & P(10|01) & P(10|11)\\
                P(01|00) & P(01|10) & P(01|01) & P(01|11)\\
                P(11|00) & P(11|10) & P(11|01) & P(11|11)
                \end{pmatrix}

        With :math:`P(AB|CD)` meaning the probability of reading :math:`AB` given the circuit is prepared to give :math:`CD`.

        The construction also supports the independent qubit approximation, which for two qubits means that:

        .. math::
            P(\tilde{q}_1 \tilde{q}_0|q_1 q_0) = P(\tilde{q}_1|q_1)P(\tilde{q}_0|q_0)

        Under this approximation only :math:`\left<00\right|` and :math:`\left<11\right|` need to be measured,
        in order the gain enough information to construct :math:`M`.

        The read-out correlation take the following form (for two qubits):

        .. math::
            M = \begin{pmatrix}
                P_{q1}(0|0)P_{q0}(0|0) & P_{q1}(0|1)P_{q0}(0|0) & P_{q1}(0|0)P_{q0}(0|1) & P_{q1}(0|1)P_{q0}(0|1)\\
                P_{q1}(1|0)P_{q0}(0|0) & P_{q1}(1|1)P_{q0}(0|0) & P_{q1}(1|0)P_{q0}(0|1) & P_{q1}(1|1)P_{q0}(0|1)\\
                P_{q1}(0|0)P_{q0}(1|0) & P_{q1}(0|1)P_{q0}(1|0) & P_{q1}(0|0)P_{q0}(1|1) & P_{q1}(0|1)P_{q0}(1|1)\\
                P_{q1}(1|0)P_{q0}(1|0) & P_{q1}(1|1)P_{q0}(1|0) & P_{q1}(1|0)P_{q0}(1|1) & P_{q1}(1|1)P_{q0}(1|1)
                \end{pmatrix}

        The construct also support the building of the read-out correlation matrix when the ansatz is included:

        .. math::
            \left<00\right| \rightarrow \left<00\right|\boldsymbol{U}^\dagger\left(\boldsymbol{\theta}=\boldsymbol{0}\right)

        This way some of the gate-error can be build into the read-out correlation matrix.

        #. https://qiskit.org/textbook/ch-quantum-hardware/measurement-error-mitigation.html
        """
        if self.num_qubits > 12:
            raise ValueError("Current implementation does not scale above 12 qubits?")
        if self._do_M_ansatz0:
            ansatz = self.circuit
            # Negate the Hartree-Fock State
            ansatz = ansatz.compose(HartreeFock(self.num_orbs, self.num_elec, self.mapper))
        else:
            ansatz = QuantumCircuit(self.num_qubits)
        M = np.zeros((2**self.num_qubits, 2**self.num_qubits))
        if self._do_M_iqa:
            Pzero = self._sampler_distributions(
                Pauli("Z" * self.num_qubits), [10**-8] * len(ansatz.parameters), ansatz.copy()
            )
            ansatzX = ansatz.copy()
            for i in range(self.num_qubits):
                ansatzX.x(i)
            Pone = self._sampler_distributions(
                Pauli("Z" * self.num_qubits), [10**-8] * len(ansatz.parameters), ansatzX
            )
            for comb in itertools.product([0, 1], repeat=self.num_qubits):
                idx2 = int("".join([str(x) for x in comb]), 2)
                for comb_m in itertools.product([0, 1], repeat=self.num_qubits):
                    P = 1.0
                    idx1 = int("".join([str(x) for x in comb_m]), 2)
                    for idx, (bit, bit_m) in enumerate(zip(comb[::-1], comb_m[::-1])):
                        val = 0.0
                        if bit == 0:
                            Pout = Pzero
                        else:
                            Pout = Pone
                        for bitstring, prob in Pout.items():
                            if bitstring[idx] == str(bit_m):
                                val += prob
                        P *= val
                    M[idx1, idx2] = P
        else:
            for comb in itertools.product([0, 1], repeat=self.num_qubits):
                ansatzX = ansatz.copy()
                idx2 = int("".join([str(x) for x in comb]), 2)
                for i, bit in enumerate(comb):
                    if bit == 1:
                        ansatzX.x(i)
                Px = self._sampler_distributions(
                    Pauli("Z" * self.num_qubits), [10**-8] * len(ansatz.parameters), ansatzX
                )
                for bitstring, prob in Px.items():
                    idx1 = int(bitstring[::-1], 2)
                    M[idx1, idx2] = prob
        self._Minv = np.linalg.inv(M)


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


def make_cliques(paulis: Pauli) -> dict[str, list[str]]:
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
