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
    tUPS,
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
        n_layers: int = 1,
    ) -> None:
        """Interface to Qiskit to use IBM quantum hardware or simulator.

        Args:
            primitive: Qiskit Estimator or Sampler object
            ansatz: Name of ansatz to be used.
            mapper: Qiskit mapper object, e.g. JW or Parity.
        """
        allowed_ansatz = ("UCCSD", "PUCCD", "UCCD", "ErikD", "ErikSD", "HF", "tUPS", "pptUPS")
        if ansatz not in allowed_ansatz:
            raise ValueError("The chosen Ansatz is not availbale. Choose from: ", allowed_ansatz)
        self.ansatz = ansatz
        self._primitive = primitive
        self.mapper = mapper
        self.total_shots_used = 0
        self.total_device_calls = 0
        self.total_paulis_evaluated = 0
        self.n_layers = n_layers

    def construct_circuit(self, num_orbs: int, num_elec: tuple[int, int]) -> None:
        """Construct qiskit circuit.

        Args:
            num_orbs: Number of orbitals in spatial basis.
            num_elec: Number of electrons (alpha, beta).
        """
        self.num_orbs = num_orbs
        self.num_spin_orbs = 2 * num_orbs
        self.num_elec = num_elec
        self.grad_param_R: dict[str, int] = (
            {}
        )  # Contains information about the parameterization needed for gradient evaluations.

        if self.ansatz == "UCCSD":
            self.circuit = UCCSD(
                num_orbs,
                self.num_elec,
                self.mapper,
                initial_state=HartreeFock(
                    num_orbs,
                    self.num_elec,
                    self.mapper,
                ),
            )
        elif self.ansatz == "PUCCD":
            self.circuit = PUCCD(
                num_orbs,
                self.num_elec,
                self.mapper,
                initial_state=HartreeFock(
                    num_orbs,
                    self.num_elec,
                    self.mapper,
                ),
            )
        elif self.ansatz == "UCCD":
            self.circuit = UCC(
                num_orbs,
                self.num_elec,
                "d",
                self.mapper,
                initial_state=HartreeFock(
                    num_orbs,
                    self.num_elec,
                    self.mapper,
                ),
            )
        elif self.ansatz == "ErikD":
            if num_orbs != 2 or self.num_elec != (1, 1):
                raise ValueError(f"Chosen ansatz, {self.ansatz}, only works for (2,2)")
            if isinstance(self.mapper, JordanWignerMapper):
                self.circuit = ErikD_JW()
            elif isinstance(self.mapper, ParityMapper):
                self.circuit = ErikD_Parity()
            else:
                raise ValueError(f"Unsupported mapper, {type(self.mapper)}, for ansatz {self.ansatz}")
        elif self.ansatz == "ErikSD":
            if num_orbs != 2 or self.num_elec != (1, 1):
                raise ValueError(f"Chosen ansatz, {self.ansatz}, only works for (2,2)")
            if isinstance(self.mapper, JordanWignerMapper):
                self.circuit = ErikSD_JW()
            elif isinstance(self.mapper, ParityMapper):
                self.circuit = ErikSD_Parity()
            else:
                raise ValueError(f"Unsupported mapper, {type(self.mapper)}, for ansatz {self.ansatz}")
        elif self.ansatz == "HF":
            self.circuit = HartreeFock(num_orbs, self.num_elec, self.mapper)
        elif self.ansatz == "tUPS":
            self.circuit, self.grad_param_R = tUPS(num_orbs, self.num_elec, self.mapper, self.n_layers, False)
        elif self.ansatz == "pptUPS":
            self.circuit, self.grad_param_R = tUPS(num_orbs, self.num_elec, self.mapper, self.n_layers, True)
        if len(self.grad_param_R) == 0:
            for par in self.circuit.parameters:
                # Default value two
                self.grad_param_R[str(par)] = 2
        if len(self.grad_param_R) != len(self.circuit.parameters):
            raise ValueError(
                f"Number of elements in grad_param_R, {len(self.grad_param_R)}, does not match number of parameters, {len(self.circuit.parameters)}"
            )
        self.param_names = [str(x) for x in self.circuit.parameters]
        for name in self.param_names:
            if name not in self.grad_param_R.keys(): # pylint: disable=consider-iterating-dictionary
                raise ValueError(f"Got parameter name, {name}, that is not in grad_param_R")

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
        if hasattr(self, "distributions"):
            self.distributions.clear()
        self._parameters = parameters.copy()

    def op_to_qbit(self, op: FermionicOperator) -> SparsePauliOp:
        """Fermionic operator to qbit rep.

        Args:
            op: Operator as SlowQuant's FermionicOperator object

        Returns:
            Qubit representation of operator.
        """
        return self.mapper.map(FermionicOp(op.get_qiskit_form(self.num_orbs), 2 * self.num_orbs))

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
        if isinstance(self._primitive, BaseSampler) and custom_parameters is None:
            return self._sampler_quantum_expectation_value(op)
        if isinstance(self._primitive, BaseSampler):
            return self._sampler_quantum_expectation_value_nosave(op, run_parameters)
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
        observables = self.op_to_qbit(op)
        job = self._primitive.run(
            circuits=self.circuit,
            parameter_values=run_parameters,
            observables=observables,
        )
        if hasattr(self._primitive.options, "shots"):
            # Shot-noise simulator
            self.total_shots_used += self._primitive.options.shots * len(observables)
        elif "execution" in self._primitive.options:
            # Device
            self.total_shots_used += self._primitive.options["execution"]["shots"] * len(observables)
        self.total_device_calls += 1
        self.total_paulis_evaluated += len(observables)
        result = job.result()
        values = result.values[0]

        if isinstance(values, complex):
            if abs(values.imag) > 10**-2:
                print("Warning: Complex number detected with Im = ", values.imag)

        return values.real

    def _sampler_quantum_expectation_value(self, op: FermionicOperator) -> float:
        r"""Calculate expectation value of circuit and observables via Sampler.

        Calculated Pauli expectation values will be saved in memory.

        The expectation value over a fermionic operator is calcuated as:

        .. math::
            E = \sum_i^N c_i\left<0\left|P_i\right|0\right>

        With :math:`c_i` being the :math:`i` the coefficient and :math:`P_i` the :math:`i` the Pauli string.

        Args:
            op: SlowQuant fermionic operator.

        Returns:
            Expectation value of operator.
        """
        values = 0.0
        # Map Fermionic to Qubit
        observables = self.op_to_qbit(op)
        # Obtain cliques for operator's Pauli strings
        raw_cliques = make_cliques(observables.paulis)
        # Make sure clique head is in clique Pauli list
        for clique_pauli, clique in raw_cliques.items():
            if clique_pauli not in clique:
                clique.append(clique_pauli)
        cliques = {}

        if not hasattr(self, "distributions"):
            self.distributions: dict[str, float] = {}

        # Check if cliques have already been calculated
        for clique_pauli, clique in raw_cliques.items():
            if clique_pauli not in self.distributions:
                cliques[clique_pauli] = clique

        if len(cliques) != 0:
            # Simulate each clique head with one combined device call
            # and return a list of distributions
            distr = self._one_call_sampler_distributions(PauliList(list(cliques.keys())), self.parameters)

            # Loop over all clique Paul lists to obtain the result for each Pauli string from the clique head distribution
            for nr, clique in enumerate(cliques.values()):
                dist = distr[nr]  # Measured distribution for a given clique head
                for pauli in clique:  # Loop over all clique Pauli strings associated with one clique head
                    result = 0.0
                    for key, value in dist.items():  # build result from quasi-distribution
                        # Here we could check if we want a given key (bitstring) in the result distribution
                        result += value * get_bitstring_sign(Pauli(pauli), key)
                    self.distributions[pauli] = result

        # Loop over all Pauli strings in observable and build final result with coefficients
        for pauli, coeff in zip(observables.paulis, observables.coeffs):
            values += self.distributions[str(pauli)] * coeff

        if isinstance(values, complex):
            if abs(values.imag) > 10**-2:
                print("Warning: Complex number detected with Im = ", values.imag)

        return values.real

    def _sampler_quantum_expectation_value_nosave(
        self, op: FermionicOperator, run_parameters: list[float]
    ) -> float:
        r"""Calculate expectation value of circuit and observables via Sampler.

        Calling this function will not use any pre-calculated Pauli expectaion values.
        Nor will it save any of the calculated Pauli expectation values.

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
        # Map Fermionic to Qubit
        observables = self.op_to_qbit(op)
        # Obtain cliques for operator's Pauli strings
        cliques = make_cliques(observables.paulis)

        # Simulate each clique head with one combined device call
        # and return a list of distributions
        distr = self._one_call_sampler_distributions(PauliList(list(cliques.keys())), run_parameters)
        distributions = {}
        # Loop over all clique Paul lists to obtain the result for each Pauli string from the clique head distribution
        for nr, clique in enumerate(cliques.values()):
            dist = distr[nr]  # Measured distribution for a given clique head
            for pauli in clique:  # Loop over all clique Pauli strings associated with one clique head
                result = 0.0
                for key, value in dist.items():  # build result from quasi-distribution
                    # Here we could check if we want a given key (bitstring) in the result distribution
                    result += value * get_bitstring_sign(Pauli(pauli), key)
                distributions[pauli] = result

        # Loop over all Pauli strings in observable and build final result with coefficients
        for pauli, coeff in zip(observables.paulis, observables.coeffs):
            values += distributions[str(pauli)] * coeff

        if isinstance(values, complex):
            if abs(values.imag) > 10**-2:
                print("Warning: Complex number detected with Im = ", values.imag)

        return values.real

    def _one_call_sampler_distributions(
        self, paulis: PauliList, run_parameters: list[float]
    ) -> list[dict[str, float]]:
        r"""Get results from a sampler distribution for several Pauli strings.

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
        num_paulis = len(paulis)
        circuits = [None] * num_paulis

        # Create QuantumCircuits
        for nr, pauli in enumerate(paulis):
            ansatz_w_obs = self.circuit.compose(to_CBS_measurement(pauli))
            ansatz_w_obs.measure_all()
            circuits[nr] = ansatz_w_obs

        # Run sampler
        job = self._primitive.run(circuits, parameter_values=[run_parameters] * num_paulis)
        if hasattr(self._primitive.options, "shots"):
            # Shot-noise simulator
            self.total_shots_used += self._primitive.options.shots * num_paulis
        elif "execution" in self._primitive.options:
            # Device
            self.total_shots_used += self._primitive.options["execution"]["shots"] * num_paulis
        self.total_device_calls += 1
        self.total_paulis_evaluated += num_paulis

        # Get quasi-distribution in binary probabilities
        distr = [res.binary_probabilities() for res in job.result().quasi_dists]
        return distr

    def _sampler_distributions(self, pauli: PauliList, run_parameters: list[float]) -> dict[str, float]:
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
        ansatz_w_obs = self.circuit.compose(to_CBS_measurement(pauli))
        ansatz_w_obs.measure_all()

        # Run sampler
        job = self._primitive.run(ansatz_w_obs, parameter_values=run_parameters)
        if hasattr(self._primitive.options, "shots"):
            # Shot-noise simulator
            self.total_shots_used += self._primitive.options.shots
        elif "execution" in self._primitive.options:
            # Device
            self.total_shots_used += self._primitive.options["execution"]["shots"]
        self.total_device_calls += 1

        # Get quasi-distribution in binary probabilities
        distr = job.result().quasi_dists[0].binary_probabilities()
        return distr


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
