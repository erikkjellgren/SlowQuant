import copy
import itertools
import math
from typing import Any

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
    tUPS,
)
from slowquant.qiskit_interface.util import (
    Clique,
    correct_distribution,
    get_bitstring_sign,
    make_cliques,
    to_CBS_measurement,
)


class QuantumInterface:
    """Quantum interface class.

    This class handles the interface with qiskit and the communication with quantum hardware.
    """

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        primitive: BaseEstimator | BaseSampler,
        ansatz: str,
        mapper: FermionicMapper,
        ansatz_options: dict[str, Any] = {},
        shots: None | int = None,
        max_shots_per_run: int = 100000,
        do_M_mitigation: bool = False,
        do_M_iqa: bool = False,
        do_M_ansatz0: bool = False,
    ) -> None:
        """Interface to Qiskit to use IBM quantum hardware or simulator.

        Args:
            primitive: Qiskit Estimator or Sampler object.
            ansatz: Name of ansatz to be used.
            mapper: Qiskit mapper object, e.g. JW or Parity.
            ansatz_options: Ansatz options.
            mapper: Qiskit mapper object.
            shots: Number of shots. If not specified use shotnumber from primitive (default).
            max_shots_per_run: Maximum number of shots allowed in a single run. Set to 100000 per IBM machines.
            do_M_mitigation: Do error mitigation via read-out correlation matrix.
            do_M_iqa: Use independent qubit approximation when constructing the read-out correlation matrix.
            do_M_ansatz0: Use the ansatz with theta=0 when constructing the read-out correlation matrix
        """
        allowed_ansatz = ("UCCSD", "PUCCD", "UCCD", "ErikD", "ErikSD", "HF", "tUPS")
        if ansatz not in allowed_ansatz:
            raise ValueError("The chosen Ansatz is not available. Choose from: ", allowed_ansatz)
        self.ansatz = ansatz
        self._primitive = primitive
        self.mapper = mapper
        self.max_shots_per_run = max_shots_per_run
        self.shots = shots
        self._do_M_mitigation = do_M_mitigation
        self._do_M_iqa = do_M_iqa
        self._do_M_ansatz0 = do_M_ansatz0
        self._Minv = None
        self.total_shots_used = 0
        self.total_device_calls = 0
        self.total_paulis_evaluated = 0
        self.ansatz_options = ansatz_options

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
            if len(self.ansatz_options) != 0:
                raise ValueError(f"No options available for UCCSD got {self.ansatz_options}")
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
            if len(self.ansatz_options) != 0:
                raise ValueError(f"No options available for PUCCD got {self.ansatz_options}")
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
            if len(self.ansatz_options) != 0:
                raise ValueError(f"No options available for UCCD got {self.ansatz_options}")
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
            if len(self.ansatz_options) != 0:
                raise ValueError(f"No options available for ErikD got {self.ansatz_options}")
            if num_orbs != 2 or self.num_elec != (1, 1):
                raise ValueError(f"Chosen ansatz, {self.ansatz}, only works for (2,2)")
            if isinstance(self.mapper, JordanWignerMapper):
                self.circuit = ErikD_JW()
            elif isinstance(self.mapper, ParityMapper):
                self.circuit = ErikD_Parity()
            else:
                raise ValueError(f"Unsupported mapper, {type(self.mapper)}, for ansatz {self.ansatz}")
        elif self.ansatz == "ErikSD":
            if len(self.ansatz_options) != 0:
                raise ValueError(f"No options available for ErikSD got {self.ansatz_options}")
            if num_orbs != 2 or self.num_elec != (1, 1):
                raise ValueError(f"Chosen ansatz, {self.ansatz}, only works for (2,2)")
            if isinstance(self.mapper, JordanWignerMapper):
                self.circuit = ErikSD_JW()
            elif isinstance(self.mapper, ParityMapper):
                self.circuit = ErikSD_Parity()
            else:
                raise ValueError(f"Unsupported mapper, {type(self.mapper)}, for ansatz {self.ansatz}")
        elif self.ansatz == "HF":
            if len(self.ansatz_options) != 0:
                raise ValueError(f"No options available for HF got {self.ansatz_options}")
            self.circuit = HartreeFock(num_orbs, self.num_elec, self.mapper)
        elif self.ansatz == "tUPS":
            self.circuit, self.grad_param_R = tUPS(num_orbs, self.num_elec, self.mapper, self.ansatz_options)

        # Check that R parameter for gradient is consistent with the paramter names.
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
            if name not in self.grad_param_R.keys():  # pylint: disable=consider-iterating-dictionary
                raise ValueError(
                    f"Got parameter name, {name}, that is not in grad_param_R, {self.grad_param_R}"
                )

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
        if hasattr(self, "cliques"):
            # The distributions should only reset if the parameters are actually changed.
            if not np.array_equal(self._parameters, parameters):
                self.cliques = Clique()
        self._parameters = parameters.copy()

    @property
    def shots(self) -> int | None:
        """Get number of shots.

        Returns:
            Number of shots.
        """
        return self._shots

    @shots.setter
    def shots(
        self,
        shots: int | None,
    ) -> None:
        """Set number of shots.

        Args:
            shots: Number of shots.
        """
        # IMPORTANT: Shot number in primitive gets always overwritten if a shot number is defined in QI!
        self._circuit_multipl = 1
        # Get shot number form primitive if none defined
        if shots is None:
            if hasattr(self._primitive.options, "shots"):
                # Shot-noise simulator
                self._shots = self._primitive.options.shots
                print("Number of shots has been set to value defined in primitive option: ", self._shots)
            elif "execution" in self._primitive.options:
                # Device
                self._shots = self._primitive.options["execution"]["shots"]
                print("Number of shots has been set to value defined in primitive option: ", self._shots)
            else:
                print(
                    "WARNING: No number of shots option found in primitive. Ideal simulator is assumed and shots set to None / Zero"
                )
                self._shots = None
        else:
            self._shots = shots
        # Check if shot number is allowed
        if self._shots is not None:
            if self._shots > self.max_shots_per_run:
                if isinstance(self._primitive, BaseEstimator):
                    self._primitive.set_options(shots=self.max_shots_per_run)
                    print(
                        "WARNING: Number of shots specified exceed possibility for Estimator. Number of shots is set to ",
                        self.max_shots_per_run,
                    )
                else:
                    print("Number of requested shots exceed the limit of ", self.max_shots_per_run)
                    # Get number of circuits needed to fulfill shot number
                    self._circuit_multipl = math.floor(self._shots / self.max_shots_per_run)
                    self._primitive.set_options(shots=self.max_shots_per_run)
                    print(
                        "Maximum shots are used and additional",
                        self._circuit_multipl - 1,
                        "circuits per Pauli string are appended to circumvent limit.",
                    )
                    if self._shots % self.max_shots_per_run != 0:
                        self._shots = self._circuit_multipl * self.max_shots_per_run
                        print(
                            "WARNING: Requested shots must be multiple of max_shots_per_run. Total shots has been adjusted to ",
                            self._shots,
                        )
            else:
                if shots is not None:  # shots were defined in input and have to be written in primitive
                    self._primitive.set_options(shots=self._shots)
                    print(
                        "Number of shots in primitive has been adapted as specified for QI to ", self._shots
                    )

    @property
    def max_shots_per_run(self) -> int:
        """Get max number of shots per run.

        Returns:
            Max number of shots pers run.
        """
        return self._max_shots_per_run

    @max_shots_per_run.setter
    def max_shots_per_run(
        self,
        max_shots_per_run: int,
    ) -> None:
        """Set max number of shots per run.

        Args:
            max_shots_per_run: Max number of shots pers run.
        """
        self._max_shots_per_run = max_shots_per_run
        # Redo shot check with new max_shots_per_run
        if hasattr(self, "_shots"):  # Check if it is initialization
            self.shots = self._shots

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
            custom_parameters: optional custom circuit parameters.

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
        if self.shots is not None:  # check if ideal simulator
            self.total_shots_used += self.shots * len(observables)
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

        if not hasattr(self, "cliques"):
            self.cliques = Clique()

        new_heads = self.cliques.add_paulis([str(x) for x in observables.paulis])

        if len(new_heads) != 0:
            # Check if error mitigation is requested and if read-out matrix already exists.
            if self._do_M_mitigation and self._Minv is None:
                self._make_Minv()

            # Simulate each clique head with one combined device call
            # and return a list of distributions
            distr = self._one_call_sampler_distributions(PauliList(new_heads), self.parameters, self.circuit)
            if self._do_M_mitigation:  # apply error mitigation if requested
                for i, dist in enumerate(distr):
                    distr[i] = correct_distribution(dist, self._Minv)
            self.cliques.update_distr(new_heads, distr)

        # Loop over all Pauli strings in observable and build final result with coefficients
        for pauli, coeff in zip(observables.paulis, observables.coeffs):
            result = 0.0
            for key, value in self.cliques.get_distr(
                str(pauli)
            ).items():  # build result from quasi-distribution
                result += value * get_bitstring_sign(Pauli(pauli), key)
            values += result * coeff

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
        distr = self._one_call_sampler_distributions(
            PauliList(list(cliques.keys())), run_parameters, self.circuit
        )
        distributions = {}

        # Check if error mitigation is requested and if read-out matrix already exists.
        if self._do_M_mitigation and self._Minv is None:
            self._make_Minv()

        # Loop over all clique Paul lists to obtain the result for each Pauli string from the clique head distribution
        for nr, clique in enumerate(cliques.values()):
            dist = distr[nr]  # Measured distribution for a given clique head
            if self._do_M_mitigation:  # apply error mitigation if requested
                dist = correct_distribution(dist, self._Minv)
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

    def quantum_variance(
        self,
        op: FermionicOperator,
        do_cliques: bool = True,
        no_coeffs: bool = False,
        custom_parameters: list[float] | None = None,
    ) -> float:
        """Calculate variance (std**2) of expectation value of circuit and observables.

        Args:
            op: SlowQuant fermionic operator.
            do_cliques: boolean if cliques are used.
            no_coeffs: boolean if coefficients of each Pauli string are used or all st to 1.
            custom_parameters: optional custom circuit parameters.

        Returns:
            Variance of expectation value.
        """
        if isinstance(self._primitive, BaseEstimator):
            raise ValueError("This function does not work with Estimator.")
        if custom_parameters is None:
            run_parameters = self.parameters
        else:
            run_parameters = custom_parameters

        observables = self.op_to_qbit(op)

        if do_cliques:
            if not hasattr(self, "cliques"):
                self.cliques = Clique()

            new_heads = self.cliques.add_paulis([str(x) for x in observables.paulis])

            if len(new_heads) != 0:
                # Check if error mitigation is requested and if read-out matrix already exists.
                if self._do_M_mitigation and self._Minv is None:
                    self._make_Minv()

                # Simulate each clique head with one combined device call
                # and return a list of distributions
                distr = self._one_call_sampler_distributions(
                    PauliList(new_heads), run_parameters, self.circuit
                )
                if self._do_M_mitigation:  # apply error mitigation if requested
                    for i, dist in enumerate(distr):
                        distr[i] = correct_distribution(dist, self._Minv)
                self.cliques.update_distr(new_heads, distr)

        # Loop over all Pauli strings in observable and build final result with coefficients
        result = 0.0
        for pauli, coeff in zip(observables.paulis, observables.coeffs):
            if no_coeffs:
                coeff = 1
            # Get distribution from cliques
            if do_cliques:
                dist = self.cliques.get_distr(str(pauli))
                # Calculate p1: Probability of measuring one
                p1 = 0.0
                for key, value in dist.items():
                    if get_bitstring_sign(pauli, key) == 1:
                        p1 += value
            else:
                p1 = self._sampler_distribution_p1(pauli, run_parameters)
            if self.shots is None:
                sigma_p = 2 * np.abs(coeff.real) * ((p1 - p1**2) ** (1 / 2))
            else:
                sigma_p = 2 * np.abs(coeff.real) * ((p1 - p1**2) ** (1 / 2)) / (self.shots ** (1 / 2))
            result += sigma_p**2
        return result

    def _one_call_sampler_distributions(
        self,
        paulis: PauliList | Pauli,
        run_parameters: list[list[float]] | list[float],
        circuits_in: list[QuantumCircuit] | QuantumCircuit,
    ) -> list[dict[str, float]]:
        r"""Get results from a sampler distribution for several Pauli strings measured on several circuits.

        The expectation value of a Pauli string is calcuated as:

        .. math::
            E = \sum_i^N p_i\left<b_i\left|P\right|b_i\right>

        With :math:`p_i` being the :math:`i` th probability and :math:`b_i` being the `i` th bit-string.

        Args:
            paulis: (List of) Pauli strings to measure.
            run_paramters: List of parameters of each circuit.
            circuits_in: List of circuits

        Returns:
            Array of quasi-distributions in order of all circuits results for a given Pauli String first.
            E.g.: [PauliString[0] for Circuit[0], PauliString[0] for Circuit[1], ...]
        """
        if isinstance(paulis, Pauli):
            paulis = PauliList(paulis)
        num_paulis = len(paulis)
        if isinstance(circuits_in, QuantumCircuit):
            circuits_in = [circuits_in]
        num_circuits = len(circuits_in)

        circuits = [None] * (num_paulis * num_circuits)
        # Create QuantumCircuits
        for nr_pauli, pauli in enumerate(paulis):
            pauli_circuit = to_CBS_measurement(pauli)
            for nr_circuit, circuit in enumerate(circuits_in):
                ansatz_w_obs = circuit.compose(pauli_circuit)
                ansatz_w_obs.measure_all()
                circuits[(nr_circuit + (nr_pauli * num_circuits))] = ansatz_w_obs
        circuits = circuits * self._circuit_multipl

        # Run sampler
        if num_circuits == 1:
            parameter_values = [run_parameters] * (num_paulis * self._circuit_multipl)
        else:
            parameter_values = run_parameters * (num_paulis * self._circuit_multipl)  # type: ignore
        job = self._primitive.run(circuits, parameter_values=parameter_values)
        if self.shots is not None:  # check if ideal simulator
            self.total_shots_used += self.shots * num_paulis * num_circuits * self._circuit_multipl
        self.total_device_calls += 1
        self.total_paulis_evaluated += num_paulis * num_circuits * self._circuit_multipl

        # Get quasi-distribution in binary probabilities
        distr = [res.binary_probabilities() for res in job.result().quasi_dists]
        if self._circuit_multipl == 1:
            return distr

        # Post-process multiple circuit runs together
        length = num_paulis * num_circuits
        dist_combined = copy.deepcopy(distr[:length])
        for nr, dist in enumerate(distr[length:]):
            for key, value in dist.items():
                dist_combined[nr % length][key] = value + dist_combined[nr % length].get(key, 0)
        for dist in dist_combined:
            for key in dist:
                dist[key] /= self._circuit_multipl
        return dist_combined

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
            custom_circ: Specific circuit to run.

        Returns:
            Quasi-distributions.
        """
        if self._circuit_multipl > 1:
            print(
                "WARNING: The chosen function does not allow for appending circuits. Choose _one_call_sampler_distributions instead."
            )
            print(
                "Simulation will be run without appending circuits with ", self.max_shots_per_run, " shots."
            )
            shots: int | None = self.max_shots_per_run
        else:
            shots = self.shots

        # Create QuantumCircuit
        if custom_circ is None:
            ansatz_w_obs = self.circuit.compose(to_CBS_measurement(pauli))
        else:
            ansatz_w_obs = custom_circ.compose(to_CBS_measurement(pauli))
        ansatz_w_obs.measure_all()

        # Run sampler
        job = self._primitive.run(ansatz_w_obs, parameter_values=run_parameters)
        if shots is not None:  # check if ideal simulator
            self.total_shots_used += shots
        self.total_device_calls += 1
        self.total_paulis_evaluated += 1

        # Get quasi-distribution in binary probabilities
        distr = job.result().quasi_dists[0].binary_probabilities()
        return distr

    def _sampler_distribution_p1(
        self, pauli: Pauli, run_parameters: list[float], custom_circ: None | QuantumCircuit = None
    ) -> float:
        """Sample the probability of measuring one for a given Pauli string.

            pauli: Pauli string.
            run_paramters: Ansatz parameters.
            custom_circ: Custom circuit to run.

        Returns:
            p1 probability.
        """
        # Get quasi-distribution in binary probabilities
        distr = self._sampler_distributions(pauli, run_parameters, custom_circ)

        p1 = 0.0
        for key, value in distr.items():
            if get_bitstring_sign(pauli, key) == 1:
                p1 += value
        return p1

    def _make_Minv(self) -> None:
        r"""Make inverse of read-out correlation matrix with one device call.

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
        print("Measuring error mitigation read-out matrix.")
        if self.num_qubits > 12:
            raise ValueError("Current implementation does not scale above 12 qubits?")
        if "transpilation" in self._primitive.options:
            if self._primitive.options["transpilation"]["initial_layout"] is None:
                raise ValueError(
                    "Doing read-out correlation matrix requires qubits to be fixed. Got ['transpilation']['initial_layout'] as None"
                )
        else:
            print("No transpilation option found in primitive. Run via simulator is assumed.")
        if self._do_M_ansatz0:
            ansatz = self.circuit
            # Negate the Hartree-Fock State
            ansatz = ansatz.compose(HartreeFock(self.num_orbs, self.num_elec, self.mapper))
        else:
            ansatz = QuantumCircuit(self.num_qubits)
        M = np.zeros((2**self.num_qubits, 2**self.num_qubits))
        if self._do_M_iqa:
            ansatzX = ansatz.copy()
            for i in range(self.num_qubits):
                ansatzX.x(i)
            [Pzero, Pone] = self._one_call_sampler_distributions(
                Pauli("Z" * self.num_qubits),
                [[10**-8] * len(ansatz.parameters)] * 2,
                [ansatz.copy(), ansatzX],
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
            ansatz_list = [None] * 2**self.num_qubits
            for nr, comb in enumerate(itertools.product([0, 1], repeat=self.num_qubits)):
                ansatzX = ansatz.copy()
                idx2 = int("".join([str(x) for x in comb]), 2)
                for i, bit in enumerate(comb):
                    if bit == 1:
                        ansatzX.x(i)
                # Make list of custom ansatz
                ansatz_list[nr] = ansatzX
            # Simulate all elements with one device call
            Px_list = self._one_call_sampler_distributions(
                Pauli("Z" * self.num_qubits),
                [[10**-8] * len(ansatz.parameters)] * len(ansatz_list),
                ansatz_list,
            )
            # Construct M
            for idx2, Px in enumerate(Px_list):
                for bitstring, prob in Px.items():
                    idx1 = int(bitstring[::-1], 2)
                    M[idx1, idx2] = prob
        self._Minv = np.linalg.inv(M)
