import qiskit_nature.second_q.mappers as Mappers
from qiskit_nature.second_q.circuit.library import PUCCD, UCCSD, HartreeFock
from qiskit_nature.second_q.operators import FermionicOp

from slowquant.qiskit_interface.base import FermionicOperator


class QuantumInterface:
    def __init__(
        self,
        estimator,
        ansatz: str,
        mapper: Mappers,
    ) -> None:
        """
        Interface to Qiskit to use IBM quantum hardware or simulator.

        Args:
            estimator: Qiskit Estimator object
            ansatz: Name of qiskit ansatz to be used. Currenly supported: UCCSD and PUCCD
            mapper: Qiskit mapper object, e.g. JW or Parity
        """
        allowed_ansatz = ["UCCSD", "PUCCD"]
        if not ansatz in allowed_ansatz:
            raise ValueError("The chosen Ansatz is not availbale. Choose from: ", allowed_ansatz)
        self.ansatz = ansatz
        self.estimator = estimator
        self.mapper = mapper

    def construct_circuit(self, num_orbs: int, num_parts: int) -> None:
        """
        Construct qiskit circuit

        Args:
            num_orbs: number of orbitals
            num_parts: number of particles/electrons
        """

        self.num_orbs = (
            num_orbs  # that might be a dirty and stupid solution for the num_orbs problem. revisit it!
        )

        if self.ansatz == "UCCSD":
            self.circuit = UCCSD(
                num_orbs,
                (num_parts // 2, num_parts // 2),
                self.mapper,
                initial_state=HartreeFock(
                    num_orbs,
                    (num_parts // 2, num_parts // 2),
                    self.mapper,
                ),
            )
        elif self.ansatz == "PUCCD":
            self.circuit = PUCCD(
                num_orbs,
                (num_parts // 2, num_parts // 2),
                self.mapper,
                initial_state=HartreeFock(
                    num_orbs,
                    (num_parts // 2, num_parts // 2),
                    self.mapper,
                ),
            )

        # Set parameter to HarteeFock
        self._parameters = [0.0] * self.circuit.num_parameters

    @property
    def parameters(self) -> list[float]:
        return self._parameters

    @parameters.setter
    def parameters(
        self,
        parameters: list[float],
    ) -> None:
        if len(parameters) != self.circuit.num_parameters:
            raise ValueError(
                "The length of the parameter list does not fit the chosen circuit for the Ansatz ",
                self.ansatz,
            )
        self._parameters = parameters.copy()

    def op_to_qbit(self, op: FermionicOperator):
        """
        Fermionic operator to qbit rep

        Args:
            op: Operator as SlowQuant's FermionicOperator object
        """
        return self.mapper.map(FermionicOp(op.get_qiskit_form(self.num_orbs), 2 * self.num_orbs))

    def quantum_expectation_value(
        self, op: FermionicOperator, custom_parameters: list[float] | None = None
    ) -> float:
        """
        Calculate expectation value of circuit from vqe result  with op operator

        Args:
            op: Operator as SlowQuant's FermionicOperator object
        """
        if custom_parameters is None:
            run_parameters = self.parameters
        else:
            run_parameters = custom_parameters
        job = self.estimator.run(
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
