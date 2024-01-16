import qiskit_nature.second_q.mappers as Mappers
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.problems import ElectronicStructureResult


class QuantumInterface:
    def __init__(
        self,
        estimator,
        vqe: ElectronicStructureResult,  # ansatz: string
        mapper: Mappers,
    ) -> None:
        """
        Interface to IBM quantum hardware or simulator.
        """
        self.estimator = estimator
        self.vqe = vqe
        self.mapper = mapper
        self.num_orbs = vqe.raw_result.optimal_circuit.num_qubits

    # construct circuit: args: ansatz string, num_spatial orbitals, num_particle

    # update parameter: parameter: list of floats

    def op_to_qbit(self, op):
        """
        Fermionic operator to qbit rep
        """
        return self.mapper.map(FermionicOp(op.get_qiskit_form(self.num_orbs)))

    def quantum_expectation_value(self, op):
        """
        Calculate expectation value of circuit from vqe result  with op operator
        """

        job = self.estimator.run(
            circuits=self.vqe.raw_result.optimal_circuit,  # constructed circuit self
            parameter_values=self.vqe.raw_result.optimal_point,  # parameter self
            observables=self.op_to_qbit(op),
        )
        result = job.result()
        values = result.values[0]

        if isinstance(values, complex):
            if abs(values.imag) > 0:
                print("Warning: Complex number detected with Im = ", values.imag)  # threshold 10**-2

        return values.real


def to_qbit(op, mapper, num_orbs):
    return mapper.map(FermionicOp(op.get_qiskit_form(num_orbs)))
