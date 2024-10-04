# pylint: disable=too-many-lines
import copy
import itertools
import math
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister
from qiskit.primitives import (
    BaseEstimator,
    BaseEstimatorV2,
    BaseSamplerV1,
    BaseSamplerV2,
)
from qiskit.providers import Backend
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.operators import FermionicOp

from slowquant.qiskit_interface.custom_ansatz import SDSfUCC, fUCC, tUPS
from slowquant.qiskit_interface.util import (
    Clique,
    correct_distribution,
    get_bitstring_sign,
    get_determinant_reference,
    get_determinant_superposition_reference,
    get_reordering_sign,
    postselection,
    to_CBS_measurement,
)
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator


class QuantumInterface:
    """Quantum interface class.

    This class handles the interface with qiskit and the communication with quantum hardware.
    """

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        primitive: BaseEstimator | BaseSamplerV1 | BaseSamplerV2,
        ansatz: str | QuantumCircuit,
        mapper: FermionicMapper,
        ISA: bool = False,
        pass_manager: None | tuple[PassManager, Backend] = None,
        ansatz_options: dict[str, Any] | None = None,
        shots: None | int = None,
        max_shots_per_run: int = 100000,
        do_M_mitigation: bool = False,
        do_M_ansatz0: bool = False,
        do_postselection: bool = True,
    ) -> None:
        """Interface to Qiskit to use IBM quantum hardware or simulator.

        Args:
            primitive: Qiskit Estimator or Sampler object.
            ansatz: Name of ansatz to be used.
            mapper: Qiskit mapper object, e.g. JW or Parity.
            ansatz_options: Ansatz options.
            mapper: Qiskit mapper object.
            ISA: Use ISA for submitting to IBM quantum. Locally transpiling is performed.
            pass_manager: Tuple of custom PassManager and backend for transpilation.
            shots: Number of shots. None means ideal simulator.
            max_shots_per_run: Maximum number of shots allowed in a single run. Set to 100000 per IBM machines.
            do_M_mitigation: Do error mitigation via read-out correlation matrix.
            do_M_ansatz0: Use the ansatz with theta=0 when constructing the read-out correlation matrix.
            do_postselection: Use postselection to preserve number of particles in the computational basis.
        """
        if ansatz_options is None:
            ansatz_options = {}
        allowed_ansatz = (
            "fpUCCD",
            "fUCCD",
            "tUPS",
            "fUCCSD",
            "QNP",
            "kSAfUpCCGSD",
            "SDSfUCCSD",
            "kSASDSfUpCCGSD",
            "fUCC",
            "SDSfUCC",
            "HF",
        )
        if not isinstance(ansatz, QuantumCircuit) and ansatz not in allowed_ansatz:
            raise ValueError(
                "The chosen Ansatz is not available. Choose from: ",
                allowed_ansatz,
                "or pass custom QuantumCircuit object",
            )
        if isinstance(primitive, BaseEstimatorV2):
            raise TypeError("EstimatorV2 is not currently supported.")
        if isinstance(primitive, BaseSamplerV2):
            print("WARNING: Using SamplerV2 is an experimental feature.")
        self.ansatz = ansatz
        self._transpiled = False  # Check if circuit has been transpiled
        self.max_shots_per_run = max_shots_per_run
        self._primitive = primitive
        self.ISA = ISA
        self.shots = shots
        self.mapper = mapper
        self.pass_manager = pass_manager
        self.do_M_mitigation = do_M_mitigation
        self.do_M_ansatz0 = do_M_ansatz0
        self._Minv = None
        self.total_shots_used = 0
        self.total_device_calls = 0
        self.total_paulis_evaluated = 0
        self.ansatz_options = ansatz_options
        self.do_postselection = do_postselection
        self._save_layout = False
        self._save_paulis = True  # hard switch to stop using Pauli saving (debugging tool).
        self._do_cliques = True  # hard switch to stop using QWC (debugging tool).
        self._M_shots = None  # define a separate number of shots for M

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

        # State prep circuit
        if isinstance(self.ansatz, QuantumCircuit):
            self.state_circuit: QuantumCircuit = QuantumCircuit(
                self.ansatz.num_qubits
            )  # empty state as custom circuit is passed
        elif self.ansatz == "tUPS" and "do_pp" in self.ansatz_options.keys() and self.ansatz_options["do_pp"]:
            # HF in pp-tUPS ordering
            if not isinstance(self.mapper, JordanWignerMapper):
                raise ValueError(f"pp-tUPS only implemented for JW mapper, got: {type(self.mapper)}")
            if np.sum(num_elec) != num_orbs:
                raise ValueError(
                    f"pp-tUPS only implemented for number of electrons and number of orbitals being the same, got: ({np.sum(num_elec)}, {num_orbs}), (elec, orbs)"
                )
            self.state_circuit = QuantumCircuit(2 * num_orbs)
            for p in range(0, 2 * num_orbs):
                if p % 2 == 0:
                    self.state_circuit.x(p)
        else:
            self.state_circuit = HartreeFock(num_orbs, num_elec, self.mapper)
        self.num_qubits = self.state_circuit.num_qubits

        # Ansatz Circuit
        if isinstance(self.ansatz, QuantumCircuit):
            print("QI was initialized with a custom QuantumCircuit object.")
            self.circuit = self.ansatz
        elif self.ansatz == "fpUCCD":
            self.ansatz_options["pD"] = True
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.circuit, self.grad_param_R = fUCC(num_orbs, self.num_elec, self.mapper, self.ansatz_options)
        elif self.ansatz == "fUCCD":
            self.ansatz_options["D"] = True
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.circuit, self.grad_param_R = fUCC(num_orbs, self.num_elec, self.mapper, self.ansatz_options)
        elif self.ansatz == "HF":
            if len(self.ansatz_options) != 0:
                raise ValueError(f"No options available for HF got {self.ansatz_options}")
            self.circuit = QuantumCircuit(self.num_qubits)  # empty ansatz circuit
        elif self.ansatz == "tUPS":
            self.circuit, self.grad_param_R = tUPS(num_orbs, self.num_elec, self.mapper, self.ansatz_options)
        elif self.ansatz == "QNP":
            self.ansatz_options["do_qnp"] = True
            self.circuit, self.grad_param_R = tUPS(num_orbs, self.num_elec, self.mapper, self.ansatz_options)
        elif self.ansatz == "fUCCSD":
            self.ansatz_options["S"] = True
            self.ansatz_options["D"] = True
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.circuit, self.grad_param_R = fUCC(num_orbs, self.num_elec, self.mapper, self.ansatz_options)
        elif self.ansatz == "kSAfUpCCGSD":
            self.ansatz_options["SAGS"] = True
            self.ansatz_options["GpD"] = True
            self.circuit, self.grad_param_R = fUCC(num_orbs, self.num_elec, self.mapper, self.ansatz_options)
        elif self.ansatz == "SDSfUCCSD":
            self.ansatz_options["D"] = True
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.circuit, self.grad_param_R = SDSfUCC(
                num_orbs, self.num_elec, self.mapper, self.ansatz_options
            )
        elif self.ansatz == "kSASDSfUpCCGSD":
            self.ansatz_options["GpD"] = True
            self.circuit, self.grad_param_R = SDSfUCC(
                num_orbs, self.num_elec, self.mapper, self.ansatz_options
            )
        elif self.ansatz == "fUCC":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.circuit, self.grad_param_R = fUCC(num_orbs, self.num_elec, self.mapper, self.ansatz_options)
        elif self.ansatz == "SDSfUCC":
            if "n_layers" not in self.ansatz_options.keys():
                # default option
                self.ansatz_options["n_layers"] = 1
            self.circuit, self.grad_param_R = SDSfUCC(
                num_orbs, self.num_elec, self.mapper, self.ansatz_options
            )
        else:
            raise ValueError(f"Unknown ansatz: {self.ansatz}")

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

        if not hasattr(self, "_parameters"):
            # Set parameter to HarteeFock
            self._parameters = [0.0] * self.circuit.num_parameters

    @property
    def ISA(self) -> bool:
        """Get ISA setting.

        Returns:
            ISA setting
        """
        return self._ISA

    @ISA.setter
    def ISA(self, ISA: bool) -> None:
        """Set ISA and handle transpile arguments.

        Args:
            ISA: ISA bool
        """
        if isinstance(self._primitive, BaseSamplerV2):
            print("ISA is set automatically to True for SamplerV2.")
            self._ISA: bool = True
        else:
            self._ISA = ISA

        if self._ISA:
            # Get backend from primitive.
            if hasattr(self._primitive, "_backend"):
                self._primitive_backend = self._primitive._backend  # pylint: disable=protected-access
            else:
                self._primitive_backend = None

            # Get optimization level from backend. Only for v1 primitives.
            self._primitive_level = 3
            if hasattr(self._primitive, "_transpile_options") and hasattr(
                self._primitive._transpile_options, "optimization_level"  # pylint: disable=protected-access
            ):
                self._primitive_level = (
                    self._primitive._transpile_options[  # pylint: disable=protected-access
                        "optimization_level"
                    ]
                )
            elif hasattr(self._primitive, "options"):
                if hasattr(self._primitive.options, "optimization_level"):
                    self._primitive_level = self._primitive.options["optimization_level"]

            # Check if circuit has been transpiled
            # In case of switching to ISA in later workflow
            if not self._transpiled and hasattr(self, "circuit"):
                if self.pass_manager is None:
                    # We have switched to ISA and had no PassManager before.
                    # This will initialize the standard internal transpilation.
                    self.pass_manager = None
                elif self._internal_pm:
                    # We have used internal PassManager before but re-transpilation was requested (probs via change_primitive in WF)
                    self.pass_manager = None
                self.construct_circuit(self.num_orbs, self.num_elec)

    @property
    def pass_manager(self) -> None | PassManager:
        """Get PassManager.

        Returns:
            PassManager
        """
        return self._pass_manager

    @pass_manager.setter
    def pass_manager(self, pass_manager: None | tuple[PassManager, Backend]) -> None:
        """Set PassManager.

        Args:
            pass_manager: PassManager object from Qiskit.
        """
        self._internal_pm = False
        if not self.ISA and isinstance(pass_manager, tuple):
            raise ValueError("You need to enable ISA if you want to use a custom PassManager.")
        # Custom pass manager
        if self.ISA and isinstance(pass_manager, tuple):
            print(
                "Warning: Using custom pass manager is an experimental feature if used together with the quantum_expectation_value_csfs function, as requiered for SA wave functions."
            )
            self._pass_manager: None | PassManager = pass_manager[0]
            self._layoutfree_pm = generate_preset_pass_manager(
                self._primitive_level,
                backend=pass_manager[1],
                layout_method="trivial",
            )
        # Default pass manager
        elif self.ISA and not isinstance(pass_manager, tuple):
            self._internal_pm = True
            self._pass_manager = generate_preset_pass_manager(
                self._primitive_level, backend=self._primitive_backend
            )
            self._layoutfree_pm = generate_preset_pass_manager(
                self._primitive_level,
                backend=self._primitive_backend,
                layout_method="trivial",
            )
            print(
                f"You selected ISA but did not pass a PassManager. Standard internal transpilation will use backend {self._primitive_backend} with optimization level {self._primitive_level}"
            )
        else:
            self._pass_manager = None

        # Check if circuit has been set
        # In case of switching to new PassManager in later workflow
        if hasattr(self, "circuit"):
            self.construct_circuit(self.num_orbs, self.num_elec)

    def redo_M_mitigation(self, shots: int | None = None) -> None:
        """Redo M_mitigation.

        Args:
            shots: Overwrites QI internal shot number if int is defined.
        """
        self._make_Minv(shots=shots)

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
    def circuit(self) -> QuantumCircuit:
        """Get circuit.

        Returns:
            circuit (State + Ansatz circuit)
        """
        return self._circuit

    @circuit.setter
    def circuit(
        self,
        ansatz_circuit: QuantumCircuit,
    ) -> None:
        """Set circuit.

        Args:
            ansatz_circuit: Ansatz circuit
        """
        # Check if ISA is selected. If yes, pre-transpile circuit for later use.
        if self.ISA:
            self.ansatz_circuit: QuantumCircuit = self._transpile_circuit(ansatz_circuit)
            self._transpiled = True
            # Add state preparation circuit (e.g. HF)
            self._circuit: QuantumCircuit = self.ansatz_circuit.compose(
                self.state_circuit, qubits=self._circuit_indices, front=True
            )
        else:
            self.ansatz_circuit = ansatz_circuit
            # Add state preparation circuit (e.g. HF)
            self._circuit = self.ansatz_circuit.compose(self.state_circuit, front=True)

    def _transpile_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Transpile circuit with default or set PassManager.

        Args:
            circuit: circuit

        Returns:
            Transpiled Circuit.
        """
        if self.pass_manager is None:
            raise ValueError("Missing PassManager for transpilation.")
        if self.pass_manager is None and not self.ISA:  # for init
            raise ValueError(
                "You have not defined a PassManager but switched on ISA and try to transpile a circuit."
            )

        circuit_return = self.pass_manager.run(circuit)
        # Get layout indices
        if circuit_return.layout is None:
            self._measurement_indices = np.arange(circuit_return.num_qubits)
            self._circuit_indices = np.arange(circuit_return.num_qubits)
        else:
            self._measurement_indices = circuit_return.layout.final_index_layout()
            self._circuit_indices = circuit_return.layout.initial_index_layout(filter_ancillas=True)

        # Transpile X and Y measurement gates: only translation to basis gates and optimization.
        self._transp_xy = [
            self.pass_manager.optimization.run(self.pass_manager.translation.run(to_CBS_measurement("X"))),
            self.pass_manager.optimization.run(self.pass_manager.translation.run(to_CBS_measurement("Y"))),
        ]

        return circuit_return

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
        # IMPORTANT: Shot number in primitive initialization gets always overwritten by QI!
        self._circuit_multipl = 1
        if hasattr(self, "cliques"):
            self._reset_cliques()
        if hasattr(self, "_Minv") and self._Minv is not None:
            self._reset_M()
        # Get shot number form primitive if none defined
        if shots is None:
            if isinstance(self._primitive, BaseSamplerV2):
                print(
                    "SamplerV2 does not support ideal simulator. Number of shots is set to 10,000 by default"
                )
                self._shots: int | None = 10000
            else:
                print("Number of shots is None. Ideal simulator is assumed.")
                self._shots = None
        else:
            self._shots = shots
        # Check if shot number is allowed
        if self._shots is not None:
            if self._shots > self.max_shots_per_run:
                if isinstance(self._primitive, BaseEstimator):
                    self.shots = self.max_shots_per_run
                    print(
                        "WARNING: Number of shots specified exceed possibility for Estimator. Number of shots is set to ",
                        self.max_shots_per_run,
                    )
                else:
                    print("Number of requested shots exceed the limit of ", self.max_shots_per_run)
                    # Get number of circuits needed to fulfill shot number
                    self._circuit_multipl = math.floor(self._shots / self.max_shots_per_run)
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

    def _reset_cliques(self, verbose: bool = True) -> None:
        """Reset cliques to empty."""
        self.cliques = Clique()
        if verbose:
            print("Pauli saving has been reset.")

    def _reset_M(self, verbose: bool = True) -> None:
        """Reset M to None."""
        self._Minv = None
        if verbose:
            print("M matrix for error mitigation has been reset.")

    def op_to_qbit(self, op: FermionicOperator) -> SparsePauliOp:
        """Fermionic operator to qbit rep.

        Args:
            op: Operator as SlowQuant's FermionicOperator object

        Returns:
            Qubit representation of operator.
        """
        mapped_op = self.mapper.map(FermionicOp(op.get_qiskit_form(self.num_orbs), self.num_spin_orbs))
        if not isinstance(mapped_op, SparsePauliOp):
            raise TypeError(f"The qubit form of the operator is not SparsePauliOp got, {type(mapped_op)}")
        return mapped_op

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
        save_paulis = self._save_paulis
        if custom_parameters is None:
            run_parameters = self.parameters
        else:
            run_parameters = custom_parameters
            save_paulis = False

        # Check if estimator or sampler
        if isinstance(self._primitive, BaseEstimator):
            return self._estimator_quantum_expectation_value(op, run_parameters, self.circuit)
        if isinstance(self._primitive, (BaseSamplerV1, BaseSamplerV2)) and save_paulis:
            return self._sampler_quantum_expectation_value(op)
        if isinstance(self._primitive, (BaseSamplerV1, BaseSamplerV2)):
            return self._sampler_quantum_expectation_value_nosave(
                op,
                run_parameters,
                self.circuit,
                do_cliques=self._do_cliques,
            )
        raise ValueError(
            "The Quantum Interface was initiated with an unknown Qiskit primitive, {type(self._primitive)}"
        )

    def quantum_expectation_value_csfs(
        self,
        bra_csf: tuple[list[float], list[str]],
        op: FermionicOperator,
        ket_csf: tuple[list[float], list[str]],
        custom_parameters: list[float] | None = None,
    ) -> float:
        r"""Calculate expectation value using different bra and ket of a Hermitian operator.

        I.e. expectation values of the type,

        .. math::
            M_{IJ} = \left<\text{CSF}_I\left|\boldsymbol{U}^\dagger\hat{O}_\text{H}\boldsymbol{U}\right|\text{CSF}_J\right>

        The expectation value is calculated as:

        .. math::
            M_{IJ} = \sum_{\text{det}_i\in\text{CSF}_I}\sum_{\text{det}_j\in\text{CSF}_J}m_{ij}

        with,

        .. math::
            \begin{align}
            m_{ij} &= \left<\text{det}_i\left|\boldsymbol{U}^\dagger\hat{O}_\text{H}\boldsymbol{U}\right|\text{det}_j\right>\\
                   &= \left<\frac{1}{\sqrt{2}}
                   \left(\text{det}_i+\text{det}_j\right)\left|\boldsymbol{U}^\dagger\hat{O}_\text{H}\boldsymbol{U}\right|\frac{1}{\sqrt{2}}\left(\text{det}_i+\text{det}_j\right)\right>
                   - \frac{1}{2}\left<\text{det}_i\left|\boldsymbol{U}^\dagger\hat{O}_\text{H}\boldsymbol{U}\right|\text{det}_i\right>
                   - \frac{1}{2}\left<\text{det}_j\left|\boldsymbol{U}^\dagger\hat{O}_\text{H}\boldsymbol{U}\right|\text{det}_j\right>
            \end{align}

        #. 10.1103/PhysRevResearch.1.033062, Eq. (1)

        Args:
            bra_csf: Bra CSF.
            op: Hermitian fermionic operator.
            ket_csf: Ket CSF.
            custom_parameters: Non-default run parameters.

        Returns:
            Expectation value of operator.
        """
        if not isinstance(self.mapper, JordanWignerMapper):
            raise TypeError(
                f"Expectation values for custom CSFs only implemented for JordanWignerMapper. Got; {type(self.mapper)}"
            )
        if not isinstance(self._primitive, (BaseEstimator, BaseSamplerV1, BaseSamplerV2)):
            raise ValueError(
                "quantum_expectation_value_csfs got unsupported Qiskit primitive, {type(self._primitive)}"
            )
        if custom_parameters is None:
            run_parameters = self.parameters
        else:
            run_parameters = custom_parameters

        # Check if ISA beforehand.
        if self.ISA:
            connection_order = self._circuit_indices
        else:
            connection_order = np.arange(self.num_qubits)
        val = 0.0
        # Loop over dets in bra CSF
        for bra_coeff, bra_det in zip(bra_csf[0], bra_csf[1]):
            sign_bra = get_reordering_sign(bra_det)
            # Loop over dets in ket CSF
            for ket_coeff, ket_det in zip(ket_csf[0], ket_csf[1]):
                sign_ket = get_reordering_sign(ket_det)
                N = bra_coeff * ket_coeff * sign_bra * sign_ket  # pre-factor
                # I == J (diagonals)
                if bra_det == ket_det:
                    # Get det circuit. Only X-Gates -> no transpilation.
                    circuit = get_determinant_reference(bra_det, self.num_orbs, self.mapper)
                    # Combine: circuit/det + Ansatz. Map det circuit onto transpiled ansatz circuit order.
                    circuit = self.ansatz_circuit.compose(circuit, qubits=connection_order, front=True)
                    if isinstance(self._primitive, BaseEstimator):
                        val += N * self._estimator_quantum_expectation_value(op, run_parameters, circuit)
                    if isinstance(self._primitive, (BaseSamplerV1, BaseSamplerV2)):
                        val += N * self._sampler_quantum_expectation_value_nosave(
                            op,
                            run_parameters,
                            circuit,
                            do_cliques=self._do_cliques,
                        )
                # I != J (off-diagonals)
                else:
                    # First term of off-diagonal element involving I and J
                    # I and J superposition state of determinants
                    circuit = get_determinant_superposition_reference(
                        bra_det, ket_det, self.num_orbs, self.mapper
                    )
                    # Transpile superposition state. All stages needed due to H and CNOTs
                    if self.ISA:
                        self.circuit_superpos = circuit
                        circuit = self._layoutfree_pm.run(
                            circuit
                        )  # this is automatically in measurement index order
                        self.circuit_superpos_transp = circuit
                        # Combine: circuit/det + Ansatz. Map det circuit onto transpiled ansatz circuit order.
                        circuit = self.ansatz_circuit.compose(
                            circuit, qubits=self._measurement_indices, front=True
                        )
                    else:
                        circuit = self.ansatz_circuit.compose(circuit, front=True)
                    if isinstance(self._primitive, BaseEstimator):
                        val += N * self._estimator_quantum_expectation_value(op, run_parameters, self.circuit)
                    if isinstance(self._primitive, (BaseSamplerV1, BaseSamplerV2)):
                        val += N * self._sampler_quantum_expectation_value_nosave(
                            op,
                            run_parameters,
                            circuit,
                            do_cliques=self._do_cliques,
                        )

                    # Second term of off-diagonal element involving only I
                    # Get det circuit. Only X-Gates -> no transpilation.
                    circuit = get_determinant_reference(bra_det, self.num_orbs, self.mapper)
                    # Combine: circuit/det + Ansatz. Map det circuit onto transpiled ansatz circuit order.
                    circuit = self.ansatz_circuit.compose(circuit, qubits=connection_order, front=True)
                    if isinstance(self._primitive, BaseEstimator):
                        val -= (
                            0.5 * N * self._estimator_quantum_expectation_value(op, run_parameters, circuit)
                        )
                    if isinstance(self._primitive, (BaseSamplerV1, BaseSamplerV2)):
                        val -= (
                            0.5
                            * N
                            * self._sampler_quantum_expectation_value_nosave(
                                op,
                                run_parameters,
                                circuit,
                                do_cliques=self._do_cliques,
                            )
                        )

                    # Third term of off-diagonal element involving only J
                    # Get det circuit. Only X-Gates -> no transpilation.
                    circuit = get_determinant_reference(ket_det, self.num_orbs, self.mapper)
                    # Combine: circuit/det + Ansatz. Map det circuit onto transpiled ansatz circuit order.
                    circuit = self.ansatz_circuit.compose(circuit, qubits=connection_order, front=True)
                    if isinstance(self._primitive, BaseEstimator):
                        val -= (
                            0.5 * N * self._estimator_quantum_expectation_value(op, run_parameters, circuit)
                        )
                    if isinstance(self._primitive, (BaseSamplerV1, BaseSamplerV2)):
                        val -= (
                            0.5
                            * N
                            * self._sampler_quantum_expectation_value_nosave(
                                op,
                                run_parameters,
                                circuit,
                                do_cliques=self._do_cliques,
                            )
                        )
        return val

    def _estimator_quantum_expectation_value(
        self,
        op: FermionicOperator,
        run_parameters: list[float],
        run_circuit: QuantumCircuit,
    ) -> float:
        """Calculate expectation value of circuit and observables via Estimator.

        Args:
            op: SlowQuant fermionic operator.
            run_parameters: Circuit parameters.

        Returns:
            Expectation value of operator.
        """
        observables = self.op_to_qbit(op)
        if self.ISA:
            observables = observables.apply_layout(self.circuit.layout)
        job = self._primitive.run(
            circuits=run_circuit, parameter_values=run_parameters, observables=observables, shots=self.shots
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

    def _sampler_quantum_expectation_value(self, op: FermionicOperator | SparsePauliOp) -> float:
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
        if isinstance(op, FermionicOperator):
            observables = self.op_to_qbit(op)
        elif isinstance(op, SparsePauliOp):
            observables = op
        else:
            raise ValueError(
                f"Got unknown operator type {type(op)}, expected FermionicOperator or SparsePauliOp"
            )

        if not hasattr(self, "cliques"):
            self.cliques = Clique()

        paulis_str = [str(x) for x in observables.paulis]
        new_heads = self.cliques.add_paulis(paulis_str)

        # Check if error mitigation is requested and if read-out matrix already exists.
        if self.do_M_mitigation and self._Minv is None:
            self._make_Minv(shots=self._M_shots)

        if len(new_heads) != 0:
            # Simulate each clique head with one combined device call
            # and return a list of distributions
            distr = self._one_call_sampler_distributions(new_heads, self.parameters, self.circuit)
            if self.do_M_mitigation:  # apply error mitigation if requested
                for i, dist in enumerate(distr):
                    distr[i] = correct_distribution(dist, self._Minv)
            if self.do_postselection:
                for i, (dist, head) in enumerate(zip(distr, new_heads)):
                    if "X" not in head and "Y" not in head:
                        distr[i] = postselection(dist, self.mapper, self.num_elec, self.num_qubits)
            self.cliques.update_distr(new_heads, distr)

        # Loop over all Pauli strings in observable and build final result with coefficients
        for pauli, coeff in zip(paulis_str, observables.coeffs):
            result = 0.0
            for key, value in self.cliques.get_distr(pauli).items():  # build result from quasi-distribution
                result += value * get_bitstring_sign(pauli, key)
            values += result * coeff

        if isinstance(values, complex):
            if abs(values.imag) > 10**-2:
                print("Warning: Complex number detected with Im = ", values.imag)

        return values.real

    def _sampler_quantum_expectation_value_nosave(
        self,
        op: FermionicOperator | SparsePauliOp,
        run_parameters: list[float],
        run_circuit: QuantumCircuit,
        do_cliques: bool = True,
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
        if isinstance(op, FermionicOperator):
            observables = self.op_to_qbit(op)
        elif isinstance(op, SparsePauliOp):
            observables = op
        else:
            raise ValueError(
                f"Got unknown operator type {type(op)}, expected FermionicOperator or SparsePauliOp"
            )

        # Check if error mitigation is requested and if read-out matrix already exists.
        if self.do_M_mitigation and self._Minv is None:
            self._make_Minv(shots=self._M_shots)

        paulis_str = [str(x) for x in observables.paulis]
        if do_cliques:
            # Obtain cliques for operator's Pauli strings
            cliques = Clique()

            new_heads = cliques.add_paulis(paulis_str)

            # Simulate each clique head with one combined device call
            # and return a list of distributions
            distr = self._one_call_sampler_distributions(new_heads, run_parameters, run_circuit)
            if self.do_M_mitigation:  # apply error mitigation if requested
                for i, dist in enumerate(distr):
                    distr[i] = correct_distribution(dist, self._Minv)
            if self.do_postselection:
                for i, (dist, head) in enumerate(zip(distr, new_heads)):
                    if "X" not in head and "Y" not in head:
                        distr[i] = postselection(dist, self.mapper, self.num_elec, self.num_qubits)
            cliques.update_distr(new_heads, distr)

            # Loop over all Pauli strings in observable and build final result with coefficients
            for pauli, coeff in zip(paulis_str, observables.coeffs):
                result = 0.0
                for key, value in cliques.get_distr(pauli).items():  # build result from quasi-distribution
                    result += value * get_bitstring_sign(pauli, key)
                values += result * coeff
        else:
            # Simulate each Pauli string with one combined device call
            distr = self._one_call_sampler_distributions(paulis_str, run_parameters, run_circuit)
            if self.do_M_mitigation:  # apply error mitigation if requested
                for i, dist in enumerate(distr):
                    distr[i] = correct_distribution(dist, self._Minv)
            if self.do_postselection:
                for i, (dist, pauli) in enumerate(zip(distr, paulis_str)):
                    if "X" not in pauli and "Y" not in pauli:
                        distr[i] = postselection(dist, self.mapper, self.num_elec, self.num_qubits)

            # Loop over all Pauli strings in observable and build final result with coefficients
            for pauli, coeff, dist in zip(paulis_str, observables.coeffs, distr):
                result = 0.0
                for key, value in dist.items():  # build result from quasi-distribution
                    result += value * get_bitstring_sign(pauli, key)
                values += result * coeff

        if isinstance(values, complex):
            if abs(values.imag) > 10**-2:
                print("Warning: Complex number detected with Im = ", values.imag)

        return values.real

    def quantum_variance(
        self,
        op: FermionicOperator | SparsePauliOp,
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

        # Map Fermionic to Qubit
        if isinstance(op, FermionicOperator):
            observables = self.op_to_qbit(op)
        elif isinstance(op, SparsePauliOp):
            observables = op
        else:
            raise ValueError(
                f"Got unknown operator type {type(op)}, expected FermionicOperator or SparsePauliOp"
            )

        if do_cliques:
            if not hasattr(self, "cliques"):
                self.cliques = Clique()

            new_heads = self.cliques.add_paulis([str(x) for x in observables.paulis])

            if len(new_heads) != 0:
                # Check if error mitigation is requested and if read-out matrix already exists.
                if self.do_M_mitigation and self._Minv is None:
                    self._make_Minv()

                # Simulate each clique head with one combined device call
                # and return a list of distributions
                distr = self._one_call_sampler_distributions(new_heads, run_parameters, self.circuit)
                if self.do_M_mitigation:  # apply error mitigation if requested
                    for i, dist in enumerate(distr):
                        distr[i] = correct_distribution(dist, self._Minv)
                if self.do_postselection:
                    for i, (dist, head) in enumerate(zip(distr, new_heads)):
                        if "X" not in head and "Y" not in head:
                            distr[i] = postselection(dist, self.mapper, self.num_elec, self.num_qubits)
                self.cliques.update_distr(new_heads, distr)
        else:
            print(
                "WARNING: REM and Post-Selection not implemented for quantum variance without the use of cliques."
            )

        # Loop over all Pauli strings in observable and build final result with coefficients
        result = 0.0
        paulis_str = [str(x) for x in observables.paulis]
        for pauli, coeff in zip(paulis_str, observables.coeffs):
            if no_coeffs:
                coeff = 1
            # Get distribution from cliques
            if do_cliques:
                dist = self.cliques.get_distr(pauli)
                # Calculate p1: Probability of measuring one
                p1 = 0.0
                for key, value in dist.items():
                    if get_bitstring_sign(pauli, key) == 1:
                        p1 += value
            else:
                p1 = self._sampler_distribution_p1(pauli, run_parameters)
            if self.shots is None:
                var_p = 4 * np.abs(coeff.real) ** 2 * np.abs(p1 - p1**2)
            else:
                var_p = 4 * np.abs(coeff.real) ** 2 * np.abs(p1 - p1**2) / (self.shots)
            result += var_p
        return result

    def _one_call_sampler_distributions(
        self,
        paulis: list[str] | str,
        run_parameters: list[list[float]] | list[float],
        circuits_in: list[QuantumCircuit] | QuantumCircuit,
        overwrite_shots: int | None = None,
    ) -> list[dict[int, float]]:
        r"""Get results from a sampler distribution for several Pauli strings measured on several circuits.

        The expectation value of a Pauli string is calcuated as:

        .. math::
            E = \sum_i^N p_i\left<b_i\left|P\right|b_i\right>

        With :math:`p_i` being the :math:`i` th probability and :math:`b_i` being the `i` th bit-string.

        Args:
            paulis: (List of) Pauli strings to measure.
            run_paramters: List of parameters of each circuit.
            circuits_in: List of circuits
            overwrite_shots: Overwrite QI shot number.

        Returns:
            Array of quasi-distributions in order of all circuits results for a given Pauli String first.
            E.g.: [PauliString[0] for Circuit[0], PauliString[0] for Circuit[1], ...]
        """
        if self._circuit_multipl > 1:
            shots: int | None = self.max_shots_per_run
        else:
            shots = self.shots
        if overwrite_shots is not None:
            print("Warning: Overwriting QI shots has been used.")
            shots = overwrite_shots

        if isinstance(paulis, str):
            paulis = [paulis]
        num_paulis = len(paulis)
        if isinstance(circuits_in, QuantumCircuit):
            circuits_in = [circuits_in]
        num_circuits = len(circuits_in)

        # Check V1 vs. V2
        if isinstance(self._primitive, BaseSamplerV2):
            # make parameter list 2d for one circuit.
            if num_circuits == 1:
                run_parameters = [run_parameters]  # type: ignore

            # Create pubs for V2
            pubs = []
            for nr_pauli, pauli in enumerate(paulis):
                pauli_circuit = to_CBS_measurement(pauli, self._transp_xy)
                for nr_circuit, circuit in enumerate(circuits_in):
                    # Add measurement in correct layout
                    ansatz_w_obs = circuit.compose(pauli_circuit, qubits=self._measurement_indices)
                    # Create classic register and measure relevant qubits
                    ansatz_w_obs.add_register(ClassicalRegister(self.num_qubits, name="meas"))
                    ansatz_w_obs.measure(self._measurement_indices, np.arange(self.num_qubits))
                    pubs.append((ansatz_w_obs, run_parameters[nr_circuit]))
            pubs = pubs * self._circuit_multipl

            # Run sampler
            job = self._primitive.run(pubs, shots=shots)
        else:
            if not self.ISA:  # No own layout-design needed: this might be faster. So we leave it for now.
                circuits = [None] * (num_paulis * num_circuits)
                # Create QuantumCircuits for V1
                for nr_pauli, pauli in enumerate(paulis):
                    pauli_circuit = to_CBS_measurement(pauli)
                    for nr_circuit, circuit in enumerate(circuits_in):
                        ansatz_w_obs = circuit.compose(pauli_circuit)
                        ansatz_w_obs.measure_all()
                        circuits[(nr_circuit + (nr_pauli * num_circuits))] = ansatz_w_obs
                circuits = circuits * self._circuit_multipl
            else:
                circuits = [None] * (num_paulis * num_circuits)
                # Create QuantumCircuits for V1
                for nr_pauli, pauli in enumerate(paulis):
                    pauli_circuit = to_CBS_measurement(pauli, self._transp_xy)
                    for nr_circuit, circuit in enumerate(circuits_in):
                        ansatz_w_obs = circuit.compose(pauli_circuit, qubits=self._measurement_indices)
                        ansatz_w_obs.add_register(ClassicalRegister(self.num_qubits))
                        ansatz_w_obs.measure(self._measurement_indices, np.arange(self.num_qubits))
                        circuits[(nr_circuit + (nr_pauli * num_circuits))] = ansatz_w_obs
                circuits = circuits * self._circuit_multipl

            # Create parameters array for V1
            if num_circuits == 1:
                parameter_values = [run_parameters] * (num_paulis * self._circuit_multipl)
            else:
                parameter_values = run_parameters * (num_paulis * self._circuit_multipl)  # type: ignore

            # Run sampler
            job = self._primitive.run(circuits, parameter_values=parameter_values, shots=shots)

        if self.shots is not None:  # check if ideal simulator
            self.total_shots_used += self.shots * num_paulis * num_circuits
        self.total_device_calls += 1
        self.total_paulis_evaluated += num_paulis * num_circuits

        # Get quasi-distribution in binary probabilities
        if isinstance(self._primitive, BaseSamplerV2):
            result = job.result()
            distr = [{}] * len(result)  # type: list[dict]
            for nr, job in enumerate(result):
                distr[nr] = job.data.meas.get_counts()
                for key in list(distr[nr].keys()):
                    distr[nr][int(key, 2)] = distr[nr].pop(key) / shots
        else:
            distr = job.result().quasi_dists

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
        self, pauli: str, run_parameters: list[float], custom_circ: None | QuantumCircuit = None
    ) -> dict[int, float]:
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
        if self.ISA:
            raise ValueError("Function _sampler_distribution does not work with ISA.")
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
        job = self._primitive.run(ansatz_w_obs, parameter_values=run_parameters, shots=shots)
        if shots is not None:  # check if ideal simulator
            self.total_shots_used += shots
        self.total_device_calls += 1
        self.total_paulis_evaluated += 1

        # Get quasi-distribution in binary probabilities
        distr = job.result().quasi_dists[0]
        return distr

    def _sampler_distribution_p1(
        self, pauli: str, run_parameters: list[float], custom_circ: None | QuantumCircuit = None
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

    def _make_Minv(self, shots: None | int = None) -> None:
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

        Args:
            shots: Number of shots if they are meant to differ from QI internal shot number.

        """
        print("Measuring error mitigation read-out matrix.")
        self._save_layout = True
        if self.num_qubits > 8:
            raise ValueError("Current implementation does not scale above 8 qubits?")
        if self.do_M_ansatz0:
            ansatz = self.ansatz_circuit
        else:
            ansatz = QuantumCircuit(self.num_qubits)  # empty circuit
        M = np.zeros((2**self.num_qubits, 2**self.num_qubits))
        ansatz_list = [None] * 2**self.num_qubits
        if self.ISA:
            for nr, comb in enumerate(itertools.product([0, 1], repeat=self.num_qubits)):
                ansatzX = ansatz.copy()
                for i, bit in enumerate(comb[::-1]):  # because of Qiskit ordering
                    if bit == 1:
                        ansatzX.x(self._measurement_indices[i])
                # Make list of custom ansatz
                ansatz_list[nr] = ansatzX
            # Simulate all elements with one device call
            Px_list = self._one_call_sampler_distributions(
                "Z" * self.num_qubits,
                [[10**-8] * len(ansatz.parameters)] * len(ansatz_list),
                ansatz_list,
                overwrite_shots=shots,
            )
        else:
            for nr, comb in enumerate(itertools.product([0, 1], repeat=self.num_qubits)):
                ansatzX = ansatz.copy()
                for i, bit in enumerate(comb[::-1]):  # because of Qiskit ordering
                    if bit == 1:
                        ansatzX.x(i)
                # Make list of custom ansatz
                ansatz_list[nr] = ansatzX
            # Simulate all elements with one device call
            Px_list = self._one_call_sampler_distributions(
                "Z" * self.num_qubits,
                [[10**-8] * len(ansatz.parameters)] * len(ansatz_list),
                ansatz_list,
                overwrite_shots=shots,
            )
        # Construct M
        for idx2, Px in enumerate(Px_list):
            for idx1, prob in Px.items():
                M[idx1, idx2] = prob
        self._Minv = np.linalg.inv(M)

    def get_info(self) -> None:
        """Get infos about settings."""
        if isinstance(self.ansatz, QuantumCircuit):
            data = f"Your settings are:\n {'Ansatz:':<20} {'custom circuit'}\n {'Number of shots:':<20} {self.shots}\n"
        else:
            data = f"Your settings are:\n {'Ansatz:':<20} {self.ansatz}\n {'Number of shots:':<20} {self.shots}\n"
        data += f" {'ISA':<20} {self.ISA}\n {'Transpiled circuit':<20} {self._transpiled}\n {'Primitive:':<20} {self._primitive.__class__.__name__}\n"
        data += f" {'Post-processing:':<20} {self.do_postselection}"
        if self.do_M_mitigation:
            data += (
                f"\n {'M mitigation:':<20} {self.do_M_mitigation}\n {'M Ansatz0:':<20} {self.do_M_ansatz0}"
            )
        if self.ISA:
            data += f"\n {'Circuit layout:':<20} {self._measurement_indices}"
            if self._internal_pm:
                data += f"\n {'Transpiled backend:':<20} {self._primitive_backend}\n {'Transpiled opt. level:':<20} {self._primitive_level}"
            if isinstance(self._primitive, BaseSamplerV2) and hasattr(self._primitive.options, "twirling"):
                data += f"\n {'Pauli twirling:':<20} {self._primitive.options.twirling.enable_gates}\n {'Dynamic decoupling:':<20} {self._primitive.options.dynamical_decoupling.enable}"
        print(data)
