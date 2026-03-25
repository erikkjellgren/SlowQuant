import pickle
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import (
    BaseEstimatorV1,
    BaseEstimatorV2,
    BaseSamplerV1,
    BaseSamplerV2,
)
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import IBMBackend

from slowquant.qiskit_interface.util import (
    Clique,
    MitigationFlags,
    get_bitstring_sign,
    postselection,
    fit_in_clique
)
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.qiskit_interface import (tiled_m0_main, tiled_m0_helper)

class QuantumInterfaceTiled:
    """
    Quantum interface class for tiledM0
    This class handles the interface with qiskit and the communication with quantum hardware.
    """

    def __init__(
        self,
        backend : IBMBackend | AerSimulator,
        primitive: BaseSamplerV1 | BaseSamplerV2,
        ansatz: str | QuantumCircuit,
        mapper: FermionicMapper,
        pass_manager_options: dict[str, Any] | None = None,
        ansatz_options: dict[str, Any] | None = None,
        shots: int = 10000,
        tiledM0_shots: int = 15000,
        do_postselection: bool = False,
    ) -> None:
        """Interface to Qiskit to use IBM quantum hardware or simulator.

        Args:
            backend: The backend to run on.
            primitive: Qiskit Sampler object.
            ansatz: Name of ansatz to be used.
            mapper: Qiskit mapper object, e.g. JW or Parity.
            pass_manager_options: Dictionary to define custom pass manager.
            ansatz_options: Ansatz options.
            shots: Number of shots used for expectation values per Pauli string or Pauli head.
            tiledM0_shots: Number of shots used for tiled M0 per column of each assignment matrix.
            do_postselection: Use postselection to preserve number of particles in the computational basis.
        """
        if ansatz_options is None:
            ansatz_options = {"layers" : 1}
        elif "layers" not in ansatz_options:
            print("Number of layers not specified. Set to 1.")
            ansatz_options["layers"] = 1
        allowed_ansatz = (
            "tUPS",
        )
        if not isinstance(ansatz, QuantumCircuit) and ansatz not in allowed_ansatz:
            raise ValueError(
                "The chosen Ansatz is not available. Choose from: ",
                allowed_ansatz
            )
        if pass_manager_options is None:
            pass_manager_options = {}
        self.pass_manager_options = pass_manager_options
        allowed_pm_options = (
            "optimization_level",
            "backend",
            "layout_method",
            "routing_method",
            "translation_method",
            "seed_transpiler",
            "optimization_method",
        )
        wrong_items = [
            item for item in list(self.pass_manager_options.keys()) if item not in allowed_pm_options
        ]
        if len(wrong_items) > 0:
            raise ValueError(
                "The specified pass manager options do not exist. You have specified",
                wrong_items,
                " which is not in allowed options",
                allowed_pm_options,
            )
        if isinstance(primitive, (BaseEstimatorV1, BaseEstimatorV2)):
            raise ValueError("Estimator is not supported.")
        elif not isinstance(primitive, (BaseSamplerV1, BaseSamplerV2)):
            raise TypeError(f"Unsupported Qiskit primitive, {type(primitive)}")
        self.ansatz = ansatz
        self.backend = backend
        self._transpiled = True  # Always true in tiledM0
        self._primitive = primitive
        self.mapper = mapper
        self.mitigation_flags: MitigationFlags = MitigationFlags(
            do_postselection = do_postselection, # Not yet implemented for tiled M0
            do_tiledM0 = True   # Always do tiled mitigation
        )
        if do_postselection == True:
            print("Post selection is not yet implemented for tiled M0.")
        self.shots = shots
        self.total_shots_used = 0
        self.total_device_calls = 0
        self.total_paulis_evaluated = 0
        self.ansatz_options = ansatz_options
        self.saver: dict[int, Clique] = {}
        self._save_paulis = True  # hard switch to stop using Pauli saving (debugging tool).
        self._do_cliques = True  # hard switch to stop using QWC (debugging tool).
        self.tiledM0_shots = tiledM0_shots  # define a separate number of shots for M
        
        self.tiledM0 = None

    def construct_circuit(self, num_orbs: int, num_elec: tuple[int, int]) -> None:
        """Construct qiskit circuit.

        Args:
            num_orbs: Number of orbitals in spatial basis.
            num_elec: Number of electrons (alpha, beta).
        """
        self.num_orbs = num_orbs
        self.num_spin_orbs = 2 * num_orbs
        self.num_elec = num_elec
        self.grad_param_R: dict[
            str, int
        ] = {}  # Contains information about the parameterization needed for gradient evaluations.

        # Note: ``self.grad_param_R`` is passed by reference here
        tileCircuits, tileQubits = tiled_m0_helper.GetTileCircuitsAndQubits_TUPS(self.num_orbs, self.backend.num_qubits, self.ansatz_options["layers"], self.grad_param_R)

        ppInputState = "1100" * (self.num_spin_orbs // 4) # pp input state by default
        self.tiledM0 = tiled_m0_main.TiledM0(
            tileCircuits = tileCircuits,
            tileQubits = tileQubits,
            layerCount = self.ansatz_options["layers"],
            elecCount = sum(self.num_elec),
            backend = self.backend,
            shots = self.shots,
            sampler = self._primitive,
            mitigatorShots = self.tiledM0_shots,
            inputState = ppInputState,
            doPatchParallelization = False
        )

        self.state_circuit = self.tiledM0.inputStatePrepQc
        self.ansatz_circuit = self.tiledM0.posttFullNonParameterizedCircuit
        self.circuit = self.state_circuit.compose(self.ansatz_circuit)

        # Get number of qubits
        self.num_qubits = self.state_circuit.num_qubits

        # Check that R parameter for gradient is consistent with the parameter names.
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
            if name not in self.grad_param_R.keys():
                raise ValueError(
                    f"Got parameter name, {name}, that is not in grad_param_R, {self.grad_param_R}"
                )

        if not hasattr(self, "_parameters"):
            # Set parameter to HarteeFock
            self._parameters = [0.0] * self.circuit.num_parameters

    def update_mitigation_flags(self, **kwargs) -> None:
        """Update mitigation flags.

        Args:
            **kwargs: Keyword arguments to update mitigation flags.
        """
        self.mitigation_flags.update_flags(**kwargs)

    def redo_M_mitigation(self, shots: int | None = None) -> None:
        """
        Set flag to redo tiled M0 mitigators. The mitigators are remade when a call to get new probability vectors for new Pauli strings are made.
        
        Args:
            shots: Overwrites QI internal shot number if int is defined.
        """
        self.tiledM0.redoMitigators = True

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
        # The distributions should only reset if the parameters are actually changed.
        if not np.array_equal(self._parameters, parameters):
            self.saver = {}
        self._parameters = parameters.copy()

    def _reset_cliques(self, verbose: bool = True) -> None:
        """Reset cliques to empty.

        Args:
            verbose: Print additional details.
        """
        self.saver = {}
        if verbose:
            print("Pauli saving has been reset.")

    def _reset_M(self, verbose: bool = True) -> None:
        """Reset M to None.

        Args:
            verbose: Print additional details.
        """
        self._Minv = None # reset a matrices ToDo
        if verbose:
            print("M matrix for error mitigation has been reset.")

    def op_to_qbit(self, op: FermionicOperator) -> SparsePauliOp:
        """Fermionic operator to qbit rep.

        The Pauli string representation is in qiskit's format of qN,qN-1,...,q0.

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

        # Check if saving is requested
        if isinstance(self._primitive, (BaseSamplerV1, BaseSamplerV2)) and save_paulis:
            return self._sampler_quantum_expectation_value(op)
        if isinstance(self._primitive, (BaseSamplerV1, BaseSamplerV2)):
            return self._sampler_quantum_expectation_value_nosave(
                op,
                run_parameters,
                do_cliques=self._do_cliques,
            )
        raise ValueError(
            "The Quantum Interface was initiated with an unknown Qiskit primitive, {type(self._primitive)}"
        )

    def _sampler_quantum_expectation_value(
        self,
        op: FermionicOperator | SparsePauliOp,
    ) -> float:
        r"""Calculate expectation value of circuit and observables via Sampler.

        Calculated Pauli expectation values will be saved in memory.

        The expectation value over a fermionic operator is calculated as:

        .. math::
            E = \sum_i^N c_i\left<0\left|P_i\right|0\right>

        With :math:`c_i` being the :math:`i` the coefficient and :math:`P_i` the :math:`i` the Pauli string.

        Args:
            op: SlowQuant fermionic operator.
            run_circuit: custom circuit to be run. If not specified, HF+Ansatz circuit is used.
            det: Classify state (determinant) of circuit for Pauli saving.
                Specified in chemistry form, i.e. left-to-right, alternating alpha and beta.
            circuit_M: custom circuit for M_Ansatz0 (correlation matrix is not stored). If not specified, M0 of Ansatz is used.
            csfs_option: Option for how superposition of initial states was handled. Default is 1.

        Returns:
            Expectation value of operator.
        """
        # Get HF determinant string
        det = "1" * (self.num_elec[0] + self.num_elec[1]) + "0" * (
            self.num_spin_orbs - (self.num_elec[0] + self.num_elec[1])
        )
        det_int = int(det, 2)

        # Map Fermionic to Qubit
        if isinstance(op, FermionicOperator):
            observables = self.op_to_qbit(op)
        elif isinstance(op, SparsePauliOp):
            observables = op
        else:
            raise ValueError(
                f"Got unknown operator type {type(op)}, expected FermionicOperator or SparsePauliOp"
            )

        if det_int not in self.saver:
            self.saver[det_int] = Clique()
        
        paulis_str = [str(x) for x in observables.paulis]
        new_heads = self.saver[det_int].add_paulis(paulis_str)

        if len(new_heads) != 0:
            # Simulate each clique head with one combined device call
            distrAllHeads = self.tiledM0.GetDistributionsForPauliStringHeads(new_heads, self.parameters, self.shots)

            # save raw results
            self.saver[det_int].update_distr(new_heads, distrAllHeads)

        """
        Note: The mitigated distributions are not saved in the tiled M0 version. Only the raw distributions are saved (as dictionaries)
        It is not viable to save many mitigated distributions for large qubit numbers because the mitigated distributions
        are not sparse the same way the raw distributions are. The memory demands will be much larger.
        When a mitigated distribution is needed, the code below generates the mitigated distribution on the spot.
        The downside is that there might be cases where we apply the same mitigation procedure to the same raw distribution
        multiple times. This will not be the case for a single call to this function, however.
        """
        mitigatedExpectationValue, rawExpectationValue = 0.0, 0.0
        pauliChosen = [False for i in range(len(paulis_str))] # Keep track of which Paulis we have calculated the expectation values of.
        for head in new_heads:
            compatiblePaulisAndWeights = dict() # Identify Paulis that are compatible with the current head (so we can use the same raw distribution and then generate the mitigated distribution only once)
            pauliIdx = 0
            for pauli, coeff in zip(paulis_str, observables.coeffs):
                if fit_in_clique(pauli, head)[0] and pauliChosen[pauliIdx] == False:
                    compatiblePaulisAndWeights[pauli] = coeff
                    pauliChosen[pauliIdx] = True
                pauliIdx += 1
            
            distr = self.saver[det_int].get_distr(head)
            p_All = self.tiledM0.GetProbabilityVectorsFromDistribution(distr, self.shots)
            mitigatedExpectationValue_ThisClique, rawExpectationValue_ThisClique = self.tiledM0.GetExpectationValues(p_All, compatiblePaulisAndWeights)

            mitigatedExpectationValue += mitigatedExpectationValue_ThisClique
            rawExpectationValue += rawExpectationValue_ThisClique

        if isinstance(mitigatedExpectationValue, complex):
            if abs(mitigatedExpectationValue.imag) > 10**-2:
                print("Warning: Complex number detected with Im = ", mitigatedExpectationValue.imag)

        self.tiledM0.LogTiledM0([mitigatedExpectationValue.real], [rawExpectationValue.real])

        print("Tiled M0 mitigated:", mitigatedExpectationValue.real, "\nTiled M0 raw:", rawExpectationValue.real)
        return mitigatedExpectationValue.real

    def _sampler_quantum_expectation_value_nosave(
        self,
        op: FermionicOperator | SparsePauliOp,
        run_parameters: list[float],
        do_cliques: bool = True,
    ) -> float:
        r"""Calculate expectation value of circuit and observables via Sampler.

        Calling this function will not use any pre-calculated Pauli expectation values.
        Nor will it save any of the calculated Pauli expectation values.

        The expectation value over a fermionic operator is calculated as:

        .. math::
            E = \sum_i^N c_i\left<0\left|P_i\right|0\right>

        With :math:`c_i` being the :math:`i` the coefficient and :math:`P_i` the :math:`i` the Pauli string.

        Args:
            op: SlowQuant fermionic operator.
            run_parameters: Circuit parameters.
            do_cliques: If True, use cliques (QWC).

        Returns:
            Expectation value of operator.
        """
        # NOT TESTED FOR tiledM0

        # Map Fermionic to Qubit
        if isinstance(op, FermionicOperator):
            observables = self.op_to_qbit(op)
        elif isinstance(op, SparsePauliOp):
            observables = op
        else:
            raise ValueError(
                f"Got unknown operator type {type(op)}, expected FermionicOperator or SparsePauliOp"
            )

        paulis_str = [str(x) for x in observables.paulis]
        mitigatedExpectationValue, rawExpectationValue = 0.0, 0.0
        if do_cliques:
            # Obtain cliques for operator's Pauli strings
            cliques = Clique()

            new_heads = cliques.add_paulis(paulis_str)

            # Simulate each clique head with one combined device call
            # and return a list of distributions
            distrAllPaulis = self.tiledM0.GetDistributionsForPauliStringHeads(new_heads, run_parameters, self.shots)                

            pauliChosen = [False for i in range(len(paulis_str))] # Keep track of which Paulis we have calculated the expectation values of.
            for i in range(len(new_heads)):
                compatiblePaulisAndWeights = dict() # Identify Paulis that are compatible with the current head (so we can use the same raw distribution and then generate the mitigated distribution only once)
                pauliIdx = 0
                for pauli, coeff in zip(paulis_str, observables.coeffs):
                    if fit_in_clique(pauli, new_heads[i])[0] and pauliChosen[pauliIdx] == False:
                        compatiblePaulisAndWeights[pauli] = coeff
                        pauliChosen[pauliIdx] = True
                    pauliIdx += 1
                
                p_All = self.tiledM0.GetProbabilityVectorsFromDistribution(distrAllPaulis[i], self.shots)
                mitigatedExpectationValue_ThisClique, rawExpectationValue_ThisClique = self.tiledM0.GetExpectationValues(p_All, compatiblePaulisAndWeights)

                mitigatedExpectationValue += mitigatedExpectationValue_ThisClique
                rawExpectationValue += rawExpectationValue_ThisClique
        else:
            # Simulate each Pauli string with one combined device call
            # Note that ``paulis_str`` is passed here.
            distrAllPaulis = self.tiledM0.GetDistributionsForPauliStringHeads(paulis_str, run_parameters, self.shots)

            for i in range(len(paulis_str)):
                p_All = self.tiledM0.GetProbabilityVectorsFromDistribution(distrAllPaulis[i], self.shots)
                pauli = dict()
                pauli[paulis_str[i]] = observables.coeffs[i]
                mitigatedExpectationValue_ThisPauli, rawExpectationValue_ThisPauli = self.tiledM0.GetExpectationValues(p_All, pauli)
                mitigatedExpectationValue += mitigatedExpectationValue_ThisPauli
                rawExpectationValue += rawExpectationValue_ThisPauli

        if isinstance(mitigatedExpectationValue, complex):
            if abs(mitigatedExpectationValue.imag) > 10**-2:
                print("Warning: Complex number detected with Im = ", mitigatedExpectationValue.imag)

        print("Tiled M0 mitigated:", mitigatedExpectationValue.real, "\nTiled M0 raw:", rawExpectationValue.real)

        return mitigatedExpectationValue.real

    def quantum_variance(
        # MIGHT NOT WORK YET
        self,
        op: FermionicOperator | SparsePauliOp,
        do_cliques: bool = True,
        no_coeffs: bool = False,
        custom_parameters: list[float] | None = None,
    ) -> float:
        """Calculate variance (std**2) of expectation value of circuit and observables.

        This works either by accessing the saver distributions or by calculating each expectation value from scratch.
        If they are calculated from scratch, no error mitigation can be applied.

        Args:
            op: SlowQuant fermionic operator.
            do_cliques: boolean if cliques are used. They are accessed via the saver.
            no_coeffs: boolean if coefficients of each Pauli string are used or all st to 1.
            custom_parameters: optional custom circuit parameters.

        Returns:
            Variance of expectation value.
        """
        det_int = int(
            "1" * (self.num_elec[0] + self.num_elec[1])
            + "0" * (self.num_spin_orbs - (self.num_elec[0] + self.num_elec[1])),
            2,
        )
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

        # Loop over all Pauli strings in observable and build final result with coefficients
        result = 0.0
        paulis_str = [str(x) for x in observables.paulis]
        for pauli, coeff in zip(paulis_str, observables.coeffs):
            if no_coeffs:
                coeff = 1
            # Get distribution from cliques
            if do_cliques:
                dist = self.saver[det_int].get_distr(pauli, self.mitigation_flags)
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

    def _apply_postselection(self, distr: list[dict[int, float]], heads: list[str]) -> None:
        ### HAS TO BE ADAPTED FOR BATCHES!!!!!
        """Apply post-selection to the distributions.

        Args:
            distr: List of distributions to be post-selected.
            heads: List of Pauli strings (heads).
        """
        for i, (dist, head) in enumerate(zip(distr, heads)):
            if "X" not in head and "Y" not in head:
                distr[i] = postselection(dist, self.mapper, self.num_elec, self.num_qubits)


    def save_paulis_to_file(self, filename: str) -> None:
        """Save all Pauli strings and their distributions to a file.

        Args:
            filename: Name of the file to save the data to.
        """
        with open(filename, "wb") as file:
            pickle.dump(self.saver, file)

    def load_paulis_from_file(self, filename: str) -> None:
        """Load Pauli strings and their distributions from a file.

        Args:
            filename: Name of the file to load the data from.
        """
        with open(filename, "rb") as file:
            self.saver = pickle.load(file)
        print(f"Loaded Pauli strings from {filename}.")

    def get_info(self) -> None:
        if self.tiledM0 == None:
            print("Info not available before ``construct_circuit`` is called.")
            return
        """Get infos about settings."""

        data = f"Your settings are:\n{'Ansatz:':<50} {self.ansatz}\n{'Number of expectation value shots (per QC):':<50} {self.shots}\n{'Number of tiled M0 shots (per QC):':<50} {self.tiledM0_shots}"
        data += f"\n{'Primitive:':<50} {self._primitive.__class__.__name__}"
        data += f"\n{'Backend:':<50} {self.backend.name}" 
        data += f"\n{'Patch:':<50} {self.tiledM0.patches[0]}" # Only a single patch as of the current implementation

        data += f"\n{'Non-local gates:':<50} {self.ansatz_circuit.num_nonlocal_gates()}"
        data += f"\n{'Transpiler settings:'}"
        for key, value in self.pass_manager_options.items():
            data += f"\n{key:<50} {value}"
        if isinstance(self._primitive, BaseSamplerV2) and hasattr(
            self._primitive.options, "twirling"
        ):
            data += f"\n{'Pauli twirling:':<50} {self._primitive.options.twirling.enable_gates}\n{'Dynamic decoupling:':<50} {self._primitive.options.dynamical_decoupling.enable}"

        print(f"{data}\nMitigation flags:\n{self.mitigation_flags.status_report()}")