import numpy as np
from numpy._typing import NDArray
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import (SamplerV2, IBMBackend)
from qiskit_aer import AerSimulator
from slowquant.qiskit_interface.util import add_permutation_gate
from slowquant.qiskit_interface.tiled_m0_helper import (ApplyMitigator, CalculateExpectationValueOfPauliString)

class TiledM0:
    def __init__(
        self,
        tileCircuits : list[QuantumCircuit],
        tileQubits : list[list[int]],
        layerCount: int,
        elecCount: int,
        backend: IBMBackend | AerSimulator,
        shots: int,
        sampler : SamplerV2 = None,
        mitigatorShots : int = 15000,
        inputState : str = None,
        doPatchParallelization : bool = False,
    ) -> None:
        """
        Initialize the tiled M0 driver.

        Args:
            tileCircuits: A list with the quantum circuits for every tile in the ansatz (pre-transpilation) - NOT just first layer. The order of the quantum circuits should be the same as the order in which the tiles appear in the ansatz, starting with the tile that is used on the input state first.
            tileQubit: A tile-wise list with the qubits in the FIRST LAYER ONLY of the ansatz (pre-transpilation). tileQubits[i] should be the qubits associated with tileCircuits[i].
            layerCount: The number of layers to be used in the ansatz.
            elecCount: The number of electrons in the wave function.
            backend: The backend used to execute quantum circuits.
            shots: Shots per expectation value circuit.
            sampler: Qiskit object to facilitate quantum circuit execution
            mitigationShots: The number of shots to use per column of the assignment matrices
            inputState: The input state on which the ansatz acts. Perfect pairing (PP) and Hartree-Fock (HF) supported.
            doPatchParallelization: Flag to turn on patch parallelization.
        """
        # NOTE Lists of tile circuits contain tiles from all layers
        # All layers are assumed to have the same number of tiles
        # All tiles must have the same number of qubits
        # The total number of qubits must be a multiple of 4.

        self.pretTileCircuits = tileCircuits # pre-transpilation
        self.pretTileQubits = list() # pre-transpilation
        for qubits in tileQubits:
            # Sort pre-transpilation (virtual) qubits within tiles in ascending order. This makes it easier to adhere to the particular bitstring order that is used in the implementation. Whenever a quantum circuit is executed and counts are received the bitstrings are of a longer length than what we want for a particular probability vector. 
            # For example, when we measure on an n-qubit device we get bitstrings on length n, but when we create an assignment matrix for a given tile we need to pick out the results from four qubits (when four qubits constitute a tile as in tUPS). That is, we make bitstrings of length 4 from the bitstring of length n, and we must know
            # exactly which bit in this four qubit bitstring corresponds to which physical qubit that we measured. The order chosen here for the "reduced bitstrings" is one where the most significant bit in the bitstring always corresponds to the physical qubit that the most significant pre-transpilation qubit was mapped to, the second most significant
            # bit corresponds to the second most significant pre-transpilation qubit etc. If the pre-transpilation qubits for a tile are q0, q1, q3, q4, for example, and if the mapping to physical qubits is, for example, q0 --> pq113, q1 --> pq50, q2 --> pq49, q3 --> pq112 then the reduced bitstring order will be (pq112)(pq49)(pq50)(pq113) i.e. for "1001" pq112
            # was measured to be 1, pq49 was measured to be 0 etc. We shall refer to this order as the "tiled M0 bitstring order". Note that we measure the qubits in a way such that the bitstrings of length n (i.e. the non-reduced bitstrings) follow the standard qiskit order with the most significant physical qubit being mapped to the most significant bit.
            # However, we always post-process those n-length bitstrings and cast them to the canonical tiled M0 bitstring order for reduced bitstrings.
            self.pretTileQubits.append(sorted(qubits).copy())
        self.tileParameters = list()
        self.layerCount = layerCount
        self.elecCount = elecCount
        self.backend = backend
        self.expectationValueShots = shots
        self.doPatchParallelization = doPatchParallelization
        self.totQubitCount = 4 * (len(tileQubits) + 1) // 2
        self.inputState = inputState
        if len(inputState) != self.totQubitCount:
            raise ValueError(
                "The number of qubits in the input state does not match the total qubit count according to the tile qubits passed. Note that the total number of qubits must be a multiple of 4, and the number of qubits in every tile must be exactly 4."
            )
        self.qubitsPerTile_Count = len(self.pretTileQubits[0])
        self.posttFullNonParameterizedCircuit = None # The transpiled and non-parameterized quantum circuit with the ansatz on all patches. Doesn't include state preparation or measurement modifications.
        self.posttFullParameterizedCircuit = None # The transpiled and parameterized quantum circuit with the ansatz on all patches. Doesn't include state preparation or measurement modifications.
        self.patches = None # A list of the physical qubits in every patch. ``self.patches[i]`` gives the physical qubits in patch number i. ``self.patches[i][j]`` is the physical qubit in patch i that the virtual qubit j (pre-transpilation) is mapped to i.e. ``self.patches[i]`` is a mapping from virtual to physical qubits in patch i.
        self.nonOverlappingFullyQubitCoveringTiles_Indices = list() # A list of indices corresponding to tiles in ``self.pretTileCircuits`` whose tile qubits combined cover all qubits in the ansatz and with no overlap of qubits between tiles.
        self.tileMitigators_All = list() # The tile mitigators for all tiles in all patches. ``self.tileMitigators_All[i]`` gives the list of all tile mitigators for patch i. The order of the mitigators in ``self.tileMitigators_All[i]`` is the same as the order of tiles in ``self.pretTileCircuits`` i.e. the first element corresponds to the tile mitigator for the first tile in the ansatz (the first tile that is used on the input state).
        self.tileRems_All = list() # The RE mitigators for all patches. ``self.tileRems_All[i]`` gives the list of RE mitigators for patch i. The RE mitigators are 16x16 in the current implementation i.e. they cover 4 qubits each.
        self.posttTileQubits_All = list() # A list of physical qubits in every patch. Similar to ``self.patches` but within a patch qubits are listed tile-wise. ``self.posttTileQubits_All[i][j]`` gives the physical qubits for tile j in patch i. The order of tile qubits in ``self.posttTileQubits_All[i]`` is the same as the order in ``self.pretTileQubits`` i.e. the first element corresponds to the physical qubits that the virtual qubits in the first element of ``self.pretTileQubits`` were mapped to, for the given patch.
        self.mitigatedExpectationValue_All = None # A list of the error mitigated expectation values of the observable ``self.operator`` as they were calculated for every patch.
        self.rawExpectationValue_All = None # A list of the raw expectation values as they were calculated for every patch.

        self.nonOverLappingTiles_Indices = list() # A list of lists of tiles with no overlapping qubits. The tiles are referred to by indices corresponding to elements in ``self.pretTileCircuits``.
        self.backendQubitCount = self.backend.num_qubits
        
        self.logData = ""
        self.mitigatorShots = mitigatorShots # The number of shots per basis state for noise characterization
        self.depth_All = list() # The quantum circuit depth of all patches
        self.gateCount_All = list() # The gate complexity of all patches
        self.nonLocalGates_All = list() # The number of non-local gates for all patches
        self.sampler = sampler
        self.inputStatePrepQc = None
        self.tilesInFirstLayer_Count = len(self.pretTileCircuits) // self.layerCount
        self.redoMitigators = False
        self.conditionNumbers_All = list()

        self.InitializePatches()
        self.TranspileTileCircuits()
        self.ComposeFullCircuit() # non-parameterized full circuit with the ansatz on all patches
        self.GetInputStatePreparationCircuit()
        self.FindNonOverlappingTiles()
        self.GetCircuitComplexityInformation()

    def PrepareAssignmentMatrices(
        self,
        markedTiles : list[int],
        posttTileCircuits : list[QuantumCircuit] | list[None]
    ) -> list[list[NDArray[np.float64]]]:
        """
        Given a list of indices of tiles corresponding to elements in ``self.posttTileCircuits_All[i]``
        and ``self.posttTileQubits_All[i]`` where i is any patch, and given the quantum circuits for the
        marked tiles - in any order and for all patches - get the assignment matrices for all the marked
        tiles for all patches. There must be no overlap of qubits for any of the marked tiles. If execution
        on hardware is turned on, a call to this function with ``len(self.jobResults)`` equal to 0 will 
        only add quantum circuits to ``self.quantumCircuitsBatch``, and no circuits will be executed.
        If ``len(self.jobResults)`` is greater than 0, ``self.jobResults``will be accessed, and the
        assignment matrices will be built.

        Args:
            markedTiles: The indices of the tiles whose assignment matrices should be determined. The indices should correspond to elements in ``self.posttTileCircuits_All[i]`` and ``self.posttTileQubits_All[i]`` where i is any patch. This is equivalent to indices in ``self.pretTileCircuits``.
            posttTileCircuits: The quantum circuits for the marked tiles for ALL patches in any order. The circuits must not be parameterized! Can also be an empty list if it is the second pass of a hardware run.

        Returns:
            For all patches, a list of tile assignment matrices corresponding to the tiles in ``markedTiles``
            for the circuits in ``posttTileCircuits``. Zero-matrices associated with the tiles that are NOT in
            ``markedTiles`` are also returned, so that the desired assignment matrices can be accessed
            conveniently by the caller as (.)[patchIdx][markedTiles[i]]. If ``self.runOnHardware`` is true
            and ``len(self.jobResults)`` is 0, then quantum circuits will be added to ``self.quantumCircuitsBatch``
            and no assignment matrices will be built. Zero-matrices will be returned.
        """
        # Hoeffding's inequality can be used to calculate the necessary number of shots
        # to determine the elements of the assignment matrices with a given accuracy
        # with a given certainty. To get an accuracy of at least 0.01 with a probability
        # of at least 90%, the necessary number of shots is approximately 15000.
        # delta, q = 0.01, 0.90
        # shotCount = int(np.log((2 / (1 - q))) / (2 * delta * delta)) + 1
        # ~15000

        # Compose all the quantum circuits and assign all parameters to 0
        ansatzQc = QuantumCircuit()
        for i in range(len(posttTileCircuits)):
            tmpCircuit = posttTileCircuits[i].copy()
            for param in tmpCircuit.parameters:
                tmpCircuit.assign_parameters({param : 0.0}, inplace = True)

            if i == 0:
                ansatzQc = tmpCircuit.copy()
            else:
                ansatzQc.compose(tmpCircuit, inplace = True)

        assignmentMatrices_All = [
            [np.zeros((2**self.qubitsPerTile_Count, 2**self.qubitsPerTile_Count)) for i in range(self.tilesInFirstLayer_Count)] # Zero-matrices for all tiles within a patch are added - also tiles that are not in ``markedTiles``. But only those in ``markedTiles`` are actually built.
            for j in range(len(self.patches))
        ]

        qcsBatch = list()
        # Loop over all basis states to determine one column at a time of the assignment matrices.
        for i in range(0, 2**self.qubitsPerTile_Count):
            counts = dict()
            
            basisState = format(i, 'b') # rewrite as a bitstring
            basisState = '0' * (self.qubitsPerTile_Count - len(basisState)) + basisState
            
            preQc = ansatzQc.copy_empty_like()
            postQc = ansatzQc.copy_empty_like()

            for patchIdx in range(len(self.patches)):
                for tileIdx in markedTiles:
                    for j in range(0, self.qubitsPerTile_Count):
                        if basisState[j] == '1':
                            preQc.x(self.posttTileQubits_All[patchIdx][tileIdx][self.qubitsPerTile_Count - j - 1])

                        postQc.measure(
                            self.posttTileQubits_All[patchIdx][tileIdx][self.qubitsPerTile_Count - j - 1],
                            self.posttTileQubits_All[patchIdx][tileIdx][self.qubitsPerTile_Count - j - 1]
                        )

            compositeQc = preQc.compose(ansatzQc)
            compositeQc.compose(postQc, inplace = True)

            qcsBatch.append((compositeQc.copy(), None, self.mitigatorShots))

        # Send off all the circuits for execution
        jobCount = len(qcsBatch)
        jobs = [None for i in range(jobCount)]
        for i in range(jobCount):
            jobs[i] = self.sampler.run([qcsBatch[i]])
            print(f">>> Job ID: {jobs[i].job_id()}")
    
        jobResults = list()
        for i in range(0, len(jobs)):
            for j in range(0, len(jobs[i].result())):
                jobResults.append(jobs[i].result()[j])

        for i in range(0, 2**self.qubitsPerTile_Count):
            counts = dict()
            for key, val in jobResults[i].data.items():
                counts = val.get_counts()

            # ``counts`` will contain many, many bitstrings of length equal to the number of qubits on the backend
            for patchIdx in range(len(self.patches)):
                for measurementResult in counts.keys():
                    for tileIdx in markedTiles:
                        reducedBitString = ""
                        for j in range(0, self.qubitsPerTile_Count):
                            reducedBitString += measurementResult[self.backendQubitCount - self.posttTileQubits_All[patchIdx][tileIdx][self.qubitsPerTile_Count-j-1] - 1] # We pick out the bits corresponding to the tile with index ``tileIdx`` in patch ``patchIdx`` following the tiled M0 bitstring order.
                        assignmentMatrices_All[patchIdx][tileIdx][int(reducedBitString, 2)][i] += counts[measurementResult] / self.mitigatorShots

        return assignmentMatrices_All

    def PrepareMitigators(
        self
    ):
        """
        Prepares all the mitigators (both gate and readout) required for tiled M0.
        """
        self.tileMitigators_All = [([0]) * len(self.pretTileQubits) for i in range(len(self.patches))] # Use placeholder 0's for the mitigators
        self.tileRems_All = [([0]) * len(self.pretTileQubits) for i in range(len(self.patches))] # A REM for every tile to remove the readout part of the tile mitigators

        for i in range(len(self.nonOverLappingTiles_Indices)):
            tileCircuitsSubset = list()
            for patchIndex in range(len(self.patches)):                            
                for tileIndex in self.nonOverLappingTiles_Indices[i]:
                    tileCircuitsSubset.append(self.posttTileCircuits_All[patchIndex][tileIndex].copy())

            # Determine the tile A-matrices
            tmpTileAssignmentMatrices_All = self.PrepareAssignmentMatrices(self.nonOverLappingTiles_Indices[i], tileCircuitsSubset)
            
            # Determine the A-matrices for the readout part only
            remCircuit = [self.posttTileCircuits_All[0][0].copy_empty_like()]
            tmp_RE_AssignmentMatrices_All = self.PrepareAssignmentMatrices(self.nonOverLappingTiles_Indices[i], remCircuit)
            
            for patchIndex in range(len(self.patches)):
                for tileIndex in self.nonOverLappingTiles_Indices[i]:
                    self.tileMitigators_All[patchIndex][tileIndex] = np.linalg.inv(tmpTileAssignmentMatrices_All[patchIndex][tileIndex])
                    self.tileRems_All[patchIndex][tileIndex] = np.linalg.inv(tmp_RE_AssignmentMatrices_All[patchIndex][tileIndex])

        # Remove the readout part from the tile mitigators (not for the first pass of a hardware run)
        for patchIndex in range(len(self.patches)):
            for tileIndex in range(self.tilesInFirstLayer_Count):
                self.tileMitigators_All[patchIndex][tileIndex] = np.dot(
                    self.tileMitigators_All[patchIndex][tileIndex],
                    np.linalg.inv(self.tileRems_All[patchIndex][tileIndex])
                )
        
        self.GetConditionNumbers()

    def GetDistributionsForPauliStringHeads(
            self,
            psHeads : list[str],
            parameters : list[float], 
            shotCount : int
    )  -> list[dict[str, int]] | None:
        """
        Gets the raw distributions for the Pauli strings in ``psHeads``.

        Args:
            psHeads: Each of the Pauli string heads to execute a quantum circuit for (with measurement modifications applied as dictated by the head).
            parameters: The variational parameters to insert in the circuit. The order should be the same as the order in ``self.pretTileCircuits``.
            shotCount: The number of shots to use for each quantum circuit (i.e. for each head).
        
        Returns:
            A list with the raw distributions for the Pauli string heads in the same order as ``psHeads``.
        """
        # If the mitigators have not been made yet: make them
        if len(self.tileMitigators_All) == 0 or self.redoMitigators == True:
            self.PrepareMitigators()
            self.redoMitigators = False

        self.ParameterizeAndComposeFullCircuit(parameters)

        qcsBatch = list()
        for ps in psHeads:
            measurementModsQc = self.posttFullParameterizedCircuit.copy_empty_like()
            for patch in self.patches:
                for i in range(len(ps)):
                    posttQubitIndex = patch[self.totQubitCount - i - 1] # patch is a mapping from virtual to physical qubit index
                    if ps[i] == 'X':
                        measurementModsQc.h(posttQubitIndex)
                    elif ps[i] == 'Y':
                        measurementModsQc.sdg(posttQubitIndex)
                        measurementModsQc.h(posttQubitIndex)

                for posttQubitIndex in patch:
                    measurementModsQc.measure(posttQubitIndex, posttQubitIndex)

            measurementModsQc = transpile(measurementModsQc, backend = self.backend, layout_method = "trivial", optimization_level = 0, routing_method = "none", seed_transpiler = 1)
            compositeCircuit = self.inputStatePrepQc.compose(self.posttFullParameterizedCircuit)
            compositeCircuit.compose(measurementModsQc, inplace = True)

            qcsBatch.append((compositeCircuit, None, shotCount))

        # Send off all the circuits for execution
        jobCount = len(qcsBatch)
        jobs = [None for i in range(jobCount)]
        for i in range(jobCount):
            jobs[i] = self.sampler.run([qcsBatch[i]])
            print(f">>> Job ID: {jobs[i].job_id()}")
    
        jobResults = list()
        for i in range(0, len(jobs)):
            for j in range(0, len(jobs[i].result())):
                jobResults.append(jobs[i].result()[j])

        distributionsForAllHeads = list()
        for i in range(len(jobResults)):
            for key, val in jobResults[i].data.items():
                distributionsForAllHeads.append(val.get_counts().copy())
            
        return distributionsForAllHeads
    
    def GetProbabilityVectorsFromDistribution(
        self,
        distribution: dict[str, int],
        shotCount: int
    )   -> list[list[float]]:
        """
        Returns probability vectors (one for each patch) from a distribution. 

        Args:
            distribution: The distribution as a dictionary.
            shotCount: The number of shots that was used to make the distribution.

        Returns:
            The distribution as a probability vector.
        """
        p_All = list()
        for patch in self.patches:
            p = [0 for i in range(2**len(patch))]
            for measurementResult in distribution.keys():
                reducedBitString = ""
                for i in range(0, len(patch)):
                    reducedBitString += measurementResult[self.backendQubitCount - patch[len(patch)-i-1] - 1] # We pick out the bits from the relevant patch following the tiled M0 bitstring order.
                p[int(reducedBitString, 2)] += distribution[measurementResult] / shotCount
            p_All.append(p.copy())

        return p_All

    def GetExpectationValues(
        self,
        p_All: list[list[float]],
        paulisAndWeights: dict[str, float]
    )   -> tuple[float, float]:
        """
        Calculates the sum of the mitigated and raw expectation values (weighted) of the Pauli strings in ``paulisAndWeights``.

        Args:
            p_All: The probability vectors (one for each patch) from which the expectation values are calculated.
            paulisAndWeights: A dictionary with the Pauli strings (keys) whose expectation values are calculated. The weights are the dictionary values.
        
        Returns:
            The sum of the mitigated and raw expectation values (weighted) of the Pauli strings.
        """
        mitigatedExpectationValues_All = [0.0 for i in range(len(self.patches))]
        rawExpectationValues_All = [0.0 for i in range(len(self.patches))]
        
        for patchIdx in range(len(self.patches)):
            # Make a list of all the mitigators. The last mitigators in the list will be applied to the probability vector first. Also make a list
            # of the virtual qubits corresponding to all the mitigators. We can use the virtual qubit indices (instead of physical qubit indices)
            # because we know that the probability vectors and assignment matrices follow the tiled M0 bitstring order. This means that at this point
            # we can ignore that there ever was a mapping to physical qubits.
            mitigators = self.tileMitigators_All[patchIdx] * self.layerCount
            mitigatedQubits = self.pretTileQubits * self.layerCount
            for tileIdx in self.nonOverlappingFullyQubitCoveringTiles_Indices:
                mitigators.append(self.tileRems_All[patchIdx][tileIdx])
                mitigatedQubits.append(self.pretTileQubits[tileIdx])
            
            pNoisy = p_All[patchIdx].copy()
            p = p_All[patchIdx].copy()

            for j in range(len(mitigators) - 1, -1, -1):
                ApplyMitigator(p, mitigatedQubits[j], mitigators[j])

            for ps, weight in paulisAndWeights.items():
                if ps == 'I' * self.totQubitCount:
                    mitigatedExpectationValues_All[patchIdx] += weight
                    rawExpectationValues_All[patchIdx] += weight
                    continue
                mitigatedExpectationValues_All[patchIdx] += CalculateExpectationValueOfPauliString(ps, p) * weight
                rawExpectationValues_All[patchIdx] += CalculateExpectationValueOfPauliString(ps, pNoisy) * weight

        return mitigatedExpectationValues_All[0], rawExpectationValues_All[0] # patches turned off for SQ version, so there is only a zeroth element
    
    def InitializePatches(
        self,
        savedPatches : list[list[int]] = None
    ):
        """
        Gets patches from ``savedPatches`` (only if a previously executed job is loaded)
        or calls ``self.FindPatches()`` to find patches. If patch parallelization is turned off,
        a circuit is transpiled and the initial index layout is used for the virtual to physical mapping.

        Args:
            savedPatches: Only for ``self.loadJob == True``. A list of saved patches that matches the patches from the jobs that will be loaded.
        """
        tmpCircuit = self.pretTileCircuits[0].copy()
        for i in range(1, self.tilesInFirstLayer_Count):
            tmpCircuit.compose(self.pretTileCircuits[i], inplace = True)
        tmpCircuit = transpile(tmpCircuit, self.backend, optimization_level = 3, seed_transpiler = 1)                    
        self.patches = [tmpCircuit.layout.initial_index_layout()[0:self.totQubitCount]]

    
    def TranspileTileCircuits(self):
        """
        Transpiles the tile circuits in ``self.pretTileCircuits`` and inserts swap gates to undo routing swaps.
        Also determines the physical qubits associated with all the transpiled tiles. ``self.patches`` must
        be initialized before calling.
        """
        # Transpile all the tile circuits in the first layer
        self.posttTileCircuits_All = list()
        for patch in self.patches:
            passManager = generate_preset_pass_manager(
                optimization_level = 3,
                backend = self.backend,
                initial_layout = patch, # Specifying the initial layout here is necessary to map the tiles to the correct patches.
                seed_transpiler = 1
            )

            transpiledTileCircuits = [0 for i in range(self.tilesInFirstLayer_Count)]
            for i in range(len(self.pretTileCircuits)):
                tmpCircuit = passManager.run(self.pretTileCircuits[i])
                add_permutation_gate(tmpCircuit, tmpCircuit.layout.routing_permutation(), self.backend.coupling_map) # Add swap gates to undo the swaps inserted during routing
                transpiledTileCircuits[i] = transpile(tmpCircuit, backend = self.backend, optimization_level = 1, layout_method = "trivial", routing_method = "none", seed_transpiler = 1) # Transpile the swap gates. Set flags to make sure qiskit does not reorder anything.
            
            self.posttTileCircuits_All.append(transpiledTileCircuits.copy())
        
        # Determine the physical qubits associated with all the post-transpilation tiles.
        # Since the pre-transpilation qubits in ``self.pretTileQubits`` are sorted in
        # ascending order for all tiles this means that the resulting post-transpilation tile qubits
        # will be subsequences of their corresponding ``patch``. This makes it easier to adhere
        # to the tiled M0 bitstring order later and to apply the tile mitigators to the probability vectors
        # when we calculate expectation values.
        self.posttTileQubits_All = list()
        for patch in self.patches:
            posttTileQubits = list()
            for tileIndex in range(len(self.pretTileQubits)):
                tmpQubits = list()
                for pretQubit in self.pretTileQubits[tileIndex]:
                    tmpQubits.append(patch[pretQubit])
                posttTileQubits.append(tmpQubits.copy())

            self.posttTileQubits_All.append(posttTileQubits.copy())
    
    def ParameterizeAndComposeFullCircuit(self, parameters):
        """
        Parameterizes and composes the full post-transpilation circuit with the ansatz on all patches.
        ``self.posttTileCircuits_All`` must be initialized beforehand. The result is stored in
        ``self.posttFullParameterizedCircuit``
        """
        expectedParameterCount = len(self.posttTileCircuits_All[0][0].parameters) * len(self.pretTileCircuits)
        if (len(parameters) != expectedParameterCount):
            raise ValueError(
                f"The number of parameters passed, {len(parameters)}, does not match the number of parameters expected, {expectedParameterCount}. This is under the assumption that all tiles have the same number of parameters."
            )

        self.posttFullParameterizedCircuit = self.posttTileCircuits_All[0][0].copy_empty_like()

        for patchIndex in range(len(self.patches)):
            idx = 0
            for i in range(len(self.pretTileCircuits)):
                tileCircuitParameterized = self.posttTileCircuits_All[patchIndex][i].copy()
                paramCount = len(tileCircuitParameterized.parameters)
                for j in range(paramCount):
                    # ``tileCircuitParameterized.parameters[0]`` with 0 as index because ``assign_parameters`` with ``inplace = True`` mutates the circuit
                    tileCircuitParameterized.assign_parameters({tileCircuitParameterized.parameters[0] : parameters[idx]}, inplace = True)
                    idx += 1

                self.posttFullParameterizedCircuit.compose(tileCircuitParameterized, inplace = True)

    def ComposeFullCircuit(self):
        """
        Composes the full post-transpilation circuit with the ansatz on all patches (not parameterized).
        ``self.posttTileCircuits_All`` must be initialized beforehand. The result is stored in
        ``self.posttFullNonParameterizedCircuit``.
        """
        self.posttFullNonParameterizedCircuit = self.posttTileCircuits_All[0][0].copy_empty_like()
        
        for patchIndex in range(len(self.patches)):
            for i in range(len(self.pretTileCircuits)):
                self.posttFullNonParameterizedCircuit.compose(self.posttTileCircuits_All[patchIndex][i].copy(), inplace = True)

    def FindNonOverlappingTiles(
    self
    ):
        """
        Identify tiles that have no overlapping qubits and whose mitigators can thus be determined in parallel.
        Results are saved in ``self.nonOverlappingTiles_Indices`` as a list of lists with indices of tiles corresponding
        to elements in ``self.pretTileCircuits``.
        """
        isProcessed = [False for i in range(self.tilesInFirstLayer_Count)]

        for i in range(self.tilesInFirstLayer_Count):
            if isProcessed[i] == True:
                continue
            # -- else --
            markedTiles = list()
            markedQubits = list()
            for j in range(i, self.tilesInFirstLayer_Count):
                if isProcessed[j] == True:
                    continue
                # -- else --
                giveMark = True
                tmpMarkedQubits = list()
                for k in range(len(self.pretTileQubits[j])):
                    if not (self.pretTileQubits[j][k] in markedQubits):
                        tmpMarkedQubits.append(self.pretTileQubits[j][k])
                    else:
                        giveMark = False
                        break
                if giveMark:
                    markedTiles.append(j)
                    isProcessed[j] = True
                    for qubitIdx in tmpMarkedQubits:
                        markedQubits.append(qubitIdx)
            
            if len(markedTiles) > 0:
                self.nonOverLappingTiles_Indices.append(markedTiles)

            # Check if the marked tiles cover all qubits (used for choosing REMs later)
            if (len(markedQubits) == self.totQubitCount):
                self.nonOverlappingFullyQubitCoveringTiles_Indices = markedTiles
    
    def GetCircuitComplexityInformation(
        self
    ):
        """
        For all patches, gets the circuit depth and gate complexity.
        Also gets the number of non-local gates in each patch.
        Results are saved in ``self.depth_All``, ``self.gateCount_All``
        and ``self.nonLocalGates_All``.
        """
        self.depth_All = [0 for i in range(len(self.patches))]
        self.gateCount_All = [0 for i in range(len(self.patches))]
        self.nonLocalGates_All = [0 for i in range(len(self.patches))]
        for patchIndex in range(len(self.patches)):
            patchQc = self.posttTileCircuits_All[0][0].copy_empty_like()
            for layerNumber in range(self.layerCount):
                for i in range(len(self.pretTileQubits)):
                    patchQc.compose(self.posttTileCircuits_All[patchIndex][i], inplace = True)
            
            self.depth_All[patchIndex] = patchQc.depth()
            self.gateCount_All[patchIndex] = patchQc.size()
            self.nonLocalGates_All[patchIndex] = patchQc.num_nonlocal_gates()

    
    def GetInputStatePreparationCircuit(
        self
    ):
        """
        Prepare the circuit that prepares the input state (on all patches).
        The circuit is saved in ``self.inputStatePrepQc``.
        """
        self.inputStatePrepQc = self.posttTileCircuits_All[0][0].copy_empty_like()
        for patch in self.patches:        
            for i in range(len(self.inputState)):
                if (self.inputState[i] == '1'):
                    # Remember to change to qiskit ordering  
                    idxInQiskitOrder = None
                    if i % 2 == 0:
                        idxInQiskitOrder = i // 2
                    else:
                        idxInQiskitOrder = (i // 2) + (self.totQubitCount // 2)

                    self.inputStatePrepQc.x(patch[idxInQiskitOrder])
    
    def GetConditionNumbers(self):
        """
        Get the condition numbers for the tile mitigators in all patches.
        Results are stored in ``self.conditionNumbers_All``.
        """
        self.conditionNumbers_All = list()
        for i in range(len(self.patches)):
            conditionNumbers = list()
            for j in range(len(self.tileMitigators_All[i])):
                U, S, Vdag = np.linalg.svd(self.tileMitigators_All[i][j], full_matrices = False)
                conditionNumbers.append(S[0] / S[-1])
            
            self.conditionNumbers_All.append(conditionNumbers.copy())
        
    def LogTiledM0(
        self,
        mitigatedExpectationValue_All,
        rawExpectationValue_All
    ):
        """
        Logs results from tiled M0. Also returns the average error mitigated expectation value over all good patches (based on condition numbers).
        """
        self.logData = "\n\n---- ERROR MITIGATION ----\n"
        self.logData += "\nInput state: " + self.inputState
        self.logData += "\n\nPatches:\n"
        for patch in self.patches:
            for qubit in patch:
                self.logData += str(qubit) + ", "
            self.logData += "\n"

        for i in range(len(self.patches)):
            self.logData += "\n\n--- PATCH " + str(i + 1)
            self.logData += "\nCondition numbers: " + str(self.conditionNumbers_All[i])
            self.logData += "\nTiled M0 expectation value: " + str(mitigatedExpectationValue_All[i])
            self.logData += "\nRaw expectation value: " + str(rawExpectationValue_All[i])
            self.logData += "\nDepth complexity: " + str(self.depth_All[i])
            self.logData += "\nGate complexity: " + str(self.gateCount_All[i])
            self.logData += "\nNumber of non-local gates: " + str(self.nonLocalGates_All[i])

        f = open("tiledM0.log", 'a')
        f.write(self.logData)
        f.close()