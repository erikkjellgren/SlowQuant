import numpy as np
from numpy import (ndarray, dtype)
from numpy._typing import NDArray
import random
import sys
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.transpiler.passes import SabreLayout
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import (SamplerV2, Batch, IBMBackend, QiskitRuntimeService)
from qiskit_aer import AerSimulator
from slowquant.qiskit_interface.util import add_permutation_gate
from tiled_m0_helper import (ApplyMitigator,
                             CalculateExpectationValueOfPauliString,
                             GroupPauliStringsByQwc,
                             CalculateAllPsGroupOneNorms)

class TiledM0:
    def __init__(
        self,
        tileCircuits : list[QuantumCircuit],
        tileQubits : list[list[int]],
        layerCount: int,
        elecCount: int,
        operator: dict[str, np.float64],
        backend: IBMBackend | AerSimulator,
        expectationValueShots: int,
        mitigatorShots : int = 15000,
        inputState : str = "PP",
        doPatchParallelization : bool = False,
        loadJob : bool = False,
        savedPatches : list[list[int]] = None,
        jobNames : list[str] = None,
        service : QiskitRuntimeService = None
    ) -> None:
        """
        Initialize the tiled M0 driver.

        Args:
            tileCircuits: A list with the quantum circuits for every tile in the first layer of the ansatz (pre-transpilation). The order of the quantum circuits should be the same as the order in which the tiles appear in the ansatz, starting with the tile that is used on the input state first.
            tileQubit: A tile-wise list with the qubits in every tile in the first layer of the ansatz (pre-transpilation). tileQubits[i] should be the qubits associated with tileCircuits[i].
            layerCount: The number of layers to be used in the ansatz.
            elecCount: The number of electrons in the wave function.
            operator: The operator to be measured. A weighted sum of Pauli strings.
            backend: The backend used to execute quantum circuits.
            expectationValueShots: The number of shots to use in total to determine the expectation value of ``operator``.
            mitigatorShots: The number of shots per column in the assignment matrices.
            inputState: The input state on which the ansatz acts. Perfect pairing (PP) and Hartree-Fock (HF) supported.
            doPatchParallelization: Flag to turn on patch parallelization.
            loadJob: Flag for loading previously executed jobs.
            savedPatches: If ``loadJob`` is true, the patches that were used for the job that is loaded.
            jobNames: If ``loadJob`` is true, the job names.
            service: Used to get data from IBM servers if ``loadJob`` is true.
        """
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
        self.layerCount = layerCount
        self.elecCount = elecCount
        self.operator = operator
        self.backend = backend
        self.expectationValueShots = expectationValueShots
        self.mitigatorShots = mitigatorShots
        print("OPERATOR", self.operator)
        self.totQubitCount = len(list(self.operator.keys())[0])
        if (self.totQubitCount % 4 != 0):
            raise ValueError("The total number of qubits must be a multiple of 4.") # Because the current implementation assumes RE mitigators of size 16x16.

        self.inputState = inputState
        if len(self.inputState) != self.totQubitCount:
            raise ValueError("The number of qubits in the input state doesn't match the number of qubits in the operator Pauli strings.")
        self.doPatchParallelization = doPatchParallelization

        self.qubitsPerTile_Count = len(self.pretTileQubits[0])
        
        self.runOnHardware = False
        self.backendClops = None
        if not (self.backend.name[0:3] == "aer"):
            self.backendClops = self.backend.configuration()._data["clops_h"]
            #self.backendResetDuration = backend.target["reset"][(0,)].duration
            #self.backendReadoutDuration = backend.target["measure"][(0,)].duration
            self.runOnHardware = True
        if self.runOnHardware == False and self.doPatchParallelization == True:
            print("Patch parallelization is turned off for simulations.")
            self.doPatchParallelization = False
        
        self.posttFullNonParameterizedCircuit = None # The transpiled and non-parameterized quantum circuit with the ansatz on all patches. Doesn't include state preparation or measurement modifications.
        self.posttFullParameterizedCircuit = None # The transpiled and parameterized quantum circuit with the ansatz on all patches. Doesn't include state preparation or measurement modifications.
        self.patches = None # A list of the physical qubits in every patch. ``self.patches[i]`` gives the physical qubits in patch number i. ``self.patches[i][j]`` is the physical qubit in patch i that the virtual qubit j (pre-transpilation) is mapped to i.e. ``self.patches[i]`` is a mapping from virtual to physical qubits in patch i.
        self.nonOverlappingFullyQubitCoveringTiles_Indices = list() # A list of indices corresponding to tiles in ``self.pretTileCircuits`` whose tile qubits combined cover all qubits in the ansatz and with no overlap of qubits between tiles.
        self.tileMitigators_All = list() # The tile mitigators for all tiles in all patches. ``self.tileMitigators_All[i]`` gives the list of all tile mitigators for patch i. The order of the mitigators in ``self.tileMitigators_All[i]`` is the same as the order of tiles in ``self.pretTileCircuits`` i.e. the first element corresponds to the tile mitigator for the first tile in the ansatz (the first tile that is used on the input state).
        self.tileRems_All = list() # The RE mitigators for all patches. ``self.tileRems_All[i]`` gives the list of RE mitigators for patch i. The RE mitigators are 16x16 in the current implementation i.e. they cover 4 qubits each.
        self.posttTileQubits_All = list() # A list of physical qubits in every patch. Similar to ``self.patches` but within a patch qubits are listed tile-wise. ``self.posttTileQubits_All[i][j]`` gives the physical qubits for tile j in patch i. The order of tile qubits in ``self.posttTileQubits_All[i]`` is the same as the order in ``self.pretTileQubits`` i.e. the first element corresponds to the physical qubits that the virtual qubits in the first element of ``self.pretTileQubits`` were mapped to, for the given patch.
        self.mitigatedExpectationValue_All = None # A list of the error mitigated expectation values of the observable ``self.operator`` as they were calculated for every patch.
        self.rawExpectationValue_All = None # A list of the raw expectation values as they were calculated for every patch.
        self.quantumCircuitsBatch = list() # A list to which all quantum circuits that need to be executed are added (only used for hardware runs). This is used to gather all circuits before they are sent off for execution.
        self.jobResults = list() # A list with the results from all quantum circuit executions (both related to noise characterization and expectation value estimation). Only used for hardware runs.
        self.jobResultsIndex = 0 # An index to keep track of which job result to process next.
        self.conditionNumbers_All = list() # A single condition number for every patch to judge the quality of the tile assignment matrices / mitigators. An average of the condition numbers for all unique tile assignment matrices within a patch is calculated.
        self.conditionNumbersOverTime = list() # Only used for ``NoiseCharacterizationRun``. ``self.conditionNumbersOverTime[i]`` is the same as ``self.conditionNumbers_All`` but with the index i referring to a specific noise characterization round. Used to see how the quality of a patch, as judged by the condition number metric, can change over time.
        self.chosenTileMitigators_All = list()
        self.conditionNumbersProduct_All = list()
        self.operatorGroupedByQwc = GroupPauliStringsByQwc(self.operator) # A list with lists of Pauli strings in ``operator`` that display qubit wise commutativity i.e. they can be measured simultaneously.
        self.qwcGroupsOneNorms = CalculateAllPsGroupOneNorms(self.operatorGroupedByQwc, self.operator) # For all qwc groups (the elements in ``self.operatorGroupedByQwc``), the sum of the absolute value of the weights of the Pauli strings. This is used for shot allocation.
        self.operatorSumOfWeights = 0.0 # The sum of the weights of all the Pauli strings in ``self.operator``, excluding the weight of the all-identity Pauli string.
        for groupOneNorm in self.qwcGroupsOneNorms:
            self.operatorSumOfWeights += groupOneNorm
        self.nonOverLappingTiles_Indices = list() # A list of lists of tiles with no overlapping qubits. The tiles are referred to by indices corresponding to elements in ``self.pretTileCircuits``.
        self.firstPass = True # A flag to indicate if we are on the first pass of a hardware run
        self.backendQubitCount = self.backend.num_qubits
        self.qwcGroups_ShotCounts = [0 for i in range(len(self.operatorGroupedByQwc))] # The number of shots used for the QWC groups for the expectation value estimations. ``self.qwcGroups_ShotCounts[i]`` is the number of shots used for the group ``self.operatorGroupedByQwc[i]``.
        self.posttFullTrivialCircuit = None # The transpiled quantum circuit with the ansatz on all patches will all parameters set to 0 (only used for normal M0 runs)
        self.doNormalM0 = False # Gets set to true if ``self.NormalM0_Run`` is called.
        self.normal_M0_Mitigators_All = list() # The normal M0 mitigators for every patch (only used for normal M0 runs)
        self.logData = ""
        self.depth_All = list() # The quantum circuit depth of all patches
        self.gateCount_All = list() # The gate complexity of all patches
        self.nonLocalGates_All = list() # The number of non-local gates for all patches
        self.loadJob = loadJob
        self.jobNames = jobNames
        self.service = service
        self.inputStatePrepQc = None
        self.parameters = None

        if self.backend.name != 'aer_simulator':
            self.InitializePatches(savedPatches)
        elif self.backend.name == 'aer_simulator':
            self.patches = [[i for i in range(self.totQubitCount)]]
        self.TranspileTileCircuits()
        self.GetInputStatePreparationCircuit()
        self.FindNonOverlappingTiles()
        self.GetCircuitComplexityInformation()

    def Run(
        self,
        parameters : list[float]
    )   -> np.float64:
        """
        Calculate the expectation value of ``self.operator`` using tiled M0.

        Args:
            The parameters to parameterize the circuit with.

        Returns:
            The average of the error mitigated expectation values for all patches which satisfy
            that the condition numbers for the tile mitigators are below a certain threshold. This is to
            assure the quality of a given patch. Also outputs a log file with results for all patches
            including raw expectation values.
        """
        self.ParameterizeAndComposeFullCircuit(parameters)
        self.quantumCircuitsBatch = list()
        self.jobResults = list()
        self.jobResultsIndex = 0

        if self.runOnHardware == False:
            self.RunPass()
        else:
            if self.loadJob == False:
                self.RunPass() # Pass 1: fills ``self.quantumCircuitsBatch`` with circuits.

                jobNamesToSave = list()
                with Batch(backend = self.backend) as batch:
                    sampler = SamplerV2()
                    #sampler.options.dynamical_decoupling.enable = True
                    #sampler.options.twirling.enable_gates = True

                    circuitsPerJob = 1
                    jobCount = int((len(self.quantumCircuitsBatch) + circuitsPerJob - 1) / circuitsPerJob)
                    jobs = [None for i in range(jobCount)]
                    for i in range(jobCount):
                        jobs[i] = sampler.run(self.quantumCircuitsBatch[i * circuitsPerJob : (i + 1) * circuitsPerJob])
                        print(f">>> Job ID: {jobs[i].job_id()}")
                        jobNamesToSave.append(jobs[i].job_id())

                f = open("jobNames.log", 'a')
                f.write("\n###################\n")
                for jobName in jobNamesToSave:
                    f.write(str(jobName) + '\n')
                f.close()
                        
            else:
                jobs = [None for i in range(len(self.jobNames))]
                for i in range(len(self.jobNames)):
                    jobs[i] = self.service.job(self.jobNames[i])
                    print(f">>> Job retrieved: {self.jobNames[i]}")

            for i in range(0, len(jobs)):
                for j in range(0, len(jobs[i].result())):
                    self.jobResults.append(jobs[i].result()[j])
        
            self.RunPass() # Pass 2: does the actual calculations

        # Logs results. ``self.PrepareLogAndGetAverageOverPatches()`` also returns an average error mitigated expectation value using samples from good patches only.
        return self.LogTiledM0()

    def RunPass(
        self
    )   -> tuple[list[np.float64], list[np.float64]]:
        """
        If hardware execution is turned off, a single call to this function will give the error mitigated
        and raw expectation values of ``self.operator``. For hardware execution, the first call to the function
        will populate ``self.quantumCircuitsBatch`` with all the quantum circuits needed for the noise characterization
        and expectation value estimations. Once the quantum circuits have been executed and retrieved in ``self.jobResults``,
        a second call to the function will do all the calculations and return the expectation values.

        The function is called through ``self.Run``.

        Returns:
            The error mitigated and raw expectation values for every patch if they are calculated and otherwise lists of 0's
        """
        self.tileMitigators_All = [([0]) * len(self.pretTileQubits) for i in range(len(self.patches))] # Use placeholder 0's for the mitigators
        self.tileRems_All = [([0]) * len(self.pretTileQubits) for i in range(len(self.patches))] # A REM for every tile to remove the readout part of the tile mitigators
        self.firstPass = (len(self.jobResults) == 0)

        for i in range(len(self.nonOverLappingTiles_Indices)):
            tileCircuitsSubset = list()
            if (self.runOnHardware == False or self.firstPass == True): # If we run on a simulator or if it is the first pass of a hardware run
                for patchIndex in range(len(self.patches)):                            
                    for tileIndex in self.nonOverLappingTiles_Indices[i]:
                        tileCircuitsSubset.append(self.posttTileCircuits_All[patchIndex][tileIndex].copy())

            # Determine the tile A-matrices
            tmpTileAssignmentMatrices_All = self.PrepareAssignmentMatrices(self.nonOverLappingTiles_Indices[i], tileCircuitsSubset)
            
            # Determine the A-matrices for the readout part only
            remCircuit = [self.posttTileCircuits_All[0][0].copy_empty_like()]
            tmp_RE_AssignmentMatrices_All = self.PrepareAssignmentMatrices(self.nonOverLappingTiles_Indices[i], remCircuit)
            
            if (self.runOnHardware == False or self.firstPass == False): # If we run on a simulator or if it is the second pass of a hardware run
                for patchIndex in range(len(self.patches)):
                    for tileIndex in self.nonOverLappingTiles_Indices[i]:
                        self.tileMitigators_All[patchIndex][tileIndex] = np.linalg.inv(tmpTileAssignmentMatrices_All[patchIndex][tileIndex])
                        self.tileRems_All[patchIndex][tileIndex] = np.linalg.inv(tmp_RE_AssignmentMatrices_All[patchIndex][tileIndex])

        # Remove the readout part from the tile mitigators (not for the first pass of a hardware run)
        if self.runOnHardware == False or self.firstPass == False:
            for patchIndex in range(len(self.patches)):
                for tileIndex in range(len(self.pretTileCircuits)):
                    self.tileMitigators_All[patchIndex][tileIndex] = np.dot(
                        self.tileMitigators_All[patchIndex][tileIndex],
                        np.linalg.inv(self.tileRems_All[patchIndex][tileIndex])
                    )

            self.GetConditionNumbers()            

        self.mitigatedExpectationValue_All, self.rawExpectationValue_All = self.CalculateExpectationValue() # Big call


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

        if (self.runOnHardware == False or self.firstPass == True): # If we run on a simulator or if it is the first pass of a hardware run
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

        assignmentMatrices_All = list()
        if (self.runOnHardware == False or self.firstPass == False): # If we run on a simulator or if it is the second pass of a hardware run
            assignmentMatrices_All = [
                [np.zeros((2**self.qubitsPerTile_Count, 2**self.qubitsPerTile_Count)) for i in range(len(self.pretTileCircuits))] # Zero-matrices for all tiles within a patch are added - also tiles that are not in ``markedTiles``. But only those in ``markedTiles`` are actually built.
                for j in range(len(self.patches))
            ]

        # Loop over all basis states to determine one column at a time of the assignment matrices.
        for i in range(0, 2**self.qubitsPerTile_Count):
            counts = dict()
            if (self.runOnHardware == False or self.firstPass == True): # If we run on a simulator or if it is the first pass of a hardware run
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

                if self.runOnHardware == False:
                    counts = self.backend.run(compositeQc, shots = self.mitigatorShots, seed_simulator = random.randint(0, int(1e9)), method = "statevector").result().get_counts()
                else:
                    self.quantumCircuitsBatch.append((compositeQc.copy(), None, self.mitigatorShots))
                    continue # <--- OBS !!!

            else: # If it is the second pass of a hardware run
                for key, val in self.jobResults[self.jobResultsIndex].data.items():
                    counts = val.get_counts()
                self.jobResultsIndex += 1

            # ``counts`` will contain many, many bitstrings of length equal to the number of qubits on the backend
            for patchIdx in range(len(self.patches)):
                for measurementResult in counts.keys():
                    for tileIdx in markedTiles:
                        reducedBitString = ""
                        for j in range(0, self.qubitsPerTile_Count):
                            reducedBitString += measurementResult[self.backendQubitCount - self.posttTileQubits_All[patchIdx][tileIdx][self.qubitsPerTile_Count-j-1] - 1] # We pick out the bits corresponding to the tile with index ``tileIdx`` in patch ``patchIdx`` following the tiled M0 bitstring order.
                        assignmentMatrices_All[patchIdx][tileIdx][int(reducedBitString, 2)][i] += counts[measurementResult] / self.mitigatorShots

        return assignmentMatrices_All

    def CalculateExpectationValue(
        self
    )   -> tuple[list[np.float64], list[np.float64]]:
        """
        Calculates the expectation value of ``self.operator`` using tiled M0 error mitigation (or normal M0 if the flag is set)
        and a number of shots equal to ``self.expectationValueShots``. Assumes that the mitigators have already been created.
        If hardware execution is turned on and ``len(self.jobResults)`` is equal to 0 then quantum circuits are added to ``self.quantumCircuitsBatch``
        and no expectation values are calculated.

        Returns:
            A list of error mitigated expectation values and a list of raw expectation values if they are calculated and otherwise lists of 0's.
        """
        self.mitigatedExpectationValue_All = [0 for i in range(len(self.patches))]
        self.rawExpectationValue_All = [0 for i in range(len(self.patches))]
        # Loop over all Pauli string groups
        for psGroup in range(0, len(self.operatorGroupedByQwc)):
            shotCount = int((self.qwcGroupsOneNorms[psGroup] / self.operatorSumOfWeights) * self.expectationValueShots) + 1
            self.qwcGroups_ShotCounts[psGroup] = shotCount
            print(shotCount, end = ", ")
            shotCountSaved = shotCount
            maxShotCount = 10000000 # If shotCount > 10 million, we need to split up the job
            while shotCount > 0:
                shotCountRestricted = min(maxShotCount, shotCount)
                p_All = self.GetProbabilityVectorForPauliStringGroup(self.operatorGroupedByQwc[psGroup], shotCountRestricted)

                if (self.runOnHardware == True and self.firstPass == True): # If it is the first pass of a hardware run
                    shotCount -= maxShotCount
                    continue # <--- OBS
                # -- else --
                for patchIdx in range(len(self.patches)):
                    mitigators = list()
                    mitigatedQubits = list()
                    if self.doNormalM0 == False:
                        # Make a list of all the mitigators. The last mitigators in the list will be applied to the probability vector first. Also make a list
                        # of the virtual qubits corresponding to all the mitigators. We can use the virtual qubit indices (instead of physical qubit indices)
                        # because we know that the probability vectors and assignment matrices follow the tiled M0 bitstring order. This means that at this point
                        # we can ignore that there ever was a mapping to physical qubits.
                        for tileIdx in range(len(self.tileMitigators_All[patchIdx])):
                            if self.chosenTileMitigators_All[patchIdx][tileIdx] == True:
                                mitigators.append(self.tileMitigators_All[patchIdx][tileIdx])
                                mitigatedQubits.append(self.pretTileQubits[tileIdx])
                                
                        mitigators *= self.layerCount
                        mitigatedQubits *= self.layerCount
                        
                        for tileIdx in self.nonOverlappingFullyQubitCoveringTiles_Indices:
                            mitigators.append(self.tileRems_All[patchIdx][tileIdx])
                            mitigatedQubits.append(self.pretTileQubits[tileIdx])
                            
                    else: # If we run normal M0 (requires setup that is done in ``self.NormalM0_Run``)
                        mitigators = [self.normal_M0_Mitigators_All[patchIdx]]
                        mitigatedQubits = [[i for i in range(self.totQubitCount)]]

                    pNoisy = p_All[patchIdx].copy()
                    p = p_All[patchIdx].copy()

                    for mitigatorIdx in range(len(mitigators) - 1, -1, -1):
                        ApplyMitigator(p, mitigatedQubits[mitigatorIdx], mitigators[mitigatorIdx])
                    
                    for ps in self.operatorGroupedByQwc[psGroup]:
                        if ps == 'I' * self.totQubitCount:
                            self.mitigatedExpectationValue_All[patchIdx] += (shotCountRestricted / shotCountSaved) * self.operator[ps]
                            self.rawExpectationValue_All[patchIdx] += (shotCountRestricted / shotCountSaved) * self.operator[ps]
                            continue
                        self.mitigatedExpectationValue_All[patchIdx] += (shotCountRestricted / shotCountSaved) * CalculateExpectationValueOfPauliString(ps, p) * self.operator[ps]
                        self.rawExpectationValue_All[patchIdx] += (shotCountRestricted / shotCountSaved) * CalculateExpectationValueOfPauliString(ps, pNoisy) * self.operator[ps]
                        
                shotCount -= maxShotCount

        print()

        if (self.runOnHardware == False or self.firstPass == False):
            for patchIdx in range(len(self.patches)):
                print(" --- Patch", patchIdx + 1)
                print("Mitigated:", self.mitigatedExpectationValue_All[patchIdx], "\nRaw:", self.rawExpectationValue_All[patchIdx])

        return self.mitigatedExpectationValue_All, self.rawExpectationValue_All    

    
    def GetProbabilityVectorForPauliStringGroup(
            self,
            psGroup: list[str],
            shotCount: int
    )  -> list[list[np.float64]] | None:
        """
        Execute the quantum circuit needed to determine the expectation values of the Pauli strings in ``psGroup``.
        The Pauli strings are assumed to be grouped based on qubit-wise commutativity. The function uses
        ``self.posttFullParameterizedCircuit`` as the base circuit. A state preparation circuit is prepended,
        and a measurement modification circuit is appended.

        Args:
            psGroup: A list of Pauli strings. The Pauli strings should commute qubit-wise.
            shotCount: The number of shots to use.
        
        Returns:
            For each patch, the probability vector resulting from executions of ``self.posttFullParameterizedCircuit``
            that has been modified to allow for measurements of the Pauli strings in ``psGroup``. If hardware execution is turned
            on and ``len(self.jobResults)`` is equal to 0, no circuits will be executed, but they will be appended to
            ``self.quantumCircuitsBatch``. Nothing is returned in this case. If ``len(self.jobResults)`` is greater than 0,
            the probability vectors will be retrieved from ``self.jobResults``.
        """
        counts = list()
        if (self.runOnHardware == False or self.firstPass == True): # If we run on a simulator or if it is the first pass of a hardware run
            measurementModsQc = self.posttFullParameterizedCircuit.copy_empty_like()

            measure_x = [False for i in range(self.backendQubitCount)] # Keep track of whether an H-gate has been applied to a given qubit
            measure_y = measure_x.copy() # Same but for measuring Y

            # Loop over all patches to apply the appropriate measurement modification gates.
            for patch in self.patches:
                for ps in psGroup:
                    for i in range(len(ps)):
                        posttQubitIndex = patch[self.totQubitCount - i - 1] # patch is a mapping from virtual to physical qubit index
                        if ps[i] == 'X' and measure_x[posttQubitIndex] == False:
                            measurementModsQc.h(posttQubitIndex)
                            measure_x[posttQubitIndex] = True
                        elif ps[i] == 'Y' and measure_y[posttQubitIndex] == False:
                            measurementModsQc.sdg(posttQubitIndex)
                            measurementModsQc.h(posttQubitIndex)
                            measure_y[posttQubitIndex] = True

                for posttQubitIndex in patch:
                    measurementModsQc.measure(posttQubitIndex, posttQubitIndex)

            measurementModsQc = transpile(measurementModsQc, backend = self.backend, layout_method = "trivial", optimization_level = 0, routing_method = "none", seed_transpiler = 1)
            compositeCircuit = self.inputStatePrepQc.compose(self.posttFullParameterizedCircuit)
            compositeCircuit.compose(measurementModsQc, inplace = True)

            if (self.runOnHardware == False):
                counts = self.backend.run(compositeCircuit, shots = shotCount, seed_simulator = random.randint(0, int(1e9)), method = "statevector").result().get_counts() 
            else: # If we run on hardware and it is the first pass
                self.quantumCircuitsBatch.append((compositeCircuit.copy(), None, shotCount))
                return # <---- OBS !!!
            
        else: # If we run on hardware and it is the second pass
            for key, val in self.jobResults[self.jobResultsIndex].data.items():
                counts = val.get_counts()
            self.jobResultsIndex += 1

        # Make a separate probability vector for each patch.
        p_All = list()
        for patch in self.patches:
            p = [0 for i in range(2**len(patch))]
            for measurementResult in counts.keys():
                reducedBitString = ""
                for i in range(0, len(patch)):
                    reducedBitString += measurementResult[self.backendQubitCount - patch[len(patch)-i-1] - 1] # We pick out the bits from the relevant patch following the tiled M0 bitstring order.
                p[int(reducedBitString, 2)] += counts[measurementResult] / shotCount
            p_All.append(p.copy())

        return p_All
    
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
        self.patches = None
        if self.loadJob == True or savedPatches != None:
            self.patches = savedPatches
            return
        else:
            if self.doPatchParallelization:
                self.patches = self.FindPatches()
                print("PATCHES:", self.patches)
            else:   # A single patch
                tmpCircuit = self.pretTileCircuits[0].copy()
                for i in range(1, len(self.pretTileCircuits)):
                    tmpCircuit.compose(self.pretTileCircuits[i], inplace = True)
                tmpCircuit = transpile(tmpCircuit, self.backend, optimization_level = 3, seed_transpiler = 1)              
                self.patches = [tmpCircuit.layout.initial_index_layout()[0:self.totQubitCount]]
                return

        f = open("patches.log", 'a')
        for patch in self.patches:
            for qubit in patch:
                f.write(str(qubit) + ", ")
            f.write("\n")
        f.close()
    
    def TranspileTileCircuits(self):
        """
        Transpiles the tile circuits in ``self.pretTileCircuits`` and inserts swap gates to undo routing swaps.
        Also determines the physical qubits associated with all the transpiled tiles. ``self.patches`` must
        be initialized before calling.
        """
        # Transpile all the tile circuits
        self.posttTileCircuits_All = list()
        for patch in self.patches:
            passManager = generate_preset_pass_manager(
                optimization_level = 3,
                backend = self.backend,
                initial_layout = patch, # Specifying the initial layout here is necessary to map the tiles to the correct patches.
                seed_transpiler = 1
            )

            transpiledTileCircuits = [0 for i in range(len(self.pretTileCircuits))]
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
    
    def ParameterizeAndComposeFullCircuit(
        self, 
        parameters : list[float]
    ):
        """
        Args:
            parameters: The circuit parameters for all tiles in a single patch.

        Parameterizes and composes the full post-transpilation circuit with the ansatz on all patches.
        ``self.posttTileCircuits_All`` must be initialized beforehand. The result is stored in
        ``self.posttFullParameterizedCircuit``
        """
        self.parameters = parameters
        self.posttFullParameterizedCircuit = self.posttTileCircuits_All[0][0].copy_empty_like()

        for patchIndex in range(len(self.patches)):
            idx = 0
            for layerNumber in range(self.layerCount):
                for i in range(len(self.pretTileCircuits)):
                    tileCircuitParameterized = self.posttTileCircuits_All[patchIndex][i].copy()
                    paramCount = len(tileCircuitParameterized.parameters)
                    for j in range(paramCount):
                        # ``tileCircuitParameterized.parameters[0]`` with 0 as index because ``assign_parameters`` with ``inplace = True`` mutates the circuit
                        tileCircuitParameterized.assign_parameters({tileCircuitParameterized.parameters[0] : self.parameters[idx]}, inplace = True)
                        idx += 1

                    self.posttFullParameterizedCircuit.compose(tileCircuitParameterized, inplace = True)
    
    def ComposeFullTrivialCircuit(self):
        """
        Builds a quantum circuit with the ansatz on all patches with all parameters set to 0.
        Only used for normal M0 runs.
        """
        self.posttFullTrivialCircuit = self.posttTileCircuits_All[0][0].copy_empty_like()

        for patchIndex in range(len(self.patches)):
            for layerNumber in range(self.layerCount):
                for i in range(len(self.pretTileCircuits)):
                    tileCircuitTrivial = self.posttTileCircuits_All[patchIndex][i].copy()
                    paramCount = len(tileCircuitTrivial.parameters)
                    for j in range(paramCount):
                        tileCircuitTrivial.assign_parameters({tileCircuitTrivial.parameters[0] : 0.0}, inplace = True)

                    self.posttFullTrivialCircuit.compose(tileCircuitTrivial, inplace = True)

    def FindNonOverlappingTiles(
    self
    ):
        """
        Identify tiles that have no overlapping qubits and whose mitigators can thus be determined in parallel.
        Results are saved in ``self.nonOverlappingTiles_Indices`` as a list of lists with indices of tiles corresponding
        to elements in ``self.pretTileCircuits``.
        """
        isProcessed = [False for i in range(len(self.pretTileCircuits))]

        for i in range(len(self.pretTileCircuits)):
            if isProcessed[i] == True:
                continue
            # -- else --
            markedTiles = list()
            markedQubits = list()
            for j in range(i, len(self.pretTileCircuits)):
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
    
    def GetCircuitComplexityInformation(self):
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
    
    def GetInputStatePreparationCircuit(self):
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
        for i in range(len(self.patches)):
            conditionNumbers = list()
            for j in range(len(self.tileMitigators_All[i])):
                U, S, Vdag = np.linalg.svd(self.tileMitigators_All[i][j], full_matrices = False)
                conditionNumbers.append(S[0] / S[-1])
            
            self.conditionNumbers_All.append(conditionNumbers.copy())
        
        for i in range(len(self.patches)):
            self.chosenTileMitigators_All.append([False for j in range(len(self.tileMitigators_All[i]))])
    
        cnThreshold = 10
        for i in range(len(self.patches)):
            pairs = list()
            for j in range(len(self.tileMitigators_All[i])):
                pairs.append((self.conditionNumbers_All[i][j], j))
            pairs = sorted(pairs)
            cn = 1
            idx = 0
            while cn < cnThreshold and idx < len(self.tileMitigators_All[i]):
                self.chosenTileMitigators_All[i][pairs[idx][1]] = True
                cn *= (pairs[idx][0])**self.layerCount
                idx += 1
            
            self.conditionNumbersProduct_All.append(cn)

    def EstimateBackendUsage(
    self
    ):
        """
        Estimates the backend usage.
        ``self.quantumCircuitsBatch`` must be filled with the quantum circuits that will be executed.
        Returns:
            The usage in minutes.
        """
        usage = 0.0
        for i in range(len(self.quantumCircuitsBatch)):
            depth = self.quantumCircuitsBatch[i][0].depth()
            usage += self.quantumCircuitsBatch[i][2] * depth / self.backendClops
        return usage / 60.0
    
    def NoiseCharacterization_Run(
        self,
        repetitionCount : int,
    ):
        """
        Analogous to ``self.Run`` but for noise characterization instead.
        Builds the tile A-matrices for all tiles for all patches a number
        of times specified by ``repetitionCount``. This can be used to
        see how the quality of tile assignment matrices change over time.
        Results are output in a file, "cns_over_time.log".

        Args:
            repetitionCount: The number of times to build the tile mitigators for every tile in all patches.
        """
        self.quantumCircuitsBatch = list()
        self.jobResults = list()
        self.jobResultsIndex = 0

        self.NoiseCharacterization_RunPass(repetitionCount) # Pass 1
        
        if self.loadJob == False:
            jobNamesToSave = list()
            with Batch(backend = self.backend) as batch:
                sampler = SamplerV2()
                circuitsPerJob = 1
                jobCount = int((len(self.quantumCircuitsBatch) + circuitsPerJob - 1) / circuitsPerJob)
                jobs = [None for i in range(jobCount)]
                for i in range(jobCount):
                    jobs[i] = sampler.run(self.quantumCircuitsBatch[i * circuitsPerJob : (i + 1) * circuitsPerJob])     # (i + 1) * circuitsPerJob kan godt være større end længden af listen ved slicing.
                    print(f">>> Job ID: {jobs[i].job_id()}")
                    jobNamesToSave.append(jobs[i].job_id())

            f = open("jobNames.log", 'a')
            for jobName in jobNamesToSave:
                f.write(str(jobName) + '\n')
            f.close()
                    
        else:
            jobs = [None for i in range(len(self.jobNames))]
            for i in range(len(self.jobNames)):
                jobs[i] = self.service.job(self.jobNames[i])
                print(f">>> Job retrieved: {self.jobNames[i]}")

        for i in range(0, len(jobs)):
            for j in range(0, len(jobs[i].result())):
                self.jobResults.append(jobs[i].result()[j])
        
        self.NoiseCharacterization_RunPass(repetitionCount) # Pass 2

        f = open("cns_over_time.log", 'a')
        for i in range(len(self.patches)):
            f.write("PATCH " + str(i + 1) + "\n")
            for qubit in self.patches[i]:
                f.write(str(qubit) + ", ")
            f.write("\n\n")
            for j in range(repetitionCount):
                f.write(str(self.conditionNumbersOverTime[j][i]) + "\n")
            f.write("\n\n")

        f.close()

    def NoiseCharacterization_RunPass(
        self,
        repetitionCount : int
    ):
        """
        Analogous to ``self.RunPass`` but for noise characterization instead.
        Called by ``self.NoiseCharacterization_Run``.

        Args:
            repetitionCount: The number of times to build the tile mitigators.
        """
        self.firstPass = (len(self.jobResults) == 0)
        for rep in range(repetitionCount):
            for i in range(len(self.nonOverLappingTiles_Indices)):
                tileCircuitsSubset = list()
                if self.firstPass == True: # First pass only
                    for patchIndex in range(len(self.patches)):                            
                        for tileIndex in self.nonOverLappingTiles_Indices[i]:
                            tileCircuitsSubset.append(self.posttTileCircuits_All[patchIndex][tileIndex].copy())

                # Determine the tile A-matrices
                tmpTileAssignmentMatrices_All = self.PrepareAssignmentMatrices(self.nonOverLappingTiles_Indices[i], tileCircuitsSubset)

                if self.firstPass == False: # Second pass only
                    self.conditionNumbersOverTime.append([])
                    # Determine the condition numbers
                    # For a given patch: average over condition numbers for all tile mitigators
                    for patchIdx in range(len(self.patches)):
                        conditionNumber = 0
                        for j in range(len(tmpTileAssignmentMatrices_All[patchIdx])):
                            U, S, Vdag = np.linalg.svd(tmpTileAssignmentMatrices_All[patchIdx][j], full_matrices = False)
                            conditionNumber += S[0] / S[-1]
                        conditionNumber /= len(tmpTileAssignmentMatrices_All[patchIdx])
                        self.conditionNumbersOverTime[-1].append(conditionNumber)
    
    def NormalM0_Run(
        self
    )   -> np.float64:
        """
        Calculate the expectation value of ``self.operator`` using normal M0.
        The normal M0 version of ``self.Run``.
            
        Returns:
            The average of the error mitigated expectation values for all patches which satisfy
            that the condition numbers for the normal M0 mitigators are below a certain threshold. This is to
            assure the quality of a given patch. Also outputs a log file with results for all patches
            including raw expectation values.
        """
        self.quantumCircuitsBatch = list()
        self.jobResults = list()
        self.jobResultsIndex = 0

        self.doNormalM0 = True # Used in ``self.CalculateExpectationValue``. False by default.

        if self.runOnHardware == False:
            self.NormalM0_RunPass()
        else:
            if self.loadJob == False:
                self.NormalM0_RunPass()

                jobNamesToSave = list()
                with Batch(backend = self.backend) as batch:
                    sampler = SamplerV2()

                    circuitsPerJob = 1
                    jobCount = int((len(self.quantumCircuitsBatch) + circuitsPerJob - 1) / circuitsPerJob)
                    
                    jobs = [None for i in range(jobCount)]
                    for i in range(jobCount):
                        jobs[i] = sampler.run(self.quantumCircuitsBatch[i * circuitsPerJob : (i + 1) * circuitsPerJob])
                        print(f">>> Job ID: {jobs[i].job_id()}")
                        jobNamesToSave.append(jobs[i].job_id())

                f = open("jobNames.log", 'a')
                for jobName in jobNamesToSave:
                    f.write(str(jobName) + '\n')
                f.close()
                        
            else:
                jobs = [None for i in range(len(self.jobNames))]
                for i in range(len(self.jobNames)):
                    jobs[i] = self.service.job(self.jobNames[i])
                    print(f">>> Job retrieved: {self.jobNames[i]}")

            for i in range(0, len(jobs)):
                for j in range(0, len(jobs[i].result())):
                    self.jobResults.append(jobs[i].result()[j])
        
            self.NormalM0_RunPass()

        for i in range(len(self.patches)):
            U, S, Vdag = np.linalg.svd(self.normal_M0_Mitigators_All[i], full_matrices = False)
            conditionNumber = S[0] / S[-1]
            self.conditionNumbers_All.append(conditionNumber)

        return self.LogTiledM0()
    
    def NormalM0_RunPass(
        self
    )   -> tuple[list[np.float64], list[np.float64]]:
        """
        The normal M0 version of ``self.RunPass``.

        Returns:
            The error mitigated and raw expectation values for every patch if they are calculated and otherwise lists of 0's
        """
        self.firstPass = (len(self.jobResults) == 0)
        self.normal_M0_Mitigators_All = [0 for i in range(len(self.patches))] # Placeholder 0's for the mitigator for every patch

        assignmentMatrices = self.PrepareM0AssignmentMatrices()

        if self.runOnHardware == False or self.firstPass == False: # Simulator or second pass of a hardware run
            for patchIndex in range(len(self.patches)):
                self.normal_M0_Mitigators_All[patchIndex] = np.linalg.inv(assignmentMatrices[patchIndex])

        self.mitigatedExpectationValue_All, self.rawExpectationValue_All = self.CalculateExpectationValue() 

    def PrepareM0AssignmentMatrices(
        self
    )   -> list[NDArray[np.float64]]: 
        """
        Prepares the normal M0 assignment matrices for all patches.
        Very similar to ``self.PrepareAssignmentMatrices``.

        Returns:
            A list with the M0 assignment matrices for all patches (or a list with zero-matrices if it is the first pass of a hardware run)
        """
        if self.posttFullTrivialCircuit == None:
            self.ComposeFullTrivialCircuit()
        
        assignmentMatrices = list()
        if (self.runOnHardware == False or self.firstPass == False): # Simulator or second pass of a hardware run
            assignmentMatrices = [np.zeros((2**self.totQubitCount, 2**self.totQubitCount)) for i in range(len(self.patches))]

        for i in range(0, 2**self.totQubitCount):
            counts = dict()
            if (self.runOnHardware == False or self.firstPass == True):
                basisState = format(i, 'b')
                basisState = '0' * (self.totQubitCount - len(basisState)) + basisState
                
                preQc = self.posttFullTrivialCircuit.copy_empty_like()
                postQc = self.posttFullTrivialCircuit.copy_empty_like()

                for patchIdx in range(len(self.patches)):
                    for j in range(self.totQubitCount):
                        if basisState[j] == '1':
                            preQc.x(self.patches[patchIdx][self.totQubitCount - j - 1])
                        postQc.measure(self.patches[patchIdx][self.totQubitCount - j - 1], self.patches[patchIdx][self.totQubitCount - j - 1])

                compositeQc = preQc.compose(self.posttFullTrivialCircuit)
                compositeQc.compose(postQc, inplace = True)

                if self.runOnHardware == False:
                    counts = self.backend.run(compositeQc, shots = self.mitigatorShots, seed_simulator = random.randint(0, int(1e9)), method = "statevector").result().get_counts()
                else:
                    self.quantumCircuitsBatch.append((compositeQc.copy(), None, self.mitigatorShots))
                    continue # <--- OBS !!!

            else: # Second pass of a hardware run
                for key, val in self.jobResults[self.jobResultsIndex].data.items():
                    counts = val.get_counts()
                self.jobResultsIndex += 1

            for patchIdx in range(len(self.patches)):
                for measurementResult in counts.keys():
                    reducedBitString = ""
                    for j in range(0, self.totQubitCount):
                        reducedBitString += measurementResult[self.backendQubitCount - self.patches[patchIdx][self.totQubitCount - j - 1] - 1] # Pick out the bits corresponding to the patch, following the tiled M0 order
                    assignmentMatrices[patchIdx][int(reducedBitString, 2)][i] += counts[measurementResult] / self.mitigatorShots

        return assignmentMatrices
    
    def FindPatches(
        self
    )   -> list[list[int]]:
        """
        Finds patches in a brute force way by randomly mapping the virtual qubits to physical qubits on the backend
        a large number of times and keeping non-overlapping patches. The quality of a patch is estimated based
        on the sum of the error rates of CZ gates on qubits in the patch, and the highest quality patches are
        kept. The first patch, however, is the patch that is chosen when you transpile.

        Returns:
            A list with all the patches.
        """
        quantumCircuit = self.pretTileCircuits[0].copy()
        for i in range(1, len(self.pretTileCircuits)):
            quantumCircuit.compose(self.pretTileCircuits[i], inplace = True)

        patches = list()
        couplingMap = self.backend.coupling_map

        cz_errors = list()
        for key in self.backend.target["cz"]:
            cz_errors.append([self.backend.target["cz"][key].error, key])
        cz_errors = sorted(cz_errors)

        scoresAndPatches = list()
        # Generate a bunch of random patches
        for i in range(10000):
            # Find a mapping from virtual to physical qubits based on the coupling map.
            sabreLayout = SabreLayout(couplingMap, max_iterations = 10)
            sabreLayout.run(circuit_to_dag(quantumCircuit.copy()))
            patch = list(range(self.totQubitCount))
            for i in range(len(sabreLayout.property_set["layout"])):
                if sabreLayout.property_set["layout"][i]._register._name != "ancilla":
                    patch[sabreLayout.property_set["layout"][i]._index] = i

            score = self.ScorePatch(patch, cz_errors)
            scoresAndPatches.append([score, patch])

        scoresAndPatches = sorted(scoresAndPatches)

        # Choose the patch we get from transpiling as the first patch
        tmpCircuit = self.pretTileCircuits[0].copy()
        for i in range(1, len(self.pretTileCircuits)):
            tmpCircuit.compose(self.pretTileCircuits[i], inplace = True)
        tmpCircuit = transpile(tmpCircuit, self.backend, optimization_level = 3, seed_transpiler = 1)                    
        bestPatch = tmpCircuit.layout.initial_index_layout()[0:self.totQubitCount]
        print(bestPatch)

        selectedPatches = [bestPatch]
        for i in range(0, len(scoresAndPatches)):
            newPatch = scoresAndPatches[i][1]
            addPatch = True
            for oldPatch in selectedPatches:
                for newQubit in newPatch:
                    if newQubit in oldPatch:
                        addPatch = False
                        break
                if addPatch == False:
                    break
            if addPatch:
                selectedPatches.append(newPatch)
                
        return selectedPatches

    def ScorePatch(
        self,
        patch : list[int],
        cz_errors : list[list[any]]
    )   -> np.float64:
        """
        A rudimentary function to score a patch based on the error rates of CZ gates on qubits in the patch.
        A lower score is better.

        Args:
            patch: The qubits in the patch.
            cz_errors: A list of CZ error rates and the associated qubits retrieved from ``self.backend``. The elements of the list contain the error rate as the first element and a list or tuple with the two associated qubits as the second element. 
        
        Returns:
            The sum of the error rates of CZ gates on qubits in the patch. Possibly double counts.
        """
        score = 0
        for qubit in patch:
            for pair in cz_errors:
                if (qubit == pair[1][0] or qubit == pair[1][1]):
                    score += pair[0]
        
        return score

    def LogTiledM0(
        self
    )   -> np.float64:
        """
        Logs results from tiled M0. Also returns the average error mitigated expectation value over all good patches (based on condition numbers).

        Returns:
            The average error mitigated expectation value over all good patches.
        """
        currentTime = datetime.now().strftime("%H:%M:%S")
        self.logData = "\n\n---- ERROR MITIGATION ----" + '\n' + datetime.now().strftime("%d") + ' ' + datetime.now().strftime("%m") + ' ' + currentTime + '\n'
        self.logData += "\nM0 type: "
        self.logData += "tiled\n" if self.doNormalM0 == False else "normal"
        self.logData += "\nTotal expectation value shots: " + str(sum(self.qwcGroups_ShotCounts))
        self.logData += "\nNoise characterization shots per basis state: " + str(self.mitigatorShots)
        self.logData += "\nNumber of layers: " + str(self.layerCount)
        self.logData += "\nInput state: " + self.inputState
        self.logData += "\nBackend: " + str(self.backend.name) + "\n"
        self.logData += "\nVariational parameters:\n"
        for i in range(len(self.parameters)):
            self.logData += str(self.parameters[i]) + ' '
            self.logData += "\n"
        self.logData += "\nOperator:\n"
        for key in self.operator:
            self.logData += key + ", " + str(self.operator[key]) + "\n"
        self.logData += "\nPauli string groups:\n"
        for i in range(len(self.operatorGroupedByQwc)):
            for j in range(len(self.operatorGroupedByQwc[i])):
                self.logData += str(self.operatorGroupedByQwc[i][j]) + ' '
            self.logData += ", shots: " + str(self.qwcGroups_ShotCounts[i]) + "\n"
        self.logData += "\nPatches:\n"
        for patch in self.patches:
            for qubit in patch:
                self.logData += str(qubit) + ", "
            self.logData += "\n"

        conditionNumberThreshold = 4.0       # !!! HARDCODED
        finalMitigatedEvals = list()
        finalRawEvals = list()
        for i in range(len(self.patches)):
            self.logData += "\n\n--- PATCH " + str(i + 1)
            self.logData += "\nCondition numbers:\n"
            for j in range(len(self.tileMitigators_All[i])):
                self.logData += str(self.conditionNumbers_All[i][j]) + "    chosen: " + str(self.chosenTileMitigators_All[i][j]) + "\n"
            self.logData += "Average: " + str(np.average(self.conditionNumbers_All[i]))
            self.logData += "\nProduct: " + str(self.conditionNumbersProduct_All[i])
            self.logData += "\nTiled M0 expectation value: " + str(self.mitigatedExpectationValue_All[i])
            self.logData += "\nRaw expectation value: " + str(self.rawExpectationValue_All[i])
            self.logData += "\nDepth complexity: " + str(self.depth_All[i])
            self.logData += "\nGate complexity: " + str(self.gateCount_All[i])
            self.logData += "\nNumber of non-local gates: " + str(self.nonLocalGates_All[i])

            if (np.average(self.conditionNumbers_All[i]) < conditionNumberThreshold):
                finalMitigatedEvals.append(self.mitigatedExpectationValue_All[i])
                finalRawEvals.append(self.rawExpectationValue_All[i])

        avgMitigated = np.mean(finalMitigatedEvals)
        stdMitigated = np.std(finalMitigatedEvals, ddof = 1 if len(finalMitigatedEvals) > 1 else 0)

        avgRaw = np.mean(finalRawEvals)
        stdRaw = np.std(finalRawEvals, ddof = 1 if len(finalRawEvals) > 1 else 0)

        self.logData += "\n\nCondition number threshold: " + str(conditionNumberThreshold)
        self.logData += "\nNumber of accepted patches: " + str(len(finalMitigatedEvals))
        self.logData += "\n\n------------------------------"
        self.logData += "\n\nMITIGATED"
        self.logData += "{0: <20}".format("\nAverage: ") + str(avgMitigated)
        self.logData += "{0: <20}".format("\nStandard deviation: ") + str(stdMitigated)
        self.logData += "\n\nRAW"
        self.logData += "{0: <20}".format("\nAverage: ") + str(avgRaw)
        self.logData += "{0: <20}".format("\nStandard deviation: ") + str(stdRaw)

        f = open("tiledM0.log", 'a')
        f.write(self.logData)
        f.write("\n###############\n")
        f.close()

        return avgMitigated