import sys
import os
import pickle
sys.path.insert(0, os.path.abspath("../../SlowQuant"))

import tiled_m0_main
import tiled_m0_helper
from qiskit.circuit import (QuantumCircuit, Parameter)
from qiskit_ibm_runtime import (QiskitRuntimeService, Batch)
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_aer.primitives import SamplerV2
from qiskit_aer import AerSimulator
import numpy as np

def LoadJobNames(filename : str):
    f = open(filename, 'r')
    line = f.readline()
    jobNames = list()
    while line != '\n' and line != '':
        jobNames.append(line.strip())
        line = f.readline()
    return jobNames

def LoadThetas(filename : str):
    thetas = list()
    f = open(filename, 'r')
    line = f.readline()
    while line != '\n' and line != '':
        thetas.append(np.float64(line.strip()))
        line = f.readline()
    f.close()

    return thetas

def LoadOperator(filename):
    f = open(filename, 'r')
    operator = dict()
    line = f.readline()
    while line != '\n' and line != '':
        pauliString = line.strip()
        line = f.readline()
        weight = np.float64(line.strip())
        if (abs(weight) > 1e-6):
            operator[pauliString] = weight
        line = f.readline()
    f.close()
    return operator

f = open("../token.txt")
token = f.readline()
f.close()

# ------- Cloud related --------
service = QiskitRuntimeService(
    channel='ibm_cloud',
    instance= 'crn:v1:bluemix:public:quantum-computing:eu-de:a/4ef65d033cc5402196f7b9d579cd281d:e5881441-edfc-4b80-9ebd-1646bdb1f52c::', # This is the instance for running on EU computers
    token = token
)

# check backends available
# print(service.backends())

# ----------- Backend  -----------
# On hardware:
# backend = service.backend("ibm_aachen")

# Noisy simulator:
backend = AerSimulator.from_backend(service.backend("ibm_aachen"))


# ---------- Molecule -----------
atomCoords =  """N  0.0  0.0 0.0;
                H   0.0  0.0 1.0362;"""

basis = "6-31g"

activeElecCount = 2
activeSpatOrbCount = 4
name = "0.08030958600035343"


layerCount = 2

# This is for the tUPS ansatz
tileCircuits, tileQubits = tiled_m0_helper.GetTileCircuitsAndQubits(activeSpatOrbCount, backend.num_qubits)

# Parameters for the tiles starting with the top left tile in the ansatz and then moving down, going back to the top with each new column.
# parameters = [0.0 for i in range(3)] #LoadThetas("SavedStuff/LiH/thetas_L1")
parameters = np.load(f"../{name}_thetas.npy")


# The operator as a dictionary where the keys are the Pauli strings and the values are the coefficients / weights.
# operator = {"Z" * 4 : 1.0} #LoadOperator("SavedStuff/Butadiene/operator_L1")
with open(f"../{name}_N_operator", "rb") as f:
    operator = pickle.load(f)

referenceValue = 1.0 # The reference / exact expectation value of the operator (to print in the output file together with the results)

expectationValueShots = 10000
mitigatorShots = int(min(15000 * (1.2**len(tileQubits))**2, 100000))
# mitigatorShots = 15000
print(mitigatorShots)

tiledM0 = tiled_m0_main.TiledM0(tileCircuits = tileCircuits,                        # Only tile circuits for the first layer (list of qiskit QuantumCircuits, pre-transpilation, not parameterized)
                                tileQubits = tileQubits,                            # Only tile qubits for the first layer formatted like [[q0,q1,q2,q3], [q4,q5,q6,q7], ...]
                                layerCount = layerCount,
                                elecCount = activeElecCount,                        # Number of active electrons
                                operator = operator,                                # Operator as a dictionary
                                backend = backend,
                                expectationValueShots = expectationValueShots,      # Total number of shots to use for expectation value
                                mitigatorShots = mitigatorShots,                    # The number of shots to use per column in the assignment matrices. The total number of mitigation shots will be 64 * mitigatorShots (in the case of tUPS and when the qubit count is a multiple of 4 but greater than 4)
                                inputState = "10001000",                                # Passed in Fermi-order: alpha0 beta0 alpha1 beta1 ...
                                doPatchParallelization = False
)

# Output in file "tiledM0.log"
tiledM0.Run(parameters)


# Log results
f = open("tiledM0.log", 'a')
f.write("\n\nREFERENCE EXPECTATION VALUE")
f.write("{0: <19}".format("") + str(referenceValue))
f.close()