import sys
import os
import pickle
sys.path.insert(0, os.path.abspath("../../SlowQuant"))

import tiled_mo_oscar.tiled_m0_main as tiled_m0_main
import tiled_m0_helper
from qiskit.circuit import (QuantumCircuit, Parameter)
from qiskit_ibm_runtime import (QiskitRuntimeService, Batch)
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_aer.primitives import SamplerV2
from qiskit_aer import AerSimulator
import numpy as np


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
# backend = AerSimulator.from_backend(service.backend("ibm_aachen"))
backend = AerSimulator()

# ---------- Molecule -----------
atomCoords =  """N  0.0  0.0 0.0;
                H   0.0  0.0 1.0362;"""

basis = "../py_scripts/631gJ.nw"

name = "0.04020293945033737"

activeElecCount = 2
activeSpatOrbCount = 4

layerCount = 2

tileCircuits, tileQubits = tiled_m0_helper.GetTileCircuitsAndQubits(activeSpatOrbCount, backend.num_qubits)

parameters = np.load(f"../py_scripts/{name}_thetas.npy")
with open(f"../py_scripts/{name}_operator_active", "rb") as f:
    operator = pickle.load(f)

with open(f"../py_scripts/N_spinrdmelementoperator", "rb") as f:
    mapped_operator_N = pickle.load(f)

with open(f"../py_scripts/H_spinrdmelementoperator", "rb") as f:
    mapped_operator_H = pickle.load(f)

# print("operator", operator)

referenceValue = np.load(f"../py_scripts/{name}_ref_val.npy")

expectationValueShots = 10000000
mitigatorShots = 100000
# mitigatorShots = int(min(15000 * (1.2**len(tileQubits))**2, 100000))




tiledM0 = tiled_m0_main.TiledM0(tileCircuits = tileCircuits,                        # Only tile circuits for the first layer (list of qiskit QuantumCircuits, pre-transpilation, not parameterized)
                                tileQubits = tileQubits,                            # Only tile qubits for the first layer formatted like [[q0,q1,q2,q3], [q4,q5,q6,q7], ...]
                                layerCount = layerCount,
                                elecCount = activeElecCount,                        # Number of active electrons
                                operator = operator,                                # Operator as a dictionary
                                backend = backend,
                                expectationValueShots = expectationValueShots,      # Total number of shots to use for expectation value
                                mitigatorShots = mitigatorShots,                    # The number of shots to use per column in the assignment matrices. The total number of mitigation shots will be 64 * mitigatorShots (in the case of tUPS and when the qubit count is a multiple of 4 but greater than 4)
                                inputState = "10001000",                               # Passed in Fermi-order: alpha0 beta0 alpha1 beta1 ...
                                doPatchParallelization = False
)

# Output in file "tiledM0.log"
tiledM0.Run(parameters)

print("raw pauli strings")
print(tiledM0.rawPsExpectationValues_All)
print(type(tiledM0.rawPsExpectationValues_All))
tiledM0.rawPsExpectationValues_All[0]['IIIIIIII'] = 1.0


print("\n")
print("mitigated pauli strings")
print(tiledM0.mitigatedPsExpectationValues_All)
tiledM0.mitigatedPsExpectationValues_All[0]['IIIIIIII'] = 1.0


expectation_value_N_raw = 0
for pauli, coeff in mapped_operator_N.items():
    expectation_value_N_raw += coeff * tiledM0.rawPsExpectationValues_All[0][pauli]
print(expectation_value_N_raw)

expectation_value_N_mit = 0
for pauli, coeff in mapped_operator_N.items():
    expectation_value_N_mit += coeff * tiledM0.mitigatedPsExpectationValues_All[0][pauli]
print(expectation_value_N_mit)

expectation_value_H_raw = 0
for pauli, coeff in mapped_operator_H.items():
    expectation_value_H_raw += coeff * tiledM0.rawPsExpectationValues_All[0][pauli]
print(expectation_value_H_raw)

expectation_value_H_mit = 0
for pauli, coeff in mapped_operator_H.items():
    expectation_value_H_mit += coeff * tiledM0.mitigatedPsExpectationValues_All[0][pauli]
print(expectation_value_H_mit)

# Log results
f = open("tiledM0_L" + str(layerCount) + ".log", 'a')
f.write("\n\nREFERENCE EXPECTATION VALUE")
f.write("{0: <19}".format("") + str(referenceValue))
f.close()