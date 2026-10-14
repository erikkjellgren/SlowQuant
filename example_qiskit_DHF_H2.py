import time

import basis_set_exchange as bse
import numpy as np
import pyscf
from pyscf import scf
from pyscf.scf import hf
from qiskit_aer.primitives import Sampler
from qiskit_nature.second_q.mappers import JordanWignerMapper

from slowquant.qiskit_interface.generalized_circuit_wavefunction_DHF import GeneralizedWaveFunctionCircuitDHF
from slowquant.qiskit_interface.generalized_interface import QuantumInterface
from slowquant.qiskit_interface.linear_response.generalized_direct_DHF_circuit import (
    LinearResponseCircuitDirect,
)
from slowquant.unitary_coupled_cluster.DHF_integral_direct import DHFCholesky

hf.remove_overlap_zero_eigenvalue = False

mol = pyscf.M(
    atom="H 1.0215 -0.0252 0.5645; H 1.1785 -0.5748 1.0355",
    basis=bse.get_basis("dyall-v2z", elements=["H"], fmt="nwchem"),
    unit="angstrom",
    nucmod="gaussian",
    verbose=0,
)
mf = scf.dhf.DHF(mol)
mf.conv_tol = 1e-11
mf.direct_scf_tol = 1e-10
mf.kernel()

QI = QuantumInterface(
    Sampler(),
    "fUCCSD",
    JordanWignerMapper(),
    ansatz_options={"n_layers": 1, "is_spin_conserving": False},
    shots=None,
)
WF = GeneralizedWaveFunctionCircuitDHF(
    ((1, 1), 4), np.array(mf.mo_coeff, dtype=complex), mol, QI, include_active_kappa=True
)
WF.set_integral_backend(DHFCholesky(mol, tol=1e-8, cd_jk=False, screen_tol=1e-10))
rng = np.random.default_rng(42)
n = len(WF.thetas) // 2
WF.set_thetas_initial(rng.uniform(-0.05, 0.05, n), rng.uniform(-0.05, 0.05, n))

t = time.time()
WF.run_wf_optimization_2step_DHF(tol=1e-6, theta_method="l-bfgs-b")
print(
    f"OO {time.time() - t:.1f} s  E_elec = {WF.energy_elec:.10f}  energy evaluations: {WF.num_energy_evals}"
)

t = time.time()
LR = LinearResponseCircuitDirect(WF, "SD")
res = LR.get_properties(shieldings=True, sscc=True, tol=1e-5)
print(f"Properties {time.time() - t:.1f} s")
print(
    "device calls:",
    QI.total_device_calls,
    "circuits:",
    QI.total_paulis_evaluated,
    "shots:",
    QI.total_shots_used,
)
