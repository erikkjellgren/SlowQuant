import time

import numpy as np
from pyscf import gto, scf
from qiskit_aer.primitives import Sampler
from qiskit_nature.second_q.mappers import JordanWignerMapper

from slowquant.qiskit_interface.generalized_circuit_wavefunction_direct import GeneralizedWaveFunctionCircuitDirect
from slowquant.qiskit_interface.generalized_interface import QuantumInterface
from slowquant.qiskit_interface.linear_response.generalized_direct_circuit import LinearResponseCircuitDirect
from slowquant.unitary_coupled_cluster.integral_direct import GHFCholesky

mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="6-31g", verbose=0)
mf = scf.GHF(mol)
mf.conv_tol = 1e-10
mf.kernel()

QI = QuantumInterface(
    Sampler(),
    "fUCCSD",
    JordanWignerMapper(),
    ansatz_options={"n_layers": 1, "is_spin_conserving": False},
    shots=None,
)
WF = GeneralizedWaveFunctionCircuitDirect(
    ((1, 1), 4),
    np.array(mf.mo_coeff, dtype=complex),
    mol,
    QI,
    include_active_kappa=True,
    direct=GHFCholesky(mol, tol=1e-8, cd_jk=False, screen_tol=1e-10),
)
rng = np.random.default_rng(42)
n = len(WF.thetas) // 2
WF.set_thetas_initial(rng.uniform(-0.05, 0.05, n), rng.uniform(-0.05, 0.05, n))

t = time.time()
WF.run_wf_optimization_2step_direct(tol=1e-6, theta_method="l-bfgs-b", e_scf=mf.e_tot)
print(f"OO {time.time() - t:.1f} s  E_tot = {WF.energy_elec + mol.energy_nuc():.10f}  circuits: {WF.num_energy_evals}")

t = time.time()
LR = LinearResponseCircuitDirect(WF, "SD", symmetrize_qG=True)
LR.calc_excitation_energies(n_roots=4, tol=1e-6)
LR.get_oscillator_strength()
print(LR.get_formatted_oscillator_strength())
print(f"LR {time.time() - t:.1f} s")
