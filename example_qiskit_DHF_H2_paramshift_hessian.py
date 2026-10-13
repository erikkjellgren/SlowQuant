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
    ((1, 1), 4), np.array(mf.mo_coeff, dtype=complex), mol, QI, include_active_kappa=True, exact=True
)
WF.set_integral_backend(DHFCholesky(mol, tol=1e-8, cd_jk=False, screen_tol=1e-10))
rng = np.random.default_rng(42)
n = len(WF.thetas) // 2
WF.set_thetas_initial(rng.uniform(-0.05, 0.05, n), rng.uniform(-0.05, 0.05, n))

p = np.array(WF.thetas)
t = time.time()
H = WF.theta_hessian(p)
print(f"Parameter-shift Hessian ({len(p)} x {len(p)}): {time.time() - t:.2f} s")
e = np.eye(len(p))
H_fd = np.array([(WF.theta_gradient(p + 1e-4 * v) - WF.theta_gradient(p - 1e-4 * v)) / 2e-4 for v in e])
print(f"max |H_shift - H_finite_diff| = {np.abs(H - H_fd).max():.2e}")

t = time.time()
WF.run_wf_optimization_2step_DHF(tol=1e-6, theta_method="trust-exact", theta_gtol=1e-8)
print(f"OO {time.time() - t:.1f} s  E_elec = {WF.energy_elec:.10f}  E_tot = {WF.energy_elec + mol.energy_nuc():.10f}")
print("energy evaluations (circuits):", WF.num_energy_evals)
