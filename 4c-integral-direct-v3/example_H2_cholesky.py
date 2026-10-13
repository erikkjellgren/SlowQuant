import time

import basis_set_exchange as bse
import numpy as np
import pyscf
from pyscf import scf
from pyscf.scf import hf

from slowquant.unitary_coupled_cluster.DHF_integral_direct import DHFCholesky
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction_DHF import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response.generalized_direct_DHF import LinearResponseDirect

hf.remove_overlap_zero_eigenvalue = False

mol = pyscf.M(
    atom="H 1.0215 -0.0252 0.5645; H 1.1785 -0.5748 1.0355",
    basis=bse.get_basis("dyall-v2z", elements=["H"], fmt="nwchem"),
    unit="angstrom",
    nucmod="gaussian",
    max_memory=4000,
    verbose=0,
)

t = time.time()
mf = scf.dhf.DHF(mol)
mf.conv_tol = 1e-11
mf.direct_scf_tol = 1e-10
mf.kernel()
print(f"DHF {time.time() - t:.1f} s  E = {mf.e_tot:.10f}")

WF = GeneralizedWaveFunctionUPS(
    ((1, 1), 6),
    np.array(mf.mo_coeff, dtype=complex),
    mol,
    [],
    False,
    "fUCCSD",
    {"n_layers": 1, "is_spin_conserving": False},
    include_active_kappa=True,
)

t = time.time()
WF.set_integral_backend(DHFCholesky(mol, tol=1e-8, cd_jk=False, screen_tol=1e-10))
print(f"Cholesky vectors: {len(WF._direct.V)} ({time.time() - t:.1f} s)")

t = time.time()
WF.run_wf_optimization_direct_DHF(tol=1e-6)
print(f"OO {time.time() - t:.1f} s  E_elec = {WF.energy_elec:.10f}")

t = time.time()
LR = LinearResponseDirect(WF, "SD")
res = LR.get_properties(shieldings=True, sscc=False, tol=1e-5)
print(f"Shieldings {time.time() - t:.1f} s:", res["shieldings"])
