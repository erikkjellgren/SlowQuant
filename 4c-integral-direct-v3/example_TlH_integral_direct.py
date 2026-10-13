import time

import basis_set_exchange as bse
import numpy as np
import pyscf
from pyscf import lib, scf
from pyscf.scf import hf

from slowquant.unitary_coupled_cluster.DHF_integral_direct import DHFDirect
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction_DHF import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response.generalized_direct_DHF import LinearResponseDirect

hf.remove_overlap_zero_eigenvalue = False

basis = bse.get_basis("dyall-cv4z", elements=["Tl", "H"], fmt="nwchem")
mol = pyscf.M(
    atom="Tl 0 0 0; H 0 0 1.870",
    basis=basis,
    unit="angstrom",
    nucmod="gaussian",
    max_memory=50000,
    verbose=4,
)
print("n2c", mol.nao_2c(), "threads", lib.num_threads())

t = time.time()
mf = scf.dhf.DHF(mol)
mf.conv_tol = 1e-10
mf.direct_scf_tol = 1e-10
mf.kernel()
np.save("TlH_dhf_mo.npy", mf.mo_coeff)
print(f"DHF {time.time() - t:.0f} s")

t = time.time()
WF = GeneralizedWaveFunctionUPS(
    ((2, 2), 8),
    np.array(mf.mo_coeff, dtype=complex),
    mol,
    [],
    False,
    "fUCCSD",
    {"n_layers": 1, "is_spin_conserving": False},
    include_active_kappa=True,
)
WF.set_integral_backend(DHFDirect(mol, screen_tol=1e-10))
WF.run_wf_optimization_direct_DHF(tol=1e-5)
np.savez("TlH_wf.npz", c_mo=WF.c_mo, thetas_real=WF.thetas_real, thetas_imag=WF.thetas_imag)
print(f"OO {time.time() - t:.0f} s")

t = time.time()
LR = LinearResponseDirect(WF, "SD")
LR.get_properties(shieldings=True, sscc=True, tol=1e-5)
print(f"Properties {time.time() - t:.0f} s")
