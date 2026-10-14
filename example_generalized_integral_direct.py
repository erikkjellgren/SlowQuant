import time

import numpy as np
from pyscf import gto, scf

from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.integral_direct import GHFCholesky
from slowquant.unitary_coupled_cluster.linear_response.generalized_direct import LinearResponseDirect

mol = gto.M(atom="H 0 0 0; I 0 0 1.60916", basis="x2c-tzvppall-2c", verbose=0)
mf = scf.GHF(mol).x2c()
mf.conv_tol = 1e-10
mf.kernel()

WF = GeneralizedWaveFunctionUPS(
    ((3, 3), 12),
    np.array(mf.mo_coeff, dtype=complex),
    mol,
    "fUCCSD",
    do_x2c=False,
    ansatz_options={"n_layers": 1, "is_spin_conserving": False},
    include_active_kappa=True,
)
WF.set_integral_backend(GHFCholesky(mol, tol=1e-8, cd_jk=False, screen_tol=1e-10))
t = time.time()
WF.run_wf_optimization_direct(tol=1e-6, e_scf=mf.e_tot)
print(f"OO {time.time() - t:.1f} s  E_tot = {WF.energy_elec + mol.energy_nuc():.10f}")

t = time.time()
LR = LinearResponseDirect(WF, "SD", symmetrize_qG=True)
LR.calc_excitation_energies(n_roots=6, tol=1e-6)
#LR.get_oscillator_strength()
#print(LR.get_formatted_oscillator_strength())
#print("Static polarizability:\n", LR.get_polarizability(omega=0.0, tol=1e-6))
print(f"LR {time.time() - t:.1f} s")
