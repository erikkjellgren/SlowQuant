import numpy as np
import pyscf
from pyscf import scf, mcscf, fci
from pyscf.data import nist
import sys
import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS

mol = pyscf.M(atom="N   0.0  0.0           0.0; H   0.0  0.0 1.0362", basis="aug-cc-pVTZ-J.nw", unit="angstrom", spin=2)
uhf = pyscf.scf.UHF(mol).run()

mc = mcscf.UCASCI(uhf, (4,2), 6)

h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
g_eri = mol.intor("int2e")

#Slowquant
WF = UnrestrictedWaveFunctionUPS(
    mol.nelectron,
    ((4,2),6),
    mc.mo_coeff,
    h_core,
    g_eri,
    "utups",
    {"n_layers":2},
    include_active_kappa=True,
)

WF.thetas = (2*np.pi*np.random.random(len(WF.thetas)) - np.pi).tolist()   
WF.run_wf_optimization_1step("bfgs", orbital_optimization=True, tol=1e-4, maxiter=5000)

print("")
print("E0:", WF.energy_elec_RDM) # Vigtigt at printe den her så man har noget at grep efter

name = str(np.random.random())
print(name)
np.save(f"{name}_thetas.npy", np.array(WF.thetas))
np.save(f"{name}_a_mo.npy", np.array(WF.c_a_mo))
np.save(f"{name}_b_mo.npy", np.array(WF.c_b_mo))

np.save(f"{name}_a_rdm", np.array(WF.rdm1aa))
np.save(f"{name}_b_rdm", np.array(WF.rdm1bb))