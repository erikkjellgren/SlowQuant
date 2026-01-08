import numpy as np
import pyscf
from pyscf import mcscf, scf

from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS

# PySCF
mol = pyscf.M(atom="O 0 0 0; H 0.0  0.0  0.9697", basis="sto-3g", unit="angstrom", spin=1)
mol.build()
mf = scf.UHF(mol)
mf.kernel()

active_space = ((3, 2), 4)

mc = mcscf.UCASCI(mf, active_space[1], active_space[0])
res = mc.kernel(mf.mo_coeff)

h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
g_eri = mol.intor("int2e")

# SlowQuant

WF = UnrestrictedWaveFunctionUPS(
    mol.nelectron,
    active_space,
    mf.mo_coeff,
    h_core,
    g_eri,
    "fuccsdtq",
    {"n_layers": 2},
    include_active_kappa=True,
)
# WF.run_wf_optimization_1step("slsqp", False)
WF.run_wf_optimization_1step("bfgs", True)
print("Energy elec", WF.energy_elec_RDM)


# print(WF.orbital_hessian_unrestricted_A)
# print(WF.manual_hessian_unrestricted_A())
# with np.printoptions(precision=4, suppress=True):
#     print(WF.orbital_hessian_unrestricted_A - WF.manual_hessian_unrestricted_A())

# with np.printoptions(precision=4, suppress=True):
#     print(WF.orbital_hessian_unrestricted_B - WF.manual_hessian_unrestricted_B())

print(f"rdm A: {WF.orbital_hessian_unrestricted_B}")
print(f"manual A: {WF.manual_hessian_unrestricted_B()}")
with np.printoptions(precision=4, suppress=True):
    print(WF.orbital_hessian_unrestricted_A - WF.manual_hessian_unrestricted_A())
