import pyscf
from pyscf import fci, mcscf, scf

import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS

mol = pyscf.M(atom="H 0 0 0; H 0.0  0.0  0.74", basis="6-31g", unit="angstrom", spin=0)
# mol = pyscf.M(atom="O 0 0 0; H 0.0  0.0  0.9697", basis="sto-3g", unit="angstrom", spin=1)
mol.build()
mf = scf.UHF(mol)
mf.kernel()
print(mol.intor("int1e_r"))

mc = mcscf.UCASCI(mf, 2, (1, 1))
res = mc.kernel(mf.mo_coeff)
print(mf.mo_coeff)


ci_coeff = mc.ci  # CI coefficients
norb = mc.ncas  # Number of active orbitals
nelec = mc.nelecas  # Number of active electrons

# Compute the 1,2-RDMs, https://pyscf.org/_modules/pyscf/fci/direct_spin1.html
rdms = fci.direct_uhf.make_rdm12s(ci_coeff, norb, nelec)
# print(rdms)
# printed are (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) see the link above

"""
Test for unrestricted rdms
"""
# Slowquant Object with parameters and setup
SQobj = sq.SlowQuant()
# SQobj.set_molecule(
#    """O  0.0   0.0  0.0;
#        H  0.0  0.0  0.9697;""",
#    distance_unit="angstrom",
# )
SQobj.set_molecule(
    """H  0.0   0.0  0.0;
        H  0.0  0.0  0.74;""",
    distance_unit="angstrom",
)
SQobj.set_basis_set("6-31g")
# HF
SQobj.init_hartree_fock()
SQobj.hartree_fock.use_diis = False
SQobj.hartree_fock.run_unrestricted_hartree_fock()
h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
g_eri = SQobj.integral.electron_repulsion_tensor
# print("hej", SQobj.hartree_fock.E_uhf + SQobj.molecule.nuclear_repulsion, SQobj.hartree_fock.E_uhf - mf.energy_elec()[0])

WF = UnrestrictedWaveFunctionUPS(
    SQobj.molecule.number_electrons,
    ((1, 1), 2),
    mf.mo_coeff,
    h_core,
    g_eri,
    "fuccsd",
    {"n_layers": 1},
    include_active_kappa=True,
)

# print(WF.manual_gradient())
# print(WF.orbital_gradient_RDM)
# with np.printoptions(precision=3, suppress=True):
#    print(WF.orbital_gradient_RDM - WF.manual_gradient())
WF.run_wf_optimization_1step("slsqp", True)

# print(WF.orbital_response_hessian_unrestricted)
# print(WF.manual_hessian_block_unrestricted())
# with np.printoptions(precision=4, suppress=True):
#     print(WF.orbital_response_hessian_unrestricted - WF.manual_hessian_block_unrestricted())

# print(WF.orbital_response_metric_sigma_unrestricted)
# print(WF.manual_metric_sigma_unrestricted())
# with np.printoptions(precision=4, suppress=True):
#     print(WF.orbital_response_metric_sigma_unrestricted - WF.manual_metric_sigma_unrestricted())


# print("hej2", WF.energy_elec + SQobj.molecule.nuclear_repulsion, WF.energy_elec  + SQobj.molecule.nuclear_repulsion - res[0])
# print("aa", WF.rdm1aa, "bb", WF.rdm1bb,"aaaa", WF.rdm2aaaa, "bbbb", WF.rdm2bbbb, "aabb", WF.rdm2aabb)

# print("RDM" , WF.energy_elec_RDM, "elec", WF.energy_elec, "pyscf", mf.energy_elec()[0])

# print(WF.orbital_gradient_RDM)

# print(WF.rdm2aabb)
# print(WF.rdm2bbaa.transpose(2,3,0,1)) # rdm2aabb[i,j,k,l] = rdm2bbaa[k,l,i,j]


print(WF.orbital_hessian_unrestricted_A)
