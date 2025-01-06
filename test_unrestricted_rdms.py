import numpy as np

import pyscf
from pyscf import scf, mcscf, mp
from pyscf import gto, scf, mcscf, fci

import slowquant.SlowQuant as sq
import slowquant.unitary_coupled_cluster.linear_response.allstatetransfer as allstatetransfer  # pylint: disable=consider-using-from-import
import slowquant.unitary_coupled_cluster.linear_response.naive as naive  # pylint: disable=consider-using-from-import
import slowquant.unitary_coupled_cluster.linear_response.projected as projected  # pylint: disable=consider-using-from-import
import slowquant.unitary_coupled_cluster.linear_response.statetransfer as statetransfer  # pylint: disable=consider-using-from-import
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS




mol = pyscf.M(atom="O 0 0 0; H 0.0  0.0  0.9697", basis="6-31G", unit="angstrom", spin=1)
mol.build()
mf = scf.UHF(mol)
mf.kernel()



mc = mcscf.UCASCI(mf, 3, (2,1))
res = mc.kernel(mf.mo_coeff)


ci_coeff = mc.ci  # CI coefficients
norb = mc.ncas  # Number of active orbitals
nelec = mc.nelecas  # Number of active electrons

# Compute the 1,2-RDMs
rdms = fci.direct_uhf.make_rdm12s(ci_coeff, norb, nelec)



"""
Test for unrestricted rdms
"""
# Slowquant Object with parameters and setup
SQobj = sq.SlowQuant()
SQobj.set_molecule(
    """O  0.0   0.0  0.0;
        H  0.0  0.0  0.9697;""",
    distance_unit="angstrom",
)
SQobj.set_basis_set("6-31G")
# HF
SQobj.init_hartree_fock()
SQobj.hartree_fock.use_diis = False
SQobj.hartree_fock.run_unrestricted_hartree_fock()
h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
g_eri = SQobj.integral.electron_repulsion_tensor
print("hej", SQobj.hartree_fock.E_uhf + SQobj.molecule.nuclear_repulsion, SQobj.hartree_fock.E_uhf - mf.energy_elec()[0])
    
WF = UnrestrictedWaveFunctionUPS(
    SQobj.molecule.number_bf * 2,
    SQobj.molecule.number_electrons,
    ((2,1), 3),
    (SQobj.hartree_fock.mo_coeff_alpha, SQobj.hartree_fock.mo_coeff_beta),
    h_core,
    g_eri,
    "fUCC",
    {"n_layers": 2}
)
WF.run_ups(False)
print("hej2", WF.energy_elec + SQobj.molecule.nuclear_repulsion, WF.energy_elec  + SQobj.molecule.nuclear_repulsion - res[0])
    
    
    #print(WF.rdm1aa, WF.rdm1bb, WF.rdm2aaaa, WF.rdm2bbbb, WF.rdm2aabb)