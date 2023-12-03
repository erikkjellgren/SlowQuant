import pyscf

from slowquant.unitary_coupled_cluster.linear_response.statetransfer_sf import (
    LinearResponseSFUCC,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC

mol = pyscf.M(
    atom="""C  0.00000000000  0.00000000000  0.66101626407;
C  0.00000000000  0.00000000000 -0.66101626407;
H  0.90994570873  0.00000000000  1.25394440185;
H -0.90994570873  0.00000000000  1.25394440185;
H  0.00000000000  0.90994570873 -1.25394440185;
H  0.00000000000 -0.90994570873 -1.25394440185;""",
    basis="ccpvdz",
    unit="angstrom",
)
myhf = mol.RHF().run()

WF = WaveFunctionUCC(
    mol.nao * 2,
    mol.nelectron,
    (6, 6),
    myhf.mo_coeff,
    mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
    mol.intor("int2e"),
)
WF.run_ucc("SD", False)
E_singlet = WF.energy_elec

WF = WaveFunctionUCC(
    mol.nao * 2,
    mol.nelectron,
    (6, 6),
    myhf.mo_coeff,
    mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
    mol.intor("int2e"),
    spin=1,
)
WF.run_ucc("SD", False)
LR = LinearResponseSFUCC(WF, "SD")
LR.calc_excitation_energies()
print(LR.excitation_energies)
print("")
print("")
print("E_singlet:", E_singlet)
print("E_triplet:", WF.energy_elec)
print("E_SF:", WF.energy_elec + LR.excitation_energies[0])
