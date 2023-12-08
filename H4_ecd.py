import pyscf
from pyscf import mcscf, mp
from slowquant.unitary_coupled_cluster.linear_response.statetransfer import (
    LinearResponseUCC as stLR,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC

mol = pyscf.M(
        atom="""H            0.000000000000    -0.750000000000    -0.324759526419;
                H           -0.375000000000    -0.760000000000     0.324759526419;
                H            0.000000000000     0.750000000000    -0.324759526419;
                H            0.375000000000     0.850000000000     0.45;""",
        basis="STO-3G",
        unit="angstrom")
myhf = mol.RHF().run()
mymp = mp.MP2(myhf).run()
noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)


WF = WaveFunctionUCC(
    mol.nao * 2,
    mol.nelectron,
    (4, 4),
    natorbs,
    mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
    mol.intor("int2e"),
)
print(WF.check_orthonormality(mol.intor("int1e_ovlp")))

WF.run_ucc("SDTQ", False, convergence_threshold=10**-5)
WF.run_ucc("SDTQ", True)

x, y, z = mol.intor("int1e_r", comp=3)
dipole_integrals = (x, y, z)
x, y, z = mol.intor("int1e_cg_irxp", comp=3)
magpole_integrals = (x, y, z)


LR = stLR(WF, excitations="SD")
LR.calc_excitation_energies()
print(LR.get_ecd_output(dipole_integrals, magpole_integrals))


