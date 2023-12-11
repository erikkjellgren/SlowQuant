import pyscf
from pyscf import mcscf, mp
from slowquant.unitary_coupled_cluster.linear_response.naive import (
    LinearResponseUCC as naiveLR,
)
from slowquant.unitary_coupled_cluster.linear_response.selfconsistent import (
    LinearResponseUCC as scLR,
)
from slowquant.unitary_coupled_cluster.linear_response.statetransfer import (
    LinearResponseUCC as stLR,
)
from slowquant.unitary_coupled_cluster.linear_response.generic import (
    LinearResponseUCC as genericLR,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC

mol = pyscf.M(
    atom="""O     0.000000     0.000000     0.000000;
                O     0.000000     0.000000     1.480000;
                H     0.895669     0.000000    -0.316667;
                H    -0.895669     0.000000     1.796667""",
    basis="sto-3g",
    unit="angstrom",
)
myhf = mol.RHF().run()
mymp = mp.MP2(myhf).run()
noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)

WF = WaveFunctionUCC(
    mol.nao * 2,
    mol.nelectron,
    (2, 2),
    natorbs,
    mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
    mol.intor("int2e"),
)

WF.run_ucc("SD", False)
WF.run_ucc("SD", True)

x, y, z = mol.intor("int1e_r", comp=3)
dipole_integrals = (x, y, z)
with mol.with_common_origin((10, 10, 10)):
    x, y, z = mol.intor("int1e_cg_irxp", comp=3)
magetic_moment_integrals = (x, y, z)

LR = stLR(WF, excitations="SD")
LR.calc_excitation_energies()
print(LR.get_ecd_output(dipole_integrals, magetic_moment_integrals))
