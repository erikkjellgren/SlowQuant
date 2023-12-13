import pyscf
from pyscf import mcscf, mp

from slowquant.unitary_coupled_cluster.linear_response.generic import (
    LinearResponseUCC as genericLR,
)
from slowquant.unitary_coupled_cluster.linear_response.naive import (
    LinearResponseUCC as naiveLR,
)
from slowquant.unitary_coupled_cluster.linear_response.selfconsistent import (
    LinearResponseUCC as scLR,
)
from slowquant.unitary_coupled_cluster.linear_response.statetransfer import (
    LinearResponseUCC as stLR,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC

mol = pyscf.M(
    atom="""O  1.39839733250  0.0000000000  0.0000000000;
            O -1.39839733250  0.0000000000  0.0000000000;
            H  1.05666657906 -1.46580784571  -0.8462845543;
            H -1.05666657906  1.6925691086  0.0000000000;""",
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
with mol.with_common_origin((0, 0, 0)):
    x, y, z = mol.intor("int1e_cg_irxp", comp=3)
magetic_moment_integrals = (x, y, z)

print("NAIVE NAIVE NAIVE")
LR = naiveLR(WF, excitations="SD")
LR.calc_excitation_energies()
print(LR.get_ecd_output(dipole_integrals, magetic_moment_integrals))

print("")
print("")
print("ST ST ST")
LR = stLR(WF, excitations="SD")
LR.calc_excitation_energies()
print(LR.get_ecd_output(dipole_integrals, magetic_moment_integrals))
