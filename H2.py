import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.linear_response.allprojected import (
    LinearResponseUCC as allprojLR,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC

SQobj = sq.SlowQuant()
SQobj.set_molecule(
    """H  0.0          0.0  0.0;
       H   1.671707274  0.0  0.0;""",
    distance_unit="angstrom",
)
SQobj.set_basis_set("6-31G")
SQobj.init_hartree_fock()
SQobj.hartree_fock.run_restricted_hartree_fock()
h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
g_eri = SQobj.integral.electron_repulsion_tensor
WF = WaveFunctionUCC(
    SQobj.molecule.number_bf * 2,
    SQobj.molecule.number_electrons,
    (2, 2),
    SQobj.hartree_fock.mo_coeff,
    h_core,
    g_eri,
)

WF.run_ucc("SD", True)
print(f"E_tot: {WF.energy_elec+SQobj.molecule.nuclear_repulsion} Hartree")

dipole_integrals = (
    SQobj.integral.get_multipole_matrix([1, 0, 0]),
    SQobj.integral.get_multipole_matrix([0, 1, 0]),
    SQobj.integral.get_multipole_matrix([0, 0, 1]),
)

LR = allprojLR(WF, excitations="SD")
LR.calc_excitation_energies()
print("self-consistent operators")
print(LR.get_nice_output(dipole_integrals))
print("")
