from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
import slowquant.SlowQuant as sq
import numpy as np

A = sq.SlowQuant()
A.set_molecule(
    """H  0.0  0.0  0.0;
           H  1.4  0.0  0.0;
           H  2.8  0.0  0.0;
           H  4.2  0.0  0.0;""",
    distance_unit="bohr",
)
A.set_basis_set("sto-3g")
A.init_hartree_fock()
A.hartree_fock.run_restricted_hartree_fock()
num_bf = A.molecule.number_bf
h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
g_eri = A.integral.electron_repulsion_tensor
#Lambda_S, L_S = np.linalg.eigh(A.integral.overlap_matrix)
#S_sqrt = np.dot(np.dot(L_S, np.diag(Lambda_S ** (-1 / 2))), np.transpose(L_S))
WF = WaveFunctionUCC(A.molecule.number_bf*2, A.molecule.number_electrons, [0,1,2,3,4,5,6,7], A.hartree_fock.mo_coeff, h_core, g_eri)
WF.run_UCC('SD', True)
