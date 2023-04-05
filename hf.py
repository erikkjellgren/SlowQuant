import numpy as np

import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC

if False:
    """Test restricted Hartree-Fock through the second quantization module."""
    A = sq.SlowQuant()
    A.set_molecule(
        """H  1.4  0.0  0.0;
           He 0.0  0.0  0.0;""",
        distance_unit="bohr",
        molecular_charge=1,
    )
    A.set_basis_set("sto-3g")
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    Lambda_S, L_S = np.linalg.eigh(A.integral.overlap_matrix)
    S_sqrt = np.dot(np.dot(L_S, np.diag(Lambda_S ** (-1 / 2))), np.transpose(L_S))
    WF = WaveFunctionUCC(A.molecule.number_bf*2, A.molecule.number_electrons, [], S_sqrt, h_core, g_eri)
    WF.run_HF()
    assert (abs(WF.hf_energy - (-4.262632309847)) < 10**-8)

if True:
    """Test OO-UCCD(2,2) through the second quantization module."""
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
    WF = WaveFunctionUCC(A.molecule.number_bf*2, A.molecule.number_electrons, [2,3,4,5], A.hartree_fock.mo_coeff, h_core, g_eri, include_active_kappa=True)
    WF.run_UCC('D', True)
    assert (abs(WF.ucc_energy- (-5.211066791547))< 10**-8)

