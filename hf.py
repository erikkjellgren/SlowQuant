import numpy as np

import slowquant.SlowQuant as sq
from slowquant.second_quantization_matrix.second_quant_mat_ucc import WaveFunctionUCC

if True:
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
