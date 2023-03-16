import numpy as np

import slowquant.SlowQuant as sq
from slowquant.second_quantization.second_quant_base import H, WaveFunction
from slowquant.second_quantization.second_quant_functions import (
    expectation_value,
    optimize_kappa,
    optimize_wavefunction_parameters,
)


def test_HeH_sto3g() -> None:
    """Test restricted Hartree-Fock through the second quantization module."""
    A = sq.SlowQuant()
    A.set_molecule(
        """H 1.401   0  -0.000000000000;
           He 0   0  -0.000000000000;""",
        distance_unit="bohr",
        molecular_charge=1,
    )
    A.set_basis_set("sto-3g")
    num_bf = A.molecule.number_bf
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    Lambda_S, L_S = np.linalg.eigh(A.integral.overlap_matrix)
    S_sqrt = np.dot(np.dot(L_S, np.diag(Lambda_S ** (-1 / 2))), np.transpose(L_S))
    hf_det = [1] * A.molecule.number_electrons + [0] * (num_bf * 2 - A.molecule.number_electrons)
    wavefunction = WaveFunction(num_bf * 2)
    wavefunction.add_determinant(hf_det, 1)
    wavefunction.c_mo = S_sqrt
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    wavefunction = optimize_kappa(wavefunction, h_core, g_eri)
    assert (
        abs(
            expectation_value(wavefunction, H(h_core, g_eri, wavefunction.c_mo), wavefunction)
            - (-4.261756255153)
        )
        < 10**-5
    )

def test_H2_431G() -> None:
    """Test CAS(2,2) through the second quantization module."""
    A = sq.SlowQuant()
    A.set_molecule(
    """H 1.401   0  -0.000000000000;
       H 0   0  -0.000000000000;""",
    distance_unit="bohr",
    )
    A.set_basis_set("4-31G")
    A.init_hartree_fock()
    A.hartree_fock.run_restricted_hartree_fock()
    num_bf = A.molecule.number_bf
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    hf_det = [1] * A.molecule.number_electrons + [0] * (num_bf * 2 - A.molecule.number_electrons)
    wavefunction = WaveFunction(num_bf * 2)
    wavefunction.add_determinant(hf_det, 1)
    wavefunction.add_determinant([0,0,1,1, 0, 0, 0, 0], 0)
    wavefunction.add_determinant([1,0,0,1, 0, 0, 0, 0], 0)
    wavefunction.add_determinant([0,1,1,0, 0, 0, 0, 0], 0)
    wavefunction.c_mo = A.hartree_fock.mo_coeff
    wavefunction = optimize_wavefunction_parameters(wavefunction, h_core, g_eri)
    assert (
        abs(
            expectation_value(wavefunction, H(h_core, g_eri, wavefunction.c_mo), wavefunction)
            - (-1.86003134151516)
        )
        < 10**-5
    )
