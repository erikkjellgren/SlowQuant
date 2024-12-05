#import numpy as np

import slowquant.SlowQuant as sq
import slowquant.unitary_coupled_cluster.linear_response.naive_triplet as naive_t  # pylint: disable=consider-using-from-import
from slowquant.unitary_coupled_cluster.ucc_wavefunction_triplet import WaveFunctionUCC


def test_H2_sto3g_naive_triplet():
    """
    Test of triplet excitation energies for naive LR with working equations
    """
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0   0.0  0.0;
            H  0.74  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    # OO-UCCSD
    WF = WaveFunctionUCC(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "SD",
    )
    WF.run_ucc(True)

    # Linear Response
    LR = naive_t.LinearResponseUCC(WF, excitations="SD")
    LR.calc_excitation_energies()

    thresh = 10**-4

    # Check excitation energies
    assert abs(LR.excitation_energies[0] - 0.606510) < thresh


def test_H4plus2_sto3g_naive_triplet():
    """
    Test of triplet excitation energies for naive LR with working equations
    """
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0  0.0  0.0;
            H  0.0  0.0  1.0;
            H  0.74  0.0  0.0;
            H  0.74  0.0  1.0;""",
        distance_unit="angstrom",
        molecular_charge=2
    )
    SQobj.set_basis_set("STO-3G")
    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    # OO-UCCSD
    WF = WaveFunctionUCC(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 4),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "SD",
    )
    WF.run_ucc(True)

    # Linear Response
    LR = naive_t.LinearResponseUCC(WF, excitations="SD")
    LR.calc_excitation_energies()

    thresh = 10**-4

    # Check excitation energies
    assert abs(LR.excitation_energies[0] - 0.349622) < thresh
    assert abs(LR.excitation_energies[1] - 0.640587) < thresh
    assert abs(LR.excitation_energies[2] - 1.166422) < thresh
    assert abs(LR.excitation_energies[3] - 1.324957) < thresh
    assert abs(LR.excitation_energies[4] - 1.763337) < thresh
    assert abs(LR.excitation_energies[5] - 1.996678) < thresh


def test_LiH_sto3g_naive_triplet():
    """
    Test of triplet excitation energies for naive LR with working equations
    """
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0  0.0  0.0;
            H  0.8  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    # OO-UCCSD
    WF = WaveFunctionUCC(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "SD",
    )
    WF.run_ucc(True)

    # Linear Response
    LR = naive_t.LinearResponseUCC(WF, excitations="SD")
    LR.calc_excitation_energies()

    thresh = 10**-4

    # Check excitation energies
    assert abs(LR.excitation_energies[0] - 0.102103) < thresh
    assert abs(LR.excitation_energies[1] - 0.138627) < thresh
    assert abs(LR.excitation_energies[2] - 0.138627) < thresh
    assert abs(LR.excitation_energies[3] - 0.459503) < thresh
    assert abs(LR.excitation_energies[4] - 0.676406) < thresh
    assert abs(LR.excitation_energies[5] - 0.676406) < thresh
    assert abs(LR.excitation_energies[6] - 0.786396) < thresh
    assert abs(LR.excitation_energies[7] - 2.120330) < thresh
    assert abs(LR.excitation_energies[8] - 2.195168) < thresh
    assert abs(LR.excitation_energies[9] - 2.195168) < thresh
    assert abs(LR.excitation_energies[10] - 2.597856) < thresh
    assert abs(LR.excitation_energies[11] - 3.060383) < thresh
