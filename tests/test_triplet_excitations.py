# import numpy as np

import slowquant.SlowQuant as sq
import slowquant.unitary_coupled_cluster.linear_response.naive_triplet as naive_t  # pylint: disable=consider-using-from-import
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


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
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "SD",
    )
    WF.run_wf_optimization_1step("SLSQP", True)

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
        molecular_charge=2,
    )
    SQobj.set_basis_set("STO-3G")
    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    # OO-UCCSD
    WF = WaveFunctionUCC(
        SQobj.molecule.number_electrons,
        (2, 4),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "SD",
    )
    WF.run_wf_optimization_1step("SLSQP", True)

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
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "SD",
    )
    WF.run_wf_optimization_1step("SLSQP", True)

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


def test_h10_sto3g_naive_triplet():
    """
    Test of triplet excitation energies for naive LR with working equations
    """
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0  0.0  0.0;
           H  1.4  0.0  0.0;
           H  2.8  0.0  0.0;
           H  4.2  0.0  0.0;
           H  5.6  0.0  0.0;
           H  7.0  0.0  0.0;
           H  8.4  0.0  0.0;
           H  9.8  0.0  0.0;
           H 11.2  0.0  0.0;
           H 12.6  0.0  0.0;""",
        distance_unit="bohr",
    )
    SQobj.set_basis_set("STO-3G")
    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    # OO-UCCSD
    WF = WaveFunctionUCC(
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "SD",
    )
    WF.run_wf_optimization_1step("SLSQP", True)

    # Linear Response
    LR = naive_t.LinearResponseUCC(WF, excitations="SD")
    LR.calc_excitation_energies()

    thresh = 10**-4

    # Check excitation energies
    assert abs(LR.excitation_energies[0] - 0.174084) < thresh
    assert abs(LR.excitation_energies[1] - 0.283825) < thresh
    assert abs(LR.excitation_energies[2] - 0.399352) < thresh
    assert abs(LR.excitation_energies[3] - 0.595528) < thresh
    assert abs(LR.excitation_energies[4] - 0.661147) < thresh
    assert abs(LR.excitation_energies[5] - 0.729917) < thresh
    assert abs(LR.excitation_energies[6] - 0.820365) < thresh
    assert abs(LR.excitation_energies[7] - 0.858801) < thresh
    assert abs(LR.excitation_energies[8] - 1.013200) < thresh
    assert abs(LR.excitation_energies[9] - 1.030337) < thresh
    assert abs(LR.excitation_energies[10] - 1.096057) < thresh
    assert abs(LR.excitation_energies[11] - 1.130836) < thresh
    assert abs(LR.excitation_energies[12] - 1.186165) < thresh
    assert abs(LR.excitation_energies[13] - 1.230153) < thresh
    assert abs(LR.excitation_energies[14] - 1.337101) < thresh
    assert abs(LR.excitation_energies[15] - 1.356619) < thresh
    assert abs(LR.excitation_energies[16] - 1.368000) < thresh
    assert abs(LR.excitation_energies[17] - 1.472716) < thresh
    assert abs(LR.excitation_energies[18] - 1.523427) < thresh
    assert abs(LR.excitation_energies[19] - 1.598596) < thresh
    assert abs(LR.excitation_energies[20] - 1.685950) < thresh
    assert abs(LR.excitation_energies[21] - 1.712598) < thresh
    assert abs(LR.excitation_energies[22] - 1.856746) < thresh
    assert abs(LR.excitation_energies[23] - 2.032029) < thresh
    assert abs(LR.excitation_energies[24] - 2.079022) < thresh
    assert abs(LR.excitation_energies[25] - 2.111754) < thresh
    assert abs(LR.excitation_energies[26] - 2.210035) < thresh
    assert abs(LR.excitation_energies[27] - 2.242963) < thresh
    assert abs(LR.excitation_energies[28] - 2.408660) < thresh
    assert abs(LR.excitation_energies[29] - 2.504838) < thresh
    assert abs(LR.excitation_energies[30] - 2.564356) < thresh
    assert abs(LR.excitation_energies[31] - 2.751666) < thresh
