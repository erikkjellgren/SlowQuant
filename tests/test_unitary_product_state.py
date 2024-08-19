# type: ignore
import numpy as np

import slowquant.SlowQuant as sq
import slowquant.unitary_coupled_cluster.linear_response.allstatetransfer as allstLR
import slowquant.unitary_coupled_cluster.linear_response.naive as naiveLR
from slowquant.unitary_coupled_cluster.sa_ups_wavefunction import WaveFunctionSAUPS
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS


def test_ups_naivelr() -> None:
    """Test LiH UCCSD(2,2) LR"""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li 0.0           0.0  0.0;
           H  1.6717072740  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    WF = WaveFunctionUPS(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "tUPS",
        ansatz_options={"n_layers": 1, "skip_last_singles": True},
        include_active_kappa=True,
    )
    dipole_integrals = (
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    )

    WF.run_ups(True)

    LR = naiveLR.LinearResponseUCC(WF, excitations="SD")
    LR.calc_excitation_energies()
    assert abs(LR.excitation_energies[0] - 0.129476) < 10**-4
    assert abs(LR.excitation_energies[1] - 0.178749) < 10**-4
    assert abs(LR.excitation_energies[2] - 0.178749) < 10**-4
    assert abs(LR.excitation_energies[3] - 0.604681) < 10**-4
    assert abs(LR.excitation_energies[4] - 0.646707) < 10**-4
    assert abs(LR.excitation_energies[5] - 0.740632) < 10**-4
    assert abs(LR.excitation_energies[6] - 0.740632) < 10**-4
    assert abs(LR.excitation_energies[7] - 1.002914) < 10**-4
    assert abs(LR.excitation_energies[8] - 2.074822) < 10**-4
    assert abs(LR.excitation_energies[9] - 2.137193) < 10**-4
    assert abs(LR.excitation_energies[10] - 2.137193) < 10**-4
    assert abs(LR.excitation_energies[11] - 2.455191) < 10**-4
    assert abs(LR.excitation_energies[12] - 2.954372) < 10**-4
    osc_strengths = LR.get_oscillator_strength(dipole_integrals)
    assert abs(osc_strengths[0] - 0.049920) < 10**-3
    assert abs(osc_strengths[1] - 0.241184) < 10**-3
    assert abs(osc_strengths[2] - 0.241184) < 10**-3
    assert abs(osc_strengths[3] - 0.158045) < 10**-3
    assert abs(osc_strengths[4] - 0.166539) < 10**-3
    assert abs(osc_strengths[5] - 0.010379) < 10**-3
    assert abs(osc_strengths[6] - 0.010379) < 10**-3
    assert abs(osc_strengths[7] - 0.006256) < 10**-3
    assert abs(osc_strengths[8] - 0.062386) < 10**-3
    assert abs(osc_strengths[9] - 0.128862) < 10**-3
    assert abs(osc_strengths[10] - 0.128862) < 10**-3
    assert abs(osc_strengths[11] - 0.046007) < 10**-3
    assert abs(osc_strengths[12] - 0.003904) < 10**-3


def test_LiH_sto3g_allST():
    """
    Test LiH STO-3G all-statetransfer LR oscialltor strength
    """
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0  0.0  0.0;
            H 1.67 0.0 0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    # HF
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
        "SD",
    )
    WF.run_ucc(True)
    WF2 = WaveFunctionUPS(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        WF.c_trans,
        h_core,
        g_eri,
        "tUPS",
        ansatz_options={"n_layers": 1},
    )
    WF2.run_ups(False)

    # Linear Response
    LR = allstLR.LinearResponseUCC(
        WF2,
        excitations="SD",
    )
    LR.calc_excitation_energies()

    thresh = 10**-3

    # Check excitation energies
    solutions = np.array(
        [
            0.1851181,
            0.24715136,
            0.24715136,
            0.6230648,
            0.85960395,
            2.07752209,
            2.13720198,
            2.13720198,
            2.55113802,
        ]
    )

    assert np.allclose(LR.excitation_energies, solutions, atol=thresh)

    # Calculate dipole integrals
    dipole_integrals = (
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    )

    # Get oscillator strength for each excited state
    osc_strengths = LR.get_oscillator_strength(dipole_integrals)
    assert abs(osc_strengths[0] - 0.06668878) < thresh
    assert abs(osc_strengths[1] - 0.33360367) < thresh
    assert abs(osc_strengths[2] - 0.33360367) < thresh
    assert abs(osc_strengths[3] - 0.30588158) < thresh
    assert abs(osc_strengths[4] - 0.02569977) < thresh
    assert abs(osc_strengths[5] - 0.06690658) < thresh
    assert abs(osc_strengths[6] - 0.13411942) < thresh
    assert abs(osc_strengths[7] - 0.13411942) < thresh
    assert abs(osc_strengths[8] - 0.04689274) < thresh


def test_ups_water_44() -> None:
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """O   0.0  0.0           0.1035174918;
    H   0.0  0.7955612117 -0.4640237459;
    H   0.0 -0.7955612117 -0.4640237459;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor

    WF = WaveFunctionUPS(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (4, 4),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "tUPS",
        ansatz_options={"n_layers": 2, "skip_last_singles": True},
        include_active_kappa=True,
    )

    WF.run_ups(True)
    assert abs(WF.energy_elec - -84.00398456629122) < 10**-8


def test_saups_h2_3states() -> None:
    """This should give exactly the same as FCI since all states,
    are includes in the subspace expansion.
    """
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H   0.0  0.0  0.0;
           H   0.0  0.0  0.735;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor

    WF = WaveFunctionSAUPS(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        (
            [
                [1],
                [2 ** (-1 / 2), -(2 ** (-1 / 2))],
                [1],
            ],
            [
                ["1100"],
                ["1001", "0110"],
                ["0011"],
            ],
        ),
        "tUPS",
        ansatz_options={"n_layers": 1, "skip_last_singles": True},
        include_active_kappa=True,
    )

    WF.run_saups(True)

    dipole_integrals = (
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    )

    assert abs(WF.excitation_energies[0] - 0.974553) < 10**-6
    assert abs(WF.excitation_energies[1] - 1.632364) < 10**-6
    osc = WF.get_oscillator_strenghts(dipole_integrals)
    assert abs(osc[0] - 0.8706) < 10**-3
    assert abs(osc[1] - 0.0) < 10**-3


def test_saups_h3_3states() -> None:
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H   -0.45  -0.3897114317  0.0;
           H   0.45  -0.3897114317  0.0;
           H   0.0  0.3897114317  0.0;""",
        distance_unit="angstrom",
        molecular_charge=1,
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor

    WF = WaveFunctionSAUPS(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 3),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        (
            [
                [1],
                [2 ** (-1 / 2), -(2 ** (-1 / 2))],
                [2 ** (-1 / 2), -(2 ** (-1 / 2))],
            ],
            [
                ["110000"],
                ["100100", "011000"],
                ["100001", "010010"],
            ],
        ),
        "tUPS",
        ansatz_options={"n_layers": 2, "skip_last_singles": True},
        include_active_kappa=True,
    )

    WF.run_saups(True)

    dipole_integrals = (
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    )

    assert abs(WF.excitation_energies[0] - 0.838466) < 10**-6
    assert abs(WF.excitation_energies[1] - 0.838466) < 10**-6
    osc = WF.get_oscillator_strenghts(dipole_integrals)
    assert abs(osc[0] - 0.7569) < 10**-3
    assert abs(osc[1] - 0.7569) < 10**-3
