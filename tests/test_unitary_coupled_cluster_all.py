# type: ignore
import numpy as np

import slowquant.SlowQuant as sq
import slowquant.unitary_coupled_cluster.linear_response.allprojected as allprojected  # pylint: disable=consider-using-from-import
import slowquant.unitary_coupled_cluster.linear_response.allselfconsistent as allselfconsistent  # pylint: disable=consider-using-from-import
import slowquant.unitary_coupled_cluster.linear_response.allstatetransfer as allstatetransfer  # pylint: disable=consider-using-from-import
import slowquant.unitary_coupled_cluster.linear_response.naive as naive  # pylint: disable=consider-using-from-import
import slowquant.unitary_coupled_cluster.linear_response.projected as projected  # pylint: disable=consider-using-from-import
import slowquant.unitary_coupled_cluster.linear_response.projected_statetransfer as projected_statetransfer  # pylint: disable=consider-using-from-import
import slowquant.unitary_coupled_cluster.linear_response.selfconsistent as selfconsistent  # pylint: disable=consider-using-from-import
import slowquant.unitary_coupled_cluster.linear_response.statetransfer as statetransfer  # pylint: disable=consider-using-from-import
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


def test_h2_sto3g_uccsd_lr() -> None:
    """Test Linear Response for uccsd with all transform methods."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0  0.0  0.0;
            H  0.0  0.0  0.7;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "SD",
    )
    WF.run_wf_optimization_1step("SLSQP", False)
    # SC
    LR = allselfconsistent.LinearResponseUCC(
        WF,
        excitations="SD",
    )
    LR.calc_excitation_energies()
    assert abs(LR.excitation_energies[0] - 1.0157376) < 10**-3
    assert abs(LR.excitation_energies[1] - 1.71950367) < 10**-3
    # ST
    LR = allstatetransfer.LinearResponseUCC(  # type: ignore [assigment]
        WF,
        excitations="SD",
    )
    LR.calc_excitation_energies()
    assert abs(LR.excitation_energies[0] - 1.0157376) < 10**-3
    assert abs(LR.excitation_energies[1] - 1.71950367) < 10**-3


def test_LiH_atmethods_energies() -> None:
    """Test LiH at LR methods."""

    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li   0.0           0.0  0.0;
        H   1.67  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "SD",
    )
    WF.run_wf_optimization_1step("SLSQP", True)

    threshold = 10 ** (-3)

    # atSC
    LR_atSC = allselfconsistent.LinearResponseUCC(
        WF,
        excitations="SD",
    )
    LR_atSC.calc_excitation_energies()

    solutions = np.array(
        [
            0.1850601,
            0.24646229,
            0.24646229,
            0.62305832,
            0.85960246,
            2.07742277,
            2.13715343,
            2.13715343,
            2.551118,
        ]
    )

    assert np.allclose(LR_atSC.excitation_energies, solutions, atol=threshold)

    # atST
    LR_atST = allstatetransfer.LinearResponseUCC(
        WF,
        excitations="SD",
    )
    LR_atST.calc_excitation_energies()

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

    assert np.allclose(LR_atST.excitation_energies, solutions, atol=threshold)


def test_LiH_naiveq_methods_energies() -> None:
    """Test LiH energies for naive q LR methods."""

    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li   0.0           0.0  0.0;
        H   1.67  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "SD",
    )
    WF.run_wf_optimization_1step("SLSQP", True)

    threshold = 10 ** (-5)

    # naive
    LR_naive = naive.LinearResponseUCC(WF, excitations="SD")
    LR_naive.calc_excitation_energies()

    solutions = np.array(
        [
            0.12957563,
            0.17886086,
            0.17886086,
            0.60514579,
            0.64715988,
            0.74104045,
            0.74104045,
            1.00396876,
            2.0747935,
            2.13715595,
            2.13715595,
            2.45575825,
            2.95516593,
        ]
    )
    assert np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)

    # proj:
    LR_naive = projected.LinearResponseUCC(  # type: ignore [assigment]
        WF,
        excitations="SD",
    )
    LR_naive.calc_excitation_energies()

    solutions = np.array(
        [
            0.12957561,
            0.17886086,
            0.17886086,
            0.60514593,
            0.6471598,
            0.74104045,
            0.74104045,
            1.00396873,
            2.0747935,
            2.13715595,
            2.13715595,
            2.45575825,
            2.95516593,
        ]
    )
    assert np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)

    # SC
    LR_naive = selfconsistent.LinearResponseUCC(  # type: ignore [assigment]
        WF,
        excitations="SD",
    )
    LR_naive.calc_excitation_energies()

    solutions = np.array(
        [
            0.12957564,
            0.17886086,
            0.17886086,
            0.60514581,
            0.64715989,
            0.74104045,
            0.74104045,
            1.00396876,
            2.0747935,
            2.13715595,
            2.13715595,
            2.45575825,
            2.95516593,
        ]
    )
    assert np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)

    # ST
    LR_naive = statetransfer.LinearResponseUCC(  # type: ignore [assigment]
        WF,
        excitations="SD",
    )
    LR_naive.calc_excitation_energies()

    solutions = np.array(
        [
            0.12957561,
            0.17886086,
            0.17886086,
            0.60514593,
            0.64715981,
            0.74104045,
            0.74104045,
            1.00396874,
            2.0747935,
            2.13715595,
            2.13715595,
            2.45575825,
            2.95516593,
        ]
    )
    assert np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)


def test_LiH_naiveq_methods_matrices() -> None:
    """Test LiH all matrices and their properties for naive q LR methods."""

    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li   0.0           0.0  0.0;
        H   1.67  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "SD",
    )
    WF.run_wf_optimization_1step("SLSQP", True)

    threshold = 10 ** (-5)

    # naive
    LR_naive = naive.LinearResponseUCC(
        WF,
        excitations="SD",
    )

    assert np.allclose(LR_naive.A, LR_naive.A.T, atol=threshold)
    assert np.allclose(LR_naive.B, LR_naive.B.T, atol=threshold)
    assert np.allclose(LR_naive.Delta, np.zeros_like(LR_naive.Delta), atol=threshold)

    # SC
    LR_naive = selfconsistent.LinearResponseUCC(  # type: ignore [assigment]
        WF,
        excitations="SD",
    )

    assert np.allclose(LR_naive.A, LR_naive.A.T, atol=threshold)
    assert np.allclose(LR_naive.B, LR_naive.B.T, atol=threshold)
    assert np.allclose(LR_naive.Delta, np.zeros_like(LR_naive.Delta), atol=threshold)

    # projected:
    threshold = 10 ** (-5)
    LR_naive = projected.LinearResponseUCC(  # type: ignore [assigment]
        WF,
        excitations="SD",
    )

    threshold = 10 ** (-5)
    assert np.allclose(LR_naive.A, LR_naive.A.T, atol=threshold)
    assert np.allclose(LR_naive.B, LR_naive.B.T, atol=threshold)
    assert np.allclose(LR_naive.Delta, np.zeros_like(LR_naive.Delta), atol=threshold)

    # ST
    LR_naive = statetransfer.LinearResponseUCC(  # type: ignore [assigment]
        WF,
        excitations="SD",
    )

    assert np.allclose(LR_naive.A, LR_naive.A.T, atol=threshold)
    assert np.allclose(LR_naive.B, LR_naive.B.T, atol=threshold)
    assert np.allclose(LR_naive.Delta, np.zeros_like(LR_naive.Delta), atol=threshold)


def test_LiH_allproj_energies() -> None:
    """Test LiH for all-proj LR method."""

    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li   0.0           0.0  0.0;
        H   1.67  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "SD",
    )
    WF.run_wf_optimization_1step("SLSQP", True)

    threshold = 10 ** (-5)

    LR_naive = allprojected.LinearResponseUCC(
        WF,
        excitations="SD",
    )
    LR_naive.calc_excitation_energies()

    solutions = np.array(
        [
            0.12973291,
            0.18092743,
            0.18092743,
            0.60537541,
            0.64747353,
            0.74982411,
            0.74982411,
            1.00424384,
            2.07489665,
            2.13720665,
            2.13720665,
            2.45601484,
            2.95606043,
        ]
    )
    assert np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)


def test_LiH_STproj_energies() -> None:
    """Test LiH for ST-proj LR method."""

    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li   0.0           0.0  0.0;
        H   1.67  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "SD",
    )
    WF.run_wf_optimization_1step("SLSQP", True)

    threshold = 10 ** (-5)

    LR_naive = projected_statetransfer.LinearResponseUCC(
        WF,
        excitations="SD",
    )
    LR_naive.calc_excitation_energies()

    solutions = np.array(
        [
            0.12973291,
            0.18092743,
            0.18092743,
            0.60537541,
            0.64747353,
            0.74982411,
            0.74982411,
            1.00424384,
            2.07489665,
            2.13720665,
            2.13720665,
            2.45601484,
            2.95606043,
        ]
    )

    assert np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)
