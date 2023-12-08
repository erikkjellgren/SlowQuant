import numpy as np

import slowquant.SlowQuant as sq
import slowquant.unitary_coupled_cluster.linear_response.allprojected as allprojected
import slowquant.unitary_coupled_cluster.linear_response.allselfconsistent as allselfconsistent
import slowquant.unitary_coupled_cluster.linear_response.allstatetransfer as allstatetransfer
import slowquant.unitary_coupled_cluster.linear_response.generic as generic
import slowquant.unitary_coupled_cluster.linear_response.naive as naive
import slowquant.unitary_coupled_cluster.linear_response.projected as projected
import slowquant.unitary_coupled_cluster.linear_response.projected_statetransfer as projected_statetransfer
import slowquant.unitary_coupled_cluster.linear_response.selfconsistent as selfconsistent
import slowquant.unitary_coupled_cluster.linear_response.statetransfer as statetransfer
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
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
    )
    WF.run_ucc("SD", False)
    # SC
    LR = allselfconsistent.LinearResponseUCC(
        WF,
        excitations="SD",
    )
    LR.calc_excitation_energies()
    assert abs(LR.excitation_energies[0] - 1.0157376) < 10**-3
    assert abs(LR.excitation_energies[1] - 1.71950367) < 10**-3
    # ST
    LR = allstatetransfer.LinearResponseUCC(
        WF,
        excitations="SD",
    )
    LR.calc_excitation_energies()
    assert abs(LR.excitation_energies[0] - 1.0157376) < 10**-3
    assert abs(LR.excitation_energies[1] - 1.71950367) < 10**-3


def test_h2_sto3g_uccsd_lr_matrices() -> None:
    """Test matrices of Linear Response for uccsd with all transform methods."""
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
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
    )

    WF.run_ucc("SD", False)

    # SC
    LR_naive = allselfconsistent.LinearResponseUCC(
        WF,
        excitations="SD",
    )
    LR_native = generic.LinearResponseUCC(
        WF,
        excitations="SD",
        operator_type="selfconsistent",
        do_transform_orbital_rotations=True,
    )

    threshold = 10 ** (-5)

    assert np.allclose(LR_naive.A, LR_native.A, atol=threshold)
    assert np.allclose(LR_naive.B, LR_native.B, atol=threshold)
    assert np.allclose(LR_naive.Sigma, LR_native.Sigma, atol=threshold)
    assert np.allclose(LR_naive.Delta, LR_native.Delta, atol=threshold)

    # ST
    LR_naive = allstatetransfer.LinearResponseUCC(
        WF,
        excitations="SD",
    )
    LR_native = generic.LinearResponseUCC(
        WF,
        excitations="SD",
        operator_type="statetransfer",
        do_transform_orbital_rotations=True,
    )

    assert np.allclose(LR_naive.A, LR_native.A, atol=threshold)
    assert np.allclose(LR_naive.B, LR_native.B, atol=threshold)
    assert np.allclose(LR_naive.Sigma, LR_native.Sigma, atol=threshold)
    assert np.allclose(LR_naive.Delta, LR_native.Delta, atol=threshold)


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
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
    )
    WF.run_ucc("SD", True)

    threshold = 10 ** (-3)

    # atSC
    LR_naive = allselfconsistent.LinearResponseUCC(
        WF,
        excitations="SD",
    )
    LR_naive.calc_excitation_energies()

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

    assert np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)

    # atST
    LR_naive = allstatetransfer.LinearResponseUCC(
        WF,
        excitations="SD",
    )
    LR_naive.calc_excitation_energies()

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

    assert np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)


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
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
    )
    WF.run_ucc("SD", True)

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
    LR_naive = projected.LinearResponseUCC(
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
    LR_naive = selfconsistent.LinearResponseUCC(
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
    LR_naive = statetransfer.LinearResponseUCC(
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

    # HST
    LR_HST_naive = statetransfer.LinearResponseUCC(
        WF,
        excitations="SD",
        do_approximate_hermitification=True,
    )
    LR_HST_naive.calc_excitation_energies()

    solutions = np.array(
        [
            0.12957234,
            0.17886086,
            0.17886086,
            0.60515489,
            0.64717441,
            0.74104045,
            0.74104045,
            1.003942,
            2.07479277,
            2.13715595,
            2.13715595,
            2.45576414,
            2.95517029,
        ]
    )

    assert np.allclose(LR_HST_naive.excitation_energies, solutions, atol=threshold * 10)

    # HSC
    LR_HSC_naive = selfconsistent.LinearResponseUCC(
        WF,
        excitations="SD",
        do_approximate_hermitification=True,
    )
    LR_HSC_naive.calc_excitation_energies()

    solutions = np.array(
        [
            0.12957234,
            0.17886086,
            0.17886086,
            0.60515489,
            0.64717441,
            0.74104045,
            0.74104045,
            1.003942,
            2.07479277,
            2.13715595,
            2.13715595,
            2.45576414,
            2.95517029,
        ]
    )

    assert np.allclose(LR_HSC_naive.excitation_energies, solutions, atol=threshold * 10)


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
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
    )
    WF.run_ucc("SD", True)

    threshold = 10 ** (-5)

    # naive
    LR_naive = naive.LinearResponseUCC(
        WF,
        excitations="SD",
    )
    LR_generic = generic.LinearResponseUCC(
        WF,
        excitations="SD",
        operator_type="naive",
    )

    assert np.allclose(LR_naive.A, LR_generic.A, atol=threshold)
    assert np.allclose(LR_naive.B, LR_generic.B, atol=threshold)
    assert np.allclose(LR_naive.Sigma, LR_generic.Sigma, atol=threshold)
    assert np.allclose(LR_naive.Delta, LR_generic.Delta, atol=threshold)

    assert np.allclose(LR_naive.A, LR_naive.A.T, atol=threshold)
    assert np.allclose(LR_naive.B, LR_naive.B.T, atol=threshold)
    assert np.allclose(LR_naive.Delta, np.zeros_like(LR_naive.Delta), atol=threshold)

    assert np.allclose(LR_generic.A, LR_generic.A.T, atol=threshold)
    assert np.allclose(LR_generic.B, LR_generic.B.T, atol=threshold)
    assert np.allclose(LR_generic.Delta, np.zeros_like(LR_generic.Delta), atol=threshold)

    # SC
    LR_naive = selfconsistent.LinearResponseUCC(
        WF,
        excitations="SD",
    )
    LR_generic = generic.LinearResponseUCC(
        WF,
        excitations="SD",
        operator_type="selfconsistent",
    )

    assert np.allclose(LR_naive.A, LR_generic.A, atol=threshold)
    assert np.allclose(LR_naive.B, LR_generic.B, atol=threshold)
    assert np.allclose(LR_naive.Sigma, LR_generic.Sigma, atol=threshold)
    assert np.allclose(LR_naive.Delta, LR_generic.Delta, atol=threshold)

    assert np.allclose(LR_naive.A, LR_naive.A.T, atol=threshold)
    assert np.allclose(LR_naive.B, LR_naive.B.T, atol=threshold)
    assert np.allclose(LR_naive.Delta, np.zeros_like(LR_naive.Delta), atol=threshold)

    # projected:
    threshold = 10 ** (-5)
    LR_naive = projected.LinearResponseUCC(
        WF,
        excitations="SD",
    )
    LR_generic = generic.LinearResponseUCC(
        WF,
        excitations="SD",
        operator_type="projected",
    )

    assert np.allclose(LR_naive.A, LR_generic.A, atol=threshold)
    assert np.allclose(LR_naive.B, LR_generic.B, atol=threshold)
    assert np.allclose(LR_naive.Sigma, LR_generic.Sigma, atol=threshold)
    assert np.allclose(LR_naive.Delta, LR_generic.Delta, atol=threshold)

    threshold = 10 ** (-5)
    assert np.allclose(LR_naive.A, LR_naive.A.T, atol=threshold)
    assert np.allclose(LR_naive.B, LR_naive.B.T, atol=threshold)
    assert np.allclose(LR_naive.Delta, np.zeros_like(LR_naive.Delta), atol=threshold)

    # ST
    LR_naive = statetransfer.LinearResponseUCC(
        WF,
        excitations="SD",
    )
    LR_generic = generic.LinearResponseUCC(
        WF,
        excitations="SD",
        operator_type="statetransfer",
    )

    assert np.allclose(LR_naive.A, LR_generic.A, atol=threshold)
    assert np.allclose(LR_naive.B, LR_generic.B, atol=threshold)
    assert np.allclose(LR_naive.Sigma, LR_generic.Sigma, atol=threshold)
    assert np.allclose(LR_naive.Delta, LR_generic.Delta, atol=threshold)

    assert np.allclose(LR_naive.A, LR_naive.A.T, atol=threshold)
    assert np.allclose(LR_naive.B, LR_naive.B.T, atol=threshold)
    assert np.allclose(LR_naive.Delta, np.zeros_like(LR_naive.Delta), atol=threshold)

    # HST
    LR_HST_naive = statetransfer.LinearResponseUCC(
        WF,
        excitations="SD",
        do_approximate_hermitification=True,
    )

    assert np.allclose(LR_HST_naive.A, LR_HST_naive.A.T, atol=threshold)
    assert np.allclose(LR_HST_naive.B, LR_HST_naive.B.T, atol=threshold)


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
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
    )
    WF.run_ucc("SD", True)

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
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
    )
    WF.run_ucc("SD", True)

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

    # HSTproj
    LR_HSTproj_naive = projected_statetransfer.LinearResponseUCC(
        WF,
        excitations="SD",
        do_approximate_hermitification=True,
    )
    LR_HSTproj_naive.calc_excitation_energies()

    solutions = np.array(
        [
            0.12972911,
            0.18092787,
            0.18092787,
            0.6053739,
            0.64747858,
            0.74982806,
            0.74982806,
            1.00424528,
            2.07489505,
            2.13720683,
            2.13720683,
            2.45602345,
            2.95608694,
        ]
    )

    assert np.allclose(LR_HSTproj_naive.excitation_energies, solutions, atol=threshold * 10)
