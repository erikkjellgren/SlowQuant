import sys

import numpy as np

import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.linear_response_alltransform import (
    LinearResponseUCC,
)
from slowquant.unitary_coupled_cluster.linear_response_fast import (
    LinearResponseUCC as LinearResponseUCCFast,
)
from slowquant.unitary_coupled_cluster.linear_response_old import (
    LinearResponseUCC as LinearResponseUCCRef,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


def test_h2_sto3g_uccsd_lr() -> None:
    """Test Linear Response for uccsd with all transform methods."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0  0.0  0.0;
            H  0.0  0.0  0.7;""",
        distance_unit='angstrom',
    )
    SQobj.set_basis_set('sto-3g')
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
    dipole_integrals = (
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    )
    WF.run_ucc('SD', False)
    # SC
    LR = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=True,
        do_statetransfer_operators=False,
        do_debugging=True,
    )
    LR.calc_excitation_energies()
    print(LR.excitation_energies)
    assert abs(LR.excitation_energies[0] - 1.0157376) < 10**-3
    assert abs(LR.excitation_energies[1] - 1.71950367) < 10**-3
    assert abs(LR.get_excited_state_overlap(0) - 0.0) < 10**-3
    assert abs(LR.get_excited_state_overlap(1) - 0.0) < 10**-3
    assert abs(abs(LR.get_transition_dipole(0, dipole_integrals)[2]) - 1.1440534325663108) < 10**-3
    assert abs(LR.get_transition_dipole(1, dipole_integrals)[2] - 0.0) < 10**-3
    # ST
    LR = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=False,
        do_statetransfer_operators=True,
        do_debugging=True,
    )
    LR.calc_excitation_energies()
    print(LR.excitation_energies)
    assert abs(LR.excitation_energies[0] - 1.0157376) < 10**-3
    assert abs(LR.excitation_energies[1] - 1.71950367) < 10**-3
    assert abs(LR.get_excited_state_overlap(0) - 0.0) < 10**-3
    assert abs(LR.get_excited_state_overlap(1) - 0.0) < 10**-3
    assert abs(abs(LR.get_transition_dipole(0, dipole_integrals)[2]) - 1.144053440679731) < 10**-3
    assert abs(LR.get_transition_dipole(1, dipole_integrals)[2] - 0.0) < 10**-3


def test_h2_sto3g_uccsd_lr_matrices() -> None:
    """Test matrices of Linear Response for uccsd with all transform methods."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0  0.0  0.0;
            H  0.0  0.0  0.7;""",
        distance_unit='angstrom',
    )
    SQobj.set_basis_set('sto-3g')
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

    WF.run_ucc('SD', False)

    # SC
    LR_naive = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=True,
        do_statetransfer_operators=False,
        do_debugging=False,
    )
    LR_native = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=True,
        do_statetransfer_operators=False,
        do_debugging=True,
    )

    threshold = 10 ** (-5)

    assert (np.allclose(LR_naive.M, LR_native.M, atol=threshold)) is True
    assert (np.allclose(LR_naive.Q, LR_native.Q, atol=threshold)) is True
    assert (np.allclose(LR_naive.V, LR_native.V, atol=threshold)) is True
    assert (np.allclose(LR_naive.W, LR_native.W, atol=threshold)) is True

    # ST
    LR_naive = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=False,
        do_statetransfer_operators=True,
        do_debugging=False,
    )
    LR_native = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=False,
        do_statetransfer_operators=True,
        do_debugging=True,
    )

    assert np.allclose(LR_naive.M, LR_native.M, atol=threshold) is True
    assert np.allclose(LR_naive.Q, LR_native.Q, atol=threshold) is True
    assert np.allclose(LR_naive.V, LR_native.V, atol=threshold) is True
    assert np.allclose(LR_naive.W, LR_native.W, atol=threshold) is True

### START Tests for old linear response class. should still work, but are not that useful.

def test_h2_631g_oouccsd_lr() -> None:
    """Test Linear Response for OO-uccsd(2,2)."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0   0.0  0.0;
           H  0.74  0.0  0.0;""",
        distance_unit='angstrom',
    )
    SQobj.set_basis_set('6-31G')
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
    WF.run_ucc('SD', True)

    # SC
    LR = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=True,
        do_statetransfer_operators=False,
        do_debugging=True,
    )
    LR.calc_excitation_energies()
    dipole_integrals = (
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    )
    assert abs(LR.excitation_energies[0] - 0.57751624) < 10**-3
    assert abs(LR.excitation_energies[1] - 1.04796409) < 10**-3
    assert abs(LR.excitation_energies[2] - 1.63423396) < 10**-3
    assert abs(LR.excitation_energies[3] - 1.6490738) < 10**-3
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.6387323486152973) < 10**-3
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.06634550238278526) < 10**-3
    assert abs(LR.get_oscillator_strength(3, dipole_integrals) - 0.0) < 10**-3

    # ST
    LR = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=False,
        do_statetransfer_operators=True,
        do_debugging=True,
    )
    LR.calc_excitation_energies()
    dipole_integrals = (
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    )
    assert abs(LR.excitation_energies[0] - 0.5777356) < 10**-3
    assert abs(LR.excitation_energies[1] - 1.05253661) < 10**-3
    assert abs(LR.excitation_energies[2] - 1.63445649) < 10**-3
    assert abs(LR.excitation_energies[3] - 1.64921416) < 10**-3
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.6502948423608538) < 10**-3
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.06230175324668715) < 10**-3
    assert abs(LR.get_oscillator_strength(3, dipole_integrals) - 0.0) < 10**-3


def test_h2_631g_oouccsd_lr_matrices() -> None:
    """Test Linear Response for OO-uccsd(2,2)."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0   0.0  0.0;
           H  0.74  0.0  0.0;""",
        distance_unit='angstrom',
    )
    SQobj.set_basis_set('6-31G')
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
    WF.run_ucc('SD', True)

    # SC
    LR_naive = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=True,
        do_statetransfer_operators=False,
        do_debugging=False,
    )
    LR_native = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=True,
        do_statetransfer_operators=False,
        do_debugging=True,
    )

    threshold = 10 ** (-5)

    assert (np.allclose(LR_naive.M, LR_native.M, atol=threshold)) is True
    assert (np.allclose(LR_naive.Q, LR_native.Q, atol=threshold)) is True
    assert (np.allclose(LR_naive.V, LR_native.V, atol=threshold)) is True
    assert (np.allclose(LR_naive.W, LR_native.W, atol=threshold)) is True

    # ST
    LR_naive = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=False,
        do_statetransfer_operators=True,
        do_debugging=False,
    )
    LR_native = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=False,
        do_statetransfer_operators=True,
        do_debugging=True,
    )

    assert (np.allclose(LR_naive.M, LR_native.M, atol=threshold)) is True
    assert (np.allclose(LR_naive.Q, LR_native.Q, atol=threshold)) is True
    assert (np.allclose(LR_naive.V, LR_native.V, atol=threshold)) is True
    assert (np.allclose(LR_naive.W, LR_native.W, atol=threshold)) is True


def test_LiH_atmethods_matrices() -> None:
    """Test LiH all matrices and their properties for at methods."""

    print('Are you sure you want to run this test? It takes ages...')

    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li   0.0           0.0  0.0;
        H   1.67  0.0  0.0;""",
        distance_unit='angstrom',
    )
    SQobj.set_basis_set('sto-3g')
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
    WF.run_ucc('SD', True)

    threshold = 10 ** (-10)

    # atSC
    print('\nMethod: at-SC')
    LR_naive = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=True,
        do_statetransfer_operators=False,
        do_debugging=False,
    )
    LR_generic = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=True,
        do_statetransfer_operators=False,
        do_debugging=True,
    )

    print(
        'Check if implementation via work equation and generic are the same with a threshold of: ', threshold
    )
    assert (np.allclose(LR_naive.M, LR_generic.M, atol=threshold)) == True
    assert (np.allclose(LR_naive.Q, LR_generic.Q, atol=threshold)) is True
    assert (np.allclose(LR_naive.V, LR_generic.V, atol=threshold)) is True
    assert (np.allclose(LR_naive.W, LR_generic.W, atol=threshold)) is True

    print('Check if matrices fullfill their expected property:')
    assert (np.all(np.abs(LR_naive.M - LR_naive.M.T) < threshold)) == True
    assert (np.all(np.abs(LR_naive.Q - LR_naive.Q.T) < threshold)) == True
    assert (np.all(np.abs(LR_naive.V - np.eye(LR_naive.V.shape[0])) < threshold)) == True
    assert (np.all(np.abs(LR_naive.W) < threshold)) == True

    # atST
    print('\nMethod: at-ST')
    LR_naive = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=False,
        do_statetransfer_operators=True,
        do_debugging=False,
    )
    LR_generic = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=False,
        do_statetransfer_operators=True,
        do_debugging=True,
    )

    print(
        'Check if implementation via work equation and generic are the same with a threshold of: ', threshold
    )
    assert (np.allclose(LR_naive.M, LR_generic.M, atol=threshold)) is True
    assert (np.allclose(LR_naive.Q, LR_generic.Q, atol=threshold)) is True  # BUg in generic implementation?
    assert (np.allclose(LR_naive.V, LR_generic.V, atol=threshold)) is True
    assert (np.allclose(LR_naive.W, LR_generic.W, atol=threshold)) is True

    print('Check if matrices fullfill their expected property:')
    assert (np.all(np.abs(LR_naive.M - LR_naive.M.T) < threshold)) == True
    assert (np.all(np.abs(LR_naive.Q - LR_naive.Q.T) < threshold)) == True
    assert (np.all(np.abs(LR_naive.V - np.eye(LR_naive.V.shape[0])) < threshold)) == True
    assert (np.all(np.abs(LR_naive.W) < threshold)) == True


def test_LiH_allmethods_matrices() -> None:
    """Test LiH all matrices and their properties for all LR methods."""

    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li   0.0           0.0  0.0;
        H   1.67  0.0  0.0;""",
        distance_unit='angstrom',
    )
    SQobj.set_basis_set('sto-3g')
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
    WF.run_ucc('SD', True)

    threshold = 10 ** (-10)

    # naive
    print('\nMethod: naive')
    LR_naive = LinearResponseUCCRef(
        WF, excitations='SD', do_projected_operators=False, do_selfconsistent_operators=False
    )
    LR_generic = LinearResponseUCCRef(
        WF,
        excitations='SD',
        do_projected_operators=False,
        do_selfconsistent_operators=False,
        do_debugging=True,
    )

    print(
        'Check if implementation via work equation and generic are the same with a threshold of: ', threshold
    )
    assert (np.allclose(LR_naive.M, LR_generic.M, atol=threshold)) is True
    assert (np.allclose(LR_naive.Q, LR_generic.Q, atol=threshold)) is True
    assert (np.allclose(LR_naive.V, LR_generic.V, atol=threshold)) is True
    assert (np.allclose(LR_naive.W, LR_generic.W, atol=threshold)) is True

    print('Check if matrices fullfill their expected property:')
    assert (np.all(np.abs(LR_naive.M - LR_naive.M.T) < threshold)) == True
    assert (np.all(np.abs(LR_naive.Q - LR_naive.Q.T) < threshold)) == True
    assert (np.all(np.abs(LR_naive.W) < threshold)) == True

    print('Check if generic matrices fullfill their expected property:')
    assert (np.all(np.abs(LR_generic.M - LR_generic.M.T) < threshold)) == True
    assert (np.all(np.abs(LR_generic.Q - LR_generic.Q.T) < threshold)) == True
    assert (np.all(np.abs(LR_generic.W) < threshold)) == True

    # SC
    print('\nMethod: SC')
    LR_naive = LinearResponseUCCRef(
        WF, excitations='SD', do_projected_operators=False, do_selfconsistent_operators=True
    )
    LR_generic = LinearResponseUCCRef(
        WF,
        excitations='SD',
        do_projected_operators=False,
        do_selfconsistent_operators=True,
        do_debugging=True,
    )

    print(
        'Check if implementation via work equation and generic are the same with a threshold of: ', threshold
    )
    assert (np.allclose(LR_naive.M, LR_generic.M, atol=threshold)) is True
    assert (np.allclose(LR_naive.Q, LR_generic.Q, atol=threshold)) is True
    assert (np.allclose(LR_naive.V, LR_generic.V, atol=threshold)) is True
    assert (np.allclose(LR_naive.W, LR_generic.W, atol=threshold)) is True

    print('Check if matrices fullfill their expected property:')
    assert (np.all(np.abs(LR_naive.M - LR_naive.M.T) < threshold)) == True
    assert (np.all(np.abs(LR_naive.Q - LR_naive.Q.T) < threshold)) == True
    assert (np.all(np.abs(LR_naive.W) < threshold)) == True

    # projected: only generic is implemented atm
    print('\nMethod: projected')
    LR_generic = LinearResponseUCCRef(
        WF,
        excitations='SD',
        do_projected_operators=True,
        do_selfconsistent_operators=False,
        do_debugging=True,
    )

    print('Check if matrices fullfill their expected property:')
    assert (np.all(np.abs(LR_generic.M - LR_generic.M.T) < threshold)) == True
    assert (np.all(np.abs(LR_generic.Q - LR_generic.Q.T) < threshold)) == True
    assert (np.all(np.abs(LR_generic.W) < threshold)) == True

    # ST
    print('\nMethod: ST')
    LR_naive = LinearResponseUCCRef(
        WF,
        excitations='SD',
        do_projected_operators=False,
        do_selfconsistent_operators=False,
        do_statetransfer_operators=True,
    )
    LR_generic = LinearResponseUCCRef(
        WF,
        excitations='SD',
        do_projected_operators=False,
        do_selfconsistent_operators=False,
        do_statetransfer_operators=True,
        do_debugging=True,
    )

    print(
        'Check if implementation via work equation and generic are the same with a threshold of: ', threshold
    )
    assert (np.allclose(LR_naive.M, LR_generic.M, atol=threshold)) is True
    assert (np.allclose(LR_naive.Q, LR_generic.Q, atol=threshold)) is True
    assert (np.allclose(LR_naive.V, LR_generic.V, atol=threshold)) is True
    assert (np.allclose(LR_naive.W, LR_generic.W, atol=threshold)) is True

    print('Check if matrices fullfill their expected property:')
    assert (np.all(np.abs(LR_naive.M - LR_naive.M.T) < threshold)) == True
    assert (np.all(np.abs(LR_naive.Q - LR_naive.Q.T) < threshold)) == True
    assert (np.all(np.abs(LR_naive.W) < threshold)) == True

    # HST
    print('\nMethod: ST')
    LR_HST_naive = LinearResponseUCCRef(
        WF,
        excitations='SD',
        do_projected_operators=False,
        do_selfconsistent_operators=False,
        do_statetransfer_operators=False,
        do_hermitian_statetransfer_operators=True,
    )
    LR_HST_tracked = LinearResponseUCCRef(
        WF,
        excitations='SD',
        do_projected_operators=False,
        do_selfconsistent_operators=False,
        do_statetransfer_operators=False,
        do_hermitian_statetransfer_operators=True,
        track_hermitian_statetransfer=True,
    )

    print('Check if matrices fullfill their expected property:')
    assert (np.all(np.abs(LR_HST_naive.M - LR_HST_naive.M.T) < threshold)) == True
    assert (np.all(np.abs(LR_HST_naive.Q - LR_HST_naive.Q.T) < threshold)) == True

    print('Check if matrices fullfill their expected property:')
    assert (np.all(np.abs(LR_HST_tracked.M - LR_HST_tracked.M.T) < threshold)) == True
    assert (np.all(np.abs(LR_HST_tracked.Q - LR_HST_tracked.Q.T) < threshold)) == True

    print('Compare HST implementations:')
    assert (np.allclose(LR_HST_naive.M, LR_HST_tracked.M, atol=threshold)) is True
    assert (np.allclose(LR_HST_naive.Q, LR_HST_tracked.Q, atol=threshold)) is True
    assert (np.allclose(LR_HST_naive.V, LR_HST_tracked.V, atol=threshold)) is True
    assert (np.allclose(LR_HST_naive.W, LR_HST_tracked.W, atol=threshold)) is True


def test_LiH_allmethods_energies() -> None:
    """Test LiH all matrices and their properties for all LR methods."""

    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li   0.0           0.0  0.0;
        H   1.67  0.0  0.0;""",
        distance_unit='angstrom',
    )
    SQobj.set_basis_set('sto-3g')
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
    WF.run_ucc('SD', True)

    threshold = 10 ** (-5)

    # atSC
    print('\nMethod: at-SC')
    LR_naive = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=True,
        do_statetransfer_operators=False,
        do_debugging=False,
    )
    LR_naive.calc_excitation_energies()
    print(LR_naive.excitation_energies)

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
    assert (np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)) is True

    # atST
    print('\nMethod: at-ST')
    LR_naive = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=False,
        do_statetransfer_operators=True,
        do_debugging=False,
    )
    LR_naive.calc_excitation_energies()
    print(LR_naive.excitation_energies)

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
    assert (np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)) is True

    # naive
    print('\nMethod: naive')
    LR_naive = LinearResponseUCCRef(
        WF, excitations='SD', do_projected_operators=False, do_selfconsistent_operators=False
    )
    LR_naive.calc_excitation_energies()
    print(LR_naive.excitation_energies)

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
    assert (np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)) is True

    # proj: only generic implemented atm
    print('\nMethod: proj')
    LR_naive = LinearResponseUCCRef(
        WF,
        excitations='SD',
        do_projected_operators=True,
        do_selfconsistent_operators=False,
        do_debugging=True,
    )
    LR_naive.calc_excitation_energies()
    print(LR_naive.excitation_energies)

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
    assert (np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)) is True

    # SC
    print('\nMethod: SC')
    LR_naive = LinearResponseUCCRef(
        WF, excitations='SD', do_projected_operators=False, do_selfconsistent_operators=True
    )
    LR_naive.calc_excitation_energies()
    print(LR_naive.excitation_energies)

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
    assert (np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)) is True

    # ST
    print('\nMethod: ST')
    LR_naive = LinearResponseUCCRef(
        WF,
        excitations='SD',
        do_projected_operators=False,
        do_selfconsistent_operators=False,
        do_statetransfer_operators=True,
    )
    LR_naive.calc_excitation_energies()
    print(LR_naive.excitation_energies)

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
    assert (np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)) is True

    # HST
    print('\nMethod: ST')
    LR_HST_naive = LinearResponseUCCRef(
        WF,
        excitations='SD',
        do_projected_operators=False,
        do_selfconsistent_operators=False,
        do_statetransfer_operators=False,
        do_hermitian_statetransfer_operators=True,
    )
    LR_HST_naive.calc_excitation_energies()
    print(LR_HST_naive.excitation_energies)

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

    assert (np.allclose(LR_HST_naive.excitation_energies, solutions, atol=threshold)) is True

### END OLD TESTS

def test_LiH_atmethods_energies() -> None:
    """Test LiH at LR methods."""

    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li   0.0           0.0  0.0;
        H   1.67  0.0  0.0;""",
        distance_unit='angstrom',
    )
    SQobj.set_basis_set('sto-3g')
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
    WF.run_ucc('SD', True)

    threshold = 10 ** (-5)

    # atSC
    print('\nMethod: at-SC')
    LR_naive = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=True,
        do_statetransfer_operators=False,
        do_debugging=False,
    )
    LR_naive.calc_excitation_energies()
    print(LR_naive.excitation_energies)

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
    
    assert (np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)) is True

    # atST
    print('\nMethod: at-ST')
    LR_naive = LinearResponseUCC(
        WF,
        excitations='SD',
        do_selfconsistent_operators=False,
        do_statetransfer_operators=True,
        do_debugging=False,
    )
    LR_naive.calc_excitation_energies()
    print(LR_naive.excitation_energies)

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

    assert (np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)) is True

def test_LiH_allmethods_energies_fast() -> None:
    """Test LiH all matrices and their properties for all LR methods."""

    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li   0.0           0.0  0.0;
        H   1.67  0.0  0.0;""",
        distance_unit='angstrom',
    )
    SQobj.set_basis_set('sto-3g')
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
    WF.run_ucc('SD', True)

    threshold = 10 ** (-5)

    # naive
    print('\nMethod: naive')
    LR_naive = LinearResponseUCCFast(
        WF, excitations='SD', do_projected_operators=False, do_selfconsistent_operators=False
    )
    LR_naive.calc_excitation_energies()
    print(LR_naive.excitation_energies)

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
    assert (np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)) is True

    # proj:
    print('\nMethod: proj')
    LR_naive = LinearResponseUCCFast(
        WF,
        excitations='SD',
        do_projected_operators=True,
        do_selfconsistent_operators=False,
        # do_debugging=True,
    )
    LR_naive.calc_excitation_energies()
    print(LR_naive.excitation_energies)

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
    assert (np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)) is True

    # SC
    print('\nMethod: SC')
    LR_naive = LinearResponseUCCFast(
        WF, excitations='SD', do_projected_operators=False, do_selfconsistent_operators=True
    )
    LR_naive.calc_excitation_energies()
    print(LR_naive.excitation_energies)

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
    assert (np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)) is True

    # ST
    print('\nMethod: ST')
    LR_naive = LinearResponseUCCFast(
        WF,
        excitations='SD',
        do_projected_operators=False,
        do_selfconsistent_operators=False,
        do_statetransfer_operators=True,
    )
    LR_naive.calc_excitation_energies()
    print(LR_naive.excitation_energies)

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
    assert (np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)) is True

    # HST
    print('\nMethod: HST')
    LR_HST_naive = LinearResponseUCCFast(
        WF,
        excitations='SD',
        do_projected_operators=False,
        do_selfconsistent_operators=False,
        do_statetransfer_operators=False,
        do_hermitian_statetransfer_operators=True,
    )
    LR_HST_naive.calc_excitation_energies()
    print(LR_HST_naive.excitation_energies)

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

    print(LR_HST_naive.excitation_energies - solutions)
    assert (np.allclose(LR_HST_naive.excitation_energies, solutions, atol=threshold)) is True


def test_LiH_allmethods_matrices_fast() -> None:
    """Test LiH all matrices and their properties for all LR methods."""

    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li   0.0           0.0  0.0;
        H   1.67  0.0  0.0;""",
        distance_unit='angstrom',
    )
    SQobj.set_basis_set('sto-3g')
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
    WF.run_ucc('SD', True)

    threshold = 10 ** (-10)
    
    # naive
    print('\nMethod: naive')
    LR_naive = LinearResponseUCCFast(
        WF, excitations='SD', do_projected_operators=False, do_selfconsistent_operators=False
    )
    LR_generic = LinearResponseUCCFast(
        WF,
        excitations='SD',
        do_projected_operators=False,
        do_selfconsistent_operators=False,
        do_debugging=True,
    )

    print(
        'Check if implementation via work equation and generic are the same with a threshold of: ', threshold
    )
    assert (np.allclose(LR_naive.M, LR_generic.M, atol=threshold)) is True
    assert (np.allclose(LR_naive.Q, LR_generic.Q, atol=threshold)) is True
    assert (np.allclose(LR_naive.V, LR_generic.V, atol=threshold)) is True
    assert (np.allclose(LR_naive.W, LR_generic.W, atol=threshold)) is True

    print('Check if matrices fullfill their expected property:')
    assert (np.all(np.abs(LR_naive.M - LR_naive.M.T) < threshold)) == True
    assert (np.all(np.abs(LR_naive.Q - LR_naive.Q.T) < threshold)) == True
    assert (np.all(np.abs(LR_naive.W) < threshold)) == True

    print('Check if generic matrices fullfill their expected property:')
    assert (np.all(np.abs(LR_generic.M - LR_generic.M.T) < threshold)) == True
    assert (np.all(np.abs(LR_generic.Q - LR_generic.Q.T) < threshold)) == True
    assert (np.all(np.abs(LR_generic.W) < threshold)) == True

    # SC
    print('\nMethod: SC')
    LR_naive = LinearResponseUCCFast(
        WF, excitations='SD', do_projected_operators=False, do_selfconsistent_operators=True
    )
    LR_generic = LinearResponseUCCFast(
        WF,
        excitations='SD',
        do_projected_operators=False,
        do_selfconsistent_operators=True,
        do_debugging=True,
    )

    print(
        'Check if implementation via work equation and generic are the same with a threshold of: ', threshold
    )
    assert (np.allclose(LR_naive.M, LR_generic.M, atol=threshold)) is True
    assert (np.allclose(LR_naive.Q, LR_generic.Q, atol=threshold)) is True
    assert (np.allclose(LR_naive.V, LR_generic.V, atol=threshold)) is True
    assert (np.allclose(LR_naive.W, LR_generic.W, atol=threshold)) is True

    print('Check if matrices fullfill their expected property:')
    assert (np.all(np.abs(LR_naive.M - LR_naive.M.T) < threshold)) == True
    assert (np.all(np.abs(LR_naive.Q - LR_naive.Q.T) < threshold)) == True
    assert (np.all(np.abs(LR_naive.W) < threshold)) == True
    
    # projected:
    threshold = 10 ** (
        -6
    )  # Change threshold to gradient since in projected work equations, we remove terms that are just the gradient. Thus, the error is minimum the gradient's deviation from zero
    print('\nMethod: projected')
    LR_naive = LinearResponseUCCFast(
        WF,
        excitations='SD',
        do_projected_operators=True,
        do_selfconsistent_operators=False,
    )
    LR_generic = LinearResponseUCCFast(
        WF,
        excitations='SD',
        do_projected_operators=True,
        do_selfconsistent_operators=False,
        do_debugging=True,
    )

    print(
        'Check if implementation via work equation and generic are the same with a threshold of: ', threshold
    )

    assert (np.allclose(LR_naive.M, LR_generic.M, atol=threshold)) is True
    assert (np.allclose(LR_naive.Q, LR_generic.Q, atol=threshold)) is True
    assert (np.allclose(LR_naive.V, LR_generic.V, atol=threshold)) is True
    assert (np.allclose(LR_naive.W, LR_generic.W, atol=threshold)) is True

    threshold = 10 ** (-10)
    print('Check if matrices fullfill their expected property:')
    assert (np.all(np.abs(LR_naive.M - LR_naive.M.T) < threshold)) == True
    assert (np.all(np.abs(LR_naive.Q - LR_naive.Q.T) < threshold)) == True
    assert (np.all(np.abs(LR_naive.W) < threshold)) == True

    # ST
    print('\nMethod: ST')
    LR_naive = LinearResponseUCCFast(
        WF,
        excitations='SD',
        do_projected_operators=False,
        do_selfconsistent_operators=False,
        do_statetransfer_operators=True,
    )
    LR_generic = LinearResponseUCCFast(
        WF,
        excitations='SD',
        do_projected_operators=False,
        do_selfconsistent_operators=False,
        do_statetransfer_operators=True,
        do_debugging=True,
    )

    print(
        'Check if implementation via work equation and generic are the same with a threshold of: ', threshold
    )
    assert (np.allclose(LR_naive.M, LR_generic.M, atol=threshold)) is True
    assert (np.allclose(LR_naive.Q, LR_generic.Q, atol=threshold)) is True
    assert (np.allclose(LR_naive.V, LR_generic.V, atol=threshold)) is True
    assert (np.allclose(LR_naive.W, LR_generic.W, atol=threshold)) is True

    print('Check if matrices fullfill their expected property:')
    assert (np.all(np.abs(LR_naive.M - LR_naive.M.T) < threshold)) == True
    assert (np.all(np.abs(LR_naive.Q - LR_naive.Q.T) < threshold)) == True
    assert (np.all(np.abs(LR_naive.W) < threshold)) == True

    # HST
    print('\nMethod: ST')
    LR_HST_naive = LinearResponseUCCFast(
        WF,
        excitations='SD',
        do_projected_operators=False,
        do_selfconsistent_operators=False,
        do_statetransfer_operators=False,
        do_hermitian_statetransfer_operators=True,
    )
    LR_HST_tracked = LinearResponseUCCFast(
        WF,
        excitations='SD',
        do_projected_operators=False,
        do_selfconsistent_operators=False,
        do_statetransfer_operators=False,
        do_hermitian_statetransfer_operators=True,
        track_hermitian_statetransfer=True,
    )

    print('Check if matrices fullfill their expected property:')
    assert (np.all(np.abs(LR_HST_naive.M - LR_HST_naive.M.T) < threshold)) == True
    assert (np.all(np.abs(LR_HST_naive.Q - LR_HST_naive.Q.T) < threshold)) == True

    print('Check if matrices fullfill their expected property:')
    assert (np.all(np.abs(LR_HST_tracked.M - LR_HST_tracked.M.T) < threshold)) == True
    assert (np.all(np.abs(LR_HST_tracked.Q - LR_HST_tracked.Q.T) < threshold)) == True

    print('Compare HST implementations:')
    assert (np.allclose(LR_HST_naive.M, LR_HST_tracked.M, atol=threshold)) is True
    assert (np.allclose(LR_HST_naive.Q, LR_HST_tracked.Q, atol=threshold)) is True
    assert (np.allclose(LR_HST_naive.V, LR_HST_tracked.V, atol=threshold)) is True
    assert (np.allclose(LR_HST_naive.W, LR_HST_tracked.W, atol=threshold)) is True


def test_LiH_allproj_energies_fast() -> None:
    """Test LiH for all-proj LR method."""

    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li   0.0           0.0  0.0;
        H   1.67  0.0  0.0;""",
        distance_unit='angstrom',
    )
    SQobj.set_basis_set('sto-3g')
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
    WF.run_ucc('SD', True)

    threshold = 10 ** (-5)

    print('\nMethod: all-projected')
    LR_naive = LinearResponseUCCFast(
        WF,
        excitations='SD',
        do_projected_operators=False,
        do_selfconsistent_operators=False,
        do_all_projected_operators=True,
    )
    LR_naive.calc_excitation_energies()
    print(LR_naive.excitation_energies)

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
    assert (np.allclose(LR_naive.excitation_energies, solutions, atol=threshold)) is True
