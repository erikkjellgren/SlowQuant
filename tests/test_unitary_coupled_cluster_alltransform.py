import numpy as np

import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.linear_response_alltransform import (
    LinearResponseUCC,
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
    assert abs(LR.excitation_energies[0] - 1.0157) < 10**-3
    assert abs(LR.excitation_energies[1] - 1.7195) < 10**-3
    assert abs(LR.get_excited_state_overlap(0) - 0.0) < 10**-3
    assert abs(LR.get_excited_state_overlap(1) - 0.0) < 10**-3
    assert abs(abs(LR.get_transition_dipole(0, dipole_integrals)[2]) - 1.1441) < 10**-3
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
    assert abs(LR.excitation_energies[0] - 1.0157) < 10**-3
    assert abs(LR.excitation_energies[1] - 1.7195) < 10**-3
    assert abs(LR.get_excited_state_overlap(0) - 0.0) < 10**-3
    assert abs(LR.get_excited_state_overlap(1) - 0.0) < 10**-3
    assert abs(abs(LR.get_transition_dipole(0, dipole_integrals)[2]) - 1.1441) < 10**-3
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
