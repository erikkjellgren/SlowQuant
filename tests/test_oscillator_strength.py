import copy

import numpy as np

import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.linear_response_fast import LinearResponseUCC
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


def test_H2_631g_naive():
    """
    Test of oscialltor strength for naive LR with working equations
    """
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0   0.0  0.0;
            H  0.74  0.0  0.0;""",
        distance_unit='angstrom',
    )
    SQobj.set_basis_set('6-31G')
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
    )
    WF.run_ucc('SD', True)

    # Linear Response
    LR = LinearResponseUCC(WF, excitations='SD', do_selfconsistent_operators=False)  # naive
    LR_nn = copy.deepcopy(LR)
    LR.calc_excitation_energies()

    thresh = 10**-4

    # Check excitation energies
    print('Check excitation energies')
    assert abs(LR.excitation_energies[0] - 0.574413) < thresh
    assert abs(LR.excitation_energies[1] - 1.043177) < thresh
    assert abs(LR.excitation_energies[2] - 1.139481) < thresh
    assert abs(LR.excitation_energies[3] - 1.365960) < thresh
    assert abs(LR.excitation_energies[4] - 1.831196) < thresh
    assert abs(LR.excitation_energies[5] - 2.581273) < thresh

    # Calculate dipole integrals
    dipole_integrals = (
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    )

    # Get oscillator strength for each excited state
    print('Check oscillator strength')
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.6338) < thresh
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.0) < thresh
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.0) < thresh
    assert abs(LR.get_oscillator_strength(3, dipole_integrals) - 0.0311) < thresh
    assert abs(LR.get_oscillator_strength(4, dipole_integrals) - 0.0421) < thresh
    assert abs(LR.get_oscillator_strength(5, dipole_integrals) - 0.0) < thresh

    # Compare generic and working equation implementation
    print('Difference generic vs. working equations')
    assert (
        abs(
            LR.get_oscillator_strength(0, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(0, dipole_integrals)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(1, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(1, dipole_integrals)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(2, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(2, dipole_integrals)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(3, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(3, dipole_integrals)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(4, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(4, dipole_integrals)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(5, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(5, dipole_integrals)
        )
        < thresh
    )

    # Check working equations of excited state norm
    LR_nn.calc_excitation_energies(do_working_equations=True)
    print('Norm via working equations')
    print('Difference generic vs. working equations')
    assert (
        abs(
            LR.get_oscillator_strength(0, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(0, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(1, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(1, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(2, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(2, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(3, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(3, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(4, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(4, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(5, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(5, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )

    print('Compare normed response vectors')
    assert np.allclose(LR.normed_response_vectors, LR_nn.normed_response_vectors, atol=10**-20) is True


def test_LiH_sto3g_naive():
    """
    Test LiH Sto-3G naive LR oscialltor strength
    """
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0  0.0  0.0;
            H 1.671707274 0.0 0.0;""",
        distance_unit='angstrom',
    )
    SQobj.set_basis_set('sto-3g')
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
    )
    WF.run_ucc('SD', True)

    # Linear Response
    LR = LinearResponseUCC(WF, excitations='SD', do_selfconsistent_operators=False)  # naive
    LR_nn = copy.deepcopy(LR)
    LR.calc_excitation_energies()

    thresh = 10**-4

    # Check excitation energies
    print('Check excitation energies')
    assert abs(LR.excitation_energies[0] - 0.129471) < thresh
    assert abs(LR.excitation_energies[1] - 0.178744) < thresh
    assert abs(LR.excitation_energies[2] - 0.178744) < thresh
    assert abs(LR.excitation_energies[3] - 0.604674) < thresh
    assert abs(LR.excitation_energies[4] - 0.646694) < thresh
    assert abs(LR.excitation_energies[5] - 0.740616) < thresh
    assert abs(LR.excitation_energies[6] - 0.740616) < thresh
    assert abs(LR.excitation_energies[7] - 1.002882) < thresh
    assert abs(LR.excitation_energies[8] - 2.074820) < thresh
    assert abs(LR.excitation_energies[9] - 2.137192) < thresh
    assert abs(LR.excitation_energies[10] - 2.137192) < thresh
    assert abs(LR.excitation_energies[11] - 2.455124) < thresh
    assert abs(LR.excitation_energies[12] - 2.9543838) < thresh

    # Calculate dipole integrals
    dipole_integrals = (
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    )

    # Get oscillator strength for each excited state
    print('Check oscillator strength')
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.049952) < thresh
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.241200) < thresh
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.241200) < thresh
    assert abs(LR.get_oscillator_strength(3, dipole_integrals) - 0.1580497) < thresh
    assert abs(LR.get_oscillator_strength(4, dipole_integrals) - 0.166598) < thresh
    assert abs(LR.get_oscillator_strength(5, dipole_integrals) - 0.010376) < thresh
    assert abs(LR.get_oscillator_strength(6, dipole_integrals) - 0.010376) < thresh
    assert abs(LR.get_oscillator_strength(7, dipole_integrals) - 0.006250) < thresh
    assert abs(LR.get_oscillator_strength(8, dipole_integrals) - 0.062374) < thresh
    assert abs(LR.get_oscillator_strength(9, dipole_integrals) - 0.128854) < thresh
    assert abs(LR.get_oscillator_strength(10, dipole_integrals) - 0.128854) < thresh
    assert abs(LR.get_oscillator_strength(11, dipole_integrals) - 0.046008) < thresh
    assert abs(LR.get_oscillator_strength(12, dipole_integrals) - 0.003907) < thresh

    # Compare generic and working equation implementation
    print('Difference generic vs. working equations')
    assert (
        abs(
            LR.get_oscillator_strength(0, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(0, dipole_integrals)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(1, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(1, dipole_integrals)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(2, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(2, dipole_integrals)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(3, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(3, dipole_integrals)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(4, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(4, dipole_integrals)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(5, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(5, dipole_integrals)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(6, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(6, dipole_integrals)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(7, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(7, dipole_integrals)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(8, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(8, dipole_integrals)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(9, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(9, dipole_integrals)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(10, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(10, dipole_integrals)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(11, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(11, dipole_integrals)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(12, dipole_integrals, do_working_equations=True)
            - LR.get_oscillator_strength(12, dipole_integrals)
        )
        < thresh
    )

    # Check working equations of excited state norm
    LR_nn.calc_excitation_energies(do_working_equations=True)
    print('Norm via working equations')
    print('Difference generic vs. working equations')
    assert (
        abs(
            LR.get_oscillator_strength(0, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(0, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(1, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(1, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(2, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(2, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(3, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(3, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(4, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(4, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(5, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(5, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(6, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(6, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(7, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(7, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(8, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(8, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(9, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(9, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(10, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(10, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(11, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(11, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )
    assert (
        abs(
            LR.get_oscillator_strength(12, dipole_integrals, do_working_equations=True)
            - LR_nn.get_oscillator_strength(12, dipole_integrals, do_working_equations=True)
        )
        < thresh
    )

    print('Compare normed response vectors')
    assert np.allclose(LR.normed_response_vectors, LR_nn.normed_response_vectors, atol=10**-20) is True
