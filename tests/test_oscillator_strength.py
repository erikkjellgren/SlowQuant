import numpy as np

from SlowQuant.slowquant.unitary_coupled_cluster.linear_response import generalized_allstatetransfer, generalized_naive, generalized_projected
import slowquant.SlowQuant as sq
from SlowQuant.slowquant.unitary_coupled_cluster.linear_response import (
    generalized_statetransfer,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


def test_H2_631g_naive():
    """Test of oscialltor strength for naive LR with working equations."""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0   0.0  0.0;
            H  0.74  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("6-31G")
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
    LR = generalized_naive.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies()
    genericLR = generalized_naive.LinearResponse(WF, excitations="SD")
    genericLR.calc_excitation_energies()

    thresh = 10**-4

    # Check excitation energies
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
    osc_strengths = LR.get_oscillator_strength(dipole_integrals)
    assert abs(osc_strengths[0] - 0.6338) < thresh
    assert abs(osc_strengths[1] - 0.0) < thresh
    assert abs(osc_strengths[2] - 0.0) < thresh
    assert abs(osc_strengths[3] - 0.0311) < thresh
    assert abs(osc_strengths[4] - 0.0421) < thresh
    assert abs(osc_strengths[5] - 0.0) < thresh

    # Compare generic and working equation implementation
    osc_strengths_generic = genericLR.get_oscillator_strength(dipole_integrals)
    assert np.allclose(osc_strengths, osc_strengths_generic, atol=thresh)


def test_LiH_sto3g_naive():
    """Test LiH Sto-3G naive LR oscialltor strength."""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0  0.0  0.0;
            H 1.671707274 0.0 0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    # oo-UCCSD
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
    LR = generalized_naive.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies()
    genericLR = generalized_naive.LinearResponse(WF, excitations="SD")
    genericLR.calc_excitation_energies()

    thresh = 10**-4

    # Check excitation energies
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
    osc_strengths = LR.get_oscillator_strength(dipole_integrals)
    assert abs(osc_strengths[0] - 0.049952) < thresh
    assert abs(osc_strengths[1] - 0.241200) < thresh
    assert abs(osc_strengths[2] - 0.241200) < thresh
    assert abs(osc_strengths[3] - 0.1580497) < thresh
    assert abs(osc_strengths[4] - 0.166598) < thresh
    assert abs(osc_strengths[5] - 0.010376) < thresh
    assert abs(osc_strengths[6] - 0.010376) < thresh
    assert abs(osc_strengths[7] - 0.006250) < thresh
    assert abs(osc_strengths[8] - 0.062374) < thresh
    assert abs(osc_strengths[9] - 0.128854) < thresh
    assert abs(osc_strengths[10] - 0.128854) < thresh
    assert abs(osc_strengths[11] - 0.046008) < thresh
    assert abs(osc_strengths[12] - 0.003907) < thresh

    # Compare generic and working equation implementation
    osc_strengths_generic = genericLR.get_oscillator_strength(dipole_integrals)
    assert np.allclose(osc_strengths, osc_strengths_generic, atol=thresh)


def test_H2_631g_projLR():
    """Test of oscialltor strength for projected LR with working equations."""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0   0.0  0.0;
            H  0.74  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("6-31G")
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
    LR = generalized_projected.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies()

    thresh = 10**-4

    # Check excitation energies
    print("Check excitation energies")
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
    osc_strengths = LR.get_oscillator_strength(dipole_integrals)
    assert abs(osc_strengths[0] - 0.6338231953094923) < thresh
    assert abs(osc_strengths[1] - 0.0) < thresh
    assert abs(osc_strengths[2] - 0.0) < thresh
    assert abs(osc_strengths[3] - 0.031089763125846485) < thresh
    assert abs(osc_strengths[4] - 0.04212982876590235) < thresh
    assert abs(osc_strengths[5] - 0.0) < thresh


def test_LiH_sto3g_proj():
    """Test LiH Sto-3G projected LR oscialltor strength."""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0  0.0  0.0;
            H 1.671707274 0.0 0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    # HF
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    # oo-UCCSD
    WF = WaveFunctionUCC(
        SQobj.molecule.number_electrons,
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        "SD",
    )
    WF.run_wf_optimization_1step("SLSQP", True)

    LR = generalized_projected.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies()

    thresh = 10**-4

    # Check excitation energies
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
    osc_strengths = LR.get_oscillator_strength(dipole_integrals)
    assert abs(osc_strengths[0] - 0.049919878841153974) < thresh
    assert abs(osc_strengths[1] - 0.24118483531266577) < thresh
    assert abs(osc_strengths[2] - 0.24118483534591598) < thresh
    assert abs(osc_strengths[3] - 0.15804974985474457) < thresh
    assert abs(osc_strengths[4] - 0.16653189079808411) < thresh
    assert abs(osc_strengths[5] - 0.010379091370812886) < thresh
    assert abs(osc_strengths[6] - 0.010379091373763447) < thresh
    assert abs(osc_strengths[7] - 0.006256710161922168) < thresh
    assert abs(osc_strengths[8] - 0.062488043049451776) < thresh
    assert abs(osc_strengths[9] - 0.12886225822034553) < thresh
    assert abs(osc_strengths[10] - 0.12886225822019629) < thresh
    assert abs(osc_strengths[11] - 0.046007031170702296) < thresh
    assert abs(osc_strengths[12] - 0.0039034101562325234) < thresh


def test_H2_631g_STLR():
    """Test of oscialltor strength for projected LR with working equations."""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0   0.0  0.0;
            H  0.74  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("6-31G")
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
    LR = generalized_statetransfer.LinearResponse(
        WF,
        excitations="SD",
    )
    LR.calc_excitation_energies()

    thresh = 10**-4

    # Check excitation energies
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
    osc_strengths = LR.get_oscillator_strength(dipole_integrals)
    assert abs(osc_strengths[0] - 0.6338231953094933) < thresh
    assert abs(osc_strengths[1] - 0.0) < thresh
    assert abs(osc_strengths[2] - 0.0) < thresh
    assert abs(osc_strengths[3] - 0.03108976312584539) < thresh
    assert abs(osc_strengths[4] - 0.042129828765903814) < thresh
    assert abs(osc_strengths[5] - 0.0) < thresh


def test_LiH_sto3g_st():
    """Test LiH Sto-3G projected LR oscialltor strength."""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0  0.0  0.0;
            H 1.671707274 0.0 0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
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
    LR = generalized_statetransfer.LinearResponse(
        WF,
        excitations="SD",
    )
    LR.calc_excitation_energies()

    thresh = 10**-4

    # Check excitation energies
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
    osc_strengths = LR.get_oscillator_strength(dipole_integrals)
    assert abs(osc_strengths[0] - 0.0499198684945157) < thresh
    assert abs(osc_strengths[1] - 0.2411848353126639) < thresh
    assert abs(osc_strengths[2] - 0.24118483534591595) < thresh
    assert abs(osc_strengths[3] - 0.15805070049553024) < thresh
    assert abs(osc_strengths[4] - 0.16653094112270908) < thresh
    assert abs(osc_strengths[5] - 0.010379091370809963) < thresh
    assert abs(osc_strengths[6] - 0.010379091373763017) < thresh
    assert abs(osc_strengths[7] - 0.0062567030068973305) < thresh
    assert abs(osc_strengths[8] - 0.06248802645793188) < thresh
    assert abs(osc_strengths[9] - 0.12886225822029365) < thresh
    assert abs(osc_strengths[10] - 0.12886225822018843) < thresh
    assert abs(osc_strengths[11] - 0.04600702378157588) < thresh
    assert abs(osc_strengths[12] - 0.0039034084421841943) < thresh


def test_H2_631g_allST():
    """Test of oscialltor strength for projected LR with working equations."""
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0   0.0  0.0;
            H  0.74  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("6-31G")
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
    LR = generalized_allstatetransfer.LinearResponse(
        WF,
        excitations="SD",
    )
    LR.calc_excitation_energies()

    thresh = 10**-4

    # Check excitation energies
    assert abs(LR.excitation_energies[0] - 0.57773553) < thresh
    assert abs(LR.excitation_energies[1] - 1.05253656) < thresh
    assert abs(LR.excitation_energies[2] - 1.63445659) < thresh
    assert abs(LR.excitation_energies[3] - 1.64921366) < thresh

    # Calculate dipole integrals
    dipole_integrals = (
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    )

    # Get oscillator strength for each excited state
    osc_strengths = LR.get_oscillator_strength(dipole_integrals)
    assert abs(osc_strengths[0] - 0.650294311) < thresh
    assert abs(osc_strengths[1] - 0.0) < thresh
    assert abs(osc_strengths[2] - 6.23019972e-02) < thresh
    assert abs(osc_strengths[3] - 0.0) < thresh


def test_LiH_sto3g_allST():
    """Test LiH STO-3G all-statetransfer LR oscialltor strength."""
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
    LR = generalized_allstatetransfer.LinearResponse(
        WF,
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
