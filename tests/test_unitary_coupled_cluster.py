import numpy as np

import slowquant.SlowQuant as sq
import slowquant.unitary_coupled_cluster.linear_response.generic as genericLR
import slowquant.unitary_coupled_cluster.linear_response.naive as naiveLR
import slowquant.unitary_coupled_cluster.linear_response.selfconsistent as selfconsistentLR
from slowquant.unitary_coupled_cluster.operator_pauli import (
    expectation_value_pauli,
    hamiltonian_pauli,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


def test_heh_sto3g_hf() -> None:
    """Test restricted Hartree-Fock through the unitary coupled cluster module."""
    A = sq.SlowQuant()
    A.set_molecule(
        """H  1.4  0.0  0.0;
           He 0.0  0.0  0.0;""",
        distance_unit="bohr",
        molecular_charge=1,
    )
    A.set_basis_set("sto-3g")
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    Lambda_S, L_S = np.linalg.eigh(A.integral.overlap_matrix)
    S_sqrt = np.dot(np.dot(L_S, np.diag(Lambda_S ** (-1 / 2))), np.transpose(L_S))
    WF = WaveFunctionUCC(A.molecule.number_bf * 2, A.molecule.number_electrons, (2, 1), S_sqrt, h_core, g_eri)
    WF.run_ucc("S", True)
    assert abs(WF.energy_elec - (-4.262632309847)) < 10**-8


def test_lih_sto3g_hf() -> None:
    """Test restricted Hartree-Fock through the unitary coupled cluster module."""
    A = sq.SlowQuant()
    A.set_molecule(
        """H  0.0  0.0  0.0;
           Li 3.0  0.0  0.0;""",
        distance_unit="bohr",
    )
    A.set_basis_set("sto-3g")
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    Lambda_S, L_S = np.linalg.eigh(A.integral.overlap_matrix)
    S_sqrt = np.dot(np.dot(L_S, np.diag(Lambda_S ** (-1 / 2))), np.transpose(L_S))
    WF = WaveFunctionUCC(A.molecule.number_bf * 2, A.molecule.number_electrons, (2, 1), S_sqrt, h_core, g_eri)
    WF.run_ucc("S", True)
    assert abs(WF.energy_elec - (-8.862246324082243)) < 10**-8


def test_heh_sto3g_uccs() -> None:
    """Test restricted UCCS."""
    A = sq.SlowQuant()
    A.set_molecule(
        """H  1.4  0.0  0.0;
           He 0.0  0.0  0.0;""",
        distance_unit="bohr",
        molecular_charge=1,
    )
    A.set_basis_set("sto-3g")
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    Lambda_S, L_S = np.linalg.eigh(A.integral.overlap_matrix)
    S_sqrt = np.dot(np.dot(L_S, np.diag(Lambda_S ** (-1 / 2))), np.transpose(L_S))
    WF = WaveFunctionUCC(A.molecule.number_bf * 2, A.molecule.number_electrons, (2, 2), S_sqrt, h_core, g_eri)
    WF.run_ucc("S")
    assert abs(WF.energy_elec - (-4.262632309847)) < 10**-8


def test_h10_sto3g_uccsd() -> None:
    """Test UCCSD(2,2).
    Test made after bug found where more than two inactive orbitals would not work.
    """
    A = sq.SlowQuant()
    A.set_molecule(
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
    A.set_basis_set("sto-3g")
    A.init_hartree_fock()
    A.hartree_fock.run_restricted_hartree_fock()
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        A.molecule.number_bf * 2,
        A.molecule.number_electrons,
        (2, 2),
        A.hartree_fock.mo_coeff,
        h_core,
        g_eri,
    )
    WF.run_ucc("SD", False)
    assert abs(WF.energy_elec - (-18.839645894737956)) < 10**-8


def test_h2_431g_oouccsd() -> None:
    """Test OO-UCCSD(2,2)."""
    A = sq.SlowQuant()
    A.set_molecule(
        """H  1.4  0.0  0.0;
           H  0.0  0.0  0.0;""",
        distance_unit="bohr",
    )
    A.set_basis_set("4-31G")
    A.init_hartree_fock()
    A.hartree_fock.run_restricted_hartree_fock()
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        A.molecule.number_bf * 2,
        A.molecule.number_electrons,
        (2, 2),
        A.hartree_fock.mo_coeff,
        h_core,
        g_eri,
    )
    WF.run_ucc("SD", True)
    assert abs(WF.energy_elec - (-1.860533598715)) < 10**-8


def test_h2_431g_oouccd() -> None:
    """Test OO-UCCD(2,2)."""
    A = sq.SlowQuant()
    A.set_molecule(
        """H  1.4  0.0  0.0;
           H  0.0  0.0  0.0;""",
        distance_unit="bohr",
    )
    A.set_basis_set("4-31G")
    A.init_hartree_fock()
    A.hartree_fock.run_restricted_hartree_fock()
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        A.molecule.number_bf * 2,
        A.molecule.number_electrons,
        (2, 2),
        A.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        include_active_kappa=True,
    )
    WF.run_ucc("D", True)
    assert abs(WF.energy_elec - (-1.860533598715)) < 10**-8


def test_h2_431g_uccsd() -> None:
    """Test UCCSD(2,2)."""
    A = sq.SlowQuant()
    A.set_molecule(
        """H  1.4  0.0  0.0;
           H  0.0  0.0  0.0;""",
        distance_unit="bohr",
    )
    A.set_basis_set("4-31G")
    A.init_hartree_fock()
    A.hartree_fock.run_restricted_hartree_fock()
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        A.molecule.number_bf * 2,
        A.molecule.number_electrons,
        (2, 2),
        A.hartree_fock.mo_coeff,
        h_core,
        g_eri,
    )
    WF.run_ucc("SD", False)
    assert abs(WF.energy_elec - (-1.8466860579618674)) < 10**-8


def test_h4_sto3g_oouccsd() -> None:
    """Test OO-UCCSD(2,2)."""
    A = sq.SlowQuant()
    A.set_molecule(
        """H  0.0  0.0  0.0;
           H  1.4  0.0  0.0;
           H  2.8  0.0  0.0;
           H  4.2  0.0  0.0;""",
        distance_unit="bohr",
    )
    A.set_basis_set("sto-3g")
    A.init_hartree_fock()
    A.hartree_fock.run_restricted_hartree_fock()
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        A.molecule.number_bf * 2,
        A.molecule.number_electrons,
        (2, 2),
        A.hartree_fock.mo_coeff,
        h_core,
        g_eri,
    )
    WF.run_ucc("SD", True)
    assert abs(WF.energy_elec - (-5.211066791547)) < 10**-8
    # Test sparse matrix also works
    H = hamiltonian_pauli(h_core, g_eri, WF.c_trans, WF.num_spin_orbs, WF.num_elec)
    assert (
        abs(WF.energy_elec - expectation_value_pauli(WF.state_vector, H, WF.state_vector, use_csr=0))
        < 10**-8
    )


def test_h4_sto3g_oouccd() -> None:
    """Test OO-UCCD(2,2)."""
    A = sq.SlowQuant()
    A.set_molecule(
        """H  0.0  0.0  0.0;
           H  1.4  0.0  0.0;
           H  2.8  0.0  0.0;
           H  4.2  0.0  0.0;""",
        distance_unit="bohr",
    )
    A.set_basis_set("sto-3g")
    A.init_hartree_fock()
    A.hartree_fock.run_restricted_hartree_fock()
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        A.molecule.number_bf * 2,
        A.molecule.number_electrons,
        (2, 2),
        A.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        include_active_kappa=True,
    )
    WF.run_ucc("D", True)
    assert abs(WF.energy_elec - (-5.211066791547)) < 10**-8
    # Test sparse matrix also works
    H = hamiltonian_pauli(h_core, g_eri, WF.c_trans, WF.num_spin_orbs, WF.num_elec)
    assert (
        abs(WF.energy_elec - expectation_value_pauli(WF.state_vector, H, WF.state_vector, use_csr=0))
        < 10**-8
    )


def test_h2_sto3g_uccsd_lr() -> None:
    """Test Linear Response for uccsd."""
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
    dipole_integrals = (
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    )
    WF.run_ucc("SD", False)
    LR = genericLR.LinearResponseUCC(WF, excitations="SD", operator_type="selfconsistent")
    LR.calc_excitation_energies()
    assert abs(LR.excitation_energies[0] - 1.015738) < 10**-4
    assert abs(LR.excitation_energies[1] - 1.719504) < 10**-4
    assert abs(abs(LR.get_transition_dipole(0, dipole_integrals)[2]) - 1.1440534325680685) < 10**-4
    assert abs(LR.get_transition_dipole(1, dipole_integrals)[2] - 0.0) < 10**-4


def test_h4_sto3g_uccdq() -> None:
    """Test UCCDQ(4,4).
    For this particular system only D and Q contributes to the energy.
    I think :))))
    """
    A = sq.SlowQuant()
    A.set_molecule(
        """H  0.0  0.0  0.0;
           H  0.0  1.5  0.0;
           H  1.8  0.0  0.0;
           H  1.8  1.5  0.0;""",
        distance_unit="angstrom",
    )
    A.set_basis_set("sto-3g")
    A.init_hartree_fock()
    A.hartree_fock.run_restricted_hartree_fock()
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        A.molecule.number_bf * 2,
        A.molecule.number_electrons,
        (4, 4),
        A.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        include_active_kappa=True,
    )
    WF.run_ucc("DQ", False)
    assert abs(WF.energy_elec + A.molecule.nuclear_repulsion - (-1.968914822185857)) < 10**-7


def test_h2_631g_hf_lr() -> None:
    """Test Linear Response for OO-uccsd(2,2)."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0   0.0  0.0;
           H  0.74  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("6-31G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (2, 1),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
    )
    WF.run_ucc("SD", True)
    LR = genericLR.LinearResponseUCC(WF, excitations="SD", operator_type="selfconsistent")
    LR.calc_excitation_energies()
    dipole_integrals = (
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    )
    assert abs(LR.excitation_energies[0] - 0.551961) < 10**-5
    assert abs(LR.excitation_energies[1] - 1.051638) < 10**-5
    assert abs(LR.excitation_energies[2] - 1.603563) < 10**-5
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.650948) < 10**-3
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.063496) < 10**-3


def test_h2_631g_oouccsd_lr() -> None:
    """Test Linear Response for OO-uccsd(2,2)."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0   0.0  0.0;
           H  0.74  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("6-31G")
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
    LR = genericLR.LinearResponseUCC(WF, excitations="SD", operator_type="selfconsistent")
    LR.calc_excitation_energies()
    dipole_integrals = (
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    )
    assert abs(LR.excitation_energies[0] - 0.574413) < 10**-5
    assert abs(LR.excitation_energies[1] - 1.043177) < 10**-5
    assert abs(LR.excitation_energies[2] - 1.139482) < 10**-5
    assert abs(LR.excitation_energies[3] - 1.365962) < 10**-5
    assert abs(LR.excitation_energies[4] - 1.831197) < 10**-5
    assert abs(LR.excitation_energies[5] - 2.581279) < 10**-5
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.633823) < 10**-3
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(3, dipole_integrals) - 0.031090) < 10**-3
    assert abs(LR.get_oscillator_strength(4, dipole_integrals) - 0.042130) < 10**-3
    assert abs(LR.get_oscillator_strength(5, dipole_integrals) - 0.0) < 10**-3


def test_h4_sto3g_uccsd_lr_naive() -> None:  # pylint: disable=R0915
    """Test Linear Response for uccsd(4,4)."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0  0.0  0.0;
           H  1.8  0.0  0.0;
           H  0.0  1.5  0.0;
           H  1.8  1.5  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        (4, 4),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
    )
    WF.run_ucc("SD", False)
    LR = genericLR.LinearResponseUCC(WF, excitations="SD", operator_type="naive")
    LR.calc_excitation_energies()
    dipole_integrals = (
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    )
    assert abs(LR.excitation_energies[0] - 0.162961) < 10**-5
    assert abs(LR.excitation_energies[1] - 0.418771) < 10**-5
    assert abs(LR.excitation_energies[2] - 0.550513) < 10**-5
    assert abs(LR.excitation_energies[3] - 0.585337) < 10**-5
    assert abs(LR.excitation_energies[4] - 0.600209) < 10**-5
    assert abs(LR.excitation_energies[5] - 0.602964) < 10**-5
    assert abs(LR.excitation_energies[6] - 0.680440) < 10**-5
    assert abs(LR.excitation_energies[7] - 0.705532) < 10**-5
    assert abs(LR.excitation_energies[8] - 0.805980) < 10**-5
    assert abs(LR.excitation_energies[9] - 0.843321) < 10**-5
    assert abs(LR.excitation_energies[10] - 0.923462) < 10**-5
    assert abs(LR.excitation_energies[11] - 1.189881) < 10**-5
    assert abs(LR.excitation_energies[12] - 1.512350) < 10**-5
    assert abs(LR.excitation_energies[13] - 1.515402) < 10**-5
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.026095) < 10**-3
    assert abs(LR.get_oscillator_strength(3, dipole_integrals) - 0.000298) < 10**-3
    assert abs(LR.get_oscillator_strength(4, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(5, dipole_integrals) - 0.977033) < 10**-3
    assert abs(LR.get_oscillator_strength(6, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(7, dipole_integrals) - 0.110734) < 10**-3
    assert abs(LR.get_oscillator_strength(8, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(9, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(10, dipole_integrals) - 0.002530) < 10**-3
    assert abs(LR.get_oscillator_strength(11, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(12, dipole_integrals) - 0.000019) < 10**-3
    assert abs(LR.get_oscillator_strength(13, dipole_integrals) - 0.0) < 10**-3
    LR = genericLR.LinearResponseUCC(WF, excitations="SD", operator_type="selfconsistent")
    LR.calc_excitation_energies()
    assert abs(LR.excitation_energies[0] - 0.162962) < 10**-5
    assert abs(LR.excitation_energies[1] - 0.385979) < 10**-5
    assert abs(LR.excitation_energies[2] - 0.516725) < 10**-5
    assert abs(LR.excitation_energies[3] - 0.585337) < 10**-5
    assert abs(LR.excitation_energies[4] - 0.600210) < 10**-5
    assert abs(LR.excitation_energies[5] - 0.602570) < 10**-5
    assert abs(LR.excitation_energies[6] - 0.671853) < 10**-5
    assert abs(LR.excitation_energies[7] - 0.705532) < 10**-5
    assert abs(LR.excitation_energies[8] - 0.805981) < 10**-5
    assert abs(LR.excitation_energies[9] - 0.843321) < 10**-5
    assert abs(LR.excitation_energies[10] - 0.923462) < 10**-5
    assert abs(LR.excitation_energies[11] - 1.189882) < 10**-5
    assert abs(LR.excitation_energies[12] - 1.512350) < 10**-5
    assert abs(LR.excitation_energies[13] - 1.515402) < 10**-5
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.007799) < 10**-3
    assert abs(LR.get_oscillator_strength(3, dipole_integrals) - 0.000296) < 10**-3
    assert abs(LR.get_oscillator_strength(4, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(5, dipole_integrals) - 0.996467) < 10**-3
    assert abs(LR.get_oscillator_strength(6, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(7, dipole_integrals) - 0.110723) < 10**-3
    assert abs(LR.get_oscillator_strength(8, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(9, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(10, dipole_integrals) - 0.002539) < 10**-3
    assert abs(LR.get_oscillator_strength(11, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(12, dipole_integrals) - 0.000019) < 10**-3
    assert abs(LR.get_oscillator_strength(13, dipole_integrals) - 0.0) < 10**-3


def test_be_sto3g_uccsd_lr_naive() -> None:  # pylint: disable=R0915
    """Test Linear Response for uccsd(2,2)."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Be  0.0  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
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
    LR = genericLR.LinearResponseUCC(WF, excitations="SD", operator_type="selfconsistent")
    LR.calc_excitation_energies()
    dipole_integrals = (
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    )
    assert abs(LR.excitation_energies[0] - 0.000001) < 10**-5
    assert abs(LR.excitation_energies[1] - 0.000001) < 10**-5
    assert abs(LR.excitation_energies[2] - 0.246512) < 10**-5
    assert abs(LR.excitation_energies[3] - 0.246512) < 10**-5
    assert abs(LR.excitation_energies[4] - 0.259511) < 10**-5
    assert abs(LR.excitation_energies[5] - 0.380953) < 10**-5
    assert abs(LR.excitation_energies[6] - 4.168480) < 10**-5
    assert abs(LR.excitation_energies[7] - 4.168480) < 10**-5
    assert abs(LR.excitation_energies[8] - 4.188686) < 10**-5
    assert abs(LR.excitation_energies[9] - 4.401923) < 10**-5
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.358883) < 10**-3
    assert abs(LR.get_oscillator_strength(3, dipole_integrals) - 0.358883) < 10**-3
    assert abs(LR.get_oscillator_strength(4, dipole_integrals) - 0.300382) < 10**-3
    assert abs(LR.get_oscillator_strength(5, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(6, dipole_integrals) - 0.137059) < 10**-3
    assert abs(LR.get_oscillator_strength(7, dipole_integrals) - 0.137059) < 10**-3
    assert abs(LR.get_oscillator_strength(8, dipole_integrals) - 0.127297) < 10**-3
    assert abs(LR.get_oscillator_strength(9, dipole_integrals) - 0.0) < 10**-3
    LR = genericLR.LinearResponseUCC(WF, excitations="SD", operator_type="naive")
    LR.calc_excitation_energies()
    assert abs(LR.excitation_energies[0] - 0.000001) < 10**-5
    assert abs(LR.excitation_energies[1] - 0.000001) < 10**-5
    assert abs(LR.excitation_energies[2] - 0.246512) < 10**-5
    assert abs(LR.excitation_energies[3] - 0.246512) < 10**-5
    assert abs(LR.excitation_energies[4] - 0.259511) < 10**-5
    assert abs(LR.excitation_energies[5] - 0.380954) < 10**-5
    assert abs(LR.excitation_energies[6] - 4.168480) < 10**-5
    assert abs(LR.excitation_energies[7] - 4.168480) < 10**-5
    assert abs(LR.excitation_energies[8] - 4.188686) < 10**-5
    assert abs(LR.excitation_energies[9] - 4.401923) < 10**-5
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.358883) < 10**-3
    assert abs(LR.get_oscillator_strength(3, dipole_integrals) - 0.358883) < 10**-3
    assert abs(LR.get_oscillator_strength(4, dipole_integrals) - 0.300382) < 10**-3
    assert abs(LR.get_oscillator_strength(5, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(6, dipole_integrals) - 0.137059) < 10**-3
    assert abs(LR.get_oscillator_strength(7, dipole_integrals) - 0.137059) < 10**-3
    assert abs(LR.get_oscillator_strength(8, dipole_integrals) - 0.127297) < 10**-3
    assert abs(LR.get_oscillator_strength(9, dipole_integrals) - 0.0) < 10**-3


def test_lih_sto3g_uccsd_lr_naive() -> None:  # pylint: disable=R0915
    """Test Linear Response for uccsd.
    This examples was used to find and fix a bug :)
    """
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """Li  0.0  0.0  0.0;
           H 1.671707274 0.0 0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
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
    WF.run_ucc("SD", True)
    LR = genericLR.LinearResponseUCC(WF, excitations="SD", operator_type="selfconsistent")
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
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.049920) < 10**-3
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.241184) < 10**-3
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.241184) < 10**-3
    assert abs(LR.get_oscillator_strength(3, dipole_integrals) - 0.158045) < 10**-3
    assert abs(LR.get_oscillator_strength(4, dipole_integrals) - 0.166539) < 10**-3
    assert abs(LR.get_oscillator_strength(5, dipole_integrals) - 0.010379) < 10**-3
    assert abs(LR.get_oscillator_strength(6, dipole_integrals) - 0.010379) < 10**-3
    assert abs(LR.get_oscillator_strength(7, dipole_integrals) - 0.006256) < 10**-3
    assert abs(LR.get_oscillator_strength(8, dipole_integrals) - 0.062386) < 10**-3
    assert abs(LR.get_oscillator_strength(9, dipole_integrals) - 0.128862) < 10**-3
    assert abs(LR.get_oscillator_strength(10, dipole_integrals) - 0.128862) < 10**-3
    assert abs(LR.get_oscillator_strength(11, dipole_integrals) - 0.046007) < 10**-3
    assert abs(LR.get_oscillator_strength(12, dipole_integrals) - 0.003904) < 10**-3
    LR = genericLR.LinearResponseUCC(WF, excitations="SD", operator_type="naive")
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
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.049920) < 10**-3
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.241184) < 10**-3
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.241184) < 10**-3
    assert abs(LR.get_oscillator_strength(3, dipole_integrals) - 0.158045) < 10**-3
    assert abs(LR.get_oscillator_strength(4, dipole_integrals) - 0.166539) < 10**-3
    assert abs(LR.get_oscillator_strength(5, dipole_integrals) - 0.010379) < 10**-3
    assert abs(LR.get_oscillator_strength(6, dipole_integrals) - 0.010379) < 10**-3
    assert abs(LR.get_oscillator_strength(7, dipole_integrals) - 0.006256) < 10**-3
    assert abs(LR.get_oscillator_strength(8, dipole_integrals) - 0.062386) < 10**-3
    assert abs(LR.get_oscillator_strength(9, dipole_integrals) - 0.128862) < 10**-3
    assert abs(LR.get_oscillator_strength(10, dipole_integrals) - 0.128862) < 10**-3
    assert abs(LR.get_oscillator_strength(11, dipole_integrals) - 0.046007) < 10**-3
    assert abs(LR.get_oscillator_strength(12, dipole_integrals) - 0.003904) < 10**-3


def test_OHm_sto3g_uccsd_lr() -> None:  # pylint: disable=R0915
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """O  0.0           0.0  0.0;
           H  1.4  0.0  0.0;""",
        distance_unit="angstrom",
        molecular_charge=-1,
    )
    SQobj.set_basis_set("STO-3G")
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

    WF.run_ucc("SD", True)

    LR = naiveLR.LinearResponseUCC(WF, excitations="SD")
    LR.calc_excitation_energies()
    assert abs(LR.excitation_energies[0] - 0.000000) < 10**-4
    assert abs(LR.excitation_energies[1] - 0.000000) < 10**-4
    assert abs(LR.excitation_energies[2] - 0.137490) < 10**-4
    assert abs(LR.excitation_energies[3] - 0.143084) < 10**-4
    assert abs(LR.excitation_energies[4] - 0.433389) < 10**-4
    assert abs(LR.excitation_energies[5] - 0.568480) < 10**-4
    assert abs(LR.excitation_energies[6] - 0.659111) < 10**-4
    assert abs(LR.excitation_energies[7] - 1.049113) < 10**-4
    assert abs(LR.excitation_energies[8] - 19.547355) < 10**-4
    assert abs(LR.excitation_energies[9] - 19.838150) < 10**-4
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.000000) < 10**-3
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.000000) < 10**-3
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.000304) < 10**-3
    assert abs(LR.get_oscillator_strength(3, dipole_integrals) - 0.000360) < 10**-3
    assert abs(LR.get_oscillator_strength(4, dipole_integrals) - 0.045164) < 10**-3
    assert abs(LR.get_oscillator_strength(5, dipole_integrals) - 0.854043) < 10**-3
    assert abs(LR.get_oscillator_strength(6, dipole_integrals) - 0.010548) < 10**-3
    assert abs(LR.get_oscillator_strength(7, dipole_integrals) - 0.041511) < 10**-3
    assert abs(LR.get_oscillator_strength(8, dipole_integrals) - 0.094685) < 10**-3
    assert abs(LR.get_oscillator_strength(9, dipole_integrals) - 0.001133) < 10**-3
    LR = selfconsistentLR.LinearResponseUCC(WF, excitations="SD")
    LR.calc_excitation_energies()
    assert abs(LR.excitation_energies[0] - 0.000000) < 10**-4
    assert abs(LR.excitation_energies[1] - 0.000000) < 10**-4
    assert abs(LR.excitation_energies[2] - 0.137490) < 10**-4
    assert abs(LR.excitation_energies[3] - 0.143084) < 10**-4
    assert abs(LR.excitation_energies[4] - 0.433389) < 10**-4
    assert abs(LR.excitation_energies[5] - 0.568480) < 10**-4
    assert abs(LR.excitation_energies[6] - 0.659111) < 10**-4
    assert abs(LR.excitation_energies[7] - 1.049113) < 10**-4
    assert abs(LR.excitation_energies[8] - 19.547355) < 10**-4
    assert abs(LR.excitation_energies[9] - 19.838150) < 10**-4
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.000000) < 10**-3
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.000000) < 10**-3
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.000304) < 10**-3
    assert abs(LR.get_oscillator_strength(3, dipole_integrals) - 0.000360) < 10**-3
    assert abs(LR.get_oscillator_strength(4, dipole_integrals) - 0.045164) < 10**-3
    assert abs(LR.get_oscillator_strength(5, dipole_integrals) - 0.854043) < 10**-3
    assert abs(LR.get_oscillator_strength(6, dipole_integrals) - 0.010548) < 10**-3
    assert abs(LR.get_oscillator_strength(7, dipole_integrals) - 0.041511) < 10**-3
    assert abs(LR.get_oscillator_strength(8, dipole_integrals) - 0.094685) < 10**-3
    assert abs(LR.get_oscillator_strength(9, dipole_integrals) - 0.001133) < 10**-3
