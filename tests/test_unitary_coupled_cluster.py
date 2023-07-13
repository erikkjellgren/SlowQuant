import numpy as np

import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.base import Hamiltonian, expectation_value
from slowquant.unitary_coupled_cluster.linear_response_matrix import (
    LinearResponseUCCMatrix,
)
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


def test_HeH_sto3g_HF() -> None:
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
    WF.run_UCC("S", True)
    assert abs(WF.ucc_energy - (-4.262632309847)) < 10**-8


def test_LiH_sto3g_HF() -> None:
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
    WF.run_UCC("S", True)
    assert abs(WF.ucc_energy - (-8.862246324082243)) < 10**-8


def test_HeH_sto3g_UCCS() -> None:
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
    WF.run_UCC("S")
    assert abs(WF.ucc_energy - (-4.262632309847)) < 10**-8


def test_H10_STO3G_UCCSD() -> None:
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
    WF.run_UCC("SD", False)
    assert abs(WF.ucc_energy - (-18.839645894737956)) < 10**-8


def test_H2_431G_OOUCCSD() -> None:
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
    WF.run_UCC("SD", True)
    assert abs(WF.ucc_energy - (-1.860533598715)) < 10**-8


def test_H2_431G_OOUCCD() -> None:
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
    WF.run_UCC("D", True)
    assert abs(WF.ucc_energy - (-1.860533598715)) < 10**-8


def test_H2_431G_UCCSD() -> None:
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
    WF.run_UCC("SD", False)
    assert abs(WF.ucc_energy - (-1.8466833679296024)) < 10**-8


def test_H4_STO3G_OOUCCSD() -> None:
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
    WF.run_UCC("SD", True)
    assert abs(WF.ucc_energy - (-5.211066791547)) < 10**-8
    # Test sparse matrix also works
    H = Hamiltonian(h_core, g_eri, WF.c_trans, WF.num_spin_orbs, WF.num_elec)
    assert abs(WF.ucc_energy - expectation_value(WF.state_vector, H, WF.state_vector, use_csr=0)) < 10**-8


def test_H4_STO3G_OOUCCD() -> None:
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
    WF.run_UCC("D", True)
    assert abs(WF.ucc_energy - (-5.211066791547)) < 10**-8
    # Test sparse matrix also works
    H = Hamiltonian(h_core, g_eri, WF.c_trans, WF.num_spin_orbs, WF.num_elec)
    assert abs(WF.ucc_energy - expectation_value(WF.state_vector, H, WF.state_vector, use_csr=0)) < 10**-8


def test_H2_STO3G_UCCSD_LR() -> None:
    """Test Linear Response for UCCSD."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0  0.0  0.0;
           H  0.0  0.0  0.7;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("sto-3g")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    num_bf = SQobj.molecule.number_bf
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
    WF.run_UCC("SD", False)
    LR = LinearResponseUCCMatrix(WF, excitations="SD")
    LR.calc_excitation_energies()
    assert abs(LR.excitation_energies[0] - 1.0157) < 10**-3
    assert abs(LR.excitation_energies[1] - 1.7195) < 10**-3
    assert abs(LR.get_excited_state_overlap(0) - 0.0) < 10**-3
    assert abs(LR.get_excited_state_overlap(1) - 0.0) < 10**-3
    dipole_int = SQobj.integral.get_multipole_matrix([0, 0, 1])
    assert abs(abs(LR.get_transition_dipole(0, dipole_int)) - 1.1441) < 10**-3
    assert abs(LR.get_transition_dipole(1, dipole_int) - 0.0) < 10**-3


def test_H4_STO3G_UCCDQ() -> None:
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
    num_bf = A.molecule.number_bf
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
    WF.run_UCC("DQ", False)
    assert abs(WF.ucc_energy + A.molecule.nuclear_repulsion - (-1.968914822185857)) < 10**-7


def test_H2_631G_HF_LR() -> None:
    """Test Linear Response for OO-UCCSD(2,2)."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0   0.0  0.0;
           H  0.74  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("6-31G")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    num_bf = SQobj.molecule.number_bf
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
    WF.run_UCC("SD", True)
    LR = LinearResponseUCCMatrix(WF, excitations="SD")
    LR.calc_excitation_energies()
    dipole_integrals = [
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    ]
    assert abs(LR.excitation_energies[0] - 0.551961) < 10**-3
    assert abs(LR.excitation_energies[1] - 1.051638) < 10**-3
    assert abs(LR.excitation_energies[2] - 1.603563) < 10**-3
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.6509) < 10**-3
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.0635) < 10**-3


def test_H2_631G_OOUCCSD_LR() -> None:
    """Test Linear Response for OO-UCCSD(2,2)."""
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
    WF.run_UCC("SD", True)
    LR = LinearResponseUCCMatrix(WF, excitations="SD")
    LR.calc_excitation_energies()
    dipole_integrals = [
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    ]
    assert abs(LR.excitation_energies[0] - 0.574413) < 10**-3
    assert abs(LR.excitation_energies[1] - 1.043177) < 10**-3
    assert abs(LR.excitation_energies[2] - 1.139481) < 10**-3
    assert abs(LR.excitation_energies[3] - 1.365960) < 10**-3
    assert abs(LR.excitation_energies[4] - 1.831196) < 10**-3
    assert abs(LR.excitation_energies[5] - 2.581273) < 10**-3
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.6338) < 10**-3
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.0) < 10**-3
    assert abs(LR.get_oscillator_strength(3, dipole_integrals) - 0.0311) < 10**-3
    assert abs(LR.get_oscillator_strength(4, dipole_integrals) - 0.0421) < 10**-3
    assert abs(LR.get_oscillator_strength(5, dipole_integrals) - 0.0) < 10**-3


def test_H4_STO3G_UCCSD_LR_naive() -> None:
    """Test Linear Response for UCCSD(4,4)."""
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
    WF.run_UCC("SD", False)
    LR = LinearResponseUCCMatrix(WF, excitations="SD", do_selfconsistent_operators=False)
    LR.calc_excitation_energies()
    dipole_integrals = [
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    ]
    assert abs(LR.excitation_energies[0] - 0.162970) < 10**-5
    assert abs(LR.excitation_energies[1] - 0.418787) < 10**-5
    assert abs(LR.excitation_energies[2] - 0.550528) < 10**-5
    assert abs(LR.excitation_energies[3] - 0.585344) < 10**-5
    assert abs(LR.excitation_energies[4] - 0.600218) < 10**-5
    assert abs(LR.excitation_energies[5] - 0.602967) < 10**-5
    assert abs(LR.excitation_energies[6] - 0.680446) < 10**-5
    assert abs(LR.excitation_energies[7] - 0.705539) < 10**-5
    assert abs(LR.excitation_energies[8] - 0.805982) < 10**-5
    assert abs(LR.excitation_energies[9] - 0.843330) < 10**-5
    assert abs(LR.excitation_energies[10] - 0.923469) < 10**-5
    assert abs(LR.excitation_energies[11] - 1.189888) < 10**-5
    assert abs(LR.excitation_energies[12] - 1.512358) < 10**-5
    assert abs(LR.excitation_energies[13] - 1.515410) < 10**-5
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.026095) < 10**-5
    assert abs(LR.get_oscillator_strength(3, dipole_integrals) - 0.000298) < 10**-5
    assert abs(LR.get_oscillator_strength(4, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(5, dipole_integrals) - 0.977033) < 10**-5
    assert abs(LR.get_oscillator_strength(6, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(7, dipole_integrals) - 0.110734) < 10**-5
    assert abs(LR.get_oscillator_strength(8, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(9, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(10, dipole_integrals) - 0.002530) < 10**-5
    assert abs(LR.get_oscillator_strength(11, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(12, dipole_integrals) - 0.000019) < 10**-5
    assert abs(LR.get_oscillator_strength(13, dipole_integrals) - 0.0) < 10**-5
    LR = LinearResponseUCCMatrix(WF, excitations="SD")
    LR.calc_excitation_energies()
    dipole_integrals = [
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    ]
    assert abs(LR.excitation_energies[0] - 0.162960) < 10**-5
    assert abs(LR.excitation_energies[1] - 0.385979) < 10**-5
    assert abs(LR.excitation_energies[2] - 0.516725) < 10**-5
    assert abs(LR.excitation_energies[3] - 0.585337) < 10**-5
    assert abs(LR.excitation_energies[4] - 0.600210) < 10**-5
    assert abs(LR.excitation_energies[5] - 0.602570) < 10**-5
    assert abs(LR.excitation_energies[6] - 0.671853) < 10**-5
    assert abs(LR.excitation_energies[7] - 0.705532) < 10**-5
    assert abs(LR.excitation_energies[8] - 0.805982) < 10**-5
    assert abs(LR.excitation_energies[9] - 0.843322) < 10**-5
    assert abs(LR.excitation_energies[10] - 0.923462) < 10**-5
    assert abs(LR.excitation_energies[11] - 1.189882) < 10**-5
    assert abs(LR.excitation_energies[12] - 1.512350) < 10**-5
    assert abs(LR.excitation_energies[13] - 1.515402) < 10**-5
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.007791) < 10**-5
    assert abs(LR.get_oscillator_strength(3, dipole_integrals) - 0.000298) < 10**-5
    assert abs(LR.get_oscillator_strength(4, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(5, dipole_integrals) - 0.996432) < 10**-5
    assert abs(LR.get_oscillator_strength(6, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(7, dipole_integrals) - 0.110734) < 10**-5
    assert abs(LR.get_oscillator_strength(8, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(9, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(10, dipole_integrals) - 0.002530) < 10**-5
    assert abs(LR.get_oscillator_strength(11, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(12, dipole_integrals) - 0.000019) < 10**-5
    assert abs(LR.get_oscillator_strength(13, dipole_integrals) - 0.0) < 10**-5


def test_Be_STO3G_UCCSD_LR_naive() -> None:
    """Test Linear Response for UCCSD(4,4)."""
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
    WF.run_UCC("SD", True)
    LR = LinearResponseUCCMatrix(WF, excitations="SD")
    LR.calc_excitation_energies()
    dipole_integrals = [
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    ]
    assert abs(LR.excitation_energies[0] - 0.000015) < 10**-5
    assert abs(LR.excitation_energies[1] - 0.000020) < 10**-5
    assert abs(LR.excitation_energies[2] - 0.246521) < 10**-5
    assert abs(LR.excitation_energies[3] - 0.246521) < 10**-5
    assert abs(LR.excitation_energies[4] - 0.259511) < 10**-5
    assert abs(LR.excitation_energies[5] - 0.380952) < 10**-5
    assert abs(LR.excitation_energies[6] - 4.168478) < 10**-5
    assert abs(LR.excitation_energies[7] - 4.168478) < 10**-5
    assert abs(LR.excitation_energies[8] - 4.188693) < 10**-5
    assert abs(LR.excitation_energies[9] - 4.401753) < 10**-5
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.358890) < 10**-5
    assert abs(LR.get_oscillator_strength(3, dipole_integrals) - 0.358890) < 10**-5
    assert abs(LR.get_oscillator_strength(4, dipole_integrals) - 0.300338) < 10**-5
    assert abs(LR.get_oscillator_strength(5, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(6, dipole_integrals) - 0.137051) < 10**-5
    assert abs(LR.get_oscillator_strength(7, dipole_integrals) - 0.137051) < 10**-5
    assert abs(LR.get_oscillator_strength(8, dipole_integrals) - 0.127283) < 10**-5
    assert abs(LR.get_oscillator_strength(9, dipole_integrals) - 0.0) < 10**-5
    LR = LinearResponseUCCMatrix(WF, excitations="SD", do_selfconsistent_operators=False)
    LR.calc_excitation_energies()
    dipole_integrals = [
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    ]
    assert abs(LR.excitation_energies[0] - 0.000015) < 10**-5
    assert abs(LR.excitation_energies[1] - 0.000020) < 10**-5
    assert abs(LR.excitation_energies[2] - 0.246521) < 10**-5
    assert abs(LR.excitation_energies[3] - 0.246521) < 10**-5
    assert abs(LR.excitation_energies[4] - 0.259528) < 10**-5
    assert abs(LR.excitation_energies[5] - 0.380969) < 10**-5
    assert abs(LR.excitation_energies[6] - 4.168478) < 10**-5
    assert abs(LR.excitation_energies[7] - 4.168478) < 10**-5
    assert abs(LR.excitation_energies[8] - 4.188694) < 10**-5
    assert abs(LR.excitation_energies[9] - 4.401754) < 10**-5
    assert abs(LR.get_oscillator_strength(0, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(1, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(2, dipole_integrals) - 0.358890) < 10**-5
    assert abs(LR.get_oscillator_strength(3, dipole_integrals) - 0.358890) < 10**-5
    assert abs(LR.get_oscillator_strength(4, dipole_integrals) - 0.300362) < 10**-5
    assert abs(LR.get_oscillator_strength(5, dipole_integrals) - 0.0) < 10**-5
    assert abs(LR.get_oscillator_strength(6, dipole_integrals) - 0.137051) < 10**-5
    assert abs(LR.get_oscillator_strength(7, dipole_integrals) - 0.137051) < 10**-5
    assert abs(LR.get_oscillator_strength(8, dipole_integrals) - 0.127283) < 10**-5
    assert abs(LR.get_oscillator_strength(9, dipole_integrals) - 0.0) < 10**-5
