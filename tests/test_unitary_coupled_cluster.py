import numpy as np

import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.base import Hamiltonian, expectation_value
from slowquant.unitary_coupled_cluster.linear_response import LinearResponseUCC
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
    WF = WaveFunctionUCC(A.molecule.number_bf * 2, A.molecule.number_electrons, [], S_sqrt, h_core, g_eri)
    WF.run_HF()
    assert abs(WF.hf_energy - (-4.262632309847)) < 10**-8


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
    WF = WaveFunctionUCC(A.molecule.number_bf * 2, A.molecule.number_electrons, [], S_sqrt, h_core, g_eri)
    WF.run_HF()
    assert abs(WF.hf_energy - (-8.862246324082243)) < 10**-8


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
    WF = WaveFunctionUCC(
        A.molecule.number_bf * 2, A.molecule.number_electrons, [0, 1, 2, 3], S_sqrt, h_core, g_eri
    )
    WF.run_UCC("S")
    assert abs(WF.ucc_energy - (-4.262632309847)) < 10**-8


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
    num_bf = A.molecule.number_bf
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        A.molecule.number_bf * 2,
        A.molecule.number_electrons,
        [0, 1, 2, 3],
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
    num_bf = A.molecule.number_bf
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        A.molecule.number_bf * 2,
        A.molecule.number_electrons,
        [0, 1, 2, 3],
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
    num_bf = A.molecule.number_bf
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        A.molecule.number_bf * 2,
        A.molecule.number_electrons,
        [0, 1, 2, 3],
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
    num_bf = A.molecule.number_bf
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        A.molecule.number_bf * 2,
        A.molecule.number_electrons,
        [2, 3, 4, 5],
        A.hartree_fock.mo_coeff,
        h_core,
        g_eri,
    )
    WF.run_UCC("SD", True)
    assert abs(WF.ucc_energy - (-5.211066791547)) < 10**-8
    # Test sparse matrix also works
    H = Hamiltonian(
        h_core, g_eri, WF.c_trans, WF.num_spin_orbs, WF.num_elec
    )
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
    num_bf = A.molecule.number_bf
    h_core = A.integral.kinetic_energy_matrix + A.integral.nuclear_attraction_matrix
    g_eri = A.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        A.molecule.number_bf * 2,
        A.molecule.number_electrons,
        [2, 3, 4, 5],
        A.hartree_fock.mo_coeff,
        h_core,
        g_eri,
        include_active_kappa=True,
    )
    WF.run_UCC("D", True)
    assert abs(WF.ucc_energy - (-5.211066791547)) < 10**-8
    # Test sparse matrix also works
    H = Hamiltonian(
        h_core, g_eri, WF.c_trans, WF.num_spin_orbs, WF.num_elec
    )
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
        [0, 1, 2, 3],
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
    )
    WF.run_UCC("SD", False)
    LR = LinearResponseUCC(WF)
    LR.calc_excitation_energies()
    assert abs(LR.excitation_energies[0] - 0.6577) < 10**-3
    assert abs(LR.excitation_energies[1] - 0.6577) < 10**-3
    assert abs(LR.excitation_energies[2] - 0.6577) < 10**-3
    assert abs(LR.excitation_energies[3] - 1.0157) < 10**-3
    assert abs(LR.excitation_energies[4] - 1.7195) < 10**-3
    assert abs(LR.get_excited_state_overlap(0) - 0.0) < 10**-3
    assert abs(LR.get_excited_state_overlap(1) - 0.0) < 10**-3
    assert abs(LR.get_excited_state_overlap(2) - 0.0) < 10**-3
    assert abs(LR.get_excited_state_overlap(3) - 0.0) < 10**-3
    assert abs(abs(LR.get_excited_state_overlap(4)) - abs(0.1029)) < 10**-3
    dipole_int = SQobj.integral.get_multipole_matrix([0, 0, 1])
    assert abs(LR.get_transition_dipole(0, dipole_int) - 0.0) < 10**-3
    assert abs(LR.get_transition_dipole(1, dipole_int) - 0.0) < 10**-3
    assert abs(LR.get_transition_dipole(2, dipole_int) - 0.0) < 10**-3
    assert abs(LR.get_transition_dipole(3, dipole_int) - 1.1441) < 10**-3
    assert abs(LR.get_transition_dipole(4, dipole_int) - 0.0) < 10**-3
