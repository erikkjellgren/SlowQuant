import numpy as np
from scipy.linalg import solve

import slowquant.SlowQuant as sq
import slowquant.unitary_coupled_cluster.linear_response.naive as naive  # pylint: disable=consider-using-from-import
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


def test_H2_sto3g_naive():
    """
    Test of polarisability for naive LR with working equations
    """
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0   0.0  0.0;
            H  0.74  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
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
        "SD",
    )
    WF.run_ucc(True)

    # Linear Response
    LR = naive.LinearResponseUCC(WF, excitations="SD")
    LR.calc_excitation_energies()
    
    # Calculate dipole integrals
    dipole_integrals = (
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    )

    property_gradient = LR.get_property_gradient(dipole_integrals)
    resp = (solve(LR.hessian, property_gradient))
    alpha = np.einsum('xi,xi->i', resp, property_gradient)

    thresh = 10**-4

    # Check excitation energies - reference dalton mcscf
    assert abs(alpha[0] - 2.775271948863) < thresh
    assert abs(alpha[1] - 0.0) < thresh
    assert abs(alpha[2] - 0.0) < thresh


def test_LiH_sto3g_naive():
    """
    Test of polarisability for naive LR with working equations
    """
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0   0.0  0.0;
            Li  0.8  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
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
        "SD",
    )
    WF.run_ucc(True)

    # Linear Response
    LR = naive.LinearResponseUCC(WF, excitations="SD")
    LR.calc_excitation_energies()
    
    # Calculate dipole integrals
    dipole_integrals = (
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    )

    property_gradient = LR.get_property_gradient(dipole_integrals)
    resp = (solve(LR.hessian, property_gradient))
    alpha = np.einsum('xi,xi->i', resp, property_gradient)

    thresh = 10**-2

    # Check excitation energies - reference dalton mcscf
    assert abs(alpha[0] - 0.5238650005008) < thresh
    assert abs(alpha[1] - 20.01552907544) < thresh
    assert abs(alpha[2] - 20.01552907544) < thresh


def test_H10_sto3g_naive():
    """
    Test of polarisability for naive LR with working equations
    """
    # Slowquant Object with parameters and setup
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
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
    SQobj.set_basis_set("STO-3G")
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
        "SD",
    )
    WF.run_ucc(True)

    # Linear Response
    LR = naive.LinearResponseUCC(WF, excitations="SD")
    LR.calc_excitation_energies()
    
    # Calculate dipole integrals
    dipole_integrals = (
        SQobj.integral.get_multipole_matrix([1, 0, 0]),
        SQobj.integral.get_multipole_matrix([0, 1, 0]),
        SQobj.integral.get_multipole_matrix([0, 0, 1]),
    )

    property_gradient = LR.get_property_gradient(dipole_integrals)
    resp = (solve(LR.hessian, property_gradient))
    alpha = np.einsum('xi,xi->i', resp, property_gradient)

    thresh = 10**-3

    # Check excitation energies - reference dalton mcscf
    assert abs(alpha[0] - 72.343635) < thresh
    assert abs(alpha[1] - 0.0) < thresh
    assert abs(alpha[2] - 0.0) < thresh
