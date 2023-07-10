import numpy as np

import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.base import Hamiltonian, expectation_value
from slowquant.unitary_coupled_cluster.linear_response_matrix import LinearResponseUCCMatrix
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
)
if True:
    """Test Linear Response for UCCSD."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0  0.0  0.0;
           H  0.7  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
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
    dipole_int = SQobj.integral.get_multipole_matrix([1, 0, 0])
    assert abs(LR.excitation_energies[0] - 0.980695) < 10**-3
    assert abs(abs(LR.get_transition_dipole(0, dipole_int)) - abs(1.1741)) < 10**-3
if True:
    """Test Linear Response for UCCSD."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0  0.0  0.0;
           H  0.7  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
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
    WF.run_UCC("S", False)
    LR = LinearResponseUCCMatrix(WF, excitations="S")
    LR.calc_excitation_energies()
    dipole_int = SQobj.integral.get_multipole_matrix([1, 0, 0])
    assert abs(LR.excitation_energies[0] - 0.980695) < 10**-3
    assert abs(abs(LR.get_transition_dipole(0, dipole_int)) - abs(1.1741)) < 10**-3


if True:
    """Test Linear Response for UCCSD."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0  0.0  0.0;
           H  0.7  0.0  0.0;""",
        distance_unit="angstrom",
    )
    SQobj.set_basis_set("STO-3G")
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
    WF.run_UCC("SD", True)
    LR = LinearResponseUCCMatrix(WF, excitations="SD")
    LR.calc_excitation_energies()
    dipole_int = SQobj.integral.get_multipole_matrix([1, 0, 0])
    assert abs(LR.excitation_energies[0] - 1.0157) < 10**-3
    assert abs(LR.excitation_energies[1] - 1.7195) < 10**-3
    assert abs(abs(LR.get_transition_dipole(0, dipole_int)) - abs(-1.1441)) < 10**-3
    assert abs(abs(LR.get_transition_dipole(1, dipole_int)) - abs(0.0)) < 10**-3



if True:
    """Test Linear Response for UCCSD."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0  0.0  0.0;
           H  0.7  0.0  0.0;""",
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
        (2, 2),
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
    )
    WF.run_UCC("SD", True)
    LR = LinearResponseUCCMatrix(WF, excitations="SD")
    LR.calc_excitation_energies()
    dipole_int = SQobj.integral.get_multipole_matrix([1, 0, 0])
    print(LR.get_nice_output([SQobj.integral.get_multipole_matrix([1, 0, 0]),
                              SQobj.integral.get_multipole_matrix([0, 1, 0]),
                              SQobj.integral.get_multipole_matrix([0, 0, 1])]))
    assert abs(LR.excitation_energies[0] - 0.587915) < 10**-3
    assert abs(LR.excitation_energies[1] - 1.038151) < 10**-3
    assert abs(LR.excitation_energies[2] - 1.205344) < 10**-3
    assert abs(LR.excitation_energies[3] - 1.394958) < 10**-3
    assert abs(LR.excitation_energies[4] - 1.861310) < 10**-3
    assert abs(LR.excitation_energies[5] - 2.706723) < 10**-3
    assert abs(abs(LR.get_transition_dipole(0, dipole_int)) - abs(1.2545)) < 10**-3
    assert abs(abs(LR.get_transition_dipole(1, dipole_int)) - abs(0.0)) < 10**-3
    assert abs(abs(LR.get_transition_dipole(2, dipole_int)) - abs(0.0)) < 10**-3
    assert abs(abs(LR.get_transition_dipole(3, dipole_int)) - abs(0.1957)) < 10**-3
    assert abs(abs(LR.get_transition_dipole(4, dipole_int)) - abs(0.2119)) < 10**-3
    assert abs(abs(LR.get_transition_dipole(5, dipole_int)) - abs(0.0)) < 10**-3


