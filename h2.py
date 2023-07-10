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
    print(LR.get_nice_output([SQobj.integral.get_multipole_matrix([1, 0, 0]),
                              SQobj.integral.get_multipole_matrix([0, 1, 0]),
                              SQobj.integral.get_multipole_matrix([0, 0, 1])]))
    #LR = LinearResponseUCC(WF, excitations="SD")
    #LR.calc_excitation_energies()
    #print(LR.excitation_energies)
    print(SQobj.integral.get_multipole_matrix([1, 0, 0]))
