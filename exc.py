import numpy as np

import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.base import Hamiltonian, expectation_value
from slowquant.unitary_coupled_cluster.linear_response import LinearResponseUCC
from slowquant.unitary_coupled_cluster.linear_response_matrix import LinearResponseUCCMatrix
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC


if True:
    """Test Linear Response for UCCSD."""
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(
        """H  0.0  0.0  0.0;
           H  0.0  2.0  0.0;
           H  1.5  0.0  0.0;
           H  1.5  2.0  0.0;""",
        distance_unit="bohr",
    )
    SQobj.set_basis_set("sto-3g")
    SQobj.init_hartree_fock()
    SQobj.hartree_fock.run_restricted_hartree_fock()
    print(SQobj.molecule.nuclear_repulsion)
    num_bf = SQobj.molecule.number_bf
    h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
    g_eri = SQobj.integral.electron_repulsion_tensor
    WF = WaveFunctionUCC(
        SQobj.molecule.number_bf * 2,
        SQobj.molecule.number_electrons,
        [0, 1, 2, 3, 4, 5, 6, 7],
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
    )
    WF.run_UCC("SD", False)
    LR = LinearResponseUCCMatrix(WF, excitations="SD")
    LR.calc_excitation_energies()
    print(LR.excitation_energies)
    #LR = LinearResponseUCC(WF, excitations="SD")
    #LR.calc_excitation_energies()
    #print(LR.excitation_energies)

