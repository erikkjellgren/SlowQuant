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
        """Li  0.0  0.0  0.0; H 0.0 0.0 1.0""",
        distance_unit="angstrom",
        molecular_charge=0,
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
        [0,1,2,3,4,5,6,7],
        SQobj.hartree_fock.mo_coeff,
        h_core,
        g_eri,
    )
    #WF.kappa = [0.005988364755839179, -2.1885759944946952e-08, 1.5992531886343553e-08, -3.516924219931774e-08, 7.816039325805208e-05, 7.3538681690042274e-09, 4.538762183575819e-08, -2.7997073396202625e-08, -0.011994201364541091, 2.8133116089391523e-07, 1.250306835954386e-07, 6.164180144858557e-07, -4.123182449737314e-06, 0.3971811749771395, -6.561192457502697e-07, -1.7913441741716648e-07, -1.1168340847315917e-06, 2.134835445476741e-06, 6.669001190677096e-07, 0.39718370041019396, 5.509887617663071e-06, -1.435576170805175e-07, 0.3971841273920861, 4.843119677257789e-08]
    #WF.theta1 = [1.6511022986138332e-06, 0, 0, -9.78466632952094e-07, 7.008380902357084e-07, 0, 0, -1.4041639953474357e-06, 1.429690887486518e-06, 0, 0, -5.327713851957565e-07]
    #WF.theta2 = [-0.22527390589698376, 0, -6.894163119200487e-08, 0, 1.431496299386387e-07, 3.179303759345898e-08, 0, 1.8154135538136842e-07, 0, -0.22527324508872285, 0, 8.123373892036201e-07, 6.722465220871827e-07, 0, -0.22527322899344374]
    WF.run_UCC("SD", False)
    exit()
    LR = LinearResponseUCCMatrix(WF, excitations="SD", is_spin_conserving=True)
    LR.calc_excitation_energies()
    print(LR.excitation_energies)
