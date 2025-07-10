<<<<<<< HEAD
=======
# type: ignore
>>>>>>> 60937ff (adapt)
import numpy as np

import slowquant.SlowQuant as sq
import slowquant.unitary_coupled_cluster.linear_response.allstatetransfer as allstlr
import slowquant.unitary_coupled_cluster.linear_response.naive as naivelr
<<<<<<< HEAD
from slowquant.unitary_coupled_cluster.sa_ups_wavefunction import WaveFunctionSAUPS
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS


"""Test LiH UCCSD(2,2) LR."""
SQobj = sq.SlowQuant()
SQobj.set_molecule(
     """O 0.000000000000  -0.143225816552   0.000000000000;
           H 1.638036840407   1.136548822547  -0.000000000000;
           H -1.638036840407   1.136548822547  -0.000000000000;""",
=======
from slowquant.unitary_coupled_cluster.sa_adapt_wavefunction import WaveFunctionSAADAPT
#from slowquant.unitary_coupled_cluster.sa_ups_wavefunction import WaveFunctionSAUPS
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS

"""This should give exactly the same as FCI.

Since all states, are includes in the subspace expansion.
"""
SQobj = sq.SlowQuant()
SQobj.set_molecule(
    """H   0.0  0.0  0.0;
       Li   0.0  0.0  0.735;""",
>>>>>>> 60937ff (adapt)
    distance_unit="angstrom",
)
SQobj.set_basis_set("STO-3G")
SQobj.init_hartree_fock()
SQobj.hartree_fock.run_restricted_hartree_fock()
<<<<<<< HEAD
h_core = SQobj.integral.kinetic_energy_matrix + \
    SQobj.integral.nuclear_attraction_matrix
g_eri = SQobj.integral.electron_repulsion_tensor
WF = WaveFunctionUPS(
=======
h_core = SQobj.integral.kinetic_energy_matrix + SQobj.integral.nuclear_attraction_matrix
g_eri = SQobj.integral.electron_repulsion_tensor
WF = WaveFunctionSAADAPT(
>>>>>>> 60937ff (adapt)
    SQobj.molecule.number_electrons,
    (4, 4),
    SQobj.hartree_fock.mo_coeff,
    h_core,
    g_eri,
<<<<<<< HEAD
    "tUPS",
    ansatz_options={"n_layers": 2, "skip_last_singles": True},
    include_active_kappa=True,
)
print(WF.energy_elec)
WF.run_wf_optimization_1step("SLSQP", True)
=======
    (
        [
            #[1],
            [1],
            [1],
            [1],
        ],
        [
            #["00001111"],
            ["11100100"],
            ["11101000"],
            ["11110000"],
            #["1100"],
        ],
    ),
    #(
    #    [
    #        [1],
    #        [2 ** (-1 / 2), -(2 ** (-1 / 2))],
    #        [1],
    #    ],
    #    [
    #        ["1100"],
    #        ["1001", "0110"],
    #        ["0011"],
    #    ],
    #),
    "tUPS",
    ansatz_options={"n_layers": 1, "skip_last_singles": True},
    include_active_kappa=True,
)


#WF.try_toCalculate_gradient()
#WF.gradient(WF.thetas, True, False)

#WF.run_wf_optimization_1step("bfgs", True)
#WF.do_adapt()

WF.do_adapt()
#print(WF.ups_layout.excitation_indices)
#print(WF.ups_layout.excitation_operator_type)

print("Final energy after eigh digonalisation")
print(WF.energy_states)
#print(WF.excitation_energies[0])
#print(WF.excitation_energies[1])
#print(WF.excitation_energies[2])
>>>>>>> 60937ff (adapt)
