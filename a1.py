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
    """C  4.673795    6.280948  0.00 ;
C  5.901190    5.572311  0.00 ;
C  5.901190    4.155037  0.00 ;
C  4.673795    3.446400  0.00 ;
C  3.446400    4.155037  0.00 ;
C  3.446400    5.572311  0.00 ;
H  4.673795    7.376888  0.00 ;
H  6.850301    6.120281  0.00 ;
H  6.850301    3.607068  0.00 ;
H  4.673795    2.350461  0.00 ;
H  2.497289    3.607068  0.00 ;
H  2.497289    6.120281  0.00 ;
""",
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
help(WF)
dipole_integrals = (
    SQobj.integral.get_multipole_matrix([1, 0, 0]),
    SQobj.integral.get_multipole_matrix([0, 1, 0]),
    SQobj.integral.get_multipole_matrix([0, 0, 1]),
)
WF.run_wf_optimization_1step("SLSQP", True)

print(WF.g_mo.shape)
print(WF.h_mo.shape)
print("+++++++++++++++++")
print(WF.ansatz_options)
print(WF.ups_layout)
print(WF.ups_layout.excitation_indices)
print(WF.ups_layout.excitation_operator_type)
LR = naivelr.LinearResponseUCC(WF, excitations="S")
LR.calc_excitation_energies()
print(LR.excitation_energies[0])
print(LR.excitation_energies[1])
print(LR.excitation_energies[2])
print(LR.excitation_energies[3])
print(LR.excitation_energies[4])
print(LR.excitation_energies[5])
print(LR.excitation_energies[6])
print(LR.excitation_energies[7])
print(LR.excitation_energies[8])

=======
    (
        [
            #[1],
            #[1],
            [1],
            [1],
        ],
        [
            #["00001111"],
            #["00100111"],
            ["00011011"],
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
