import numpy as np

import slowquant.SlowQuant as sq
import slowquant.unitary_coupled_cluster.linear_response.allstatetransfer as allstlr
import slowquant.unitary_coupled_cluster.linear_response.naive as naivelr
from slowquant.unitary_coupled_cluster.sa_ups_wavefunction import WaveFunctionSAUPS
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS


"""Test LiH UCCSD(2,2) LR."""
SQobj = sq.SlowQuant()
SQobj.set_molecule(
     """O 0.000000000000  -0.143225816552   0.000000000000;
           H 1.638036840407   1.136548822547  -0.000000000000;
           H -1.638036840407   1.136548822547  -0.000000000000;""",
    distance_unit="angstrom",
)
SQobj.set_basis_set("STO-3G")
SQobj.init_hartree_fock()
SQobj.hartree_fock.run_restricted_hartree_fock()
h_core = SQobj.integral.kinetic_energy_matrix + \
    SQobj.integral.nuclear_attraction_matrix
g_eri = SQobj.integral.electron_repulsion_tensor
WF = WaveFunctionUPS(
    SQobj.molecule.number_electrons,
    (4, 4),
    SQobj.hartree_fock.mo_coeff,
    h_core,
    g_eri,
    "tUPS",
    ansatz_options={"n_layers": 2, "skip_last_singles": True},
    include_active_kappa=True,
)
print(WF.energy_elec)
WF.run_wf_optimization_1step("SLSQP", True)
