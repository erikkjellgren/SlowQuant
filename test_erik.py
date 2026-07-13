import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response import generalized_naive
from slowquant.qiskit_interface.generalized_circuit_wavefunction import GeneralizedWaveFunctionCircuit
from qiskit_aer.primitives import Sampler
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from slowquant.qiskit_interface.generalized_interface import QuantumInterface
import slowquant.qiskit_interface.linear_response.generalized_naive as q_generalized_naive
import scipy
from slowquant.unitary_coupled_cluster.generalized_density_matrix import RDM1


# geometry = """H  0.0   0.0  0.0;
#     H  0.0  0.0  0.74"""
# basis = "STO-3g"
# active_space = ((1, 1), 4) #spin orbitaler or spinor basis
# charge = 0
# spin = 0




# mol = pyscf.M(atom=geometry, basis=basis, unit='angstrom', charge=charge, spin=spin, nucmod=1)
# mol.build()


# mf = scf.GHF(mol).x2c()
# mf.kernel()

# coeff=np.array(mf.mo_coeff, dtype=complex)

# WF =GeneralizedWaveFunctionUPS(
#     # mol.nelectron,
#     active_space,
#     coeff,
#     #C_u,
#     mol,
#     "fUCCSD",
#     True, #Do x2c
#     {"n_layers": 1, "is_spin_conserving" : False},
#     include_active_kappa=True,
# )
# WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)



import numpy
from pyscf import gto, lib

mol = gto.M(atom='Li 0 0 0', basis='ccpvdz', spin=1, verbose=0)
mol.set_rinv_origin(mol.atom_coord(0))   # sæt rinv-origo i kernen


# tjek at navnene findes hos dig (libcint-version kan variere lidt)
t1 = mol.intor('int1e_iprinvip').reshape(3,3,mol.nao,mol.nao)   # <∇_v i|1/r|∇_u j>
t2 = mol.intor('int1e_ipiprinv').reshape(3,3,mol.nao,mol.nao)  # <i|1/r|∇∇j> #den jeg har vender den forkerte vej tror jeg....

hyp = -(t1 + t2)     # svarer til cint1e_hyp_sph
print(hyp)