import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c, lib, gto
from scipy.stats import unitary_group
from pyscf.lib import chkfile
from scipy.linalg import expm
import basis_set_exchange as bse
import struct


# from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response import generalized_naive
from slowquant.unitary_coupled_cluster.operator_state_algebra import expectation_value
from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import generalized_expectation_value_energy
from slowquant.unitary_coupled_cluster.generalized_operators import generalized_hamiltonian_full_space, generalized_hamiltonian_0i_0a, generalized_hamiltonian_1i_1a, get_HcoreB
from slowquant.unitary_coupled_cluster.generalized_density_matrix import get_orbital_gradient_generalized_real_imag, get_orbital_gradient_expvalue_real_imag, get_nonsplit_gradient_expvalue, get_gradient_finite_diff, get_electronic_energy_generalized

from slowquant.unitary_coupled_cluster.fermionic_operator import (
    FermionicOperator, 
)

from slowquant.molecularintegrals.integralfunctions import DHF_one_electron_transform, DHF_two_electron_transform


# qWF imports:
from qiskit_aer.primitives import Sampler, SamplerV2
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, InterleavedQubitMapper
from slowquant.qiskit_interface.generalized_circuit_wavefunction import GeneralizedWaveFunctionCircuit
from slowquant.qiskit_interface.generalized_interface import QuantumInterface
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_aer.noise import NoiseModel



def NR(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """.........."""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin, cart = False)
    mol.build()

    mf = scf.GHF(mol)

    # Modyfing the Hcore
    # B in T
    B = np.array([.2, .5, .3]) ; B = (B * (2.3505 * 1e5) / np.linalg.norm(B).tolist()).tolist() ; orig = (0, 0, 0)

    # Integrals:
    hcore = mf.get_hcore().astype(complex)
    hcoreB_corr = get_HcoreB(mf, mol, B = B, orig = orig)
    g_eri = mol.intor("int2e")

    # qWF settings:
    mapper = JordanWignerMapper()
    backend = FakeTorino()
    sampler = SamplerV2(options={"backend_options":{"noise_model":NoiseModel.from_backend(backend)}})
    shots = 15000
    M0 = True
    post_select = False
    method = "fUCCSD"
    spin_consv = False
    active_k = True
    nl = 1

    # Load data:
    data = np.load("data_H7_UCCSD_03.npz")
    c_mo, thetas_real, thetas_imag = data["c_mo"], data["thetas_real"], data["thetas_imag"]

    QI = QuantumInterface(
        sampler,
        method, # Ansatz
        mapper,
        ansatz_options = {"n_layers": nl, "is_spin_conserving" : spin_consv},
        shots = shots,
        do_M_ansatz0=M0, # default is false
        do_postselection=post_select,
    )

    qWF = GeneralizedWaveFunctionCircuit(
        mol.nelectron,
        active_space,
        c_mo,
        hcore + hcoreB_corr,
        g_eri,
        QI,
        include_active_kappa = active_k,
    )

    QI.get_info()

    qWF.set_thetas_initial(thetas_real, thetas_imag)

    print("Depth of circuit:", qWF.QI.circuit.depth())

    for i in range(100):
        print(qWF.energy_elec)





def h2():
    geometry = """H  0.0   0.0  0;
                  H  0.0   0.0  0.74"""
    #basis = "cc-pvtz"
    basis = "631-g"
    #basis = "sto-3g"
    #basis = "sto-6g"
    dyall2zp_H = bse.get_basis('dyall-v2z', elements=['H'], fmt='nwchem')
    with open('dyall2zp_H.nwchem', 'w') as f:
        f.write(dyall2zp_H)
        f.close()
    #basis = {'H': gto.basis.load('dyall2zp_H.nwchem', 'H')}
    active_space = ((1, 1), 4)
    #active_space = (2, 4)
    charge = 0
    spin = 0

    # restricted(
    #     geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    # )
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    # unrestricted(
    #     geometry=geometry, basis=basis, active_space=active_space_u, charge=charge, spin=spin, unit="angstrom"
    # )

def h3():
    geometry = """H  0.000000   0.000000       0.000000;
                  H  1.000000   0.000000       0.000000;
                  H  0.500000   0.8660254038   0.000000"""
    #basis = "cc-pvdz"
    #basis = "631-g"
    basis = "def2-svp"
    #basis = ""
    active_space = ((2, 1), 6)
    #active_space = (2, 4)
    charge = 0
    spin = 1

    # restricted(
    #     geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    # )
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    # unrestricted(
    #     geometry=geometry, basis=basis, active_space=active_space_u, charge=charge, spin=spin, unit="angstrom"
    # )

def LiH():
    geometry = """H  0.0   0.0  0.0;
                  Li  0.0  0.0  1.595"""
    #basis = "cc-pvdz"
    #basis = "631-g"
    basis = "sto-3g"
    active_space = ((1, 1), 6)
    #active_space = (2, 4)
    charge = 0
    spin = 0

    # restricted(
    #     geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    # )
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    # unrestricted(
    #     geometry=geometry, basis=basis, active_space=active_space_u, charge=charge, spin=spin, unit="angstrom"
    # )

def HF():
    geometry = """H  0.0   0.0  0.0;
                  F  0.0  0.0  0.917"""
    #basis = "cc-pvdz"
    basis = "631-g"
    #basis = "sto-3g"
    #active_space = ((2, 2), 12)
    #active_space = ((2, 2), 10)
    #active_space = ((3,3), 8)
    active_space = ((2, 2), 6)
    #active_space = ((1,1,), 4)
    charge = 0
    spin = 0

    # restricted(
    #     geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    # )
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    # unrestricted(
    #     geometry=geometry, basis=basis, active_space=active_space_u, charge=charge, spin=spin, unit="angstrom"
    # )


def h2o():
    geometry = """
    O  0.0   0.0  0.11779 
    H  0.0   0.75545  -0.47116;
    H  0.0  -0.75545  -0.47116"""
    all_bases = bse.get_all_basis_names()
    dyall = [b for b in all_bases if 'dyall' in b.lower()]
    #print(dyall)
    dyall2zp_H = bse.get_basis('dyall-v2z', elements=['H'], fmt='nwchem')
    dyall2zp_O = bse.get_basis('dyall-v2z', elements=['O'], fmt='nwchem')
    with open('dyall2zp_H.nwchem', 'w') as f:
        f.write(dyall2zp_H)
        f.close()
    with open('dyall2zp_O.nwchem', 'w') as f:
        f.write(dyall2zp_O)
        f.close()
    #basis = {'H': gto.basis.load('dyall2zp_H.nwchem', 'H'),'O': gto.basis.load('dyall2zp_O.nwchem', 'O')}
    #basis = "dyallv2z"
    #basis = "cc-pvdz"
    basis = "631-g"
    #basis = "sto-3g"
    #basis = "sto-6g"
    #active_space = ((5, 5), 14)
    active_space = ((2,2),8)
    charge = 0
    spin = 0

    # restricted(
    #     geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    # )
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    # unrestricted(
    #     geometry=geometry, basis=basis, active_space=active_space_u, charge=charge, spin=spin, unit="angstrom"
    # )

def HI():
    geometry = """H  0.0   0.0  0.0;
        I  0.0  0.0  1.60916 """
    #basis = "dyall-v2z"
    basis = "cc-pvdz"
    active_space = (4, 6)
    charge = 0
    spin = 0

    #print("Restricted HI")
    #restricted(
    #    geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    #)
    print("Nonrelativistic HI")
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def HCl():
    geometry = """H  0.0   0.0  0.0;
        Cl  0.0  0.0  1.1275 """
    #basis = "dyall-v2z"
    #basis = "cc-pvdz"
    basis = "sto-3g"
    active_space = ((3,3), 8)
    charge = 0
    spin = 0
    #print("Restricted HBr")
    #restricted(
    #    geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    #)
    #print("Nonrelativistic HBr")
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom",
    )

def HBr():
    geometry = """H  0.0   0.0  0.0;
        Br  0.0  0.0  1.41443 """
    #basis = "dyall-v2z"
    #basis = "cc-pvdz"
    basis = "sto-3g"
    active_space = ((18,18), 38)
    charge = 0
    spin = 0
    #print("Restricted HBr")
    #restricted(
    #    geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    #)
    #print("Nonrelativistic HBr")
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom",
    )

def BeH():
    geometry = """Be 0.0 0.0 0.0;
                  H 0.0 0.0 1.3426"""
    basis="sto-3g"
    active_space = ((1,2),6)
    charge = 0
    spin = 1
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom",
    )
  

###SPIN ELLER RUMLIGE ORBITALER###

#h2o()
#LiH()
h3()
#h2o()
# HI()
# HBr()