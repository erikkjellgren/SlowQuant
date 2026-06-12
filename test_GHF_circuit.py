import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
from scipy.stats import unitary_group
from pyscf.lib import chkfile
from scipy.linalg import expm
import matplotlib


# from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response import generalized_naive
from slowquant.unitary_coupled_cluster.operator_state_algebra import expectation_value
from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import generalized_expectation_value_energy
from slowquant.unitary_coupled_cluster.generalized_operators import generalized_hamiltonian_full_space, generalized_hamiltonian_0i_0a, generalized_hamiltonian_1i_1a
from slowquant.unitary_coupled_cluster.generalized_density_matrix import get_orbital_gradient_generalized_real_imag, get_orbital_gradient_expvalue_real_imag, get_nonsplit_gradient_expvalue, get_gradient_finite_diff, get_electronic_energy_generalized

from slowquant.unitary_coupled_cluster.fermionic_operator import (
    FermionicOperator, 
)

from slowquant.molecularintegrals.integralfunctions import DHF_one_electron_transform, DHF_two_electron_transform

# qWF imports:
from qiskit_aer.primitives import Sampler
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, InterleavedQubitMapper
from slowquant.qiskit_interface.generalized_circuit_wavefunction import GeneralizedWaveFunctionCircuit
from slowquant.qiskit_interface.generalized_interface import QuantumInterface
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import SparsePauliOp

# The customized mappper:     conserves orbital ordering from PySCF
class DirectJordanWignerMapper(JordanWignerMapper):
    """
    JW mapper that preserves spinor index ordering exactly.
    Bypasses Qiskit Nature's alpha/beta blocked reordering.
    """

    def map(self, op: FermionicOp, *, register_length=None) -> SparsePauliOp:
        n = op.num_spin_orbitals
        result = SparsePauliOp.from_list([('I' * n, 0.0)])

        for term, coeff in op.items():
            if not term:  # empty string = identity
                result += SparsePauliOp.from_list([('I' * n, coeff)])
                continue
            pauli = self._jw_term(term, n)
            result += coeff * pauli

        return result.simplify()

    def _jw_term(self, term, n):
        pauli = SparsePauliOp.from_list([('I' * n, 1.0)])
        for op_str in term.split(' '):
            action, idx = op_str.split('_')
            pauli = pauli @ self._jw_single(action, int(idx), n)  # no permutation
        return pauli

    def _jw_single(self, action, idx, n):
        """a†_idx or a_idx via JW, direct index."""
        z_string = 'I' * (n - idx - 1) + 'Z' * idx  # Z on all qubits below idx
        if action == '+':
            x_part = 'I' * (n - idx - 1) + 'X' + z_string[n - idx:]
            y_part = 'I' * (n - idx - 1) + 'Y' + z_string[n - idx:]
            # a†_i = 0.5 * (X - iY) * Z...Z
            return SparsePauliOp.from_list([(
                'I' * (n - idx - 1) + 'X' + 'Z' * idx, 0.5),
                ('I' * (n - idx - 1) + 'Y' + 'Z' * idx, -0.5j)
            ])
        else:  # '-'
            return SparsePauliOp.from_list([(
                'I' * (n - idx - 1) + 'X' + 'Z' * idx, 0.5),
                ('I' * (n - idx - 1) + 'Y' + 'Z' * idx, 0.5j)
            ])
        
        
class DirectJordanWignerMapper_new(JordanWignerMapper):
    """
    Jordan–Wigner mapper preserving the exact PySCF spin-orbital ordering.

    No alpha/beta regrouping.
    Orbital i -> qubit i directly.
    """

    def map(self, op: FermionicOp, *, register_length=None) -> SparsePauliOp:
        n = op.num_spin_orbitals

        result = SparsePauliOp.from_list([("I" * n, 0.0)])

        for term, coeff in op.items():

            # identity term
            if term == "":
                result += SparsePauliOp.from_list([("I" * n, coeff)])
                continue

            mapped = self._map_term(term, n)
            result += coeff * mapped

        return result.simplify()

    def _map_term(self, term: str, n: int) -> SparsePauliOp:
        """
        Map a product like '+_0 -_1'.
        """
        op = SparsePauliOp.from_list([("I" * n, 1.0)])

        for tok in term.split():
            action, idx = tok.split("_")
            idx = int(idx)

            op = op @ self._single_jw(action, idx, n)

        return op

    def _single_jw(self, action: str, idx: int, n: int) -> SparsePauliOp:
        """
        Direct JW transform for one fermionic operator.

        a†_i = 1/2 (X_i - iY_i) Z_0 ... Z_{i-1}
        a_i  = 1/2 (X_i + iY_i) Z_0 ... Z_{i-1}

        Qiskit Pauli strings are little-endian:
        rightmost char = qubit 0
        """

        x_label = ["I"] * n
        y_label = ["I"] * n

        # parity string on lower qubits
        for q in range(idx):
            x_label[n - 1 - q] = "Z"
            y_label[n - 1 - q] = "Z"

        # operator on target qubit
        x_label[n - 1 - idx] = "X"
        y_label[n - 1 - idx] = "Y"

        x_label = "".join(x_label)
        y_label = "".join(y_label)

        if action == "+":
            return SparsePauliOp.from_list([
                (x_label, 0.5),
                (y_label, -0.5j),
            ])

        elif action == "-":
            return SparsePauliOp.from_list([
                (x_label, 0.5),
                (y_label, 0.5j),
            ])

        else:
            raise ValueError(f"Invalid fermionic action: {action}")


def NR(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """.........."""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()
    

    mf = scf.GHF(mol)
    mf.conv_tol = 1e-8        # Energy convergence (Hartree)
    mf.conv_tol_grad = 1e-8   # Optional: gradient convergence
    mf.max_cycle = 1000

    mf.scf()
    mf.kernel()


    coeff = np.array(mf.mo_coeff,dtype=complex)



    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")
   

    # small random anti-Hermitian
    eps = 0.1  # controls "step size"
    X_anti = np.random.randn(coeff.shape[0],coeff.shape[0]) + 1j*np.random.randn(coeff.shape[0],coeff.shape[0])
    A_mat = eps * (X_anti - X_anti.conj().T)/2  # make anti-Hermitian

    U_step = expm(A_mat)

    coeff_u = coeff @ U_step


    #print(np.round(coeff.real, 2))

    spin_cons = False


    WF = GeneralizedWaveFunctionUPS(
        active_space,
        coeff_u,
        mol,
        "fUCCD",
        ansatz_options = {"n_layers": 0, "is_spin_conserving" : spin_cons},
        include_active_kappa=True,
    )

    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True)

    WF2 = GeneralizedWaveFunctionUPS(
        active_space,
        WF.c_mo,
        mol,
        "fUCCSD",
        ansatz_options = {"n_layers": 1, "is_spin_conserving" : spin_cons},
        include_active_kappa=True,
    )

    WF3 = GeneralizedWaveFunctionUPS(
        active_space,
        WF.c_mo,
        mol,
        "fUCCSD",
        ansatz_options = {"n_layers": 1, "is_spin_conserving" : spin_cons},
        include_active_kappa=True,
    )

    WF2.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True)

    WF3.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True)

    # Setting the correct mapper:    The problem is that we do NOT have perfectly interleaved orbital ordering
    # base_mapper = JordanWignerMapper()
    # mapper = InterleavedQubitMapper(base_mapper)
    # mapper = DirectJordanWignerMapper_new()
    mapper = JordanWignerMapper()


    QI = QuantumInterface(
        Sampler(run_options={"shots": None}),
        "fUCCSD", # Ansatz
        mapper,
        ansatz_options = {"n_layers": 1, "is_spin_conserving" : spin_cons},
        ISA=False, # default is false
        do_M_mitigation=False, # default is false
        do_M_ansatz0=False, # default is false
        do_postselection=False, # default is false
    )

    qWF = GeneralizedWaveFunctionCircuit(
        mol.nelectron,
        active_space,
        WF3.c_mo,
        h_core,
        g_eri,
        QI,
        include_active_kappa = True,
    )

    #clean = [tuple(int(y) for y in x) for x in WF3.ups_layout.excitation_indices]
    #print("idx from classical excitaion operators:", clean)
    

    #qWF.set_thetas_initial(WF2.thetas_real, WF2.thetas_imag)
    qWF.set_thetas_initial(WF3.thetas_real, WF3.thetas_imag)

    #qWF.set_thetas_initial(np.add(WF2.thetas_real, 0.002), np.add(WF2.thetas_imag, 0.002))

    print("real components of thetas     :", np.round(WF3.thetas_real,10))
    print("imaginary components of thetas:", np.round(WF3.thetas_imag,10))

    print("norm of thetas:", np.round(qWF.thetas_real,10))
    print("phi of thetas :", np.round(qWF.thetas_imag,10))


    print("HF Classical PySCF                :", mf.energy_elec()[0])

    print("HF Classical WF                   :", WF.energy_elec)

    print("oo-UCCSD Classical                :", WF2.energy_elec)

    print("oo-UCCSD Classical no active kappa:", WF3.energy_elec)

    print("oo-UCCSD Quantum                  :", qWF.energy_elec)

    
    fig = qWF.QI.circuit.draw("mpl")
    fig.savefig("circuit_GHF.png")

    #print("param_names:", qWF.QI.param_names)

    qWF.run_wf_optimization_1step("bfgs", orbital_optimization=True, tol=1e-6)

    print("norm of thetas:", np.round(qWF.thetas_real,10))
    print("phi of thetas :", np.round(qWF.thetas_imag,10))








def h2():
    geometry = """H  0.0   0.0  0.0;
        H  0.0  0.0  0.74"""
    #basis = "cc-pvdz"
    basis = "631-g"
    #basis = "sto-3g"
    #basis = "sto-6g"
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
    basis = "sto-6g"
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
        Li  0.0  0.0  1"""
    #basis = "cc-pvdz"
    #basis = "631-g"
    basis = "sto-3g"
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

def h2o():
    geometry = """
    O  0.0   0.0  0.11779 
    H  0.0   0.75545  -0.47116;
    H  0.0  -0.75545  -0.47116"""
    #basis = "dyall-v2z"
    #basis = "cc-pvdz"
    #basis = "631-g"
    basis = "sto-3g"
    #basis = "sto-6g"
    #active_space = ((5, 5), 14)
    active_space = ((2,2),6)
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
    basis = "dyall-v2z"
    active_space = (4, 6)
    charge = 0
    spin = 0

    print("Restricted HI")
    restricted(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    print("Nonrelativistic HI")
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def HBr():
    geometry = """H  0.0   0.0  0.0;
        Br  0.0  0.0  1.41443 """
    basis = "dyall-v2z"
    active_space = (4, 6)
    charge = 0
    spin = 0
    print("Restricted HBr")
    restricted(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
    print("Nonrelativistic HBr")
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

def HF():
    geometry = """H  0.0   0.0  0.91680;
        F  0.0  0.0  0.0 """
    basis = 'sto-3g'
    active_space = ((1,1), 4) #spin orbitaler or spinor basis
    # active_space = ((2,2), 6) #spin orbitaler or spinor basis
    # active_space = (2, 4)
    charge = 0
    spin = 1
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )  

# Run simulation:

h3()

