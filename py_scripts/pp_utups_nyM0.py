import numpy as np
import pickle
import pyscf
from pyscf import scf, mcscf
import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from qiskit_aer.primitives import Sampler
from qiskit_nature.second_q.mappers import JordanWignerMapper
from slowquant.unitary_coupled_cluster.unrestricted_operators import one_elec_op_0i_0a_HFC
from slowquant.unitary_coupled_cluster.operator_state_algebra import expectation_value
from slowquant.qiskit_interface.interface import FermionicMapper, FermionicOperator, FermionicOp

name = "0.04020293945033737"

mol = pyscf.M(atom="N   0.0  0.0    0.0; H   0.0  0.0 1.0362", basis="631gJ.nw", unit="angstrom", spin=2)
uhf = pyscf.scf.UHF(mol).run()
mc = mcscf.UCASCI(uhf, (2,0), 4)

c_mo_a = np.load(f"{name}_a_mo.npy")
c_mo_b = np.load(f"{name}_b_mo.npy")

c_mo = np.stack([c_mo_a, c_mo_b], axis=0)

WF = UnrestrictedWaveFunctionUPS(
    ((2,0),4),
    c_mo,
    mol,
    ansatz = "utups",
    ansatz_options= {"n_layers":2},
    include_active_kappa=True,
)

WF.thetas = np.load(f"{name}_thetas.npy")

WF.run_wf_optimization_1step("bfgs", orbital_optimization=True, tol=1e-8, maxiter=5000)

print("")
print("E0:", WF.energy_elec_RDM)

print("\n")
print("name", name)



mapper = JordanWignerMapper()

list_h1mo_a = []
list_h1mo_b = []

for atom in mol._atom:
    atom_name = atom[0]
    print("\n")
    print("atom name", atom_name)
    amp_basis = mol.eval_gto("GTOval_sph", coords=[atom[1]])[0]
    mo_basis_a = amp_basis@WF.c_a_mo
    mo_basis_b = amp_basis@WF.c_b_mo
    h1mo_a = np.outer(np.conj(mo_basis_a), mo_basis_a)[:WF.num_inactive_orbs + WF.num_active_orbs, :WF.num_inactive_orbs + WF.num_active_orbs]
    h1mo_b = np.outer(np.conj(mo_basis_b), mo_basis_b)[:WF.num_inactive_orbs + WF.num_active_orbs, :WF.num_inactive_orbs + WF.num_active_orbs]
    list_h1mo_a.append(h1mo_a)
    list_h1mo_b.append(h1mo_b)

    operator = one_elec_op_0i_0a_HFC(h1mo_a, h1mo_b, num_inactive_orbs=WF.num_inactive_orbs, num_active_orbs=WF.num_active_orbs)
    operator_active = operator.get_folded_operator(WF.num_inactive_orbs, WF.num_active_orbs, WF. num_virtual_orbs)

    reference_atom = expectation_value(WF.ci_coeffs, [operator_active], WF.ci_coeffs, WF.ci_info, do_folding=False)-operator_active.operators[()]
    print("\n")
    print("Atom specific reference", reference_atom)

    operator_active.operators.pop(())
    print("\n")
    print("Atom specific operator (fermionic notation)", operator_active.operators_readable)
    mapped_op = mapper.map(FermionicOp(operator_active.get_qiskit_form(WF.num_active_orbs), WF.num_active_spin_orbs))
    
    t_mo_operator = {}
    for c, p in zip(mapped_op.coeffs, mapped_op.paulis):
        t_mo_operator[p.to_label()] = c.real
    print("\n")
    print("Atom specific operator (Jordan-wigner)",t_mo_operator)
    with open(f"{atom_name}_spinrdmelementoperator", "wb") as f:
        pickle.dump(t_mo_operator, f)


h1mo_a = np.maximum(list_h1mo_a[0], list_h1mo_a[1])
h1mo_b = np.maximum(list_h1mo_b[0], list_h1mo_b[1])

operator = one_elec_op_0i_0a_HFC(h1mo_a, h1mo_b, num_inactive_orbs=WF.num_inactive_orbs, num_active_orbs=WF.num_active_orbs)
operator_active = operator.get_folded_operator(WF.num_inactive_orbs, WF.num_active_orbs, WF.num_virtual_orbs)

reference = expectation_value(WF.ci_coeffs, [operator_active], WF.ci_coeffs, WF.ci_info, do_folding=False)-operator_active.operators[()]
np.save(f"{name}_ref_val.npy", reference)
print("\n")
print("reference value", reference)

operator_active.operators.pop(())
print("Active operator (fermionic notation)", operator_active.operators_readable)
mapped_op = mapper.map(FermionicOp(operator_active.get_qiskit_form(WF.num_active_orbs), WF.num_active_spin_orbs))
t_mo_operator = {}
for c, p in zip(mapped_op.coeffs, mapped_op.paulis):
    t_mo_operator[p.to_label()] = c.real
print("\n")
print("Active operator (Jordan-wigner mapped)", t_mo_operator)
with open(f"{name}_operator_active", "wb") as f:
    pickle.dump(t_mo_operator, f)