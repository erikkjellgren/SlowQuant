import numpy as np
import pyscf
import pytest
from pyscf import scf
from qiskit.quantum_info import Statevector
from qiskit_aer.primitives import Sampler
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp

from slowquant.qiskit_interface.fast_pauli import FermionPauliMap, PauliEstimator, fermionic_to_sparse_pauli
from slowquant.qiskit_interface.generalized_circuit_wavefunction_DHF import GeneralizedWaveFunctionCircuitDHF
from slowquant.qiskit_interface.generalized_interface import QuantumInterface
from slowquant.qiskit_interface.linear_response.generalized_direct_DHF_circuit import (
    LinearResponseCircuitDirect,
)
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction_DHF import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response.generalized_direct_DHF import LinearResponseDirect
from slowquant.unitary_coupled_cluster.operators import a_op_spin


def make_qi():
    return QuantumInterface(
        Sampler(), "fUCCSD", JordanWignerMapper(), ansatz_options={"n_layers": 1, "is_spin_conserving": False}
    )


_LIH = {}


def lih():
    if _LIH:
        return _LIH["mol"], _LIH["C"]
    mol = pyscf.M(
        atom="Li 0 0 0; H 0.1 0.2 1.5", basis="sto-3g", unit="angstrom", nucmod="gaussian", verbose=0
    )
    mf = scf.dhf.DHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    _LIH.update(mol=mol, C=np.array(mf.mo_coeff, dtype=complex))
    return mol, _LIH["C"]


def random_wf(exact=None):
    mol, C = lih()
    wf = GeneralizedWaveFunctionCircuitDHF(
        ((1, 1), 4), C, mol, make_qi(), include_active_kappa=True, exact=exact
    )
    rng = np.random.default_rng(3)
    n = len(wf.thetas) // 2
    wf.set_thetas(rng.uniform(0.0, 0.3, n), rng.uniform(-1, 1, n))
    return wf


def test_fast_jw_and_rdms():
    na = 6
    QI = make_qi()
    QI.construct_circuit([0, 1], [2, 3, 4, 5], na, (1, 1))
    rng = np.random.default_rng(1)
    QI.parameters = rng.uniform(-0.6, 0.6, QI.circuit.num_parameters).tolist()
    H = FermionicOperator({(): 0.3})
    for _ in range(40):
        t, u, v, w = (int(i) for i in rng.integers(0, na, 4))
        c = complex(rng.normal(), rng.normal())
        H += c * a_op_spin(t, True) * a_op_spin(v, True) * a_op_spin(w, False) * a_op_spin(u, False)
        H += c * a_op_spin(t, True) * a_op_spin(u, False)
    ref = JordanWignerMapper().map(FermionicOp(H.get_qiskit_form(na // 2), na))
    assert np.abs((ref - fermionic_to_sparse_pauli(H, na // 2, na)).simplify(atol=1e-12).coeffs).max() < 1e-12
    pm = FermionPauliMap(QI, na)
    sv = Statevector(QI.circuit.assign_parameters(QI.parameters))
    for exact in (True, False):
        d1, d2 = pm.rdms(PauliEstimator(QI, exact)(pm.keys))
        for p, q, r, s in ((0, 1, 2, 3), (1, 0, 0, 1), (5, 0, 1, 2)):
            op = a_op_spin(p, True) * a_op_spin(r, True) * a_op_spin(s, False) * a_op_spin(q, False)
            assert abs(sv.expectation_value(QI.op_to_qbit(op)) - d2[p, q, r, s]) < 1e-12
        assert (
            abs(sv.expectation_value(QI.op_to_qbit(a_op_spin(4, True) * a_op_spin(1, False))) - d1[4, 1])
            < 1e-12
        )


def test_parameter_shift_gradient_and_hessian():
    wf = random_wf()
    p = np.array(wf.thetas)
    e = np.eye(len(p))
    g_fd = np.array([(wf.theta_energy(p + 1e-5 * v) - wf.theta_energy(p - 1e-5 * v)) / 2e-5 for v in e])
    assert np.abs(wf.theta_gradient(p) - g_fd).max() < 1e-8
    H_fd = np.array([(wf.theta_gradient(p + 1e-4 * v) - wf.theta_gradient(p - 1e-4 * v)) / 2e-4 for v in e])
    assert np.abs(wf.theta_hessian(p) - H_fd).max() < 1e-6
    assert abs(wf.energy_elec - wf.theta_energy(p)) < 1e-12


def test_sampler_path_matches_exact():
    a, b = random_wf(True), random_wf(False)
    assert np.abs(a.theta_gradient() - b.theta_gradient()).max() < 1e-12
    assert np.abs(a.rdm2 - b.rdm2).max() < 1e-12


@pytest.mark.slow
def test_circuit_oo_and_linear_response_match_classical():
    mol, C = lih()
    ref = GeneralizedWaveFunctionUPS(
        ((1, 1), 4),
        C,
        mol,
        [],
        False,
        "fUCCSD",
        {"n_layers": 1, "is_spin_conserving": False},
        include_active_kappa=True,
    )
    ref.run_wf_optimization_direct_DHF(tol=1e-9)
    wf = GeneralizedWaveFunctionCircuitDHF(((1, 1), 4), C, mol, make_qi(), include_active_kappa=True)
    wf.run_wf_optimization_2step_DHF(tol=1e-7)
    assert abs(wf.energy_elec - ref.energy_elec) < 1e-10
    wf = GeneralizedWaveFunctionCircuitDHF(((1, 1), 4), ref.c_mo, mol, make_qi(), include_active_kappa=True)
    wf._optimize_thetas(1e-12, 2000)
    r0 = LinearResponseDirect(ref, "SD").get_properties(tol=1e-8)
    r1 = LinearResponseCircuitDirect(wf, "SD").get_properties(tol=1e-8)
    assert np.abs(r1["shieldings"] - r0["shieldings"]).max() < 1e-4
    assert abs(r1["sscc"][0, 1] - r0["sscc"][0, 1]) < 1e-4


def test_batched_simulator_matches_qiskit():
    from qiskit.quantum_info import Statevector

    from slowquant.qiskit_interface.fast_pauli import CircuitSimulator

    QI = make_qi()
    QI.construct_circuit([0, 1, 2], [3, 4, 5], 6, (2, 1))
    P = np.random.default_rng(2).uniform(-1, 1, (3, QI.circuit.num_parameters))
    S = CircuitSimulator(QI.circuit).statevectors(P)
    for p, s in zip(P, S):
        assert np.abs(Statevector(QI.circuit.assign_parameters(p)).data - s).max() < 1e-12


@pytest.mark.slow
def test_circuit_lr_symmetrize_qG_matches_classical():
    mol = pyscf.M(
        atom="H 0 0 0; H 0.1 0.2 0.74; H 0.9 0.1 0.3", basis="6-31g", spin=1, nucmod="gaussian", verbose=0
    )
    mf = scf.dhf.DHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    ref = GeneralizedWaveFunctionUPS(
        ((2, 1), 6),
        np.array(mf.mo_coeff, dtype=complex),
        mol,
        [],
        False,
        "fUCCSD",
        {"n_layers": 1, "is_spin_conserving": False},
        include_active_kappa=True,
    )
    ref.run_wf_optimization_direct_DHF(tol=1e-9)
    wf = GeneralizedWaveFunctionCircuitDHF(((2, 1), 6), ref.c_mo, mol, make_qi(), include_active_kappa=True)
    wf._optimize_thetas(1e-12, 3000)
    for sym in (False, True):
        r0 = LinearResponseDirect(ref, "SD", symmetrize_qG=sym)
        r1 = LinearResponseCircuitDirect(wf, "SD", symmetrize_qG=sym)
        E0, _ = r0.explicit()
        E1, _ = r1.explicit()
        assert np.abs(E0 - E1).max() < 1e-5 * np.abs(E0).max()
        assert np.abs(r0.DB - r1.DB).max() < 1e-5
        w0 = r0.calc_excitation_energies(4, tol=1e-6)
        w1 = r1.calc_excitation_energies(4, tol=1e-6)
        assert np.abs(w0 - w1).max() < 1e-6


def test_adjoint_gradient_and_hessian_match_parameter_shift():
    wf = random_wf(True)
    p = np.array(wf.thetas)
    g, H = wf.theta_gradient(p), wf.theta_hessian(p)
    wf.parameter_shift = True
    assert np.abs(g - wf.theta_gradient(p)).max() < 1e-12
    assert np.abs(H - wf.theta_hessian(p)).max() < 1e-10
