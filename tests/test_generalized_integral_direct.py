import numpy as np
import pytest
from pyscf import gto, scf

from slowquant.unitary_coupled_cluster.generalized_density_matrix import (
    get_electronic_energy_generalized,
    get_orbital_gradient_generalized_real_imag,
)
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.integral_direct import GHFCholesky, GHFDirect
from slowquant.unitary_coupled_cluster.linear_response import generalized_naive
from slowquant.unitary_coupled_cluster.linear_response.generalized_direct import LinearResponseDirect

OPTS = {"n_layers": 1, "is_spin_conserving": False}


def h3():
    mol = gto.M(atom="H 0 0 0; H 0.1 0.2 0.74; H 0.9 0.1 0.3", basis="sto-3g", spin=1, verbose=0)
    mf = scf.GHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    return mol, np.array(mf.mo_coeff, dtype=complex)


def test_backends():
    mol = gto.M(atom="O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59", basis="6-31g", verbose=0)
    n = mol.nao
    rng = np.random.default_rng(0)
    C = [rng.normal(size=(2 * n, k)) + 1j * rng.normal(size=(2 * n, k)) for k in (3, 2, 4, 2)]
    eri = mol.intor("int2e")
    ref = sum(
        np.einsum("pi,qj,rk,sl,pqrs->ijkl", C[0][s : s + n].conj(), C[1][s : s + n], C[2][t : t + n].conj(), C[3][t : t + n], eri)
        for s in (0, n)
        for t in (0, n)
    )
    for incore in (256.0, 0.0):
        assert np.abs(GHFDirect(mol, incore_memory=incore).eri(*C) - ref).max() < 1e-10
    D = rng.normal(size=(2, 2 * n, 2 * n)) + 1j * rng.normal(size=(2, 2 * n, 2 * n))
    j0, k0 = GHFDirect(mol).jk(D)
    cd = GHFCholesky(mol, tol=1e-10)
    j1, k1 = cd.jk(D)
    assert np.abs(j0 - j1).max() < 1e-7 and np.abs(k0 - k1).max() < 1e-7
    assert np.abs(cd.eri(*C) - ref).max() < 1e-7


@pytest.mark.parametrize("extra", [{}, {"do_x2c": True}, {"B": [0.0, 500.0, 1000.0]}])
def test_energy_and_gradients_vs_mo_integrals(extra):
    mol, C = h3()
    wf = GeneralizedWaveFunctionUPS(((2, 1), 6), C, mol, "fUCCSD", ansatz_options=OPTS, include_active_kappa=True, **extra)
    rng = np.random.default_rng(5)
    n = len(wf.thetas)
    wf.set_thetas(rng.uniform(-0.3, 0.3, n).tolist(), rng.uniform(-0.3, 0.3, n).tolist())
    nk = len(wf.kappa_spin_idx)
    p = np.concatenate((rng.uniform(-0.05, 0.05, 2 * nk), wf.thetas_real, wf.thetas_imag))
    wf._old_opt_parameters = np.zeros_like(p) + 1e20
    e = wf._calc_energy_optimization(p, True, True)
    g = wf._calc_gradient_optimization(p, True, True)
    nI, nA = wf.num_inactive_spin_orbs, wf.num_active_spin_orbs
    e_ref = get_electronic_energy_generalized(wf.h_mo, wf.g_mo, nI, nA, wf.rdm1, wf.rdm2)
    g_ref = get_orbital_gradient_generalized_real_imag(wf.h_mo, wf.g_mo, wf.kappa_spin_idx, nI, nA, wf.rdm1, wf.rdm2)
    assert abs(e - e_ref.real) < 1e-10
    assert np.abs(g[: 2 * nk] - g_ref).max() < 1e-10
    assert abs(wf.energy_elec - e) < 1e-10


def test_oo_h2():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="631-g", verbose=0)
    mf = scf.GHF(mol)
    mf.kernel()
    C = np.array(mf.mo_coeff, dtype=complex)
    for backend in (None, GHFCholesky(mol, tol=1e-10, cd_jk=False, screen_tol=1e-12)):
        wf = GeneralizedWaveFunctionUPS(((1, 1), 4), C, mol, "fUCCSD", ansatz_options=OPTS, include_active_kappa=True)
        if backend is not None:
            wf.set_integral_backend(backend)
        wf.run_wf_optimization_direct(tol=1e-7)
        assert abs(wf.energy_elec - -1.8613387609037015) < 1e-8


def test_lr_vs_naive():
    mol = gto.M(atom="H 0 0 0; H 0.1 0 0.8; H 0 1.2 0.8; H 0.2 1.1 -0.1", basis="sto-3g", verbose=0)
    mf = scf.GHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    C = np.array(mf.mo_coeff, dtype=complex)
    wf = GeneralizedWaveFunctionUPS(((2, 2), 6), C, mol, "fUCCSD", ansatz_options=OPTS, include_active_kappa=True)
    wf.run_wf_optimization_direct(tol=1e-9)
    lr0 = generalized_naive.LinearResponse(wf, excitations="SD")
    lr0.calc_excitation_energies()
    lr = LinearResponseDirect(wf, "SD")
    assert np.abs(lr.DB).max() > 1e-4
    E2, S = lr.explicit()
    assert np.abs(E2 - lr0.hessian).max() < 1e-8
    assert np.abs(S - lr0.metric).max() < 1e-8
    w = lr.calc_excitation_energies(6, tol=1e-8)
    assert np.abs(w - lr0.excitation_energies[:6]).max() < 1e-7
    f0 = lr0.get_oscillator_strength(mol.intor("int1e_r", comp=3))
    assert np.abs(lr.get_oscillator_strength() - f0[:6]).max() < 1e-7
    V = lr0.get_property_gradient(mol.intor_symmetric("int1e_r"))
    a0 = (V.conj().T @ np.linalg.solve(lr0.hessian, V)).real
    assert np.abs(lr.get_polarizability(tol=1e-10) - a0).max() < 1e-6
    lr_plain = LinearResponseDirect(wf, "SD", symmetrize_qG=False)
    assert np.abs(lr_plain.DA).max() == 0.0


def test_ups_engine_matches_reference_algebra():
    from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import (
        generalized_construct_ups_state_test_erik,
        generalized_get_grad_action,
    )

    mol, C = h3()
    for ans in ("fUCCSD", "fUCCSDT", "gtups"):
        wf = GeneralizedWaveFunctionUPS(((2, 1), 6), C, mol, ans, ansatz_options={"n_layers": 1, "is_spin_conserving": False}, include_active_kappa=True)
        rng = np.random.default_rng(1)
        n = len(wf.thetas)
        th = rng.uniform(-0.5, 0.5, n) + 1j * rng.uniform(-0.5, 0.5, n)
        s0 = generalized_construct_ups_state_test_erik(wf.csf_coeffs, wf.ci_info, list(th), wf.ups_layout)
        assert np.abs(wf.ups_engine.state(wf.csf_coeffs, th) - s0).max() < 1e-12
        for k in (0, n // 2, n - 1):
            a = generalized_get_grad_action(s0, k, wf.ci_info, list(th), wf.ups_layout)
            b = wf.ups_engine.grad_action(k, th[k], s0)
            assert max(np.abs(a[0] - b[0]).max(), np.abs(a[1] - b[1]).max()) < 1e-12


def test_onestep_oo_matches_twostep():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.5; H 0 0 3.0; H 0 0 4.5", basis="6-31g", verbose=0)
    mf = scf.GHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    e = []
    for name in ("run_wf_optimization_direct", "run_wf_optimization_onestep_direct"):
        wf = GeneralizedWaveFunctionUPS(
            ((2, 2), 8), np.array(mf.mo_coeff, dtype=complex), mol, "fUCCSD", ansatz_options=dict(OPTS),
            include_active_kappa=True,
        )
        getattr(wf, name)(tol=1e-7, verbose=False)
        e.append(wf.energy_elec)
    assert abs(e[0] - e[1]) < 1e-9
