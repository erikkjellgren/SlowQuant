import numpy as np
import pyscf
import pytest
from pyscf import scf

from slowquant.unitary_coupled_cluster.ci_sparse_DHF import ActiveCI
from slowquant.unitary_coupled_cluster.DHF_integral_direct import DHFCholesky, DHFDirect, giao_integrals
from slowquant.unitary_coupled_cluster.generalized_density_matrix_DHF import (
    RDM1,
    RDM2,
    get_electronic_energy_generalized,
    get_orbital_gradient_generalized_real_imag,
)
from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import generalized_propagate_state
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction_DHF import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response.generalized_direct_DHF import LinearResponseDirect
from slowquant.unitary_coupled_cluster.operators import a_op_spin


def make_wf(cas, seed=3, scale=0.05):
    mol = pyscf.M(atom="Li 0 0 0; H 0.1 0.2 1.5", basis="sto-3g", unit="angstrom", nucmod="gaussian", verbose=0)
    mf = scf.dhf.DHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    wf = GeneralizedWaveFunctionUPS(
        cas, np.array(mf.mo_coeff, dtype=complex), mol, [], False, "fUCCSD",
        {"n_layers": 1, "is_spin_conserving": False}, include_active_kappa=True,
    )
    rng = np.random.default_rng(seed)
    if len(wf.thetas):
        wf.set_thetas(rng.uniform(-0.1, 0.1, len(wf.thetas)), rng.uniform(-0.1, 0.1, len(wf.thetas)))
    if scale:
        nk = len(wf.kappa_real)
        wf.set_kappa_cep(rng.uniform(-scale, scale, nk), rng.uniform(-scale, scale, nk))
        wf._set_orbitals(wf.c_mo)
    return mol, wf


def test_energy_and_orbital_gradient():
    _, wf = make_wf(((1, 1), 4))
    args = (wf.num_spin_orbs_NES, wf.num_inactive_spin_orbs, wf.num_active_spin_orbs, wf.rdm1, wf.rdm2)
    e_ref = get_electronic_energy_generalized(wf.h_mo, wf.g_mo, *args)
    g_ref = get_orbital_gradient_generalized_real_imag(wf.h_mo, wf.g_mo, wf.kappa_spin_idx, *args)
    assert abs(wf.energy_elec - e_ref.real) < 1e-10
    g = wf._calc_gradient_optimization_DHF([0.0] * 2 * len(wf.kappa_real), False, True, False)
    assert np.abs(g - g_ref).max() < 1e-9


def test_active_ci_engine():
    _, wf = make_wf(((2, 2), 6))
    ci = ActiveCI(wf.ci_info)
    A = wf.active_spin_idx
    c = np.random.default_rng(0).normal(size=ci.ndet) + 0j
    for t, u in ((0, 3), (2, 1), (4, 4)):
        ref = generalized_propagate_state([a_op_spin(A[t], True) * a_op_spin(A[u], False)], c, wf.ci_info)
        assert np.abs(ci.E[t][u] @ c - ref).max() < 1e-12
    H = ci.hamiltonian(*wf.ds.active_integrals())
    assert np.abs(H @ c - generalized_propagate_state([wf.ds.hamiltonian()], c, wf.ci_info)).max() < 1e-10


def test_transformed_operator():
    _, wf = make_wf(((1, 1), 4))
    ds = wf.ds
    ds.set_rdms(wf.rdm1, wf.rdm2)
    h, g = wf.h_mo, wf.g_mo
    N, A, occ = h.shape[0], ds.A, ds.occ
    nes, ni, na = wf.num_spin_orbs_NES, wf.num_inactive_spin_orbs, wf.num_active_spin_orbs
    D = np.zeros((N, N), complex)
    G = np.zeros((N,) * 4, complex)
    for p in occ:
        for q in occ:
            D[p, q] = RDM1(p, q, nes, ni, na, wf.rdm1)
            for r in occ:
                for s in occ:
                    G[p, q, r, s] = RDM2(p, q, r, s, nes, ni, na, wf.rdm1, wf.rdm2)
    rng = np.random.default_rng(5)
    Am = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    Bm = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    ht = Am @ h + h @ Bm
    gt = (np.einsum("px,xqrs->pqrs", Am, g) + np.einsum("pxrs,xq->pqrs", g, Bm)
          + np.einsum("rx,pqxs->pqrs", Am, g) + np.einsum("pqrx,xs->pqrs", g, Bm))
    F_ref = (ht @ D.T + np.einsum("npqr,mpqr->nm", gt, G))[:, occ]
    v = ds.direct.veff(np.array(ds.transformed_densities(Am, Bm)))
    F, _, gA = ds.transformed_operator(Am, Bm, v[0], v[1])
    assert np.abs(F - F_ref).max() < 1e-10 * np.abs(F_ref).max()
    assert np.abs(gA - gt[np.ix_(A, A, A, A)]).max() < 1e-12


def test_giao_integrals_match_g_B():
    mol, wf = make_wf(((1, 1), 4))
    n2c = mol.nao_2c()
    n = 2 * n2c
    L, S = slice(0, n2c), slice(n2c, n)
    gB = wf.int_gen.g_B
    G = np.zeros((3, n, n, n, n), complex)
    for x in range(3):
        G[x][L, L, L, L], G[x][S, S, S, S], G[x][L, L, S, S], G[x][S, S, L, L] = gB[x]
    rng = np.random.default_rng(1)
    D = rng.normal(size=(2, n, n)) + 1j * rng.normal(size=(2, n, n))
    C = wf.c_mo
    CA = C[:, wf.active_spin_idx]
    W, g = giao_integrals(wf._direct, C, CA, D, max_memory=1)
    W_ref = np.einsum("xpqrs,nsr->nxpq", G, D) - np.einsum("xpsrq,nsr->nxpq", G, D)
    g_ref = np.einsum("xmnls,mN,nu,lv,sw->xNuvw", G, C.conj(), CA, CA.conj(), CA, optimize=True)
    assert np.abs(W - W_ref).max() < 1e-12
    assert np.abs(g - g_ref).max() < 1e-12


def test_hessian_is_energy_second_derivative():
    _, wf = make_wf(((1, 1), 4), scale=0.0)
    lr = LinearResponseDirect(wf, "SD", screen=-1.0)
    E2, S = lr.explicit()
    assert np.abs(E2 - E2.conj().T).max() < 1e-6 * np.abs(E2).max()
    assert np.abs(S - S.conj().T).max() < 1e-12


def test_dhf_limit_sscc_matches_pyscf():
    from pyscf.data import nist
    from pyscf.prop.ssc.dhf import SSC

    _, wf = make_wf(((1, 1), 2), scale=0.0)
    lr = LinearResponseDirect(wf, "SD")
    mf = scf.dhf.DHF(wf.int_gen.int_obj)
    mf.conv_tol = 1e-11
    mf.kernel()
    ssc = SSC(mf)
    ssc.cphf, ssc.mb, ssc.conv_tol = True, "RKB", 1e-10
    e11 = ssc.kernel()
    k_ref = nist.HARTREE2J / nist.PLANCK * (0.5 * nist.E_MASS / nist.PROTON_MASS) ** 2 * np.trace(e11[0]) / 3
    k = lr.get_SSCC_4comp_iso(tol=1e-6)
    assert abs(k[0, 1] - k_ref) < 1e-6 * abs(k_ref)


def test_cholesky_and_ssss_options():
    mol, _ = make_wf(((1, 1), 2), scale=0.0)
    ex = DHFDirect(mol)
    n = 2 * mol.nao_2c()
    rng = np.random.default_rng(0)
    D = rng.normal(size=(2, n, n)) + 1j * rng.normal(size=(2, n, n))
    C = rng.normal(size=(n, 4)) + 1j * rng.normal(size=(n, 4))
    v0 = ex.veff(D)
    v0n = ex._jk_exact(D, False)
    e0 = ex.eri(C, C[:, :2], C, C[:, 1:3])
    for kw in ({"blocks": "LS"}, {"blocks": "S"}, {"blocks": "S", "eri_tol": 1e-13}, {"cd_jk": False}):
        cd = DHFCholesky(mol, tol=1e-13, **kw)
        assert np.abs(cd.veff(D) - v0).max() < 1e-10
        assert np.abs(cd.eri(C, C[:, :2], C, C[:, 1:3]) - e0).max() < 1e-9
    g = DHFDirect(mol, ssss="ground")
    assert np.abs(g.veff(D) - v0).max() < 1e-14
    assert np.abs(g.veff(D, response=True) - (v0n[0] - v0n[1])).max() < 1e-14


def test_subset_of_nuclei():
    _, wf = make_wf(((1, 1), 4), scale=0.0)
    wf.run_wf_optimization_direct_DHF(tol=1e-8)
    lr = LinearResponseDirect(wf, "SD")
    r0 = lr.get_properties(tol=1e-9)
    r1 = lr.get_properties(tol=1e-9, nuclei=[0])
    assert np.abs(r1["shieldings"] - r0["shieldings"]).max() < 1e-5
    assert abs(r1["sscc"][0, 1] - r0["sscc"][0, 1]) < 1e-6


@pytest.mark.slow
def test_h2_rmb_giao_shieldings_and_sscc():
    import basis_set_exchange as bse
    from pyscf.scf import hf

    hf.remove_overlap_zero_eigenvalue = False
    geometry = "H 1.0215 -0.0252 0.5645; H 1.1785 -0.5748 1.0355"
    basis = bse.get_basis("dyall-v2z", elements=["H"], fmt="nwchem")
    mol = pyscf.M(atom=geometry, basis=basis, unit="angstrom", nucmod="gaussian", verbose=0)
    mf = scf.dhf.DHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    wf = GeneralizedWaveFunctionUPS(
        ((1, 1), 6), np.array(mf.mo_coeff, dtype=complex), mol, [], False, "fUCCSD",
        {"n_layers": 1, "is_spin_conserving": False}, include_active_kappa=True,
    )
    wf.run_wf_optimization_direct_DHF(tol=1e-7)
    assert abs(wf.energy_elec + 1.8716658443) < 1e-9
    res = LinearResponseDirect(wf, "SD").get_properties(tol=1e-6)
    assert np.abs(res["shieldings"] - 26.977661).max() < 2e-5
    assert abs(res["sscc"][0, 1] - 7.85540) < 2e-4


@pytest.mark.slow
def test_symmetrize_qG_matches_naive():
    import scipy

    from slowquant.unitary_coupled_cluster.linear_response import generalized_naive_DHF
    from slowquant.unitary_coupled_cluster.linear_response.generalized_direct_DHF import LinearResponseDirect as LRD

    mol = pyscf.M(
        atom="H 0 0 0; H 0.1 0 0.8; H 0 1.2 0.8; H 0.2 1.1 -0.1", basis="sto-3g", nucmod="gaussian", verbose=0
    )
    mf = scf.dhf.DHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    wf = GeneralizedWaveFunctionUPS(
        ((2, 2), 6),
        np.array(mf.mo_coeff, dtype=complex),
        mol,
        [],
        False,
        "fUCCSD",
        {"n_layers": 1, "is_spin_conserving": False},
        include_active_kappa=True,
    )
    wf.run_wf_optimization_direct_DHF(tol=1e-9)
    E2, S = LRD(wf, "SD", symmetrize_qG=True).explicit()
    E2p, _ = LRD(wf, "SD").explicit()
    lr0 = generalized_naive_DHF.LinearResponse(wf, excitations="SD")
    lr0.calc_excitation_energies()
    assert np.abs(E2 - lr0.hessian).max() < 1e-8
    assert np.abs(S - lr0.metric).max() < 1e-10
    assert np.abs(E2p - lr0.hessian).max() > 1e-3


def test_onestep_oo_matches_twostep():
    e = []
    for name in ("run_wf_optimization_direct_DHF", "run_wf_optimization_onestep_direct_DHF"):
        _, wf = make_wf(((2, 2), 6), scale=0.0)
        wf.set_thetas([0.0] * len(wf.thetas), [0.0] * len(wf.thetas))
        getattr(wf, name)(tol=1e-7)
        e.append(wf.energy_elec)
    assert abs(e[0] - e[1]) < 1e-10
