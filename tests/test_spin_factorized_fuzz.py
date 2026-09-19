"""Randomised differential test of the spin-factorized algebra.

The reference is build_operator_matrix, which builds the operator through the general
determinant kernel, so this compares the two independent implementations directly. Random
operators over random spin spaces reach many more shapes than the fixed cases in
test_spin_factorized_algebra, in particular unequal alpha and beta occupations, where a phase
error in the factorization would show up first.
"""

import numpy as np

import slowquant.unitary_coupled_cluster.spin_factorized_algebra as sfa
from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing
from slowquant.unitary_coupled_cluster.operator_state_algebra import build_operator_matrix
from slowquant.unitary_coupled_cluster.operators import (
    G1,
    G2,
    G3,
    Epq,
    G1_sa,
    G2_sa,
    epqrs,
    hamiltonian_0i_0a,
    hamiltonian_1i_1a,
)
from slowquant.unitary_coupled_cluster.spin_factorized_algebra import propagate_state_factorized
from slowquant.unitary_coupled_cluster.spin_ordering import alpha_idx, beta_idx

NUM_TRIALS = 250
SEED = 20240915


def random_two_electron_integrals(rng: np.random.Generator, num_orbs: int) -> np.ndarray:
    """Draw random two-electron integrals with the right permutational symmetry.

    Args:
        rng: Random number generator.
        num_orbs: Number of spatial orbitals.

    Returns:
        Two-electron integrals.
    """
    g_mo = rng.standard_normal((num_orbs,) * 4)
    g_mo = g_mo + np.transpose(g_mo, (1, 0, 3, 2))
    return g_mo + np.transpose(g_mo, (2, 3, 0, 1))


def random_operator(rng: np.random.Generator, num_orbs: int, num_inactive_orbs: int, num_active_orbs: int):
    """Draw a random spin conserving operator.

    Args:
        rng: Random number generator.
        num_orbs: Number of spatial orbitals.
        num_inactive_orbs: Number of inactive spatial orbitals.
        num_active_orbs: Number of active spatial orbitals.

    Returns:
        Name of the operator kind and the operator.
    """
    kind = int(rng.integers(0, 9))
    h_mo = rng.standard_normal((num_orbs, num_orbs))
    h_mo = h_mo + h_mo.T
    num_virtual_orbs = num_orbs - num_inactive_orbs - num_active_orbs

    def pick(count: int) -> list[int]:
        return [
            int(idx) for idx in rng.integers(num_inactive_orbs, num_inactive_orbs + num_active_orbs, count)
        ]

    def spin(idx: int):
        return alpha_idx(idx, num_orbs) if rng.integers(0, 2) else beta_idx(idx, num_orbs)

    if kind == 0:
        g_mo = random_two_electron_integrals(rng, num_orbs)
        return "hamiltonian_0i_0a", hamiltonian_0i_0a(
            h_mo, g_mo, num_inactive_orbs, num_active_orbs, num_virtual_orbs
        )
    if kind == 1:
        g_mo = random_two_electron_integrals(rng, num_orbs)
        return "hamiltonian_1i_1a", hamiltonian_1i_1a(
            h_mo, g_mo, num_inactive_orbs, num_active_orbs, num_virtual_orbs
        )
    if kind == 2:
        p, q = pick(2)
        return "Epq", Epq(p, q, num_orbs)
    if kind == 3:
        p, q, r, s = pick(4)
        return "epqrs", epqrs(p, q, r, s, num_orbs)
    if kind == 4:
        idx = pick(6)
        # A product of three excitation operators, as the three-electron density matrix uses.
        operator = Epq(idx[0], idx[1], num_orbs) * Epq(idx[2], idx[3], num_orbs)
        return "Epq_cubed", operator * Epq(idx[4], idx[5], num_orbs)
    if kind == 5:
        p, q = pick(2)
        # Both indices get the same spin, otherwise the operator does not conserve spin.
        to_spin = alpha_idx if rng.integers(0, 2) else beta_idx
        return "G1", G1(to_spin(p, num_orbs), to_spin(q, num_orbs), bool(rng.integers(0, 2)))
    if kind == 6:
        p, q, r, s = pick(4)
        return "G2", G2(spin(p), spin(q), spin(r), spin(s), bool(rng.integers(0, 2)))
    if kind == 7:
        p, q, r, s, t, u = pick(6)
        return "G3", G3(spin(p), spin(q), spin(r), spin(s), spin(t), spin(u), bool(rng.integers(0, 2)))
    p, q, r, s = pick(4)
    case = int(rng.integers(1, 6))
    operator = G2_sa(p, q, r, s, case, bool(rng.integers(0, 2)), num_orbs=num_orbs)
    return f"G2_sa_{case}", operator + G1_sa(p, q, True, num_orbs=num_orbs)


def test_factorized_algebra_matches_general_kernel_on_random_operators() -> None:
    """Test random operators over random spin spaces against the general determinant kernel.

    Also asserts that every one of the four application paths is reached. A clean result would
    otherwise be worthless if the heuristics happened to send everything to the fallbacks.
    """
    # Patch in call counters so a clean result cannot come from the heuristics quietly
    # sending everything to the scalar fallbacks.
    watched = (
        ("sigma3", "apply_sigma3_terms"),
        ("mixed_scalar", "apply_mixed_terms"),
        ("pure_dense", "build_pure_spin_matrix"),
        ("pure_scalar", "apply_pure_spin_terms"),
    )
    coverage = dict.fromkeys((name for name, _ in watched), 0)
    original = {attribute: getattr(sfa, attribute) for _, attribute in watched}

    def counted(name, function):
        def wrapper(*args, **kwargs):
            coverage[name] += 1
            return function(*args, **kwargs)

        return wrapper

    for name, attribute in watched:
        setattr(sfa, attribute, counted(name, original[attribute]))
    try:
        rng = np.random.default_rng(SEED)
        num_checked = 0
        num_declined = 0
        worst = 0.0
        for _ in range(NUM_TRIALS):
            num_active_orbs = int(rng.integers(1, 7))
            num_inactive_orbs = int(rng.integers(0, 3))
            num_virtual_orbs = int(rng.integers(0, 3))
            num_orbs = num_inactive_orbs + num_active_orbs + num_virtual_orbs
            num_active_elec_alpha = int(rng.integers(0, num_active_orbs + 1))
            num_active_elec_beta = int(rng.integers(0, num_active_orbs + 1))
            ci_info = get_indexing(
                num_inactive_orbs,
                num_active_orbs,
                num_virtual_orbs,
                num_active_elec_alpha,
                num_active_elec_beta,
            )
            _, operator = random_operator(rng, num_orbs, num_inactive_orbs, num_active_orbs)
            folded = operator.get_folded_operator(num_inactive_orbs, num_active_orbs, num_virtual_orbs)
            if len(folded.operators) == 0:
                continue
            state = rng.standard_normal(len(ci_info.idx2det))
            factorized = propagate_state_factorized(folded, state, ci_info, np.zeros_like(state))
            if factorized is None:
                # An operator that changes the electron count of a spin, which the factorized
                # algebra declines. The general kernel would leave the CI space here.
                num_declined += 1
                continue
            reference = build_operator_matrix(folded, ci_info) @ state
            scale = max(float(np.max(np.abs(reference))), 1e-30)
            worst = max(worst, float(np.max(np.abs(factorized - reference))) / scale)
            num_checked += 1
        assert num_checked > NUM_TRIALS // 3, f"only {num_checked} trials were comparable"
        assert num_declined > 0, "no operator exercised the fall back to the general kernel"
        assert worst < 1e-10, f"worst relative deviation {worst:.3e}"
    finally:
        for _, attribute in watched:
            setattr(sfa, attribute, original[attribute])
    unused = [name for name, count in coverage.items() if count == 0]
    assert not unused, f"application paths never exercised: {unused}, coverage {coverage}"
