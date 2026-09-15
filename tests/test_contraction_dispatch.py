"""Test the choice between the sigma3 contraction and pairing the surviving strings.

Both kernels are correct, so the choice is only about speed. What is asserted here is that the
dense operator of the code, the Hamiltonian, takes the contraction and is faster for it, while
an excitation generator does not and would be slower for it.
"""

import time

import numpy as np
import pytest

import slowquant.SlowQuant as sq
import slowquant.unitary_coupled_cluster.spin_factorized_algebra as sfa
from slowquant.unitary_coupled_cluster.operators import G2, Epq, G2_sa
from slowquant.unitary_coupled_cluster.spin_ordering import alpha_idx, beta_idx
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS

NITROGEN = "N 0.0 0.0 0.0; N 0.0 0.0 1.2"


@pytest.fixture(scope="module")
def wave_function():
    """Build a wave function over an active space big enough for the two kernels to differ.

    Returns:
        Wave function.
    """
    obj = sq.SlowQuant()
    obj.set_molecule(NITROGEN, distance_unit="angstrom")
    obj.set_basis_set("6-31G")
    obj.init_hartree_fock()
    obj.hartree_fock.run_restricted_hartree_fock()
    wf = WaveFunctionUPS((8, 8), obj.hartree_fock.mo_coeff, obj, "tUPS", ansatz_options={"n_layers": 1})
    wf.thetas = list(0.1 * (np.random.default_rng(2).random(len(wf.thetas)) - 0.5))
    return wf


def operators(wave_function):
    """Build the Hamiltonian and a few generators, all folded into the active space.

    Args:
        wave_function: Wave function.

    Returns:
        Name and operator pairs.
    """
    ci_info = wave_function.ci_info
    num_orbs = ci_info.num_active_orbs
    fold = lambda op: op.get_folded_operator(  # noqa: E731
        ci_info.num_inactive_orbs, num_orbs, ci_info.num_virtual_orbs
    )
    return [
        ("Hamiltonian", fold(wave_function.energy_hamiltonian), True),
        (
            "mixed spin double",
            G2(
                alpha_idx(0, num_orbs),
                beta_idx(1, num_orbs),
                alpha_idx(num_orbs - 2, num_orbs),
                beta_idx(num_orbs - 1, num_orbs),
                True,
            ).get_folded_operator(0, num_orbs, 0),
            False,
        ),
        (
            "spin-adapted double",
            G2_sa(0, 1, num_orbs - 2, num_orbs - 1, 1, True, num_orbs=num_orbs).get_folded_operator(
                0, num_orbs, 0
            ),
            False,
        ),
        (
            "product of two singles",
            (Epq(0, num_orbs - 1, num_orbs) * Epq(1, num_orbs - 2, num_orbs)).get_folded_operator(
                0, num_orbs, 0
            ),
            False,
        ),
    ]


def apply_with(op, state, ci_info, contract):
    """Apply an operator with the choice of kernel forced.

    Args:
        op: Operator, already folded.
        state: State.
        ci_info: Information about the CI space.
        contract: Force the contraction if true, the pairing if false.

    Returns:
        New state and the seconds the fastest of a few applications took.
    """
    factorized = sfa.factorize_operator(op, ci_info)
    assert factorized is not None
    factorized.sigma3_layout = None
    factorized.alpha_matrix = None
    factorized.beta_matrix = None
    real = sfa.prefer_contraction_over_pairs
    sfa.prefer_contraction_over_pairs = lambda *args, **kwargs: contract
    try:
        sfa.build_derived_forms(factorized, ci_info)
    finally:
        sfa.prefer_contraction_over_pairs = real
    out = np.zeros_like(state)
    sfa.apply_factorized_operator(factorized, state, ci_info, out)
    best = np.inf
    for _ in range(5):
        out[:] = 0.0
        start = time.perf_counter()
        result = sfa.apply_factorized_operator(factorized, state, ci_info, out)
        best = min(best, time.perf_counter() - start)
    return np.copy(result), best


def test_the_hamiltonian_is_contracted_and_generators_are_not(wave_function) -> None:
    """Test which operators the dispatch sends to the contraction.

    Args:
        wave_function: Wave function.
    """
    for name, op, expected in operators(wave_function):
        factorized = sfa.factorize_operator(op, wave_function.ci_info)
        assert factorized is not None, name
        assert sfa.prefer_contraction_over_pairs(factorized, wave_function.ci_info) is expected, name


def test_the_two_kernels_agree(wave_function) -> None:
    """Test that forcing either kernel gives the same state.

    Args:
        wave_function: Wave function.
    """
    state = np.array(wave_function.ci_coeffs)
    for name, op, _ in operators(wave_function):
        paired, _ = apply_with(op, state, wave_function.ci_info, contract=False)
        contracted, _ = apply_with(op, state, wave_function.ci_info, contract=True)
        assert np.allclose(paired, contracted, atol=1e-12), name


def test_the_contraction_is_faster_for_the_hamiltonian(wave_function) -> None:
    """Test that the Hamiltonian really is quicker to contract than to pair.

    Both paths are timed in the same process on the same operator, so this compares the two
    kernels rather than the machine. The margin asserted is well below what is measured, and the
    point is that the Hamiltonian must not quietly fall back to the pairing again.

    Args:
        wave_function: Wave function.
    """
    state = np.array(wave_function.ci_coeffs)
    op = operators(wave_function)[0][1]
    _, paired = apply_with(op, state, wave_function.ci_info, contract=False)
    _, contracted = apply_with(op, state, wave_function.ci_info, contract=True)
    assert contracted < paired / 1.5, (
        f"contraction {contracted * 1e3:.2f} ms vs pairing {paired * 1e3:.2f} ms"
    )


def test_a_generator_would_be_slower_to_contract(wave_function) -> None:
    """Test that the dispatch is right to keep an excitation generator on the pairing.

    Args:
        wave_function: Wave function.
    """
    state = np.array(wave_function.ci_coeffs)
    op = operators(wave_function)[1][1]
    _, paired = apply_with(op, state, wave_function.ci_info, contract=False)
    _, contracted = apply_with(op, state, wave_function.ci_info, contract=True)
    assert paired < contracted, f"pairing {paired * 1e3:.2f} ms vs contraction {contracted * 1e3:.2f} ms"
