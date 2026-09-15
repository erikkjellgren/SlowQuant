"""Benchmark wave function optimization over a range of active space sizes.

Runs a BFGS optimization with and without orbital optimization for each active space, with the
iteration count capped so the whole sweep finishes in a few minutes. The point is the cost per
iteration and per energy evaluation, not the converged energy, so a capped run is the right
measurement, and the energies printed are not converged.

Examples:
    python benchmarks/benchmark_active_space.py
    python benchmarks/benchmark_active_space.py --max-orbs 12 --maxiter 5
    python benchmarks/benchmark_active_space.py --compare gram
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time
import warnings

import slowquant.SlowQuant as sq
import slowquant.unitary_coupled_cluster.ups_wavefunction as ups_module
from slowquant.unitary_coupled_cluster import spin_factorized_algebra
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS

MOLECULES = {
    "n2": "N 0.0 0.0 0.0; N 0.0 0.0 1.2",
    "h2o": """O 0.0 0.0 0.1035174918; H 0.0 0.7955612117 -0.4640237459;
              H 0.0 -0.7955612117 -0.4640237459""",
    "beh2": "Be 0.0 0.0 0.0; H 0.0 0.0 1.3; H 0.0 0.0 -1.3",
    "h6": "; ".join(f"H 0.0 0.0 {1.0 * i}" for i in range(6)),
    "h10": "; ".join(f"H 0.0 0.0 {1.0 * i}" for i in range(10)),
}


def build_hartree_fock(molecule: str, basis: str):
    """Run restricted Hartree-Fock, which every active space below starts from.

    Args:
        molecule: Molecule specification.
        basis: Basis set name.

    Returns:
        SlowQuant object with Hartree-Fock done.
    """
    obj = sq.SlowQuant()
    obj.set_molecule(MOLECULES[molecule], distance_unit="angstrom")
    obj.set_basis_set(basis)
    obj.init_hartree_fock()
    with contextlib.redirect_stdout(io.StringIO()):
        obj.hartree_fock.run_restricted_hartree_fock()
    return obj


def time_optimization(
    obj, num_elec: int, num_orbs: int, ansatz: str, layers: int, maxiter: int, orbital_optimization: bool
) -> dict:
    """Time a capped BFGS optimization of one active space.

    Args:
        obj: SlowQuant object with Hartree-Fock done.
        num_elec: Number of active electrons.
        num_orbs: Number of active spatial orbitals.
        ansatz: Name of ansatz.
        layers: Number of ansatz layers.
        maxiter: Cap on the number of optimizer iterations.
        orbital_optimization: Optimize the orbitals alongside the ansatz parameters.

    Returns:
        Measurements of the run, or the reason it could not be made.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            wf = WaveFunctionUPS(
                (num_elec, num_orbs),
                obj.hartree_fock.mo_coeff,
                obj,
                ansatz,
                ansatz_options={"n_layers": layers},
            )
        except ValueError as exc:
            return {"skipped": str(exc)}
        # num_energy_evals counts the measurements a quantum device would need, not classical
        # calls, so the classical ones are counted here instead.
        counts = {"energy": 0, "gradient": 0}
        calc_energy = wf._calc_energy_optimization
        calc_gradient = wf._calc_gradient_optimization

        def counted_energy(*args, **kwargs):
            counts["energy"] += 1
            return calc_energy(*args, **kwargs)

        def counted_gradient(*args, **kwargs):
            counts["gradient"] += 1
            return calc_gradient(*args, **kwargs)

        wf._calc_energy_optimization = counted_energy  # type: ignore[method-assign]
        wf._calc_gradient_optimization = counted_gradient  # type: ignore[method-assign]
        start = time.perf_counter()
        with warnings.catch_warnings():
            # Stopping at the iteration cap is the intent here, not a problem to report.
            warnings.simplefilter("ignore")
            wf.run_wf_optimization(
                maxiter=maxiter,
                orbital_optimization=orbital_optimization,
                one_step_optimizer="BFGS",
            )
        elapsed = time.perf_counter() - start
    return {
        "seconds": elapsed,
        "num_dets": len(wf.ci_coeffs),
        "num_thetas": len(wf.thetas),
        "num_kappas": len(wf.kappa_idx) if orbital_optimization else 0,
        "num_energy": counts["energy"],
        "num_gradient": counts["gradient"],
    }


def best_of(
    repeat: int, obj, num_orbs: int, ansatz: str, layers: int, maxiter: int, orbital_optimization: bool
) -> dict:
    """Time an active space a few times and keep the fastest run.

    The runs are short, so a single one is easily disturbed by whatever else the machine is
    doing.

    Args:
        repeat: Number of runs to take the fastest of.
        obj: SlowQuant object with Hartree-Fock done.
        num_orbs: Number of active spatial orbitals, also used as the electron count.
        ansatz: Name of ansatz.
        layers: Number of ansatz layers.
        maxiter: Cap on the number of optimizer iterations.
        orbital_optimization: Optimize the orbitals alongside the ansatz parameters.

    Returns:
        Measurements of the fastest run.
    """
    best = None
    for _ in range(max(repeat, 1)):
        result = time_optimization(obj, num_orbs, num_orbs, ansatz, layers, maxiter, orbital_optimization)
        if "skipped" in result:
            return result
        if best is None or result["seconds"] < best["seconds"]:
            best = result
    assert best is not None
    return best


def main() -> None:
    """Run the sweep and print the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", default="n2", choices=sorted(MOLECULES))
    parser.add_argument("--basis", default="6-31G")
    parser.add_argument("--ansatz", default="tUPS")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument(
        "--maxiter", type=int, default=3, help="cap on optimizer iterations, keeps the sweep short"
    )
    parser.add_argument("--min-orbs", type=int, default=2)
    parser.add_argument("--max-orbs", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1, help="take the fastest of this many runs per point")
    parser.add_argument(
        "--compare",
        choices=("off", "all", "gram", "algebra"),
        default="off",
        help=(
            "also run with something turned off, to show what it buys: 'gram' the Gram density "
            "matrices, 'algebra' the spin-factorized algebra, 'all' both. Note 'all' is not the "
            "full pre-branch code, the memoized operator folding applies either way"
        ),
    )
    args = parser.parse_args()

    obj = build_hartree_fock(args.molecule, args.basis)
    original_factorize = spin_factorized_algebra.factorize_operator
    original_gram = ups_module.can_build_rdm12_as_gram
    # Warm up both code paths. They compile different Numba kernels, so whichever ran first
    # would otherwise carry that compilation into the table and flatter the other one.
    for warm_general in (False, True):
        if warm_general:
            # Warm the slow side of both switches, whichever the run ends up comparing.
            spin_factorized_algebra.factorize_operator = lambda *a, **k: None
            ups_module.can_build_rdm12_as_gram = lambda *a, **k: False
        try:
            for num_orbs in (2, 4):
                time_optimization(obj, num_orbs, num_orbs, args.ansatz, args.layers, 1, True)
                time_optimization(obj, num_orbs, num_orbs, args.ansatz, args.layers, 1, False)
        finally:
            spin_factorized_algebra.factorize_operator = original_factorize
            ups_module.can_build_rdm12_as_gram = original_gram
    print(
        f"{args.molecule}/{args.basis}, {args.ansatz} with {args.layers} layer(s), "
        f"BFGS capped at {args.maxiter} iterations"
    )
    print("Energies are from a capped run and are not converged.\n")
    header = (
        f"{'active space':>13} {'dets':>9} {'params':>12} {'E':>5} {'grad':>5} {'seconds':>9} {'s/call':>9}"
    )
    if args.compare != "off":
        header += f" {'without':>9} {'speedup':>8}"
    print(header)

    for num_orbs in range(args.min_orbs, args.max_orbs + 1, 2):
        for orbital_optimization in (False, True):
            result = best_of(
                args.repeat,
                obj,
                num_orbs,
                args.ansatz,
                args.layers,
                args.maxiter,
                orbital_optimization,
            )
            label = f"({num_orbs},{num_orbs}){'+oo' if orbital_optimization else '   '}"
            if "skipped" in result:
                print(f"{label:>13} {'skipped':>9}  {result['skipped'][:48]}")
                continue
            params = f"{result['num_thetas']}t"
            if result["num_kappas"]:
                params += f"+{result['num_kappas']}k"
            num_calls = result["num_energy"] + result["num_gradient"]
            row = (
                f"{label:>13} {result['num_dets']:9d} {params:>12} "
                f"{result['num_energy']:5d} {result['num_gradient']:5d} "
                f"{result['seconds']:9.2f} {result['seconds'] / max(num_calls, 1):9.4f}"
            )
            if args.compare != "off":
                if args.compare in ("all", "algebra"):
                    spin_factorized_algebra.factorize_operator = lambda *a, **k: None
                if args.compare in ("all", "gram"):
                    ups_module.can_build_rdm12_as_gram = lambda *a, **k: False
                try:
                    general = best_of(
                        args.repeat,
                        obj,
                        num_orbs,
                        args.ansatz,
                        args.layers,
                        args.maxiter,
                        orbital_optimization,
                    )
                finally:
                    spin_factorized_algebra.factorize_operator = original_factorize
                    ups_module.can_build_rdm12_as_gram = original_gram
                row += f" {general['seconds']:9.2f} {general['seconds'] / result['seconds']:7.1f}x"
            print(row, flush=True)


if __name__ == "__main__":
    main()
