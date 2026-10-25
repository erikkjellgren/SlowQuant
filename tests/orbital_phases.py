r"""Fix the arbitrary phases of Hartree-Fock orbitals, so pinned test values do not depend on them.

An eigensolver returns each molecular orbital only up to a sign, and a set of degenerate orbitals
only up to a rotation among themselves. Neither changes the Hartree-Fock state, but a test that
pins a value at fixed or flat-start ansatz parameters depends on both: flipping the sign of an
orbital flips the sign of every excitation that touches it, so the same parameters describe a
different state. Which choice the solver makes is not part of its contract. It changed for water
and LiH in STO-3G with the update to OpenBLAS 0.3.34, which supplies NumPy's eigensolver, and
that broke twenty-six tests at once.

canonicalize_mo_coeff removes both freedoms with a fixed rule that does not depend on the
solver. Within each group of degenerate orbitals it picks the basis that diagonalizes

.. math::
    M = \boldsymbol{C}_g^T\boldsymbol{W}\boldsymbol{C}_g,\qquad W_{\mu\nu} = w_\mu\delta_{\mu\nu}

for a fixed, strictly increasing weight per atomic orbital, which keeps the columns orbitals
of the same Fock matrix and orthonormal in the overlap metric. It then fixes the sign of every
orbital so that :math:`\sum_\mu w_\mu C_{\mu i} > 0`. Both steps assert a safe margin, so a
molecule for which the rule is ambiguous fails loudly instead of silently picking a phase.
"""

import numpy as np


def canonicalize_mo_coeff(
    mo_coeff: np.ndarray,
    fock_ao: np.ndarray,
    degeneracy_tol: float = 1e-8,
    margin: float = 1e-6,
) -> np.ndarray:
    r"""Return the orbitals with their degenerate rotations and signs fixed by a fixed rule.

    Args:
        mo_coeff: Orbital coefficients, one orbital per column, as the eigensolver returned them.
        fock_ao: Converged Fock matrix in the atomic orbital basis, used for the orbital energies
            :math:`\varepsilon_i = \left(\boldsymbol{C}^T\boldsymbol{F}\boldsymbol{C}\right)_{ii}`.
        degeneracy_tol: Orbitals whose energies differ by less than this are treated as one
            degenerate group.
        margin: Smallest relative separation the rule needs, between the eigenvalues of
            :math:`M` within a group and of the weighted sum from zero.

    Returns:
        Orbital coefficients in canonical phase.
    """
    coeff = np.array(mo_coeff, dtype=float, copy=True)
    num_ao, num_mo = coeff.shape
    weights = 1.0 + np.arange(num_ao) / num_ao
    energies = np.einsum("mi,mn,ni->i", coeff, fock_ao, coeff)
    start = 0
    while start < num_mo:
        stop = start + 1
        while stop < num_mo and abs(energies[stop] - energies[start]) < degeneracy_tol:
            stop += 1
        if stop - start > 1:
            block = coeff[:, start:stop]
            values, vectors = np.linalg.eigh(block.T @ (weights[:, None] * block))
            gaps = np.diff(values)
            if np.min(gaps) <= margin * max(np.max(np.abs(values)), 1.0):
                raise ValueError(f"Cannot fix a basis for degenerate orbitals {start}..{stop - 1}.")
            coeff[:, start:stop] = block @ vectors
        start = stop
    for orbital in range(num_mo):
        column = coeff[:, orbital]
        weighted = float(weights @ column)
        if abs(weighted) <= margin * float(np.sum(np.abs(weights * column))):
            raise ValueError(f"Cannot fix the sign of orbital {orbital}.")
        if weighted < 0:
            coeff[:, orbital] = -column
    return coeff
