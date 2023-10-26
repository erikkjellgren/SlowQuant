import numpy as np


def dipole_moment(
    dipole_integrals: np.ndarray,
    rdm1: np.ndarray,
    atom_charges: np.ndarray,
    atom_coordinates: np.ndarray,
    dipole_origin: np.ndarray,
) -> np.ndarray:
    r"""Calculate molecular dipole moment.

    .. math::
        \left<\vec{\mu}\right> = -\sum_{\mu\nu}D_{\mu\nu}\left<\chi_\mu\left|\vec{\mu}\right|\chi_\nu\right> + \sum_I Z_I R_{I0}

    Args:
        dipole_integrals: Cartesian dipole integrals (x, y, z).
        rdm1: One-electron reduced density matrix.
        atom_charges: Atom charges.
        atom_coordinates: Atom coordinates.
        dipole_origin: Origin with respect to which the dipole moment is calculated.

    Returns:
        Molecular dipolemoment (x, y, z).
    """
    return -np.einsum("ij,kij->k", rdm1, dipole_integrals) + np.einsum(
        "i,ik->k", atom_charges, atom_coordinates - dipole_origin
    )
