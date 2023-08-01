import numpy as np

from slowquant.molecularintegrals.integralfunctions import expansion_coefficients
from slowquant.molecule.moleculeclass import _Molecule


def kinetic_energy_integral_driver(mol_obj: _Molecule) -> np.ndarray:
    """Driver function for calculating kinetic energy integrals.

    Args:
        mol_obj: Molecule object.

    Returns:
        Kinetic energy integrals.
    """
    T = np.zeros((mol_obj.number_bf, mol_obj.number_bf))
    for i, shell1 in enumerate(mol_obj.shells):
        for j, shell2 in enumerate(mol_obj.shells):
            if j > i:  # Matrix is symmetric
                break
            T_slice = kinetic_energy_integral(
                shell1.center,
                shell2.center,
                shell1.exponents,
                shell2.exponents,
                shell1.contraction_coefficients,
                shell2.contraction_coefficients,
                shell1.normalization,
                shell2.normalization,
                shell1.angular_moments,
                shell2.angular_moments,
            )
            start_idx1 = shell1.bf_idx[0]
            end_idx1 = shell1.bf_idx[-1] + 1
            start_idx2 = shell2.bf_idx[0]
            end_idx2 = shell2.bf_idx[-1] + 1
            T[start_idx1:end_idx1, start_idx2:end_idx2] = T_slice
            T[start_idx2:end_idx2, start_idx1:end_idx1] = T_slice.transpose()
    return T


def kinetic_energy_integral(
    center1: np.ndarray,
    center2: np.ndarray,
    exponents1: np.ndarray,
    exponents2: np.ndarray,
    contra_coeff1: np.ndarray,
    contra_coeff2: np.ndarray,
    norm1: np.ndarray,
    norm2: np.ndarray,
    angular_moments1: np.ndarray,
    angular_moments2: np.ndarray,
) -> np.ndarray:
    r"""Calculate kinetic energy integral over shells.

    .. math::
        D^2_{ij} = 4a^2S_{i+2,j} - 2a(2i+1)S_{ij}+i(i-1)S_[i-2,j]

    .. math::
        T_\mathrm{primitive} = -\frac{1}{2}\left( D^2_{ij}S_{kl}S_{mn} + S_{ij}D^2_{kl}S_{mn} + S_{ij}S_{kl}D^2_{mn} \right)

    .. math::
        T = \sum_{ij}N_iN_jc_ic_jT_{\mathrm{primitive},ij}

    With :math:`E` being the expansion coefficients, :math:`p` the sum of the exponents of the Gaussians, :math:`N` the normalization constant, and, :math:`c` the contraction coefficients.

    Reference: Molecular Electronic-Structure Theory, https://onlinelibrary.wiley.com/doi/book/10.1002/9781119019572

    Args:
        center1: Center of first shell.
        center2: Center of second shell.
        exponents1: Exponents of primitives in first shell.
        exponents2: Exponents of primitives in second shell.
        contra_coeff1: Contraction coefficients of primitives in first shell.
        contra_coeff2: Contraction coefficients of primitives in second shell.
        norm1: Normalization constant of basis-functions in first shell.
        norm2: Normalization constant of basis-functions in second shell.
        angular_moments1: Cartesian angular moments of basis-functions in first shell.
        angular_moments2: Cartesian angular moments of basis-functions in second shell.

    Returns:
        Kinetic energy integral between two shells.
    """
    number_primitives1 = len(exponents1)
    number_primitives2 = len(exponents2)
    number_bf1 = len(angular_moments1)
    number_bf2 = len(angular_moments2)
    max_ang1 = np.max(angular_moments1) + 2
    max_ang2 = np.max(angular_moments2) + 2
    T_primitive = np.zeros((number_bf1, number_bf2, number_primitives1, number_primitives2))
    for i in range(number_primitives1):
        for j in range(number_primitives2):
            E_x = expansion_coefficients(
                center1[0], center2[0], exponents1[i], exponents2[j], max_ang1, max_ang2
            )
            E_y = expansion_coefficients(
                center1[1], center2[1], exponents1[i], exponents2[j], max_ang1, max_ang2
            )
            E_z = expansion_coefficients(
                center1[2], center2[2], exponents1[i], exponents2[j], max_ang1, max_ang2
            )
            for bf_i, (x1, y1, z1) in enumerate(angular_moments1):
                temp = norm1[bf_i, i]
                for bf_j, (x2, y2, z2) in enumerate(angular_moments2):
                    D2_x = (
                        4 * exponents1[i] ** 2 * E_x[x1 + 2, x2, 0]
                        - 2 * exponents1[i] * (2 * x1 + 1) * E_x[x1, x2, 0]
                        + x1 * (x1 - 1) * E_x[x1 - 2, x2, 0]
                    )
                    D2_y = (
                        4 * exponents1[i] ** 2 * E_y[y1 + 2, y2, 0]
                        - 2 * exponents1[i] * (2 * y1 + 1) * E_y[y1, y2, 0]
                        + y1 * (y1 - 1) * E_y[y1 - 2, y2, 0]
                    )
                    D2_z = (
                        4 * exponents1[i] ** 2 * E_z[z1 + 2, z2, 0]
                        - 2 * exponents1[i] * (2 * z1 + 1) * E_z[z1, z2, 0]
                        + z1 * (z1 - 1) * E_z[z1 - 2, z2, 0]
                    )
                    T_primitive[bf_i, bf_j, i, j] = (
                        temp
                        * norm2[bf_j, j]
                        * (
                            D2_x * E_y[y1, y2, 0] * E_z[z1, z2, 0]
                            + E_x[x1, x2, 0] * D2_y * E_z[z1, z2, 0]
                            + E_x[x1, x2, 0] * E_y[y1, y2, 0] * D2_z
                        )
                    )
            p = exponents1[i] + exponents2[j]
            T_primitive[:, :, i, j] *= -1 / 2 * (np.pi / p) ** (3 / 2)

    T_slice = np.einsum('i,j,klij->kl', contra_coeff1, contra_coeff2, T_primitive)
    return T_slice
