import numpy as np

from slowquant.molecularintegrals.integralfunctions import (
    expansion_coefficients,
    hermite_multipole_integral,
)
from slowquant.molecule.moleculeclass import _Molecule


def multipole_integral_driver(mol_obj: _Molecule, multipole_moment: np.ndarray) -> np.ndarray:
    """Driver function for calculating overlap integrals.

    Args:
        mol_obj: Molecule object.
        multipole_moment: Cartesian multipole moment (x, y, z).

    Returns:
        Multipole integrals.
    """
    S = np.zeros((mol_obj.number_bf, mol_obj.number_bf))
    for i, shell1 in enumerate(mol_obj.shells):
        for j, shell2 in enumerate(mol_obj.shells):
            if j > i:  # Matrix is symmetric
                break
            S_slice = multipole_integral(
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
                mol_obj.center_of_mass,
                multipole_moment,
            )
            start_idx1 = shell1.bf_idx[0]
            end_idx1 = shell1.bf_idx[-1] + 1
            start_idx2 = shell2.bf_idx[0]
            end_idx2 = shell2.bf_idx[-1] + 1
            S[start_idx1:end_idx1, start_idx2:end_idx2] = S_slice
            S[start_idx2:end_idx2, start_idx1:end_idx1] = S_slice.transpose()
    return S


def multipole_integral(
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
    multipole_origin: np.ndarray,
    multipole_moment: np.ndarray,
) -> np.ndarray:
    r"""Calculate overlap integral over shells.

    .. math::
        S^{efg}_\mathrm{primitive} = \sum_{t=0}^{\mathrm{min}(i+j,e)}\sum_{r=0}^{\mathrm{min}(k+l,f)}\sum_{p=0}^{\mathrm{min}(m+n,g)}E_t^{ij}E_s^{kl}E_k^{mn}M^e_tM^f_rM^g_p

    .. math::
        S = \sum_{ij}N_iN_jc_ic_jS_{\mathrm{primitive},ij}

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
        multipole_origin: Origin with respect to the multipole moment. Does not matter for the case when all the moments are zero (overlap integral).
        multipole_moment: Cartesian multipole moment (x, y, z).

    Returns:
        Overlap integral between two shells.
    """
    number_primitives1 = len(exponents1)
    number_primitives2 = len(exponents2)
    number_bf1 = len(angular_moments1)
    number_bf2 = len(angular_moments2)
    max_ang1 = np.max(angular_moments1)
    max_ang2 = np.max(angular_moments2)
    S_primitive = np.zeros((number_bf1, number_bf2, number_primitives1, number_primitives2))
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
            M_x = hermite_multipole_integral(
                center1[0], center2[0], multipole_origin[0], exponents1[i], exponents2[j], multipole_moment[0]
            )
            M_y = hermite_multipole_integral(
                center1[1], center2[1], multipole_origin[1], exponents1[i], exponents2[j], multipole_moment[1]
            )
            M_z = hermite_multipole_integral(
                center1[2], center2[2], multipole_origin[2], exponents1[i], exponents2[j], multipole_moment[2]
            )
            for bf_i, (x1, y1, z1) in enumerate(angular_moments1):
                temp = norm1[bf_i, i]
                for bf_j, (x2, y2, z2) in enumerate(angular_moments2):
                    for t1 in range(min(x1 + x2, multipole_moment[0]) + 1):
                        for t2 in range(min(y1 + y2, multipole_moment[1]) + 1):
                            for t3 in range(min(z1 + z2, multipole_moment[2]) + 1):
                                S_primitive[bf_i, bf_j, i, j] += (
                                    M_x[multipole_moment[0], t1]
                                    * M_y[multipole_moment[1], t2]
                                    * M_z[multipole_moment[2], t3]
                                    * E_x[x1, x2, t1]
                                    * E_y[y1, y2, t2]
                                    * E_z[z1, z2, t3]
                                )
                    S_primitive[bf_i, bf_j, i, j] *= temp * norm2[bf_j, j]

    S_slice = np.einsum("i,j,klij->kl", contra_coeff1, contra_coeff2, S_primitive)
    return S_slice
