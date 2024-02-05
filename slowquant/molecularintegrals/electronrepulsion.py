import numpy as np
from numba import float64, int64, jit

from slowquant.molecularintegrals.integralfunctions import (
    expansion_coefficients,
    hermite_coulomb_integral,
)
from slowquant.molecule.moleculeclass import _Molecule


def electron_repulsion_integral_driver(
    mol_obj: _Molecule, cauchy_schwarz_matrix: np.ndarray = None, cauchy_schwarz_threshold: float = 10**-10
) -> np.ndarray:
    r"""Driver function for calculating electron-repulsion integrals.

    Might use the Cauchy Schwarz inequility to do integral screening:

    .. math::
        |g_{abcd}| \geq \sqrt{g_{abab}}\sqrt{{g_{cdcd}}}

    Reference: Molecular Electronic-Structure Theory, https://onlinelibrary.wiley.com/doi/book/10.1002/9781119019572

    Args:
        mol_obj: Molecule object.
        cauchy_schwarz_matrix: Cauchy Schwarz matrix, :math:`\sqrt{g_{ijij}}`.
        cauchy_schwarz_threshold: Integral threshold using Cauchy Scwartz inequlity.

    Returns:
        Electron-repulsion integrals.
    """
    V = np.zeros((mol_obj.number_bf, mol_obj.number_bf, mol_obj.number_bf, mol_obj.number_bf))
    for i, shell1 in enumerate(mol_obj.shells):
        for j, shell2 in enumerate(mol_obj.shells):
            if j < i:
                continue
            ij = i * (i + 1) // 2 + j
            for k, shell3 in enumerate(mol_obj.shells):
                for l, shell4 in enumerate(mol_obj.shells):
                    if l < k:
                        continue
                    kl = k * (k + 1) // 2 + l
                    if ij < kl:
                        continue
                    if cauchy_schwarz_matrix is not None:
                        if (
                            cauchy_schwarz_matrix[i, j] * cauchy_schwarz_matrix[k, l]
                            < cauchy_schwarz_threshold
                        ):
                            continue
                    V_slice = electron_repulsion_integral(
                        shell1.center,
                        shell2.center,
                        shell3.center,
                        shell4.center,
                        shell1.exponents,
                        shell2.exponents,
                        shell3.exponents,
                        shell4.exponents,
                        shell1.contraction_coefficients,
                        shell2.contraction_coefficients,
                        shell3.contraction_coefficients,
                        shell4.contraction_coefficients,
                        shell1.normalization,
                        shell2.normalization,
                        shell3.normalization,
                        shell4.normalization,
                        shell1.angular_moments,
                        shell2.angular_moments,
                        shell3.angular_moments,
                        shell4.angular_moments,
                    )
                    start_idx1 = shell1.bf_idx[0]
                    end_idx1 = shell1.bf_idx[-1] + 1
                    start_idx2 = shell2.bf_idx[0]
                    end_idx2 = shell2.bf_idx[-1] + 1
                    start_idx3 = shell3.bf_idx[0]
                    end_idx3 = shell3.bf_idx[-1] + 1
                    start_idx4 = shell4.bf_idx[0]
                    end_idx4 = shell4.bf_idx[-1] + 1
                    V[start_idx1:end_idx1, start_idx2:end_idx2, start_idx3:end_idx3, start_idx4:end_idx4] = (
                        V_slice
                    )
                    V[start_idx2:end_idx2, start_idx1:end_idx1, start_idx3:end_idx3, start_idx4:end_idx4] = (
                        V_slice.transpose([1, 0, 2, 3])
                    )
                    V[start_idx1:end_idx1, start_idx2:end_idx2, start_idx4:end_idx4, start_idx3:end_idx3] = (
                        V_slice.transpose([0, 1, 3, 2])
                    )
                    V[start_idx2:end_idx2, start_idx1:end_idx1, start_idx4:end_idx4, start_idx3:end_idx3] = (
                        V_slice.transpose([1, 0, 3, 2])
                    )
                    V[start_idx3:end_idx3, start_idx4:end_idx4, start_idx1:end_idx1, start_idx2:end_idx2] = (
                        V_slice.transpose([2, 3, 0, 1])
                    )
                    V[start_idx3:end_idx3, start_idx4:end_idx4, start_idx2:end_idx2, start_idx1:end_idx1] = (
                        V_slice.transpose([2, 3, 1, 0])
                    )
                    V[start_idx4:end_idx4, start_idx3:end_idx3, start_idx1:end_idx1, start_idx2:end_idx2] = (
                        V_slice.transpose([3, 2, 0, 1])
                    )
                    V[start_idx4:end_idx4, start_idx3:end_idx3, start_idx2:end_idx2, start_idx1:end_idx1] = (
                        V_slice.transpose([3, 2, 1, 0])
                    )
    return V


@jit(
    float64[:, :, :, :](
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:, :],
        float64[:, :],
        float64[:, :],
        float64[:, :],
        int64[:, :],
        int64[:, :],
        int64[:, :],
        int64[:, :],
    ),
    nopython=True,
    cache=True,
)
def electron_repulsion_integral(
    center1: np.ndarray,
    center2: np.ndarray,
    center3: np.ndarray,
    center4: np.ndarray,
    exponents1: np.ndarray,
    exponents2: np.ndarray,
    exponents3: np.ndarray,
    exponents4: np.ndarray,
    contra_coeff1: np.ndarray,
    contra_coeff2: np.ndarray,
    contra_coeff3: np.ndarray,
    contra_coeff4: np.ndarray,
    norm1: np.ndarray,
    norm2: np.ndarray,
    norm3: np.ndarray,
    norm4: np.ndarray,
    angular_moments1: np.ndarray,
    angular_moments2: np.ndarray,
    angular_moments3: np.ndarray,
    angular_moments4: np.ndarray,
) -> np.ndarray:
    r"""Calculate electron-repulsion integral over shells.

    Args:

    Returns:
        Electron-repulsion integral between two shells.
    """
    number_primitives1 = len(exponents1)
    number_primitives2 = len(exponents2)
    number_primitives3 = len(exponents3)
    number_primitives4 = len(exponents4)
    number_bf1 = len(angular_moments1)
    number_bf2 = len(angular_moments2)
    number_bf3 = len(angular_moments3)
    number_bf4 = len(angular_moments4)
    max_ang1 = np.max(angular_moments1)
    max_ang2 = np.max(angular_moments2)
    max_ang3 = np.max(angular_moments3)
    max_ang4 = np.max(angular_moments4)
    V_primitive = np.zeros(
        (
            number_bf1,
            number_bf2,
            number_bf3,
            number_bf4,
            number_primitives1,
            number_primitives2,
            number_primitives3,
            number_primitives4,
        )
    )
    pre_E_xk = np.zeros(
        (number_primitives3, number_primitives4, max_ang3 + 1, max_ang4 + 1, max_ang3 + max_ang4 + 1)
    )
    pre_E_yk = np.zeros(
        (number_primitives3, number_primitives4, max_ang3 + 1, max_ang4 + 1, max_ang3 + max_ang4 + 1)
    )
    pre_E_zk = np.zeros(
        (number_primitives3, number_primitives4, max_ang3 + 1, max_ang4 + 1, max_ang3 + max_ang4 + 1)
    )
    for k in range(number_primitives3):
        for l in range(number_primitives4):
            pre_E_xk[k, l, :, :, :] = expansion_coefficients(
                center3[0], center4[0], exponents3[k], exponents4[l], max_ang3, max_ang4
            )
            pre_E_yk[k, l, :, :, :] = expansion_coefficients(
                center3[1], center4[1], exponents3[k], exponents4[l], max_ang3, max_ang4
            )
            pre_E_zk[k, l, :, :, :] = expansion_coefficients(
                center3[2], center4[2], exponents3[k], exponents4[l], max_ang3, max_ang4
            )
    for i in range(number_primitives1):
        for j in range(number_primitives2):
            E_xb = expansion_coefficients(
                center1[0], center2[0], exponents1[i], exponents2[j], max_ang1, max_ang2
            )
            E_yb = expansion_coefficients(
                center1[1], center2[1], exponents1[i], exponents2[j], max_ang1, max_ang2
            )
            E_zb = expansion_coefficients(
                center1[2], center2[2], exponents1[i], exponents2[j], max_ang1, max_ang2
            )
            p = exponents1[i] + exponents2[j]
            P = (exponents1[i] * center1 + exponents2[j] * center2) / p
            for k in range(number_primitives3):
                for l in range(number_primitives4):
                    q = exponents3[k] + exponents4[l]
                    Q = (exponents3[k] * center3 + exponents4[l] * center4) / q
                    alpha = p * q / (p + q)
                    R = hermite_coulomb_integral(
                        max_ang1 + max_ang2 + max_ang3 + max_ang4,
                        max_ang1 + max_ang2 + max_ang3 + max_ang4,
                        max_ang1 + max_ang2 + max_ang3 + max_ang4,
                        alpha,
                        P - Q,
                    )
                    for bf_i, (x1, y1, z1) in enumerate(angular_moments1):
                        for bf_j, (x2, y2, z2) in enumerate(angular_moments2):
                            for bf_k, (x3, y3, z3) in enumerate(angular_moments3):
                                for bf_l, (x4, y4, z4) in enumerate(angular_moments4):
                                    for tau in range(x3 + x4 + 1):
                                        for nu in range(y3 + y4 + 1):
                                            for phi in range(z3 + z4 + 1):
                                                E_temp = (
                                                    (-1.0) ** (tau + nu + phi)
                                                    * pre_E_xk[k, l, x3, x4, tau]
                                                    * pre_E_yk[k, l, y3, y4, nu]
                                                    * pre_E_zk[k, l, z3, z4, phi]
                                                )
                                                for t in range(x1 + x2 + 1):
                                                    for u in range(y1 + y2 + 1):
                                                        for v in range(z1 + z2 + 1):
                                                            V_primitive[
                                                                bf_i, bf_j, bf_k, bf_l, i, j, k, l
                                                            ] += (
                                                                E_temp
                                                                * E_xb[x1, x2, t]
                                                                * E_yb[y1, y2, u]
                                                                * E_zb[z1, z2, v]
                                                                * R[t + tau, u + nu, v + phi]
                                                            )
                                    V_primitive[bf_i, bf_j, bf_k, bf_l, i, j, k, l] *= (
                                        norm1[bf_i, i] * norm2[bf_j, j] * norm3[bf_k, k] * norm4[bf_l, l]
                                    )
                    V_primitive[:, :, :, :, i, j, k, l] *= 2 * (np.pi) ** (5 / 2) / (p * q * (p + q) ** 0.5)

    V_slice = np.zeros((number_bf1, number_bf2, number_bf3, number_bf4))
    for bf_i in range(number_bf1):
        for bf_j in range(number_bf2):
            for bf_k in range(number_bf3):
                for bf_l in range(number_bf4):
                    for i in range(number_primitives1):
                        for j in range(number_primitives2):
                            for k in range(number_primitives3):
                                for l in range(number_primitives4):
                                    V_slice[bf_i, bf_j, bf_k, bf_l] += (
                                        contra_coeff1[i]
                                        * contra_coeff2[j]
                                        * contra_coeff3[k]
                                        * contra_coeff4[l]
                                        * V_primitive[bf_i, bf_j, bf_k, bf_l, i, j, k, l]
                                    )
    return V_slice


def get_cauchy_schwarz_matrix(mol_obj: _Molecule) -> np.ndarray:
    r"""Calculate Cauchy Schwarz inequality matrix on shell level.

    Integrals of the form :math:`\sqrt{g_{ijij}}` will be the Cauchy Schwarz marix.

    Args:
        mol_obj: Molecule object.

    Returns:
        Cauchy Schwarz inequality matrix.
    """
    cauchy_schwarz_matrix = np.zeros((mol_obj.number_shell, mol_obj.number_shell))
    for i, shell1 in enumerate(mol_obj.shells):
        for j, shell2 in enumerate(mol_obj.shells):
            if j < i:
                continue
            V_slice = electron_repulsion_integral(
                shell1.center,
                shell2.center,
                shell1.center,
                shell2.center,
                shell1.exponents,
                shell2.exponents,
                shell1.exponents,
                shell2.exponents,
                shell1.contraction_coefficients,
                shell2.contraction_coefficients,
                shell1.contraction_coefficients,
                shell2.contraction_coefficients,
                shell1.normalization,
                shell2.normalization,
                shell1.normalization,
                shell2.normalization,
                shell1.angular_moments,
                shell2.angular_moments,
                shell1.angular_moments,
                shell2.angular_moments,
            )
            cauchy_schwarz_matrix[i, j] = cauchy_schwarz_matrix[j, i] = np.max(V_slice) ** 0.5
    return cauchy_schwarz_matrix
