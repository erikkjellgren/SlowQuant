import numpy as np

from slowquant.molecule.moleculeclass import _Molecule


def expansion_coefficients(
    A_x: float, B_x: float, a: float, b: float, angular_a: int, angular_b: int
) -> np.ndarray:
    r"""Calculate expansion coefficients for McMurchie-Davidson scheme.

    .. math::
        E_t^{i,j} = 0\ \ \mathrm{for}\ \ t<0\ \mathrm{or}\ t>i+j

    .. math::
        E_t^{i+1,j} = \frac{1}{2p}E_{t-1}^{i,j} + X_{PA}E_t^{i,j} + (t+1)E_{t+1}^{i,j}

    .. math::
        E_t^{i,j+1} = \frac{1}{2p}E_{t-1}^{i,j} + X_{PB}E_t^{i,j} + (t+1)E_{t+1}^{i,j}

    .. math::
        E_0^{0,0} = \exp\left(-\mu X^2_{AB}\right)

    With :math:`p=a+b`, :math:`\mu=\frac{ab}{a+b}`, :math:`X_{AB}=A_x-B_x`, :math:`P_x=\frac{aA_x+bB_x}{p}`, and,
    for :math:`a` and :math:`b` being Gaussian exponents, and :math:`A_x` :math:`B_x` being the Gaussian centers.

    Reference: Molecular Electronic-Structure Theory, https://onlinelibrary.wiley.com/doi/book/10.1002/9781119019572

    Args:
        A_x: x-coordinate of first Gaussian.
        B_x: x-coordinate of second Gaussian.
        a: Exponent of first Gaussian.
        b: Exponent of second Gaussian.
        angular_a: x-angular moment of first Gaussian.
        angular_b: x-angular moment of second Gaussian.

    Returns:
        Returns expansion coefficients for McMurchie-Davidson scheme
    """
    X_AB = A_x - B_x
    p = a + b
    mu = a * b / (a + b)
    P_x = (a * A_x + b * B_x) / p
    X_PA = P_x - A_x
    X_PB = P_x - B_x
    e_coeff = np.zeros((angular_a + 1, angular_b + 1, angular_a + angular_b + 1))
    for i in range(angular_a + 1):
        for j in range(angular_b + 1):
            for t in range(angular_a + angular_b + 1):
                value = 0.0
                if i == j == t == 0:  # Boundary condition
                    value = np.exp(-mu * X_AB**2)
                elif i == 0:  # Increment j
                    if j > 0 and t > 0:
                        value += 1 / (2 * p) * e_coeff[i, j - 1, t - 1]
                    if j > 0:
                        value += X_PB * e_coeff[i, j - 1, t]
                    if j > 0 and t + 1 <= i + j:
                        value += (t + 1) * e_coeff[i, j - 1, t + 1]
                else:  # Increment i
                    if i > 0 and t > 0:
                        value += 1 / (2 * p) * e_coeff[i - 1, j, t - 1]
                    if i > 0:
                        value += X_PA * e_coeff[i - 1, j, t]
                    if i > 0 and t + 1 <= i + j:
                        value += (t + 1) * e_coeff[i - 1, j, t + 1]
                e_coeff[i, j, t] = value
    return e_coeff


def overlap_integral_driver(mol_obj: _Molecule) -> np.ndarray:
    """Driver function for calculating overlap integrals.

    Args:
        mol_obj: Molecule object.

    Returns:
        Overlap integrals.
    """
    S = np.zeros((mol_obj.number_bf, mol_obj.number_bf))
    for i, shell1 in enumerate(mol_obj.shells):
        for j, shell2 in enumerate(mol_obj.shells):
            if j > i:  # Matrix is symmetric
                break
            S_slice = overlap_integral(
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
            S[start_idx1:end_idx1, start_idx2:end_idx2] = S_slice
            S[start_idx2:end_idx2, start_idx1:end_idx1] = S_slice.transpose()
    return S


def overlap_integral(
    center1,
    center2,
    exponents1,
    exponents2,
    contra_coeff1,
    contra_coeff2,
    norm1,
    norm2,
    angular_moments1,
    angular_moments2,
) -> np.ndarray:
    r"""Calculate overlap integral over shells.

    .. math::
        S_\mathrm{primitive} = \left(\frac{\pi}{p}\right)E_0^{ij}E_0^{kl}E_0^{mn}

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
            for bf_i, (x1, y1, z1) in enumerate(angular_moments1):
                temp = norm1[bf_i, i]
                for bf_j, (x2, y2, z2) in enumerate(angular_moments2):
                    S_primitive[bf_i, bf_j, i, j] = (
                        temp * norm2[bf_j, j] * E_x[x1, x2, 0] * E_y[y1, y2, 0] * E_z[z1, z2, 0]
                    )
            p = exponents1[i] + exponents2[j]
            S_primitive[:, :, i, j] *= (np.pi / p) ** (3 / 2)

    S_slice = np.einsum("i,j,klij->kl", contra_coeff1, contra_coeff2, S_primitive)
    return S_slice
