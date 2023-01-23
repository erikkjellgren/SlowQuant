import numpy as np
from numba import float64, jit


# Is float64. int64 gives overflow.
# Only approximate value as a float is needed.
@jit(float64(float64), nopython=True, cache=True)
def factorial2(number: int) -> int:
    r"""Double factorial.

    .. math::
        n!! = \prod^{\lceil n/2\rceil-1}_{k=0}(n-2k)

    Reference: https://en.wikipedia.org/wiki/Double_factorial

    Args:
        number: Integer.

    Returns:
        Double factorial of number.
    """
    out = 1
    if number > 0:
        for i in range(0, int(number + 1) // 2):
            out = out * (number - 2 * i)
    return out


def primitive_normalization(exponent: float, angular_moment: np.ndarray) -> float:
    r"""Normalize primitive Gaussian.

    .. math::
        N = \left(\frac{2}{\pi}\right)^{3/4} \frac{2^{l+m+n}\alpha^{(2l+2m+2n+3)/4}}{[(2l-1)!!(2m-1)!!(2n-1)!!]^{1/2}}

    Args:
        exponent: Gaussian exponent.
        angular_moment: Cartesian angular moment of primitive Gaussian.

    Returns:
        Normalization constant for primitive Gaussian.
    """
    l, m, n = angular_moment
    return (
        (2 / np.pi) ** (3 / 4)
        * (2 ** (l + m + n) * exponent ** ((2 * l + 2 * m + 2 * n + 3) / 4))
        / (factorial2(2 * l - 1) * factorial2(2 * m - 1) * factorial2(2 * n - 1)) ** (1 / 2)
    )


def contracted_normalization(
    exponents: np.ndarray,
    coefficients: np.ndarray,
    angular_moments: np.ndarray,
) -> float:
    r"""Normalize contracted Gaussian.

    .. math::
        N = \left[ \frac{\pi^{3/2}(2l-1)!!(2m-1)!!(2n-1)!!}{2^{l+m+n}} \sum_{i,j}^n\frac{a_ia_j}{\left(\alpha_i+\alpha_j\right)^{l+m+n+3/2}} \right]^{-1/2}

    With the cartesian angular moments being :math:`l`, :math:`m`, and :math:`n`, and the Gaussian exponent being :math:`\alpha`, and the contraction coefficients being :math:`a`.

    Reference: Fundamentals of Molecular Integrals Evaluation, https://arxiv.org/abs/2007.12057

    Args:
      exponents: Gaussian exponents.
      coefficients: Contraction coefficients.
      angular_moments: Angular moment of Gaussian orbital.

    Returns:
      Normalization constant for contracted Gaussian.
    """
    normalization_factor = 0
    angular_x, angular_y, angular_z = angular_moments
    number_primitives = len(exponents)
    for i in range(number_primitives):
        for j in range(number_primitives):
            normalization_factor += (
                coefficients[i]
                * coefficients[j]
                / (exponents[i] + exponents[j]) ** (angular_x + angular_y + angular_z + 3 / 2)
            )
    normalization_factor *= (
        np.pi ** (3 / 2)
        * factorial2(2 * angular_x - 1)
        * factorial2(2 * angular_y - 1)
        * factorial2(2 * angular_z - 1)
    ) / (2 ** (angular_x + angular_y + angular_z))
    return normalization_factor ** (-1 / 2)
