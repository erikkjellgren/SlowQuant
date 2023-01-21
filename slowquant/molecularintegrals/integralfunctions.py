import numpy as np


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
