import math

import numpy as np

from slowquant.molecule.moleculefunctions import factorial2


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


def hermite_coulomb_integral(
    angular_x: int, angular_y: int, angular_z: int, p: float, PC: np.ndarray
) -> np.ndarray:
    r"""Calculate Hermite coulomb integral.

    .. math::
        R^n_{t+1,u,v} = t*R^{n+1}_{t-1,u,v} + X_{PC}R^{n+1}_{tuv}

    .. math::
        R^n_{t,u+1,v} = u*R^{n+1}_{t,u-1,v} + Y_{PC}R^{n+1}_{tuv}

    .. math::
        R^n_{t,u,v+1} = v*R^{n+1}_{t,u,v-1} + Z_{PC}R^{n+1}_{tuv}

    .. math::
        R^{n}_{000} = (-2p)^nF_n\left(pR^2_{PC}\right)

    Reference: Molecular Electronic-Structure Theory, https://onlinelibrary.wiley.com/doi/book/10.1002/9781119019572
    """
    X_PC, Y_PC, Z_PC = PC
    R_PC = (X_PC**2 + Y_PC**2 + Z_PC**2) ** 0.5
    r_integral = np.zeros(
        (angular_x + 1, angular_y + 1, angular_z + 1, angular_x + angular_y + angular_z + 2)
    )
    for t in range(angular_x + 1):
        for u in range(angular_y + 1):
            for v in range(angular_z + 1):
                for n in range(angular_x + angular_y + angular_z + 1):
                    value = 0.0
                    if t == u == v == 0:
                        value = (-2.0 * p) ** n * boys_function(n, p * R_PC**2)
                    elif t == u == 0:  # Increment v
                        if v > 1:
                            value += (v - 1) * r_integral[t, u, v - 2, n + 1]
                        if v > 0:
                            value += Z_PC * r_integral[t, u, v - 1, n + 1]
                    elif t == 0:  # Increment u
                        if u > 1:
                            value += (u - 1) * r_integral[t, u - 2, v, n + 1]
                        if u > 0:
                            value += Y_PC * r_integral[t, u - 1, v, n + 1]
                    else:  # Increment t
                        if t > 1:
                            value += (t - 1) * r_integral[t - 2, u, v, n + 1]
                        if t > 0:
                            value += X_PC * r_integral[t - 1, u, v, n + 1]
                    r_integral[t, u, v, n] = value
    return r_integral[:, :, :, 0]


def boys_function(n: int, z: float) -> float:
    r"""Calculate Boys function."""
    if n == 0 and z < 10**-10:
        F = 1.0
    elif n == 0:
        F = (np.pi / (4.0 * z)) ** 0.5 * math.erf(z**0.5)
    elif z > 25:
        # Long range approximation
        F = factorial2(2 * n - 1) / (2.0 ** (n + 1.0)) * (np.pi / (z ** (2.0 * n + 1.0))) ** 0.5
    elif z < 10**-10:
        # special case of z = 0
        F = 1.0 / (2.0 * n + 1.0)
    else:
        F = 0.0
        threshold = 10**-12
        for i in range(0, 1000):
            Fcheck = F
            F += (2.0 * z) ** i / (factorial2(2 * n + 2 * i + 1))
            Fcheck -= F
            if abs(Fcheck) < threshold:
                break
        F *= np.exp(-z) * factorial2(2 * n - 1)
    return F
