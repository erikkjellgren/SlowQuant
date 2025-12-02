import numba as nb
import numpy as np



@nb.jit(nopython=True)
def RDM1(p: int, q: int, num_inactive_spin_orbs: int, num_active_spin_orbs: int, rdm1: np.ndarray) -> float:
    r"""Get full space one-electron reduced density matrix element.

    The only non-zero elements are:

    .. math::
        \Gamma^{[1]}_{pq} = \left\{\begin{array}{ll}
                            2\delta_{ij} & pq = ij\\
                            \left<0\left|\hat{E}_{vw}\right|0\right> & pq = vw\\
                            0 & \text{otherwise} \\
                            \end{array} \right.

    and the symmetry `\Gamma^{[1]}_{pq}=\Gamma^{[1]}_{qp}`:math:.

    Args:
        p: Spatial orbital index.
        q: Spatial orbital index.
        num_inactive_spin_orbs: Number of spatial inactive orbitals.
        num_active_spin_orbs: Number of spatial active orbitals.
        rdm1: Active part of 1-RDM.

    Returns:
        One-electron reduced density matrix element.
    """
    virt_start = num_inactive_spin_orbs + num_active_spin_orbs
    if p >= virt_start or q >= virt_start:
        # Zero if any virtual index
        return 0
    elif p >= num_inactive_spin_orbs and q >= num_inactive_spin_orbs:
        # All active index
        return rdm1[p - num_inactive_spin_orbs, q - num_inactive_spin_orbs]
    elif p < num_inactive_spin_orbs and q < num_inactive_spin_orbs:
        # All inactive indx
        if p == q:
            return 1
        return 0
    # One inactive and one active index
    return 0


@nb.jit(nopython=True)
def RDM2(
    p: int,
    q: int,
    r: int,
    s: int,
    num_inactive_spin_orbs: int,
    num_active_spin_orbs: int,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
) -> float:
    r"""Get full space two-electron reduced density matrix element.

    .. math::
        \Gamma^{[2]}_{pqrs} = \left\{\begin{array}{ll}
                              4\delta_{ij}\delta_{kl} - 2\delta_{jk}\delta_{il} & pqrs = ijkl\\
                              2\delta_{ij} \Gamma^{[1]}_{vw} & pqrs = vwij\\
                              - \delta_{ij}\Gamma^{[1]}_{vw} & pqrs = ivwj\\
                              \left<0\left|\hat{e}_{vwxy}\right|0\right> & pqrs = vwxy\\
                              0 & \text{otherwise} \\
                              \end{array} \right.

    and the symmetry `\Gamma^{[2]}_{pqrs}=\Gamma^{[2]}_{rspq}=\Gamma^{[2]}_{qpsr}=\Gamma^{[2]}_{srqp}`:math:.

    Args:
        p: Spatial orbital index.
        q: Spatial orbital index.
        r: Spatial orbital index.
        s: Spatial orbital index.
        num_inactive_spin_orbs: Number of spatial inactive orbitals.
        num_active_spin_orbs: Number of spatial active orbitals.
        rdm1: Active part of 1-RDM.
        rdm2: Active part of 2-RDM.

    Returns:
        Two-electron reduced density matrix element.
    """
    virt_start = num_inactive_spin_orbs + num_active_spin_orbs
    if p >= virt_start or q >= virt_start or r >= virt_start or s >= virt_start:
        # Zero if any virtual index
        return 0
    elif (
        p >= num_inactive_spin_orbs
        and q >= num_inactive_spin_orbs
        and r >= num_inactive_spin_orbs
        and s >= num_inactive_spin_orbs
    ):
        return rdm2[
            p - num_inactive_spin_orbs,
            q - num_inactive_spin_orbs,
            r - num_inactive_spin_orbs,
            s - num_inactive_spin_orbs,
        ]
    elif (
        p < num_inactive_spin_orbs and q >= num_inactive_spin_orbs and r >= num_inactive_spin_orbs and s < num_inactive_spin_orbs
    ):
        # iuvj type index
        if p == s:
            return -rdm1[q - num_inactive_spin_orbs, r - num_inactive_spin_orbs]
        return 0
    elif (
        p >= num_inactive_spin_orbs and q < num_inactive_spin_orbs and r < num_inactive_spin_orbs and s >= num_inactive_spin_orbs
    ):
        # uijv type index
        if q == r:
            return -rdm1[p - num_inactive_spin_orbs, s - num_inactive_spin_orbs]
        return 0
    elif (
        p >= num_inactive_spin_orbs and q >= num_inactive_spin_orbs and r < num_inactive_spin_orbs and s < num_inactive_spin_orbs
    ):
        # uvij type index
        if r == s:
            return rdm1[p - num_inactive_spin_orbs, q - num_inactive_spin_orbs]
        return 0
    elif (
        p < num_inactive_spin_orbs and q < num_inactive_spin_orbs and r >= num_inactive_spin_orbs and s >= num_inactive_spin_orbs
    ):
        # ijuv type index
        if p == q:
            return rdm1[r - num_inactive_spin_orbs, s - num_inactive_spin_orbs]
        return 0
    elif p < num_inactive_spin_orbs and q < num_inactive_spin_orbs and r < num_inactive_spin_orbs and s < num_inactive_spin_orbs:
        # All inactive index
        val = 0
        if p == q and r == s:
            val = 1
        if q == r and p == s:
            val = -1
        return val
    # Everything else
    return 0


@nb.jit(nopython=True)
def get_orbital_gradient(
    h_int: np.ndarray,
    g_int: np.ndarray,
    kappa_idx: list[tuple[int, int]],
    num_inactive_spin_orbs: int,
    num_active_spin_orbs: int,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
) -> tuple[np.ndarray]:
    r"""Calculate the first order orbital gradient.

    .. math::
        g_{pq}^{\hat{\kappa}} = \left<0\left|\left[\hat{\kappa}_{pq},\hat{H}\right]\right|0\right>

    Args:
        h_int: One-electron integrals in MO in Hamiltonian.
        g_int: Two-electron integrals in MO in Hamiltonian.
        kappa_idx: Orbital parameter indices in Spin basis.
        num_inactive_spin_orbs: Number of inactive orbitals in spin basis.
        num_active_spin_orbs: Number of active orbitals in spin basis.
        rdm1: Active part of 1-RDM.
        rdm2: Active part of 2-RDM.

    Returns:
        Orbital gradient.
    """
    gradient_R = np.zeros(len(kappa_idx))
    gradient_I = np.zeros(len(kappa_idx))
    for idx, (Q, P) in enumerate(kappa_idx):
        # 1e contribution
        for T in range(num_inactive_spin_orbs + num_active_spin_orbs):
            if Q == P:
                # Imaginary
                gradient_I[idx] += h_int[P, T] * RDM1(P, T, num_inactive_spin_orbs, num_active_spin_orbs, rdm1)
                gradient_I[idx] -= h_int[T, P] * RDM1(T, P, num_inactive_spin_orbs, num_active_spin_orbs, rdm1)
            else:
                # Real
                gradient_R[idx] += h_int[Q, T] * RDM1(P, T, num_inactive_spin_orbs, num_active_spin_orbs, rdm1)
                gradient_R[idx] += h_int[T, Q] * RDM1(T, P, num_inactive_spin_orbs, num_active_spin_orbs, rdm1)

                gradient_R[idx] -= h_int[P, T] * RDM1(Q, T, num_inactive_spin_orbs, num_active_spin_orbs, rdm1)
                gradient_R[idx] -= h_int[P, Q] * RDM1(T, Q, num_inactive_spin_orbs, num_active_spin_orbs, rdm1)

                # Imaginary
                gradient_I[idx] += h_int[Q, T] * RDM1(P, T, num_inactive_spin_orbs, num_active_spin_orbs, rdm1)
                gradient_I[idx] += h_int[P, T] * RDM1(Q, T, num_inactive_spin_orbs, num_active_spin_orbs, rdm1)

                gradient_I[idx] -= h_int[T, P] * RDM1(T, Q, num_inactive_spin_orbs, num_active_spin_orbs, rdm1)
                gradient_I[idx] -= h_int[T, Q] * RDM1(T, P, num_inactive_spin_orbs, num_active_spin_orbs, rdm1)
        # 2e contribution
        for T in range(num_inactive_spin_orbs + num_active_spin_orbs):
            for R in range(num_inactive_spin_orbs + num_active_spin_orbs):
                for S in range(num_inactive_spin_orbs + num_active_spin_orbs):
                    if Q == P:
                        gradient_I[idx] += g_int[P, T, R, S] * RDM2(
                            P, T, R, S, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )
                        gradient_I[idx] -= g_int[T, S, R, P] * RDM2(
                            T, S, R, P, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )
                        gradient_I[idx] += g_int[T, R, P, S] * RDM2(
                            T, R, P, S, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )
                        gradient_I[idx] -= g_int[T, P, R, S] * RDM2(
                            T, P, R, S, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )
                    else:
                        # Real
                        gradient_I[idx] += g_int[Q, T, R, S] * RDM2(
                            P, T, R, S, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )
                        gradient_I[idx] -= g_int[S, T, R, P] * RDM2(
                            S, T, R, Q, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )
                        gradient_I[idx] += g_int[S, T, R, Q] * RDM2(
                            S, T, R, P, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )
                        gradient_I[idx] -= g_int[P, T, R, S] * RDM2(
                            Q, T, R, S, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )
                        gradient_I[idx] += g_int[T, Q, R, S] * RDM2(
                            T, P, R, S, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )
                        gradient_I[idx] -= g_int[T, P, R, S] * RDM2(
                            T, Q, R, S, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )
                        gradient_I[idx] += g_int[T, R, Q, S] * RDM2(
                            T, R, P, S, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )
                        gradient_I[idx] -= g_int[T, R, P, S] * RDM2(
                            T, R, Q, S, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )

                        # Imaginary
                        gradient_I[idx] += g_int[Q, T, R, S] * RDM2(
                            P, T, R, S, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )
                        gradient_I[idx] -= g_int[T, S, R, P] * RDM2(
                            T, S, R, Q, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )
                        gradient_I[idx] += g_int[T, R, Q, S] * RDM2(
                            T, R, P, S, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )
                        gradient_I[idx] -= g_int[T, P, R, S] * RDM2(
                            T, Q, R, S, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )
                        gradient_I[idx] += g_int[P, T, R, S] * RDM2(
                            Q, T, R, S, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )
                        gradient_I[idx] -= g_int[T, S, R, Q] * RDM2(
                            T, S, R, P, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )
                        gradient_I[idx] += g_int[T, R, P, S] * RDM2(
                            T, R, Q, S, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )
                        gradient_I[idx] -= g_int[T, Q, R, S] * RDM2(
                            T, P, R, S, num_inactive_spin_orbs, num_active_spin_orbs, rdm1, rdm2
                        )

                        
    return gradient_R, 1j*gradient_I

