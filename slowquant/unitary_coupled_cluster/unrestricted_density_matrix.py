import numba as nb
import numpy as np
import io

@nb.jit(nopython=True)
def RDM1xx(p: int, q: int, num_inactive_orbs: int, num_active_orbs: int, rdm1xx: np.ndarray) -> float:
    r"""Get one-electron unrestricted reduced density matrix element

    The only non-zero elements are:

    .. math::
        \Gamma^{[1]}_{pq} = \left\{\begin{array}{ll}
                            2\delta_{ij} & pq = ij\\
                            \left<0\left|\hat{E}_{vw}\right|0\right> & pq = vw\\
                            0 & \text{otherwise} \\
                            \end{array} \right.

    and the symmetry `\Gamma^{[1]}_{pq}=\Gamma^{[1]}_{qp}`:math:.


    Args:
        p: Spatial orbital index
        q: Spatial orbital index

    Returns:
        One-electron unrestricted reduced density matrix element.
    """
    virt_start = num_inactive_orbs + num_active_orbs
    if p >= virt_start or q >= virt_start:
        # Zero if any virtual index
        return 0
    elif p >= num_inactive_orbs and q >= num_inactive_orbs:
        # All active index
        return rdm1xx[p - num_inactive_orbs, q - num_inactive_orbs]
    elif p < num_inactive_orbs and q < num_inactive_orbs:
        # All inactive index
        if p == q:
            return 1
        return 0
    # One inactive and one active index
    return 0


@nb.jit(nopython=True)
def RDM2xxxx(
    p: int,
    r: int,
    s: int,
    q: int,
    num_inactive_orbs: int,
    num_active_orbs: int,
    rdm1xx: np.ndarray,
    rdm2xxxx: np.ndarray,
) -> float:
    r"""Get two-elelctron unrestricted reduced density matrix element.

    .. math::
    \Gamma^{[2]}_{p_{\sigma}q_{\sigma}r_{\tau}s_{\tau}} = \left\{\begin{array}{ll}
                              \delta_{ij}\delta_{\sigma \sigma}\delta_{kl}\delta_{\tau \tau} - \delta_{il}\delta_{\sigma \tau}\delta_{kj}\delta_{\tau \sigma} & pqrs = ijkl\\
                              \delta_{ij}\delta_{\tau\tau} \Gamma^{[1]}_{v_{\sigma}w_{\sigma}} & pqrs = vwij\\
                               - \delta_{ij}\delta_{\sigma\tau}\Gamma^{[1]}_{v_{\tau}w_{\sigma}} & pqrs = ivwj\\
                              \left<0\left|a^{\dagger}_{v_{\sigma}}a^{\dagger}_{x_{\tau}}a_{y_{\tau}}a_{w_{\sigma}}\right|0\right> & pqrs = vwxy\\
                              0 & \text{otherwise} \\
                              \end{array} .. math:.

    Args:
        p: Spatial orbital index
        q: Spatial orbital index
        r: Spatial obrital index
        s: Spatial orbital index

    Returns:
        Two-electron unrestricted reduced density matrix element.
    """
    virt_start = num_inactive_orbs + num_active_orbs
    if p >= virt_start or q >= virt_start or r >= virt_start or s >= virt_start:
        # Zero if any virtual index
        return 0
    elif (
        p >= num_inactive_orbs
        and q >= num_inactive_orbs
        and r >= num_inactive_orbs
        and s >= num_inactive_orbs
    ):
        return rdm2xxxx[
            p - num_inactive_orbs,
            q - num_inactive_orbs,
            r - num_inactive_orbs,
            s - num_inactive_orbs,
        ]
    elif (
        p < num_inactive_orbs and q >= num_inactive_orbs and r >= num_inactive_orbs and s < num_inactive_orbs
    ):
        if p == s:
            return -rdm1xx[r - num_inactive_orbs, q - num_inactive_orbs]
        return 0
    elif (
        p >= num_inactive_orbs and q < num_inactive_orbs and r < num_inactive_orbs and s >= num_inactive_orbs
    ):
        if q == r:
            return -rdm1xx[p - num_inactive_orbs, s - num_inactive_orbs]
        return 0
    elif (
        p >= num_inactive_orbs and q >= num_inactive_orbs and r < num_inactive_orbs and s < num_inactive_orbs
    ):
        if r == s:
            return rdm1xx[p - num_inactive_orbs, q - num_inactive_orbs]
        return 0
    elif (
        p < num_inactive_orbs and q < num_inactive_orbs and r >= num_inactive_orbs and s >= num_inactive_orbs
    ):
        if p == q:
            return rdm1xx[r - num_inactive_orbs, s - num_inactive_orbs]
        return 0
    if p < num_inactive_orbs and q < num_inactive_orbs and r < num_inactive_orbs and s < num_inactive_orbs:
        val = 0
        if p == q and r == s:
            val += 1
        if p == s and q == r:
            val -= 1
        return val
    return 0


@nb.jit(nopython=True)
def RDM2xxyy(
    p: int,
    r: int,
    s: int,
    q: int,
    num_inactive_orbs: int,
    num_active_orbs: int,
    rdm1xx: np.ndarray,
    rdm1yy: np.ndarray,
    rdm2xxyy: np.ndarray,
) -> float:
    r"""Get two-elelctron unrestricted reduced density matrix element.

    .. math::
    \Gamma^{[2]}_{p_{\sigma}q_{\sigma}r_{\tau}s_{\tau}} = \left\{\begin{array}{ll}
                              \delta_{ij}\delta_{\sigma \sigma}\delta_{kl}\delta_{\tau \tau} - \delta_{il}\delta_{\sigma \tau}\delta_{kj}\delta_{\tau \sigma} & pqrs = ijkl\\
                              \delta_{ij}\delta_{\tau\tau} \Gamma^{[1]}_{v_{\sigma}w_{\sigma}} & pqrs = vwij\\
                               - \delta_{ij}\delta_{\sigma\tau}\Gamma^{[1]}_{v_{\tau}w_{\sigma}} & pqrs = ivwj\\
                              \left<0\left|a^{\dagger}_{v_{\sigma}}a^{\dagger}_{x_{\tau}}a_{y_{\tau}}a_{w_{\sigma}}\right|0\right> & pqrs = vwxy\\
                              0 & \text{otherwise} \\
                              \end{array} .. math:.

    Args:
        p: Spatial orbital index
        q: Spatial orbital index
        r: Spatial obrital index
        s: Spatial orbital index

    Returns:
        Two-electron unrestricted reduced density matrix element.
    """
    virt_start = num_inactive_orbs + num_active_orbs
    if p >= virt_start or q >= virt_start or r >= virt_start or s >= virt_start:
        # Zero if any virtual index
        return 0
    elif (
        p >= num_inactive_orbs
        and q >= num_inactive_orbs
        and r >= num_inactive_orbs
        and s >= num_inactive_orbs
    ):
        return rdm2xxyy[
            p - num_inactive_orbs,
            q - num_inactive_orbs,
            r - num_inactive_orbs,
            s - num_inactive_orbs,
        ]
    elif (
        p >= num_inactive_orbs and q >= num_inactive_orbs and r < num_inactive_orbs and s < num_inactive_orbs
    ):
        if r == s:
            return rdm1xx[p - num_inactive_orbs, q - num_inactive_orbs]
        return 0
    elif (
        p < num_inactive_orbs and q < num_inactive_orbs and r >= num_inactive_orbs and s >= num_inactive_orbs
    ):
        if p == q:
            return rdm1yy[r - num_inactive_orbs, s - num_inactive_orbs]
        return 0
    elif p < num_inactive_orbs and q < num_inactive_orbs and r < num_inactive_orbs and s < num_inactive_orbs:
        if p == q and r == s:
            return 1
        return 0
    return 0


@nb.jit(nopython=True)
def get_electronic_energy_unrestricted(
    h_int_aa: np.ndarray,
    h_int_bb: np.ndarray,
    g_int_aaaa: np.ndarray,
    g_int_bbbb: np.ndarray,
    g_int_aabb: np.ndarray,
    g_int_bbaa: np.ndarray,
    num_inactive_orbs: int,
    num_active_orbs: int,
    rdm1aa: np.ndarray,
    rdm1bb: np.ndarray,
    rdm2aaaa: np.ndarray,
    rdm2bbbb: np.ndarray,
    rdm2aabb: np.ndarray,
    rdm2bbaa: np.ndarray,
) -> float:
    """Calcualte unrestricted electronic energy.

    .. math::
        E = \\sum_{pq}(h_{pq,\alpha}\\Gamma^{[1]}_{pq, \alpha} + h_{pq}_{\beta}\\Gamma^{[1]}_{pq, \beta})
            + \frac{1}{2}\\sum_{pqrs}(g_{pqrs, \alpha\alpha}\\Gamma^{[2]}_{pqrs, \alpha\alpha} + g_{pqrs, \beta\beta}\\Gamma^{[2]}_{pqrs, \beta\beta}
            + g_{pqrs, \alpha\beta}\\Gamma^{[2]}_{pqrs, \alpha\beta} + g_{pqrs, \beta\alpha}\\Gamma^{[2]}_{pqrs, \beta\alpha})

    Args:
        h_int: One-electron integrals in MO. For alpha alpha (aa) and beta beta (bb)
        g_int: Two-electron integrals in MO. For aaaa, bbbb, aabb, bbaa
        num_inactive_orbs: Number of inactive orbitals.
        num_active_orbs: Number of active orbitals.

    Returns:
        Electronic energy.
    """
    energy = 0
    for p in range(num_inactive_orbs + num_active_orbs):
        for q in range(num_inactive_orbs + num_active_orbs):
            energy += h_int_aa[p, q] * RDM1xx(p, q, num_inactive_orbs, num_active_orbs, rdm1aa) + h_int_bb[
                p, q
            ] * RDM1xx(p, q, num_inactive_orbs, num_active_orbs, rdm1bb)
    for p in range(num_inactive_orbs + num_active_orbs):
        for q in range(num_inactive_orbs + num_active_orbs):
            for r in range(num_inactive_orbs + num_active_orbs):
                for s in range(num_inactive_orbs + num_active_orbs):
                    energy += (
                        1
                        / 2
                        * (
                            g_int_aaaa[p, q, r, s]
                            * RDM2xxxx(p, r, s, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                            + g_int_bbbb[p, q, r, s]
                            * RDM2xxxx(p, r, s, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                            + g_int_aabb[p, q, r, s]
                            * RDM2xxyy(
                                p, r, s, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb
                            )
                            + g_int_bbaa[p, q, r, s]
                            * RDM2xxyy(
                                p, r, s, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa
                            )
                        )
                    )
    return energy


@nb.jit(nopython=True)
def get_orbital_gradient_unrestricted(
    h_int_aa: np.ndarray,
    h_int_bb: np.ndarray,
    g_int_aaaa: np.ndarray,
    g_int_bbbb: np.ndarray,
    g_int_aabb: np.ndarray,
    g_int_bbaa: np.ndarray,
    kappa_idx: list[tuple[int, int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
    rdm1aa: np.ndarray,
    rdm1bb: np.ndarray,
    rdm2aaaa: np.ndarray,
    rdm2bbbb: np.ndarray,
    rdm2aabb: np.ndarray,
    rdm2bbaa: np.ndarray,
) -> np.ndarray:
    r"""Calculate the unrestricted orbital gradient.

    .. math::
        g^{\hat{\kappa}_{mn}} = \left<0\left|\left[\hat{\kappa}_{mn},\hat{H}\right]\right|0\right>

    Args:
        rdms: Unrestricted reduced density matrix class.
        h_int: One-electron integrals in MO in Hamiltonian. aa and bb
        g_int: Two-electron integrals in MO in Hamiltonian. aaaa, bbbb, aabb, bbaa
        kappa_idx: Orbital parameter indices in spatial basis.
        num_inactive_orbs: Number of inactive orbitals in spatial basis.
        num_active_orbs: Number of active orbitals in spatial basis.

    Returns:
        Unrestricted orbital gradient.

    """
    gradient = np.zeros(2 * len(kappa_idx))
    for idx, (m, n) in enumerate(kappa_idx):
        # 1e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            gradient[idx] += h_int_aa[n, p] * RDM1xx(m, p, num_inactive_orbs, num_active_orbs, rdm1aa)
            gradient[idx] -= h_int_aa[m, p] * RDM1xx(n, p, num_inactive_orbs, num_active_orbs, rdm1aa)
            gradient[idx] -= h_int_aa[p, m] * RDM1xx(p, n, num_inactive_orbs, num_active_orbs, rdm1aa)
            gradient[idx] += h_int_aa[p, n] * RDM1xx(p, m, num_inactive_orbs, num_active_orbs, rdm1aa)
            gradient[idx + len(kappa_idx)] += h_int_bb[n, p] * RDM1xx(
                m, p, num_inactive_orbs, num_active_orbs, rdm1bb
            )
            gradient[idx + len(kappa_idx)] -= h_int_bb[m, p] * RDM1xx(
                n, p, num_inactive_orbs, num_active_orbs, rdm1bb
            )
            gradient[idx + len(kappa_idx)] -= h_int_bb[p, m] * RDM1xx(
                p, n, num_inactive_orbs, num_active_orbs, rdm1bb
            )
            gradient[idx + len(kappa_idx)] += h_int_bb[p, n] * RDM1xx(
                p, m, num_inactive_orbs, num_active_orbs, rdm1bb
            )
        # 2e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            for q in range(num_inactive_orbs + num_active_orbs):
                for r in range(num_inactive_orbs + num_active_orbs):
                    # aaaa
                    gradient[idx] += (
                        0.5
                        * g_int_aaaa[n, p, q, r]
                        * RDM2xxxx(m, q, r, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx] -= (
                        0.5
                        * g_int_aaaa[p, q, n, r]
                        * RDM2xxxx(m, p, r, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx] -= (
                        0.5
                        * g_int_aaaa[m, p, q, r]
                        * RDM2xxxx(n, q, r, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx] += (
                        0.5
                        * g_int_aaaa[p, q, m, r]
                        * RDM2xxxx(n, p, r, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx] -= (
                        0.5
                        * g_int_aaaa[p, m, q, r]
                        * RDM2xxxx(p, q, r, n, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx] += (
                        0.5
                        * g_int_aaaa[p, q, r, m]
                        * RDM2xxxx(p, r, q, n, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx] += (
                        0.5
                        * g_int_aaaa[p, n, q, r]
                        * RDM2xxxx(p, q, r, m, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx] -= (
                        0.5
                        * g_int_aaaa[p, q, r, n]
                        * RDM2xxxx(p, r, q, m, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    # bbbb
                    gradient[idx + len(kappa_idx)] += (
                        0.5
                        * g_int_bbbb[n, p, q, r]
                        * RDM2xxxx(m, q, r, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + len(kappa_idx)] -= (
                        0.5
                        * g_int_bbbb[p, q, n, r]
                        * RDM2xxxx(m, p, r, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + len(kappa_idx)] -= (
                        0.5
                        * g_int_bbbb[m, p, q, r]
                        * RDM2xxxx(n, q, r, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + len(kappa_idx)] += (
                        0.5
                        * g_int_bbbb[p, q, m, r]
                        * RDM2xxxx(n, p, r, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + len(kappa_idx)] -= (
                        0.5
                        * g_int_bbbb[p, m, q, r]
                        * RDM2xxxx(p, q, r, n, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + len(kappa_idx)] += (
                        0.5
                        * g_int_bbbb[p, q, r, m]
                        * RDM2xxxx(p, r, q, n, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + len(kappa_idx)] += (
                        0.5
                        * g_int_bbbb[p, n, q, r]
                        * RDM2xxxx(p, q, r, m, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + len(kappa_idx)] -= (
                        0.5
                        * g_int_bbbb[p, q, r, n]
                        * RDM2xxxx(p, r, q, m, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    # aabb
                    # kappa a with aabb
                    gradient[idx] += (
                        0.5
                        * g_int_aabb[n, p, q, r]
                        * RDM2xxyy(m, q, r, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    gradient[idx] -= (
                        0.5
                        * g_int_aabb[m, p, q, r]
                        * RDM2xxyy(n, q, r, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    gradient[idx] -= (
                        0.5
                        * g_int_aabb[p, m, q, r]
                        * RDM2xxyy(p, q, r, n, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    gradient[idx] += (
                        0.5
                        * g_int_aabb[p, n, q, r]
                        * RDM2xxyy(p, q, r, m, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    # kappa a with bbaa
                    gradient[idx] += (
                        0.5
                        * g_int_bbaa[p, q, n, r]
                        * RDM2xxyy(p, m, r, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    gradient[idx] -= (
                        0.5
                        * g_int_bbaa[p, q, m, r]
                        * RDM2xxyy(p, n, r, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    gradient[idx] -= (
                        0.5
                        * g_int_bbaa[p, q, r, m]
                        * RDM2xxyy(p, r, n, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    gradient[idx] += (
                        0.5
                        * g_int_bbaa[p, q, r, n]
                        * RDM2xxyy(p, r, m, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    # kappa b with aabb
                    gradient[idx + len(kappa_idx)] += (
                        0.5
                        * g_int_aabb[p, q, n, r]
                        * RDM2xxyy(p, m, r, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    gradient[idx + len(kappa_idx)] -= (
                        0.5
                        * g_int_aabb[p, q, m, r]
                        * RDM2xxyy(p, n, r, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    gradient[idx + len(kappa_idx)] -= (
                        0.5
                        * g_int_aabb[p, q, r, m]
                        * RDM2xxyy(p, r, n, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    gradient[idx + len(kappa_idx)] += (
                        0.5
                        * g_int_aabb[p, q, r, n]
                        * RDM2xxyy(p, r, m, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    # kappa b with bbaa
                    gradient[idx + len(kappa_idx)] += (
                        0.5
                        * g_int_bbaa[n, p, q, r]
                        * RDM2xxyy(m, q, r, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    gradient[idx + len(kappa_idx)] -= (
                        0.5
                        * g_int_bbaa[m, p, q, r]
                        * RDM2xxyy(n, q, r, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    gradient[idx + len(kappa_idx)] -= (
                        0.5
                        * g_int_bbaa[p, m, q, r]
                        * RDM2xxyy(p, q, r, n, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    gradient[idx + len(kappa_idx)] += (
                        0.5
                        * g_int_bbaa[p, n, q, r]
                        * RDM2xxyy(p, q, r, m, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
    return gradient


@nb.jit(nopython=True)
def get_orbital_gradient_response_unrestricted(
    h_int_aa: np.ndarray,
    h_int_bb: np.ndarray,
    g_int_aaaa: np.ndarray,
    g_int_bbbb: np.ndarray,
    g_int_aabb: np.ndarray,
    g_int_bbaa: np.ndarray,
    kappa_idx: list[tuple[int, int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
    rdm1aa: np.ndarray,
    rdm1bb: np.ndarray,
    rdm2aaaa: np.ndarray,
    rdm2bbbb: np.ndarray,
    rdm2aabb: np.ndarray,
    rdm2bbaa: np.ndarray,
) -> np.ndarray:
    r"""Calculate the unrestricted response orbital parameter gradient.

    ..math::
        g^{\hat{q_{mn}}} = \left<0\left|\left[\hat{q}_{mn},\hat{H}\right]\right|0\right>

    Args:
        rdms: Unrestricted reduced density matrix class.
        h_int_aa: One-electron integrals in MO in Hamiltonian, alpha part.
        h_int_bb: One-electron integrals in MO in Hamiltonian, beta part.
        g_int_aaaa: Two-electron integrals in MO in Hamiltonian, alpha-alpha part.
        g_int_bbbb: Two-electron integrals in MO in Hamiltonian, beta-beta part.
        g_int_aabb: Two-electron integrals in MO in Hamiltonian, alpha-beta part.
        g_int_bbaa: Two-electron integrals in MO in Hamiltonian, beta-alpha part.
        num_inactive_orbs: Number of inactive orbitals in spatial basis.
        num_active_orbs: Number of active orbitals in spatial basis.

    Returns:
        Unrestricted response parameter gradient

    """
    gradient = np.zeros(4 * len(kappa_idx))
    shift = len(kappa_idx)
    shift2 = len(2 * kappa_idx)
    shift3 = len(3 * kappa_idx)
    for idx, (m, n) in enumerate(kappa_idx):
        # 1e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            gradient[idx] += h_int_aa[n, p] * RDM1xx(m, p, num_inactive_orbs, num_active_orbs, rdm1aa)
            gradient[idx] -= h_int_aa[m, p] * RDM1xx(n, p, num_inactive_orbs, num_active_orbs, rdm1aa)
            gradient[idx] -= h_int_aa[p, m] * RDM1xx(p, n, num_inactive_orbs, num_active_orbs, rdm1aa)
            gradient[idx] += h_int_aa[p, n] * RDM1xx(p, m, num_inactive_orbs, num_active_orbs, rdm1aa)
            gradient[idx + shift] += h_int_bb[n, p] * RDM1xx(m, p, num_inactive_orbs, num_active_orbs, rdm1bb)
            gradient[idx + shift] -= h_int_bb[m, p] * RDM1xx(n, p, num_inactive_orbs, num_active_orbs, rdm1bb)
            gradient[idx + shift] -= h_int_bb[p, m] * RDM1xx(p, n, num_inactive_orbs, num_active_orbs, rdm1bb)
            gradient[idx + shift] += h_int_bb[p, n] * RDM1xx(p, m, num_inactive_orbs, num_active_orbs, rdm1bb)
        # 2e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            for q in range(num_inactive_orbs + num_active_orbs):
                for r in range(num_inactive_orbs + num_active_orbs):
                    # aaaa
                    gradient[idx] += (
                        0.5
                        * g_int_aaaa[n, p, q, r]
                        * RDM2xxxx(m, q, r, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx] -= (
                        0.5
                        * g_int_aaaa[p, q, n, r]
                        * RDM2xxxx(m, p, r, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx] -= (
                        0.5
                        * g_int_aaaa[m, p, q, r]
                        * RDM2xxxx(n, q, r, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx] += (
                        0.5
                        * g_int_aaaa[p, q, m, r]
                        * RDM2xxxx(n, p, r, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx] -= (
                        0.5
                        * g_int_aaaa[p, m, q, r]
                        * RDM2xxxx(p, q, r, n, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx] += (
                        0.5
                        * g_int_aaaa[p, q, r, m]
                        * RDM2xxxx(p, r, q, n, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx] += (
                        0.5
                        * g_int_aaaa[p, n, q, r]
                        * RDM2xxxx(p, q, r, m, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx] -= (
                        0.5
                        * g_int_aaaa[p, q, r, n]
                        * RDM2xxxx(p, r, q, m, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    # bbbb
                    gradient[idx + shift] += (
                        0.5
                        * g_int_bbbb[n, p, q, r]
                        * RDM2xxxx(m, q, r, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + shift] -= (
                        0.5
                        * g_int_bbbb[p, q, n, r]
                        * RDM2xxxx(m, p, r, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + shift] -= (
                        0.5
                        * g_int_bbbb[m, p, q, r]
                        * RDM2xxxx(n, q, r, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + shift] += (
                        0.5
                        * g_int_bbbb[p, q, m, r]
                        * RDM2xxxx(n, p, r, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + shift] -= (
                        0.5
                        * g_int_bbbb[p, m, q, r]
                        * RDM2xxxx(p, q, r, n, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + shift] += (
                        0.5
                        * g_int_bbbb[p, q, r, m]
                        * RDM2xxxx(p, r, q, n, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + shift] += (
                        0.5
                        * g_int_bbbb[p, n, q, r]
                        * RDM2xxxx(p, q, r, m, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + shift] -= (
                        0.5
                        * g_int_bbbb[p, q, r, n]
                        * RDM2xxxx(p, r, q, m, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    # aabb
                    # kappa a with aabb
                    gradient[idx] += (
                        0.5
                        * g_int_aabb[n, p, q, r]
                        * RDM2xxyy(m, q, r, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    gradient[idx] -= (
                        0.5
                        * g_int_aabb[m, p, q, r]
                        * RDM2xxyy(n, q, r, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    gradient[idx] -= (
                        0.5
                        * g_int_aabb[p, m, q, r]
                        * RDM2xxyy(p, q, r, n, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    gradient[idx] += (
                        0.5
                        * g_int_aabb[p, n, q, r]
                        * RDM2xxyy(p, q, r, m, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    # kappa a with bbaa
                    gradient[idx] += (
                        0.5
                        * g_int_bbaa[p, q, n, r]
                        * RDM2xxyy(p, m, r, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    gradient[idx] -= (
                        0.5
                        * g_int_bbaa[p, q, m, r]
                        * RDM2xxyy(p, n, r, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    gradient[idx] -= (
                        0.5
                        * g_int_bbaa[p, q, r, m]
                        * RDM2xxyy(p, r, n, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    gradient[idx] += (
                        0.5
                        * g_int_bbaa[p, q, r, n]
                        * RDM2xxyy(p, r, m, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    # kappa b with aabb
                    gradient[idx + shift] += (
                        0.5
                        * g_int_aabb[p, q, n, r]
                        * RDM2xxyy(p, m, r, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    gradient[idx + shift] -= (
                        0.5
                        * g_int_aabb[p, q, m, r]
                        * RDM2xxyy(p, n, r, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    gradient[idx + shift] -= (
                        0.5
                        * g_int_aabb[p, q, r, m]
                        * RDM2xxyy(p, r, n, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    gradient[idx + shift] += (
                        0.5
                        * g_int_aabb[p, q, r, n]
                        * RDM2xxyy(p, r, m, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    # kappa b with bbaa
                    gradient[idx + shift] += (
                        0.5
                        * g_int_bbaa[n, p, q, r]
                        * RDM2xxyy(m, q, r, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    gradient[idx + shift] -= (
                        0.5
                        * g_int_bbaa[m, p, q, r]
                        * RDM2xxyy(n, q, r, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    gradient[idx + shift] -= (
                        0.5
                        * g_int_bbaa[p, m, q, r]
                        * RDM2xxyy(p, q, r, n, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    gradient[idx + shift] += (
                        0.5
                        * g_int_bbaa[p, n, q, r]
                        * RDM2xxyy(p, q, r, m, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )

    for idx, (n, m) in enumerate(kappa_idx):
        # 1e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            gradient[idx+ shift2] += h_int_aa[n, p] * RDM1xx(
                m, p, num_inactive_orbs, num_active_orbs, rdm1aa
            )
            gradient[idx+ shift2] -= h_int_aa[m, p] * RDM1xx(
                n, p, num_inactive_orbs, num_active_orbs, rdm1aa
            )
            gradient[idx+ shift2] -= h_int_aa[p, m] * RDM1xx(
                p, n, num_inactive_orbs, num_active_orbs, rdm1aa
            )
            gradient[idx+ shift2] += h_int_aa[p, n] * RDM1xx(
                p, m, num_inactive_orbs, num_active_orbs, rdm1aa
            )
            gradient[idx + shift3] += h_int_bb[n, p] * RDM1xx(
                m, p, num_inactive_orbs, num_active_orbs, rdm1bb
            )
            gradient[idx + shift3] -= h_int_bb[m, p] * RDM1xx(
                n, p, num_inactive_orbs, num_active_orbs, rdm1bb
            )
            gradient[idx + shift3] -= h_int_bb[p, m] * RDM1xx(
                p, n, num_inactive_orbs, num_active_orbs, rdm1bb
            )
            gradient[idx + shift3] += h_int_bb[p, n] * RDM1xx(
                p, m, num_inactive_orbs, num_active_orbs, rdm1bb
            )
        # 2e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            for q in range(num_inactive_orbs + num_active_orbs):
                for r in range(num_inactive_orbs + num_active_orbs):
                    # aaaa
                    gradient[idx+ shift2] += (
                        0.5
                        * g_int_aaaa[n, p, q, r]
                        * RDM2xxxx(m, q, r, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx+ shift2] -= (
                        0.5
                        * g_int_aaaa[p, q, n, r]
                        * RDM2xxxx(m, p, r, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx+ shift2] -= (
                        0.5
                        * g_int_aaaa[m, p, q, r]
                        * RDM2xxxx(n, q, r, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx+ shift2] += (
                        0.5
                        * g_int_aaaa[p, q, m, r]
                        * RDM2xxxx(n, p, r, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx+ shift2] -= (
                        0.5
                        * g_int_aaaa[p, m, q, r]
                        * RDM2xxxx(p, q, r, n, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx+ shift2] += (
                        0.5
                        * g_int_aaaa[p, q, r, m]
                        * RDM2xxxx(p, r, q, n, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx+ shift2] += (
                        0.5
                        * g_int_aaaa[p, n, q, r]
                        * RDM2xxxx(p, q, r, m, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    gradient[idx+ shift2] -= (
                        0.5
                        * g_int_aaaa[p, q, r, n]
                        * RDM2xxxx(p, r, q, m, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa)
                    )
                    # bbbb
                    gradient[idx + shift3] += (
                        0.5
                        * g_int_bbbb[n, p, q, r]
                        * RDM2xxxx(m, q, r, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + shift3] -= (
                        0.5
                        * g_int_bbbb[p, q, n, r]
                        * RDM2xxxx(m, p, r, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + shift3] -= (
                        0.5
                        * g_int_bbbb[m, p, q, r]
                        * RDM2xxxx(n, q, r, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + shift3] += (
                        0.5
                        * g_int_bbbb[p, q, m, r]
                        * RDM2xxxx(n, p, r, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + shift3] -= (
                        0.5
                        * g_int_bbbb[p, m, q, r]
                        * RDM2xxxx(p, q, r, n, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + shift3] += (
                        0.5
                        * g_int_bbbb[p, q, r, m]
                        * RDM2xxxx(p, r, q, n, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + shift3] += (
                        0.5
                        * g_int_bbbb[p, n, q, r]
                        * RDM2xxxx(p, q, r, m, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    gradient[idx + shift3] -= (
                        0.5
                        * g_int_bbbb[p, q, r, n]
                        * RDM2xxxx(p, r, q, m, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb)
                    )
                    # aabb
                    # kappa a with aabb
                    gradient[idx + shift2] += (
                        0.5
                        * g_int_aabb[n, p, q, r]
                        * RDM2xxyy(m, q, r, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    gradient[idx + shift2] -= (
                        0.5
                        * g_int_aabb[m, p, q, r]
                        * RDM2xxyy(n, q, r, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    gradient[idx + shift2] -= (
                        0.5
                        * g_int_aabb[p, m, q, r]
                        * RDM2xxyy(p, q, r, n, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    gradient[idx + shift2] += (
                        0.5
                        * g_int_aabb[p, n, q, r]
                        * RDM2xxyy(p, q, r, m, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    # kappa a with bbaa
                    gradient[idx + shift2] += (
                        0.5
                        * g_int_bbaa[p, q, n, r]
                        * RDM2xxyy(p, m, r, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    gradient[idx + shift2] -= (
                        0.5
                        * g_int_bbaa[p, q, m, r]
                        * RDM2xxyy(p, n, r, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    gradient[idx + shift2] -= (
                        0.5
                        * g_int_bbaa[p, q, r, m]
                        * RDM2xxyy(p, r, n, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    gradient[idx + shift2] += (
                        0.5
                        * g_int_bbaa[p, q, r, n]
                        * RDM2xxyy(p, r, m, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    # kappa b with aabb
                    gradient[idx + shift3] += (
                        0.5
                        * g_int_aabb[p, q, n, r]
                        * RDM2xxyy(p, m, r, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    gradient[idx + shift3] -= (
                        0.5
                        * g_int_aabb[p, q, m, r]
                        * RDM2xxyy(p, n, r, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    gradient[idx + shift3] -= (
                        0.5
                        * g_int_aabb[p, q, r, m]
                        * RDM2xxyy(p, r, n, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    gradient[idx + shift3] += (
                        0.5
                        * g_int_aabb[p, q, r, n]
                        * RDM2xxyy(p, r, m, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb)
                    )
                    # kappa b with bbaa
                    gradient[idx + shift3] += (
                        0.5
                        * g_int_bbaa[n, p, q, r]
                        * RDM2xxyy(m, q, r, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    gradient[idx + shift3] -= (
                        0.5
                        * g_int_bbaa[m, p, q, r]
                        * RDM2xxyy(n, q, r, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    gradient[idx + shift3] -= (
                        0.5
                        * g_int_bbaa[p, m, q, r]
                        * RDM2xxyy(p, q, r, n, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
                    gradient[idx + shift3] += (
                        0.5
                        * g_int_bbaa[p, n, q, r]
                        * RDM2xxyy(p, q, r, m, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa)
                    )
    return gradient


@nb.jit(nopython=True)
def get_orbital_response_metric_sigma_unrestricted(
    kappa_idx: list[tuple[int, int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
    rdm1aa: np.ndarray,
    rdm1bb: np.ndarray,
) -> np.ndarray:
    r"""Calculate the unrestricted orbital-orbital block of the sigma matrix.

    .. math::
        \Sigma_{pq,mn}^{\hat{q}^{\sigma},\hat{q}^{\nu}} = \left<0\left|\left[\hat{q}^{\sigma}_{pq}^\dagger,\hat{q}^{\nu}_{mn}\right]\right|0\right>

    Args:
       rdms: Unresticted reduced density matrix class.
       kappa_idx: Orbital parameter indicies in spatial basis.

    Returns:
        Unrestricted sigma matrix orbital-orbital block.
    """
    sigma = np.zeros((2 * len(kappa_idx), 2 * len(kappa_idx)))
    for idx1, (q, p) in enumerate(kappa_idx):
        for idx2, (m, n) in enumerate(kappa_idx):
            if p == n:
                sigma[idx1, idx2] -= RDM1xx(m, q, num_inactive_orbs, num_active_orbs, rdm1aa)
                sigma[idx1 + len(kappa_idx), idx2 + len(kappa_idx)] -= RDM1xx(m, q, num_inactive_orbs, num_active_orbs, rdm1bb)
            if m == q:
                sigma[idx1, idx2] += RDM1xx(p, n, num_inactive_orbs, num_active_orbs, rdm1aa)
                sigma[idx1 + len(kappa_idx), idx2 + len(kappa_idx)] += RDM1xx(p, n, num_inactive_orbs, num_active_orbs, rdm1bb)
    return sigma


@nb.jit(nopython=True)
def get_orbital_response_property_gradient_unrestricted(
    mo_a: np.ndarray,
    mo_b: np.ndarray,
    kappa_idx: list[tuple[int, int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
    rdm1aa: np.ndarray,
    rdm1bb: np.ndarray,
    response_vectors: np.ndarray,
    state_number: int,
    number_excitations: int,
) -> np.ndarray:
    r"""Calculate the orbital part of property gradient for oscillator strengths.

    .. math::
       P^{\hat{\mu}_{\gamma}} = \sum_k \left<0\left|\left[\hat{\mu}_{\gamma}, \hat{O}_k\right]\right|0\right>
       Where
       \hat{O}_k =\sum_{\mu}(Z_{k, \mu}\hat{q}_{\mu}^{\dagger} + Y_{k, \mu}\hat{q}_{\mu})

    Args:
        rdms: Unrestricted reduced density matrix class.
        mo: Property integral in MO basis.
        kappa_idx: Orbital parameter indicies in spatial basis.
        num_inactive_orbs: Number of inactive orbitals in spatial basis.
        num_active_orbs: Number of active orbitals in spatial basis.
        response_vectors: Response vectors.
        state_number: state number counting from zero.
        number_excitations: Total number of excitations.

    Returns:
        Unrestricted orbital part of property gradient for oscillator strengths.
    """
    prop_grad = 0
    for idx, (m, n) in enumerate(kappa_idx):
        for p in range(num_inactive_orbs, num_active_orbs):
            prop_grad += (
                (
                    response_vectors[idx, state_number]
                    - response_vectors[idx + number_excitations, state_number]
                )
                * mo_a[n, p]
                * RDM1xx(m, p, num_inactive_orbs, num_active_orbs, rdm1aa)
            )
            prop_grad -= (
                (
                    response_vectors[idx, state_number]
                    - response_vectors[idx + number_excitations, state_number]
                )
                * mo_a[p, m]
                * RDM1xx(p, n, num_inactive_orbs, num_active_orbs, rdm1aa)
            )
            prop_grad += (
                (
                    response_vectors[idx, state_number]
                    - response_vectors[idx + number_excitations, state_number]
                )
                * mo_b[n, p]
                * RDM1xx(m, p, num_inactive_orbs, num_active_orbs, rdm1bb)
            )
            prop_grad -= (
                (
                    response_vectors[idx, state_number]
                    - response_vectors[idx + number_excitations, state_number]
                )
                * mo_b[p, m]
                * RDM1xx(p, n, num_inactive_orbs, num_active_orbs, rdm1bb)
            )
    return prop_grad


# @nb.jit(nopython=True)
def get_orbital_response_hessian_block_unrestricted(
    h_int_aa: np.ndarray,
    h_int_bb: np.ndarray,
    g_int_aaaa: np.ndarray,
    g_int_bbbb: np.ndarray,
    g_int_aabb: np.ndarray,
    g_int_bbaa: np.ndarray,
    kappa_idx1: list[tuple[int, int]],
    kappa_idx2: list[tuple[int, int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
    rdm1aa: np.ndarray,
    rdm1bb: np.ndarray,
    rdm2aaaa: np.ndarray,
    rdm2bbbb: np.ndarray,
    rdm2aabb: np.ndarray,
    rdm2bbaa: np.ndarray,
) -> np.ndarray:
    r"""Calculate the unrestricted Hessian-like orbital-orbital block.

    .. math::
        H^{\hat{q}^{\mu},\hat{q}^{\nu}}_{tu,mn} = \left<0\left|\left[\hat{q}^{\mu}_{tu},\left[\hat{H},\hat{q}^{\nu}_{mn}\right]\right]\right|0\right>

    Args:
       rdms: Reduced density matrix class.
       h_int_aa: One-electron integrals in MO in Hamiltonian, alpha part.
       h_int_bb: One-electron integrals in MO in Hamiltonian, beta part.
       g_int_aaaa: Two-electron integrals in MO in Hamiltonian, alpha-alpha part.
       g_int_bbbb: Two-electron integrals in MO in Hamiltonian, beta-beta part.
       g_int_aabb: Two-electron integrals in MO in Hamiltonian, alpha-beta part.
       g_int_bbaa: Two-electron integrals in MO in Hamiltonian, beta-alpha part.
       kappa_idx1: Orbital parameter indicies in spatial basis.
       kappa_idx2: Orbital parameter indicies in spatial basis.
       num_inactive_orbs: Number of inactive orbitals in spatial basis.
       num_active_orbs: Number of active orbitals in spatial basis.

    Returns:
        Unrestricted Hessian-like orbital-orbital block.
    """
    # with io.open("/mnt/c/Users/Pernille/Seafile/phd/code/SlowQuant/slowquant/b.txt", 'w', encoding='utf-8') as file:
    #     file.write(f"start\n\n")
    # file.close()
    A1e = np.zeros((2 * len(kappa_idx1), 2 * len(kappa_idx2)))
    A2e = np.zeros((2 * len(kappa_idx1), 2 * len(kappa_idx2)))
    for idx1, (t, u) in enumerate(kappa_idx1):
        for idx2, (m, n) in enumerate(kappa_idx2):
            # with io.open("/mnt/c/Users/Pernille/Seafile/phd/code/SlowQuant/slowquant/b.txt", 'a+', encoding='utf-8') as file:
            #     file.write(f"idx1:{idx1} and idx2:{idx2}\n\n")                
            # file.close()
            A1e[idx1, idx2] += h_int_aa[u, m] * RDM1xx(
                t, n, num_inactive_orbs, num_active_orbs, rdm1aa
            )
            A1e[idx1, idx2] += h_int_aa[n, t] * RDM1xx(
                m, u, num_inactive_orbs, num_active_orbs, rdm1aa
            )
            A1e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] += h_int_bb[u, m] * RDM1xx(
                t, n, num_inactive_orbs, num_active_orbs, rdm1bb
            )
            A1e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] += h_int_bb[n, t] * RDM1xx(
                m, u, num_inactive_orbs, num_active_orbs, rdm1bb
            )
            for p in range(num_inactive_orbs + num_active_orbs):
                if m == u:
                    A1e[idx1, idx2] -= h_int_aa[n, p] * RDM1xx(
                        t, p, num_inactive_orbs, num_active_orbs, rdm1aa
                    )
                    A1e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] -= h_int_bb[n, p] * RDM1xx(
                        t, p, num_inactive_orbs, num_active_orbs, rdm1bb
                    )
                if t == n:
                    A1e[idx1, idx2] -= h_int_aa[p, m] * RDM1xx(
                        p, u, num_inactive_orbs, num_active_orbs, rdm1aa
                    )
                    A1e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] -= h_int_bb[p, m] * RDM1xx(
                        p, u, num_inactive_orbs, num_active_orbs, rdm1bb
                    )
            # et forsøg på at lave hessian matrix symmetrisk
            # print(idx1, idx2)
            # A1e[idx1, idx2] = A1e[idx2, idx1]
            # A1e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] = A1e[idx2 + len(kappa_idx1), idx1 + len(kappa_idx1)]
            for p in range(num_inactive_orbs + num_active_orbs):
                for q in range(num_inactive_orbs + num_active_orbs):
                    # mu, nu, sigma, tau = alpha
                    A2e[idx1, idx2] += g_int_aaaa[n, p, u, q] * RDM2xxxx(
                        t, m, q, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa
                    )
                    A2e[idx1, idx2] += g_int_aaaa[n, t, p, q] * RDM2xxxx(
                        m, p, q, u, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa
                    )
                    A2e[idx1, idx2] -= g_int_aaaa[n, p, q, t] * RDM2xxxx(
                        m, q, p, u, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa
                    )
                    A2e[idx1, idx2] -= g_int_aaaa[u, p, n, q] * RDM2xxxx(
                        t, m, q, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa
                    )
                    A2e[idx1, idx2] -= g_int_aaaa[p, t, n, q] * RDM2xxxx(
                        m, p, q, u, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa
                    )
                    A2e[idx1, idx2] += g_int_aaaa[p, q, n, t] * RDM2xxxx(
                        m, p, q, u, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa
                    )
                    A2e[idx1, idx2] += g_int_aaaa[u, m, p, q] * RDM2xxxx(
                        t, p, q, n, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa
                    )
                    A2e[idx1, idx2] -= g_int_aaaa[p, m, u, q] * RDM2xxxx(
                        t, p, q, n, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa
                    )
                    A2e[idx1, idx2] += g_int_aaaa[p, m, q, t] * RDM2xxxx(
                        p, q, n, u, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa
                    )
                    A2e[idx1, idx2] -= g_int_aaaa[u, p, q, m] * RDM2xxxx(
                        t, q, p, n, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa
                    )
                    A2e[idx1, idx2] += g_int_aaaa[p, q, u, m] * RDM2xxxx(
                        t, p, q, n, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa
                    )
                    A2e[idx1, idx2] -= g_int_aaaa[p, t, q, m] * RDM2xxxx(
                        p, q, n, u, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa
                    )

                    # mu, nu, sigma, tau = beta
                    A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] += g_int_bbbb[n, p, u, q] * RDM2xxxx(
                        t, m, q, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb
                    )
                    A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] += g_int_bbbb[n, t, p, q] * RDM2xxxx(
                        m, p, q, u, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb
                    )
                    A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] -= g_int_bbbb[n, p, q, t] * RDM2xxxx(
                        m, q, p, u, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb
                    )
                    A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] -= g_int_bbbb[u, p, n, q] * RDM2xxxx(
                        t, m, q, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb
                    )
                    A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] -= g_int_bbbb[p, t, n, q] * RDM2xxxx(
                        m, p, q, u, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb
                    )
                    A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] += g_int_bbbb[p, q, n, t] * RDM2xxxx(
                        m, p, q, u, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb
                    )
                    A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] += g_int_bbbb[u, m, p, q] * RDM2xxxx(
                        t, p, q, n, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb
                    )
                    A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] -= g_int_bbbb[p, m, u, q] * RDM2xxxx(
                        t, p, q, n, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb
                    )
                    A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] += g_int_bbbb[p, m, q, t] * RDM2xxxx(
                        p, q, n, u, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb
                    )
                    A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] -= g_int_bbbb[u, p, q, m] * RDM2xxxx(
                        t, q, p, n, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb
                    )
                    A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] += g_int_bbbb[p, q, u, m] * RDM2xxxx(
                        t, p, q, n, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb
                    )
                    A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] -= g_int_bbbb[p, t, q, m] * RDM2xxxx(
                        p, q, n, u, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb
                    )

                    # nu, mu, tau = alpha, sigma = beta
                    A2e[idx1, idx2] += g_int_bbaa[p, q, u, m] * RDM2xxyy(
                        t, p, q, n, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb
                    )
                    A2e[idx1, idx2] += g_int_bbaa[p, q, n, t] * RDM2xxyy(
                        m, p, q, u, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb
                    )

                    # nu, mu, sigma = alpha, tau = beta
                    A2e[idx1, idx2] += g_int_aabb[u, m, p, q] * RDM2xxyy(
                        t, p, q, n, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb
                    )
                    A2e[idx1, idx2] += g_int_aabb[n, t, p, q] * RDM2xxyy(
                        m, p, q, u, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb
                    )

                    # nu, mu, tau = beta, sigma = alpha
                    A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] += g_int_aabb[p, q, u, m] * RDM2xxyy(
                        t, p, q, n, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa
                    )
                    A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] += g_int_aabb[p, q, n, t] * RDM2xxyy(
                        m, p, q, u, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa
                    )

                    # nu, mu, sigma = beta, tau = alpha
                    A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] += g_int_bbaa[u, m, p, q] * RDM2xxyy(
                        t, p, q, n, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa
                    )
                    A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] += g_int_bbaa[n, t, p, q] * RDM2xxyy(
                        m, p, q, u, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa
                    )

                    # mu, sigma = beta, nu, tau = alpha
                    A2e[idx1 + len(kappa_idx1), idx2] += g_int_bbaa[u, p, q, m] * RDM2xxyy(
                        t, q, n, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa
                    )  
                    A2e[idx1 + len(kappa_idx1), idx2] -= g_int_bbaa[u, p, n, q] * RDM2xxyy(
                        t, m, q, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa
                    )
                    A2e[idx1 + len(kappa_idx1), idx2] -= g_int_bbaa[p, t, q, m] * RDM2xxyy(
                        p, q, n, u, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa
                    )
                    A2e[idx1 + len(kappa_idx1), idx2] += g_int_bbaa[p, t, n, q] * RDM2xxyy(
                        p, m, q, u, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa
                    )  

                    # mu, tau = beta, nu, sigma = alpha
                    A2e[idx1 + len(kappa_idx1), idx2] += g_int_aabb[p, m, u, q] * RDM2xxyy(
                        p, t, q, n, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb
                    )  
                    A2e[idx1 + len(kappa_idx1), idx2] -= g_int_aabb[n, p, u, q] * RDM2xxyy(
                        m, t, q, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb
                    )  
                    A2e[idx1 + len(kappa_idx1), idx2] -= g_int_aabb[p, m, q, t] * RDM2xxyy(
                        p, q, u, n, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb
                    )  
                    A2e[idx1 + len(kappa_idx1), idx2] += g_int_aabb[n, p, q, t] * RDM2xxyy(
                        m, q, u, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb
                    )  

                    # mu, sigma = alpha, nu, tau = beta
                    A2e[idx1, idx2 + len(kappa_idx1)] += g_int_aabb[u, p, q, m] * RDM2xxyy(
                        t, q, n, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb
                    )  
                    A2e[idx1, idx2 + len(kappa_idx1)] -= g_int_aabb[u, p, n, q] * RDM2xxyy(
                        t, m, q, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb
                    )
                    A2e[idx1, idx2 + len(kappa_idx1)] -= g_int_aabb[p, t, q, m] * RDM2xxyy(
                        p, q, n, u, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb
                    )
                    A2e[idx1, idx2 + len(kappa_idx1)] += g_int_aabb[p, t, n, q] * RDM2xxyy(
                        p, m, q, u, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb
                    )  

                    # mu, tau = alpha, nu, sigma = beta
                    A2e[idx1, idx2 + len(kappa_idx1)] += g_int_bbaa[p, m, u, q] * RDM2xxyy(
                        p, t, q, n, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa
                    )  
                    A2e[idx1, idx2 + len(kappa_idx1)] -= g_int_bbaa[n, p, u, q] * RDM2xxyy(
                        m, t, q, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa
                    )  
                    A2e[idx1, idx2 + len(kappa_idx1)] -= g_int_bbaa[p, m, q, t] * RDM2xxyy(
                        p, q, u, n, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa
                    )  
                    A2e[idx1, idx2 + len(kappa_idx1)] += g_int_bbaa[n, p, q, t] * RDM2xxyy(
                        m, q, u, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa
                    )  

            for p in range(num_inactive_orbs + num_active_orbs):
                for q in range(num_inactive_orbs + num_active_orbs):
                    for r in range(num_inactive_orbs + num_active_orbs):
                        if m == u:
                            A2e[idx1, idx2] -= g_int_aaaa[n, p, q, r] * RDM2xxxx(
                                t, q, r, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa
                            )
                            A2e[idx1, idx2] += g_int_aaaa[p, q, n, r] * RDM2xxxx(
                                t, p, r, q, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa
                            )
                            A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] -= g_int_bbbb[n, p, q, r] * RDM2xxxx(
                                t, q, r, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb
                            )
                            A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] += g_int_bbbb[p, q, n, r] * RDM2xxxx(
                                t, p, r, q, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb
                            )
                            A2e[idx1, idx2] -= g_int_bbaa[p, q, n, r] * RDM2xxyy(
                                t, p, q, r, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb
                            )  
                            A2e[idx1, idx2] -= g_int_aabb[n, p, q, r] * RDM2xxyy(
                                t, q, r, p, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb
                            )
                            A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] -= g_int_aabb[p, q, n, r] * RDM2xxyy(
                                t, p, q, r, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa
                            )  
                            A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] -= g_int_bbaa[n, p, q, r] * RDM2xxyy(
                                t, q, r, p, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa
                            )
                        if t == n:
                            A2e[idx1, idx2] -= g_int_aaaa[p, m, q, r] * RDM2xxxx(
                                p, q, r, u, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa
                            )
                            A2e[idx1, idx2] += g_int_aaaa[p, q, r, m] * RDM2xxxx(
                                p, r, q, u, num_inactive_orbs, num_active_orbs, rdm1aa, rdm2aaaa
                            )
                            A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] -= g_int_bbbb[p, m, q, r] * RDM2xxxx(
                                p, q, r, u, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb
                            )
                            A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] += g_int_bbbb[p, q, r, m] * RDM2xxxx(
                                p, r, q, u, num_inactive_orbs, num_active_orbs, rdm1bb, rdm2bbbb
                            )
                            A2e[idx1, idx2] -= g_int_bbaa[p, q, r, m] * RDM2xxyy(
                                r, p, q, u, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb
                            )  
                            A2e[idx1, idx2] -= g_int_aabb[p, m, q, r] * RDM2xxyy(
                                p, q, r, u, num_inactive_orbs, num_active_orbs, rdm1aa, rdm1bb, rdm2aabb
                            )
                            A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] -= g_int_aabb[p, q, r, m] * RDM2xxyy(
                                r, p, q, u, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa
                            )  
                            A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] -= g_int_bbaa[p, m, q, r] * RDM2xxyy(
                                p, q, r, u, num_inactive_orbs, num_active_orbs, rdm1bb, rdm1aa, rdm2bbaa
                            )
            # A2e[idx1, idx2] = A2e[idx2, idx1]
            # A2e[idx1 + len(kappa_idx1), idx2 + len(kappa_idx1)] = A2e[idx2 + len(kappa_idx1), idx1 + len(kappa_idx1)]
            # A2e[idx1, idx2 + len(kappa_idx1)] = A2e[idx2 + len(kappa_idx1), idx1]
            # A2e[idx1 + len(kappa_idx1), idx2] = A2e[idx2, idx1 + len(kappa_idx1)]
            # with io.open("/mnt/c/Users/Pernille/Seafile/phd/code/SlowQuant/slowquant/b.txt", 'a+', encoding='utf-8') as file:
            #     file.write(f"{A1e + 1/2 * A2e}\n\n")                
            # file.close()                

    # return 1/2* (A1e + A1e.T + (1 / 2 * A2e) + (1/2*A2e.T))
    return A1e + (1/2 *A2e)

