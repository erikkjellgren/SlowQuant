import numpy as np


class ReducedDenstiyMatrix:
    """Reduced density matrix class."""

    __slots__ = ("inactive_idx", "virtual_idx", "active_idx", "idx_shift", "rdm1", "rdm2", "t_rdm2")

    def __init__(
        self,
        num_inactive_orbs: int,
        num_active_orbs: int,
        num_virtual_orbs: int,
        rdm1: np.ndarray,
        rdm2: np.ndarray | None = None,
        t_rdm2: np.ndarray | None = None,
    ) -> None:
        """Initialize reduced density matrix class.

        Args:
            num_inactive_orbs: Number of inactive orbitals in spatial basis.
            num_active_orbs: Number of active orbitals in spatial basis.
            num_virtual_orbs: Number of virtual orbitals in spatial basis.
            rdm1: One-electron reduced density matrix in the active space.
            rdm2: Two-electron reduced density matrix in the active space.
            t_rdm2: Triplet two-electron reduced density matrix in the active space.
        """
        self.inactive_idx = []
        self.active_idx = []
        self.virtual_idx = []
        for idx in range(num_inactive_orbs):
            self.inactive_idx.append(idx)
        for idx in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
            self.active_idx.append(idx)
        for idx in range(
            num_inactive_orbs + num_active_orbs,
            num_inactive_orbs + num_active_orbs + num_virtual_orbs,
        ):
            self.virtual_idx.append(idx)
        self.idx_shift = num_inactive_orbs
        self.rdm1 = rdm1
        self.rdm2 = rdm2
        self.t_rdm2 = t_rdm2

    def RDM1(self, p: int, q: int) -> float:
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

        Returns:
            One-electron reduced density matrix element.
        """
        if p in self.active_idx and q in self.active_idx:
            return self.rdm1[p - self.idx_shift, q - self.idx_shift]
        if p in self.inactive_idx and q in self.inactive_idx:
            if p == q:
                return 2
            return 0
        return 0

    def RDM2(self, p: int, q: int, r: int, s: int) -> float:  # pylint: disable=R0911
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

        Returns:
            Two-electron reduced density matrix element.
        """
        if self.rdm2 is None:
            raise ValueError("RDM2 is not given.")
        if p in self.active_idx and q in self.active_idx and r in self.active_idx and s in self.active_idx:
            return self.rdm2[
                p - self.idx_shift,
                q - self.idx_shift,
                r - self.idx_shift,
                s - self.idx_shift,
            ]
        if (
            p in self.inactive_idx
            and q in self.active_idx
            and r in self.active_idx
            and s in self.inactive_idx
        ):
            if p == s:
                return -self.rdm1[q - self.idx_shift, r - self.idx_shift]
            return 0
        if (
            p in self.active_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.active_idx
        ):
            if q == r:
                return -self.rdm1[p - self.idx_shift, s - self.idx_shift]
            return 0
        if (
            p in self.active_idx
            and q in self.active_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
        ):
            if r == s:
                return 2 * self.rdm1[p - self.idx_shift, q - self.idx_shift]
            return 0
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.active_idx
            and s in self.active_idx
        ):
            if p == q:
                return 2 * self.rdm1[r - self.idx_shift, s - self.idx_shift]
            return 0
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
        ):
            val = 0
            if p == q and r == s:
                val += 4
            if q == r and p == s:
                val -= 2
            return val
        return 0

    def t_RDM2(self, p: int, q: int, r: int, s: int) -> float:  # pylint: disable=R0911
        r"""Get full space triplet two-electron reduced density matrix element.

        .. math::
            \Gamma^{T [2]}_{pqrs} = \left\{\begin{array}{ll}
                                  - 2\delta_{jk}\delta_{il} & pqrs = ijkl\\
                                  - \delta_{ij}\Gamma^{[1]}_{vw} & pqrs = ivwj\\
                                  \left<0\left|\hat{e}_{vwxy}\right|0\right> & pqrs = vwxy\\
                                  0 & \text{otherwise} \\
                                  \end{array} \right.

        and the symmetry `\Gamma^{T [2]}_{pqrs}=\Gamma^{T [2]}_{rspq}=\Gamma^{T [2]}_{qpsr}=\Gamma^{T [2]}_{srqp}`:math:.

        Args:
            p: Spatial orbital index.
            q: Spatial orbital index.
            r: Spatial orbital index.
            s: Spatial orbital index.

        Returns:
            Triplet two-electron reduced density matrix element.
        """
        if self.t_rdm2 is None:
            raise ValueError("triplet RDM2 is not given.")
        if p in self.active_idx and q in self.active_idx and r in self.active_idx and s in self.active_idx:
            return self.t_rdm2[
                p - self.idx_shift,
                q - self.idx_shift,
                r - self.idx_shift,
                s - self.idx_shift,
            ]
        if (
            p in self.inactive_idx
            and q in self.active_idx
            and r in self.active_idx
            and s in self.inactive_idx
        ):
            if p == s:
                return -self.rdm1[q - self.idx_shift, r - self.idx_shift]
            return 0
        if (
            p in self.active_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.active_idx
        ):
            if q == r:
                return -self.rdm1[p - self.idx_shift, s - self.idx_shift]
            return 0
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
        ):
            if q == r and p == s:
                return -2
            return 0
        return 0


def get_electronic_energy(
    rdms: ReducedDenstiyMatrix,
    h_int: np.ndarray,
    g_int: np.ndarray,
    num_inactive_orbs: int,
    num_active_orbs: int,
) -> float:
    r"""Calculate electronic energy.

    .. math::
        E = \sum_{pq}h_{pq}\Gamma^{[1]}_{pq} + \frac{1}{2}\sum_{pqrs}g_{pqrs}\Gamma^{[2]}_{pqrs}

    Args:
        h_int: One-electron integrals in MO.
        g_int: Two-electron integrals in MO.
        num_inactive_orbs: Number of inactive orbitals.
        num_active_orbs: Number of active orbitals.

    Returns:
        Electronic energy.
    """
    energy = 0
    for p in range(num_inactive_orbs + num_active_orbs):
        for q in range(num_inactive_orbs + num_active_orbs):
            energy += h_int[p, q] * rdms.RDM1(p, q)
    for p in range(num_inactive_orbs + num_active_orbs):
        for q in range(num_inactive_orbs + num_active_orbs):
            for r in range(num_inactive_orbs + num_active_orbs):
                for s in range(num_inactive_orbs + num_active_orbs):
                    energy += 1 / 2 * g_int[p, q, r, s] * rdms.RDM2(p, q, r, s)
    return energy


def get_orbital_gradient(
    rdms: ReducedDenstiyMatrix,
    h_int: np.ndarray,
    g_int: np.ndarray,
    kappa_idx: list[tuple[int, int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
) -> np.ndarray:
    r"""Calculate the orbital gradient.

    .. math::
        g_{pq}^{\hat{\kappa}} = \left<0\left|\left[\hat{\kappa}_{pq},\hat{H}\right]\right|0\right>

    Args:
       rdms: Reduced density matrix class.
       h_int: One-electron integrals in MO in Hamiltonian.
       g_int: Two-electron integrals in MO in Hamiltonian.
       kappa_idx: Orbital parameter indices in spatial basis.
       num_inactive_orbs: Number of inactive orbitals in spatial basis.
       num_active_orbs: Number of active orbitals in spatial basis.

    Returns:
        Orbital gradient.
    """
    gradient = np.zeros(len(kappa_idx))
    for idx, (m, n) in enumerate(kappa_idx):
        # 1e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            gradient[idx] += 2 * h_int[n, p] * rdms.RDM1(m, p)
            gradient[idx] -= 2 * h_int[p, m] * rdms.RDM1(p, n)
        # 2e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            for q in range(num_inactive_orbs + num_active_orbs):
                for r in range(num_inactive_orbs + num_active_orbs):
                    gradient[idx] += g_int[n, p, q, r] * rdms.RDM2(m, p, q, r)
                    gradient[idx] -= g_int[p, m, q, r] * rdms.RDM2(p, n, q, r)
                    gradient[idx] -= g_int[m, p, q, r] * rdms.RDM2(n, p, q, r)
                    gradient[idx] += g_int[p, n, q, r] * rdms.RDM2(p, m, q, r)
    return gradient


def get_orbital_gradient_response(
    rdms: ReducedDenstiyMatrix,
    h_int: np.ndarray,
    g_int: np.ndarray,
    kappa_idx: list[tuple[int, int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
) -> np.ndarray:
    r"""Calculate the response orbital parameter gradient.

    .. math::
        g_{pq}^{\hat{q}} = \left<0\left|\left[\hat{q}_{pq},\hat{H}\right]\right|0\right>

    Args:
       rdms: Reduced density matrix class.
       h_int: One-electron integrals in MO in Hamiltonian.
       g_int: Two-electron integrals in MO in Hamiltonian.
       kappa_idx: Orbital parameter indices in spatial basis.
       num_inactive_orbs: Number of inactive orbitals in spatial basis.
       num_active_orbs: Number of active orbitals in spatial basis.

    Returns:
        Orbital response parameter gradient.
    """
    gradient = np.zeros(2 * len(kappa_idx))
    for idx, (m, n) in enumerate(kappa_idx):
        # 1e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            gradient[idx] += h_int[n, p] * rdms.RDM1(m, p)
            gradient[idx] -= h_int[p, m] * rdms.RDM1(p, n)
        # 2e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            for q in range(num_inactive_orbs + num_active_orbs):
                for r in range(num_inactive_orbs + num_active_orbs):
                    gradient[idx] += 1 / 2 * g_int[n, p, q, r] * rdms.RDM2(m, p, q, r)
                    gradient[idx] -= 1 / 2 * g_int[p, m, q, r] * rdms.RDM2(p, n, q, r)
                    gradient[idx] -= 1 / 2 * g_int[m, p, q, r] * rdms.RDM2(n, p, q, r)
                    gradient[idx] += 1 / 2 * g_int[p, n, q, r] * rdms.RDM2(p, m, q, r)
    shift = len(kappa_idx)
    for idx, (n, m) in enumerate(kappa_idx):
        # 1e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            gradient[idx + shift] += h_int[n, p] * rdms.RDM1(m, p)
            gradient[idx + shift] -= h_int[p, m] * rdms.RDM1(p, n)
        # 2e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            for q in range(num_inactive_orbs + num_active_orbs):
                for r in range(num_inactive_orbs + num_active_orbs):
                    gradient[idx + shift] += 1 / 2 * g_int[n, p, q, r] * rdms.RDM2(m, p, q, r)
                    gradient[idx + shift] -= 1 / 2 * g_int[p, m, q, r] * rdms.RDM2(p, n, q, r)
                    gradient[idx + shift] -= 1 / 2 * g_int[m, p, q, r] * rdms.RDM2(n, p, q, r)
                    gradient[idx + shift] += 1 / 2 * g_int[p, n, q, r] * rdms.RDM2(p, m, q, r)
    return 2 ** (-1 / 2) * gradient


def get_orbital_response_metric_sigma(
    rdms: ReducedDenstiyMatrix, kappa_idx: list[tuple[int, int]]
) -> np.ndarray:
    r"""Calculate the Sigma matrix orbital-orbital block.

    .. math::
        \Sigma_{pq,pq}^{\hat{q},\hat{q}} = \left<0\left|\left[\hat{q}_{pq}^\dagger,\hat{q}_{pq}\right]\right|0\right>

    Args:
       rdms: Reduced density matrix class.
       kappa_idx: Orbital parameter indices in spatial basis.

    Returns:
        Sigma matrix orbital-orbital block.
    """
    sigma = np.zeros((len(kappa_idx), len(kappa_idx)))
    for idx1, (n, m) in enumerate(kappa_idx):
        for idx2, (p, q) in enumerate(kappa_idx):
            if p == n:
                sigma[idx1, idx2] += rdms.RDM1(m, q)
            if m == q:
                sigma[idx1, idx2] -= rdms.RDM1(p, n)
    return -1 / 2 * sigma


def get_orbital_response_vector_norm(
    rdms: ReducedDenstiyMatrix,
    kappa_idx: list[list[int]],
    response_vectors: np.ndarray,
    state_number: int,
    number_excitations: int,
) -> float:
    r"""Calculate the orbital part of excited state norm.

    .. math::
        N^{\hat{q}} = \sum_k\left<0\left|\left[\hat{O}_{k},\hat{O}_{k}^\dagger\right]\right|0\right>

    Args:
       rdms: Reduced density matrix class.
       kappa_idx: Orbital parameter indices in spatial basis.
       response_vectors: Response vectors.
       state_number: State number counting from zero.
       number_excitations: Total number of excitations.

    Returns:
        Orbital part of excited state norm.
    """
    norm = 0
    for i, (m, n) in enumerate(kappa_idx):
        for j, (t, u) in enumerate(kappa_idx):
            if n == u:
                norm += (
                    response_vectors[i, state_number] * response_vectors[j, state_number] * rdms.RDM1(m, t)
                )
            if m == t:
                norm -= (
                    response_vectors[i, state_number] * response_vectors[j, state_number] * rdms.RDM1(n, u)
                )
            if m == t:
                norm += (
                    response_vectors[i + number_excitations, state_number]
                    * response_vectors[j + number_excitations, state_number]
                    * rdms.RDM1(n, u)
                )
            if n == u:
                norm -= (
                    response_vectors[i + number_excitations, state_number]
                    * response_vectors[j + number_excitations, state_number]
                    * rdms.RDM1(m, t)
                )
    return 1 / 2 * norm


def get_orbital_response_property_gradient(
    rdms: ReducedDenstiyMatrix,
    x_mo: np.ndarray,
    kappa_idx: list[tuple[int, int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
    response_vectors: np.ndarray,
    state_number: int,
    number_excitations: int,
) -> float:
    r"""Calculate the orbital part of property gradient.

    .. math::
        P^{\hat{q}} = \sum_k\left<0\left|\left[\hat{O}_{k},\hat{X}\right]\right|0\right>

    Args:
       rdms: Reduced density matrix class.
       x_mo: Property integral in MO basis.
       kappa_idx: Orbital parameter indices in spatial basis.
       num_inactive_orbs: Number of inactive orbitals in spatial basis.
       num_active_orbs: Number of active orbitals in spatial basis.
       response_vectors: Response vectors.
       state_number: State number counting from zero.
       number_excitations: Total number of excitations.

    Returns:
        Orbital part of property gradient.
    """
    prop_grad = 0
    for i, (m, n) in enumerate(kappa_idx):
        for p in range(num_inactive_orbs + num_active_orbs):
            prop_grad += (
                (response_vectors[i + number_excitations, state_number] - response_vectors[i, state_number])
                * x_mo[n, p]
                * rdms.RDM1(m, p)
            )
            prop_grad += (
                (response_vectors[i, state_number] - response_vectors[i + number_excitations, state_number])
                * x_mo[m, p]
                * rdms.RDM1(n, p)
            )
    return 2 ** (-1 / 2) * prop_grad


def get_orbital_response_hessian_block(
    rdms: ReducedDenstiyMatrix,
    h: np.ndarray,
    g: np.ndarray,
    kappa_idx1: list[tuple[int, int]],
    kappa_idx2: list[tuple[int, int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
) -> np.ndarray:
    r"""Calculate Hessian-like orbital-orbital block.

    .. math::
        H^{\hat{q},\hat{q}}_{tu,mn} = \left<0\left|\left[\hat{q}_{tu},\left[\hat{H},\hat{q}_{mn}\right]\right]\right|0\right>

    Args:
       rdms: Reduced density matrix class.
       kappa_idx1: Orbital parameter indices in spatial basis.
       kappa_idx1: Orbital parameter indices in spatial basis.
       num_inactive_orbs: Number of inactive orbitals in spatial basis.
       num_active_orbs: Number of active orbitals in spatial basis.

    Returns:
        Hessian-like orbital-orbital block.
    """
    A1e = np.zeros((len(kappa_idx1), len(kappa_idx1)))
    A2e = np.zeros((len(kappa_idx1), len(kappa_idx1)))
    for idx1, (t, u) in enumerate(kappa_idx1):
        for idx2, (m, n) in enumerate(kappa_idx2):
            # 1e contribution
            A1e[idx1, idx2] += h[n, t] * rdms.RDM1(m, u)
            A1e[idx1, idx2] += h[u, m] * rdms.RDM1(t, n)
            for p in range(num_inactive_orbs + num_active_orbs):
                if m == u:
                    A1e[idx1, idx2] -= h[n, p] * rdms.RDM1(t, p)
                if t == n:
                    A1e[idx1, idx2] -= h[p, m] * rdms.RDM1(p, u)
            # 2e contribution
            for p in range(num_inactive_orbs + num_active_orbs):
                for q in range(num_inactive_orbs + num_active_orbs):
                    A2e[idx1, idx2] += g[n, t, p, q] * rdms.RDM2(m, u, p, q)
                    A2e[idx1, idx2] -= g[n, p, u, q] * rdms.RDM2(m, p, t, q)
                    A2e[idx1, idx2] += g[n, p, q, t] * rdms.RDM2(m, p, q, u)
                    A2e[idx1, idx2] += g[u, m, p, q] * rdms.RDM2(t, n, p, q)
                    A2e[idx1, idx2] += g[p, m, u, q] * rdms.RDM2(p, n, t, q)
                    A2e[idx1, idx2] -= g[p, m, q, t] * rdms.RDM2(p, n, q, u)
                    A2e[idx1, idx2] -= g[u, p, n, q] * rdms.RDM2(t, p, m, q)
                    A2e[idx1, idx2] += g[p, t, n, q] * rdms.RDM2(p, u, m, q)
                    A2e[idx1, idx2] += g[p, q, n, t] * rdms.RDM2(p, q, m, u)
                    A2e[idx1, idx2] += g[u, p, q, m] * rdms.RDM2(t, p, q, n)
                    A2e[idx1, idx2] -= g[p, t, q, m] * rdms.RDM2(p, u, q, n)
                    A2e[idx1, idx2] += g[p, q, u, m] * rdms.RDM2(p, q, t, n)
            for p in range(num_inactive_orbs + num_active_orbs):
                for q in range(num_inactive_orbs + num_active_orbs):
                    for r in range(num_inactive_orbs + num_active_orbs):
                        if m == u:
                            A2e[idx1, idx2] -= g[n, p, q, r] * rdms.RDM2(t, p, q, r)
                        if t == n:
                            A2e[idx1, idx2] -= g[p, m, q, r] * rdms.RDM2(p, u, q, r)
                        if m == u:
                            A2e[idx1, idx2] -= g[p, q, n, r] * rdms.RDM2(p, q, t, r)
                        if t == n:
                            A2e[idx1, idx2] -= g[p, q, r, m] * rdms.RDM2(p, q, r, u)
    return 1 / 2 * A1e + 1 / 4 * A2e


def get_triplet_orbital_response_hessian_block(
    rdms: ReducedDenstiyMatrix,
    h: np.ndarray,
    g: np.ndarray,
    kappa_idx1: list[tuple[int, int]],
    kappa_idx2: list[tuple[int, int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
) -> np.ndarray:
    r"""Calculate Hessian-like orbital-orbital block for triplet response.

    .. math::
        H^{\hat{q}^T,\hat{q}^T}_{tu,mn} = \left<0\left|\left[\hat{q}^T_{tu},\left[\hat{H},\hat{q}^T_{mn}\right]\right]\right|0\right>

    Args:
       rdms: Reduced density matrix class.
       kappa_idx1: Orbital parameter indicies in spatial basis.
       kappa_idx1: Orbital parameter indicies in spatial basis.
       num_inactive_orbs: Number of inactive orbitals in spatial basis.
       num_active_orbs: Number of active orbitals in spatial basis.

    Returns:
        Hessian-like triplet orbital-orbital block.
    """
    A1e = np.zeros((len(kappa_idx1), len(kappa_idx1)))
    A2e = np.zeros((len(kappa_idx1), len(kappa_idx1)))
    for idx1, (t, u) in enumerate(kappa_idx1):
        for idx2, (m, n) in enumerate(kappa_idx2):
            # 1e contribution
            A1e[idx1, idx2] += h[n, t] * rdms.RDM1(m, u)
            A1e[idx1, idx2] += h[u, m] * rdms.RDM1(t, n)
            for p in range(num_inactive_orbs + num_active_orbs):
                if m == u:
                    A1e[idx1, idx2] -= h[n, p] * rdms.RDM1(t, p)
                if t == n:
                    A1e[idx1, idx2] -= h[p, m] * rdms.RDM1(p, u)
            # 2e contribution
            for p in range(num_inactive_orbs + num_active_orbs):
                for q in range(num_inactive_orbs + num_active_orbs):
                    A2e[idx1, idx2] += g[n, t, p, q] * rdms.RDM2(m, u, p, q)
                    A2e[idx1, idx2] -= g[n, p, u, q] * rdms.t_RDM2(m, p, t, q)
                    A2e[idx1, idx2] += g[n, p, q, t] * rdms.t_RDM2(m, p, q, u)
                    A2e[idx1, idx2] += g[u, m, p, q] * rdms.RDM2(t, n, p, q)
                    A2e[idx1, idx2] += g[p, m, u, q] * rdms.t_RDM2(p, n, t, q)
                    A2e[idx1, idx2] -= g[p, m, q, t] * rdms.t_RDM2(p, n, q, u)
                    A2e[idx1, idx2] -= g[u, p, n, q] * rdms.t_RDM2(t, p, m, q)
                    A2e[idx1, idx2] += g[p, t, n, q] * rdms.t_RDM2(p, u, m, q)
                    A2e[idx1, idx2] += g[p, q, n, t] * rdms.RDM2(p, q, m, u)
                    A2e[idx1, idx2] += g[u, p, q, m] * rdms.t_RDM2(t, p, q, n)
                    A2e[idx1, idx2] -= g[p, t, q, m] * rdms.t_RDM2(p, u, q, n)
                    A2e[idx1, idx2] += g[p, q, u, m] * rdms.RDM2(p, q, t, n)
            for p in range(num_inactive_orbs + num_active_orbs):
                for q in range(num_inactive_orbs + num_active_orbs):
                    for r in range(num_inactive_orbs + num_active_orbs):
                        if m == u:
                            A2e[idx1, idx2] -= g[n, p, q, r] * rdms.RDM2(t, p, q, r)
                        if t == n:
                            A2e[idx1, idx2] -= g[p, m, q, r] * rdms.RDM2(p, u, q, r)
                        if m == u:
                            A2e[idx1, idx2] -= g[p, q, n, r] * rdms.RDM2(p, q, t, r)
                        if t == n:
                            A2e[idx1, idx2] -= g[p, q, r, m] * rdms.RDM2(p, q, r, u)
    return 1 / 2 * A1e + 1 / 4 * A2e


def get_orbital_response_static_property_gradient(
    rdms: ReducedDenstiyMatrix,
    mo: np.ndarray,
    kappa_idx: list[tuple[int, int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
) -> np.ndarray:
    r"""Calculate the orbital part of static property gradient.

    .. math::
        P^{\hat{q}} = \frac{1}{\sqrt{2}}\sum_{p}\left(x_{np}\Gamma^{[1]}_{mp} - x_{pm}\Gamma^{[1]}_{pn}\right)

    Args:
       rdms: Reduced density matrix class.
       mo: Property integral in MO basis.
       kappa_idx: Orbital parameter indicies in spatial basis.
       num_inactive_orbs: Number of inactive orbitals in spatial basis.
       num_active_orbs: Number of active orbitals in spatial basis.

    Returns:
        Orbital part of static property gradient.
    """
    prop_grad = np.zeros((len(kappa_idx), len(mo)))
    for idx, (n, m) in enumerate(kappa_idx):
        for p in range(num_inactive_orbs + num_active_orbs):
            prop_grad[idx, :] += mo[:, n, p] * rdms.RDM1(m, p)
            prop_grad[idx, :] -= mo[:, p, m] * rdms.RDM1(p, n)
    return 2 ** (-1 / 2) * prop_grad
