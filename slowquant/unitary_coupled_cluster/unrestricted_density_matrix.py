import numpy as np


class UnrestrictedReducedDensityMatrix:
    """Unrestricted reduced density matrix class """

    def __init__(
        self,
        num_inactive_orbs : int,
        num_active_orbs: int,
        num_virtual_orbs: int,
        rdm1aa: np.ndarray,
        rdm1bb: np.ndarray,
        rdm2aaaa: np.ndarray | None = None,
        rdm2bbbb: np.ndarray | None = None,
        rdm2aabb: np.ndarray | None = None,
        rdm2bbaa: np.ndarray | None = None,
    ) -> None:
        """Initialize unrestricted reduced density matrix class.
    
        Args: 
            num_inactive_orbs: Number of inactive orbitals in spatial basis.
            num_active_orbs: Number of active orbitals in spatial basis.
            num_virtual_orbs: Number of virtual orbitals in spatial basis.
            rdm1: One-electron reduced density matrix in the active space.
            rdm2: Two-electron reduced density matrix in the active space.  
        """ 
        self.inactive_idx = []
        self.actitve_idx = [] #skal der gøres noget ved at active er stavet forkert?
        self.virtual_idx = []
        for idx in range(num_inactive_orbs):
            self.inactive_idx.append(idx)
        for idx in range(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
            self.actitve_idx.append(idx)
        for idx in range(
            num_inactive_orbs + num_active_orbs,
            num_inactive_orbs + num_active_orbs + num_virtual_orbs,
        ):
            self.virtual_idx.append(idx)
        self.idx_shift = num_inactive_orbs
        self.rdm1aa = rdm1aa
        self.rdm1bb = rdm1bb
        self.rdm2aaaa = rdm2aaaa
        self.rdm2bbbb = rdm2bbbb
        self.rdm2aabb = rdm2aabb
        self.rdm2bbaa = rdm2bbaa
# what about rdm1_C, rdm1_S, rdm2_C? What are they used for?

    def RDM1aa(self, p: int, q: int) -> float:
        """Get one-electron unrestricted reduced density matrix element
        
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
        if p in self.actitve_idx and q in self.actitve_idx:
            return self.rdm1aa[p - self.idx_shift, q - self.idx_shift]
        if p in self.inactive_idx and q in self.inactive_idx:
            if p == q:
                return 1
            return 0
        return 0 

    def RDM1bb(self, p: int, q: int) -> float:
        """Get one-electron unrestricted reduced density matrix element
        
        Think about what the non-zero elements are
        
        Args:
            p: Spatial orbital index
            q: Spatial orbital index
            
        Returns:
            One-electron unrestricted reduced density matrix element.
         """
        if p in self.actitve_idx and q in self.actitve_idx:
            return self.rdm1bb[p - self.idx_shift, q - self.idx_shift]
        if p in self.inactive_idx and q in self.inactive_idx:
            if p == q:
                return 1
            return 0
        return 0        

    def RDM2aaaa(self, p: int, r: int, s: int, q: int) -> float:
        """Get two-elelctron unrestricted reduced density matrix element.
        
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

        if self.rdm2aaaa is None:
            raise ValueError("RDM2aaaa is not given.")
        if (
            p in self.actitve_idx
            and q in self.actitve_idx
            and r in self.actitve_idx
            and s in self.actitve_idx
        ):
            return self.rdm2aaaa[
                p - self.idx_shift,
                q - self.idx_shift,
                r - self.idx_shift,
                s - self.idx_shift,
            ]
        if (
            p in self.inactive_idx
            and q in self.actitve_idx
            and r in self.actitve_idx
            and s in self.inactive_idx
        ):
            if p == s:
                return -self.rdm1aa[r - self.idx_shift, q - self.idx_shift]
            return 0
        if (
            p in self.actitve_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.actitve_idx
        ):
            if q == r:
                return -self.rdm1aa[p - self.idx_shift, s - self.idx_shift]
            return 0
        if (
            p in self.actitve_idx
            and q in self.actitve_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
        ):
            if r == s:
                return self.rdm1aa[p - self.idx_shift, q - self.idx_shift]
            return 0
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.actitve_idx
            and s in self.actitve_idx
        ):
            if p == q:
                return self.rdm1aa[r - self.idx_shift, s - self.idx_shift]
            return 0
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
        ):
            val = 0
            if p == q and r == s:
                val += 1
            if p == s and q == r:
                val -= 1
            return val
        return 0

    def RDM2bbbb(self, p: int, r: int, s: int, q: int) -> float:
        """Get two-elelctron unrestricted reduced density matrix element.
        
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
        if self.rdm2bbbb is None:
            raise ValueError("RDM2bbbb is not given.")
        if (
            p in self.actitve_idx
            and q in self.actitve_idx
            and r in self.actitve_idx
            and s in self.actitve_idx
        ):
            return self.rdm2bbbb[
                p - self.idx_shift,
                q - self.idx_shift,
                r - self.idx_shift,
                s - self.idx_shift,
            ]
        if (
            p in self.inactive_idx
            and q in self.actitve_idx
            and r in self.actitve_idx
            and s in self.inactive_idx
        ):
            if p == s:
                return -self.rdm1bb[r - self.idx_shift, q - self.idx_shift]
            return 0
        if (
            p in self.actitve_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.actitve_idx
        ):
            if q == r:
                return -self.rdm1bb[p - self.idx_shift, s - self.idx_shift]
            return 0
        if (
            p in self.actitve_idx
            and q in self.actitve_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
        ):
            if r == s:
                return self.rdm1bb[p - self.idx_shift, q - self.idx_shift]
            return 0
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.actitve_idx
            and s in self.actitve_idx
        ):
            if p == q:
                return self.rdm1bb[r - self.idx_shift, s - self.idx_shift]
            return 0
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
        ):
            val = 0
            if p == q and r == s:
                val += 1
            if p == s and q == r:
                val -= 1
            return val
        return 0

    def RDM2aabb(self, p: int, r: int, s: int, q: int) -> float:
        """Get two-elelctron unrestricted reduced density matrix element.
        
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
        if self.rdm2aabb is None:
            raise ValueError("RDM2aabb is not given.")
        if (
            p in self.actitve_idx
            and q in self.actitve_idx
            and r in self.actitve_idx
            and s in self.actitve_idx
        ):
            return self.rdm2aabb[
                p - self.idx_shift,
                q - self.idx_shift,
                r - self.idx_shift,
                s - self.idx_shift,
            ]
        if (
            p in self.inactive_idx
            and q in self.actitve_idx
            and r in self.actitve_idx
            and s in self.inactive_idx
        ):
            return 0
        if (
            p in self.actitve_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.actitve_idx
        ):
            return 0
        if (
            p in self.actitve_idx
            and q in self.actitve_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
        ):
            if r == s:
                return self.rdm1aa[p - self.idx_shift, q - self.idx_shift]
            return 0
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.actitve_idx
            and s in self.actitve_idx
        ):
            if p == q:
                return self.rdm1bb[r - self.idx_shift, s - self.idx_shift]
            return 0
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
        ):
            if p == q and r == s:
                return 1
            return 0
        return 0

    def RDM2bbaa(self, p: int, r: int, s: int, q: int) -> float:
        """Get two-elelctron unrestricted reduced density matrix element.
        
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
        if self.rdm2bbaa is None:
            raise ValueError("RDM2bbaa is not given.")
        if (
            p in self.actitve_idx
            and q in self.actitve_idx
            and r in self.actitve_idx
            and s in self.actitve_idx
        ):
            return self.rdm2bbaa[
                p - self.idx_shift,
                q - self.idx_shift,
                r - self.idx_shift,
                s - self.idx_shift,
            ]
        if (
            p in self.inactive_idx
            and q in self.actitve_idx
            and r in self.actitve_idx
            and s in self.inactive_idx
        ):
            return 0
        if (
            p in self.actitve_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.actitve_idx
        ):
            return 0
        if (
            p in self.actitve_idx
            and q in self.actitve_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
        ):
            if r == s:
                return self.rdm1bb[p - self.idx_shift, q - self.idx_shift]
            return 0
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.actitve_idx
            and s in self.actitve_idx
        ):
            if p == q:
                return self.rdm1aa[r - self.idx_shift, s - self.idx_shift]
            return 0
        if (
            p in self.inactive_idx
            and q in self.inactive_idx
            and r in self.inactive_idx
            and s in self.inactive_idx
        ):
            if p == q and r == s:
                return 1
            return 0
        return 0

def get_electronic_energy_unrestricted(
    rdms : UnrestrictedReducedDensityMatrix,
    h_int_aa: np.ndarray,
    h_int_bb: np.ndarray,
    g_int_aaaa: np.ndarray,
    g_int_bbbb: np.ndarray,
    g_int_aabb: np.ndarray,
    g_int_bbaa: np.ndarray,
    num_inactive_orbs: int,
    num_active_orbs: int,
) -> float:
    """Calcualte unrestricted electronic energy.
    

    Args:
        h_int: One-electron integrals in MO. For alpha alpha (aa) and beta beta (bb)
        g_int: Two-electron integrals in MO. For the combinations aaaa, bbbb, aabb, bbaa
        num_inactive_orbs: Number of inactive orbitals.
        num_active_orbs: Number of active orbitals.

    Returns:
        Electronic energy.
    """
    energy = 0
    for p in range(num_inactive_orbs + num_active_orbs):
        for q in range(num_inactive_orbs + num_active_orbs):
            energy += h_int_aa[p, q] * rdms.RDM1aa(p, q) + h_int_bb[p, q] * rdms.RDM1bb(p, q)
    for p in range(num_inactive_orbs + num_active_orbs):
        for q in range(num_inactive_orbs + num_active_orbs):
            for r in range(num_inactive_orbs + num_active_orbs):
                for s in range(num_inactive_orbs + num_active_orbs):
                    energy += 1 / 2 * (g_int_aaaa[p, q, r, s] * rdms.RDM2aaaa(p, r, s, q) + g_int_bbbb[p, q, r, s] * rdms.RDM2bbbb(p, r, s, q) + g_int_aabb[p, q, r, s] * rdms.RDM2aabb(p, r, s, q) + g_int_bbaa[p, q, r, s] * rdms.RDM2bbaa(p, r, s, q))

    return energy

def get_orbital_gradient_unrestricted(
        rdms: UnrestrictedReducedDensityMatrix,
    h_int_aa: np.ndarray,
    h_int_bb: np.ndarray,
    g_int_aaaa: np.ndarray,
    g_int_bbbb: np.ndarray,
    g_int_aabb: np.ndarray,
    g_int_bbaa: np.ndarray,
    kappa_idx: list[list[int]],
    num_inactive_orbs: int,
    num_active_orbs: int,
) -> np.ndarray:
    """ Figure out your math first!"""
    gradient = np.zeros(2*len(kappa_idx))
    for idx, (m, n) in enumerate(kappa_idx):
        #1e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            gradient[idx] +=  h_int_aa[n, p] * rdms.RDM1aa(m, p)
            gradient[idx] -=  h_int_aa[m, p] * rdms.RDM1aa(n, p)
            gradient[idx] -=  h_int_aa[p, m] * rdms.RDM1aa(p, n)
            gradient[idx] +=  h_int_aa[p, n] * rdms.RDM1aa(p, m)
            gradient[idx+len(kappa_idx)] +=  h_int_bb[n, p] * rdms.RDM1bb(m, p)
            gradient[idx+len(kappa_idx)] -=  h_int_bb[m, p] * rdms.RDM1bb(n, p)
            gradient[idx+len(kappa_idx)] -=  h_int_bb[p, m] * rdms.RDM1bb(p, n)
            gradient[idx+len(kappa_idx)] +=  h_int_bb[p, n] * rdms.RDM1bb(p, m)
        #2e contribution
        for p in range(num_inactive_orbs + num_active_orbs):
            for q in range(num_inactive_orbs + num_active_orbs):
                for r in range(num_inactive_orbs + num_active_orbs):
                    #aaaa
                    gradient[idx] += 0.5 * g_int_aaaa[n, p, q, r] * rdms.RDM2aaaa(m, q, r, p)
                    gradient[idx] -= 0.5 * g_int_aaaa[p, q, n, r] * rdms.RDM2aaaa(m, p, r, q)
                    gradient[idx] -= 0.5 * g_int_aaaa[m, p, q, r] * rdms.RDM2aaaa(n, q, r, p)
                    gradient[idx] += 0.5 * g_int_aaaa[p, q, m, r] * rdms.RDM2aaaa(n, p, r, q)
                    gradient[idx] -= 0.5 * g_int_aaaa[p, m, q, r] * rdms.RDM2aaaa(p, q, r, n)
                    gradient[idx] += 0.5 * g_int_aaaa[p, q, r, m] * rdms.RDM2aaaa(p, r, q, n)
                    gradient[idx] += 0.5 * g_int_aaaa[p, n, q, r] * rdms.RDM2aaaa(p, q, r, m)
                    gradient[idx] -= 0.5 * g_int_aaaa[p, q, r, n] * rdms.RDM2aaaa(p, r, q, m)
                    #bbbb
                    gradient[idx+len(kappa_idx)] += 0.5 * g_int_bbbb[n, p, q, r] * rdms.RDM2bbbb(m, q, r, p)
                    gradient[idx+len(kappa_idx)] -= 0.5 * g_int_bbbb[p, q, n, r] * rdms.RDM2bbbb(m, p, r, q)
                    gradient[idx+len(kappa_idx)] -= 0.5 * g_int_bbbb[m, p, q, r] * rdms.RDM2bbbb(n, q, r, p)
                    gradient[idx+len(kappa_idx)] += 0.5 * g_int_bbbb[p, q, m, r] * rdms.RDM2bbbb(n, p, r, q)
                    gradient[idx+len(kappa_idx)] -= 0.5 * g_int_bbbb[p, m, q, r] * rdms.RDM2bbbb(p, q, r, n)
                    gradient[idx+len(kappa_idx)] += 0.5 * g_int_bbbb[p, q, r, m] * rdms.RDM2bbbb(p, r, q, n)
                    gradient[idx+len(kappa_idx)] += 0.5 * g_int_bbbb[p, n, q, r] * rdms.RDM2bbbb(p, q, r, m)
                    gradient[idx+len(kappa_idx)] -= 0.5 * g_int_bbbb[p, q, r, n] * rdms.RDM2bbbb(p, r, q, m)
                    #aabb
                    #kappa a with aabb
                    gradient[idx] += 0.5 * g_int_aabb[n, p, q, r] * rdms.RDM2aabb(m, q, r, p) # [0, 0.08, 0, -0.01, 0, -0.06, 0.13, 0, 0]
                    gradient[idx] -= 0.5 * g_int_aabb[m, p, q, r] * rdms.RDM2aabb(n, q, r, p) # [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    gradient[idx] -= 0.5 * g_int_aabb[p, m, q, r] * rdms.RDM2aabb(p, q, r, n) # [0, 0, 0, 0, 0, 0, -0.13, 0, 0]
                    gradient[idx] += 0.5 * g_int_aabb[p, n, q, r] * rdms.RDM2aabb(p, q, r, m) # [0, 0.08, 0, -0.01, 0, -0.06, 0.13, 0, 0]
                    # kappa a with bbaa
                    gradient[idx] += 0.5 * g_int_bbaa[p, q, n, r] * rdms.RDM2bbaa(p, m, r, q) 
                    gradient[idx] -= 0.5 * g_int_bbaa[p, q, m, r] * rdms.RDM2bbaa(p, n, r, q) 
                    gradient[idx] -= 0.5 * g_int_bbaa[p, q, r, m] * rdms.RDM2bbaa(p, r, n, q) 
                    gradient[idx] += 0.5 * g_int_bbaa[p, q, r, n] * rdms.RDM2bbaa(p, r, m, q)
                    #kappa b with aabb
                    gradient[idx + len(kappa_idx)] += 0.5 * g_int_aabb[p, q, n, r] * rdms.RDM2aabb(p, m, r, q)
                    gradient[idx + len(kappa_idx)] -= 0.5 * g_int_aabb[p, q, m, r] * rdms.RDM2aabb(p, n, r, q)
                    gradient[idx + len(kappa_idx)] -= 0.5 * g_int_aabb[p, q, r, m] * rdms.RDM2aabb(p, r, n, q)
                    gradient[idx + len(kappa_idx)] += 0.5 * g_int_aabb[p, q, r, n] * rdms.RDM2aabb(p, r, m, q)
                    # kappa b with bbaa
                    gradient[idx + len(kappa_idx)] += 0.5 * g_int_bbaa[n, p, q, r] * rdms.RDM2bbaa(m, q, r, p)
                    gradient[idx + len(kappa_idx)] -= 0.5 * g_int_bbaa[m, p, q, r] * rdms.RDM2bbaa(n, q, r, p)
                    gradient[idx + len(kappa_idx)] -= 0.5 * g_int_bbaa[p, m, q, r] * rdms.RDM2bbaa(p, q, r, n)
                    gradient[idx + len(kappa_idx)] += 0.5 * g_int_bbaa[p, n, q, r] * rdms.RDM2bbaa(p, q, r, m)
    return gradient


# tror ikke det er gradienten jeg laver, men hessian block
def get_orbital_response_hessian_block_unrestricted(
        rdms: UnrestrictedReducedDensityMatrix,
        h_int_aa: np.ndarray,
        h_int_bb: np.ndarray,
        g_int_aaaa: np.ndarray,
        g_int_bbbb: np.ndarray,
        g_int_aabb: np.ndarray,
        g_int_bbaa: np.ndarray,
        kappa_idx1: list[list[int]],
        kappa_idx2: list[list[int]],
        num_inactive_orbs: int,
        num_active_orbs: int,
) -> np.ndarray:

    r"""Calculate Hessian-like orbital-orbital block in an unrestricted scheme.

    .. math::
        H^{\hat{q},\hat{q}}_{tu,mn} = \left<0\left|\left[\hat{q}_{tu},\left[\hat{H},\hat{q}_{mn}\right]\right]\right|0\right>

    Args:
       rdms: Reduced density matrix class.
       kappa_idx1: Orbital parameter indicies in spatial basis.
       kappa_idx2: Orbital parameter indicies in spatial basis.
       num_inactive_orbs: Number of inactive orbitals in spatial basis.
       num_active_orbs: Number of active orbitals in spatial basis.

    Returns:
        Hessian-like orbital-orbital block.
    """
    A1e = np.zeros((2*len(kappa_idx1), 2*len(kappa_idx2)))
    A2e = np.zeros((2*len(kappa_idx1), 2*len(kappa_idx2)))
    shift1 = len(kappa_idx1)
    shift2 = len(kappa_idx2)
    for idx1, (t, u) in enumerate(kappa_idx1):
        for idx2, (m, n) in enumerate(kappa_idx2):
            
            A1e[idx1, idx2] += h_int_aa[u, m] * rdms.RDM1aa(t, n)
            A1e[idx1, idx2] += h_int_aa[n, t] * rdms.RDM1aa(m, u)
            A1e[idx1 + shift1, idx2 + shift2] += h_int_bb[u, m] * rdms.RDM1bb(t, n)
            A1e[idx1 + shift1, idx2 + shift2] += h_int_bb[n, t] * rdms.RDM1bb(m, u)
            for p in range(num_inactive_orbs + num_active_orbs):
                if m == u: 
                    A1e[idx1, idx2] -= h_int_aa[n, p] * rdms.RDM1aa(t, p)
                    A1e[idx1 + shift1, idx2 + shift2] -= h_int_bb[n, p] * rdms.RDM1bb(t, p)
                if t == n:
                    A1e[idx1, idx2] -= h_int_aa[p, m] * rdms.RDM1aa(p, u)
                    A1e[idx1 + shift1, idx2 + shift2] -= h_int_bb[p, m] * rdms.RDM1bb(p, u)
            
            for p in range(num_inactive_orbs + num_active_orbs):
                for q in range(num_inactive_orbs + num_active_orbs):
                    
                    #mu, nu, sigma, tau = alpha
                    A2e[idx1, idx2] += g_int_aaaa[n, p, u, q] * rdms.RDM2aaaa(t, m, q, p)
                    A2e[idx1, idx2] += g_int_aaaa[n, t, p, q] * rdms.RDM2aaaa(m, p, q, u)
                    A2e[idx1, idx2] -= g_int_aaaa[n, p, q, t] * rdms.RDM2aaaa(m, q, p, u)
                    A2e[idx1, idx2] -= g_int_aaaa[u, p, n, q] * rdms.RDM2aaaa(t, m, q, p)
                    A2e[idx1, idx2] -= g_int_aaaa[p, t, n, q] * rdms.RDM2aaaa(m, p, q, u)
                    A2e[idx1, idx2] += g_int_aaaa[p, q, n, t] * rdms.RDM2aaaa(m, p, q, u)
                    A2e[idx1, idx2] += g_int_aaaa[u, m, p, q] * rdms.RDM2aaaa(t, p, q, n)
                    A2e[idx1, idx2] -= g_int_aaaa[p, m, u, q] * rdms.RDM2aaaa(t, p, q, n)
                    A2e[idx1, idx2] += g_int_aaaa[p, m, q, t] * rdms.RDM2aaaa(p, q, n, u) 
                    A2e[idx1, idx2] -= g_int_aaaa[u, p, q, m] * rdms.RDM2aaaa(t, q, p, n)
                    A2e[idx1, idx2] += g_int_aaaa[p, q, u, m] * rdms.RDM2aaaa(t, p, q, n)
                    A2e[idx1, idx2] -= g_int_aaaa[p, t, q, m] * rdms.RDM2aaaa(p, q, n, u)
                    
                    #mu, nu, sigma, tau = beta
                    A2e[idx1 + shift1, idx2 + shift2] += g_int_bbbb[n, p, u, q] * rdms.RDM2bbbb(t, m, q, p)
                    A2e[idx1 + shift1, idx2 + shift2] += g_int_bbbb[n, t, p, q] * rdms.RDM2bbbb(m, p, q, u)
                    A2e[idx1 + shift1, idx2 + shift2] -= g_int_bbbb[n, p, q, t] * rdms.RDM2bbbb(m, q, p, u)
                    A2e[idx1 + shift1, idx2 + shift2] -= g_int_bbbb[u, p, n, q] * rdms.RDM2bbbb(t, m, q, p)
                    A2e[idx1 + shift1, idx2 + shift2] -= g_int_bbbb[p, t, n, q] * rdms.RDM2bbbb(m, p, q, u)
                    A2e[idx1 + shift1, idx2 + shift2] += g_int_bbbb[p, q, n, t] * rdms.RDM2bbbb(m, p, q, u)
                    A2e[idx1 + shift1, idx2 + shift2] += g_int_bbbb[u, m, p, q] * rdms.RDM2bbbb(t, p, q, n)
                    A2e[idx1 + shift1, idx2 + shift2] -= g_int_bbbb[p, m, u, q] * rdms.RDM2bbbb(t, p, q, n)
                    A2e[idx1 + shift1, idx2 + shift2] += g_int_bbbb[p, m, q, t] * rdms.RDM2bbbb(p, q, n, u) 
                    A2e[idx1 + shift1, idx2 + shift2] -= g_int_bbbb[u, p, q, m] * rdms.RDM2bbbb(t, q, p, n)
                    A2e[idx1 + shift1, idx2 + shift2] += g_int_bbbb[p, q, u, m] * rdms.RDM2bbbb(t, p, q, n)
                    A2e[idx1 + shift1, idx2 + shift2] -= g_int_bbbb[p, t, q, m] * rdms.RDM2bbbb(p, q, n, u)
                    
                    
                    #nu, mu, tau = alpha, sigma = beta
                    A2e[idx1, idx2] += g_int_bbaa[p, q, u, m] * rdms.RDM2aabb(t, p, q, n)
                    A2e[idx1, idx2] += g_int_bbaa[p, q, n, t] * rdms.RDM2aabb(m, p, q, u)

                    #nu, mu, sigma = alpha, tau = beta
                    A2e[idx1, idx2] += g_int_aabb[u, m, p, q] * rdms.RDM2aabb(t, p, q, n)
                    A2e[idx1, idx2] += g_int_aabb[n, t, p, q] * rdms.RDM2aabb(m, p, q, u)
                    
                    #nu, mu, tau = beta, sigma = alpha
                    A2e[idx1 + shift1, idx2 + shift2] += g_int_aabb[p, q, u, m] * rdms.RDM2bbaa(t, p, q, n)
                    A2e[idx1 + shift1, idx2 + shift2] += g_int_aabb[p, q, n, t] * rdms.RDM2bbaa(m, p, q, u)

                    #nu, mu, sigma = beta, tau = alpha
                    A2e[idx1 + shift1, idx2 + shift2] += g_int_bbaa[u, m, p, q] * rdms.RDM2bbaa(t, p, q, n)
                    A2e[idx1 + shift1, idx2 + shift2] += g_int_bbaa[n, t, p, q] * rdms.RDM2bbaa(m, p, q, u)
                    
                    #mu, sigma = beta, nu, tau = alpha
                    A2e[idx1 + shift1, idx2] += g_int_bbaa[u, p, q, m] * rdms.RDM2bbaa(t, q, n, p) #ombyttet indices 3
                    A2e[idx1 + shift1, idx2] -= g_int_bbaa[u, p, n, q] * rdms.RDM2bbaa(t, m, q, p)
                    A2e[idx1 + shift1, idx2] -= g_int_bbaa[p, t, q, m] * rdms.RDM2bbaa(p, q, n, u)
                    A2e[idx1 + shift1, idx2] += g_int_bbaa[p, t, n, q] * rdms.RDM2bbaa(p, m, q, u) #ombyttet indices 15

                    #mu, tau = beta, nu, sigma = alpha
                    A2e[idx1 + shift1, idx2] += g_int_aabb[p, m, u, q] * rdms.RDM2aabb(p, t, q, n) #ombyttet indices 2 
                    A2e[idx1 + shift1, idx2] -= g_int_aabb[n, p, u, q] * rdms.RDM2aabb(m, t, q, p) #ombyttet indices 6 
                    A2e[idx1 + shift1, idx2] -= g_int_aabb[p, m, q, t] * rdms.RDM2aabb(p, q, u, n) #ombyttet indices 10
                    A2e[idx1 + shift1, idx2] += g_int_aabb[n, p, q, t] * rdms.RDM2aabb(m, q, u, p) #ombyttet indices 14

                    #mu, sigma = alpha, nu, tau = beta
                    A2e[idx1, idx2 + shift2] += g_int_aabb[u, p, q, m] * rdms.RDM2aabb(t, q, n, p) #ombyttet indices 3
                    A2e[idx1, idx2 + shift2] -= g_int_aabb[u, p, n, q] * rdms.RDM2aabb(t, m, q, p)
                    A2e[idx1, idx2 + shift2] -= g_int_aabb[p, t, q, m] * rdms.RDM2aabb(p, q, n, u)
                    A2e[idx1, idx2 + shift2] += g_int_aabb[p, t, n, q] * rdms.RDM2aabb(p, m, q, u) #ombyttet indices 15

                    #mu, tau = alpha, nu, sigma = beta
                    A2e[idx1, idx2 + shift2] += g_int_bbaa[p, m, u, q] * rdms.RDM2bbaa(p, t, q, n) #ombyttet indices 2
                    A2e[idx1, idx2 + shift2] -= g_int_bbaa[n, p, u, q] * rdms.RDM2bbaa(m, t, q, p) #ombyttet indices 6
                    A2e[idx1, idx2 + shift2] -= g_int_bbaa[p, m, q, t] * rdms.RDM2bbaa(p, q, u, n) #ombyttet indices 10
                    A2e[idx1, idx2 + shift2] += g_int_bbaa[n, p, q, t] * rdms.RDM2bbaa(m, q, u, p) #ombyttet indeices 14
                    
            for p in range(num_inactive_orbs + num_active_orbs):
                for q in range(num_inactive_orbs + num_active_orbs):
                    for r in range(num_inactive_orbs + num_active_orbs):
                        if m == u:
                            A2e[idx1, idx2] -= g_int_aaaa[n, p, q, r] * rdms.RDM2aaaa(t, q, r, p)
                            A2e[idx1, idx2] += g_int_aaaa[p, q, n, r] * rdms.RDM2aaaa(t, p, r, q)
                            A2e[idx1 + shift1, idx2 + shift2] -= g_int_bbbb[n, p, q, r] * rdms.RDM2bbbb(t, q, r, p)
                            A2e[idx1 + shift1, idx2 + shift2] += g_int_bbbb[p, q, n, r] * rdms.RDM2bbbb(t, p, r, q)
                            A2e[idx1, idx2] -= g_int_bbaa[p, q, n, r] * rdms.RDM2aabb(t, p, q, r) # ombytte indices 
                            A2e[idx1, idx2] -= g_int_aabb[n, p, q, r] * rdms.RDM2aabb(t, q, r, p) 
                            A2e[idx1 + shift1, idx2 + shift2] -= g_int_aabb[p, q, n, r] * rdms.RDM2bbaa(t, p, q, r) # ombytte indices (skal flyttes) 7
                            A2e[idx1 + shift1, idx2 + shift2] -= g_int_bbaa[n, p, q, r] * rdms.RDM2bbaa(t, q, r, p) 
                        if t == n:
                            A2e[idx1, idx2] -= g_int_aaaa[p, m, q, r] * rdms.RDM2aaaa(p, q, r, u)
                            A2e[idx1, idx2] += g_int_aaaa[p, q, r, m] * rdms.RDM2aaaa(p, r, q, u)
                            A2e[idx1 + shift1, idx2 + shift2] -= g_int_bbbb[p, m, q, r] * rdms.RDM2bbbb(p, q, r, u)
                            A2e[idx1 + shift1, idx2 + shift2] += g_int_bbbb[p, q, r, m] * rdms.RDM2bbbb(p, r, q, u)
                            A2e[idx1, idx2] -= g_int_bbaa[p, q, r, m] * rdms.RDM2aabb(r, p, q, u) # ombytte indices 
                            A2e[idx1, idx2] -= g_int_aabb[p, m, q, r] * rdms.RDM2aabb(p, q, r, u)
                            A2e[idx1 + shift1, idx2 + shift2] -= g_int_aabb[p, q, r, m] * rdms.RDM2bbaa(r, p, q, u) # ombytte indices (skal flyttes) 11             
                            A2e[idx1 + shift1, idx2 + shift2] -= g_int_bbaa[p, m, q, r] * rdms.RDM2bbaa(p, q, r, u)    
                                
    return 1/2 * A1e + 1/4 * A2e