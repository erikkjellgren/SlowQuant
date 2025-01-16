import numpy as np


class UnrestrictedReducedDensityMatrix:
    """Unrestricted reduced density matrix class """

    def __init__(
        self,
        num_inactive_orbs : int,
        num_active_orbs: int,
        num_virtual_orbs: int,
        rdm1: np.ndarray, #Should this be aa, bb
        rdm2: np.ndarray | None = None, # and this aaaa, bbbb and aabb?
    ) -> None:
    """ Initialize unrestricted reduced density matrix class.
    
    Args: 
        num_inactive_orbs: Number of inactive orbitals in spatial basis.
        num_active_orbs: Number of active orbitals in spatial basis.
        num_virtual_orbs: Number of virtual orbitals in spatial basis.
        rdm1: One-electron reduced density matrix in the active space.
        rdm2: Two-electron reduced density matrix in the active space.C
 D   """ 
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
    self._rdm1aa = rdm1aa
    self._rdm1bb = rdm1bb
    self._rdm2aaaa = rdm2aaaa
    self._rdm2bbbb = rdm2bbbb
    self._rdm2aabb = rdm2aabb
    self._rdm2bbaa = rdm2bbaa
# what about rdm1_C, rdm1_S, rdm2_C? What are they used for?

    def RDM1(self, p: int, q: int) -> float:
        """Get one-electron unrestricted reduced density matrix element
        
        Think about what the non-zero elements are
        
        Args:
            p: Spatial orbital index
            q: Spatial orbital index
            
        Returns:
            One-electron unrestricted reduced density matrix element.
         """

    def RDM2(self, p: int, q: int, r: int, s: int) -> float:
        """Get two-elelctron unrestricted reduced density matrix element.
        
        Think about what the non-zero elements are
        
        Args:
            p: Spatial orbital index
            q: Spatial orbital index
            r: Spatial obrital index
            s: Spatial orbital index

        Returns:
            Two-electron unrestricted reduced density matrix element.
         """
