import functools
import itertools
from collections.abc import Sequence

import numpy as np
import scipy.sparse as ss


@functools.cache
def kronecker_product_cached(
    num_prior: int, num_after: int, pauli_mat_symbol: str, is_csr: bool
) -> np.ndarray | ss.csr_matrix:
    r"""Get operator in matrix form.

    The operator is returned in the form:

    .. math::
        I x I x .. o .. x I x I

    Args:
       num_prior: Number of left-hand side identity matrices.
       num_after: Number of right-hand side identity matrices.
       is_csr: If the resulting matrix representation should be a sparse matrix.

    Returns:
       Matrix representation of an operator.
    """
    mat = pauli_to_mat(pauli_mat_symbol)
    if is_csr:
        I1 = ss.identity(int(2**num_prior))
        I2 = ss.identity(int(2**num_after))
        mat = ss.csr_matrix(mat)
        return ss.csr_matrix(ss.kron(I1, ss.kron(mat, I2)))
    I1 = np.identity(int(2**num_prior))
    I2 = np.identity(int(2**num_after))
    return np.kron(I1, np.kron(mat, I2))


def kronecker_product(A: Sequence[np.ndarray]) -> np.ndarray:
    r"""Get Kronecker product of a list of matricies.

    Does:

    .. math::
        P x P x P ...

    Args:
       A: List of matrices.

    Returns:
       Kronecker product of matrices.
    """
    if len(A) < 2:
        return A
    total = np.kron(A[0], A[1])
    for operator in A[2:]:
        total = np.kron(total, operator)
    return total


@functools.cache
def pauli_to_mat(pauli: str) -> np.ndarray:
    """Convert Pauli matrix symbol to matrix representation.

    Args:
        pauli: Pauli matrix symbol.

    Returns:
        Pauli matrix.
    """
    if pauli == "I":
        return np.array([[1, 0], [0, 1]], dtype=float)
    if pauli == "Z":
        return np.array([[1, 0], [0, -1]], dtype=float)
    if pauli == "X":
        return np.array([[0, 1], [1, 0]], dtype=float)
    if pauli == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    raise ValueError(f"Got unknown string: {pauli}")


class StateVector:
    """State vector."""

    def __init__(
        self, inactive: list[np.ndarray], active: list[np.ndarray], virtual: list[np.ndarray]
    ) -> None:
        """Initialize state vector.

        Args:
            inactive: Kronecker representation of inactive orbitals (reference).
            active: Kronecker representation of active orbitals (reference).
            virtual: Kronecker representation of virtual orbitals (reference).
        """
        self.inactive = np.transpose(inactive)
        self._active_onvector = active
        self._active = np.transpose(kronecker_product(active))
        self.active = np.transpose(kronecker_product(active)) * 1.0
        self.active_csr = ss.csr_matrix(self.active)
        self.virtual = np.transpose(virtual)
        o = np.array([0, 1])
        z = np.array([1, 0])
        num_active_elec = 0
        num_active_alpha_elec = 0
        num_active_beta_elec = 0
        num_active_spin_orbs = len(self._active_onvector)
        if num_active_spin_orbs != 0:
            for idx, vec in enumerate(self._active_onvector):
                if vec[0] == 0 and vec[1] == 1:
                    num_active_elec += 1
                    if idx % 2 == 0:
                        num_active_alpha_elec += 1
                    else:
                        num_active_beta_elec += 1
            self.allowed_active_states_number_conserving = np.zeros(len(self._active), dtype=bool)
            self.allowed_active_states_number_spin_conserving = np.zeros(len(self._active), dtype=bool)
            for comb in itertools.product([o, z], repeat=num_active_spin_orbs):
                num_elec = 0
                num_alpha_elec = 0
                num_beta_elec = 0
                for idx, vec in enumerate(comb):
                    if vec[0] == 0 and vec[1] == 1:
                        num_elec += 1
                        if idx % 2 == 0:
                            num_alpha_elec += 1
                        else:
                            num_beta_elec += 1
                if num_elec == num_active_elec:
                    idx = np.argmax(kronecker_product(comb))
                    self.allowed_active_states_number_conserving[idx] = True
                    if num_alpha_elec == num_active_alpha_elec and num_beta_elec == num_active_beta_elec:
                        self.allowed_active_states_number_spin_conserving[idx] = True

    @property
    def bra_inactive(self) -> list[np.ndarray]:
        """Get bra configuration for inactive orbitals."""
        return np.conj(self.inactive).transpose()

    @property
    def ket_inactive(self) -> list[np.ndarray]:
        """Get ket configuration for inactive orbitals."""
        return self.inactive

    @property
    def bra_virtual(self) -> list[np.ndarray]:
        """Get bra configuration for virtual orbitals."""
        return np.conj(self.virtual).transpose()

    @property
    def ket_virtual(self) -> list[np.ndarray]:
        """Get ket configuration for virtual orbitals."""
        return self.virtual

    @property
    def bra_active(self) -> np.ndarray:
        """Get bra state-vector for active orbitals."""
        return np.conj(self.active).transpose()

    @property
    def ket_active(self) -> np.ndarray:
        """Get ket state-vector for inactive orbitals."""
        return self.active

    @property
    def bra_active_csr(self) -> ss.csr_matrix:
        """Get bra state-vector for inactive orbitals."""
        return self.active_csr

    @property
    def ket_active_csr(self) -> ss.csr_matrix:
        """Get ket state-vector for inactive orbitals."""
        return self.active_csr.conj().transpose()

    def new_u(self, U: np.ndarray, allowed_states: np.ndarray | None = None) -> None:
        """Create active state-vector by applying transformation matrix to reference.

        Args:
            U: Transformation matrix.
            allowed_states: State to be transformed.
        """
        if allowed_states is None:
            self.active = np.matmul(U, self._active)
            self.active_csr = ss.csr_matrix(self.active)
        else:
            if isinstance(U, np.ndarray):
                tmp_active = np.matmul(U, self._active[allowed_states])
            else:
                tmp_active = U.dot(ss.csr_matrix(self._active[allowed_states]).transpose()).toarray()
            idx = 0
            for i, allowed in enumerate(allowed_states):
                if allowed:
                    self.active[i] = tmp_active[idx]
                    idx += 1
            self.active_csr = ss.csr_matrix(self.active)
