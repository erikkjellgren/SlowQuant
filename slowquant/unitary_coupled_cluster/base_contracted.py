from __future__ import annotations
import numpy as np
import scipy.sparse as ss

class PauliOperatorContracted:
    def __init__(self, operator: np.ndarray | ss.csr_matrix) -> None:
        """The key is the Pauli-string of inactive + virtual,
        i.e. the active part does not contribute to the key.
        """
        self.operators = operator

    def __add__(self, hybridop: PauliOperatorContracted) -> PauliOperatorContracted:

    def __sub__(self, hybridop: PauliOperatorContracted) -> PauliOperatorContracted:

    def __mul__(self, pauliop: PauliOperatorContracted) -> PauliOperatorContracted:

    def __rmul__(self, number: float) -> PauliOperatorContracted:

    @property
    def dagger(self) -> PauliOperatorContracted:

    def apply_U_from_right(self, U: np.ndarray | ss.csr_matrix) -> PauliOperatorContracted:

    def apply_U_from_left(self, U: np.ndarray | ss.csr_matrix) -> PauliOperatorContracted:
