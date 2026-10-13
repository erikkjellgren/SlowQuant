 @property
    def c_mo_int(self) -> np.ndarray:
        """Get molecular orbital coefficients.

        Returns:
            Molecular orbital coefficients.
        """
        # Construct anti-hermitian kappa matrix
        kappa_mat = np.zeros_like(self._c_mo)
        if len(self.kappa_real) != 0:
            # The MO transformation is calculated as a difference between current kappa and kappa old.
            # This is to make the moving of the expansion point to work with SciPy optimization algorithms.
            # Resetting kappa to zero would mess with any algorithm that has any memory f.x. BFGS.
            if np.max(np.abs(np.array(self.kappa_real) - np.array(self._kappa_real_old))) > 0.0:
                for kappa_val, kappa_old, (p, q) in zip(
                    self.kappa_real, self._kappa_real_old, self.kappa_spin_idx
                ):
                    if p == q:
                        continue
                    kappa_mat[p, q] = kappa_val - kappa_old
                    kappa_mat[q, p] = -(kappa_val - kappa_old)
            if np.max(np.abs(np.array(self.kappa_imag) - np.array(self._kappa_imag_old))) > 0.0:
                for kappa_val, kappa_old, (p, q) in zip(
                    self.kappa_imag, self._kappa_imag_old, self.kappa_spin_idx
                ):
                    kappa_mat[p, q] += (kappa_val - kappa_old) * 1.0j
                    kappa_mat[q, p] += (kappa_val - kappa_old) * 1.0j
        # Apply orbital rotation unitary to MO coefficients
        return np.matmul(self._c_mo, scipy.linalg.expm(-kappa_mat))




@property
    def c_mo_old(self) -> np.ndarray:
        """Get molecular orbital coefficients.

        Returns:
            Molecular orbital coefficients.
        """

        M = self.c_mo.conj().T @ self._S_int @ Trev(self.c_mo)

        # Construct anti-hermitian kappa matrix
        kappa_mat = np.zeros_like(self._c_mo)
        if len(self.kappa_real) != 0:
            # The MO transformation is calculated as a difference between current kappa and kappa old.
            # This is to make the moving of the expansion point to work with SciPy optimization algorithms.
            # Resetting kappa to zero would mess with any algorithm that has any memory f.x. BFGS.
            if np.max(np.abs(np.array(self.kappa_real) - np.array(self._kappa_real_old))) > 0.0:
                for kappa_val, kappa_old, (p, q) in zip(
                    self.kappa_real, self._kappa_real_old, self.kappa_spin_idx
                ):
                    if p == q:
                        continue

                    delta = kappa_val - kappa_old

                    # Original rotation
                    kappa_mat[p, q] = delta
                    kappa_mat[q, p] = -delta

                    # Kramers partner rotation
                    if self._Kp:
                        pb = self._kpartner[p]
                        qb = self._kpartner[q]

                        kappa_mat[pb, qb] = delta
                        kappa_mat[qb, pb] = -delta


            if np.max(np.abs(np.array(self.kappa_imag) - np.array(self._kappa_imag_old))) > 0.0:
                for kappa_val, kappa_old, (p, q) in zip(
                    self.kappa_imag, self._kappa_imag_old, self.kappa_spin_idx
                ):
                    delta = (kappa_val - kappa_old) * 1.0j

                    # Original rotation
                    kappa_mat[p, q] += delta
                    kappa_mat[q, p] += delta

                    # Kramers partner rotation
                    if self._Kp:
                        pb = self._kpartner[p]
                        qb = self._kpartner[q]

                        kappa_mat[pb, qb] -= delta
                        kappa_mat[qb, pb] -= delta

        # Apply orbital rotation unitary to MO coefficients
        return np.matmul(self._c_mo, scipy.linalg.expm(-kappa_mat))

    @property
    def c_mo_ep_old(self) -> np.ndarray:  # Positronic
        """Get molecular orbital coefficients.

        Returns:
            Molecular orbital coefficients.
        """
        # Construct anti-hermitian kappa matrix
        kappa_mat = np.zeros_like(self._c_mo)
        if len(self.kappa_real_ep) != 0:
            # The MO transformation is calculated as a difference between current kappa and kappa old.
            # This is to make the moving of the expansion point to work with SciPy optimization algorithms.
            # Resetting kappa to zero would mess with any algorithm that has any memory f.x. BFGS.
            if np.max(np.abs(np.array(self.kappa_real_ep) - np.array(self._kappa_real_old_ep))) > 0.0:
                for kappa_val, kappa_old, (p, q) in zip(
                    self.kappa_real_ep, self._kappa_real_old_ep, self.kappa_spin_idx_ep
                ):
                    if p == q:
                        continue

                    delta = kappa_val - kappa_old

                    # Original rotation
                    kappa_mat[p, q] = delta
                    kappa_mat[q, p] = -delta

                    # Kramers partner rotation
                    if self._Kp:
                        pb = self._kpartner[p]
                        qb = self._kpartner[q]

                        kappa_mat[pb, qb] = delta
                        kappa_mat[qb, pb] = -delta

            if np.max(np.abs(np.array(self.kappa_imag_ep) - np.array(self._kappa_imag_old_ep))) > 0.0:
                for kappa_val, kappa_old, (p, q) in zip(
                    self.kappa_imag_ep, self._kappa_imag_old_ep, self.kappa_spin_idx_ep
                ):
                    delta = (kappa_val - kappa_old) * 1.0j

                    # Original rotation
                    kappa_mat[p, q] += delta
                    kappa_mat[q, p] += delta

                    # Kramers partner rotation
                    if self._Kp:
                        pb = self._kpartner[p]
                        qb = self._kpartner[q]

                        kappa_mat[pb, qb] -= delta
                        kappa_mat[qb, pb] -= delta
        # Apply orbital rotation unitary to MO coefficients
        return np.matmul(self._c_mo, scipy.linalg.expm(-kappa_mat))