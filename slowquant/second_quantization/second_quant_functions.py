import copy

import numpy as np

from slowquant.second_quantization.second_quant_base import (
    Epq,
    FermionicString,
    H,
    WaveFunction,
    a_op,
    collapse_operator_on_determinant,
    comm,
)


def expectation_value(bra: WaveFunction, operators: FermionicString, ket: WaveFunction) -> float:
    value = 0.0
    if np.sum(np.abs(ket.kappa)) != 0:
        print("WARNING: Wavefunction is not canonical")
    for factor, operator in zip(operators.factors, operators.operators):
        for ket_determinant, ket_coeff in zip(ket.determinants, ket.coefficients):
            collapsed_determinant, phase = collapse_operator_on_determinant(operator, ket_determinant)
            if phase == 0:
                continue
            for bra_determinant, bra_coeff in zip(bra.determinants, bra.coefficients):
                if bra_determinant == collapsed_determinant:
                    value += factor * phase * bra_coeff * ket_coeff
    return value


def optimize_kappa(wavefunction: WaveFunction, h_core: np.ndarray, g_eri: np.ndarray) -> WaveFunction:
    kappa_idx = wavefunction.get_non_redundant_kappa()
    E_old = expectation_value(wavefunction, H(h_core, g_eri, wavefunction.c_mo), wavefunction)
    for j in range(200):
        H_op = H(h_core, g_eri, wavefunction.c_mo)
        grad = []
        for p, q in kappa_idx:
            grad.append(2 * expectation_value(wavefunction, comm(Epq(p, q), H_op), wavefunction))
        kappa = np.zeros(len(kappa_idx))
        for i in range(len(grad)):
            kappa[i] = -0.1 * grad[i]
        wavefunction.update_kappa(kappa, kappa_idx)
        E_new = expectation_value(wavefunction, H(h_core, g_eri, wavefunction.c_mo), wavefunction)
        print(j + 1, E_new, abs(E_new - E_old), np.max(np.abs(grad)))
        if abs(E_new - E_old) < 10**-6 and np.max(np.abs(grad)) < 10**-6:
            break
        E_old = E_new
    return wavefunction
