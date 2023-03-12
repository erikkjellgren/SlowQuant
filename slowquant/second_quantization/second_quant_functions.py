import copy

import numpy as np

from slowquant.second_quantization.second_quant_base import (
    Epq,
    Eminuspq,
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
    for _ in range(200):
        H_op = H(h_core, g_eri, wavefunction.c_mo)
        grad = np.zeros(len(kappa_idx))
        hess = np.zeros((len(kappa_idx),len(kappa_idx)))
        grad_op = {}
        for i, (p, q) in enumerate(kappa_idx):
            if str(p)+str(q) not in grad_op:
                grad_op[str(p)+str(q)] = comm(Epq(p, q), H_op) 
            #grad[i] = expectation_value(wavefunction, comm(Eminuspq(p, q), H_op), wavefunction)
            #grad[i] = 2*expectation_value(wavefunction, comm(Epq(p, q), H_op), wavefunction)
            grad[i] = 2*expectation_value(wavefunction, grad_op[str(p)+str(q)], wavefunction)
        for i, (p,q) in enumerate(kappa_idx):
            for j, (r,s) in enumerate(kappa_idx):
                if i < j:
                    continue
                #hess_op = 0.5*comm(Eminuspq(p, q), comm(Eminuspq(r, s), H_op)) + comm(Eminuspq(r, s), comm(Eminuspq(p, q), H_op))  
                #hess_op = comm(Eminuspq(p, q), comm(Epq(r, s), H_op)) + comm(Eminuspq(r, s), comm(Epq(p, q), H_op))  
                hess_op = comm(Eminuspq(p, q), grad_op[str(r)+str(s)]) + comm(Eminuspq(r, s), grad_op[str(p)+str(q)])  
                hess[i,j] = hess[j,i] = expectation_value(wavefunction, hess_op, wavefunction)
        kappa = np.zeros(len(kappa_idx))
        hess_eig, _ = np.linalg.eigh(hess)
        if np.min(hess_eig) >= 0.0:
            delta = np.matmul(np.linalg.inv(hess), grad)
            for i in range(len(delta)):
                delta[i] = np.sign(delta[i]) * min(abs(delta[i]), 0.3)
            did_newton = True
        else:
            delta = grad
            for i in range(len(delta)):
                delta[i] = np.sign(delta[i]) * min(abs(delta[i]), 0.3)
            did_newton = False
        for i in range(len(delta)):
            kappa[i] = -delta[i]
        wavefunction.update_kappa(kappa, kappa_idx)
        E_new = expectation_value(wavefunction, H(h_core, g_eri, wavefunction.c_mo), wavefunction)
        if abs(E_new - E_old) < 10**-6 and np.max(np.abs(grad)) < 10**-3:
            break
        print('Micro:', E_new, did_newton)
        E_old = E_new
    return wavefunction


def build_H_CI(wavefunction: WaveFunction, h_core: np.ndarray, g_eri: np.ndarray) -> np.ndarray:
    H_op = H(h_core, g_eri, wavefunction.c_mo)
    num_det = len(wavefunction.determinants)
    H_ci = np.zeros((num_det, num_det))
    for i, det1 in enumerate(wavefunction.determinants):
        wf1 = WaveFunction(wavefunction.number_spin_orbitals)
        wf1.add_determinant(det1, 1.0)
        for j, det2 in enumerate(wavefunction.determinants):
            if i < j:
                continue
            wf2 = WaveFunction(wavefunction.number_spin_orbitals)
            wf2.add_determinant(det2, 1.0)
            H_ci[i,j] = H_ci[j,i] = expectation_value(wf1, H_op, wf2)
    return H_ci

def optimize_wavefunction_parameters(wavefunction: WaveFunction, h_core: np.ndarray, g_eri: np.ndarray) -> WaveFunction:
    E_old = expectation_value(wavefunction, H(h_core, g_eri, wavefunction.c_mo), wavefunction)
    for _ in range(200):
        wavefunction = optimize_kappa(wavefunction, h_core, g_eri) 
        H_ci = build_H_CI(wavefunction, h_core, g_eri)
        _, eigvec = np.linalg.eigh(H_ci)
        wavefunction.coefficients = eigvec[:,0]
        E_new = expectation_value(wavefunction, H(h_core, g_eri, wavefunction.c_mo), wavefunction)
        if abs(E_new - E_old) < 10**-6:
            break
        print('****Macro:', E_new)
        E_old = E_new
    return wavefunction
