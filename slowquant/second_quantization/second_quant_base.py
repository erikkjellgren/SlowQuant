from __future__ import annotations

import copy

import numpy as np
import scipy

from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform,
    two_electron_integral_transform,
)


class a_op:
    def __init__(self, spinless_idx: int, spin: str, dagger: bool) -> None:
        self.spinless_idx = spinless_idx
        self.idx = 2 * self.spinless_idx
        self.dagger = dagger
        self.spin = spin
        if self.spin == "beta":
            self.idx += 1


class FermionicString:
    def __init__(self, annihilation_operator: list[list[a_op]], factor: list[float]) -> None:
        if not isinstance(annihilation_operator, list):
            self.operators = [[annihilation_operator]]
            self.factors = [factor]
        else:
            self.operators = annihilation_operator
            self.factors = factor

    def __add__(self, fermistring: FermionicString) -> FermionicString:
        operators = copy.copy(self.operators)
        factors = copy.copy(self.factors)
        for operator, factor in zip(fermistring.operators, fermistring.factors):
            operators.append(operator)
            factors.append(factor)
        return FermionicString(operators, factors)

    def __sub__(self, fermistring: FermionicString) -> FermionicString:
        operators = copy.copy(self.operators)
        factors = copy.copy(self.factors)
        for operator, factor in zip(fermistring.operators, fermistring.factors):
            operators.append(operator)
            factors.append(-factor)
        return FermionicString(operators, factors)

    def __mul__(self, fermistring: FermionicString) -> FermionicString:
        operators = []
        factors = []
        for operator, factor in zip(fermistring.operators, fermistring.factors):
            for operator2, factor2 in zip(self.operators, self.factors):
                operators.append(operator2 + operator)
                factors.append(factor * factor2)
        return FermionicString(operators, factors)

    def __rmul__(self, number: float) -> FermionicString:
        operators = []
        factors = []
        for operator, factor in zip(self.operators, self.factors):
            operators.append(operator)
            factors.append(factor * number)
        return FermionicString(operators, factors)

    def get_latex(self, do_spin_idx: bool = False) -> str:
        latex_code = ""
        for term, factor in zip(self.operators, self.factors):
            if factor > 0:
                if factor != 1:
                    latex_code += f" +{factor} "
                else:
                    latex_code += " + "
            else:
                if factor != -1:
                    latex_code += f" {factor} "
                else:
                    latex_code += " - "
            for a in term:
                if do_spin_idx:
                    if a.dagger:
                        latex_code += f"\hat{{a}}_{a.idx}^\dagger "
                    else:
                        latex_code += f"\hat{{a}}_{a.idx} "
                else:
                    if a.dagger:
                        latex_code += f"\hat{{a}}_{{{a.spinless_idx}, \\{a.spin}}}^\dagger "
                    else:
                        latex_code += f"\hat{{a}}_{{{a.spinless_idx}, \\{a.spin}}} "
        return latex_code


class WaveFunction:
    def __init__(self, number_spin_orbitals: int) -> None:
        self.determinants = []
        self.coefficients = []
        self.number_spin_orbitals = number_spin_orbitals
        self.number_spatial_orbitals = number_spin_orbitals // 2
        self.kappa = np.zeros((self.number_spatial_orbitals, self.number_spatial_orbitals))
        self.c_mo = scipy.linalg.expm(-self.kappa)

    def add_determinant(self, determinant: list[int], coefficient: float) -> None:
        self.determinants.append(determinant)
        self.coefficients.append(coefficient)

    def get_non_redundant_kappa(self) -> list[int, int]:
        kappa_index = []
        for p in range(self.number_spatial_orbitals):
            for q in range(self.number_spatial_orbitals):
                if p <= q:
                    continue
                if len(apply_on_ket(Eminuspq(p, q), self)[0]) == 0:
                    continue
                kappa_index.append([p, q])
        return kappa_index

    def update_kappa(self, kappa_values: list[float], kappa_indicies: list[int, int]) -> None:
        for value, (p, q) in zip(kappa_values, kappa_indicies):
            if p == q:
                print("WARNING: kappa_pp is changed")
            self.kappa[p, q] = value
            self.kappa[q, p] = -value
        self.c_mo = np.matmul(self.c_mo, scipy.linalg.expm(-self.kappa))
        self.kappa[:, :] = 0.0


def collapse_operator_on_determinant(operator: list[a_op], determinant: list[int]) -> tuple[list[int], int]:
    determinant_out = copy.copy(determinant)
    phase = 1
    for a in operator[::-1]:
        idx = a.idx
        if a.dagger:
            if determinant_out[idx] == 1:
                return [], 0
            determinant_out[idx] = 1
        else:
            if determinant_out[idx] == 0:
                return [], 0
            determinant_out[idx] = 0
        if int(np.sum(determinant_out[:idx])) % 2 == 1:
            phase *= -1
    return determinant_out, phase


def apply_on_ket(operators: FermionicString, ket: WaveFunction) -> tuple[list[int], list[int]]:
    determinants = []
    phases = []
    for factor, operator in zip(operators.factors, operators.operators):
        for ket_determinant, ket_coeff in zip(ket.determinants, ket.coefficients):
            collapsed_determinant, phase = collapse_operator_on_determinant(operator, ket_determinant)
            if phase == 0:
                continue
            determinants.append(collapsed_determinant)
            phases.append(phase)
    return determinants, phases


def Epq(p: int, q: int) -> FermionicString:
    E = FermionicString(a_op(p, "alpha", dagger=True), 1) * FermionicString(a_op(q, "alpha", dagger=False), 1)
    E += FermionicString(a_op(p, "beta", dagger=True), 1) * FermionicString(a_op(q, "beta", dagger=False), 1)
    return E


def epqrs(p: int, q: int, r: int, s: int) -> FermionicString:
    if q == r:
        return Epq(p, q) * Epq(r, s) - Epq(p, s)
    return Epq(p, q) * Epq(r, s)


def Eminuspq(p: int, q: int) -> FermionicString:
    return Epq(p, q) - Epq(q, p)


def H(h: np.ndarray, g: np.ndarray, c_mo: np.ndarray) -> FermionicString:
    H_operator = FermionicString([], [])
    h_mo = one_electron_integral_transform(c_mo, h)
    g_mo = two_electron_integral_transform(c_mo, g)
    num_bf = len(c_mo)
    for p in range(num_bf):
        for q in range(num_bf):
            H_operator += h_mo[p, q] * Epq(p, q)
    for p in range(num_bf):
        for q in range(num_bf):
            for r in range(num_bf):
                for s in range(num_bf):
                    H_operator += 1 / 2 * g_mo[p, q, r, s] * epqrs(p, q, r, s)
    return H_operator


def comm(A: FermionicString, B: FermionicString) -> FermionicString:
    return A * B - B * A
