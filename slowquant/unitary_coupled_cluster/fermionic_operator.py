from __future__ import annotations

import copy
import re
from dataclasses import dataclass


@dataclass(repr=False, eq=False, match_args=False)
class a_op:
    __slots__ = ("spinless_idx", "idx", "dagger", "spin")

    def __init__(self, spinless_idx: int, spin: str, dagger: bool) -> None:
        """Initialize fermionic annihilation operator.

        Args:
            spinless_idx: Spatial orbital index.
            spin: Alpha or beta spin.
            dagger: If creation operator.
        """
        if spin not in ("alpha", "beta"):
            raise ValueError(f'spin must be "alpha" or "beta" got {spin}')
        self.spinless_idx = spinless_idx
        self.idx = 2 * self.spinless_idx
        self.dagger = dagger
        self.spin = spin
        if self.spin == "beta":
            self.idx += 1


def a_op_spin(spin_idx: int, dagger: bool) -> a_op:
    """Get fermionic annihilation operator.

    Args:
        spin_idx: Spin orbital index.
        dagger: If creation operator.

    Returns:
        Annihilation operator.
    """
    if spin_idx % 2 == 0:
        return a_op(spin_idx // 2, "alpha", dagger)
    return a_op(spin_idx // 2, "beta", dagger)


def operator_string_to_key(operator_string: list[a_op]) -> str:
    """Make key string to index a fermionic operator in a dict structure.

    Args:
        operator_string: Fermionic operators.

    Returns:
        Dictionary key.
    """
    string_key = ""
    for a in operator_string:
        if a.dagger:
            string_key += f"c{a.idx}"
        else:
            string_key += f"a{a.idx}"
    return string_key


def operator_to_qiskit_key(operator_string: list[a_op], remapping: dict[int, int]) -> str:
    """Make key string to index a fermionic operator in a dict structure.

    Args:
        operator_string: Fermionic operators.
        remapping: Map that takes indices from alpha,beta,alpha,beta
                   to alpha,alpha,beta,beta ordering.

    Returns:
        Dictionary key.
    """
    string_key = ""
    for a in operator_string:
        if a.dagger:
            string_key += f" +_{remapping[a.idx]}"
        else:
            string_key += f" -_{remapping[a.idx]}"
    return string_key[1:]


def do_extended_normal_ordering(
    fermistring: FermionicOperator,
) -> tuple[dict[str, list[a_op]], dict[str, float]]:
    """Reorder fermionic operator string.

    The string will be ordered such that all creation operators are first,
    and annihilation operators are second.
    Within a block of creation or annihilation operators the largest spin index
    will be first and the ordering will be descending.

    Returns:
        Reordered operator dict and factor dict.
    """
    operator_queue = []
    factor_queue = []
    new_operators = {}
    new_factors = {}
    for key in fermistring.operators:
        operator_queue.append(fermistring.operators[key])
        factor_queue.append(fermistring.factors[key])
    while len(operator_queue) > 0:
        next_operator = operator_queue.pop(0)
        factor = factor_queue.pop(0)
        # Doing a dumb version of cycle-sort (it is easy, but N**2)
        while True:
            current_idx = 0
            changed = False
            is_zero = False
            while True:
                a = next_operator[current_idx]
                b = next_operator[current_idx + 1]
                i = current_idx
                j = current_idx + 1
                if a.dagger and b.dagger:
                    if a.idx == b.idx:
                        is_zero = True
                    elif a.idx < b.idx:
                        next_operator[i], next_operator[j] = next_operator[j], next_operator[i]
                        factor *= -1
                        changed = True
                elif not a.dagger and b.dagger:
                    if a.idx == b.idx:
                        new_op = copy.copy(next_operator)
                        new_op.pop(j)
                        new_op.pop(i)
                        if len(new_op) > 0:
                            operator_queue.append(new_op)
                            factor_queue.append(factor)
                        next_operator[i], next_operator[j] = next_operator[j], next_operator[i]
                        factor *= -1
                        changed = True
                    else:
                        next_operator[i], next_operator[j] = next_operator[j], next_operator[i]
                        factor *= -1
                        changed = True
                elif a.dagger and not b.dagger:
                    pass
                else:
                    if a.idx == b.idx:
                        is_zero = True
                    elif a.idx < b.idx:
                        next_operator[i], next_operator[j] = next_operator[j], next_operator[i]
                        factor *= -1
                        changed = True
                current_idx += 1
                if current_idx + 1 == len(next_operator) or is_zero:
                    break
            if not changed or is_zero:
                if not is_zero:
                    key_string = operator_string_to_key(next_operator)
                    if key_string not in new_operators:
                        new_operators[key_string] = next_operator
                        new_factors[key_string] = factor
                    else:
                        new_factors[key_string] += factor
                        if abs(new_factors[key_string]) < 10**-14:
                            del new_operators[key_string]
                            del new_factors[key_string]
                break
    return new_operators, new_factors


class FermionicOperator:
    __slots__ = ("operators", "factors")

    def __init__(
        self, annihilation_operator: dict[str, list[a_op]] | a_op, factor: dict[str, float] | float
    ) -> None:
        """Initialize fermionic operator class.

        Fermionic operators are defined via an annihilation_operator dictionary where each entry is one addend of the operator.
        Each entry is a strings of annihilation operator specified via its string (key) and list of a_op (item).
        The dictionary factor contains the factor for each of the addend of the fermionic operator.

        Args:
            annihilation_operator: Annihilation operator.
            factor: Factor in front of operator.
        """
        if isinstance(annihilation_operator, dict) and not isinstance(factor, dict):
            raise ValueError(f"factor cannot be {type(factor)} when annihilation_operator is dict")
        if not isinstance(annihilation_operator, dict) and isinstance(factor, float):
            raise ValueError(f"factor cannot be dict when annihilation_operator is {type(a_op)}")
        if not isinstance(annihilation_operator, dict) and not isinstance(factor, dict):
            string_key = operator_string_to_key([annihilation_operator])
            self.operators = {}
            self.operators[string_key] = [annihilation_operator]
            self.factors = {}
            self.factors[string_key] = factor
        elif isinstance(annihilation_operator, dict) and isinstance(factor, dict):
            self.operators = annihilation_operator
            self.factors = factor
        else:
            raise ValueError(
                f"Could not assign operator of {type(annihilation_operator)} with factor of {type(factor)}"
            )

    def __add__(self, fermistring: FermionicOperator) -> FermionicOperator:
        """Addition of two fermionic operators.

        Args:
            fermistring: Fermionic operator.

        Returns:
            New fermionic operator.
        """
        # Combine annihilation string entries of two FermionicOperators.
        operators = copy.copy(self.operators)
        factors = copy.copy(self.factors)
        for string_key in fermistring.operators.keys():
            if string_key in operators.keys():
                factors[string_key] += fermistring.factors[string_key]
                if abs(factors[string_key]) < 10**-14:
                    del factors[string_key]
                    del operators[string_key]
            else:
                operators[string_key] = fermistring.operators[string_key]
                factors[string_key] = fermistring.factors[string_key]
        return FermionicOperator(operators, factors)

    def __sub__(self, fermistring: FermionicOperator) -> FermionicOperator:
        """Subtraction of two fermionic operators.

        Args:
            fermistring: Fermionic operator.

        Returns:
            New fermionic operator.
        """
        # Combine annihilation string entries of two FermionicOperators with relevant sign flip.
        operators = copy.copy(self.operators)
        factors = copy.copy(self.factors)
        for string_key in fermistring.operators.keys():
            if string_key in operators.keys():
                factors[string_key] -= fermistring.factors[string_key]
                if abs(factors[string_key]) < 10**-14:
                    del factors[string_key]
                    del operators[string_key]
            else:
                operators[string_key] = fermistring.operators[string_key]
                factors[string_key] = -fermistring.factors[string_key]
        return FermionicOperator(operators, factors)

    def __mul__(self, fermistring: FermionicOperator) -> FermionicOperator:
        """Multiplication of two fermionic operators.

        Args:
            fermistring: Fermionic operator.

        Returns:
            New fermionic operator.
        """
        operators: dict[str, list[a_op]] = {}
        factors: dict[str, float] = {}
        # Iterate over all strings in both FermionicOperators
        for string_key1 in fermistring.operators.keys():
            for string_key2 in self.operators.keys():
                # Build new strings and factors via normal ordering of product of two strings
                new_ops, new_facs = do_extended_normal_ordering(
                    FermionicOperator(
                        {
                            string_key1
                            + string_key2: self.operators[string_key2]
                            + fermistring.operators[string_key1]
                        },
                        {
                            string_key1
                            + string_key2: self.factors[string_key2] * fermistring.factors[string_key1]
                        },
                    )
                )
                for str_key in new_ops:  # pylint: disable=C0206
                    if str_key not in operators.keys():  # pylint: disable=C0201
                        operators[str_key] = new_ops[str_key]
                        factors[str_key] = new_facs[str_key]
                    else:
                        factors[str_key] += new_facs[str_key]
                        if abs(factors[str_key]) < 10**-14:
                            del factors[str_key]
                            del operators[str_key]
        return FermionicOperator(operators, factors)

    def __rmul__(self, number: float) -> FermionicOperator:
        """Multiplication of number with fermionic operator.

        Args:
            number: Number.

        Returns:
            New fermionic operator.
        """
        operators = {}
        factors = {}
        for key_string in self.operators:
            operators[key_string] = self.operators[key_string]
            factors[key_string] = self.factors[key_string] * number
        return FermionicOperator(operators, factors)

    @property
    def dagger(self) -> FermionicOperator:
        """Complex conjugation of fermionic operator.

        Returns:
            New fermionic operator.
        """
        operators = {}
        factors = {}
        for key_string in self.operators.keys():
            new_op = []
            for op in reversed(self.operators[key_string]):
                if op.dagger:
                    new_op.append(a_op(op.spinless_idx, op.spin, False))
                else:
                    new_op.append(a_op(op.spinless_idx, op.spin, True))
            new_string_key = operator_string_to_key(new_op)
            operators[new_string_key] = new_op
            factors[new_string_key] = self.factors[key_string]
        # Build new strings and factors via normal ordering of product of two strings
        operators_ordered: dict[str, list[a_op]] = {}
        factors_ordered: dict[str, float] = {}
        new_ops, new_facs = do_extended_normal_ordering(FermionicOperator(operators, factors))
        for str_key in new_ops:  # pylint: disable=C0206
            if str_key not in operators_ordered.keys():  # pylint: disable=C0201
                operators_ordered[str_key] = new_ops[str_key]
                factors_ordered[str_key] = new_facs[str_key]
            else:
                factors_ordered[str_key] += new_facs[str_key]
        return FermionicOperator(operators_ordered, factors_ordered)

    @property
    def operator_count(self) -> dict[int, int]:
        """Count number of operators of different lengths.

        Returns:
            Number of operators of every length.
        """
        op_count = {}
        for string_key in self.operators.keys():
            op_lenght = len(self.operators[string_key])
            if op_lenght not in op_count:
                op_count[op_lenght] = 1
            else:
                op_count[op_lenght] += 1
        return op_count

    def get_qiskit_form(self, num_orbs: int) -> dict[str, float]:
        """Get fermionic operator on qiskit form.

        Args:
            num_orbs: Number of spatial orbitals.

        Returns:
            Fermionic operators on qiskit form.
        """
        qiskit_form = {}
        remapping = {}
        #  Map indices from alpha,beta,alpha,beta to alpha,alpha,beta,beta.
        for i in range(2 * num_orbs):
            if i < num_orbs:
                remapping[2 * i] = i
            else:
                remapping[2 * i + 1 - 2 * num_orbs] = i
        for key_string in self.operators:
            qiskit_str = operator_to_qiskit_key(self.operators[key_string], remapping)
            qiskit_form[qiskit_str] = self.factors[key_string]
        return qiskit_form

    def get_folded_operator(
        self, num_inactive_orbs: int, num_active_orbs: int, num_virtual_orbs: int
    ) -> FermionicOperator:
        r"""Get folded operator.

        Operator is split into spaces

        .. math::
            \hat{O} = \hat{O}_I\otimes \hat{O}_A\otimes \hat{O}_V

        giving the expectation values as

        .. math::
            \left<0\left(\boldsymbol{\theta}\right)\left|\hat{O}\right|0\left(\boldsymbol{\theta}\right)\right>
            = \left<I\left|\hat{O}_{I}\right|I\right>\otimes \left<A\left(\boldsymbol{\theta}\right)\left|\hat{O}_{A}\right|A\left(\boldsymbol{\theta}\right)\right>
                \otimes\left<V\left|\hat{O}_{V}\right|V\right>

        where the inactive and virtual parts follow simple annihilation operator arguments, leaving just the active part.

        Warning, multiplication of folded operators, might give wrong operators.
        (I have not quite figured out a good programming structure that will not allow multiplication after folding)

        Note, that the indices of the folded operator is remapped, such that idx=0 is the first index in the active space.

        Args:
            num_inactive_orbs: Number of spatial inactive orbitals.
            num_active_orbs: Number of spatial active orbitals.
            num_virtual_orbs: Number of spatial virtual orbitals.

        Returns:
           Folded fermionic operator.
        """
        operators = {}
        factors: dict[str, float] = {}
        inactive_idx = []
        active_idx = []
        virtual_idx = []
        # Get indices of spaces
        for i in range(2 * num_inactive_orbs + 2 * num_active_orbs + 2 * num_virtual_orbs):
            if i < 2 * num_inactive_orbs:
                inactive_idx.append(i)
            elif i < 2 * num_inactive_orbs + 2 * num_active_orbs:
                active_idx.append(i)
            else:
                virtual_idx.append(i)

        # Loop over string of annihilation operators
        for key_string in self.operators.keys():
            virtual = []
            virtual_dagger = []
            inactive = []
            inactive_dagger = []
            active = []
            active_dagger = []
            fac = 1
            # Loop over individual annihilation operator and sort into spaces
            for anni in self.operators[key_string]:
                if anni.dagger:
                    if anni.idx in inactive_idx:
                        inactive_dagger.append(anni.idx)
                    elif anni.idx in active_idx:
                        active_dagger.append(
                            a_op(anni.spinless_idx - num_inactive_orbs, anni.spin, anni.dagger)
                        )
                    elif anni.idx in virtual_idx:
                        virtual_dagger.append(anni.idx)
                else:
                    if anni.idx in inactive_idx:
                        inactive.append(anni.idx)
                    elif anni.idx in active_idx:
                        active.append(a_op(anni.spinless_idx - num_inactive_orbs, anni.spin, anni.dagger))
                    elif anni.idx in virtual_idx:
                        virtual.append(anni.idx)
            # Any virtual indices will make the operator evaluate to zero.
            if len(virtual) != 0 or len(virtual_dagger) != 0:
                continue
            active_op = active_dagger + active  # list
            bra_side = inactive_dagger
            ket_side = inactive
            # The inactive bra and ket side must end up giving identical state vectors.
            if bra_side != ket_side:
                continue
            if len(inactive_dagger) % 2 == 1 and len(active_dagger) % 2 == 1:
                fac *= -1
            # Calculate sign coming from flipping the order of the ket side.
            # It has to be "flipped" to match the order on the bra side.
            ket_flip_fac = 1
            for i in range(1, len(ket_side) + 1):
                if i % 2 == 0:
                    ket_flip_fac *= -1
            fac *= ket_flip_fac
            new_key = operator_string_to_key(active_op)
            if new_key in factors:
                factors[new_key] += fac * self.factors[key_string]
            else:
                factors[new_key] = fac * self.factors[key_string]
                operators[new_key] = active_op
        return FermionicOperator(operators, factors)

    def get_info(self) -> tuple[list[list[int]], list[list[int]], list[float]]:
        """Return operator excitation in ordered strings with coefficient."""
        excitations = list(self.factors.keys())
        coefficients = list(self.factors.values())
        creation = []
        annihilation = []
        for op_string in excitations:
            numbers = re.findall(r"\d+", op_string)
            numbers = [int(num) for num in numbers]
            midpoint = len(numbers) // 2
            c = numbers[:midpoint]
            a = numbers[midpoint:]
            creation.append(c)
            annihilation.append(a)
        return annihilation, creation, coefficients
