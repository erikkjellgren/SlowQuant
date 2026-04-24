from __future__ import annotations

import copy
import re

from collections import defaultdict
from collections.abc import Generator
import itertools


def operator_to_qiskit_key(operator_string: tuple[tuple[int, ...], tuple[int, ...]]) -> str:
    """Make key string to index a hardcoreboson operator in a dict structure.

    Args:
        operator_string: Hardcoreboson operators.

    Returns:
        Dictionary key.
    """
    op_key = ""
    for a in operator_string[0]:
        op_key += f" +_{a}"
    for a in operator_string[1]:
        op_key += f" -_{a}"
    return op_key[1:]


def do_product_extended_normal_ordering(
    fermistring1: tuple[tuple[int, ...], tuple[int, ...]],
    fermistring2: tuple[tuple[int, ...], tuple[int, ...]],
    dagger1_set: set[int],
    dagger2_set: set[int],
    nondagger1_set: set[int],
    nondagger2_set: set[int],
) -> Generator[tuple[tuple[tuple[int, ...], tuple[int, ...]], int], None, None]:
    if nondagger1_set.isdisjoint(dagger2_set):
        # No index overlap between non-dagger left-side and dagger right-side.
        is_zero = False
        if not dagger1_set.isdisjoint(dagger2_set):
            # Same index creation operator.
            is_zero = True
        elif not nondagger1_set.isdisjoint(nondagger2_set):
            # Same index annihilation operator.
            is_zero = True
        if not is_zero:
            phase = 1
            if len(fermistring1[1]) % 2 != 0 and len(fermistring2[0]) % 2 != 0:
                # Only phase change if both are an odd lenght.
                phase *= -1
            # sort the dagger part
            dagger_list = [*fermistring1[0], *fermistring2[0]]
            dagger_list.sort()
            # sort non-dagger part
            nondagger_list = [*fermistring1[1], *fermistring2[1]]
            nondagger_list.sort()
            yield (tuple(dagger_list), tuple(nondagger_list)), phase
    else:
        overlap_idxs = nondagger1_set.intersection(dagger2_set)
        for k in range(0, len(overlap_idxs) + 1):
            for contract_idxs in itertools.combinations(overlap_idxs, k):
                nondagger_tmp = list(fermistring1[1])
                dagger_tmp = list(fermistring2[0])
                phase = 1
                for contract_idx in contract_idxs:
                    nondagger_loc = nondagger_tmp.index(contract_idx)
                    dagger_loc = dagger_tmp.index(contract_idx)
                    # phase *= (-1)**(len(nondagger_tmp) - 1 - nondagger_loc) * (-1)**(dagger_loc)
                    phase *= 1 if (len(nondagger_tmp) - 1 - nondagger_loc + dagger_loc) % 2 == 0 else -1
                    nondagger_tmp.pop(nondagger_loc)
                    dagger_tmp.pop(dagger_loc)
                if len(nondagger_tmp) % 2 == 1 and len(dagger_tmp) % 2 == 1:
                    phase *= -1
                dagger_tmp_set = set(dagger_tmp)
                nondagger_tmp_set = set(nondagger_tmp)
                if not dagger1_set.isdisjoint(dagger_tmp_set):
                    # Same index creation operator.
                    continue
                elif not nondagger_tmp_set.isdisjoint(nondagger2_set):
                    # Same index annihilation operator.
                    continue
                # sort the dagger part
                dagger_list = [*fermistring1[0], *dagger_tmp]
                # Doing insertion sort, left-side part is already sorted.
                # Hence not starting from 1.
                for i in range(len(fermistring1[0]), len(dagger_list)):
                    j = i
                    while j > 0 and dagger_list[j] > dagger_list[j - 1]:
                        dagger_list[j], dagger_list[j - 1] = dagger_list[j - 1], dagger_list[j]
                        phase *= -1
                        j -= 1
                # sort non-dagger part
                nondagger_list = [*nondagger_tmp, *fermistring2[1]]
                # Doing insertion sort, left-side part is already sorted.
                # Hence not starting from 1.
                for i in range(len(nondagger_tmp), len(nondagger_list)):
                    j = i
                    while j > 0 and nondagger_list[j] > nondagger_list[j - 1]:
                        nondagger_list[j], nondagger_list[j - 1] = nondagger_list[j - 1], nondagger_list[j]
                        phase *= -1
                        j -= 1
                yield (tuple(dagger_list), tuple(nondagger_list)), phase


class HardcorebosonOperator:
    __slots__ = ("_operator_sets", "operators")

    def __init__(
        self,
        annihilation_operator: dict[tuple[tuple[int, ...], tuple[int, ...]], float],
    ) -> None:
        """Initialize fermionic operator class.

        Fermionic operators are defined via an annihilation_operator dictionary where each entry is one of the annihilation operators.
        Each entry is a tuples (key) of an integer (spin orbital index) and a bool (dagger/not dagger)
        which defines the keys and a float (value) that is the factor in front of the annihilation string.

        Args:
            annihilation_operator: Annihilation operator.
        """
        if isinstance(annihilation_operator, dict):
            self.operators = annihilation_operator
            self._operator_sets: (
                dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[set[int], set[int]]] | None
            ) = None
        else:
            raise ValueError(f"Could not assign operator of {type(annihilation_operator)}.")

    def operator_sets(self, key: tuple[tuple[int, ...], tuple[int, ...]]) -> tuple[set[int], set[int]]:
        """Get set represtion of fermionic string.

        Args:
            key: Fermionic string.

        Returns:
            Set of creation operators and set of annihilation operators.
        """
        if self._operator_sets is None:
            self._operator_sets = {}
            for op_key in self.operators:
                self._operator_sets[op_key] = (set(op_key[0]), set(op_key[1]))
        return self._operator_sets[key]

    def __add__(self, fermistring: HardcorebosonOperator) -> HardcorebosonOperator:
        """Addition of two fermionic operators.

        Args:
            fermistring: Fermionic operator.

        Returns:
            New fermionic operator.
        """
        # Combine annihilation string entries of two HardcorebosonOperators.
        operators = copy.copy(self.operators)
        for op_key, fac in fermistring.operators.items():
            if op_key in operators.keys():
                operators[op_key] += fac
                if abs(operators[op_key]) < 10**-14:
                    del operators[op_key]
            else:
                operators[op_key] = fac
        return HardcorebosonOperator(operators)

    def __iadd__(self, fermistring: HardcorebosonOperator) -> HardcorebosonOperator:
        """Inplace addition of two fermionic operators.

        Args:
            fermistring: Fermionic operator.

        Returns:
            Updated fermionic operator.
        """
        for op_key, fac in fermistring.operators.items():
            if op_key in self.operators.keys():
                self.operators[op_key] += fac
                if abs(self.operators[op_key]) < 10**-14:
                    del self.operators[op_key]
            else:
                self.operators[op_key] = fac
                self._operator_sets = None
        return self

    def __sub__(self, fermistring: HardcorebosonOperator) -> HardcorebosonOperator:
        """Subtraction of two fermionic operators.

        Args:
            fermistring: Fermionic operator.

        Returns:
            New fermionic operator.
        """
        # Combine annihilation string entries of two HardcorebosonOperators with relevant sign flip.
        operators = copy.copy(self.operators)
        for op_key, fac in fermistring.operators.items():
            if op_key in operators.keys():
                operators[op_key] -= fac
                if abs(operators[op_key]) < 10**-14:
                    del operators[op_key]
            else:
                operators[op_key] = -fac
        return HardcorebosonOperator(operators)

    def __isub__(self, fermistring: HardcorebosonOperator) -> HardcorebosonOperator:
        """Inplace subtraction of two fermionic operators.

        Args:
            fermistring: Fermionic operator.

        Returns:
            Update fermionic operator.
        """
        # Combine annihilation string entries of two HardcorebosonOperators with relevant sign flip.
        for op_key, fac in fermistring.operators.items():
            if op_key in self.operators.keys():
                self.operators[op_key] -= fac
                if abs(self.operators[op_key]) < 10**-14:
                    del self.operators[op_key]
            else:
                self.operators[op_key] = -fac
                self._operator_sets = None
        return self

    def __mul__(self, fermistring: HardcorebosonOperator | float | int) -> HardcorebosonOperator:
        """Multiplication of two fermionic operators.

        Args:
            fermistring: Fermionic operator.

        Returns:
            New fermionic operator.
        """
        if type(fermistring) in (float, int):
            operators = copy.copy(self.operators)
            for op_key in self.operators.keys():
                # The name fermistring is misleading here.
                operators[op_key] *= fermistring  # type: ignore
        elif type(fermistring) is HardcorebosonOperator:
            operators = defaultdict(float)
            # Iterate over all strings in both HardcorebosonOperators
            for op_key1, fac1 in fermistring.operators.items():
                dagger1_set, nondagger1_set = fermistring.operator_sets(op_key1)
                for op_key2, fac2 in self.operators.items():
                    fac = fac1 * fac2
                    if abs(fac) < 10**-14:
                        continue
                    dagger2_set, nondagger2_set = self.operator_sets(op_key2)
                    # Build new strings and factors via normal ordering of product of two strings
                    for op_key, phase in do_product_extended_normal_ordering(
                        op_key2, op_key1, dagger2_set, dagger1_set, nondagger2_set, nondagger1_set
                    ):
                        operators[op_key] += fac * phase
        else:
            raise TypeError(f"Got unknown type of fermistring: {type(fermistring)}")
        operators = {op: fac for op, fac in operators.items() if abs(fac) >= 1e-14}
        return HardcorebosonOperator(operators)

    def __imul__(self, fermistring: HardcorebosonOperator | float | int) -> HardcorebosonOperator:
        """Inplace multiplication of two fermionic operators.

        Args:
            fermistring: Fermionic operator.

        Returns:
            Updated fermionic operator.
        """
        if type(fermistring) in (float, int):
            for op_key in self.operators.keys():
                # The name fermistring is misleading here.
                self.operators[op_key] *= fermistring  # type: ignore
            self.operators = {op: fac for op, fac in self.operators.items() if abs(fac) >= 1e-14}
        elif type(fermistring) is HardcorebosonOperator:
            operators: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = defaultdict(float)
            # Iterate over all strings in both HardcorebosonOperators
            for op_key1, fac1 in fermistring.operators.items():
                dagger1_set, nondagger1_set = fermistring.operator_sets(op_key1)
                for op_key2, fac2 in self.operators.items():
                    fac = fac1 * fac2
                    if abs(fac) < 10**-14:
                        continue
                    dagger2_set, nondagger2_set = self.operator_sets(op_key2)
                    # Build new strings and factors via normal ordering of product of two strings
                    for op_key, phase in do_product_extended_normal_ordering(
                        op_key2, op_key1, dagger2_set, dagger1_set, nondagger2_set, nondagger1_set
                    ):
                        operators[op_key] += fac * phase
            self.operators = {op: fac for op, fac in operators.items() if abs(fac) >= 1e-14}
            self._operator_sets = None
        else:
            raise TypeError(f"Got unknown type of fermistring: {type(fermistring)}")
        return self

    def __rmul__(self, number: float) -> HardcorebosonOperator:
        """Multiplication of number with fermionic operator.

        Args:
            number: Number.

        Returns:
            New fermionic operator.
        """
        operators = {op: fac * number for op, fac in self.operators.items() if abs(fac * number) >= 1e-14}
        return HardcorebosonOperator(operators)

    def __neg__(self) -> HardcorebosonOperator:
        """Negate the factors in a fermionic operator.

        Retunrs:
            New fermionic operator.
        """
        operators = {op: -fac for op, fac in self.operators.items()}
        return HardcorebosonOperator(operators)

    @property
    def dagger(self) -> HardcorebosonOperator:
        r"""Complex conjugation of fermionic operator.

        After dagger'ing, the operator blocks need to be reversed.
        This give a number phase change that follows a shifted triangular number sequence,

        .. math::
            \Gamma = (-1)^{(k(k-1)/2 + l(l-1)/2}

        with :math:`k` being the number of creation and l the number of annihilation operators.

        Returns:
            New fermionic operator.
        """
        operators = {}
        for op_key, fac in self.operators.items():
            k = len(op_key[1])
            l = len(op_key[0])
            phase_changes = (k * k - k + l * l - l) // 2
            sign = 1.0 - 2.0 * (phase_changes & 1)
            operators[(op_key[1][::-1], op_key[0][::-1])] = fac * sign
        return HardcorebosonOperator(operators)

    @property
    def operator_count(self) -> dict[int, int]:
        """Count number of operators of different lengths.

        Returns:
            Number of operators of every length.
        """
        op_count = {}
        for op_key in self.operators.keys():
            op_lenght = len(op_key)
            if op_lenght not in op_count:
                op_count[op_lenght] = 1
            else:
                op_count[op_lenght] += 1
        return op_count

    @property
    def operators_readable(self) -> dict[str, float]:
        """Get the operator in human readable format.

        Returns:
            Operator in humanreable format.
        """
        operator = {}
        for (dagger_string, nondagger_string), fac in self.operators.items():
            op_key = ""
            for a in dagger_string:
                op_key += f"c{a}"
            for a in nondagger_string:
                op_key += f"a{a}"
            operator[op_key] = fac
        return operator

    def get_qiskit_form(self) -> dict[str, float]:
        """Get fermionic operator on qiskit form.

        Returns:
            Fermionic operators on qiskit form.
        """
        qiskit_form = {}
        for op_key in self.operators.keys():
            qiskit_str = operator_to_qiskit_key(op_key)
            qiskit_form[qiskit_str] = self.operators[op_key]
        return qiskit_form

    def get_folded_operator(
        self, num_inactive_orbs: int, num_active_orbs: int, num_virtual_orbs: int
    ) -> HardcorebosonOperator:
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
        operators: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
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
        for op_key, fac in self.operators.items():
            virtual = []
            virtual_dagger = []
            inactive = []
            inactive_dagger = []
            active = []
            active_dagger = []
            phase = 1
            # Loop over individual annihilation operator and sort into spaces
            # Loop over daggers
            for anni in op_key[0]:
                if anni in inactive_idx:
                    inactive_dagger.append(anni)
                elif anni in active_idx:
                    active_dagger.append(anni - 2 * num_inactive_orbs)
                elif anni in virtual_idx:
                    virtual_dagger.append(anni)
            # Loop over non-daggers
            for anni in op_key[1]:
                if anni in inactive_idx:
                    inactive.append(anni)
                elif anni in active_idx:
                    active.append(anni - 2 * num_inactive_orbs)
                elif anni in virtual_idx:
                    virtual.append(anni)
            # Any virtual indices will make the operator evaluate to zero.
            if len(virtual) != 0 or len(virtual_dagger) != 0:
                continue
            active_op = (tuple(active_dagger), tuple(active))
            bra_side = inactive_dagger
            ket_side = inactive
            # The inactive bra and ket side must end up giving identical state vectors.
            if bra_side != ket_side:
                continue
            if len(inactive_dagger) % 2 == 1 and len(active_dagger) % 2 == 1:
                phase *= -1
            # Calculate sign coming from flipping the order of the ket side.
            # It has to be "flipped" to match the order on the bra side.
            ket_flip_fac = 1
            for i in range(1, len(ket_side) + 1):
                if i % 2 == 0:
                    ket_flip_fac *= -1
            phase *= ket_flip_fac
            if active_op in operators.keys():
                operators[active_op] += fac * phase
            else:
                operators[active_op] = fac * phase
        return HardcorebosonOperator(operators)

    def get_info(self) -> tuple[list[list[int]], list[list[int]], list[float]]:
        """Return operator excitation in ordered strings with coefficient."""
        operator = self.operators_readable
        excitations = list(operator.keys())
        coefficients = list(operator.values())
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

