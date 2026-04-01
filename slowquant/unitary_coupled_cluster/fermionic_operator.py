from __future__ import annotations

import copy
import re


def operator_to_qiskit_key(operator_string: tuple[tuple[int, bool], ...], remapping: dict[int, int]) -> str:
    """Make key string to index a fermionic operator in a dict structure.

    Args:
        operator_string: Fermionic operators.
        remapping: Map that takes indices from alpha,beta,alpha,beta
                   to alpha,alpha,beta,beta ordering.

    Returns:
        Dictionary key.
    """
    op_key = ""
    for a in operator_string:
        if a[1]:
            op_key += f" +_{remapping[a[0]]}"
        else:
            op_key += f" -_{remapping[a[0]]}"
    return op_key[1:]


def nondagger_dagger_sort(
    fermistring: list[int],
    daggers: list[bool],
    phase: int,
):
    """Reorder fermionic operator string."""

    factor = phase
    next_operator = fermistring
    # Doing a dumb version of cycle-sort
    while True:
        changed = False
        is_zero = False
        i = 0

        while i < len(next_operator) - 1:
            idx_a = next_operator[i]
            idx_b = next_operator[i + 1]
            is_cr_a = daggers[i]
            is_cr_b = daggers[i + 1]

            if not is_cr_a and is_cr_b:  # Annihilation / Creation
                if idx_a == idx_b:
                    if len(next_operator) > 2:
                        new_op = next_operator[:i] + next_operator[i + 2:]
                        new_daggers = daggers[:i] + daggers[i + 2:]
                        yield from nondagger_dagger_sort(new_op, new_daggers, factor)

                next_operator[i], next_operator[i + 1] = next_operator[i + 1], next_operator[i]
                daggers[i], daggers[i + 1] = daggers[i + 1], daggers[i]
                factor *= -1
                changed = True

            # If it's Creation / Annihilation, it's already in correct order
            i += 1

        if is_zero:
            break

        if not changed:
            num_daggers = sum(daggers)
            yield next_operator[:num_daggers], next_operator[num_daggers:], factor
            break


def insertion_sort(indices: list[int]) -> tuple[list[int], int]:
    phase = 1
    for i in range(1, len(indices)):
        j = i
        while j > 0 and indices[j] > indices[j-1]:
            indices[j], indices[j-1] = indices[j-1], indices[j]
            phase *= -1
            j -= 1
    return indices, phase


def do_product_extended_normal_ordering(
    fermistring1: tuple[tuple[int, ...], tuple[int, ...]],
    fermistring2: tuple[tuple[int, ...], tuple[int, ...]],
) -> tuple[list[tuple[tuple[int, ...], tuple[int, ...]]], list[int]]:
    """Reorder fermionic operator string.

    The string will be ordered such that all creation operators are first,
    and annihilation operators are second.
    Within a block of creation or annihilation operators the largest spin index
    will be first and the ordering will be descending.

    Returns:
        Reordered operator dict and factor dict.
    """
    dagger1_set = set(fermistring1[0])
    dagger2_set = set(fermistring2[0])
    nondagger1_set = set(fermistring1[1])
    nondagger2_set = set(fermistring2[1])
    if len(nondagger1_set) + len(dagger2_set) == len(nondagger1_set.union(dagger2_set)):
        # No index overlap between non-dagger left-side and dagger right-side.
        if len(dagger1_set) + len(dagger2_set) != len(dagger1_set.union(dagger2_set)):
            # Same index creation operator.
            return [], []
        elif len(nondagger1_set) + len(nondagger2_set) != len(nondagger1_set.union(nondagger2_set)):
            # Same index annihilation operator.
            return [], []
        phase = 1
        if len(fermistring1[1]) % 2 != 0 and len(fermistring2[0]) % 2 != 0:
            # Only phase change if both are an odd lenght.
            phase *= -1
        dagger_list, phase_fac = insertion_sort(list(fermistring1[0]) + list(fermistring2[0]))
        phase *= phase_fac
        nondagger_list, phase_fac = insertion_sort(list(fermistring1[1]) + list(fermistring2[1]))
        phase *= phase_fac
        return [(tuple(dagger_list), tuple(nondagger_list))], [phase] 

    new_operators = []
    new_phases = []
    fermistring = list(fermistring1[1]) + list(fermistring2[0])
    daggers = [False]*len(fermistring1[1]) + [True]*len(fermistring2[0])
    for dagger_tmp, nondagger_tmp, phase in nondagger_dagger_sort(fermistring, daggers, 1):
        dagger_tmp_set = set(dagger_tmp)
        nondagger_tmp_set = set(nondagger_tmp)
        if len(dagger1_set) + len(dagger_tmp_set) != len(dagger1_set.union(dagger_tmp_set)):
            # Same index creation operator.
            continue
        elif len(nondagger_tmp_set) + len(nondagger2_set) != len(nondagger_tmp_set.union(nondagger2_set)):
            # Same index annihilation operator.
            continue
        dagger_list, phase_fac = insertion_sort(list(fermistring1[0]) + dagger_tmp)
        phase *= phase_fac
        nondagger_list, phase_fac = insertion_sort(nondagger_tmp + list(fermistring2[1]))
        phase *= phase_fac
        new_operators.append((tuple(dagger_list), tuple(nondagger_list)))
        new_phases.append(phase)
    return new_operators, new_phases


class FermionicOperator:
    __slots__ = ("operators",)

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
        else:
            raise ValueError(f"Could not assign operator of {type(annihilation_operator)}.")

    def __add__(self, fermistring: FermionicOperator) -> FermionicOperator:
        """Addition of two fermionic operators.

        Args:
            fermistring: Fermionic operator.

        Returns:
            New fermionic operator.
        """
        # Combine annihilation string entries of two FermionicOperators.
        operators = copy.copy(self.operators)
        for op_key in fermistring.operators.keys():
            if op_key in operators.keys():
                operators[op_key] += fermistring.operators[op_key]
                if abs(operators[op_key]) < 10**-14:
                    del operators[op_key]
            else:
                operators[op_key] = fermistring.operators[op_key]
        return FermionicOperator(operators)

    def __iadd__(self, fermistring: FermionicOperator) -> FermionicOperator:
        """Inplace addition of two fermionic operators.

        Args:
            fermistring: Fermionic operator.

        Returns:
            Updated fermionic operator.
        """
        for op_key in fermistring.operators.keys():
            if op_key in self.operators.keys():
                self.operators[op_key] += fermistring.operators[op_key]
                if abs(self.operators[op_key]) < 10**-14:
                    del self.operators[op_key]
            else:
                self.operators[op_key] = fermistring.operators[op_key]
        return self

    def __sub__(self, fermistring: FermionicOperator) -> FermionicOperator:
        """Subtraction of two fermionic operators.

        Args:
            fermistring: Fermionic operator.

        Returns:
            New fermionic operator.
        """
        # Combine annihilation string entries of two FermionicOperators with relevant sign flip.
        operators = copy.copy(self.operators)
        for op_key in fermistring.operators.keys():
            if op_key in operators.keys():
                operators[op_key] -= fermistring.operators[op_key]
                if abs(operators[op_key]) < 10**-14:
                    del operators[op_key]
            else:
                operators[op_key] = -fermistring.operators[op_key]
        return FermionicOperator(operators)

    def __isub__(self, fermistring: FermionicOperator) -> FermionicOperator:
        """Inplace subtraction of two fermionic operators.

        Args:
            fermistring: Fermionic operator.

        Returns:
            Update fermionic operator.
        """
        # Combine annihilation string entries of two FermionicOperators with relevant sign flip.
        for op_key in fermistring.operators.keys():
            if op_key in self.operators.keys():
                self.operators[op_key] -= fermistring.operators[op_key]
                if abs(self.operators[op_key]) < 10**-14:
                    del self.operators[op_key]
            else:
                self.operators[op_key] = -fermistring.operators[op_key]
        return self

    def __mul__(self, fermistring: FermionicOperator | float | int) -> FermionicOperator:
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
        elif type(fermistring) is FermionicOperator:
            operators = {}  # type: ignore
            # Iterate over all strings in both FermionicOperators
            for op_key1 in fermistring.operators.keys():
                for op_key2 in self.operators.keys():
                    factor = self.operators[op_key2] * fermistring.operators[op_key1]
                    if abs(factor) < 10**-14:
                        continue
                    # Build new strings and factors via normal ordering of product of two strings
                    new_ops, phases = do_product_extended_normal_ordering(self.operators[op_key2], fermistring.operators[op_key1])
                    for op_key, phase in zip(new_ops, phases):
                        if op_key not in operators.keys():
                            operators[op_key] = factor*phase
                        else:
                            operators[op_key] += factor*phase
                            if abs(operators[op_key]) < 10**-14:
                                del operators[op_key]
        else:
            raise TypeError(f"Got unknown type of fermistring: {type(fermistring)}")
        return FermionicOperator(operators)

    def __imul__(self, fermistring: FermionicOperator | float | int) -> FermionicOperator:
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
        elif type(fermistring) is FermionicOperator:
            operators: dict[tuple[tuple[int, bool], ...], float] = {}
            # Iterate over all strings in both FermionicOperators
            for op_key1 in fermistring.operators.keys():
                for op_key2 in self.operators.keys():
                    factor = self.operators[op_key2] * fermistring.operators[op_key1]
                    if abs(factor) < 10**-14:
                        continue
                    # Build new strings and factors via normal ordering of product of two strings
                    new_ops, phases = do_product_extended_normal_ordering(self.operators[op_key2], fermistring.operators[op_key1])
                    for op_key, phase in zip(new_ops, phases):
                        if op_key not in operators.keys():
                            operators[op_key] = factor*phase
                        else:
                            operators[op_key] += factor*phase
                            if abs(operators[op_key]) < 10**-14:
                                del operators[op_key]
            self.operators = operators
        else:
            raise TypeError(f"Got unknown type of fermistring: {type(fermistring)}")
        return self

    def __rmul__(self, number: float) -> FermionicOperator:
        """Multiplication of number with fermionic operator.

        Args:
            number: Number.

        Returns:
            New fermionic operator.
        """
        operators = {}
        for op_key in self.operators.keys():
            operators[op_key] = self.operators[op_key] * number
        return FermionicOperator(operators)

    def __neg__(self):
        """Negate the factors in a fermionic operator.

        Retunrs:
            New fermionic operator.
        """
        operators = copy.copy(self.operators)
        for op_key in self.operators.keys():
            operators[op_key] = -operators[op_key]
        return FermionicOperator(operators)

    @property
    def dagger(self) -> FermionicOperator:
        """Complex conjugation of fermionic operator.

        Returns:
            New fermionic operator.
        """
        operators = {}
        for op_key in self.operators.keys():
            new_op = []
            for op in reversed(op_key):
                if op[1]:
                    new_op.append((op[0], False))
                else:
                    new_op.append((op[0], True))
            new_op_key = tuple(new_op)
            operators[new_op_key] = self.operators[op_key]
        # Do normal ordering of comlex conjugated operator.
        operators_ordered = do_extended_normal_ordering(FermionicOperator(operators))
        return FermionicOperator(operators_ordered)

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
        for string, fac in self.operators.items():
            op_key = ""
            for a in string:
                if a[1]:
                    op_key += f"c{a[0]}"
                else:
                    op_key += f"a{a[0]}"
            operator[op_key] = fac
        return operator

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
        for op_key in self.operators.keys():
            qiskit_str = operator_to_qiskit_key(op_key, remapping)
            qiskit_form[qiskit_str] = self.operators[op_key]
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
        operators: dict[tuple[tuple[int, bool], ...], float] = {}
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
        for op_key in self.operators.keys():
            virtual = []
            virtual_dagger = []
            inactive = []
            inactive_dagger = []
            active = []
            active_dagger = []
            fac = 1
            # Loop over individual annihilation operator and sort into spaces
            for anni in op_key:
                if anni[1]:
                    if anni[0] in inactive_idx:
                        inactive_dagger.append(anni[0])
                    elif anni[0] in active_idx:
                        active_dagger.append((anni[0] - 2 * num_inactive_orbs, anni[1]))
                    elif anni[0] in virtual_idx:
                        virtual_dagger.append(anni[0])
                elif anni[0] in inactive_idx:
                    inactive.append(anni[0])
                elif anni[0] in active_idx:
                    active.append((anni[0] - 2 * num_inactive_orbs, anni[1]))
                elif anni[0] in virtual_idx:
                    virtual.append(anni[0])
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
            new_key = tuple(active_op)
            if new_key in operators.keys():
                operators[new_key] += fac * self.operators[op_key]
            else:
                operators[new_key] = fac * self.operators[op_key]
        return FermionicOperator(operators)

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
