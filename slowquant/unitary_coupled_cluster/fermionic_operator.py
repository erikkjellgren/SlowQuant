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


def do_extended_normal_ordering(
    fermistring: FermionicOperator,
) -> dict[tuple[tuple[int, bool], ...], float | complex]:
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
    for key in fermistring.operators.keys():
        operator_queue.append(list(key))
        factor_queue.append(fermistring.operators[key])
    while len(operator_queue) > 0:
        next_operator = operator_queue.pop(0)
        factor = factor_queue.pop(0)
        # Doing a dumb version of cycle-sort (it is easy, but N**2)
        while True:
            current_idx = 0
            changed = False
            is_zero = False
            while True:
                if len(next_operator) == 0:
                    break
                a = next_operator[current_idx]
                b = next_operator[current_idx + 1]
                i = current_idx
                j = current_idx + 1
                if a[1] and b[1]:
                    if a[0] == b[0]:
                        is_zero = True
                    elif a[0] < b[0]:
                        next_operator[i], next_operator[j] = next_operator[j], next_operator[i]
                        factor *= -1
                        changed = True
                elif not a[1] and b[1]:
                    if a[0] == b[0]:
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
                elif a[1] and not b[1]:
                    pass
                elif a[0] == b[0]:
                    is_zero = True
                elif a[0] < b[0]:
                    next_operator[i], next_operator[j] = next_operator[j], next_operator[i]
                    factor *= -1
                    changed = True
                current_idx += 1
                if current_idx + 1 == len(next_operator) or is_zero:
                    break
            if not changed or is_zero:
                if not is_zero:
                    op_key = tuple(next_operator)
                    if op_key not in new_operators:
                        new_operators[op_key] = factor
                    else:
                        new_operators[op_key] += factor
                        if abs(new_operators[op_key]) < 10**-14:
                            del new_operators[op_key]
                break
    return new_operators


class FermionicOperator:
    __slots__ = ("operators",)

    def __init__(
        self,
        annihilation_operator: dict[tuple[tuple[int, bool], ...], float | complex],
    ) -> None:
        """Initialize fermionic operator class.

        Fermionic operators are defined via an annihilation_operator dictionary where each entry is one of the annihilation operators.
        Each entry is a tuples (key) of an integer (spin orbital index) and a bool (dagger/not dagger)
        which defines the keys and a complex (value) that is the factor in front of the annihilation string.

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

    def __mul__(self, fermistring: FermionicOperator | float | int | complex) -> FermionicOperator:
        """Multiplication of two fermionic operators.

        Args:
            fermistring: Fermionic operator.

        Returns:
            New fermionic operator.
        """
        if type(fermistring) in (float, int, complex):
            operators = copy.copy(self.operators)
            for op_key in self.operators.keys():
                # The name fermistring is misleading here.
                operators[op_key] *= fermistring  # type: ignore
        elif type(fermistring) is FermionicOperator:
            operators = {}  # type: ignore
            # Iterate over all strings in both FermionicOperators
            for op_key1 in fermistring.operators.keys():
                for op_key2 in self.operators.keys():
                    # Build new strings and factors via normal ordering of product of two strings
                    new_ops = do_extended_normal_ordering(
                        FermionicOperator(
                            {op_key2 + op_key1: self.operators[op_key2] * fermistring.operators[op_key1]}
                        )
                    )
                    for op_key in new_ops.keys():
                        if op_key not in operators.keys():
                            operators[op_key] = new_ops[op_key]
                        else:
                            operators[op_key] += new_ops[op_key]
                            if abs(operators[op_key]) < 10**-14:
                                del operators[op_key]
        else:
            raise TypeError(f"Got unknown type of fermistring: {type(fermistring)}")
        return FermionicOperator(operators)

    def __imul__(self, fermistring: FermionicOperator | float | int | complex) -> FermionicOperator:
        """Inplace multiplication of two fermionic operators.

        Args:
            fermistring: Fermionic operator.

        Returns:
            Updated fermionic operator.
        """
        if type(fermistring) in (float, int, complex):
            for op_key in self.operators.keys():
                # The name fermistring is misleading here.
                self.operators[op_key] *= fermistring  # type: ignore
        elif type(fermistring) is FermionicOperator:
            operators: dict[tuple[tuple[int, bool], ...], float | complex] = {}
            # Iterate over all strings in both FermionicOperators
            for op_key1 in fermistring.operators.keys():
                for op_key2 in self.operators.keys():
                    # Build new strings and factors via normal ordering of product of two strings
                    new_ops = do_extended_normal_ordering(
                        FermionicOperator(
                            {op_key2 + op_key1: self.operators[op_key2] * fermistring.operators[op_key1]}
                        )
                    )
                    for op_key in new_ops.keys():
                        if op_key not in operators.keys():
                            operators[op_key] = new_ops[op_key]
                        else:
                            operators[op_key] += new_ops[op_key]
                            if abs(operators[op_key]) < 10**-14:
                                del operators[op_key]
            self.operators = operators
        else:
            raise TypeError(f"Got unknown type of fermistring: {type(fermistring)}")
        return self

    def __rmul__(self, number: float | int | complex) -> FermionicOperator:
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
    def operators_readable(self) -> dict[str, float | complex]:
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

    def get_qiskit_form(self, num_orbs: int) -> dict[str, float | complex]:
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
        operators: dict[tuple[tuple[int, bool], ...], float | complex] = {}
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

    def get_info(self) -> tuple[list[list[int]], list[list[int]], list[float | complex]]:
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
