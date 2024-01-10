from __future__ import annotations

import copy


class a_op:
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


def operator_string_to_key(operator_string: list[a_op]) -> str:
    """Make key string to index a fermionic operator in a dict structure.

    Args:
        operator_string: Fermionic opreators.

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
        operator_string: Fermionic opreators.
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
) -> tuple[dict[str, float], dict[str, list[a_op]]]:
    """Reorder fermionic operator string.

    The string will be ordered such that all creation operators are first,
    and annihilation operators are second.
    Within a block of creation or annihilation operators the largest spin index
    will be first and the ordering will be descending.

    Returns:
        Reoreder operator dict and factor dict.
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
                    None
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
    def __init__(
        self, annihilation_operator: dict[str, list[a_op]] | a_op, factor: dict[str, float] | float
    ) -> None:
        if isinstance(annihilation_operator, dict) and not isinstance(factor, dict):
            raise ValueError(f"factor cannot be {type(factor)} when annihilation_operator is dict")
        if not isinstance(annihilation_operator, dict) and isinstance(factor, float):
            raise ValueError(f"factor cannot be dict when annihilation_operator is {type(a_op)}")
        if not isinstance(annihilation_operator, dict):
            string_key = operator_string_to_key([annihilation_operator])
            self.operators = {}
            self.operators[string_key] = [annihilation_operator]
            self.factors = {}
            self.factors[string_key] = factor
        if isinstance(annihilation_operator, dict) and isinstance(factor, dict):
            self.operators = annihilation_operator
            self.factors = factor

    def __add__(self, fermistring: FermionicOperator) -> FermionicOperator:
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
        operators = {}
        factors = {}
        for string_key1 in fermistring.operators.keys():
            for string_key2 in self.operators.keys():
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
                for str_key in new_ops.keys():
                    if str_key not in operators.keys():
                        operators[str_key] = new_ops[str_key]
                        factors[str_key] = new_facs[str_key]
                    else:
                        factors[str_key] += new_facs[str_key]
                        if abs(factors[str_key]) < 10**-14:
                            del factors[str_key]
                            del operators[str_key]
        return FermionicOperator(operators, factors)

    def __rmul__(self, number: float) -> FermionicOperator:
        operators = {}
        factors = {}
        for key_string in self.operators:
            operators[key_string] = self.operators[key_string]
            factors[key_string] = self.factors[key_string] * number
        return FermionicOperator(operators, factors)

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
            Fermionic opetaros on qiskit form.
        """
        qiskit_form = {}
        remapping = {}
        for i in range(2 * num_orbs):
            if i < num_orbs:
                remapping[i] = 2 * i
            else:
                remapping[i] = 2 * i + 1 - 2 * num_orbs
        for key_string in self.operators:
            qiskit_str = operator_to_qiskit_key(self.operators[key_string], remapping)
            qiskit_form[qiskit_str] = self.factors[key_string]
        return qiskit_form
