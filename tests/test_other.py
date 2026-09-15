from slowquant.unitary_coupled_cluster.operators import a_op


def test_operator_dagger():
    """Test daggering of fermionic operators."""
    O = a_op(0, "alpha", False)
    assert O.operators_readable == {"a0": 1.0}
    assert (O.dagger).operators_readable == {"c0": 1.0}
    O = a_op(0, "alpha", False) * a_op(1, "alpha", False)
    assert O.operators_readable == {"a2a0": -1.0}
    assert (O.dagger).operators_readable == {"c2c0": 1.0}
    O = (
        a_op(0, "alpha", False)
        * a_op(2, "alpha", False)
        * a_op(3, "alpha", False)
        * a_op(4, "alpha", False)
        * a_op(5, "alpha", True)
        * a_op(6, "alpha", True)
    )
    assert O.operators_readable == {"c12c10a8a6a4a0": -1.0}
    assert (O.dagger).operators_readable == {"c8c6c4c0a12a10": 1.0}
    O = a_op(0, "alpha", False) * a_op(2, "alpha", False) * a_op(5, "alpha", True) * a_op(6, "alpha", True)
    assert O.operators_readable == {"c12c10a4a0": 1.0}
    assert (O.dagger).operators_readable == {"c4c0a12a10": 1.0}
