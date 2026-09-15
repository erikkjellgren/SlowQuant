from slowquant.unitary_coupled_cluster.operators import a_op

# Spin-orbital indices are alpha/beta blocked, so the alpha orbital of spatial index p is p.
NUM_ORBS = 7


def test_operator_dagger():
    """Test daggering of fermionic operators."""
    O = a_op(0, "alpha", False, NUM_ORBS)
    assert O.operators_readable == {"a0": 1.0}
    assert (O.dagger).operators_readable == {"c0": 1.0}
    O = a_op(0, "alpha", False, NUM_ORBS) * a_op(1, "alpha", False, NUM_ORBS)
    assert O.operators_readable == {"a1a0": -1.0}
    assert (O.dagger).operators_readable == {"c1c0": 1.0}
    O = (
        a_op(0, "alpha", False, NUM_ORBS)
        * a_op(2, "alpha", False, NUM_ORBS)
        * a_op(3, "alpha", False, NUM_ORBS)
        * a_op(4, "alpha", False, NUM_ORBS)
        * a_op(5, "alpha", True, NUM_ORBS)
        * a_op(6, "alpha", True, NUM_ORBS)
    )
    assert O.operators_readable == {"c6c5a4a3a2a0": -1.0}
    assert (O.dagger).operators_readable == {"c4c3c2c0a6a5": 1.0}
    O = (
        a_op(0, "alpha", False, NUM_ORBS)
        * a_op(2, "alpha", False, NUM_ORBS)
        * a_op(5, "alpha", True, NUM_ORBS)
        * a_op(6, "alpha", True, NUM_ORBS)
    )
    assert O.operators_readable == {"c6c5a2a0": 1.0}
    assert (O.dagger).operators_readable == {"c2c0a6a5": 1.0}
