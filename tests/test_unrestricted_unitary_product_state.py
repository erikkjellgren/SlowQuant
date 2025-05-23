from pyscf import gto, scf

from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import (
    UnrestrictedWaveFunctionUPS,
)


def test_ups_oh() -> None:
    """Test unrestricted calculation for OH radical."""
    mol = gto.M(
        atom="O 0 0 0; H 0 0 1",
        basis="STO3G",
        spin=1,
        charge=0,
    )

    mf = scf.UHF(mol)
    mf.kernel()

    WF = UnrestrictedWaveFunctionUPS(
        mol.nelectron,
        ((2, 1), 3),
        mf.mo_coeff,
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        mol.intor("int2e"),
        ansatz="fuccsdt",
    )
    WF.run_wf_optimization_1step("SLSQP", True)
    assert abs(WF.energy_elec - -78.59886958623208) < 10**-8
