import numpy as np
import pyscf
from pyscf import mcscf, scf
from pyscf.data import nist

from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS


def get_hcf_fc_unrestricted(geometry, basis, active_space, unit="bohr", charge=0, spin=0):
    """Calculate hyperfine coupling constant (fermi-contact term) for a molecule"""
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()

    mc = mcscf.UCASCI(mf, active_space[1], active_space[0])

    res = mc.kernel(mf.mo_coeff)

    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g_eri = mol.intor("int2e")

    # Slowquant
    WF = UnrestrictedWaveFunctionUPS(
        mol.nelectron,
        active_space,
        mf.mo_coeff,
        h_core,
        g_eri,
        "fuccsd",
        {"n_layers": 1},
        include_active_kappa=True,
    )

    # WF.run_wf_optimization_1step("SLSQP", True)

    # FC
    for atom in mol._atom:
        print(atom)
        amp_basis = mol.eval_gto("GTOval_sph", coords=[atom[1]])[0]
        # old version of h1ao
        # h1ao = np.array([np.outer(np.conj(amp_basis), amp_basis)]) * nist.G_ELECTRON * 2/3 * np.pi
        h1ao = np.outer(np.conj(amp_basis), amp_basis) * nist.G_ELECTRON * 2 / 3 * np.pi
        # hfc here whould be in atomic units
        print((WF.num_inactive_orbs), WF.num_virtual_orbs, WF.num_active_orbs)
        print(WF.rdm1aa - WF.rdm1bb)

        print(
            h1ao[:, (WF.num_inactive_orbs - 1) : (len(amp_basis) + WF.num_virtual_orbs - 1)][
                (WF.num_inactive_orbs - 1) : (len(amp_basis) + WF.num_virtual_orbs - 1)
            ]
        )

        hfc = np.matmul(
            h1ao[:, (WF.num_inactive_orbs - 1) : (len(amp_basis) + WF.num_virtual_orbs - 1)][
                (WF.num_inactive_orbs - 1) : (len(amp_basis) + WF.num_virtual_orbs - 1)
            ],
            (WF.rdm1aa - WF.rdm1bb),
        )
        print(hfc)
    # print(WF.num_elec)


def test_OH_hfc():
    """Test of hfc for OH using unrestricted RDMs"""
    geometry = """O  0.0   0.0  0.0;
        H  0.0  0.0  0.9697;"""
    basis = "STO-3G"
    active_space = ((1, 2), 3)
    charge = 0
    # the pyscf spin parameter is the value of 2S (tne number of unpaired electrons, or the difference between the number of alpha and beta electrons)
    spin = 1

    get_hcf_fc_unrestricted(
        geometry=geometry, basis=basis, active_space=active_space, unit="angstrom", charge=charge, spin=spin
    )


test_OH_hfc()
