import slowquant.SlowQuant as sq
import pyscf
from pyscf import scf
import numpy as np

import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.linear_response import generalized_naive
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS


def test_naivelr_H2_STO3g():
    """
    Test of generalized excitation energies for naive LR with H2((1,1),4)/STO-3G
    """

    mol = pyscf.M(atom="H 0 0 0; H 0.0  0.0  0.74", basis="STO-3g", unit="angstrom", spin=0)
    mol.build()
    mf = scf.GHF(mol)
    mf.conv_tol_grad = 1e-10 #gradient tolerance form PYSCF
    mf.max_cycle = 50000
    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)
    WF =GeneralizedWaveFunctionUPS(
        ((1, 1), 4),
        coeff,
        mol,
        "fUCCSDT",
        False, #Do x2c
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    # Linear Response
    LR = generalized_naive.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies()
    assert abs(LR.excitation_energies[0] - 0.60651048) < 10**-4
    assert abs(LR.excitation_energies[1] - 0.60651048) < 10**-4
    assert abs(LR.excitation_energies[2] - 0.60651048) < 10**-4
    assert abs(LR.excitation_energies[3] - 0.9689314) < 10**-4
    assert abs(LR.excitation_energies[4] - 1.62042651) < 10**-4

def test_naivelr_H2_631g():
    """
    Test of generalized excitation energies for naive LR with H2((1,1),4)/631-G
    """

    mol = pyscf.M(atom="H 0 0 0; H 0.0  0.0  0.74", basis="631-g", unit="angstrom", spin=0)
    mol.build()
    mf = scf.GHF(mol)


    mf.conv_tol_grad = 1e-10 #gradient tolerance form PYSCF
    mf.max_cycle = 50000
    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)


    # OO-UCCSD
    WF =GeneralizedWaveFunctionUPS(
        ((1, 1), 4),
        coeff,
        mol,
        "fUCCSD",
        False, #Do x2c
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )

    # Optimize wavefunction
    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)


    # Linear Response
    LR = generalized_naive.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies()
    assert abs(LR.excitation_energies[0] - 0.39432446) < 10**-4
    assert abs(LR.excitation_energies[1] - 0.39432446) < 10**-4
    assert abs(LR.excitation_energies[2] - 0.39432446) < 10**-4
    assert abs(LR.excitation_energies[3] - 0.57441343) < 10**-4
    assert abs(LR.excitation_energies[4] - 0.84842793) < 10**-4
    assert abs(LR.excitation_energies[5] - 0.84842793) < 10**-4
    assert abs(LR.excitation_energies[6] - 0.84842793) < 10**-4
    assert abs(LR.excitation_energies[7] - 1.04317705) < 10**-4
    assert abs(LR.excitation_energies[8] - 1.13948175) < 10**-4
    assert abs(LR.excitation_energies[9] - 1.33548038) < 10**-4
    assert abs(LR.excitation_energies[10] - 1.33548038) < 10**-4
    assert abs(LR.excitation_energies[11] - 1.33548038) < 10**-4
    assert abs(LR.excitation_energies[12] -1.3659622) < 10**-4
    assert abs(LR.excitation_energies[13] - 1.44184491) < 10**-4
    assert abs(LR.excitation_energies[14] - 1.44184491) < 10**-4
    assert abs(LR.excitation_energies[15] - 1.44184491) < 10**-4
    assert abs(LR.excitation_energies[16] - 1.8311974) < 10**-4
    assert abs(LR.excitation_energies[17] - 1.88481546) < 10**-4
    assert abs(LR.excitation_energies[18] - 1.88481546) < 10**-4
    assert abs(LR.excitation_energies[19] - 1.88481546) < 10**-4
    assert abs(LR.excitation_energies[20] - 2.58127848) < 10**-4




def test_naivelr_H4_STO3g():
    """
    Test of generalized excitation energies for naive LR with H4((1,1),4)/STO3G
    """

    mol = pyscf.M(atom="""H 0.0 0.0 0.0;
                  H 0.0 0.0 0.74;
                  H 0.0 1.11 0.74;
                  H 0.0 1.11 0.0;""" , basis="sto-3g", unit="angstrom", spin=0)
    mol.build()
    mf = scf.GHF(mol)
    mf.conv_tol_grad = 1e-10 #gradient tolerance form PYSCF
    mf.max_cycle = 50000
    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)


    # OO-UCCSD
    WF =GeneralizedWaveFunctionUPS(
        ((1, 1), 4),
        coeff,
        mol,
        "fUCCSD",
        False, #Do x2c
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )

    # Optimize wavefunction
    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)


    # Linear Response
    LR = generalized_naive.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies()
    assert abs(LR.excitation_energies[4] - 2.34085080e-01) < 10**-4
    assert abs(LR.excitation_energies[5] - 2.34085080e-01) < 10**-4
    assert abs(LR.excitation_energies[6] - 2.34085080e-01) < 10**-4
    assert abs(LR.excitation_energies[7] - 4.28186413e-01) < 10**-4
    assert abs(LR.excitation_energies[8] - 5.51556420e-01) < 10**-4
    assert abs(LR.excitation_energies[9] - 5.51556420e-01) < 10**-4
    assert abs(LR.excitation_energies[10] - 5.51556420e-01) < 10**-4
    assert abs(LR.excitation_energies[11] - 7.57117763e-01) < 10**-4
    assert abs(LR.excitation_energies[12] - 7.83801830e-01) < 10**-4
    assert abs(LR.excitation_energies[13] - 7.83801830e-01) < 10**-4
    assert abs(LR.excitation_energies[14] - 7.83801830e-01) < 10**-4
    assert abs(LR.excitation_energies[15] - 8.74888889e-01) < 10**-4
    assert abs(LR.excitation_energies[16] - 9.26202641e-01) < 10**-4
    assert abs(LR.excitation_energies[17] - 9.26202641e-01) < 10**-4
    assert abs(LR.excitation_energies[18] - 9.26202641e-01) < 10**-4
    assert abs(LR.excitation_energies[19] - 1.08260105) < 10**-4
    assert abs(LR.excitation_energies[20] - 1.18490185) < 10**-4
    assert abs(LR.excitation_energies[21] - 1.25507083) < 10**-4
    assert abs(LR.excitation_energies[22] - 1.25507083) < 10**-4
    assert abs(LR.excitation_energies[23] - 1.25507083) < 10**-4
    assert abs(LR.excitation_energies[24] - 1.38827432) < 10**-4





def test_naivelr_H4_STO3g_full():
    """
    Test of generalized excitation energies for naive LR with H4((2,2),8)/STO3G
    """

    mol = pyscf.M(atom="""H 0.0 0.0 0.0;
                  H 0.0 0.0 0.74;
                  H 0.0 1.11 0.74;
                  H 0.0 1.11 0.0;""" , basis="sto-3g", unit="angstrom", spin=0)
    mol.build()
    mf = scf.GHF(mol)
    mf.conv_tol_grad = 1e-10 #gradient tolerance form PYSCF
    mf.max_cycle = 50000
    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)


    # OO-UCCSD
    WF =GeneralizedWaveFunctionUPS(
        ((2, 2), 8),
        coeff,
        mol,
        "fUCCSD",
        False, #Do x2c
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )

    # Optimize wavefunction
    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)


    # Linear Response
    LR = generalized_naive.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies()
    assert abs(LR.excitation_energies[0] - 0.2854956) < 10**-4
    assert abs(LR.excitation_energies[1] - 0.2854956) < 10**-4
    assert abs(LR.excitation_energies[2] - 0.2854956) < 10**-4
    assert abs(LR.excitation_energies[3] - 0.47095693) < 10**-4
    assert abs(LR.excitation_energies[4] - 0.60274993) < 10**-4
    assert abs(LR.excitation_energies[5] - 0.60274995) < 10**-4
    assert abs(LR.excitation_energies[6] - 0.60274995) < 10**-4
    assert abs(LR.excitation_energies[7] - 0.69340677) < 10**-4
    assert abs(LR.excitation_energies[8] - 0.91991757) < 10**-4
    assert abs(LR.excitation_energies[9] - 0.93709366) < 10**-4
    assert abs(LR.excitation_energies[10] - 0.93709366) < 10**-4
    assert abs(LR.excitation_energies[11] - 0.93709366) < 10**-4
    assert abs(LR.excitation_energies[12] - 0.97050017) < 10**-4
    assert abs(LR.excitation_energies[13] - 0.97050017) < 10**-4
    assert abs(LR.excitation_energies[14] - 0.97050027) < 10**-4
    assert abs(LR.excitation_energies[15] - 1.12156432) < 10**-4
    assert abs(LR.excitation_energies[16] - 1.2348761) < 10**-4
    assert abs(LR.excitation_energies[17] - 1.23792252) < 10**-4
    assert abs(LR.excitation_energies[18] - 1.23792252) < 10**-4
    assert abs(LR.excitation_energies[19] - 1.23792252) < 10**-4
    assert abs(LR.excitation_energies[20] - 1.23792252) < 10**-4
    assert abs(LR.excitation_energies[21] - 1.23792252) < 10**-4
    assert abs(LR.excitation_energies[22] - 1.25548344) < 10**-4
    assert abs(LR.excitation_energies[23] - 1.25548344) < 10**-4
    assert abs(LR.excitation_energies[24] - 1.25548344) < 10**-4
    assert abs(LR.excitation_energies[25] - 1.30409161) < 10**-4
    assert abs(LR.excitation_energies[26] - 1.30409161) < 10**-4
    assert abs(LR.excitation_energies[27] - 1.30409161) < 10**-4
    assert abs(LR.excitation_energies[28] - 1.42624618) < 10**-4
    assert abs(LR.excitation_energies[29] - 1.43499708) < 10**-4
    assert abs(LR.excitation_energies[30] - 1.53676515) < 10**-4
    assert abs(LR.excitation_energies[31] - 1.63324821) < 10**-4
    assert abs(LR.excitation_energies[32] - 1.63324821) < 10**-4
    assert abs(LR.excitation_energies[33] - 1.63324821) < 10**-4
    assert abs(LR.excitation_energies[34] - 1.71964207) < 10**-4
    assert abs(LR.excitation_energies[35] - 1.71964207) < 10**-4
    assert abs(LR.excitation_energies[36] - 1.71964207) < 10**-4
    assert abs(LR.excitation_energies[37] - 1.8163781 ) < 10**-4
    assert abs(LR.excitation_energies[38] -  1.8163781) < 10**-4
    assert abs(LR.excitation_energies[39] -  1.8163781) < 10**-4
    assert abs(LR.excitation_energies[40] - 1.81786397) < 10**-4
    assert abs(LR.excitation_energies[41] - 2.00692655) < 10**-4
    assert abs(LR.excitation_energies[42] - 2.00692655) < 10**-4
    assert abs(LR.excitation_energies[43] - 2.00692655) < 10**-4
    assert abs(LR.excitation_energies[44] - 2.01106057) < 10**-4
    assert abs(LR.excitation_energies[45] - 2.22626576) < 10**-4
    assert abs(LR.excitation_energies[46] - 2.22834703) < 10**-4
    assert abs(LR.excitation_energies[47] - 2.25176684) < 10**-4
    assert abs(LR.excitation_energies[48] - 2.25176684) < 10**-4
    assert abs(LR.excitation_energies[49] - 2.25176684) < 10**-4
    assert abs(LR.excitation_energies[50] - 2.77902678) < 10**-4
    assert abs(LR.excitation_energies[51] - 2.91336178) < 10**-4



def test_naivelr_H3_STO3g():
    """
    Test of generalized excitation energies for naive LR with H3((2,1),6)/STO-3G
    """

    mol = pyscf.M(atom="""H  0.000000   0.000000       0.000000;
                  H  1.000000   0.000000       0.000000;
                  H  0.500000   0.8660254038   0.000000""", basis="STO-3g", unit="angstrom", spin=1)
    mol.build()
    mf = scf.GHF(mol)
    mf.conv_tol_grad = 1e-10 #gradient tolerance form PYSCF
    mf.max_cycle = 50000
    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)
    WF =GeneralizedWaveFunctionUPS(
        ((2, 1), 6),
        coeff,
        mol,
        "fUCCSDT",
        False, #Do x2c
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )
    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    # Linear Response
    LR = generalized_naive.LinearResponse(WF, excitations="SD")
    LR.calc_excitation_energies()
    print()
    assert abs(LR.excitation_energies[3] - 3.90974121e-01) < 10**-4
    assert abs(LR.excitation_energies[4] - 3.90974121e-01) < 10**-4
    assert abs(LR.excitation_energies[5] - 3.90974121e-01) < 10**-4
    assert abs(LR.excitation_energies[6] - 3.90974121e-01) < 10**-4
    assert abs(LR.excitation_energies[7] -  6.98556873e-01) < 10**-4
    assert abs(LR.excitation_energies[8] -  6.98556873e-01) < 10**-4
    assert abs(LR.excitation_energies[9] -  6.98556873e-01) < 10**-4
    assert abs(LR.excitation_energies[10] -  7.15748171e-01) < 10**-4
    assert abs(LR.excitation_energies[11] -  8.23203906e-01) < 10**-4
    assert abs(LR.excitation_energies[12] -  8.23203906e-01) < 10**-4
    assert abs(LR.excitation_energies[13] -  8.38594532e-01) < 10**-4
    assert abs(LR.excitation_energies[14] -  8.67164806e-01) < 10**-4
    assert abs(LR.excitation_energies[15] -  1.37841350e+00) < 10**-4
    assert abs(LR.excitation_energies[16] -  1.37841351) < 10**-4
    assert abs(LR.excitation_energies[17] -  1.3784135) < 10**-4



test_naivelr_H2_631g()
test_naivelr_H2_STO3g()
test_naivelr_H4_STO3g()
test_naivelr_H4_STO3g_full()
test_naivelr_H3_STO3g()