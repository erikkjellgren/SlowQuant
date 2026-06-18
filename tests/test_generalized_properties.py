import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
from scipy.stats import unitary_group
from scipy.linalg import solve
from pyscf.x2c import sfx2c1e
from pyscf import cc
import scipy.linalg

# from pyscf.x2c.x2c import dip_moment

# from slowquant.unitary_coupled_cluster.unrestricted_ups_wavefunction import UnrestrictedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
from slowquant.unitary_coupled_cluster.linear_response import generalized_naive, naive
from slowquant.unitary_coupled_cluster.generalized_operator_state_algebra import generalized_expectation_value, generalized_propagate_state
from slowquant.unitary_coupled_cluster.generalized_operators import generalized_hamiltonian_full_space, generalized_hamiltonian_0i_0a, generalized_hamiltonian_1i_1a, generalized_one_elec_op_0i_0a
from slowquant.unitary_coupled_cluster.operators import a_op_spin
from slowquant.molecularintegrals.integralfunctions import generalized_one_electron_transform

def test_efg_dipmom_PC_HF():
    """.........."""
    from pyscf.gto.basis import load
    import pyscf.gto as gto
    mol = gto.M(
                atom = """H  0.0  0.0  0.91680 ;
                  F  0.000000   0.000000       0.000000""",
                basis = {'H': gto.uncontract(load('../x2c-SVPall.nw', 'H')),
                'F': gto.uncontract(load('../x2c-SVPall.nw', 'F'))},
                charge = 0,
                spin = 0,
    )


    c=137.036
    mf = scf.GHF(mol).x2c()

    mf.conv_tol_grad = 1e-10 #gradient tolerance form PYSCF

    mf.max_cycle = 50000
    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)
    
    WF =GeneralizedWaveFunctionUPS(
        ((1,1), 2),
        coeff,
        mol,
        "fUCCSD",
        True, #Do x2c
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=True,
    )

    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    dip_ao = build_x2c_pc_operator(mf, mol, "int1e_r", 'int1e_sprsp', c, x2c=True, picture_change=True, spin_free=False)

    "Calculate dipole moments"
    mux = generalized_one_electron_transform(WF.c_mo, dip_ao[0], x2c=True) 
    muy = generalized_one_electron_transform(WF.c_mo, dip_ao[1], x2c=True)
    muz = generalized_one_electron_transform(WF.c_mo, dip_ao[2], x2c=True)
    mu_op_x = generalized_one_elec_op_0i_0a(mux, WF.num_inactive_spin_orbs,WF.num_active_spin_orbs,)
    mu_op_y = generalized_one_elec_op_0i_0a(muy, WF.num_inactive_spin_orbs,WF.num_active_spin_orbs,)
    mu_op_z = generalized_one_elec_op_0i_0a(muz, WF.num_inactive_spin_orbs,WF.num_active_spin_orbs,)
    dip_x=generalized_expectation_value(WF.ci_coeffs, [mu_op_x], WF.ci_coeffs, WF.ci_info)
    dip_y=generalized_expectation_value(WF.ci_coeffs, [mu_op_y], WF.ci_coeffs, WF.ci_info)
    dip_z=generalized_expectation_value(WF.ci_coeffs, [mu_op_z], WF.ci_coeffs, WF.ci_info)


    charges = mol.atom_charges()
    coords = mol.atom_coords()
    nuclear_dipole = np.einsum('i,ij->j', charges, coords)

    print(-dip_z+nuclear_dipole[2])

    assert abs(-dip_z+nuclear_dipole[2] - (0.8380+0.0000*1j)) < 10**-4



    "Electric field gradients"
    coords = mol.atom_coords()
    charges = mol.atom_charges()

    for A in range(mol.natm):
        int_pc = build_x2c_pc_operator_efg(mf, mol, A, c, x2c=True, picture_change=True, spin_free=False)  # (3, 3, 2*nao_c, 2*nao_c)

        efg_elec = np.zeros((3, 3)) #create the EFG matrix
        for alpha in range(3):
            for beta in range(3):
                mo = generalized_one_electron_transform(WF.c_mo, int_pc[alpha, beta], x2c=True) #must be false for sf...
                op = generalized_one_elec_op_0i_0a(mo, WF.num_inactive_spin_orbs, WF.num_active_spin_orbs)
                efg_elec[alpha, beta] = generalized_expectation_value(
                    WF.ci_coeffs, [op], WF.ci_coeffs, WF.ci_info
                )

        # Make traceless
        trace = np.trace(efg_elec) / 3
        for alpha in range(3):
            efg_elec[alpha, alpha] -= trace

        efg_elec *= -1  # electrons charge -1

        #Nuclear part
        efg_nuc = np.zeros((3, 3))
        for B in range(mol.natm):
            if B == A:
                continue
            R_AB = coords[B] - coords[A] #A os expansion point
            r = np.linalg.norm(R_AB)
            for alpha in range(3):
                for beta in range(3):
                    efg_nuc[alpha, beta] += charges[B] * (3 * R_AB[alpha] * R_AB[beta]/r**5 - (alpha == beta) / r**3) 


        #Total EFG
        efg_total = efg_elec + efg_nuc
        print('Electric part of', efg_elec)
        print('Nuclear part of', efg_nuc)
        print(f"EFG at atom {A} ({mol.atom_symbol(A)}):")
        print(f"  xx={efg_total[0,0]:.4f}  xy={efg_total[0,1]:.4f}  xz={efg_total[0,2]:.4f}")
        print(f"  yy={efg_total[1,1]:.4f}  yz={efg_total[1,2]:.4f}")
        print(f"  zz={efg_total[2,2]:.4f}")
        print(f"  Trace: {np.trace(efg_total):.2e}")
        print(f"  Symmetric: {np.allclose(efg_total, efg_total.T)}")


    assert abs(efg_total[2,2]-3.1185) < 10**-4

"Calculate Properties"
def block_diagonal_matrix(mat):
    return scipy.linalg.block_diag(mat, mat)



def _sigma_dot2(prp4: np.ndarray) -> np.ndarray:
    qx, qy, qz, q0 = prp4
    return np.block([
        [q0 + 1j * qz,     qy + 1j * qx],
        [-qy + 1j * qx,   q0 - 1j * qz     ]
    ])

def build_x2c_pc_operator(mf, mol, int_LL, int_SS, c, x2c=True, picture_change=True, spin_free=True): 
        if x2c==False:
            print('NO X2C')
            return mol.intor_symmetric(int_LL)    
        else:
            if picture_change:
                if spin_free:
                    print('Picture change SF operator true')
                    xmol = mf.with_x2c.get_xmol()[0]
                    nao = xmol.nao
                    r = xmol.intor_symmetric(int_LL)                          # (3, nao_x, nao_x)
                    c1 = 0.5 / c
                    sprsp = xmol.intor_symmetric(int_SS).reshape(3, 4, nao, nao)
                    sprsp_sf = sprsp[:, 3] * (c1**2)
                    return mf.with_x2c.picture_change((r, sprsp_sf)) 
                else:
                    print('picture change true')
                    xmol = mf.with_x2c.get_xmol()[0]
                    nao = xmol.nao
                    r = xmol.intor_symmetric(int_LL)                          # (3, nao_x, nao_x)
                    r_so = np.array([block_diagonal_matrix(x) for x in r])   # (3, 2*nao_x, 2*nao_x)
                    c1 = 0.5 / c
                    sprsp = xmol.intor_symmetric(int_SS).reshape(3, 4, nao, nao)
                    sprsp_so = np.array([_sigma_dot2(x * c1**2) for x in sprsp])
                    print("int1e_ipsprinvspip shape:", xmol.intor("int1e_ipsprinvspip").shape)
                    return mf.with_x2c.picture_change((r_so, sprsp_so))       # (3, 2*nao_c, 2*nao_c) 
            else:
                print('picture change false')
                nao_c = mol.nao
                r = mol.intor_symmetric(int_LL)                           # (3, nao_c, nao_c)
                r_so = np.array([block_diagonal_matrix(x) for x in r])   # (3, 2*nao_c, 2*nao_c) 
            return r_so


def build_x2c_pc_operator_efg(mf, mol, atom_idx, c, x2c=True, picture_change=True, spin_free=False):
    with mol.with_rinv_origin(mol.atom_coord(atom_idx)):
        nao_c = mol.nao
        if x2c==False:
            print('NO X2C')
            return  (mol.intor("int1e_ipiprinv") + mol.intor("int1e_ipiprinv").transpose(0, 2, 1)+ 2 * mol.intor("int1e_iprinvip")).reshape(3, 3, nao_c, nao_c)

        else:
            xmol = mf.with_x2c.get_xmol()[0]
            nao_x = xmol.nao
            c1 = 0.5 / c

            if picture_change:
                if spin_free:
                    print("Picture change is True for EFG SF")
                    efg_ao = (
                        xmol.intor("int1e_ipiprinv")
                        + xmol.intor("int1e_ipiprinv").transpose(0, 2, 1)
                        + 2 * xmol.intor("int1e_iprinvip")
                    )  # (9, nao_x, nao_x)

                    efg_ao_ss = (
                        xmol.intor("int1e_ipipsprinvsp")
                        + xmol.intor("int1e_ipipsprinvsp").transpose(0, 2, 1)
                        + 2 * xmol.intor("int1e_ipsprinvspip")).reshape(9, 4, nao_x, nao_x)
                    
                    efg_ao_ss_sf = efg_ao_ss[:, 3] * (c1**2) # This correctly becomes (9, nao_x, nao_x)

                        
                    ao_efg = mf.with_x2c.picture_change((efg_ao, efg_ao_ss_sf )) 
                    ao_efg = 0.5 * (ao_efg + ao_efg.conj().transpose(0, 2, 1)) #enforcing hermicity
                    ao_efg = ao_efg.reshape(3, 3, nao_c, nao_c) #x2c sf
                    ao_efg = 0.5 * (ao_efg + ao_efg.transpose(1, 0, 2, 3)) #makes tensor symmetric

                else:
                    print("Picture change is True for EFG")
                    efg_ao = (
                        xmol.intor("int1e_ipiprinv")
                        + xmol.intor("int1e_ipiprinv").transpose(0, 2, 1)
                        + 2 * xmol.intor("int1e_iprinvip")
                    )  # (9, nao_x, nao_x)
                    f2_LL_spinor = np.array([block_diagonal_matrix(x) for x in efg_ao])  # (9, 2*nao_x, 2*nao_x)
                    # f2_SS = xmol.intor("int1e_ipsprinvspip").reshape(9, 4, nao_x, nao_x)

                    efg_ao_ss = (
                        xmol.intor("int1e_ipipsprinvsp")
                        + xmol.intor("int1e_ipipsprinvsp").transpose(0, 2, 1)
                        + 2 * xmol.intor("int1e_ipsprinvspip")).reshape(9, 4, nao_x, nao_x)
                        
                    f2_SS_spinor = np.array([_sigma_dot2(x) * (0.5/c)**2 for x in efg_ao_ss])
                    ao_efg = mf.with_x2c.picture_change((f2_LL_spinor, f2_SS_spinor))  #det er her den går galt
                    ao_efg = 0.5 * (ao_efg + ao_efg.conj().transpose(0, 2, 1))
                    ao_efg = ao_efg.reshape(3, 3, 2 * nao_c, 2 * nao_c)
                    ao_efg = 0.5 * (ao_efg + ao_efg.transpose(1, 0, 2, 3))

            else:
                print('Picture change False for EFG')
                efg_ao = (
                    mol.intor("int1e_ipiprinv")                       
                    + mol.intor("int1e_ipiprinv").transpose(0, 2, 1)
                    + 2 * mol.intor("int1e_iprinvip")
                )  # (9, nao_c, nao_c)
                ao_efg = np.array([block_diagonal_matrix(x) for x in efg_ao])  # (9, 2*nao_c, 2*nao_c)
                ao_efg = 0.5 * (ao_efg + ao_efg.conj().transpose(0, 2, 1))
                ao_efg = ao_efg.reshape(3, 3, 2 * nao_c, 2 * nao_c)
                ao_efg = 0.5 * (ao_efg + ao_efg.transpose(1, 0, 2, 3))

        return ao_efg



def test_polarizability_H2():
    """Test polarizability for H2."""
    mol = gto.M(
                atom = """H  0.0   0.0  0.0;
        H  0.0  0.0  0.74""",
                basis = "631-g",
                charge = 0,
                spin = 0,
    )

    mf = scf.GHF(mol)
    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)

    WF =GeneralizedWaveFunctionUPS(
            ((1, 1), 4),
            coeff,
            mol,
            "fUCCSD",
            False, #Do x2c
            {"n_layers": 1, "is_spin_conserving" : False},
            include_active_kappa=True,
        )
    WF.run_wf_optimization_1step("l-bfgs-b", orbital_optimization=True, tol=1e-10, maxiter = 2000)

    LR = generalized_naive.LinearResponse(WF, excitations="sd")
    LR.calc_excitation_energies()
    print(LR.excitation_energies)

    # "Calculate polarizability"
    prop_grad = LR.get_property_gradient( mol.intor_symmetric("int1e_r")) 
    response = solve(LR.hessian, prop_grad) 
    alpha = np.einsum('ix,ix->x', prop_grad.conj(), response)


    assert abs(alpha[2] - (5.8506+0.0000*1j)) < 10**-8




test_efg_dipmom_PC_HF()
test_polarizability_H2()

