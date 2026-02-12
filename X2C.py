import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
from scipy.linalg import eig, lstsq

def test_x2c(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
    mol.build()

    # mf = scf.HF(mol)
    mf = scf.GHF(mol)
        
        
    mf.conv_tol_grad = 1e-10 #gradient tolerance form PYSCF
    mf.max_cycle = 10000

    # mf.scf()
    mf.kernel()
    coeff=np.array(mf.mo_coeff, dtype=complex)


    s = mol.intor('int1e_ovlp_spinor')      # overlap
    t = mol.intor('int1e_kin_spinor')       # kinetic energy
    v = mol.intor('int1e_nuc_spinor')       # nuclear attraction
    w = mol.intor('int1e_spnucsp_spinor')   # sigma.p V sigma.p  (W matrix)

    nao = len(coeff[0])
    #construct matrix
    D = np.zeros((2*nao, 2*nao), dtype=complex)   #2*nao since it is in spinor basis
    S=  np.zeros((2*nao, 2*nao), dtype=complex)   #overlap in spinor basis (RKB)
    
    #create Dirac Hamiltonian
    D[:nao, :nao] = v #Upper left
    D[:nao, nao:] = t #upper right
    D[nao:, :nao] = t.conj().T #lower left
    D[nao:, nao:] = w/(4*c**2)-t #lower right
    
    
    #create overlap matrix
    S[:nao, :nao] = s
    S[nao:, nao:] = t/(2*c**2)

    # print(D@D.conj().T)
    # print('new line')
    # print(S)

    #solve 4-component generalized eigenvalue problem D@C = e*S@C
    
    E, C = eig(D,S)
    print(C)
    #take only the positive part of the spectrum
    C_positive = []
    E_positive = []
    for i in range(2*nao):
        if E[i] > -2*c**2:            
            E_positive.append(E[i])


    positive_int = np.real(E) > -2 * c**2
    C_positive = C[:, positive_int]

    print(C_positive)

    if C_positive.shape[1] != nao:
        raise ValueError(f"Expected {nao} positive-energy solutions, "
                         f"got {C_positive.shape[1]}")


    # # Split into large (upper half) and small (lower half) component blocks
    CL = C_positive[:nao]     # large component:  shape (2n, 2n)
    CS = C_positive[nao:]     # small component:  shape (2n, 2n)


    #compute the X transformation X=CS@CL**-1
    
    X = lstsq(CL.T,CS.T)[0].T
    print(X)


        
    
    
    


def h2():
    geometry = """H  0.0   0.0  0.0;
        H  0.0  0.0  0.74"""
    basis = "631-g"
    active_space_u = ((1, 1), 4) #spin orbitaler
    # active_space = (2, 4)
    charge = 0
    spin = 0
    test_x2c(
        geometry=geometry, basis=basis, active_space=active_space_u, charge=charge, spin=spin, unit="angstrom"
    )


h2()