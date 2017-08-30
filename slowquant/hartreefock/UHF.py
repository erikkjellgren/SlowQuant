import numpy as np
import scipy.linalg

def diagonlize(M):
    eigVal, eigVec = np.linalg.eigh(M)
    return eigVal, eigVec

def symm_orth(eigVal, eigVec):
    diaVal = np.diag(eigVal)
    M = np.dot(np.dot(eigVec,scipy.linalg.sqrtm(np.linalg.inv(np.diag(eigVal)))),np.matrix.transpose(eigVec))
    return M

def UnrestrictedHartreeFock(input, VNN, Te, S, VeN, Vee, deTHR=10**-6,rmsTHR=10**-6,Maxiter=100, UHF_mix=0.15, print_SCF='Yes'):
    # ###############################
    #
    # VNN = nuclear repulsion
    # Te  = kinetic energy
    # S   = overlap integrals
    # VeN = nuclear attraction
    # Vee = two electron integrals
    # 
    # ###############################
    
    #Core Hamiltonian
    Hcore = VeN+Te
    
    #Diagonalizing overlap matrix
    Lambda_S, L_S = diagonlize(S)
    #Symmetric orthogonal inverse overlap matrix
    S_sqrt = symm_orth(Lambda_S, L_S)
    
    #Assign number of beta and alpha
    Nalpha = int((input[0,0]+1)/2)
    Nbeta  = int(input[0,0]/2)

    #Initial Density
    F0prime_alpha = np.dot(np.dot(np.matrix.transpose(S_sqrt),Hcore),np.matrix.transpose(S_sqrt))
    eps0_alpha, C0prime_alpha = diagonlize(F0prime_alpha)
    F0prime_beta = np.dot(np.dot(np.matrix.transpose(S_sqrt),Hcore),np.matrix.transpose(S_sqrt))
    eps0_beta, C0prime_beta = diagonlize(F0prime_beta)
    
    C0_alpha = np.matrix.transpose(np.dot(S_sqrt, C0prime_alpha))
    C0_beta = np.matrix.transpose(np.dot(S_sqrt, C0prime_beta))
    
    k = UHF_mix
    if Nalpha == Nbeta:
        ColdHOMO = C0_alpha[Nalpha-1]
        ColdLUMO = C0_alpha[Nalpha]
        C0_alpha[Nalpha-1] = 1/((1+k**2))**0.5*(ColdHOMO+k*ColdLUMO)
        C0_alpha[Nalpha]   = 1/((1+k**2))**0.5*(-k*ColdHOMO+ColdLUMO)
    
    #Only using occupied MOs
    C0_alpha = C0_alpha[0:Nalpha]
    C0T_alpha = np.matrix.transpose(C0_alpha)
    D0_alpha = np.dot(C0T_alpha, C0_alpha) 
    C0_beta = C0_beta[0:Nbeta]
    C0T_beta = np.matrix.transpose(C0_beta)
    D0_beta = np.dot(C0T_beta, C0_beta)        
    
    # Initial Energy
    E0el = 0
    for i in range(0, len(D0_alpha)):
        for j in range(0, len(D0_alpha[0])):
            E0el += 0.5*D0_alpha[i,j]*(Hcore[i,j]+Hcore[i,j])
    for i in range(0, len(D0_beta)):
        for j in range(0, len(D0_beta[0])):
            E0el += 0.5*D0_beta[i,j]*(Hcore[i,j]+Hcore[i,j])
    
    #SCF iterations
    if print_SCF == 'Yes':
        output = open('out.txt', 'a')
        output.write('Iter')
        output.write("\t")
        output.write('Eel')
        output.write("\t \t \t \t \t")
        output.write('Etot')
        output.write("\t \t \t \t")
        output.write('dE')
        output.write("\t \t \t \t \t")
        output.write('rmsD_alpha')
        output.write("\t \t \t \t \t")
        output.write('rmsD_beta')
        output.write("\n")
        output.write('0')
        output.write("\t \t")
        output.write("{:14.10f}".format(E0el))
        output.write("\t \t")
        output.write("{:14.10f}".format(E0el+VNN[0]))
    
    for iter in range(1, Maxiter):
        if print_SCF == 'Yes':
            output.write("\n")
            
        #New Fock Matrix
        Part_alpha = np.zeros((len(S),len(S)))
        Part_alpha_beta = np.zeros((len(S),len(S)))
        Part_beta = np.zeros((len(S),len(S)))
        Part_beta_alpha = np.zeros((len(S),len(S)))
        for mu in range(0, len(S)):
            for nu in range(0, len(S)):
                for lam in range(0, len(S)):
                    for sig in range(0, len(S)):
                        Part_alpha[mu,nu] += D0_alpha[lam,sig]*(Vee[mu,nu,lam,sig]-Vee[mu,lam,nu,sig])
                        Part_alpha_beta[mu,nu] += D0_beta[lam,sig]*Vee[mu,nu,lam,sig]
                        Part_beta[mu,nu] += D0_beta[lam,sig]*(Vee[mu,nu,lam,sig]-Vee[mu,lam,nu,sig])
                        Part_beta_alpha[mu,nu] += D0_alpha[lam,sig]*Vee[mu,nu,lam,sig]
        
        
        F_alpha = Hcore + Part_alpha + Part_alpha_beta
        F_beta = Hcore + Part_beta + Part_beta_alpha
        
            
        Fprime_alpha = np.dot(np.dot(np.transpose(S_sqrt),F_alpha),S_sqrt)
        eps_alpha, Cprime_alpha = diagonlize(Fprime_alpha)
        C_alpha = np.dot(S_sqrt, Cprime_alpha)
        CT_alpha    = np.matrix.transpose(C_alpha)
        CTocc_alpha = CT_alpha[0:Nalpha]
        Cocc_alpha  = np.matrix.transpose(CTocc_alpha)
        D_alpha = np.dot(Cocc_alpha, CTocc_alpha)
        Fprime_beta = np.dot(np.dot(np.transpose(S_sqrt),F_beta),S_sqrt)
        eps_beta, Cprime_beta = diagonlize(Fprime_beta)
        C_beta = np.dot(S_sqrt, Cprime_beta)
        CT_beta    = np.matrix.transpose(C_beta)
        CTocc_beta = CT_beta[0:Nbeta]
        Cocc_beta  = np.matrix.transpose(CTocc_beta)
        D_beta = np.dot(Cocc_beta, CTocc_beta)
        
        #New SCF Energy
        Eel = 0
        for i in range(0, len(D_alpha)):
            for j in range(0, len(D_alpha[0])):
                Eel += 0.5*D_alpha[i,j]*(Hcore[i,j]+F_alpha[i,j])
                Eel += 0.5*D_beta[i,j]*(Hcore[i,j]+F_beta[i,j])

        #Convergance
        dE = Eel - E0el
        rmsD_alpha = 0
        for i in range(0, len(D0_alpha)):
            for j in range(0, len(D0_alpha[0])):
                rmsD_alpha += (D_alpha[i,j] - D0_alpha[i,j])**2
        rmsD_alpha = (rmsD_alpha)**0.5
        rmsD_beta = 0
        for i in range(0, len(D0_beta)):
            for j in range(0, len(D0_beta[0])):
                rmsD_beta += (D_beta[i,j] - D0_beta[i,j])**2
        rmsD_beta = (rmsD_beta)**0.5
        
        if print_SCF == 'Yes':
            output.write(str(iter))
            output.write("\t \t")
            output.write("{:14.10f}".format(Eel))
            output.write("\t \t")
            output.write("{:14.10f}".format(Eel+VNN[0]))
            output.write("\t \t")
            output.write("{: 12.8e}".format(dE))
            output.write("\t \t")
            output.write("{: 12.8e}".format(rmsD_alpha))
            output.write("\t \t")
            output.write("{: 12.8e}".format(rmsD_beta))
    
        D0_alpha = D_alpha
        D0_beta = D_beta
        E0el = Eel
        if dE < deTHR and rmsD_alpha < rmsTHR and rmsD_beta < rmsTHR:
            break
            
    if print_SCF == 'Yes':
        output.write('\n \n')
        output.close()
    
    return Eel+VNN[0], C_alpha, F_alpha, D_alpha, C_beta, F_beta, D_beta, iter

