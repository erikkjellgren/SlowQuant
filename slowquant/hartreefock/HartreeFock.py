import numpy as np
import scipy.linalg
from slowquant.hartreefock import DIIS

def diagonlize(M):
    eigVal, eigVec = np.linalg.eigh(M)
    return eigVal, eigVec

def symm_orth(eigVal, eigVec):
    diaVal = np.diag(eigVal)
    M = np.dot(np.dot(eigVec,scipy.linalg.sqrtm(np.linalg.inv(np.diag(eigVal)))),np.matrix.transpose(eigVec))
    return M

def HartreeFock(input, VNN, Te, S, VeN, Vee, deTHR=10**-6,rmsTHR=10**-6,Maxiter=100, DO_DIIS='Yes', DIIS_steps=6, print_SCF='Yes'):
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
    
    #Initial Density
    F0prime = np.dot(np.dot(np.matrix.transpose(S_sqrt),Hcore),np.matrix.transpose(S_sqrt))
    eps0, C0prime = diagonlize(F0prime)
    
    C0 = np.matrix.transpose(np.dot(S_sqrt, C0prime))
    
    #Only using occupied MOs
    C0 = C0[0:int(input[0,0]/2)]
    C0T = np.matrix.transpose(C0)
    D0 = np.dot(C0T, C0)        
    
    # Initial Energy
    E0el = 0
    for i in range(0, len(D0)):
        for j in range(0, len(D0[0])):
            E0el += D0[i,j]*(Hcore[i,j]+Hcore[i,j])
    
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
        output.write('rmsD')
        if DO_DIIS == 'Yes':
            output.write("\t \t \t \t \t")
            output.write('DIIS')
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
        J = np.einsum('pqrs,sr->pq', Vee,D0)
        K = np.einsum('psqr,sr->pq', Vee,D0)
        F = Hcore + 2.0*J-K
        
        if DO_DIIS == 'Yes':
            #Estimate F by DIIS
            if iter == 1:
                F, errorFock, errorDens, errorDIIS = DIIS.runDIIS(F,D0,S,iter,DIIS_steps,0,0)
            else:
                F, errorFock, errorDens, errorDIIS  = DIIS.runDIIS(F,D0,S,iter,DIIS_steps,errorFock, errorDens)
            
        Fprime = np.dot(np.dot(np.transpose(S_sqrt),F),S_sqrt)
        eps, Cprime = diagonlize(Fprime)
        
        C = np.dot(S_sqrt, Cprime)
        
        CT = np.matrix.transpose(C)
        CTocc = CT[0:int(input[0,0]/2)]
        Cocc = np.matrix.transpose(CTocc)
        
        D = np.dot(Cocc, CTocc)
        
        #New SCF Energy
        Eel = 0
        for i in range(0, len(D)):
            for j in range(0, len(D[0])):
                Eel += D[i,j]*(Hcore[i,j]+F[i,j])

        #Convergance
        dE = Eel - E0el
        rmsD = 0
        for i in range(0, len(D0)):
            for j in range(0, len(D0[0])):
                rmsD += (D[i,j] - D0[i,j])**2
        rmsD = (rmsD)**0.5
        
        if print_SCF == 'Yes':
            output.write(str(iter))
            output.write("\t \t")
            output.write("{:14.10f}".format(Eel))
            output.write("\t \t")
            output.write("{:14.10f}".format(Eel+VNN[0]))
            output.write("\t \t")
            output.write("{: 12.8e}".format(dE))
            output.write("\t \t")
            output.write("{: 12.8e}".format(rmsD))
            if DO_DIIS == 'Yes':
                if errorDIIS != 'None':
                    output.write("\t \t")
                    output.write("{: 12.8e}".format(errorDIIS))
    
        D0 = D
        E0el = Eel
        if np.abs(dE) < deTHR and rmsD < rmsTHR:
            break
            
    if print_SCF == 'Yes':
        output.write('\n \n')
        output.close()
    
    return Eel+VNN[0], C, F, D, iter
