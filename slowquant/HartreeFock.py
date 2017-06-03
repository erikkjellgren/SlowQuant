import numpy as np
import scipy.linalg
import math
from slowquant import DIIS

def diagonlize(M):
    eigVal, eigVec = np.linalg.eigh(M)
    return eigVal, eigVec

def symm_orth(eigVal, eigVec):
    diaVal = np.diag(eigVal)
    M = np.dot(np.dot(eigVec,scipy.linalg.sqrtm(np.linalg.inv(np.diag(eigVal)))),np.matrix.transpose(eigVec))
    return M

def HartreeFock(input, set, basis, VNN, Te, S, VeN, Vee, results, print_SCF='Yes'):
    # ###############################
    #
    # VNN = nuclear repulsion
    # Te  = kinetic energy
    # S   = overlap integrals
    # VeN = nuclear attraction
    # Vee = two electron integrals
    # 
    # ###############################
    
    deTHR = int(set['SCF Energy Threshold'])
    rmsTHR = int(set['SCF RMSD Threshold'])
    Maxiter = int(set['SCF Max iterations'])
    
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
        if set['DIIS'] == 'Yes':
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
        Part = np.zeros((len(basis),len(basis)))
        for mu in range(0, len(basis)):
            for nu in range(0, len(basis)):
                for lam in range(0, len(basis)):
                    for sig in range(0, len(basis)):
                        Part[mu,nu] += D0[lam,sig]*(2*Vee[mu,nu,lam,sig]-Vee[mu,lam,nu,sig])
        
        F = Hcore + Part
        
        if set['DIIS'] == 'Yes':
            #Estimate F by DIIS
            if iter == 1:
                F, errorFock, errorDens, errorDIIS = DIIS.runDIIS(F,D0,S,iter,set,basis,0,0)
            else:
                F, errorFock, errorDens, errorDIIS  = DIIS.runDIIS(F,D0,S,iter,set,basis,errorFock, errorDens)
            
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
        rmsD = math.sqrt(rmsD)
        
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
            if set['DIIS'] == 'Yes':
                if errorDIIS != 'None':
                    output.write("\t \t")
                    output.write("{: 12.8e}".format(errorDIIS))
    
        D0 = D
        E0el = Eel
        if dE < 10**(-deTHR) and rmsD < 10**(-rmsTHR):
            break
            
    if print_SCF == 'Yes':
        output.write('\n \n')
        output.close()
    results['HFenergy'] = Eel+VNN[0]
    
    return C, F, D, results
