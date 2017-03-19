import numpy as np
import scipy.linalg
import math

def diagonlize(M):
    eigVal, eigVec = np.linalg.eigh(M)
    return eigVal, eigVec

def symm_orth(eigVal, eigVec):
    diaVal = np.diag(eigVal)
    M = np.dot(np.dot(eigVec,scipy.linalg.sqrtm(np.linalg.inv(np.diag(eigVal)))),np.matrix.transpose(eigVec))
    return M

def HartreeFock(input, set, basis):
    deTHR = int(set['SCF Energy Threshold'])
    rmsTHR = int(set['SCF RMSD Threshold'])
    Maxiter = int(set['SCF Max iterations'])
    
    #Loading nuclear repulsion
    VNN = np.load('enuc.npy')
    
    #Loading kinetic energy
    Te = np.load('Ekin.npy')
    
    #Loading overlap integrals
    S = np.load('overlap.npy')
    
    #Loading nuclear attraction
    VeN = np.load('nucatt.npy')
    
    #Loading two electron integrals
    Vee = np.load('twoint.npy')
    
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
    output = open('out.txt', 'w')
    output.write('Iter')
    output.write("\t")
    output.write('Eel')
    output.write("\t \t \t \t \t")
    output.write('Etot')
    output.write("\t \t \t \t")
    output.write('dE')
    output.write("\t \t \t \t \t")
    output.write('rmsD')
    output.write("\n")
    output.write('0')
    output.write("\t \t")
    output.write("{:14.10f}".format(E0el))
    output.write("\t \t")
    output.write("{:14.10f}".format(E0el+VNN[0]))
    
    for iter in range(1, Maxiter):
        output.write("\n")
        #New Fock Matrix
        Part = np.zeros((len(basis),len(basis)))
        for mu in range(0, len(basis)):
            for nu in range(0, len(basis)):
                for lam in range(0, len(basis)):
                    for sig in range(0, len(basis)):
                        Part[mu,nu] += D0[lam,sig]*(2*Vee[mu,nu,lam,sig]-Vee[mu,lam,nu,sig])
        
        F = Hcore + Part
        #New Density Matrix
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
        
        output.write(str(iter))
        output.write("\t \t")
        output.write("{:14.10f}".format(Eel))
        output.write("\t \t")
        output.write("{:14.10f}".format(Eel+VNN[0]))
        output.write("\t \t")
        output.write("{: 12.8e}".format(dE))
        output.write("\t \t")
        output.write("{: 12.8e}".format(rmsD))
    
        D0 = D
        E0el = Eel
        if dE < 10**(-deTHR) and rmsD < 10**(-rmsTHR):
            break
    output.close()
    
    return C, F, D

    








