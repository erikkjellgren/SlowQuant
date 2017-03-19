import numpy as np
import scipy.linalg
import math
import Utilityfunc as utilF

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
    VNN = np.genfromtxt('enuc.txt', delimiter = ';')
    
    #Loading kinetic energy
    Te = {}
    kin = np.genfromtxt('kinen.txt', delimiter = ';')
    
    for i in range(0, len(kin)):
        Te[str(int(kin[i][0]))+str(int(kin[i][1]))] = kin[i][2]
    
    #Loading overlap integrals
    S = {}
    overlap = np.genfromtxt('overlap.txt', delimiter = ';')
    
    for i in range(0, len(overlap)):
        S[str(int(overlap[i][0]))+str(int(overlap[i][1]))] = overlap[i][2]
    
    #Loading nuclear attraction
    VeN = {}
    nucatt = np.genfromtxt('nucatt.txt', delimiter = ';')
    
    for i in range(0, len(nucatt)):
        VeN[str(int(nucatt[i][0]))+str(int(nucatt[i][1]))] = nucatt[i][2]
    
    #Loading two electron integrals
    Vee = utilF.load2el(basis)
            
    
    #Core Hamiltonian
    Hcore = []
    for i in range(1, len(basis)+1):
        a = []
        for j in range(1, len(basis)+1):
            if i > j:
                a.append(VeN[str(i)+str(j)]+Te[str(i)+str(j)])
            else:
                a.append(VeN[str(j)+str(i)]+Te[str(j)+str(i)])
        
        Hcore.append(a)
    

    #Overlap matrix
    Sval = []
    for i in range(1, len(basis)+1):
        a = []
        for j in range(1, len(basis)+1):
            if i > j:
                a.append(S[str(i)+str(j)])
            else:
                a.append(S[str(j)+str(i)])
        
        Sval.append(a)
    
    
    S = np.array(Sval)


    #Diagonalizing overlap matrix
    Lambda_S, L_S = diagonlize(S)
    #Symmetric orthogonal inverse overlap matrix
    S_sqrt = symm_orth(Lambda_S, L_S)

    
    #Initial Density
    F0prime = np.dot(np.dot(np.matrix.transpose(S_sqrt),Hcore),np.matrix.transpose(S_sqrt))
    eps0, C0prime = diagonlize(F0prime)

    C0 = np.matrix.transpose(np.dot(S_sqrt, C0prime))
    
    #Only using occupied MOs
    C0 = C0[0:int(input[0][0]/2)]
    C0T = np.matrix.transpose(C0)
    D0 = np.dot(C0T, C0)        
    
    # Initial Energy
    E0el = 0
    for i in range(0, len(D0)):
        for j in range(0, len(D0[0])):
            E0el += D0[i][j]*(Hcore[i][j]+Hcore[i][j])
    
    
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
    output.write("{:14.10f}".format(E0el+VNN))
    
    for iter in range(1, Maxiter):
        output.write("\n")
        #New Fock Matrix
        placeholder = []
        for i in range(1, len(basis)+1):
            a = []
            for j in range(1, len(basis)+1):
                b = 0
                for k in range(1, len(basis)+1):
                    for l in range(1, len(basis)+1):
                        ijkl = utilF.idx2el(i, j, k, l)
                        ikjl = utilF.idx2el(i, k, j, l)
                        b += D0[k-1][l-1] * (2 * Vee[ijkl] - Vee[ikjl])
                a.append(b)
            placeholder.append(a)
                        
        
        F = np.add(Hcore, placeholder)
        
        #New Density Matrix
        Fprime = np.dot(np.dot(np.transpose(S_sqrt),F),S_sqrt)
        eps, Cprime = diagonlize(Fprime)
        
        C = np.dot(S_sqrt, Cprime)
        
        CT = np.matrix.transpose(C)
        CTocc = CT[0:int(input[0][0]/2)]
        Cocc = np.matrix.transpose(CTocc)
        
        D = np.dot(Cocc, CTocc)
        
        #New SCF Energy
        Eel = 0
        for i in range(0, len(D)):
            for j in range(0, len(D[0])):
                Eel += D[i][j]*(Hcore[i][j]+F[i][j])

        #Convergance
        dE = Eel - E0el
        rmsD = 0
        for i in range(0, len(D0)):
            for j in range(0, len(D0[0])):
                rmsD += (D[i][j] - D0[i][j])**2
        rmsD = math.sqrt(rmsD)
        
        output.write(str(iter))
        output.write("\t \t")
        output.write("{:14.10f}".format(Eel))
        output.write("\t \t")
        output.write("{:14.10f}".format(Eel+VNN))
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

    








