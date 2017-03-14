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
    Vee = {}
    for i in range(1, len(basis)+1):
        for j in range(1, len(basis)+1):
            for k in range(1, len(basis)+1):
                for l in range(1, len(basis)+1):
                    if i>j:
                        ij = i * (i + 1)/2 + j
                    else:
                        ij = j * (j + 1)/2 + i
                    if k>l:
                        kl = k * (k + 1)/2 + l
                    else:
                        kl = l * (l + 1)/2 + k
                    if ij > kl:
                        ijkl = ij*(ij+1)/2 + kl
                    else:
                        ijkl = kl*(kl+1)/2 + ij
                    Vee[int(ijkl)] = 0
    
    twoint = np.genfromtxt('twoint.txt', delimiter = ';')
    
    for i in range(0, len(twoint)):
        if int(twoint[i][0])>int(twoint[i][1]):
            ij = int(twoint[i][0]) * (int(twoint[i][0]) + 1)/2 + int(twoint[i][1])
        else:
            ij = int(twoint[i][1]) * (int(twoint[i][1]) + 1)/2 + int(twoint[i][0])
        if int(twoint[i][2])>int(twoint[i][3]):
            kl = int(twoint[i][2]) * (int(twoint[i][2]) + 1)/2 + int(twoint[i][3])
        else:
            kl = int(twoint[i][3]) * (int(twoint[i][3]) + 1)/2 + int(twoint[i][2])
        if ij > kl:
            ijkl = ij*(ij+1)/2 + kl
        else:
            ijkl = kl*(kl+1)/2 + ij
        
        Vee[int(ijkl)] = twoint[i][4]
            
    
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
    
    
    SvalM = np.array(Sval)


    #Diagonalizing overlap matrix
    La, Ls = diagonlize(SvalM)
    #Symmetric orthogonal inverse overlap matrix
    Smat = symm_orth(La, Ls)

    
    #Initial Density
    F0p = np.dot(np.dot(np.matrix.transpose(Smat),Hcore),np.matrix.transpose(Smat))
    eps, C0p = diagonlize(F0p)

    C0T = np.matrix.transpose(np.dot(Smat, C0p))
    
    #Only using occupied MOs
    C0T = C0T[0:int(input[0][0]/2)]
    C0 = np.matrix.transpose(C0T)
    D0 = np.dot(C0, C0T)        
    
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
    
    for iter in range(1, 100):
        output.write("\n")
        #New Fock Matrix
        placeholder = []
        for i in range(1, len(basis)+1):
            a = []
            for j in range(1, len(basis)+1):
                b = 0
                for k in range(1, len(basis)+1):
                    for l in range(1, len(basis)+1):
                        if i>j:
                            ij = i * (i + 1)/2 + j
                        else:
                            ij = j * (j + 1)/2 + i
                        if k>l:
                            kl = k * (k + 1)/2 + l
                        else:
                            kl = l * (l + 1)/2 + k
                        if ij > kl:
                            ijkl = ij*(ij+1)/2 + kl
                        else:
                            ijkl = kl*(kl+1)/2 + ij
                        
                        if i>k:
                            ik = i * (i + 1)/2 + k
                        else:
                            ik = k * (k + 1)/2 + i
                        if j>l:
                            jl = j * (j + 1)/2 + l
                        else:
                            jl = l * (l + 1)/2 + j
                        if ik > jl:
                            ikjl = ik*(ik+1)/2 + jl
                        else:
                            ikjl = jl*(jl+1)/2 + ik
                        b += D0[k-1][l-1] * (2 * Vee[ijkl] - Vee[ikjl])
                a.append(b)
            placeholder.append(a)
                        
        
        Fnew = np.add(Hcore, placeholder)
        
        #New Density Matrix
        Fp = np.dot(np.dot(np.transpose(Smat),Fnew),Smat)
        eps, Cp = diagonlize(Fp)
        
        Cr = np.dot(Smat, Cp)
        
        CT = np.matrix.transpose(Cr)
        CT = CT[0:int(input[0][0]/2)]
        C = np.matrix.transpose(CT)
        
        D = np.dot(C, CT)
        
        #New SCF Energy
        Eel = 0
        for i in range(0, len(D)):
            for j in range(0, len(D[0])):
                Eel += D[i][j]*(Hcore[i][j]+Fnew[i][j])

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
        if dE < 10**(-12) and rmsD < 10**(-12):
            break
            
    output.close()
    
    return Cr, Fnew, D

    








