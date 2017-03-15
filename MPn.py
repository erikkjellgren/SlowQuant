import Utilityfunc as utilF
import numpy as np

def MP2(basis, input, F, C):
    #Get MO orbital energies and C_MO
    CT = np.transpose(C)
    eps = np.dot(np.dot(CT, F),C)

    #Loading two electron integrals
    Vee = utilF.load2el(basis)
    
    #Make dict for MO integrals
    VeeMO = {}
    for i in range(1, len(basis)+1):
        for j in range(1, len(basis)+1):
            for k in range(1, len(basis)+1):
                for l in range(1, len(basis)+1):
                    ijkl = utilF.idx2el(i, j, k, l)
                    VeeMO[ijkl] = 0
    
    #Transform two electron integrals to MO basis
    idxcheck = []
    for i in range(1, len(basis)+1):
        for j in range(1, len(basis)+1):
            for k in range(1, len(basis)+1):
                for l in range(1, len(basis)+1):
                    ijkl = utilF.idx2el(i, j, k, l)
                    if ijkl not in idxcheck:
                        idxcheck.append(ijkl)
                        for p in range(1, len(basis)+1):
                            for q in range(1, len(basis)+1):
                                for r in range(1, len(basis)+1):
                                    for s in range(1, len(basis)+1):
                                        pqrs = utilF.idx2el(p, q, r, s)
                                        VeeMO[ijkl] += C[p-1,i-1]*C[q-1,j-1]*C[r-1,k-1]*C[s-1,l-1]*Vee[pqrs]

    #Calc EMP2
    EMP2 = 0
    for i in range(1, int(input[0][0]/2)+1):
        for a in range(int(input[0][0]/2)+1, len(basis)+1):
            for j in range(1, int(input[0][0]/2)+1):
                for b in range(int(input[0][0]/2)+1, len(basis)+1):
                    ibja = utilF.idx2el(i, b, j, a)
                    iajb = utilF.idx2el(i, a, j, b)
                    EMP2 += VeeMO[iajb]*(2*VeeMO[iajb]-VeeMO[ibja])/(eps[i-1, i-1] + eps[j-1, j-1] -eps[a-1, a-1] -eps[b-1, b-1])

    output = open('out.txt', 'a')
    output.write('\n \n')
    output.write('MP2 Energy: \t')
    output.write("{: 10.8f}".format(EMP2))
    output.close()

def runMPn(basis, input, F, C, set):
    if set['MPn'] == 'MP2':
        MP2(basis, input, F, C)