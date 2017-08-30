import numpy as np

def CIS(occ, F, C, VeeMOspin):
    # Make the spin MO fock matrix
    Fspin = np.zeros((len(F)*2,len(F)*2))
    Cspin = np.zeros((len(F)*2,len(F)*2))
    for p in range(1,len(F)*2+1):
        for q in range(1,len(F)*2+1):
            Fspin[p-1,q-1] = F[(p+1)//2-1,(q+1)//2-1] * (p%2 == q%2)
            Cspin[p-1,q-1] = C[(p+1)//2-1,(q+1)//2-1] * (p%2 == q%2)
    FMOspin = np.dot(np.transpose(Cspin),np.dot(Fspin,Cspin))


    #Construct hamiltonian
    H = np.zeros((occ*(len(Fspin)-occ),occ*(len(Fspin)-occ)))
    jbidx = -1
    for j in range(0, occ):
        for b in range(occ, len(Fspin)):
            jbidx += 1
            iaidx = -1
            for i in range(0, occ):
                for a in range(occ, len(Fspin)):
                    iaidx += 1
                    H[iaidx,jbidx] = VeeMOspin[a,j,i,b] - VeeMOspin[a,j,b,i]
                    if i == j:
                        H[iaidx,jbidx] += FMOspin[a,b]
                    if a == b:
                        H[iaidx,jbidx] -= FMOspin[i,j]
    
    Exc = np.linalg.eigvalsh(H)
    
    return Exc
