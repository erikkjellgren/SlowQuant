import numpy as np
import scipy.linalg

def MulCharge(basis, input, D, S):
    #Loading overlap integrals
    qvec = np.zeros(len(input)-1)
    
    DS = np.dot(D,S)
    for i in range(1, len(input)):
        q = 0
        for j in range(len(S)):
            if basis[j][6] == i:
                mu = basis[j][0]-1
                q += DS[mu,mu]
        qvec[i-1] = input[i,0] - q
    return qvec
    

def LowdinCharge(basis, input, D, S):
    #Loading overlap integrals
    qvec = np.zeros(len(input)-1)
    
    S_sqrt = scipy.linalg.sqrtm(S)
    SDS = np.dot(np.dot(S_sqrt,D),S_sqrt)
    
    for i in range(1, len(input)):
        q = 0
        for j in range(len(basis)):
            if basis[j][6] == i:
                mu = basis[j][0]-1
                q += SDS[mu,mu]
        qvec[i-1] = input[i,0] - q
    return qvec


def dipolemoment(input, D, mux, muy, muz):
    nucx = []
    nucy = []
    nucz = []
    for i in range(1, len(input)):
        nucx.append(input[i,1])
        nucy.append(input[i,2])
        nucz.append(input[i,3])
    
    ux = 0
    for i in range(0, len(D)):
        for j in range(0, len(D[0])):
            ux += D[i,j]*mux[i,j]
            
    uy = 0
    for i in range(0, len(D)):
        for j in range(0, len(D[0])):
            uy += D[i,j]*muy[i,j]
            
    uz = 0
    for i in range(0, len(D)):
        for j in range(0, len(D[0])):
            uz += D[i,j]*muz[i,j]
            
    Cx = 0
    Cy = 0
    Cz = 0
    M = 0
    for i in range(1, len(input)):
        M += input[i,0]
    
    for i in range(1, len(input)):
        Cx += (input[i,0]*input[i,1])/M
        Cy += (input[i,0]*input[i,2])/M
        Cz += (input[i,0]*input[i,3])/M
        
        
    for i in range(0, len(nucx)):
        ux += input[i+1,0]*(nucx[i]-Cx)
    
    for i in range(0, len(nucx)):
        uy += input[i+1,0]*(nucy[i]-Cy)
    
    for i in range(0, len(nucx)):
        uz += input[i+1,0]*(nucz[i]-Cz)
    
    u_mag = (ux**2+uy**2+uz**2)**0.5
    
    return ux, uy, uz, u_mag
 
    
def RPA(occ, F, C, VeeMOspin):
    # Make the spin MO fock matrix
    Fspin = np.zeros((len(F)*2,len(F)*2))
    Cspin = np.zeros((len(F)*2,len(F)*2))
    for p in range(1,len(F)*2+1):
        for q in range(1,len(F)*2+1):
            Fspin[p-1,q-1] = F[(p+1)//2-1,(q+1)//2-1] * (p%2 == q%2)
            Cspin[p-1,q-1] = C[(p+1)//2-1,(q+1)//2-1] * (p%2 == q%2)
    FMOspin = np.dot(np.transpose(Cspin),np.dot(Fspin,Cspin))


    #Construct hamiltonian
    A = np.zeros((occ*(len(Fspin)-occ),occ*(len(Fspin)-occ)))
    B = np.zeros((occ*(len(Fspin)-occ),occ*(len(Fspin)-occ)))
    jbidx = -1
    for j in range(0, occ):
        for b in range(occ, len(Fspin)):
            jbidx += 1
            iaidx = -1
            for i in range(0, occ):
                for a in range(occ, len(Fspin)):
                    iaidx += 1
                    A[iaidx,jbidx] = VeeMOspin[a,j,i,b] - VeeMOspin[a,j,b,i]
                    B[iaidx,jbidx] = VeeMOspin[a,b,i,j] - VeeMOspin[a,b,j,i]
                    if i == j:
                        A[iaidx,jbidx] += FMOspin[a,b]
                    if a == b:
                        A[iaidx,jbidx] -= FMOspin[i,j]
    
    C = np.dot(A+B,A-B)
    Exc = np.sort(np.sqrt(np.linalg.eigvals(C)))
    
    return Exc

