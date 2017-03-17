import numpy as np
import MolecularIntegrals as MI

## NOT WORKING
def chrfit(basis, input, D):
    imesh = 10
    jmesh = 10
    
    # [pointidx, x, y, z, VQM]
    V = np.zeros((imesh*jmesh,4))
    
    # CHOOSE POINTS
    Xcm = 0
    Ycm = 0
    Zcm = 0
    for i in range(1, len(input)):
        Xcm += input[i, 1]/(len(input)-1)
        Ycm += input[i, 2]/(len(input)-1)
        Zcm += input[i, 3]/(len(input)-1)
    
    radius = 3.4 + np.sqrt(np.max((Xcm-input[1:,1])**2)+np.max((Ycm-input[1:,2])**2)+np.max((Zcm-input[1:,3])**2))
    
    idx = 0
    for i in range(0, imesh):
        for j in range(0, jmesh):
            V[idx, 0] = radius*np.cos(2*3.14/imesh*i)*np.sin(3.14/jmesh*j) - Xcm
            V[idx, 1] = radius*np.sin(2*3.14/imesh*i)*np.sin(3.14/jmesh*j) - Ycm
            V[idx, 2] = radius*np.cos(3.14/jmesh*j) - Zcm
            idx += 1
    # END OF CHOOSING POINTS
    
    for i in range(len(V)):
        Ve = MI.runQMESP(basis, input, V[i,0], V[i,1], V[i,2])
        
        NucESP = 0
        for j in range(1, len(input)):
            NucESP += input[j, 0]/(((input[j, 1]-V[i, 0])**2+(input[j, 2]-V[i, 1])**2+(input[j, 3]-V[i, 2])**2)**0.5)
            
        ElecESP = 0
        for j in range(1, len(basis)+1):
            for k in range(1, len(basis)+1):
                if j >= k:
                    ElecESP += D[j-1,k-1]*Ve[str(int(j))+';'+str(int(k))]
                else:
                    # Is this double counting, or should both elements be used?
                    ElecESP += D[j-1,k-1]*Ve[str(int(k))+';'+str(int(j))]
        
        V[i,3] = NucESP - ElecESP
    
    #Construct matrix
    #len(input) = atoms +1, here +1 account for constraint
    B = np.zeros(len(input))
    A = np.zeros((len(input),len(input)))
    for k in range(1, len(B)):
        #Last element will be qtot
        for i in range(len(V)):
            B[k-1] += V[i,3]/(((input[k, 1]-V[i, 0])**2+(input[k, 2]-V[i, 1])**2+(input[k, 3]-V[i, 2])**2)**0.5)
    
    B[len(B)-1] = 0
    
    for j in range(1, len(A)):
        for k in range(1, len(A)):
            for i in range(len(V)):
                rij = ((input[j, 1]-V[i, 0])**2+(input[j, 2]-V[i, 1])**2+(input[j, 3]-V[i, 2])**2)**0.5
                rik = ((input[k, 1]-V[i, 0])**2+(input[k, 2]-V[i, 1])**2+(input[k, 3]-V[i, 2])**2)**0.5
                A[j-1,k-1] += 1/(rij*rik)
    
    for i in range(len(A)-1):
        A[len(A)-1, i] = 1
        A[i, len(A)-1] = 1
    print(B)
    print(A)
    
    Ainv = np.linalg.inv(A)
    q = np.dot(Ainv, B)
    
    print(q)

    