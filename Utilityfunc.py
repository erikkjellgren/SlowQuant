import numpy as np


def idx2el(i, j, k, l):
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

    return int(ijkl)
    

def load2el(basis, **kwargs):
    if 'MO' in kwargs:
        Vee = {}
        for i in range(1, len(basis)+1):
            for j in range(1, len(basis)+1):
                for k in range(1, len(basis)+1):
                    for l in range(1, len(basis)+1):
                        ijkl = idx2el(i, j, k, l)
                        Vee[ijkl] = 0
        twoint = np.genfromtxt('twointMO.txt', delimiter = ';')
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
        
        return Vee
        
    else:
        Vee = {}
        for i in range(1, len(basis)+1):
            for j in range(1, len(basis)+1):
                for k in range(1, len(basis)+1):
                    for l in range(1, len(basis)+1):
                        ijkl = idx2el(i, j, k, l)
                        Vee[ijkl] = 0
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
        
        return Vee

def TransformMO(C, basis, set):
    #Check for key that require MO integrals
    MOcheck = 0
    if set['MPn'] == 'MP2':
        MOcheck = 1
    if MOcheck == 1:
        #Loading two electron integrals
        Vee = load2el(basis)
        
        #Make dict for MO integrals
        VeeMO = {}
        for i in range(1, len(basis)+1):
            for j in range(1, len(basis)+1):
                for k in range(1, len(basis)+1):
                    for l in range(1, len(basis)+1):
                        ijkl = idx2el(i, j, k, l)
                        VeeMO[ijkl] = 0
    
        #Transform two electron integrals to MO basis
        idxcheck = []
        for i in range(1, len(basis)+1):
            for j in range(1, len(basis)+1):
                for k in range(1, len(basis)+1):
                    for l in range(1, len(basis)+1):
                        ijkl = idx2el(i, j, k, l)
                        if ijkl not in idxcheck:
                            idxcheck.append(ijkl)
                            for p in range(1, len(basis)+1):
                                for q in range(1, len(basis)+1):
                                    for r in range(1, len(basis)+1):
                                        for s in range(1, len(basis)+1):
                                            pqrs = idx2el(p, q, r, s)
                                            VeeMO[ijkl] += C[p-1,i-1]*C[q-1,j-1]*C[r-1,k-1]*C[s-1,l-1]*Vee[pqrs]

        output = open('twointMO.txt', 'w')
        idxcheck = []
        for i in range(1, len(basis)+1):
            for j in range(1, len(basis)+1):
                for k in range(1, len(basis)+1):
                    for l in range(1, len(basis)+1):
                        ijkl = idx2el(i, j, k, l)
                        if ijkl not in idxcheck:
                            idxcheck.append(ijkl)
                            output.write(str(i)+';'+str(j)+';'+str(k)+';'+str(l))
                            output.write(";")
                            output.write(str(VeeMO[ijkl]))
                            output.write("\n")
        output.close()

