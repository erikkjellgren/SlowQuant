import numpy as np

def idx1el():
    return None

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
    
def load1el():
    return None

def load2el(basis):
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
