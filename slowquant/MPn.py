from slowquant import IntegralTransform as utilF
import numpy as np

def MP2(basis, input, F, C):
    #Get MO orbital energies and C_MO
    CT = np.transpose(C)
    eps = np.dot(np.dot(CT, F),C)

    #Loading two electron integrals
    VeeMO = np.load('slowquant/temp/twointMO.npy')

    #Calc EMP2
    EMP2 = 0
    for i in range(0, int(input[0][0]/2)):
        for a in range(int(input[0][0]/2), len(basis)):
            for j in range(0, int(input[0][0]/2)):
                for b in range(int(input[0][0]/2), len(basis)):
                    EMP2 += VeeMO[i,a,j,b]*(2*VeeMO[i,a,j,b]-VeeMO[i,b,j,a])/(eps[i, i] + eps[j, j] -eps[a, a] -eps[b, b])

    output = open('out.txt', 'a')
    output.write('\n \n')
    output.write('MP2 Energy \t')
    output.write("{: 10.8f}".format(EMP2))
    output.close()
    return EMP2

def runMPn(basis, input, F, C, set):
    if set['MPn'] == 'MP2':
        MP2(basis, input, F, C)