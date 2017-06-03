import numpy as np
import math
from slowquant import MolecularIntegrals as MI
import scipy.linalg

def MulCharge(basis, input, D):
    #Loading overlap integrals
    S = np.load('slowquant/temp/overlap.npy')
    D = 2*D
    
    DS = np.dot(D,S)
    output = open('out.txt', 'a')
    output.write('\n \n')
    output.write('Mulliken Charges \n')
    for i in range(1, len(input)):
        q = 0
        for j in range(len(basis)):
            if basis[j][6] == i:
                mu = basis[j][0]-1
                q += DS[mu,mu]
        q = input[i,0] - q
        output.write('Atom'+str(i)+'\t')
        output.write("{: 10.8f}".format(q))
        output.write('\n')
    output.close()

def LowdinCharge(basis, input, D):
    #Loading overlap integrals
    S = np.load('slowquant/temp/overlap.npy')
    D = 2*D
    
    output = open('out.txt', 'a')
    output.write('\n \n')
    output.write('Lowdin Charges \n')
    
    S_sqrt = scipy.linalg.sqrtm(S)
    SDS = np.dot(np.dot(S_sqrt,D),S_sqrt)
    
    for i in range(1, len(input)):
        q = 0
        for j in range(len(basis)):
            if basis[j][6] == i:
                mu = basis[j][0]-1
                q += SDS[mu,mu]
        q = input[i,0] - q
        output.write('Atom'+str(i)+'\t')
        output.write("{: 10.8f}".format(q))
        output.write('\n')
    output.close()

def dipolemoment(basis, input, D, results):
    nucx = []
    nucy = []
    nucz = []
    for i in range(1, len(input)):
        nucx.append(input[i,1])
        nucy.append(input[i,2])
        nucz.append(input[i,3])
    
    mux = np.load('slowquant/temp/mux.npy')
    muy = np.load('slowquant/temp/muy.npy')
    muz = np.load('slowquant/temp/muz.npy')
    
    ux = 0
    for i in range(0, len(D)):
        for j in range(0, len(D[0])):
            ux += 2*D[i,j]*mux[i,j]
            
    uy = 0
    for i in range(0, len(D)):
        for j in range(0, len(D[0])):
            uy += 2*D[i,j]*muy[i,j]
            
    uz = 0
    for i in range(0, len(D)):
        for j in range(0, len(D[0])):
            uz += 2*D[i,j]*muz[i,j]
            
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
    
    u = math.sqrt(ux**2+uy**2+uz**2)
    
    results['dipolex'] = ux
    results['dipoley'] = uy
    results['dipolez'] = uz
    results['dipoletot'] = u
    
    output = open('out.txt', 'a')
    output.write('\n \nMolecular dipole moment \n')
    output.write('X \t \t')
    output.write("{: 10.8f}".format(ux))
    output.write('\nY \t \t')
    output.write("{: 10.8f}".format(uy))
    output.write('\nZ \t \t')
    output.write("{: 10.8f}".format(uz))
    output.write('\nTotal \t')
    output.write("{: 10.8f}".format(u))
    output.close()
    
    return results

def runprop(basis, input, D, set, results):
    if set['Charge'] == 'Mulliken':
        MulCharge(basis, input, D)
    elif set['Charge'] == 'Lowdin':
        LowdinCharge(basis, input, D)
    if set['Dipole'] == 'Yes':
        MI.run_dipole_int(basis, input)
        results = dipolemoment(basis, input, D, results)
    return results