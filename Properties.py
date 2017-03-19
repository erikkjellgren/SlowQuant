import numpy as np
import math
import MolecularIntegrals as MI

def MulCharge(basis, input, D):
    #Loading overlap integrals
    S = {}
    overlap = np.genfromtxt('overlap.txt', delimiter = ';')
    for i in range(0, len(overlap)):
        S[str(int(overlap[i][0]))+str(int(overlap[i][1]))] = overlap[i][2]
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
    Smat = np.array(Sval)
    
    DS = np.dot(D,Smat)
    output = open('out.txt', 'a')
    output.write('\n \n')
    output.write('Mulliken Charges \n')
    for i in range(1, len(input)):
        q = 0
        for j in range(len(basis)):
            if basis[j][6] == i:
                mu = basis[j][0]-1
                q += DS[mu][mu]
        q = input[i][0] - 2*q
        output.write('Atom'+str(i)+'\t')
        output.write("{: 10.8f}".format(q))
        output.write('\n')
    output.close()

def dipolemoment(basis, input, D, results):
    nucx = []
    nucy = []
    nucz = []
    for i in range(1, len(input)):
        nucx.append(input[i][1])
        nucy.append(input[i][2])
        nucz.append(input[i][3])
    
    mx = {}
    muxdat = np.genfromtxt('mux.txt', delimiter = ';')
    
    for i in range(0, len(muxdat)):
        mx[str(int(muxdat[i][0]))+str(int(muxdat[i][1]))] = muxdat[i][2]
    
    my = {}
    muydat = np.genfromtxt('muy.txt', delimiter = ';')
    
    for i in range(0, len(muydat)):
        my[str(int(muxdat[i][0]))+str(int(muydat[i][1]))] = muydat[i][2]
    mz = {}
    muzdat = np.genfromtxt('muz.txt', delimiter = ';')
    
    for i in range(0, len(muzdat)):
        mz[str(int(muzdat[i][0]))+str(int(muzdat[i][1]))] = muzdat[i][2]
    
    mux = []
    for i in range(1, len(D)+1):
        a = []
        for j in range(1, len(D)+1):
            if i > j:
                a.append(mx[str(i)+str(j)])
            else:
                a.append(mx[str(j)+str(i)])
        
        mux.append(a)
    
    muy = []
    for i in range(1, len(D)+1):
        a = []
        for j in range(1, len(D)+1):
            if i > j:
                a.append(my[str(i)+str(j)])
            else:
                a.append(my[str(j)+str(i)])
        
        muy.append(a)
    
    muz = []
    for i in range(1, len(D)+1):
        a = []
        for j in range(1, len(D)+1):
            if i > j:
                a.append(mz[str(i)+str(j)])
            else:
                a.append(mz[str(j)+str(i)])
        
        muz.append(a)
    
    
    muxM = np.array(mux)
    muyM = np.array(muy)
    muzM = np.array(muz)
    
    ux = 0
    for i in range(0, len(D)):
        for j in range(0, len(D[0])):
            ux += 2*D[i][j]*muxM[i][j]
            
    uy = 0
    for i in range(0, len(D)):
        for j in range(0, len(D[0])):
            uy += 2*D[i][j]*muyM[i][j]
            
    uz = 0
    for i in range(0, len(D)):
        for j in range(0, len(D[0])):
            uz += 2*D[i][j]*muzM[i][j]
            
    Cx = 0
    Cy = 0
    Cz = 0
    M = 0
    for i in range(1, len(input)):
        M += input[i][0]
    
    for i in range(1, len(input)):
        Cx += (input[i][0]*input[i][1])/M
        Cy += (input[i][0]*input[i][2])/M
        Cz += (input[i][0]*input[i][3])/M
        
        
    for i in range(0, len(nucx)):
        ux += input[i+1][0]*(nucx[i]-Cx)
    
    for i in range(0, len(nucx)):
        uy += input[i+1][0]*(nucy[i]-Cy)
    
    for i in range(0, len(nucx)):
        uz += input[i+1][0]*(nucz[i]-Cz)
    
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
    if set['Dipole'] == 'Yes':
        MI.run_dipole_int(basis, input)
        results = dipolemoment(basis, input, D, results)
    return results