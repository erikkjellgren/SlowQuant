import numpy as np
import MolecularIntegrals as MI

# #############################################
# Charge fitting script
# Constraining dipole is not working!
# Erik Kjellgren
# 19-03-2017
# #############################################

def magvec(v1, v2):
    x = v2[0] - v1[0]
    y = v2[1] - v1[1]
    z = v2[2] - v1[2]
    return (x**2+y**2+z**2)**0.5

def chrfit(basis, input, D, set, results):
    imesh = 10
    jmesh = 10
    
    # [x, y, z, VQM]
    V = np.zeros((imesh*jmesh,4))
    
    # CHOOSE POINTS
    Xcm = 0
    Ycm = 0
    Zcm = 0
    M = 0
    for i in range(1, len(input)):
        M += input[i][0]
    
    for i in range(1, len(input)):
        Xcm += (input[i][0]*input[i][1])/M
        Ycm += (input[i][0]*input[i][2])/M
        Zcm += (input[i][0]*input[i][3])/M
    
    radius = 3.4 + np.sqrt(np.max((Xcm-input[1:,1])**2)+np.max((Ycm-input[1:,2])**2)+np.max((Zcm-input[1:,3])**2))
    
    idx = 0
    for i in range(0, imesh):
        for j in range(0, jmesh):
            V[idx, 0] = radius*np.cos(2*3.14/imesh*i)*np.sin(3.14/jmesh*j) + Xcm
            V[idx, 1] = radius*np.sin(2*3.14/imesh*i)*np.sin(3.14/jmesh*j) + Ycm
            V[idx, 2] = radius*np.cos(3.14/jmesh*j) + Zcm
            idx += 1
    # END OF CHOOSING POINTS
    
    #Caclculate QM potential
    for i in range(len(V)):
        Ve = MI.runQMESP(basis, input, V[i,0], V[i,1], V[i,2])
        
        NucESP = 0
        for j in range(1, len(input)):
            v1 = [input[j, 1], input[j, 2], input[j, 3]]
            v2 = [V[i, 0], V[i, 1], V[i, 2]]
            r12 = magvec(v1, v2)
            NucESP += input[j, 0]/r12
            
        ElecESP = 0
        for j in range(1, len(basis)+1):
            for k in range(1, len(basis)+1):
                if j >= k:
                    ElecESP += D[j-1,k-1]*Ve[str(int(j))+';'+str(int(k))]
                else:
                    ElecESP += D[j-1,k-1]*Ve[str(int(k))+';'+str(int(j))]
        
        V[i,3] = NucESP - 2*ElecESP
    # END OF calculate QM potential
    
    #Building A and B matrix
    Constr = 0
    if set['Constraint charge'] == 'Yes':
        Constr += 1
    if set['Constraint dipole'] == 'Yes':
        Constr += 3
    B = np.zeros(len(input)-1+Constr)
    A = np.zeros((len(input)-1+Constr,len(input)-1+Constr))
    for k in range(1, len(B)+1-Constr):
        for i in range(len(V)):
            B[k-1] += V[i,3]/(((input[k, 1]-V[i, 0])**2+(input[k, 2]-V[i, 1])**2+(input[k, 3]-V[i, 2])**2)**0.5)
    
    Bshift = 0
    if set['Constraint charge'] == 'Yes':
        B[len(B)-Constr+Bshift] = float(set['Qtot'])
        Bshift += 1
    if set['Constraint dipole'] == 'Yes':
        B[len(B)-Constr+Bshift] = float(results['dipolex'])
        Bshift += 1
        B[len(B)-Constr+Bshift] = float(results['dipoley'])
        Bshift += 1
        B[len(B)-Constr+Bshift] = float(results['dipolez'])
        Bshift += 1
        
    for j in range(1, len(A)+1-Constr):
        for k in range(1, len(A)+1-Constr):
            for i in range(len(V)):
                vj = [input[j, 1], input[j, 2], input[j, 3]]
                vk = [input[k, 1], input[k, 2], input[k, 3]]
                vi = [V[i, 0], V[i, 1], V[i, 2]]
                rij = magvec(vi, vj)
                rik = magvec(vi, vk)
                A[j-1,k-1] += 1/(rij*rik)
    
    Ashift = 0
    if set['Constraint charge'] == 'Yes':
        for i in range(len(A)-Constr):
                A[len(A)-Constr+Ashift, i] = 1
                A[i, len(A)-Constr+Ashift] = 1
        Ashift += 1

    if set['Constraint dipole'] == 'Yes':
        for i in range(len(A)-Constr):
            for j in range(1, len(input)):
                A[len(A)-Constr+Ashift, j-1] = input[j][1] - Xcm
                A[j-1, len(A)-Constr+Ashift] = input[j][1] - Xcm
        Ashift += 1
        for i in range(len(A)-Constr):
            for j in range(1, len(input)):
                A[len(A)-Constr+Ashift, j-1] = input[j][2] - Ycm
                A[j-1, len(A)-Constr+Ashift] = input[j][2] - Ycm
        Ashift += 1
        for i in range(len(A)-Constr):
            for j in range(1, len(input)):
                A[len(A)-Constr+Ashift, j-1] = input[j][3] - Zcm
                A[j-1, len(A)-Constr+Ashift] = input[j][3] - Zcm
        Ashift += 1
    # END OF building A and B matrix

    Ainv = np.linalg.inv(A)
    q = np.dot(Ainv, B)
    
    RMSD = 0
    for i in range(len(V)):
        E = 0
        for j in range(len(q)-Constr):
            rij = ((input[j+1, 1]-V[i, 0])**2+(input[j+1, 2]-V[i, 1])**2+(input[j+1, 3]-V[i, 2])**2)**0.5
            E += q[j]/(rij)
        RMSD += (V[i, 3] - E)**2
    RMSD = (RMSD/len(V))**0.5
    
    output = open('out.txt', 'a')
    output.write('\n \n')
    output.write('Fitted Charges \n')
    for i in range(1, len(input)):
        output.write('Atom'+str(i)+'\t')
        output.write("{: 10.8f}".format(q[i-1]))
        output.write('\n')
    output.write('RMSD \t')
    output.write("{: 12.8e}".format(RMSD))
    output.close()
