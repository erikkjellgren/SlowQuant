import numpy as np
import MolecularIntegrals as MI
#### DIPOLE FIT NOT WORKING
#### UNKNOWN IF DIPOLE CONSTRAIN WORK FOR CHARGE FITTING

def magvec(v1, v2):
    x = v2[0] - v1[0]
    y = v2[1] - v1[1]
    z = v2[2] - v1[2]
    return (x**2+y**2+z**2)**0.5

## CHARGE FIT
def chrfit(basis, input, D, set, results):
    meshtot = int(set['Gridpoints'])
    imesh = int((meshtot)**0.5)
    jmesh = int((meshtot)**0.5)
    Atoms = len(input)-1
    
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
        for j in range(0, len(basis)):
            for k in range(0, len(basis)):
                ElecESP += D[j,k]*Ve[j,k]

        V[i,3] = NucESP - 2*ElecESP #Where does the 2 come from?
    # END OF calculate QM potential
    
    #Building A and B matrix
    Constr = 0
    if set['Constraint charge'] == 'Yes':
        Constr += 1
    if set['Constraint dipole'] == 'Yes':
        Constr += 3
    B = np.zeros(Atoms+Constr)
    A = np.zeros((Atoms+Constr,Atoms+Constr))
    for k in range(1, len(B)+1-Constr):
        for i in range(len(V)):
            B[k-1] += V[i,3]/(((input[k, 1]-V[i, 0])**2+(input[k, 2]-V[i, 1])**2+(input[k, 3]-V[i, 2])**2)**0.5)
    
    Bshift = 0
    #Checking constraints in construction of B Vector
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
    #Checking constraints in construction of A matrix
    if set['Constraint charge'] == 'Yes':
        for i in range(len(A)-Constr):
            A[len(A)-Constr+Ashift, i] = 1
            A[i, len(A)-Constr+Ashift] = 1
        Ashift += 1

    if set['Constraint dipole'] == 'Yes':
        for j in range(1, len(input)):
            A[len(A)-Constr+Ashift, j-1] = input[j][1] - Xcm
            A[j-1, len(A)-Constr+Ashift] = input[j][1] - Xcm
        Ashift += 1
        for j in range(1, len(input)):
            A[len(A)-Constr+Ashift, j-1] = input[j][2] - Ycm
            A[j-1, len(A)-Constr+Ashift] = input[j][2] - Ycm
        Ashift += 1
        for j in range(1, len(input)):
            A[len(A)-Constr+Ashift, j-1] = input[j][3] - Zcm
            A[j-1, len(A)-Constr+Ashift] = input[j][3] - Zcm
        Ashift += 1
    
    #Check A for all zero rows
    for j in range(0, len(A)):
        #Double loop because A changes len when row/coloum is removed
        #If row/coloum that is all 0 the matrix will be singular
        for i in range(0, len(A)):
            checksum = np.sum(np.abs(A[i,:]))
            if checksum == 0:
                A = np.delete(A,i,axis=1)
                A = np.delete(A,i,axis=0)
                B = np.delete(B,i,axis=0)
                break
    # END OF building A and B matrix

    #WORKING EQUATIONS
    Ainv = np.linalg.inv(A)
    q = np.dot(Ainv, B)
    # END OF WORKING EQUATIONS
    
    #Calculate RMSD
    RMSD = 0
    for i in range(len(V)):
        E = 0
        for j in range(len(q)-Constr):
            rij = ((input[j+1, 1]-V[i, 0])**2+(input[j+1, 2]-V[i, 1])**2+(input[j+1, 3]-V[i, 2])**2)**0.5
            E += q[j]/(rij)
        RMSD += (V[i, 3] - E)**2
    RMSD = (RMSD/len(V))**0.5
    # END OF calculating RMSD
    
    #Write output
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
    # END OF writing output


## CHARGE AND DIPOLE FIT
def dipolefit(basis, input, D, set, results):
    meshtot = int(set['Gridpoints'])
    imesh = int((meshtot)**0.5)
    jmesh = int((meshtot)**0.5)
    Atoms = len(input)-1
    
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
        for j in range(0, len(basis)):
            for k in range(0, len(basis)):
                ElecESP += D[j,k]*Ve[j,k]

        V[i,3] = NucESP - 2*ElecESP #Where does the 2 come from?
    # END OF calculate QM potential
    
    #Building A and B matrix
    Constr = 0
    if set['Constraint charge'] == 'Yes':
        Constr += 1
    if set['Constraint dipole'] == 'Yes':
        Constr += 3
    B = np.zeros((Atoms)*4+Constr)
    A = np.zeros(((Atoms)*4+Constr,(Atoms)*4+Constr))
    
    #Assign values to B vector
    for k in range(1, len(B)+1-Constr):
        if k <= Atoms:
            kidx = (k-1)%Atoms+1
            for i in range(len(V)):
                B[k-1] += V[i,3]/(((input[kidx, 1]-V[i, 0])**2+(input[kidx, 2]-V[i, 1])**2+(input[kidx, 3]-V[i, 2])**2)**0.5)
        elif k<=(Atoms)*2:
            for i in range(len(V)):
                B[k-1] += (V[i, 0]-input[kidx, 1])*V[i,3]/((((input[kidx, 1]-V[i, 0])**2+(input[kidx, 2]-V[i, 1])**2+(input[kidx, 3]-V[i, 2])**2)**0.5)**3)
        elif k<=(Atoms)*3:
            for i in range(len(V)):
                B[k-1] += (V[i, 1]-input[kidx, 2])*V[i,3]/((((input[kidx, 1]-V[i, 0])**2+(input[kidx, 2]-V[i, 1])**2+(input[kidx, 3]-V[i, 2])**2)**0.5)**3)
        elif k<=(Atoms)*4:
            for i in range(len(V)):
                B[k-1] += (V[i, 2]-input[kidx, 3])*V[i,3]/((((input[kidx, 1]-V[i, 0])**2+(input[kidx, 2]-V[i, 1])**2+(input[kidx, 3]-V[i, 2])**2)**0.5)**3)
    
    #Assign values to A matrix
    for j in range(1, len(A)+1-Constr):
        for k in range(1, j+1):
            kidx = (k-1)%Atoms+1
            jidx = (j-1)%Atoms+1
            vj = [input[jidx, 1], input[jidx, 2], input[jidx, 3]]
            vk = [input[kidx, 1], input[kidx, 2], input[kidx, 3]]
            if j <= Atoms and k <= Atoms:
                # Charge 
                for i in range(len(V)):
                    vi = [V[i, 0], V[i, 1], V[i, 2]]
                    rij = magvec(vi, vj)
                    rik = magvec(vi, vk)
                    if j == k:
                        A[j-1,k-1] += 1/(rij*rik)
                    else:
                        A[j-1,k-1] += 1/(rij*rik)
                        A[k-1,j-1] += 1/(rij*rik)
            elif j <= (Atoms)*2 and k <= Atoms:
                # dipole_x charge
                for i in range(len(V)):
                    vi = [V[i, 0], V[i, 1], V[i, 2]]
                    rij = magvec(vi, vj)
                    rik = magvec(vi, vk)
                    A[j-1,k-1] += (V[i, 0]-input[jidx, 1])/(rij**3 * rik)
                    A[k-1,j-1] += (V[i, 0]-input[jidx, 1])/(rij**3 * rik)
            elif j <= (Atoms)*3 and k <= Atoms:
                # dipole_y charge
                for i in range(len(V)):
                    vi = [V[i, 0], V[i, 1], V[i, 2]]
                    rij = magvec(vi, vj)
                    rik = magvec(vi, vk)
                    A[j-1,k-1] += (V[i, 1]-input[jidx, 2])/(rij**3 * rik)
                    A[k-1,j-1] += (V[i, 1]-input[jidx, 2])/(rij**3 * rik)
            elif j <= (Atoms)*4 and k <= Atoms:
                # dipole_z charge
                for i in range(len(V)):
                    vi = [V[i, 0], V[i, 1], V[i, 2]]
                    rij = magvec(vi, vj)
                    rik = magvec(vi, vk)
                    A[j-1,k-1] += (V[i, 2]-input[jidx, 3])/(rij**3 * rik)
                    A[k-1,j-1] += (V[i, 2]-input[jidx, 3])/(rij**3 * rik)
            elif j <= (Atoms)*2 and k <= (Atoms)*2:
                # dipole_x
                for i in range(len(V)):
                    vi = [V[i, 0], V[i, 1], V[i, 2]]
                    rij = magvec(vi, vj)
                    rik = magvec(vi, vk)
                    A[j-1,k-1] += (V[i, 0]-input[kidx, 1])*(V[i, 0]-input[jidx, 1])/(rij**3 * rik**3)
                    A[k-1,j-1] += (V[i, 0]-input[kidx, 1])*(V[i, 0]-input[jidx, 1])/(rij**3 * rik**3)
            elif j <= (Atoms)*3 and k <= (Atoms)*3:
                # dipole_y
                for i in range(len(V)):
                    vi = [V[i, 0], V[i, 1], V[i, 2]]
                    rij = magvec(vi, vj)
                    rik = magvec(vi, vk)
                    A[j-1,k-1] += (V[i, 1]-input[kidx, 2])*(input[jidx, 2]-V[i, 1])/(rij**3 * rik**3)
                    A[k-1,j-1] += (V[i, 1]-input[kidx, 2])*(input[jidx, 2]-V[i, 1])/(rij**3 * rik**3)
            elif j <= (Atoms)*4 and k <= (Atoms)*4:
                # dipole_z
                for i in range(len(V)):
                    vi = [V[i, 0], V[i, 1], V[i, 2]]
                    rij = magvec(vi, vj)
                    rik = magvec(vi, vk)
                    A[j-1,k-1] += (V[i, 2]-input[kidx, 3])*(input[jidx, 3]-V[i, 2])/(rij**3 * rik**3)
                    A[k-1,j-1] += (V[i, 2]-input[kidx, 3])*(input[jidx, 3]-V[i, 2])/(rij**3 * rik**3)
    
    Bshift = 0
    #Checking constraints in construction of B Vector
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
        
    
    Ashift = 0
    #Checking constraints in construction of A matrix
    if set['Constraint charge'] == 'Yes':
        for j in range(0, len(A)-Constr):
            if j < Atoms:
                # Charge 
                A[len(A)-Constr+Ashift, j] = 1
                A[j, len(A)-Constr+Ashift] = 1
        Ashift += 1

    if set['Constraint dipole'] == 'Yes':
        for i in range(1, len(input)):
            # Charge x
            A[len(A)-Constr+Ashift, i-1] = input[i][1] - Xcm
            A[i-1, len(A)-Constr+Ashift] = input[i][1] - Xcm
        for i in range(1, len(input)):
            # dipole x
            A[len(A)-Constr+Ashift, i-1+Atoms] = 1
            A[i-1+Atoms, len(A)-Constr+Ashift] = 1
        Ashift += 1
        for i in range(1, len(input)):
            # Charge y
            A[len(A)-Constr+Ashift, i-1] = input[i][2] - Ycm
            A[i-1, len(A)-Constr+Ashift] = input[i][2] - Ycm
        for i in range(1, len(input)):
            # dipole y
            A[len(A)-Constr+Ashift, i-1+2*Atoms] = 1
            A[i-1+2*Atoms, len(A)-Constr+Ashift] = 1
        Ashift += 1
        for i in range(1, len(input)):
            # Charge z
            A[len(A)-Constr+Ashift, i-1] = input[i][3] - Zcm
            A[i-1, len(A)-Constr+Ashift] = input[i][3] - Zcm
        for i in range(1, len(input)):
            # dipole z
            A[len(A)-Constr+Ashift, i-1+3*Atoms] = 1
            A[i-1+3*Atoms, len(A)-Constr+Ashift] = 1
        Ashift += 1
    
    
    #Check print of z, x, and y dipole
    if set['Constraint dipole'] == 'Yes':
        check = 0
        if set['Constraint charge'] == 'Yes':
            check = 1
        checksumX = np.sum(np.abs(A[4*Atoms+check,:]))
        checksumY = np.sum(np.abs(A[4*Atoms+check+1,:]))
        checksumZ = np.sum(np.abs(A[4*Atoms+check+2,:]))
    
    #Check A for all zero rows
    for j in range(0, len(A)):
        #Double loop because A changes len when row/coloum is removed
        #If row/coloum that is all 0 the matrix will be singular
        for i in range(0, len(A)):
            checksum = np.sum(np.abs(A[i,:]))
            if checksum == 0:
                A = np.delete(A,i,axis=1)
                A = np.delete(A,i,axis=0)
                B = np.delete(B,i,axis=0)                    
                break
    # END OF building A and B matrix
    
    np.savetxt('A2.txt', A)
    #WORKING EQUATIONS
    Ainv = np.linalg.inv(A)
    q = np.dot(Ainv, B)
    # END OF WORKING EQUATIONS
    
    #Calculate RMSD
    RMSD = 0
    for i in range(len(V)):
        E = 0
        for j in range(0, Atoms):
            rij = ((input[j+1, 1]-V[i, 0])**2+(input[j+1, 2]-V[i, 1])**2+(input[j+1, 3]-V[i, 2])**2)**0.5
            E += q[j]/(rij) + q[j+Atoms]*(V[i, 0]-input[j+1, 1])/(rij)**3 + q[j+2*Atoms]*(V[i, 1]-input[j+1, 2])/(rij)**3 + q[j+3*Atoms]*(V[i, 2]-input[j+1, 3])/(rij)**3 
            
        RMSD += (V[i, 3] - E)**2
    RMSD = (RMSD/len(V))**0.5
    # END OF calculating RMSD
    
    #Write output
    output = open('out.txt', 'a')
    output.write('\n \n')
    output.write('Fitted Charges \n')
    for i in range(1, len(input)):
        output.write('Atom'+str(i)+'\t')
        output.write("{: 10.8f}".format(q[i-1]))
        output.write('\n')
    output.write('Fitted Dipoles \n')
    output.write('\t\t x\t\t\t\t y\t\t\t\t z\n')
    for i in range(1, len(input)):
        output.write('Atom'+str(i)+'\t')
        output.write("{: 10.8f}".format(q[i-1+Atoms]))
        output.write('\t \t')
        output.write("{: 10.8f}".format(q[i-1+2*Atoms]))
        output.write('\t \t')
        output.write("{: 10.8f}".format(q[i-1+3*Atoms]))
        output.write('\n')
    output.write('RMSD \t')
    output.write("{: 12.8e}".format(RMSD))
    output.close()
    # END OF writing output

def runQfit(basis, input, D, set, results):
    if set['Multipolefit'] == 'Charge':
        chrfit(basis, input, D, set, results)
    if set['Multipolefit'] == 'Dipole':
        dipolefit(basis, input, D, set, results)