import numpy as np
import MolecularIntegrals as MI
import random as rng

#### DIPOLE FIT NOT WORKING
#### UNKNOWN IF DIPOLE CONSTRAIN WORK FOR CHARGE FITTING

def magvec(v1, v2):
    x = v2[0] - v1[0]
    y = v2[1] - v1[1]
    z = v2[2] - v1[2]
    return (x**2+y**2+z**2)**0.5

def centerofcharge(input):
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
    return Xcm, Ycm, Zcm

def makepoints(Xcm, Ycm, Zcm, set, input):
    meshtot = int(set['Gridpoints'])
    # [x, y, z, VQM]
    V = np.zeros((meshtot,5))
    idx = 0
    radius= 3.4 + np.sqrt(np.max((Xcm-input[1:,1])**2)+np.max((Ycm-input[1:,2])**2)+np.max((Zcm-input[1:,3])**2))
    for i in range(0, meshtot):
        z = rng.uniform(-radius,radius)
        phi = rng.uniform(0, 2*np.pi)
        x = (radius**2-z**2)**0.5*np.cos(phi)
        y = (radius**2-z**2)**0.5*np.sin(phi)
        V[idx, 0] = x + Xcm
        V[idx, 1] = y + Ycm
        V[idx, 2] = z + Zcm
        idx += 1
    #V = np.array([[-0.19523982, -5.11636437, -1.74206068,  0,          0,        ],[ 3.78732559, -0.63947779, -4.09453905,  0,          0,        ]])
    return V

def RMSDcalc(V, q, Constr, input, Atoms, set):
    RMSD = 0
    for i in range(len(V)):
        E = 0
        for j in range(0, Atoms):
            rij = ((input[j+1, 1]-V[i, 0])**2+(input[j+1, 2]-V[i, 1])**2+(input[j+1, 3]-V[i, 2])**2)**0.5
            if set['Multipolefit'] == 'Charge':
                E += q[j]/(rij)
            elif set['Multipolefit'] == 'Dipole':
                E += q[j]/(rij) + q[j+Atoms]*(V[i, 0]-input[j+1, 1])/(rij)**3 + q[j+2*Atoms]*(V[i, 1]-input[j+1, 2])/(rij)**3 + q[j+3*Atoms]*(V[i, 2]-input[j+1, 3])/(rij)**3 
        V[i,4] = V[i, 3] - E #Difference between QM and classical potential
        RMSD += (V[i, 3] - E)**2
    RMSD = (RMSD/len(V))**0.5
    return RMSD, V
    

## CHARGE FIT
def chrfit(basis, input, D, set, results):
    
    # CHOOSE POINTS
    Atoms = len(input)-1
    Xcm, Ycm, Zcm = centerofcharge(input)
    V = makepoints(Xcm, Ycm, Zcm, set, input)
    
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
    RMSD, V = RMSDcalc(V, q, Constr, input, Atoms, set)
    
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
    if set['Write ESP'] == 'Yes':
        potentialPBD(V, input)


## CHARGE AND DIPOLE FIT
def dipolefit(basis, input, D, set, results):
    
    # CHOOSE POINTS
    Atoms = len(input)-1
    Xcm, Ycm, Zcm = centerofcharge(input)
    V = makepoints(Xcm, Ycm, Zcm, set, input)
    
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
    for k in range(0, Atoms):
        vk = np.array([input[k+1,1],input[k+1,2],input[k+1,3]])
        for i in range(len(V)):
            vi = np.array([V[i,0],V[i,1],V[i,2]])
            rik = magvec(vi, vk)
            rikx = vi[0] - vk[0]
            riky = vi[1] - vk[1]
            rikz = vi[2] - vk[2]
            
            B[k] += V[i,3]/rik
            B[k+Atoms] += rikx*V[i,3]/(rik**3)
            B[k+2*Atoms] += riky*V[i,3]/(rik**3)
            B[k+3*Atoms] += rikz*V[i,3]/(rik**3)
    
    #Assign values to A matrix
    for j in range(0, Atoms):
        for k in range(0, Atoms):
            vj = np.array([input[j+1, 1], input[j+1, 2], input[j+1, 3]])
            vk = np.array([input[k+1, 1], input[k+1, 2], input[k+1, 3]])
            for i in range(len(V)):
                vi = np.array([V[i, 0], V[i, 1], V[i, 2]])
                rij = magvec(vi, vj)
                rik = magvec(vi, vk)
                rikx = vi[0] - vk[0]
                riky = vi[1] - vk[1]
                rikz = vi[2] - vk[2]
                rijx = vi[0] - vj[0]
                rijy = vi[1] - vj[1]
                rijz = vi[2] - vj[2]
                # Charge 
                A[j,k] += 1/(rij*rik)
                # dipole_x charge
                A[j,k+Atoms] += rijx/(rij**3 * rik)
                A[j+Atoms,k] += rikx/(rik**3 * rij)
                # dipole_y charge
                A[j,k+2*Atoms] += rijy/(rij**3 * rik)
                A[j+2*Atoms,k] += riky/(rik**3 * rij)
                # dipole_z charge
                A[j,k+3*Atoms] += rijz/(rij**3 * rik)
                A[j+3*Atoms,k] += rikz/(rik**3 * rij)
                # dipole_x  
                A[j+Atoms,k+Atoms] += rikx*rijx/(rij**3 * rik**3)
                # dipole_y
                A[j+2*Atoms,k+2*Atoms] += riky*rijy/(rij**3 * rik**3)
                # dipole_z
                A[j+3*Atoms,k+3*Atoms] += rikz*rijz/(rij**3 * rik**3)
    
    
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
    np.savetxt('B2.txt', B)
    #WORKING EQUATIONS
    Ainv = np.linalg.inv(A)
    q = np.dot(Ainv, B)
    # END OF WORKING EQUATIONS
    
    #Calculate RMSD
    RMSD, V = RMSDcalc(V, q, Constr, input, Atoms, set)
    
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
    if set['Write ESP'] == 'Yes':
        potentialPBD(V, input)

def potentialPBD(V, input):
    #Write potential to pdb
    f = open('potential.pdb', 'w+')
    idx = 1
    for i in range(1, len(input)):
        f.write('{:>6}'.format('HETATM'))
        f.write('{:>5}'.format(str(idx)))
        f.write('{:>1}'.format(' '))
        if input[i,0] == 1:
            f.write('{:>4}'.format('H'))
        elif input[i,0] == 6:
            f.write('{:>4}'.format('C'))
        elif input[i,0] == 8:
            f.write('{:>4}'.format('O'))
        f.write('{:>1}'.format(' '))
        f.write('{:>3}'.format('MOL'))
        f.write('{:>1}'.format(' '))
        f.write('{:>1}'.format(' '))
        f.write('{:>4}'.format(' '))
        f.write('{:>1}'.format(' '))
        f.write('{:>3}'.format(' '))
        f.write('{: 8.3f}'.format(input[i,1]))
        f.write('{: 8.3f}'.format(input[i,2]))
        f.write('{: 8.3f}'.format(input[i,3]))
        f.write('{: 6.2f}'.format(1))
        f.write('{: 6.2f}'.format(1))
        f.write('{:>6}'.format(' '))
        f.write('{:>4}'.format(' '))
        f.write('{:>2}'.format(' '))
        f.write('{:>2}'.format(' '))
        f.write('\n')
        idx += 1
    for i in range(0, len(V)):
        f.write('{:>6}'.format('HETATM')) #Record name     "HETATM" 
        f.write('{:>5}'.format(str(idx))) #Integer         Atom serial number. 
        f.write('{:>1}'.format(' '))      #Blanck space
        f.write('{:>4}'.format('He'))     #Atom            Atom name    
        f.write('{:>1}'.format(' '))      #Character       Alternate location indicator 
        f.write('{:>3}'.format('POT'))    #Residue name    Residue name 
        f.write('{:>1}'.format(' '))      #Blanck space
        f.write('{:>1}'.format(' '))      #Character       Chain identifier 
        f.write('{:>4}'.format(' '))      #Integer         Residue sequence number   
        f.write('{:>1}'.format(' '))      #AChar           Code for insertion of residues 
        f.write('{:>3}'.format(' '))      #Blank space
        f.write('{: 8.3f}'.format(V[i,0]))#Real(8.3)       Orthogonal coordinates for X    
        f.write('{: 8.3f}'.format(V[i,1]))#Real(8.3)       Orthogonal coordinates for Y   
        f.write('{: 8.3f}'.format(V[i,2]))#Real(8.3)       Orthogonal coordinates for Z   
        f.write('{: 6.2f}'.format(V[i,3]*100))#Real(6.2)       Occupancy 
        f.write('{: 6.2f}'.format(V[i,4]*10000))#Real(6.2)       Temperature factor 
        f.write('{:>6}'.format(' ')) #Blank space
        f.write('{:>4}'.format(' ')) #LString(4)      Segment identifier, left-justified  
        f.write('{:>2}'.format(' ')) #LString(2)      Element symbol, right-justified 
        f.write('{:>2}'.format(' ')) #LString(2)      Charge on the atom 
        f.write('\n')
        idx += 1
    f.close()
    

def runQfit(basis, input, D, set, results):
    if set['Multipolefit'] == 'Charge':
        chrfit(basis, input, D, set, results)
    if set['Multipolefit'] == 'Dipole':
        dipolefit(basis, input, D, set, results)