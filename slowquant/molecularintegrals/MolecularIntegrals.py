import math
import scipy.misc as scm
import numpy as np

##INTEGRAL FUNCTIONS
def nucrep(input):
    #Classical nucleus nucleus repulsion
    Vnn = 0
    for i in range(1, len(input)):
        for j in range(1, len(input)):
            if i < j:
                Vnn += (input[i][0]*input[j][0])/(((input[i][1]-input[j][1])**2+(input[i][2]-input[j][2])**2+(input[i][3]-input[j][3])**2))**0.5
    return Vnn


def nucdiff(input, atomidx, direction):
    # direction 1 = x, 2 = y, 3 = z
    Vnn = 0
    ZA = input[atomidx][0]
    XA = input[atomidx][direction]
    for i in range(1, len(input)):
        if i != atomidx:
            ZB = input[i][0]
            XB = input[i][direction]
            Vnn += ZB*(XB-XA)/((math.sqrt((input[i][1]-input[atomidx][1])**2+(input[i][2]-input[atomidx][2])**2+(input[i][3]-input[atomidx][3])**2))**3)
    
    return ZA*Vnn


def Ndiff1(l,a):
    return ((2*l+1)*a)**0.5

def Ndiff2(l,a):
    return -2*l*(a/(2*l-1))**0.5
        
def Nrun(basisset):
    # Normalize primitive functions
    for i in range(len(basisset)):
        for j in range(len(basisset[i][5])):
            a = basisset[i][5][j][1]
            l = basisset[i][5][j][3]
            m = basisset[i][5][j][4]
            n = basisset[i][5][j][5]
            
            part1 = (2.0/math.pi)**(3.0/4.0)
            part2 = 2.0**(l+m+n) * a**((2.0*l+2.0*m+2.0*n+3.0)/(4.0))
            part3 = math.sqrt(scm.factorial2(int(2*l-1))*scm.factorial2(int(2*m-1))*scm.factorial2(int(2*n-1)))
            basisset[i][5][j][0] = part1 * ((part2)/(part3))
    """
    # Normalize contractions
    for k in range(len(basisset)):
        if len(basisset[k][5]) != 1:
            l = basisset[k][5][0][3]
            m = basisset[k][5][0][4]
            n = basisset[k][5][0][5]
            L = l+m+n
            factor = (np.pi**(3.0/2.0)*scm.factorial2(int(2*l-1))*scm.factorial2(int(2*m-1))*scm.factorial2(int(2*n-1)))/(2.0**L)
            sum = 0
            for i in range(len(basisset[k][5])):
                for j in range(len(basisset[k][5])):
                    alphai = basisset[k][5][i][1]
                    alphaj = basisset[k][5][j][1]
                    ai     = basisset[k][5][i][2]*basisset[k][5][i][0]
                    aj     = basisset[k][5][j][2]*basisset[k][5][j][0]
                    
                    sum += ai*aj/((alphai+alphaj)**(L+3.0/2.0))
            
            Nc = (factor*sum)**(-1.0/2.0)
            for i in range(len(basisset[k][5])):
                basisset[k][5][i][0] *= Nc
    """
    return basisset

def u_ObaraSaika(a1, a2, Ax, Ay, Az, Bx, By, Bz, la, lb, ma, mb, na, nb, N1, N2, c1, c2, input):
    #Used to calculate dipolemoment
    N = N1*N2*c1*c2
    p = a1 + a2
    u = a1*a2/p
    
    Px = (a1*Ax+a2*Bx)/p
    Py = (a1*Ay+a2*By)/p
    Pz = (a1*Az+a2*Bz)/p
    
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
    
    ux000 = (math.pi/p)**(1/2) * math.exp(-u*(Ax-Bx)**2)
    uy000 = (math.pi/p)**(1/2) * math.exp(-u*(Ay-By)**2)
    uz000 = (math.pi/p)**(1/2) * math.exp(-u*(Az-Bz)**2)
    
    ux = np.zeros(shape=(la+2,lb+2, 3))
    uy = np.zeros(shape=(ma+2,mb+2, 3))
    uz = np.zeros(shape=(na+2,nb+2, 3))
    ux[0][0][0] = ux000
    uy[0][0][0] = uy000
    uz[0][0][0] = uz000
    for i in range(0, la+1):
        for j in range(0, lb+1):
            for e in range(0, 2):
                ux[i+1][j][e] = (Px-Ax)*ux[i][j][e] + 1/(2*p) * (i*ux[i-1][j][e]+j*ux[i][j-1][e]+e*ux[i][j][e-1])
                ux[i][j+1][e] = (Px-Bx)*ux[i][j][e] + 1/(2*p) * (i*ux[i-1][j][e]+j*ux[i][j-1][e]+e*ux[i][j][e-1])
                ux[i][j][e+1] = (Px-Cx)*ux[i][j][e] + 1/(2*p) * (i*ux[i-1][j][e]+j*ux[i][j-1][e]+e*ux[i][j][e-1])
    for i in range(0, ma+1):
        for j in range(0, mb+1):
            for e in range(0, 2):
                uy[i+1][j][e] = (Py-Ay)*uy[i][j][e] + 1/(2*p) * (i*uy[i-1][j][e]+j*uy[i][j-1][e]+e*uy[i][j][e-1])
                uy[i][j+1][e] = (Py-By)*uy[i][j][e] + 1/(2*p) * (i*uy[i-1][j][e]+j*uy[i][j-1][e]+e*uy[i][j][e-1])
                uy[i][j][e+1] = (Py-Cy)*uy[i][j][e] + 1/(2*p) * (i*uy[i-1][j][e]+j*uy[i][j-1][e]+e*uy[i][j][e-1])
    for i in range(0, na+1):
        for j in range(0, nb+1):
            for e in range(0, 2):
                uz[i+1][j][e] = (Pz-Az)*uz[i][j][e] + 1/(2*p) * (i*uz[i-1][j][e]+j*uz[i][j-1][e]+e*uz[i][j][e-1])
                uz[i][j+1][e] = (Pz-Bz)*uz[i][j][e] + 1/(2*p) * (i*uz[i-1][j][e]+j*uz[i][j-1][e]+e*uz[i][j][e-1])
                uz[i][j][e+1] = (Pz-Cz)*uz[i][j][e] + 1/(2*p) * (i*uz[i-1][j][e]+j*uz[i][j-1][e]+e*uz[i][j][e-1])
    return -N*ux[la][lb][1]*uy[ma][mb][0]*uz[na][nb][0], -N*ux[la][lb][0]*uy[ma][mb][1]*uz[na][nb][0], -N*ux[la][lb][0]*uy[ma][mb][0]*uz[na][nb][1]