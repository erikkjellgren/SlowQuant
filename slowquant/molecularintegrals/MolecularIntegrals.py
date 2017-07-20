import numpy as np
import scipy.misc as scm
import scipy.special as scs
import math
from numba import jit

##INTEGRAL FUNCTIONS
def Overlap(a, b, la, lb, Ax, Bx):
    #Obara-Saika scheme, 9.3.8 and 9.3.9 Helgaker
    #Used in Kin integral!

    p = a + b
    u = a*b/p
    
    Px = (a*Ax+b*Bx)/p
    
    S00 = (math.pi/p)**(1/2) * math.exp(-u*(Ax-Bx)**2)
    
    Sij = np.zeros(shape=(la+2,lb+2))
    Sij[0,0] = S00
    
    
    for i in range(0, la+1):
        for j in range(0, lb+1):
            Sij[i+1,j] = (Px-Ax)*Sij[i,j] + 1/(2*p) * (i*Sij[i-1,j] + j*Sij[i,j-1])
            Sij[i,j+1] = (Px-Bx)*Sij[i,j] + 1/(2*p) * (i*Sij[i-1,j] + j*Sij[i,j-1])
    
    return Sij

def Kin(a, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma, mb, na, nb, N1, N2, c1, c2):
    #Obara-Saika scheme, 9.3.40 and 9.3.41 Helgaker
    # Calculates electronic kinetic energy and overlap integrals
    #    at the same time
    
    p = a + b
    N = N1*N2*c1*c2
    Px = (a*Ax+b*Bx)/p
    Py = (a*Ay+b*By)/p
    Pz = (a*Az+b*Bz)/p
    XPA = Px - Ax
    YPA = Py - Ay
    ZPA = Pz - Az
    XPB = Px - Bx
    YPB = Py - By
    ZPB = Pz - Bz
    
    Tijx = np.zeros(shape=(la+2,lb+2))
    Tijy = np.zeros(shape=(ma+2,mb+2))
    Tijz = np.zeros(shape=(na+2,nb+2))
    Sx = Overlap(a, b, la, lb, Ax, Bx)
    Sy = Overlap(a, b, ma, mb, Ay, By)
    Sz = Overlap(a, b, na, nb, Az, Bz)
    Tijx[0,0] = (a-2*a**2*(XPA**2+1/(2*p)))*Sx[0,0]
    Tijy[0,0] = (a-2*a**2*(YPA**2+1/(2*p)))*Sy[0,0]
    Tijz[0,0] = (a-2*a**2*(ZPA**2+1/(2*p)))*Sz[0,0]
    
    for i in range(0, la+1):
        for j in range(0, lb+1):
            Tijx[i+1,j] = XPA*Tijx[i,j] + 1/(2*p)*(i*Tijx[i-1,j]+j*Tijx[i,j-1]) + b/p*(2*a*Sx[i+1,j] - i*Sx[i-1,j])
            Tijx[i,j+1] = XPB*Tijx[i,j] + 1/(2*p)*(i*Tijx[i-1,j]+j*Tijx[i,j-1]) + a/p*(2*b*Sx[i,j+1] - j*Sx[i,j-1])
    
    for i in range(0, ma+1):
        for j in range(0, mb+1):
            Tijy[i+1,j] = YPA*Tijy[i,j] + 1/(2*p)*(i*Tijy[i-1,j]+j*Tijy[i,j-1]) + b/p*(2*a*Sy[i+1,j] - i*Sy[i-1,j])
            Tijy[i,j+1] = YPB*Tijy[i,j] + 1/(2*p)*(i*Tijy[i-1,j]+j*Tijy[i,j-1]) + a/p*(2*b*Sy[i,j+1] - j*Sy[i,j-1])
    
    for i in range(0, na+1):
        for j in range(0, nb+1):
            Tijz[i+1,j] = ZPA*Tijz[i,j] + 1/(2*p)*(i*Tijz[i-1,j]+j*Tijz[i,j-1]) + b/p*(2*a*Sz[i+1,j] - i*Sz[i-1,j])
            Tijz[i,j+1] = ZPB*Tijz[i,j] + 1/(2*p)*(i*Tijz[i-1,j]+j*Tijz[i,j-1]) + a/p*(2*b*Sz[i,j+1] - j*Sz[i,j-1])
    
    return (Tijx[la, lb]*Sy[ma,mb]*Sz[na,nb]+Tijy[ma, mb]*Sx[la,lb]*Sz[na,nb]+Tijz[na, nb]*Sy[ma,mb]*Sx[la,lb])*N, Sx[la, lb]*Sy[ma, mb]*Sz[na, nb]*N


def elnuc(P, p, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, Zc, C, Ex, Ey, Ez):
    #McMurchie-Davidson scheme    
    N = N1*N2*c1*c2
    
    RPC = ((P[0]-C[0])**2+(P[1]-C[1])**2+(P[2]-C[2])**2)**0.5
    Rpre = np.ones((l1+l2+2,m1+m2+2,n1+n2+2,l1+l2+m1+m2+n1+n2+6))
    
    val = 0
    for t in range(0, l1+l2+1):
        for u in range(0, m1+m2+1):
            for v in range(0, n1+n2+1):
                R1, Rpre = R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC,Rpre)
                val += Ex[t]*Ey[u]*Ez[v]*R1*Zc

    return -val*2*np.pi/p*N

def nucrep(input):
    #Classical nucleus nucleus repulsion
    Vnn = 0
    for i in range(1, len(input)):
        for j in range(1, len(input)):
            if i < j:
                Vnn += (input[i][0]*input[j][0])/(math.sqrt((input[i][1]-input[j][1])**2+(input[i][2]-input[j][2])**2+(input[i][3]-input[j][3])**2))
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
    
    
def electricfield(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, input, derivative, atomidx):
    #McMurchie-Davidson scheme    
    N = N1*N2*c1*c2
    p = a + b
    
    A = np.array([Ax, Ay, Az])
    B = np.array([Bx, By, Bz])
    P = (a*A+b*B)/p
    
    dx = 0
    dy = 0
    dz = 0
    if derivative == 'x':
        dx = 1
    elif derivative == 'y':
        dy = 1
    else:
        dz = 1

    val = 0
    Zc = input[atomidx][0]
    C = np.array([input[atomidx][1],input[atomidx][2],input[atomidx][3]])
    RPC = ((P[0]-C[0])**2+(P[1]-C[1])**2+(P[2]-C[2])**2)**0.5
    Rpre = np.ones((l1+l2+2,m1+m2+2,n1+n2+2,l1+l2+m1+m2+n1+n2+6))
    
    for t in range(0, l1+l2+1):
        Ex = E(l1,l2,t,A[0]-B[0],a,b,P[0]-A[0],P[0]-B[0],A[0]-B[0])
        for u in range(0, m1+m2+1):
            Ey = E(m1,m2,u,A[1]-B[1],a,b,P[1]-A[1],P[1]-B[1],A[1]-B[1])
            for v in range(0, n1+n2+1):
                Ez = E(n1,n2,v,A[2]-B[2],a,b,P[2]-A[2],P[2]-B[2],A[2]-B[2])
                R1, Rpre = R(t+dx,u+dy,v+dz,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC,Rpre)
                val += Ex*Ey*Ez*R1*Zc
    
    return val*2*np.pi/p*N

@jit
def elelrep(p,q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, N3, N4, c1, c2, c3, c4, E1, E2, E3, E4, E5, E6, Rpre):
    # #############################################################
    #
    # Used in ERI for speed up should later be used
    # for all MacMurchie davidson integrals to replace elelrep()
    #
    # #############################################################
    
    #McMurchie-Davidson scheme   
    N = N1*N2*N3*N4*c1*c2*c3*c4
    
    val = 0.0
    for tau in range(l3+l4+1):
        for nu in range(m3+m4+1):
            for phi in range(n3+n4+1):
                factor = np.power(-1,tau+nu+phi)
                for t in range(l1+l2+1):
                    for u in range(m1+m2+1):
                        for v in range(n1+n2+1):
                            val += E1[t]*E2[u]*E3[v]*E4[tau]*E5[nu]*E6[phi]*Rpre[t+tau,u+nu,v+phi]*factor


    val *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q)) 
    return val*N
    
##UTILITY FUNCTIONS
def E(i,j,t,Qx,a,b,XPA,XPB,XAB):
    #McMurchie-Davidson scheme, 9.5.6 and 9.5.7 Helgaker
    p = a + b
    q = a*b/p
    if (t < 0) or (t > (i + j)):
        return 0.0
    elif i == j == t == 0:
        return np.exp(-q*Qx*Qx)
    elif j == 0:
        return (1/(2*p))*E(i-1,j,t-1,Qx,a,b,XPA,XPB,XAB) + XPA*E(i-1,j,t,Qx,a,b,XPA,XPB,XAB) + (t+1)*E(i-1,j,t+1,Qx,a,b,XPA,XPB,XAB)
    else:
        return (1/(2*p))*E(i,j-1,t-1,Qx,a,b,XPA,XPB,XAB) + XPB*E(i,j-1,t,Qx,a,b,XPA,XPB,XAB) + (t+1)*E(i,j-1,t+1,Qx,a,b,XPA,XPB,XAB)


def R(t,u,v,n,p,PCx,PCy,PCz,RPC, Rpre):
    # #############################################################
    #
    # Used in ERI for speed up at p or higher orbitals
    #
    # #############################################################
    
    #McMurchie-Davidson scheme, 9.9.18, 9.9.19 and 9.9.20 Helgaker
    if Rpre[t,u,v,n] == 1.0:
        T = p*RPC*RPC
        val = 0.0
        if t == u == v == 0:
            val += np.power(-2*p,n)*boys(n,T)
        elif t == u == 0:
            if v > 1:
                res,Rpre = R(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC,Rpre)
                val+=(v-1)*res
            res,Rpre = R(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC,Rpre)  
            val+=PCz*res
        elif t == 0:
            if u > 1:
                res,Rpre = R(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC,Rpre) 
                val+=(u-1)*res  
            res,Rpre = R(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC,Rpre)  
            val+=PCy*res
        else:
            if t > 1:
                res,Rpre = R(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC,Rpre)  
                val+=(t-1)*res
            res,Rpre = R(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC,Rpre)  
            val+=PCx*res
        
        Rpre[t,u,v,n] = val
        return Rpre[t,u,v,n] , Rpre
    else:
        return Rpre[t,u,v,n] , Rpre
    

def boys(m,T):
    #Boys functions
    if abs(T) < 1e-12:
        return 1/(2*m + 1)
    else:
        #return scs.gammainc(m+0.5,T)*scs.gamma(m+0.5)/(2*np.power(T,m+0.5))
        return scs.hyp1f1(m+0.5,m+1.5,-T)/(2.0*m+1.0)
        

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

def Eprecalculation(basis):
    # #############################################################
    #
    # Precalculation of the expansion coefficients, used in 
    # MacMurchie davidson ERI. Speed up compared to on the fly
    # calculation. Should later be used for all MacMurchie davidson integrals
    #
    # #############################################################
    Edict  = {}
    GPdict = {}
    pdict  = {}
    for k in range(0, len(basis)):
        for l in range(0, len(basis)):
            if k >= l:
                idx = np.zeros(2)
                idx[0] = k
                idx[1] = l
                idx = idx.astype(int)
                for i in range(basis[idx[0]][4]):
                    for j in range(basis[idx[1]][4]):
                        a=basis[idx[0]][5][i][1]
                        b=basis[idx[1]][5][j][1]
                        Ax=basis[idx[0]][1]
                        Ay=basis[idx[0]][2]
                        Az=basis[idx[0]][3]
                        Bx=basis[idx[1]][1]
                        By=basis[idx[1]][2]
                        Bz=basis[idx[1]][3]
                        la=basis[idx[0]][5][i][3]
                        lb=basis[idx[1]][5][j][3]
                        ma=basis[idx[0]][5][i][4]
                        mb=basis[idx[1]][5][j][4]
                        na=basis[idx[0]][5][i][5]
                        nb=basis[idx[1]][5][j][5]
                        
                        A = np.array([Ax, Ay, Az])
                        B = np.array([Bx, By, Bz])
                        
                        p = a+b
                        P = (a*A+b*B)/p
                        
                        E1 = np.zeros(la+lb+1)
                        E2 = np.zeros(ma+mb+1)
                        E3 = np.zeros(na+nb+1)
                        
                        for t in range(la+lb+1):
                            E1[t] = E(la,lb,t,A[0]-B[0],a,b,P[0]-A[0],P[0]-B[0],A[0]-B[0])
                        for u in range(ma+mb+1):
                            E2[u] = E(ma,mb,u,A[1]-B[1],a,b,P[1]-A[1],P[1]-B[1],A[1]-B[1])
                        for v in range(na+nb+1):
                            E3[v] = E(na,nb,v,A[2]-B[2],a,b,P[2]-A[2],P[2]-B[2],A[2]-B[2])
                        
                        Edict[str(k)+str(l)+str(i)+str(j)+'E1'] = E1
                        Edict[str(k)+str(l)+str(i)+str(j)+'E2'] = E2
                        Edict[str(k)+str(l)+str(i)+str(j)+'E3'] = E3
                        GPdict[str(k)+str(l)+str(i)+str(j)]     = P
                        pdict[str(k)+str(l)+str(i)+str(j)]      = p
                        
    return Edict, GPdict, pdict

