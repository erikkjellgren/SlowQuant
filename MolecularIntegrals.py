import math
import numpy as np
import scipy.misc as scm
import scipy.special as scs
import time as time
import copy

##INTEGRAL FUNCTIONS
def Overlap(a, b, la, lb, Ax, Bx):
    #Obara-Saika scheme, 9.3.8 and 9.3.9 Helgaker
    #Used in Kin integral!, should be linked with the other Overlap, 
    #    so it is not double calculated
    
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


def elnuc(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, input):
    #McMurchie-Davidson scheme    
    N = N1*N2*c1*c2
    p = a + b
    
    A = np.array([Ax, Ay, Az])
    B = np.array([Bx, By, Bz])
    P = gaussian_product_center(a,A,b,B)
    
    
    val = 0
    for k in range(1, len(input)):
        Zc = input[k][0]
        C = np.array([input[k][1],input[k][2],input[k][3]])
        RPC = np.linalg.norm(P-C)
        
        for t in range(0, l1+l2+1):
            Ex = E(l1,l2,t,A[0]-B[0],a,b,P[0]-A[0],P[0]-B[0],A[0]-B[0])
            for u in range(0, m1+m2+1):
                Ey = E(m1,m2,u,A[1]-B[1],a,b,P[1]-A[1],P[1]-B[1],A[1]-B[1])
                for v in range(0, n1+n2+1):
                    Ez = E(n1,n2,v,A[2]-B[2],a,b,P[2]-A[2],P[2]-B[2],A[2]-B[2])
                    val += Ex*Ey*Ez*R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC)*Zc

    return -val*2*np.pi/p*N

def elelrep(a, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, N3, N4, c1, c2, c3, c4):
    #McMurchie-Davidson scheme   
    N = N1*N2*N3*N4*c1*c2*c3*c4
    A = np.array([Ax, Ay, Az])
    B = np.array([Bx, By, Bz])
    C = np.array([Cx, Cy, Cz])
    D = np.array([Dx, Dy, Dz])
    
    p = a+b
    q = c+d
    alpha = p*q/(p+q)
    P = gaussian_product_center(a,A,b,B)
    Q = gaussian_product_center(c,C,d,D)
    RPQ = np.linalg.norm(P-Q)

    E2 = np.zeros(m1+m2+1)
    E3 = np.zeros(n1+n2+1)
    E4 = np.zeros(l3+l4+1)
    E5 = np.zeros(m3+m4+1)
    E6 = np.zeros(n3+n4+1)
    
    for u in range(m1+m2+1):
        E2[u] = E(m1,m2,u,A[1]-B[1],a,b,P[1]-A[1],P[1]-B[1],A[1]-B[1])
    for v in range(n1+n2+1):
        E3[v] = E(n1,n2,v,A[2]-B[2],a,b,P[2]-A[2],P[2]-B[2],A[2]-B[2])
    for tau in range(l3+l4+1):
        E4[tau] = E(l3,l4,tau,C[0]-D[0],c,d,Q[0]-C[0],Q[0]-D[0],C[0]-D[0])
    for nu in range(m3+m4+1):
        E5[nu] = E(m3,m4,nu ,C[1]-D[1],c,d,Q[1]-C[1],Q[1]-D[1],C[1]-D[1])
    for phi in range(n3+n4+1):
        E6[phi] = E(n3,n4,phi,C[2]-D[2],c,d,Q[2]-C[2],Q[2]-D[2],C[2]-D[2])

    val = 0.0
    for t in range(l1+l2+1):
        E1 = E(l1,l2,t,A[0]-B[0],a,b,P[0]-A[0],P[0]-B[0],A[0]-B[0])
        for u in range(m1+m2+1):
            for v in range(n1+n2+1):
                for tau in range(l3+l4+1):
                    for nu in range(m3+m4+1):
                        for phi in range(n3+n4+1):
                            R1 = R(t+tau,u+nu,v+phi,0,alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ) 
                            val += E1*E2[u]*E3[v]*E4[tau]*E5[nu]*E6[phi]*np.power(-1,tau+nu+phi)*R1

    val *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q)) 
    return val*N

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
    P = gaussian_product_center(a,A,b,B)
    
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
    RPC = np.linalg.norm(P-C)
    
    for t in range(0, l1+l2+1):
        Ex = E(l1,l2,t,A[0]-B[0],a,b,P[0]-A[0],P[0]-B[0],A[0]-B[0])
        for u in range(0, m1+m2+1):
            Ey = E(m1,m2,u,A[1]-B[1],a,b,P[1]-A[1],P[1]-B[1],A[1]-B[1])
            for v in range(0, n1+n2+1):
                Ez = E(n1,n2,v,A[2]-B[2],a,b,P[2]-A[2],P[2]-B[2],A[2]-B[2])
                val += Ex*Ey*Ez*R(t+dx,u+dy,v+dz,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC)*Zc
    
    
    return val*2*np.pi/p*N


def elnucExclude(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, input, atomidx):
    #McMurchie-Davidson scheme    
    N = N1*N2*c1*c2
    p = a + b
    
    A = np.array([Ax, Ay, Az])
    B = np.array([Bx, By, Bz])
    P = gaussian_product_center(a,A,b,B)
    
    
    val = 0
    for k in range(1, len(input)):
        if k != atomidx:
            Zc = input[k][0]
            C = np.array([input[k][1],input[k][2],input[k][3]])
            RPC = np.linalg.norm(P-C)
            
            for t in range(0, l1+l2+1):
                Ex = E(l1,l2,t,A[0]-B[0],a,b,P[0]-A[0],P[0]-B[0],A[0]-B[0])
                for u in range(0, m1+m2+1):
                    Ey = E(m1,m2,u,A[1]-B[1],a,b,P[1]-A[1],P[1]-B[1],A[1]-B[1])
                    for v in range(0, n1+n2+1):
                        Ez = E(n1,n2,v,A[2]-B[2],a,b,P[2]-A[2],P[2]-B[2],A[2]-B[2])
                        val += Ex*Ey*Ez*R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC)*Zc

    return -val*2*np.pi/p*N
    
    
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
        

def R(t,u,v,n,p,PCx,PCy,PCz,RPC):
    #McMurchie-Davidson scheme, 9.9.18, 9.9.19 and 9.9.20 Helgaker
    T = p*RPC*RPC
    val = 0.0
    if t == u == v == 0:
        val += np.power(-2*p,n)*boys(n,T)
    elif t == u == 0:
        if v > 1:
            val += (v-1)*R(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC)  
        val += PCz*R(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC)
    elif t == 0:
        if u > 1:
            val += (u-1)*R(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC)  
        val += PCy*R(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC)
    else:
        if t > 1:
            val += (t-1)*R(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC)  
        val += PCx*R(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
    return val

def boys(m,T):
    #Boys functions
    if abs(T) < 1e-12:
        return 1/(2*m + 1)
    else:
        return scs.gammainc(m+0.5,T)*scs.gamma(m+0.5)/(2*np.power(T,m+0.5))

def gaussian_product_center(a,A,b,B):
    return (a*A+b*B)/(a+b)

def Ndiff1(l,a):
    return ((2*l+1)*a)**0.5

def Ndiff2(l,a):
    return -2*l*(a/(2*l-1))**0.5

def N(a, l, m, n):
    part1 = (2.0/math.pi)**(3.0/4.0)
    part2 = 2.0**(l+m+n) * a**((2.0*l+2.0*m+2.0*n+3.0)/(4.0))
    part3 = math.sqrt(scm.factorial2(int(2*l-1))*scm.factorial2(int(2*m-1))*scm.factorial2(int(2*n-1)))
    N = part1 * ((part2)/(part3))
    return N


##CALC OF INTEGRALS
def runIntegrals(input, basis):
    # Nuclear-nuclear repulsion
    E = np.zeros(1)
    E[0] = nucrep(input)
    np.save('enuc.npy',E)
    #END OF nuclear-nuclear repulsion
    
    # Two electron integrals
    start = time.time()
    ERI = np.zeros((len(basis),len(basis),len(basis),len(basis)))
    for mu in range(0, len(basis)):
        for nu in range(0, len(basis)):
            if mu >= nu:
                for lam in range(0, len(basis)):
                    for sig in range(0, len(basis)):
                        munu = mu*(mu+1)/2+nu
                        lamsig = lam*(lam+1)/2+sig
                        if lam >= sig and munu >= lamsig:
                            a = np.zeros(4)
                            a[0] = mu
                            a[1] = nu
                            a[2] = lam
                            a[3] = sig
                            a = a.astype(int)
                            calc = 0
                            for i in range(basis[a[0]][4]):
                                for j in range(basis[a[1]][4]):
                                    for k in range(basis[a[2]][4]):
                                        for l in range(basis[a[3]][4]):
                                            calc += elelrep(basis[a[0]][5][i][1], basis[a[1]][5][j][1], basis[a[2]][5][k][1], basis[a[3]][5][l][1], basis[a[0]][1], basis[a[0]][2], basis[a[0]][3], basis[a[1]][1], basis[a[1]][2], basis[a[1]][3], basis[a[2]][1], basis[a[2]][2], basis[a[2]][3], basis[a[3]][1], basis[a[3]][2], basis[a[3]][3], basis[a[0]][5][i][3], basis[a[1]][5][j][3], basis[a[2]][5][k][3], basis[a[3]][5][l][3], basis[a[0]][5][i][4], basis[a[1]][5][j][4], basis[a[2]][5][k][4], basis[a[3]][5][l][4], basis[a[0]][5][i][5], basis[a[1]][5][j][5], basis[a[2]][5][k][5], basis[a[3]][5][l][5], basis[a[0]][5][i][0], basis[a[1]][5][j][0], basis[a[2]][5][k][0], basis[a[3]][5][l][0], basis[a[0]][5][i][2], basis[a[1]][5][j][2], basis[a[2]][5][k][2], basis[a[3]][5][l][2])
                            ERI[mu,nu,lam,sig] = calc
                            ERI[nu,mu,lam,sig] = calc
                            ERI[mu,nu,sig,lam] = calc
                            ERI[nu,mu,sig,lam] = calc
                            ERI[lam,sig,mu,nu] = calc
                            ERI[sig,lam,mu,nu] = calc
                            ERI[lam,sig,nu,mu] = calc
                            ERI[sig,lam,nu,mu] = calc
    np.save('twoint.npy',ERI)
    print(time.time()-start, 'ERI')
    #END OF two electron integrals
    
    # Kinetic energy and overlap
    start = time.time()
    S = np.zeros((len(basis),len(basis)))
    T = np.zeros((len(basis),len(basis)))
    for k in range(0, len(basis)):
        for l in range(0, len(basis)):
            if k >= l:
                a = np.zeros(2)
                a[0] = k
                a[1] = l
                a = a.astype(int)
                calc = 0
                calc2 = 0
                for i in range(basis[a[0]][4]):
                    for j in range(basis[a[1]][4]):
                        calct, calct2 = Kin(basis[a[0]][5][i][1], basis[a[1]][5][j][1], basis[a[0]][1], basis[a[0]][2], basis[a[0]][3], basis[a[1]][1], basis[a[1]][2], basis[a[1]][3], basis[a[0]][5][i][3], basis[a[1]][5][j][3], basis[a[0]][5][i][4], basis[a[1]][5][j][4],basis[a[0]][5][i][5], basis[a[1]][5][j][5], basis[a[0]][5][i][0], basis[a[1]][5][j][0], basis[a[0]][5][i][2], basis[a[1]][5][j][2])
                        calc += calct
                        calc2 += calct2
                S[k,l] = calc2
                S[l,k] = calc2
                T[k,l] = calc
                T[l,k] = calc
    np.save('overlap.npy',S)
    np.save('Ekin.npy',T)
    print(time.time()-start, 'Overlap + kin')
    #END OF kinetic energy and overlap
    
    # Nucleus electron attraction
    start = time.time()
    Na = np.zeros((len(basis),len(basis)))
    for k in range(0, len(basis)):
        for l in range(0, len(basis)):
            if k >= l:
                a = np.zeros(2)
                a[0] = k
                a[1] = l
                a = a.astype(int)
                calc = 0
                for i in range(basis[a[0]][4]):
                    for j in range(basis[a[1]][4]):
                        calc += elnuc(basis[a[0]][5][i][1], basis[a[1]][5][j][1], basis[a[0]][1], basis[a[0]][2], basis[a[0]][3], basis[a[1]][1], basis[a[1]][2], basis[a[1]][3], basis[a[0]][5][i][3], basis[a[1]][5][j][3], basis[a[0]][5][i][4], basis[a[1]][5][j][4],basis[a[0]][5][i][5], basis[a[1]][5][j][5], basis[a[0]][5][i][0], basis[a[1]][5][j][0], basis[a[0]][5][i][2], basis[a[1]][5][j][2], input)
                Na[k,l] = calc
                Na[l,k] = calc
    np.save('nucatt.npy',Na)
    print(time.time()-start, 'Nuc att')
    #END OF nucleus electron attraction
    
def run_dipole_int(basis, input):
    X = np.zeros((len(basis),len(basis)))
    Y = np.zeros((len(basis),len(basis)))
    Z = np.zeros((len(basis),len(basis)))
    for k in range(0, len(basis)):
        for l in range(0, len(basis)):
            if k >= l:
                a = np.zeros(2)
                a[0] = k
                a[1] = l
                a = a.astype(int)
                calcx = 0
                calcy = 0
                calcz = 0
                for i in range(basis[a[0]][4]):
                    for j in range(basis[a[1]][4]):
                        x, y, z = u_ObaraSaika(basis[a[0]][5][i][1], basis[a[1]][5][j][1], basis[a[0]][1], basis[a[0]][2], basis[a[0]][3], basis[a[1]][1], basis[a[1]][2], basis[a[1]][3], basis[a[0]][5][i][3], basis[a[1]][5][j][3], basis[a[0]][5][i][4], basis[a[1]][5][j][4],basis[a[0]][5][i][5], basis[a[1]][5][j][5], basis[a[0]][5][i][0], basis[a[1]][5][j][0], basis[a[0]][5][i][2], basis[a[1]][5][j][2], input)
                        calcx += x
                        calcy += y
                        calcz += z
                X[k,l] = calcx
                X[l,k] = calcx
                Y[k,l] = calcy
                Y[l,k] = calcy
                Z[k,l] = calcz
                Z[l,k] = calcz
    np.save('mux.npy',X)
    np.save('muy.npy',Y)
    np.save('muz.npy',Z)

def runQMESP(basis, input, rcx, rcy ,rcz):
    # Set up indexes for integrals
    Ve = np.zeros((len(basis),len(basis)))
    point = np.array([['None''None','None','None'],[-1, rcx, rcy ,rcz]])
    for k in range(0, len(basis)):
        for l in range(0, len(basis)):
            if k >= l:
                a = np.zeros(2)
                a[0] = k
                a[1] = l
                a = a.astype(int)
                calc = 0
                for i in range(basis[a[0]][4]):
                    for j in range(basis[a[1]][4]):
                        calc += elnuc(basis[a[0]][5][i][1], basis[a[1]][5][j][1], basis[a[0]][1], basis[a[0]][2], basis[a[0]][3], basis[a[1]][1], basis[a[1]][2], basis[a[1]][3], basis[a[0]][5][i][3], basis[a[1]][5][j][3], basis[a[0]][5][i][4], basis[a[1]][5][j][4],basis[a[0]][5][i][5], basis[a[1]][5][j][5], basis[a[0]][5][i][0], basis[a[1]][5][j][0], basis[a[0]][5][i][2], basis[a[1]][5][j][2], point)
            Ve[k,l] = calc
            Ve[l,k] = calc
        
    return Ve

def rungeometric_derivatives(input, basis):
    # Calculating the normalization of the derivatives. For now only used in ERI
    # Thanks to Tom for suggesting copy.deepcopy
    Nxplus  = copy.deepcopy(basis)
    Nxminus = copy.deepcopy(basis)
    Nyplus  = copy.deepcopy(basis)
    Nyminus = copy.deepcopy(basis)
    Nzplus  = copy.deepcopy(basis)
    Nzminus = copy.deepcopy(basis)
    for i in range(len(basis)):
        for j in range(len(basis[i][5])):
            Nxplus[i][5][j][0] = N(Nxplus[i][5][j][1], Nxplus[i][5][j][3]+1, Nxplus[i][5][j][4], Nxplus[i][5][j][5])
            if Nxplus[i][5][j][3] != 0:
                Nxminus[i][5][j][0] = N(Nxminus[i][5][j][1], Nxminus[i][5][j][3]-1, Nxminus[i][5][j][4], Nxminus[i][5][j][5])
                
            Nyplus[i][5][j][0] = N(Nyplus[i][5][j][1], Nyplus[i][5][j][3], Nyplus[i][5][j][4]+1, Nyplus[i][5][j][5])
            if Nyplus[i][5][j][4] != 0:
                Nyminus[i][5][j][0] = N(Nyminus[i][5][j][1], Nyminus[i][5][j][3], Nyminus[i][5][j][4]-1, Nyminus[i][5][j][5])
                
            Nzplus[i][5][j][0] = N(Nzplus[i][5][j][1], Nzplus[i][5][j][3], Nzplus[i][5][j][4], Nzplus[i][5][j][5]+1)
            if Nzplus[i][5][j][5] != 0:
                Nzminus[i][5][j][0] = N(Nzminus[i][5][j][1], Nzminus[i][5][j][3], Nzminus[i][5][j][4], Nzminus[i][5][j][5]-1)
    
    for atomidx in range(1, len(input)):
        # Nuclear-nuclear repulsion
        E = np.zeros(1)
        E[0] = nucdiff(input, atomidx, 1)
        np.save(str(atomidx)+'dxenuc.npy',E)
        
        E = np.zeros(1)
        E[0] = nucdiff(input, atomidx, 2)
        np.save(str(atomidx)+'dyenuc.npy',E)
        
        E = np.zeros(1)
        E[0] = nucdiff(input, atomidx, 3)
        np.save(str(atomidx)+'dzenuc.npy',E)
        #END OF nuclear-nuclear repulsion
        
        # Two electron integrals x diff
        start = time.time()
        ERI = np.zeros((len(basis),len(basis),len(basis),len(basis)))
        for mu in range(0, len(basis)):
            for nu in range(0, len(basis)):
                if mu >= nu:
                    for lam in range(0, len(basis)):
                        for sig in range(0, len(basis)):
                            munu = mu*(mu+1)/2+nu
                            lamsig = lam*(lam+1)/2+sig
                            if lam >= sig and munu >= lamsig:
                                a = np.zeros(4)
                                a[0] = mu
                                a[1] = nu
                                a[2] = lam
                                a[3] = sig
                                a = a.astype(int)
                                calc = 0
                                for i in range(basis[a[0]][4]):
                                    for j in range(basis[a[1]][4]):
                                        for k in range(basis[a[2]][4]):
                                            for l in range(basis[a[3]][4]):
                                                if atomidx == basis[a[0]][6] and atomidx == basis[a[1]][6] and atomidx == basis[a[2]][6] and atomidx == basis[a[3]][6]:
                                                    calc += 0
                                                else:
                                                    a2=basis[a[0]][5][i][1]
                                                    b=basis[a[1]][5][j][1]
                                                    c=basis[a[2]][5][k][1]
                                                    d=basis[a[3]][5][l][1]
                                                    Ax=basis[a[0]][1]
                                                    Ay=basis[a[0]][2]
                                                    Az=basis[a[0]][3] 
                                                    Bx=basis[a[1]][1]
                                                    By=basis[a[1]][2]
                                                    Bz=basis[a[1]][3]
                                                    Cx=basis[a[2]][1]
                                                    Cy=basis[a[2]][2]
                                                    Cz=basis[a[2]][3]
                                                    Dx=basis[a[3]][1]
                                                    Dy=basis[a[3]][2]
                                                    Dz=basis[a[3]][3]
                                                    l1=basis[a[0]][5][i][3]
                                                    l2=basis[a[1]][5][j][3]
                                                    l3=basis[a[2]][5][k][3]
                                                    l4=basis[a[3]][5][l][3] 
                                                    m1=basis[a[0]][5][i][4]
                                                    m2=basis[a[1]][5][j][4]
                                                    m3=basis[a[2]][5][k][4]
                                                    m4=basis[a[3]][5][l][4] 
                                                    n1=basis[a[0]][5][i][5]
                                                    n2=basis[a[1]][5][j][5]
                                                    n3=basis[a[2]][5][k][5]
                                                    n4=basis[a[3]][5][l][5] 
                                                    N1=basis[a[0]][5][i][0]
                                                    N2=basis[a[1]][5][j][0]
                                                    N3=basis[a[2]][5][k][0]
                                                    N4=basis[a[3]][5][l][0]
                                                    c1=basis[a[0]][5][i][2]
                                                    c2=basis[a[1]][5][j][2]
                                                    c3=basis[a[2]][5][k][2]
                                                    c4=basis[a[3]][5][l][2]
                                                    
                                                    Nxp1=Nxplus[a[0]][5][i][0]
                                                    Nxp2=Nxplus[a[1]][5][j][0]
                                                    Nxp3=Nxplus[a[2]][5][k][0]
                                                    Nxp4=Nxplus[a[3]][5][l][0]
                                                    
                                                    Nxm1=Nxminus[a[0]][5][i][0]
                                                    Nxm2=Nxminus[a[1]][5][j][0]
                                                    Nxm3=Nxminus[a[2]][5][k][0]
                                                    Nxm4=Nxminus[a[3]][5][l][0]

                                                    if atomidx == basis[a[0]][6]:
                                                        calc += Ndiff1(l1, a2)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1+1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, Nxp1, N2, N3, N4, c1, c2, c3, c4)
                                                        if l1 != 0:
                                                            calc += Ndiff2(l1, a2)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1-1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, Nxm1, N2, N3, N4, c1, c2, c3, c4)
                                                            
                                                    if atomidx == basis[a[1]][6]:
                                                        calc += Ndiff1(l2, b)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2+1, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, Nxp2, N3, N4, c1, c2, c3, c4)
                                                        if l2 != 0:
                                                            calc += Ndiff2(l2, b)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2-1, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, Nxm2, N3, N4, c1, c2, c3, c4)
                                                    if atomidx == basis[a[2]][6]:  
                                                        calc += Ndiff1(l3, c)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3+1, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, Nxp3, N4, c1, c2, c3, c4)
                                                        if l3 != 0:
                                                            calc += Ndiff2(l3, c)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3-1, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, Nxm3, N4, c1, c2, c3, c4)
                                                        
                                                    if atomidx == basis[a[3]][6]:
                                                        calc += Ndiff1(l4, d)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4+1, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, N3, Nxp4, c1, c2, c3, c4)
                                                        if l4 != 0:
                                                            calc += Ndiff2(l4, d)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4-1, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, N3, Nxm4, c1, c2, c3, c4)
                                                                                            
                                ERI[mu,nu,lam,sig] = calc
                                ERI[nu,mu,lam,sig] = calc
                                ERI[mu,nu,sig,lam] = calc
                                ERI[nu,mu,sig,lam] = calc
                                ERI[lam,sig,mu,nu] = calc
                                ERI[sig,lam,mu,nu] = calc
                                ERI[lam,sig,nu,mu] = calc
                                ERI[sig,lam,nu,mu] = calc
        np.save(str(atomidx)+'dxtwoint.npy',ERI)
        print(time.time()-start, 'ERI x diff: atom'+str(atomidx))
        #END OF two electron integrals x diff
        
        # Two electron integrals y diff
        start = time.time()
        ERI = np.zeros((len(basis),len(basis),len(basis),len(basis)))
        for mu in range(0, len(basis)):
            for nu in range(0, len(basis)):
                if mu >= nu:
                    for lam in range(0, len(basis)):
                        for sig in range(0, len(basis)):
                            munu = mu*(mu+1)/2+nu
                            lamsig = lam*(lam+1)/2+sig
                            if lam >= sig and munu >= lamsig:
                                a = np.zeros(4)
                                a[0] = mu
                                a[1] = nu
                                a[2] = lam
                                a[3] = sig
                                a = a.astype(int)
                                calc = 0
                                for i in range(basis[a[0]][4]):
                                    for j in range(basis[a[1]][4]):
                                        for k in range(basis[a[2]][4]):
                                            for l in range(basis[a[3]][4]):
                                                if atomidx == basis[a[0]][6] and atomidx == basis[a[1]][6] and atomidx == basis[a[2]][6] and atomidx == basis[a[3]][6]:
                                                    calc += 0
                                                else:
                                                    a2=basis[a[0]][5][i][1]
                                                    b=basis[a[1]][5][j][1]
                                                    c=basis[a[2]][5][k][1]
                                                    d=basis[a[3]][5][l][1]
                                                    Ax=basis[a[0]][1]
                                                    Ay=basis[a[0]][2]
                                                    Az=basis[a[0]][3] 
                                                    Bx=basis[a[1]][1]
                                                    By=basis[a[1]][2]
                                                    Bz=basis[a[1]][3]
                                                    Cx=basis[a[2]][1]
                                                    Cy=basis[a[2]][2]
                                                    Cz=basis[a[2]][3]
                                                    Dx=basis[a[3]][1]
                                                    Dy=basis[a[3]][2]
                                                    Dz=basis[a[3]][3]
                                                    l1=basis[a[0]][5][i][3]
                                                    l2=basis[a[1]][5][j][3]
                                                    l3=basis[a[2]][5][k][3]
                                                    l4=basis[a[3]][5][l][3] 
                                                    m1=basis[a[0]][5][i][4]
                                                    m2=basis[a[1]][5][j][4]
                                                    m3=basis[a[2]][5][k][4]
                                                    m4=basis[a[3]][5][l][4] 
                                                    n1=basis[a[0]][5][i][5]
                                                    n2=basis[a[1]][5][j][5]
                                                    n3=basis[a[2]][5][k][5]
                                                    n4=basis[a[3]][5][l][5] 
                                                    N1=basis[a[0]][5][i][0]
                                                    N2=basis[a[1]][5][j][0]
                                                    N3=basis[a[2]][5][k][0]
                                                    N4=basis[a[3]][5][l][0]
                                                    c1=basis[a[0]][5][i][2]
                                                    c2=basis[a[1]][5][j][2]
                                                    c3=basis[a[2]][5][k][2]
                                                    c4=basis[a[3]][5][l][2]
                                                    
                                                    Nyp1=Nyplus[a[0]][5][i][0]
                                                    Nyp2=Nyplus[a[1]][5][j][0]
                                                    Nyp3=Nyplus[a[2]][5][k][0]
                                                    Nyp4=Nyplus[a[3]][5][l][0]
                                                    
                                                    Nym1=Nyminus[a[0]][5][i][0]
                                                    Nym2=Nyminus[a[1]][5][j][0]
                                                    Nym3=Nyminus[a[2]][5][k][0]
                                                    Nym4=Nyminus[a[3]][5][l][0]

                                                    if atomidx == basis[a[0]][6]:
                                                        calc += Ndiff1(m1, a2)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4, m1+1, m2, m3, m4, n1, n2, n3, n4, Nyp1, N2, N3, N4, c1, c2, c3, c4)
                                                        if m1 != 0:
                                                            calc += Ndiff2(m1, a2)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4, m1-1, m2, m3, m4, n1, n2, n3, n4, Nym1, N2, N3, N4, c1, c2, c3, c4)
                                                            
                                                    if atomidx == basis[a[1]][6]:
                                                        calc += Ndiff1(m2, b)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4, m1, m2+1, m3, m4, n1, n2, n3, n4, N1, Nyp2, N3, N4, c1, c2, c3, c4)
                                                        if m2 != 0:
                                                            calc += Ndiff2(m2, b)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4, m1, m2-1, m3, m4, n1, n2, n3, n4, N1, Nym2, N3, N4, c1, c2, c3, c4)
                                                    if atomidx == basis[a[2]][6]:  
                                                        calc += Ndiff1(m3, c)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4, m1, m2, m3+1, m4, n1, n2, n3, n4, N1, N2, Nyp3, N4, c1, c2, c3, c4)
                                                        if m3 != 0:
                                                            calc += Ndiff2(m3, c)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4, m1, m2, m3-1, m4, n1, n2, n3, n4, N1, N2, Nym3, N4, c1, c2, c3, c4)
                                                        
                                                    if atomidx == basis[a[3]][6]:
                                                        calc += Ndiff1(m4, d)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4, m1, m2, m3, m4+1, n1, n2, n3, n4, N1, N2, N3, Nyp4, c1, c2, c3, c4)
                                                        if m4 != 0:
                                                            calc += Ndiff2(m4, d)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4, m1, m2, m3, m4-1, n1, n2, n3, n4, N1, N2, N3, Nym4, c1, c2, c3, c4)
                                                                                            
                                ERI[mu,nu,lam,sig] = calc
                                ERI[nu,mu,lam,sig] = calc
                                ERI[mu,nu,sig,lam] = calc
                                ERI[nu,mu,sig,lam] = calc
                                ERI[lam,sig,mu,nu] = calc
                                ERI[sig,lam,mu,nu] = calc
                                ERI[lam,sig,nu,mu] = calc
                                ERI[sig,lam,nu,mu] = calc
        np.save(str(atomidx)+'dytwoint.npy',ERI)
        print(time.time()-start, 'ERI y diff: atom'+str(atomidx))
        #END OF two electron integrals y diff
        
        # Two electron integrals z diff
        start = time.time()
        ERI = np.zeros((len(basis),len(basis),len(basis),len(basis)))
        for mu in range(0, len(basis)):
            for nu in range(0, len(basis)):
                if mu >= nu:
                    for lam in range(0, len(basis)):
                        for sig in range(0, len(basis)):
                            munu = mu*(mu+1)/2+nu
                            lamsig = lam*(lam+1)/2+sig
                            if lam >= sig and munu >= lamsig:
                                a = np.zeros(4)
                                a[0] = mu
                                a[1] = nu
                                a[2] = lam
                                a[3] = sig
                                a = a.astype(int)
                                calc = 0
                                for i in range(basis[a[0]][4]):
                                    for j in range(basis[a[1]][4]):
                                        for k in range(basis[a[2]][4]):
                                            for l in range(basis[a[3]][4]):
                                                if atomidx == basis[a[0]][6] and atomidx == basis[a[1]][6] and atomidx == basis[a[2]][6] and atomidx == basis[a[3]][6]:
                                                    calc += 0
                                                else:
                                                    a2=basis[a[0]][5][i][1]
                                                    b=basis[a[1]][5][j][1]
                                                    c=basis[a[2]][5][k][1]
                                                    d=basis[a[3]][5][l][1]
                                                    Ax=basis[a[0]][1]
                                                    Ay=basis[a[0]][2]
                                                    Az=basis[a[0]][3] 
                                                    Bx=basis[a[1]][1]
                                                    By=basis[a[1]][2]
                                                    Bz=basis[a[1]][3]
                                                    Cx=basis[a[2]][1]
                                                    Cy=basis[a[2]][2]
                                                    Cz=basis[a[2]][3]
                                                    Dx=basis[a[3]][1]
                                                    Dy=basis[a[3]][2]
                                                    Dz=basis[a[3]][3]
                                                    l1=basis[a[0]][5][i][3]
                                                    l2=basis[a[1]][5][j][3]
                                                    l3=basis[a[2]][5][k][3]
                                                    l4=basis[a[3]][5][l][3] 
                                                    m1=basis[a[0]][5][i][4]
                                                    m2=basis[a[1]][5][j][4]
                                                    m3=basis[a[2]][5][k][4]
                                                    m4=basis[a[3]][5][l][4] 
                                                    n1=basis[a[0]][5][i][5]
                                                    n2=basis[a[1]][5][j][5]
                                                    n3=basis[a[2]][5][k][5]
                                                    n4=basis[a[3]][5][l][5] 
                                                    N1=basis[a[0]][5][i][0]
                                                    N2=basis[a[1]][5][j][0]
                                                    N3=basis[a[2]][5][k][0]
                                                    N4=basis[a[3]][5][l][0]
                                                    c1=basis[a[0]][5][i][2]
                                                    c2=basis[a[1]][5][j][2]
                                                    c3=basis[a[2]][5][k][2]
                                                    c4=basis[a[3]][5][l][2]
                                                    
                                                    Nzp1=Nzplus[a[0]][5][i][0]
                                                    Nzp2=Nzplus[a[1]][5][j][0]
                                                    Nzp3=Nzplus[a[2]][5][k][0]
                                                    Nzp4=Nzplus[a[3]][5][l][0]
                                                    
                                                    Nzm1=Nzminus[a[0]][5][i][0]
                                                    Nzm2=Nzminus[a[1]][5][j][0]
                                                    Nzm3=Nzminus[a[2]][5][k][0]
                                                    Nzm4=Nzminus[a[3]][5][l][0]

                                                    if atomidx == basis[a[0]][6]:
                                                        calc += Ndiff1(n1, a2)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4, m1, m2, m3, m4, n1+1, n2, n3, n4, Nzp1, N2, N3, N4, c1, c2, c3, c4)
                                                        if n1 != 0:
                                                            calc += Ndiff2(n1, a2)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4, m1, m2, m3, m4, n1-1, n2, n3, n4, Nzm1, N2, N3, N4, c1, c2, c3, c4)
                                                            
                                                    if atomidx == basis[a[1]][6]:
                                                        calc += Ndiff1(n2, b)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2+1, n3, n4, N1, Nzp2, N3, N4, c1, c2, c3, c4)
                                                        if n2 != 0:
                                                            calc += Ndiff2(n2, b)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2-1, n3, n4, N1, Nzm2, N3, N4, c1, c2, c3, c4)
                                                    if atomidx == basis[a[2]][6]:  
                                                        calc += Ndiff1(n3, c)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3+1, n4, N1, N2, Nzp3, N4, c1, c2, c3, c4)
                                                        if n3 != 0:
                                                            calc += Ndiff2(n3, c)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3-1, n4, N1, N2, Nzm3, N4, c1, c2, c3, c4)
                                                        
                                                    if atomidx == basis[a[3]][6]:
                                                        calc += Ndiff1(n4, d)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4+1, N1, N2, N3, Nzp4, c1, c2, c3, c4)
                                                        if n4 != 0:
                                                            calc += Ndiff2(n4, d)*elelrep(a2, b, c, d, Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4-1, N1, N2, N3, Nzm4, c1, c2, c3, c4)
                                                                                            
                                ERI[mu,nu,lam,sig] = calc
                                ERI[nu,mu,lam,sig] = calc
                                ERI[mu,nu,sig,lam] = calc
                                ERI[nu,mu,sig,lam] = calc
                                ERI[lam,sig,mu,nu] = calc
                                ERI[sig,lam,mu,nu] = calc
                                ERI[lam,sig,nu,mu] = calc
                                ERI[sig,lam,nu,mu] = calc
        np.save(str(atomidx)+'dztwoint.npy',ERI)
        print(time.time()-start, 'ERI z diff: atom'+str(atomidx))
        #END OF two electron integrals z diff
        
        # Kinetic energy and overlap x diff
        start = time.time()
        S = np.zeros((len(basis),len(basis)))
        T = np.zeros((len(basis),len(basis)))
        for k in range(0, len(basis)):
            for l in range(0, len(basis)):
                if k >= l:
                    a = np.zeros(2)
                    a[0] = k
                    a[1] = l
                    a = a.astype(int)
                    calc = 0
                    calc2 = 0
                    for i in range(basis[a[0]][4]):
                        for j in range(basis[a[1]][4]):
                            if atomidx == basis[a[0]][6] and atomidx == basis[a[1]][6]:
                                calc += 0
                            else:
                                a2=basis[a[0]][5][i][1]
                                b=basis[a[1]][5][j][1]
                                Ax=basis[a[0]][1]
                                Ay=basis[a[0]][2]
                                Az=basis[a[0]][3]
                                Bx=basis[a[1]][1]
                                By=basis[a[1]][2]
                                Bz=basis[a[1]][3]
                                la=basis[a[0]][5][i][3]
                                lb=basis[a[1]][5][j][3]
                                ma=basis[a[0]][5][i][4]
                                mb=basis[a[1]][5][j][4]
                                na=basis[a[0]][5][i][5]
                                nb=basis[a[1]][5][j][5]
                                #N1=basis[a[0]][5][i][0]
                                #N2=basis[a[1]][5][j][0]
                                c1=basis[a[0]][5][i][2]
                                c2=basis[a[1]][5][j][2]
                                
                                if atomidx == basis[a[0]][6]:
                                    calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la+1, lb, ma, mb, na, nb, N(a2, la+1, ma, na), N(b, lb, mb, nb), c1, c2)
                                    calc += calct*Ndiff1(la,a2)
                                    calc2 += calct2*Ndiff1(la,a2)
                                    if la != 0:
                                        calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la-1, lb, ma, mb, na, nb, N(a2, la-1, ma, na), N(b, lb, mb, nb), c1, c2)
                                        calc += calct*Ndiff2(la,a2)
                                        calc2 += calct2*Ndiff2(la,a2)
                                        
                                if atomidx == basis[a[1]][6]:
                                    calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb+1, ma, mb, na, nb, N(a2, la, ma, na), N(b, lb+1, mb, nb), c1, c2)
                                    calc += calct*Ndiff1(lb,b)
                                    calc2 += calct2*Ndiff1(lb,b)
                                    if lb != 0:
                                        calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb-1, ma, mb, na, nb, N(a2, la, ma, na), N(b, lb-1, mb, nb), c1, c2)
                                        calc += calct*Ndiff2(lb,b)
                                        calc2 += calct2*Ndiff2(lb,b)
                    S[k,l] = calc2
                    S[l,k] = calc2
                    T[k,l] = calc
                    T[l,k] = calc
        np.save(str(atomidx)+'dxoverlap.npy',S)
        np.save(str(atomidx)+'dxEkin.npy',T)
        print(time.time()-start, 'Overlap + kin diff x atom: '+str(atomidx))
        #END OF kinetic energy and overlap x diff
        
        # Kinetic energy and overlap y diff
        start = time.time()
        S = np.zeros((len(basis),len(basis)))
        T = np.zeros((len(basis),len(basis)))
        for k in range(0, len(basis)):
            for l in range(0, len(basis)):
                if k >= l:
                    a = np.zeros(2)
                    a[0] = k
                    a[1] = l
                    a = a.astype(int)
                    calc = 0
                    calc2 = 0
                    for i in range(basis[a[0]][4]):
                        for j in range(basis[a[1]][4]):
                            if atomidx == basis[a[0]][6] and atomidx == basis[a[1]][6]:
                                calc += 0
                                calc2 += 0
                            else:
                                a2=basis[a[0]][5][i][1]
                                b=basis[a[1]][5][j][1]
                                Ax=basis[a[0]][1]
                                Ay=basis[a[0]][2]
                                Az=basis[a[0]][3]
                                Bx=basis[a[1]][1]
                                By=basis[a[1]][2]
                                Bz=basis[a[1]][3]
                                la=basis[a[0]][5][i][3]
                                lb=basis[a[1]][5][j][3]
                                ma=basis[a[0]][5][i][4]
                                mb=basis[a[1]][5][j][4]
                                na=basis[a[0]][5][i][5]
                                nb=basis[a[1]][5][j][5]
                                #N1=basis[a[0]][5][i][0]
                                #N2=basis[a[1]][5][j][0]
                                c1=basis[a[0]][5][i][2]
                                c2=basis[a[1]][5][j][2]
                                
                                if atomidx == basis[a[0]][6]:
                                    calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma+1, mb, na, nb, N(a2, la, ma+1, na), N(b, lb, mb, nb), c1, c2)
                                    calc += calct*Ndiff1(ma,a2)
                                    calc2 += calct2*Ndiff1(ma,a2)
                                    if ma != 0:
                                        calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma-1, mb, na, nb, N(a2, la, ma-1, na), N(b, lb, mb, nb), c1, c2)
                                        calc += calct*Ndiff2(ma,a2)
                                        calc2 += calct2*Ndiff2(ma,a2)
                                        
                                if atomidx == basis[a[1]][6]:
                                    calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma, mb+1, na, nb, N(a2, la, ma, na), N(b, lb, mb+1, nb), c1, c2)
                                    calc += calct*Ndiff1(mb,b)
                                    calc2 += calct2*Ndiff1(mb,b)
                                    if mb != 0:
                                        calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma, mb-1, na, nb, N(a2, la, ma, na), N(b, lb, mb-1, nb), c1, c2)
                                        calc += calct*Ndiff2(mb,b)
                                        calc2 += calct2*Ndiff2(mb,b)
                    S[k,l] = calc2
                    S[l,k] = calc2
                    T[k,l] = calc
                    T[l,k] = calc
        np.save(str(atomidx)+'dyoverlap.npy',S)
        np.save(str(atomidx)+'dyEkin.npy',T)
        print(time.time()-start, 'Overlap + kin diff y atom: '+str(atomidx))
        #END OF kinetic energy and overlap y diff
        
        # Kinetic energy and overlap z diff
        start = time.time()
        S = np.zeros((len(basis),len(basis)))
        T = np.zeros((len(basis),len(basis)))
        for k in range(0, len(basis)):
            for l in range(0, len(basis)):
                if k >= l:
                    a = np.zeros(2)
                    a[0] = k
                    a[1] = l
                    a = a.astype(int)
                    calc = 0
                    calc2 = 0
                    for i in range(basis[a[0]][4]):
                        for j in range(basis[a[1]][4]):
                            if atomidx == basis[a[0]][6] and atomidx == basis[a[1]][6]:
                                calc += 0
                            else:
                                a2=basis[a[0]][5][i][1]
                                b=basis[a[1]][5][j][1]
                                Ax=basis[a[0]][1]
                                Ay=basis[a[0]][2]
                                Az=basis[a[0]][3]
                                Bx=basis[a[1]][1]
                                By=basis[a[1]][2]
                                Bz=basis[a[1]][3]
                                la=basis[a[0]][5][i][3]
                                lb=basis[a[1]][5][j][3]
                                ma=basis[a[0]][5][i][4]
                                mb=basis[a[1]][5][j][4]
                                na=basis[a[0]][5][i][5]
                                nb=basis[a[1]][5][j][5]
                                #N1=basis[a[0]][5][i][0]
                                #N2=basis[a[1]][5][j][0]
                                c1=basis[a[0]][5][i][2]
                                c2=basis[a[1]][5][j][2]
                                
                                if atomidx == basis[a[0]][6]:
                                    calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma, mb, na+1, nb, N(a2, la, ma, na+1), N(b, lb, mb, nb), c1, c2)
                                    calc += calct*Ndiff1(na,a2)
                                    calc2 += calct2*Ndiff1(na,a2)
                                    if na != 0:
                                        calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma, mb, na-1, nb, N(a2, la, ma, na-1), N(b, lb, mb, nb), c1, c2)
                                        calc += calct*Ndiff2(na,a2)
                                        calc2 += calct2*Ndiff2(na,a2)
                                        
                                if atomidx == basis[a[1]][6]:
                                    calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma, mb, na, nb+1, N(a2, la, ma, na), N(b, lb, mb, nb+1), c1, c2)
                                    calc += calct*Ndiff1(nb,b)
                                    calc2 += calct2*Ndiff1(nb,b)
                                    if nb != 0:
                                        calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma, mb, na, nb-1, N(a2, la, ma, na), N(b, lb, mb, nb-1), c1, c2)
                                        calc += calct*Ndiff2(nb,b)
                                        calc2 += calct2*Ndiff2(nb,b)
                    S[k,l] = calc2
                    S[l,k] = calc2
                    T[k,l] = calc
                    T[l,k] = calc
        np.save(str(atomidx)+'dzoverlap.npy',S)
        np.save(str(atomidx)+'dzEkin.npy',T)
        print(time.time()-start, 'Overlap + kin diff z atom: '+str(atomidx))
        #END OF kinetic energy and overlap z diff

        # Nucleus electron attraction x diff
        start = time.time()
        Na = np.zeros((len(basis),len(basis)))
        for k in range(0, len(basis)):
            for l in range(0, len(basis)):
                if k >= l:
                    a = np.zeros(2)
                    a[0] = k
                    a[1] = l
                    a = a.astype(int)
                    calc = 0
                    for i in range(basis[a[0]][4]):
                        for j in range(basis[a[1]][4]):
                            a2=basis[a[0]][5][i][1]
                            b=basis[a[1]][5][j][1]
                            Ax=basis[a[0]][1]
                            Ay=basis[a[0]][2]
                            Az=basis[a[0]][3]
                            Bx=basis[a[1]][1]
                            By=basis[a[1]][2]
                            Bz=basis[a[1]][3]
                            l1=basis[a[0]][5][i][3]
                            l2=basis[a[1]][5][j][3]
                            m1=basis[a[0]][5][i][4]
                            m2=basis[a[1]][5][j][4]
                            n1=basis[a[0]][5][i][5]
                            n2=basis[a[1]][5][j][5]
                            N1=basis[a[0]][5][i][0]
                            N2=basis[a[1]][5][j][0]
                            c1=basis[a[0]][5][i][2]
                            c2=basis[a[1]][5][j][2]
                            
                            if atomidx == basis[a[0]][6] and atomidx == basis[a[1]][6]:
                                calc += Ndiff1(l1,a2)*elnucExclude(a2, b, Ax, Ay, Az, Bx, By, Bz, l1+1, l2, m1, m2, n1, n2, N(a2, l1+1, m1, n1), N(b, l2, m2, n2), c1, c2, input, atomidx)
                                if l1 != 0:
                                    calc += Ndiff2(l1,a2)*elnucExclude(a2, b, Ax, Ay, Az, Bx, By, Bz, l1-1, l2, m1, m2, n1, n2, N(a2, l1-1, m1, n1), N(b, l2, m2, n2), c1, c2, input, atomidx)
                                
                                calc += Ndiff1(l2,b)*elnucExclude(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2+1, m1, m2, n1, n2, N(a2, l1, m1, n1), N(b, l2+1, m2, n2), c1, c2, input, atomidx)
                                if l2 != 0:
                                    calc += Ndiff2(l2,b)*elnucExclude(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2-1, m1, m2, n1, n2, N(a2, l1, m1, n1), N(b, l2-1, m2, n2), c1, c2, input, atomidx)
                            
                            else:
                                calc += electricfield(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, input, 'x', atomidx)
                                
                                if atomidx == basis[a[0]][6]:
                                    calc += Ndiff1(l1,a2)*elnuc(a2, b, Ax, Ay, Az, Bx, By, Bz, l1+1, l2, m1, m2, n1, n2, N(a2, l1+1, m1, n1), N(b, l2, m2, n2), c1, c2, input)
                                    if l1 != 0:
                                        calc += Ndiff2(l1,a2)*elnuc(a2, b, Ax, Ay, Az, Bx, By, Bz, l1-1, l2, m1, m2, n1, n2, N(a2, l1-1, m1, n1), N(b, l2, m2, n2), c1, c2, input)
        
                                if atomidx == basis[a[1]][6]:
                                    calc += Ndiff1(l2,b)*elnuc(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2+1, m1, m2, n1, n2, N(a2, l1, m1, n1), N(b, l2+1, m2, n2), c1, c2, input)
                                    if l2 != 0:
                                        calc += Ndiff2(l2,b)*elnuc(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2-1, m1, m2, n1, n2, N(a2, l1, m1, n1), N(b, l2-1, m2, n2), c1, c2, input)
        
                    Na[k,l] = calc
                    Na[l,k] = calc
        np.save(str(atomidx)+'dxnucatt.npy',Na)
        print(time.time()-start, 'Nuc att diffx atom: '+str(atomidx))
        #END OF nucleus electron attraction x diff

        # Nucleus electron attraction y diff
        start = time.time()
        Na = np.zeros((len(basis),len(basis)))
        for k in range(0, len(basis)):
            for l in range(0, len(basis)):
                if k >= l:
                    a = np.zeros(2)
                    a[0] = k
                    a[1] = l
                    a = a.astype(int)
                    calc = 0
                    for i in range(basis[a[0]][4]):
                        for j in range(basis[a[1]][4]):
                            a2=basis[a[0]][5][i][1]
                            b=basis[a[1]][5][j][1]
                            Ax=basis[a[0]][1]
                            Ay=basis[a[0]][2]
                            Az=basis[a[0]][3]
                            Bx=basis[a[1]][1]
                            By=basis[a[1]][2]
                            Bz=basis[a[1]][3]
                            l1=basis[a[0]][5][i][3]
                            l2=basis[a[1]][5][j][3]
                            m1=basis[a[0]][5][i][4]
                            m2=basis[a[1]][5][j][4]
                            n1=basis[a[0]][5][i][5]
                            n2=basis[a[1]][5][j][5]
                            N1=basis[a[0]][5][i][0]
                            N2=basis[a[1]][5][j][0]
                            c1=basis[a[0]][5][i][2]
                            c2=basis[a[1]][5][j][2]
                            
                            
                            if atomidx == basis[a[0]][6] and atomidx == basis[a[1]][6]:
                                calc += Ndiff1(m1,a2)*elnucExclude(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1+1, m2, n1, n2, N(a2, l1, m1+1, n1), N(b, l2, m2, n2), c1, c2, input, atomidx)
                                if m1 != 0:
                                    calc += Ndiff2(m1,a2)*elnucExclude(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1-1, m2, n1, n2, N(a2, l1, m1-1, n1), N(b, l2, m2, n2), c1, c2, input, atomidx)
                                
                                calc += Ndiff1(m2,b)*elnucExclude(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2+1, n1, n2, N(a2, l1, m1, n1), N(b, l2, m2+1, n2), c1, c2, input, atomidx)
                                if m2 != 0:
                                    calc += Ndiff2(m2,b)*elnucExclude(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2-1, n1, n2, N(a2, l1, m1, n1), N(b, l2, m2-1, n2), c1, c2, input, atomidx)
                            
                            else:
                                calc += electricfield(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, input, 'y', atomidx)
                                
                                if atomidx == basis[a[0]][6]:
                                    calc += Ndiff1(m1,a2)*elnuc(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1+1, m2, n1, n2, N(a2, l1, m1+1, n1), N(b, l2, m2, n2), c1, c2, input)
                                    if m1 != 0:
                                        calc += Ndiff2(m1,a2)*elnuc(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1-1, m2, n1, n2, N(a2, l1, m1-1, n1), N(b, l2, m2, n2), c1, c2, input)
        
                                if atomidx == basis[a[1]][6]:
                                    calc += Ndiff1(m2,b)*elnuc(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2+1, n1, n2, N(a2, l1, m1, n1), N(b, l2, m2+1, n2), c1, c2, input)
                                    if m2 != 0:
                                        calc += Ndiff2(m2,b)*elnuc(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2-1, n1, n2, N(a2, l1, m1, n1), N(b, l2, m2-1, n2), c1, c2, input)
                        
                    Na[k,l] = calc
                    Na[l,k] = calc
        np.save(str(atomidx)+'dynucatt.npy',Na)
        print(time.time()-start, 'Nuc att diff y atom: '+str(atomidx))
        #END OF nucleus electron attraction y diff

        # Nucleus electron attraction z diff
        start = time.time()
        Na = np.zeros((len(basis),len(basis)))
        for k in range(0, len(basis)):
            for l in range(0, len(basis)):
                if k >= l:
                    a = np.zeros(2)
                    a[0] = k
                    a[1] = l
                    a = a.astype(int)
                    calc = 0
                    for i in range(basis[a[0]][4]):
                        for j in range(basis[a[1]][4]):
                            a2=basis[a[0]][5][i][1]
                            b=basis[a[1]][5][j][1]
                            Ax=basis[a[0]][1]
                            Ay=basis[a[0]][2]
                            Az=basis[a[0]][3]
                            Bx=basis[a[1]][1]
                            By=basis[a[1]][2]
                            Bz=basis[a[1]][3]
                            l1=basis[a[0]][5][i][3]
                            l2=basis[a[1]][5][j][3]
                            m1=basis[a[0]][5][i][4]
                            m2=basis[a[1]][5][j][4]
                            n1=basis[a[0]][5][i][5]
                            n2=basis[a[1]][5][j][5]
                            N1=basis[a[0]][5][i][0]
                            N2=basis[a[1]][5][j][0]
                            c1=basis[a[0]][5][i][2]
                            c2=basis[a[1]][5][j][2]
                            
                            if atomidx == basis[a[0]][6] and atomidx == basis[a[1]][6]:
                                calc += Ndiff1(n1,a2)*elnucExclude(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1+1, n2, N(a2, l1, m1, n1+1), N(b, l2, m2, n2), c1, c2, input, atomidx)
                                if n1 != 0:
                                    calc += Ndiff2(n1,a2)*elnucExclude(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1-1, n2, N(a2, l1, m1, n1-1), N(b, l2, m2, n2), c1, c2, input, atomidx)
                                
                                calc += Ndiff1(n2,b)*elnucExclude(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2+1, N(a2, l1, m1, n1), N(b, l2, m2, n2+1), c1, c2, input, atomidx)
                                if n2 != 0:
                                    calc += Ndiff2(n2,b)*elnucExclude(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2-1, N(a2, l1, m1, n1), N(b, l2, m2, n2-1), c1, c2, input, atomidx)
                            
                            else:
                                calc += electricfield(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, input, 'z', atomidx)
                                
                                if atomidx == basis[a[0]][6]:
                                    calc += Ndiff1(n1,a2)*elnuc(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1+1, n2, N(a2, l1, m1, n1+1), N(b, l2, m2, n2), c1, c2, input)
                                    if n1 != 0:
                                        calc += Ndiff2(n1,a2)*elnuc(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1-1, n2, N(a2, l1, m1, n1-1), N(b, l2, m2, n2), c1, c2, input)
        
                                if atomidx == basis[a[1]][6]:
                                    calc += Ndiff1(n2,b)*elnuc(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2+1, N(a2, l1, m1, n1), N(b, l2, m2, n2+1), c1, c2, input)
                                    if n2 != 0:
                                        calc += Ndiff2(n2,b)*elnuc(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2-1, N(a2, l1, m1, n1), N(b, l2, m2, n2-1), c1, c2, input)
        
                    Na[k,l] = calc
                    Na[l,k] = calc
        np.save(str(atomidx)+'dznucatt.npy',Na)
        print(time.time()-start, 'Nuc att diff z atom: '+str(atomidx))
        #END OF nucleus electron attraction z diff
