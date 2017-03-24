import math
import numpy as np
import scipy.misc as scm
import scipy.special as scs

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
    for C in range(1, len(input)):
        Zc = input[C][0]
        C = np.array([input[C][1],input[C][2],input[C][3]])
        RPC = np.linalg.norm(P-C)
        
        for t in range(0, l1+l2+1):
            for u in range(0, m1+m2+1):
                for v in range(0, n1+n2+1):
                    Ex = E(l1,l2,t,A[0]-B[0],a,b,P[0]-A[0],P[0]-B[0],A[0]-B[0])
                    Ey = E(m1,m2,u,A[1]-B[1],a,b,P[1]-A[1],P[1]-B[1],A[1]-B[1])
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

    val = 0.0
    for t in range(l1+l2+1):
        for u in range(m1+m2+1):
            for v in range(n1+n2+1):
                for tau in range(l3+l4+1):
                    for nu in range(m3+m4+1):
                        for phi in range(n3+n4+1):
                            E1 = E(l1,l2,t,A[0]-B[0],a,b,P[0]-A[0],P[0]-B[0],A[0]-B[0])
                            E2 = E(m1,m2,u,A[1]-B[1],a,b,P[1]-A[1],P[1]-B[1],A[1]-B[1])
                            E3 = E(n1,n2,v,A[2]-B[2],a,b,P[2]-A[2],P[2]-B[2],A[2]-B[2])
                            E4 = E(l3,l4,tau,C[0]-D[0],c,d,Q[0]-C[0],Q[0]-D[0],C[0]-D[0])
                            E5 = E(m3,m4,nu ,C[1]-D[1],c,d,Q[1]-C[1],Q[1]-D[1],C[1]-D[1])
                            E6 = E(n3,n4,phi,C[2]-D[2],c,d,Q[2]-C[2],Q[2]-D[2],C[2]-D[2])
                            R1 = R(t+tau,u+nu,v+phi,0,alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ) 
                            val += E1*E2*E3*E4*E5*E6*np.power(-1,tau+nu+phi)*R1

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

def Velesp(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, rcx, rcy, rcz):
    #McMurchie-Davidson scheme    
    N = N1*N2*c1*c2
    p = a + b
    A = np.array([Ax, Ay, Az])
    B = np.array([Bx, By, Bz])
    C = np.array([rcx, rcy, rcz])
    P = gaussian_product_center(a,A,b,B)
    RPC = np.linalg.norm(P-C)
    val = 0
    for t in range(0, l1+l2+1):
        for u in range(0, m1+m2+1):
            for v in range(0, n1+n2+1):
                Ex = E(l1,l2,t,A[0]-B[0],a,b,P[0]-A[0],P[0]-B[0],A[0]-B[0])
                Ey = E(m1,m2,u,A[1]-B[1],a,b,P[1]-A[1],P[1]-B[1],A[1]-B[1])
                Ez = E(n1,n2,v,A[2]-B[2],a,b,P[2]-A[2],P[2]-B[2],A[2]-B[2])
                val += Ex*Ey*Ez*R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC)
    return val*2*np.pi/p*N
    
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

##CALC OF INTEGRALS
def runIntegrals(input, basis):
    # Set up indexes for integrals
    See = {}
    for i in range(1, len(basis)+1):
        for j in range(1, len(basis)+1):
            if i >= j:
                See[str(int(i))+';'+str(int(j))] = 0
    #END OF set up indexes for integrals
    
    # Nuclear-nuclear repulsion
    E = np.zeros(1)
    E[0] = nucrep(input)
    np.save('enuc.npy',E)
    #END OF nuclear-nuclear repulsion
    
    # Two electron integrals
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
    #END OF two electron integrals
    
    # Kinetic energy and overlap
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
    #END OF kinetic energy and overlap
    
    # Nucleus electron attraction
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
                        calc += Velesp(basis[a[0]][5][i][1], basis[a[1]][5][j][1], basis[a[0]][1], basis[a[0]][2], basis[a[0]][3], basis[a[1]][1], basis[a[1]][2], basis[a[1]][3], basis[a[0]][5][i][3], basis[a[1]][5][j][3], basis[a[0]][5][i][4], basis[a[1]][5][j][4],basis[a[0]][5][i][5], basis[a[1]][5][j][5], basis[a[0]][5][i][0], basis[a[1]][5][j][0], basis[a[0]][5][i][2], basis[a[1]][5][j][2], rcx, rcy, rcz)
            Ve[k,l] = calc
            Ve[l,k] = calc
        
    return Ve
