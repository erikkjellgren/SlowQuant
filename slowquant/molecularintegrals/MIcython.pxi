import cython
import numpy as np
cimport numpy as np
from scipy.special.cython_special cimport hyp1f1, gamma


cdef double [:,:,:] R(int l1l2, int m1m2, int n1n2, double Cx, double Cy, double Cz, double Px,  double Py, double Pz, double p, double [:,:,:] R1, double [:,:,:,:] Rbuffer, check=0):
    cdef double RPC, PCx, PCy, PCz, val
    cdef int t, u, v, n, exclude_from_n
    # check = 0, normal calculation. 
    # check = 1, derivative calculation
    
    PCx = Px-Cx
    PCy = Py-Cy
    PCz = Pz-Cz
    RPC = ((PCx)**2+(PCy)**2+(PCz)**2)**0.5
    if check == 0:
        for t in range(0, l1l2+1):
            for u in range(0, m1m2+1):
                for v in range(0, n1n2+1):
                    # Check the range of n, to ensure no redundent n are calculated
                    if t == u == 0:
                        exclude_from_n = v
                    elif t == 0:
                        exclude_from_n = n1n2 + u
                    else:
                        exclude_from_n = n1n2 + m1m2 + t
                    for n in range(0, l1l2+m1m2+n1n2+1-exclude_from_n):
                        val = 0.0
                        if t == u == v == 0:
                            Rbuffer[t,u,v,n] = (-2.0*p)**n*boys(n,p*RPC*RPC)
                        else:
                            if t == u == 0:
                                if v > 1:
                                    val += (v-1)*Rbuffer[t,u,v-2,n+1]
                                val += PCz*Rbuffer[t,u,v-1,n+1]  
                            elif t == 0:
                                if u > 1:
                                    val += (u-1)*Rbuffer[t,u-2,v,n+1]
                                val += PCy*Rbuffer[t,u-1,v,n+1]
                            else:
                                if t > 1:
                                    val += (t-1)*Rbuffer[t-2,u,v,n+1]
                                val += PCx*Rbuffer[t-1,u,v,n+1]
                            Rbuffer[t,u,v,n] = val
                            
                        if n == 0:
                            R1[t,u,v] = Rbuffer[t,u,v,n]
                            
                            
    elif check == 1:
        # For the first derivative +1 is needed in t, u and v
        # First the "normal" Rs are calculated
        for t in range(0, l1l2+1):
            for u in range(0, m1m2+1):
                for v in range(0, n1n2+1):
                    # Check the range of n, to ensure no redundent n are calculated
                    if t == u == 0:
                        exclude_from_n = v
                    elif t == 0:
                        exclude_from_n = n1n2 + u
                    else:
                        exclude_from_n = n1n2 + m1m2 + t
                    # +1 in n because of derivative    
                    for n in range(0, l1l2+m1m2+n1n2+1+1-exclude_from_n):
                        val = 0.0
                        if t == u == v == 0:
                            Rbuffer[t,u,v,n] = (-2.0*p)**n*boys(n,p*RPC*RPC)
                        else:
                            if t == u == 0:
                                if v > 1:
                                    val += (v-1)*Rbuffer[t,u,v-2,n+1]
                                val += PCz*Rbuffer[t,u,v-1,n+1]  
                            elif t == 0:
                                if u > 1:
                                    val += (u-1)*Rbuffer[t,u-2,v,n+1]
                                val += PCy*Rbuffer[t,u-1,v,n+1]
                            else:
                                if t > 1:
                                    val += (t-1)*Rbuffer[t-2,u,v,n+1]
                                val += PCx*Rbuffer[t-1,u,v,n+1]
                            Rbuffer[t,u,v,n] = val

                        if n == 0:
                            R1[t,u,v] = Rbuffer[t,u,v,n]
        
        # The next three blocks of code, calculates the
        # +1 incriments in the different angularmoment directions.
        # only one direction is +1 at a time.
        # eg. no need to calc R(t+1,u+1,v+1)
        # but only; R(t+1,u,v), R(t,u,v+1), R(t,u+,v)
        v = n1n2+1
        for t in range(0, l1l2+1):
            for u in range(0, m1m2+1):
                # Check the range of n, to ensure no redundent n are calculated
                if t == u == 0:
                    exclude_from_n = v
                elif t == 0:
                    exclude_from_n = n1n2 + 1 + u
                else:
                    exclude_from_n = n1n2 + 1 + m1m2 + t
                # +1 in n because of derivative    
                for n in range(0, l1l2+m1m2+n1n2+1+1-exclude_from_n):
                    val = 0.0
                    if t == u == 0:
                        if v > 1:
                            val += (v-1)*Rbuffer[t,u,v-2,n+1]
                        val += PCz*Rbuffer[t,u,v-1,n+1]  
                    elif t == 0:
                        if u > 1:
                            val += (u-1)*Rbuffer[t,u-2,v,n+1]
                        val += PCy*Rbuffer[t,u-1,v,n+1]
                    else:
                        if t > 1:
                            val += (t-1)*Rbuffer[t-2,u,v,n+1]
                        val += PCx*Rbuffer[t-1,u,v,n+1]
                    Rbuffer[t,u,v,n] = val

                    if n == 0:
                        R1[t,u,v] = Rbuffer[t,u,v,n]
        
        u = m1m2+1
        for t in range(0, l1l2+1):
            for v in range(0, n1n2+1):
                # Check the range of n, to ensure no redundent n are calculated
                if t == 0:
                    exclude_from_n = n1n2 + u
                else:
                    exclude_from_n = n1n2 + m1m2 + 1 + t
                # +1 in n because of derivative    
                for n in range(0, l1l2+m1m2+n1n2+1+1-exclude_from_n):
                    val = 0.0
                    if t == 0:
                        if u > 1:
                            val += (u-1)*Rbuffer[t,u-2,v,n+1]
                        val += PCy*Rbuffer[t,u-1,v,n+1]
                    else:
                        if t > 1:
                            val += (t-1)*Rbuffer[t-2,u,v,n+1]
                        val += PCx*Rbuffer[t-1,u,v,n+1]
                    Rbuffer[t,u,v,n] = val

                    if n == 0:
                        R1[t,u,v] = Rbuffer[t,u,v,n]

        t = l1l2+1
        for u in range(0, m1m2+1):
            for v in range(0, n1n2+1):
                # Check the range of n, to ensure no redundent n are calculated
                exclude_from_n = n1n2 + m1m2 + t
                # +1 in n because of derivative    
                for n in range(0, l1l2+m1m2+n1n2+1+1-exclude_from_n):
                    val = 0.0
                    if t > 1:
                        val += (t-1)*Rbuffer[t-2,u,v,n+1]
                    val += PCx*Rbuffer[t-1,u,v,n+1]
                    Rbuffer[t,u,v,n] = val

                    if n == 0:
                        R1[t,u,v] = Rbuffer[t,u,v,n]
        
    return R1
    
cdef double boys(double m,double T):
    return hyp1f1(m+0.5,m+1.5,-T)/(2.0*m+1.0) 

cdef double elelrep(double p, double q, int l1, int l2, int l3, int l4, int m1, int m2, int m3, int m4, int n1, int n2, int n3, int n4, double N1, double N2, double N3, double N4, double c1, double c2, double c3, double c4, double [:] E1, double [:] E2, double [:] E3, double [:] E4, double [:] E5, double [:] E6, double [:,:,:] Rpre):
    cdef double N, val, factor
    cdef int tau, nu, phi, t, u, v
    cdef double pi = 3.141592653589793238462643383279

    N = N1*N2*N3*N4*c1*c2*c3*c4
    
    val = 0.0
    for tau in range(l3+l4+1):
        for nu in range(m3+m4+1):
            for phi in range(n3+n4+1):
                factor = (-1.0)**(tau+nu+phi)*E4[tau]*E5[nu]*E6[phi]
                for t in range(l1+l2+1):
                    for u in range(m1+m2+1):
                        for v in range(n1+n2+1):
                            val += E1[t]*E2[u]*E3[v]*Rpre[t+tau,u+nu,v+phi]*factor

    val *= 2.0*pi**2.5/(p*q*(p+q)**0.5) 
    return val*N

cdef double E(int i, int j, int t, double Qx, double a, double b, double XPA, double XPB, double XAB):
    #McMurchie-Davidson scheme, 9.5.6 and 9.5.7 Helgaker
    cdef double p, q
    
    p = a + b
    q = a*b/p
    if (t < 0) or (t > (i + j)):
        return 0.0
    elif i == j == t == 0:
        return np.exp(-q*Qx*Qx)
    elif j == 0:
        return (1.0/(2.0*p))*E(i-1,j,t-1,Qx,a,b,XPA,XPB,XAB) + XPA*E(i-1,j,t,Qx,a,b,XPA,XPB,XAB) + (t+1.0)*E(i-1,j,t+1,Qx,a,b,XPA,XPB,XAB)
    else:
        return (1.0/(2.0*p))*E(i,j-1,t-1,Qx,a,b,XPA,XPB,XAB) + XPB*E(i,j-1,t,Qx,a,b,XPA,XPB,XAB) + (t+1.0)*E(i,j-1,t+1,Qx,a,b,XPA,XPB,XAB)    


cdef double elnuc(double p, int l1, int l2, int m1, int m2, int n1, int n2, double  N1, double N2, double c1, double c2, double Zc, double [:] Ex, double [:] Ey, double [:] Ez, double [:,:,:] R1):
    #McMurchie-Davidson scheme    
    cdef double N, val
    cdef double pi = 3.141592653589793238462643383279
    cdef int t, u, v
    
    N = N1*N2*c1*c2

    val = 0.0
    for t in range(0, l1+l2+1):
        for u in range(0, m1+m2+1):
            for v in range(0, n1+n2+1):
                val += Ex[t]*Ey[u]*Ez[v]*R1[t,u,v]*Zc

    return -val*2.0*pi/p*N


cdef Kin(double a, double b, double Ax, double Ay, double Az, double Bx, double By, double Bz, int la, int lb, int ma, int mb, int na, int nb, double N1, double N2, double c1, double c2):
    #Obara-Saika scheme, 9.3.40 and 9.3.41 Helgaker
    # Calculates electronic kinetic energy and overlap integrals
    #    at the same time
    cdef double p, N, Px, Py, Pz, XPA, YPA, ZPA, XPB, YPB, ZPB
    cdef double [:,:] Tijx, Tijy, Tijz, Sx, Sy, Sz
    cdef int i, j
    
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
    
    Tijx = np.zeros((la+2,lb+2))
    Tijy = np.zeros((ma+2,mb+2))
    Tijz = np.zeros((na+2,nb+2))
    Sx = Overlap(a, b, la, lb, Ax, Bx)
    Sy = Overlap(a, b, ma, mb, Ay, By)
    Sz = Overlap(a, b, na, nb, Az, Bz)
    Tijx[0,0] = (a-2.0*a**2*(XPA**2+1.0/(2.0*p)))*Sx[0,0]
    Tijy[0,0] = (a-2.0*a**2*(YPA**2+1.0/(2.0*p)))*Sy[0,0]
    Tijz[0,0] = (a-2.0*a**2*(ZPA**2+1.0/(2.0*p)))*Sz[0,0]
    
    for i in range(0, la+1):
        for j in range(0, lb+1):
            Tijx[i+1,j] = XPA*Tijx[i,j] + 1.0/(2.0*p)*(i*Tijx[i-1,j]+j*Tijx[i,j-1]) + b/p*(2.0*a*Sx[i+1,j] - i*Sx[i-1,j])
            Tijx[i,j+1] = XPB*Tijx[i,j] + 1.0/(2.0*p)*(i*Tijx[i-1,j]+j*Tijx[i,j-1]) + a/p*(2.0*b*Sx[i,j+1] - j*Sx[i,j-1])
    
    for i in range(0, ma+1):
        for j in range(0, mb+1):
            Tijy[i+1,j] = YPA*Tijy[i,j] + 1.0/(2.0*p)*(i*Tijy[i-1,j]+j*Tijy[i,j-1]) + b/p*(2.0*a*Sy[i+1,j] - i*Sy[i-1,j])
            Tijy[i,j+1] = YPB*Tijy[i,j] + 1.0/(2.0*p)*(i*Tijy[i-1,j]+j*Tijy[i,j-1]) + a/p*(2.0*b*Sy[i,j+1] - j*Sy[i,j-1])
    
    for i in range(0, na+1):
        for j in range(0, nb+1):
            Tijz[i+1,j] = ZPA*Tijz[i,j] + 1.0/(2.0*p)*(i*Tijz[i-1,j]+j*Tijz[i,j-1]) + b/p*(2.0*a*Sz[i+1,j] - i*Sz[i-1,j])
            Tijz[i,j+1] = ZPB*Tijz[i,j] + 1.0/(2.0*p)*(i*Tijz[i-1,j]+j*Tijz[i,j-1]) + a/p*(2.0*b*Sz[i,j+1] - j*Sz[i,j-1])
    
    return (Tijx[la, lb]*Sy[ma,mb]*Sz[na,nb]+Tijy[ma, mb]*Sx[la,lb]*Sz[na,nb]+Tijz[na, nb]*Sy[ma,mb]*Sx[la,lb])*N, Sx[la, lb]*Sy[ma, mb]*Sz[na, nb]*N


cdef double [:,:] Overlap(double a, double b, int la, int lb, double Ax, double Bx):
    #Obara-Saika scheme, 9.3.8 and 9.3.9 Helgaker
    #Used in Kin integral!
    cdef double p, u, Px, S00
    cdef double pi = 3.141592653589793238462643383279
    cdef double [:,:] Sij
    cdef int i, j

    p = a + b
    u = a*b/p
    
    Px = (a*Ax+b*Bx)/p
    
    S00 = (pi/p)**0.5 * np.exp(-u*(Ax-Bx)**2)
    
    Sij = np.zeros((la+2,lb+2))
    Sij[0,0] = S00
    
    
    for i in range(0, la+1):
        for j in range(0, lb+1):
            Sij[i+1,j] = (Px-Ax)*Sij[i,j] + 1.0/(2.0*p) * (i*Sij[i-1,j] + j*Sij[i,j-1])
            Sij[i,j+1] = (Px-Bx)*Sij[i,j] + 1.0/(2.0*p) * (i*Sij[i-1,j] + j*Sij[i,j-1])
    
    return Sij


cdef double electricfield(double p, double [:] Ex, double [:] Ey, double [:] Ez, double Zc, int l1, int l2, int m1, int m2, int n1, int n2, double N1, double N2, double c1, double c2, int derivative, double [:,:,:] R1):
    #McMurchie-Davidson scheme
    cdef double N, val
    cdef double pi = 3.141592653589793238462643383279
    cdef int dx, dy, dz, t, u, v
    
    N = N1*N2*c1*c2
    
    dx = 0
    dy = 0
    dz = 0
    if derivative == 1:
        dx = 1
    elif derivative == 2:
        dy = 1
    else:
        dz = 1

    val = 0.0
    for t in range(0, l1+l2+1):
        for u in range(0, m1+m2+1):
            for v in range(0, n1+n2+1):
                val += Ex[t]*Ey[u]*Ez[v]*R1[t+dx,u+dy,v+dz]*Zc
    
    return val*2.0*pi/p*N
    


"""
cdef double factorial2(double n):
    cdef int i, n_range
    cdef double out
    n_range = int(n)
    out = 1.0
    if n > 0:
        for i in range(0, int(n_range+1)//2):
            out = out*(n-2*i)
    return out

cdef double boys(double m,double T):
    cdef double pi = 3.141592653589793238462643383279
    if T == 0.0:
        # special case of T = 0
        return 1.0/(2.0*m+1.0)
    elif T > 2.4951*m + 35.451:
        # Long range approximation, table from Obara1986
        return factorial2(2.0*m-1.0)/(2.0**(m+1))*(pi/(T**(2*m+1)))**0.5        
    else:
        return hyp1f1(m+0.5,m+1.5,-T)/(2.0*m+1.0)
"""
