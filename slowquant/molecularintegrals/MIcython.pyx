import cython
import numpy as np
cimport numpy as np
from scipy.special.cython_special cimport hyp1f1   


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double [:,:,:] runR(int l1l2, int m1m2, int n1n2, double [:] C, double [:] P, double p, int check=0):
    cdef double RPC
    cdef double [:,:,:] R1
    cdef int t, u, v
    # check = 0, normal calculation. 
    # check = 1, derivative calculation
    if check == 0:
        RPC = ((P[0]-C[0])**2+(P[1]-C[1])**2+(P[2]-C[2])**2)**0.5
        R1 = np.ones((l1l2+1,m1m2+1,n1n2+1))
        for t in range(0, l1l2+1):
            for u in range(0, m1m2+1):
                for v in range(0, n1n2+1):
                    R1[t,u,v] = R2(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC)
                    
    elif check == 1:
        # For the first derivative +1 is needed in t, u and v
        # but only one at a time.
        RPC = ((P[0]-C[0])**2+(P[1]-C[1])**2+(P[2]-C[2])**2)**0.5
        R1 = np.ones((l1l2+2,m1m2+2,n1n2+2))
        for t in range(0, l1l2+1):
            for u in range(0, m1m2+1):
                R1[t,u,n1n2+1] = R2(t,u,n1n2+1,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC)
                for v in range(0, n1n2+1):
                    R1[t,u,v] = R2(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC)
                    if t == 0:
                        R1[l1l2+1,u,v] = R2(l1l2+1,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC)
                    if u == 0:
                        R1[t,m1m2+1,v] = R2(t,m1m2+1,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC) 
                        
    return R1
    

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef R2(int t, int u, int v, int n, double p, double PCx, double PCy, double PCz, double RPC):
    cdef double T, val, res    

    T = p*RPC*RPC
    val = 0.0
    if t == u == v == 0:
        val += (-2.0*p)**n*boys(n,T)
    elif t == u == 0:
        if v > 1:
            val+=(v-1)*R2(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC)
        val+=PCz*R2(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC)  
    elif t == 0:
        if u > 1:
            val+=(u-1)*R2(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC)  
        val+=PCy*R2(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC)  
    else:
        if t > 1:
            val+=(t-1)*R2(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC) 
        val+=PCx*R2(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
    
    return val


@cython.cdivision(True)
cdef double boys(double m,double T):
    return hyp1f1(m+0.5,m+1.5,-T)/(2.0*m+1.0) 
    

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double elelrep(double p, double q, int l1, int l2, int l3, int l4, int m1, int m2, int m3, int m4, int n1, int n2, int n3, int n4, double N1, double N2, double N3, double N4, double c1, double c2, double c3, double c4, double [:] E1, double [:] E2, double [:] E3, double [:] E4, double [:] E5, double [:] E6, double [:,:,:] Rpre):
    cdef double N, val, factor
    cdef int tau, nu, phi, t, u, v
    cdef double pi = 3.141592653589793238462643383279

    N = N1*N2*N3*N4*c1*c2*c3*c4
    
    val = 0.0
    for tau in range(l3+l4+1):
        for nu in range(m3+m4+1):
            for phi in range(n3+n4+1):
                factor = (-1.0)**(tau+nu+phi)
                for t in range(l1+l2+1):
                    for u in range(m1+m2+1):
                        for v in range(n1+n2+1):
                            val += E1[t]*E2[u]*E3[v]*E4[tau]*E5[nu]*E6[phi]*Rpre[t+tau,u+nu,v+phi]*factor

    val *= 2.0*pi**2.5/(p*q*(p+q)**0.5) 
    return val*N

cpdef double boysPrun(double m,double T):
    # boys_Python_run, used in tests.py to test boys function
    return boys(m, T)
