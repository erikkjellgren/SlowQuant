import cython
import numpy as np
cimport numpy as np
include "MIcython.pxi"


cpdef runCythonIntegrals(int [:,:] basisidx, double [:,:] basisfloat, int [:,:] basisint, double[:,:] input, double[:,:] Na, double[:,:] S, double[:,:] T, double[:,:,:,:] ERI):
    cdef double calc, calc2, calc3, a, b, c, d, Ax, Bx, Cx, Dx, Ay, By, Cy, Dy, Az, Bz, Cz, Dz, Zc, Px, Py, Pz, Qx, Qy, Qz, p, q, RPC, RPQ, calct, calct2, N1, N2, N3, N4, c1, c2, c3, c4, aplha
    cdef double [:] Ex, Ey, Ez, E1, E2, E3, E4, E5, E6
    cdef double [:,:,:] R1
    cdef double [:,:,:,:,:] E1arr, E2arr, E3arr
    cdef int k, l, i, j, mu, nu, lam, sig, t, u, v, atom, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4
    
    # basisidx [number of primitives, start index in basisfloat and basisint]
    # basisfloat array of float values for basis, N, zeta, c, x, y, z 
    # basisint array of integer values for basis, l, m, n, atomidx
    
    # Making array to save E
    E1arr = np.zeros((len(basisidx),len(basisidx),len(basisint),len(basisint),np.max(basisint[:,0:3])*2+1))
    E2arr = np.zeros((len(basisidx),len(basisidx),len(basisint),len(basisint),np.max(basisint[:,0:3])*2+1))
    E3arr = np.zeros((len(basisidx),len(basisidx),len(basisint),len(basisint),np.max(basisint[:,0:3])*2+1))
    
    for k in range(0, len(basisidx)):
        for l in range(0, len(basisidx)):
            if k >= l:
                calc  = 0.0
                calc2 = 0.0
                calc3 = 0.0
                for i in range(basisidx[k,1],basisidx[k,1]+basisidx[k,0]):
                    for j in range(basisidx[l,1],basisidx[l,1]+basisidx[l,0]):
                        a  = basisfloat[i,1]
                        b  = basisfloat[j,1]
                        Ax = basisfloat[i,3]
                        Ay = basisfloat[i,4]
                        Az = basisfloat[i,5]
                        Bx = basisfloat[j,3]
                        By = basisfloat[j,4]
                        Bz = basisfloat[j,5]
                        l1 = basisint[i,0]
                        l2 = basisint[j,0]
                        m1 = basisint[i,1]
                        m2 = basisint[j,1]
                        n1 = basisint[i,2]
                        n2 = basisint[j,2]
                        N1 = basisfloat[i,0]
                        N2 = basisfloat[j,0]
                        c1 = basisfloat[i,2]
                        c2 = basisfloat[j,2]
                        
                        #E1, E2, E3, p, P is also used in ERI calculation, make smart later
                        p   = a+b
                        Px  = (a*Ax+b*Bx)/p
                        Py  = (a*Ay+b*By)/p
                        Pz  = (a*Az+b*Bz)/p
                        
                        Ex = np.zeros(l1+l2+1)
                        Ey = np.zeros(m1+m2+1)
                        Ez = np.zeros(n1+n2+1)
                        
                        for t in range(l1+l2+1):
                            Ex[t] = E1arr[k,l,i,j,t] = E(l1,l2,t,Ax-Bx,a,b,Px-Ax,Px-Bx,Ax-Bx)
                        for u in range(m1+m2+1):
                            Ey[u] = E2arr[k,l,i,j,u] = E(m1,m2,u,Ay-By,a,b,Py-Ay,Py-By,Ay-By)
                        for v in range(n1+n2+1):
                            Ez[v] = E3arr[k,l,i,j,v] = E(n1,n2,v,Az-Bz,a,b,Pz-Az,Pz-Bz,Az-Bz)

                        for atom in range(1, len(input)):
                            Zc = input[atom,0]
                            Cx = input[atom,1]
                            Cy = input[atom,2]
                            Cz = input[atom,3]
                            RPC = ((Px-Cx)**2+(Py-Cy)**2+(Pz-Cz)**2)**0.5
                            R1 = runR(l1+l2, m1+m2, n1+n2, Cx, Cy, Cz, Px, Py, Pz, p)

                            calc += elnuc(p, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, Zc, Ex, Ey, Ez, R1)
                        calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2)
                        calc2 += calct
                        calc3 += calct2
                        
                Na[k,l] = Na[l,k] = calc
                S[k,l]  = S[l,k]  = calc3
                T[k,l]  = T[l,k]  = calc2
    #END OF one electron integrals
    
    # Run ERI
    for mu in range(0, len(basisidx)):
        for nu in range(0, len(basisidx)):
            if mu >= nu:
                for lam in range(0, len(basisidx)):
                    for sig in range(0, len(basisidx)):
                        munu = mu*(mu+1)/2+nu
                        lamsig = lam*(lam+1)/2+sig
                        if lam >= sig and munu >= lamsig:
                            calc = 0.0
                            for i in range(basisidx[mu,1],basisidx[mu,1]+basisidx[mu,0]):
                                for j in range(basisidx[nu,1],basisidx[nu,1]+basisidx[nu,0]):
                                    N1 = basisfloat[i,0]
                                    a  = basisfloat[i,1]
                                    c1 = basisfloat[i,2]
                                    Ax = basisfloat[i,3]
                                    Ay = basisfloat[i,4]
                                    Az = basisfloat[i,5]
                                    N2 = basisfloat[j,0]
                                    b  = basisfloat[j,1]
                                    c2 = basisfloat[j,2]
                                    Bx = basisfloat[j,3]
                                    By = basisfloat[j,4]
                                    Bz = basisfloat[j,5]
                                    l1 = basisint[i,0]
                                    m1 = basisint[i,1]
                                    n1 = basisint[i,2]
                                    l2 = basisint[j,0]
                                    m2 = basisint[j,1]
                                    n2 = basisint[j,2]

                                    p   = a+b
                                    Px  = (a*Ax+b*Bx)/p
                                    Py  = (a*Ay+b*By)/p
                                    Pz  = (a*Az+b*Bz)/p
                                    
                                    E1 = E1arr[mu,nu,i,j]
                                    E2 = E2arr[mu,nu,i,j]
                                    E3 = E3arr[mu,nu,i,j]

                                    for k in range(basisidx[lam,1],basisidx[lam,1]+basisidx[lam,0]):
                                        for l in range(basisidx[sig,1],basisidx[sig,1]+basisidx[sig,0]):
                                            N3 = basisfloat[k,0]
                                            c  = basisfloat[k,1]
                                            c3 = basisfloat[k,2]
                                            Cx = basisfloat[k,3]
                                            Cy = basisfloat[k,4]
                                            Cz = basisfloat[k,5]
                                            N4 = basisfloat[l,0]
                                            d  = basisfloat[l,1]
                                            c4 = basisfloat[l,2]
                                            Dx = basisfloat[l,3]
                                            Dy = basisfloat[l,4]
                                            Dz = basisfloat[l,5]
                                            l3 = basisint[k,0]
                                            m3 = basisint[k,1]
                                            n3 = basisint[k,2]
                                            l4 = basisint[l,0]
                                            m4 = basisint[l,1]
                                            n4 = basisint[l,2]
                                                                                        
                                            q   = c+d
                                            Qx  = (c*Cx+d*Dx)/q
                                            Qy  = (c*Cy+d*Dy)/q
                                            Qz  = (c*Cz+d*Dz)/q
        
                                            E4 = E1arr[lam,sig,k,l]
                                            E5 = E2arr[lam,sig,k,l]
                                            E6 = E3arr[lam,sig,k,l]
                                            
                                            alpha = p*q/(p+q)
                                            RPQ = ((Px-Qx)**2+(Py-Qy)**2+(Pz-Qz)**2)**0.5
                                            
                                            R1 = runR(l1+l2+l3+l4, m1+m2+m3+m4, n1+n2+n3+n4, Qx, Qy, Qz, Px, Py, Pz, alpha)
                                            calc += elelrep(p,q,l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, N3, N4, c1, c2, c3, c4, E1, E2, E3, E4, E5, E6, R1)
                                
                            ERI[mu,nu,lam,sig] = ERI[nu,mu,lam,sig] = ERI[mu,nu,sig,lam] = ERI[nu,mu,sig,lam] = ERI[lam,sig,mu,nu] = ERI[sig,lam,mu,nu] = ERI[lam,sig,nu,mu] = ERI[sig,lam,nu,mu] = calc            
                                
    #END OF run ERI
    return Na, S, T, ERI

cpdef runE(int [:,:] basisidx, double [:,:] basisfloat, int [:,:] basisint, double [:,:] input):
    # Used to precalc E for geometric derivatives
    cdef double a, b, Ax, Bx, Ay, By, Az, Bz, Px, Py, Pz, p
    cdef int k, l, i, j, t, u, v, l1, l2, m1, m2, n1, n2
    
    E1arr = np.zeros((len(basisidx),len(basisidx),len(basisint),len(basisint),np.max(basisint[:,0:3])*2+1))
    E2arr = np.zeros((len(basisidx),len(basisidx),len(basisint),len(basisint),np.max(basisint[:,0:3])*2+1))
    E3arr = np.zeros((len(basisidx),len(basisidx),len(basisint),len(basisint),np.max(basisint[:,0:3])*2+1))
    
    for k in range(0, len(basisidx)):
        for l in range(0, len(basisidx)):
            if k >= l:
                for i in range(basisidx[k,1],basisidx[k,1]+basisidx[k,0]):
                    for j in range(basisidx[l,1],basisidx[l,1]+basisidx[l,0]):
                        a  = basisfloat[i,1]
                        b  = basisfloat[j,1]
                        Ax = basisfloat[i,3]
                        Ay = basisfloat[i,4]
                        Az = basisfloat[i,5]
                        Bx = basisfloat[j,3]
                        By = basisfloat[j,4]
                        Bz = basisfloat[j,5]
                        l1 = basisint[i,0]
                        l2 = basisint[j,0]
                        m1 = basisint[i,1]
                        m2 = basisint[j,1]
                        n1 = basisint[i,2]
                        n2 = basisint[j,2]
                        
                        #E1, E2, E3 is also used in ERI calculation
                        p   = a+b
                        Px  = (a*Ax+b*Bx)/p
                        Py  = (a*Ay+b*By)/p
                        Pz  = (a*Az+b*Bz)/p
                        
                        for t in range(l1+l2+1):
                            E1arr[k,l,i,j,t] = E(l1,l2,t,Ax-Bx,a,b,Px-Ax,Px-Bx,Ax-Bx)
                        for u in range(m1+m2+1):
                            E2arr[k,l,i,j,u] = E(m1,m2,u,Ay-By,a,b,Py-Ay,Py-By,Ay-By)
                        for v in range(n1+n2+1):
                            E3arr[k,l,i,j,v] = E(n1,n2,v,Az-Bz,a,b,Pz-Az,Pz-Bz,Az-Bz)
    
    return E1arr, E2arr, E3arr


cpdef runCythonRunGeoDev(int [:,:] basisidx, double [:,:] basisfloat, int [:,:] basisint, double[:,:] input, double [:,:,:,:,:] E1arr, double [:,:,:,:,:] E2arr, double [:,:,:,:,:] E3arr, double[:,:] Sxarr, double[:,:] Syarr, double[:,:] Szarr, double[:,:] Txarr, double[:,:] Tyarr, double[:,:] Tzarr, double[:,:] VNexarr, double[:,:] VNeyarr, double[:,:] VNezarr, double[:,:,:,:] ERIx, double[:,:,:,:] ERIy, double[:,:,:,:] ERIz, int atomidx):
    cdef double a, b, c, d, Ax, Bx, Cx, Dx, Ay, By, Cy, Dy, Az, Bz, Cz, Dz, Zc, Px, Py, Pz, Qx, Qy, Qz, p, q, RPC, RPQ, N1, N2, N3, N4, c1, c2, c3, c4, aplha, Nxp1, Nxm1, Nyp1, Nym1, Nzp1, Nzm1, Nxp2, Nxm2, Nyp2, Nym2, Nzp2, Nzm2, Nxp3, Nxm3, Nyp3, Nym3, Nzp3, Nzm3, Nxp4, Nxm4, Nyp4, Nym4, Nzp4, Nzm4, Tx, Ty, Tz, Sx, Sy, Sz, VNex, VNey, VNez, calcx, calcy, calcz, calct, calct2
    cdef double [:] E1, E2, E3, E4, E5, E6, E1p, E1m, E2p, E2m, E3p, E3m, E4p, E4m, E5p, E5m, E6p, E6m, Ex, Ey, Ez, Exp, Exm, Eyp, Eym, Ezp, Ezm
    cdef double [:,:,:] R1
    cdef int k, l, i, j, mu, nu, lam, sig, t, u, v, atom, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, atomidx_k, atomidx_l, atomidx_mu, atomidx_nu, atomidx_lam, atomidx_sig
    
    # One electron integrals
    for k in range(0, len(basisidx)):
        for l in range(0, len(basisidx)):
            if k >= l:
                Tx = 0.0
                Ty = 0.0
                Tz = 0.0
                Sx = 0.0
                Sy = 0.0
                Sz = 0.0
                VNex = 0.0
                VNey = 0.0
                VNez = 0.0
                for i in range(basisidx[k,1],basisidx[k,1]+basisidx[k,0]):
                    for j in range(basisidx[l,1],basisidx[l,1]+basisidx[l,0]):
                        # basisfloat array of float values for basis, N, zeta, c, x, y, z, Nxp, Nxm, Nyp, Nym, Nzp, Nzm 
                        a  = basisfloat[i,1]
                        b  = basisfloat[j,1]
                        Ax = basisfloat[i,3]
                        Ay = basisfloat[i,4]
                        Az = basisfloat[i,5]
                        Bx = basisfloat[j,3]
                        By = basisfloat[j,4]
                        Bz = basisfloat[j,5]
                        l1 = basisint[i,0]
                        l2 = basisint[j,0]
                        m1 = basisint[i,1]
                        m2 = basisint[j,1]
                        n1 = basisint[i,2]
                        n2 = basisint[j,2]
                        N1 = basisfloat[i,0]
                        N2 = basisfloat[j,0]
                        c1 = basisfloat[i,2]
                        c2 = basisfloat[j,2]
                        
                        Nxp1=basisfloat[i,6]
                        Nxm1=basisfloat[i,7]
                        Nyp1=basisfloat[i,8]
                        Nym1=basisfloat[i,9]
                        Nzp1=basisfloat[i,10]
                        Nzm1=basisfloat[i,11]
                        
                        Nxp2=basisfloat[j,6]
                        Nxm2=basisfloat[j,7]
                        Nyp2=basisfloat[j,8]
                        Nym2=basisfloat[j,9]
                        Nzp2=basisfloat[j,10]
                        Nzm2=basisfloat[j,11]
                        
                        Ex = E1arr[k,l,i,j]
                        Ey = E2arr[k,l,i,j]
                        Ez = E3arr[k,l,i,j]
                        
                        p   = a+b
                        Px  = (a*Ax+b*Bx)/p
                        Py  = (a*Ay+b*By)/p
                        Pz  = (a*Az+b*Bz)/p
                        
                        atomidx_k = basisint[i,3]
                        atomidx_l = basisint[j,3]
                            
                        if atomidx == atomidx_k and atomidx == atomidx_l:
                            None
                        else:
                            # OVERLAP AND KINETIC ENERGY
                            # x derivative
                            if atomidx == atomidx_k:
                                calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1+1, l2, m1, m2, n1, n2, Nxp1, N2, c1, c2)
                                Tx += calct
                                Sx += calct2
                                if l1 != 0:
                                    calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1-1, l2, m1, m2, n1, n2, Nxm2, N2, c1, c2)
                                    Tx += calct
                                    Sx += calct2
                                    
                            if atomidx == atomidx_l:
                                calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2+1, m1, m2, n1, n2, N1, Nxp2, c1, c2)
                                Tx += calct
                                Sx += calct2
                                if l2 != 0:
                                    calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2-1, m1, m2, n1, n2, N1, Nxm2, c1, c2)
                                    Tx += calct
                                    Sx += calct2
                            
                            # y derivative
                            if atomidx == atomidx_k:
                                calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1+1, m2, n1, n2, Nyp1, N2, c1, c2)
                                Ty += calct
                                Sy += calct2
                                if m1 != 0:
                                    calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1-1, m2, n1, n2, Nym1, N2, c1, c2)
                                    Ty += calct
                                    Sy += calct2
                                    
                            if atomidx == atomidx_l:
                                calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2+1, n1, n2, N1, Nyp2, c1, c2)
                                Ty += calct
                                Sy += calct2
                                if m2 != 0:
                                    calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2-1, n1, n2, N1, Nym2, c1, c2)
                                    Ty += calct
                                    Sy += calct2
                            
                            # z derivative
                            if atomidx == atomidx_k:
                                calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1+1, n2, Nzp1, N2, c1, c2)
                                Tz += calct
                                Sz += calct2
                                if n1 != 0:
                                    calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1-1, n2, Nzm1, N2, c1, c2)
                                    Tz += calct
                                    Sz += calct2
                                    
                            if atomidx == atomidx_l:
                                calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2+1, N1, Nzp2, c1, c2)
                                Tz += calct
                                Sz += calct2
                                if n2 != 0:
                                    calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2-1, N1, Nzm2, c1, c2)
                                    Tz += calct
                                    Sz += calct2
                        
                        # NUCLEUS NUCLEUS ENERGY
                        for atom in range(1, len(input)):
                            if atomidx == atomidx_k or atomidx == atomidx_l:
                                Zc = input[atom,0]
                                Cx = input[atom,1]
                                Cy = input[atom,2]
                                Cz = input[atom,3]
                                RPC = ((Px-Cx)**2+(Py-Cy)**2+(Pz-Cz)**2)**0.5
                                R1 = runR(l1+l2, m1+m2, n1+n2, Cx, Cy, Cz, Px, Py, Pz, p, check=1)
                        # x derivative
                                if atomidx == atomidx_k and atomidx == atomidx_l:
                                    if atom != atomidx:
                                        Exp = np.zeros(l1+1+l2+1)
                                        for t in range(0, l1+1+l2+1):
                                            Exp[t] = E(l1+1,l2,t,Ax-Bx,a,b,Px-Ax,Px-Bx,Ax-Bx)
                                        VNex += elnuc(p, l1+1, l2, m1, m2, n1, n2, Nxp1, N2, c1, c2, Zc, Exp, Ey, Ez,R1)
                                        if l1 != 0:
                                            Exm = np.zeros(l1-1+l2+1)
                                            for t in range(0, l1-1+l2+1):
                                                Exm[t] = E(l1-1,l2,t,Ax-Bx,a,b,Px-Ax,Px-Bx,Ax-Bx)
                                            VNex += elnuc(p, l1-1, l2, m1, m2, n1, n2, Nxm1, N2, c1, c2, Zc, Exm, Ey, Ez,R1)
                                        
                                        Exp = np.zeros(l1+1+l2+1)
                                        for t in range(0, l1+1+l2+1):
                                            Exp[t] = E(l1,l2+1,t,Ax-Bx,a,b,Px-Ax,Px-Bx,Ax-Bx)
                                        VNex += elnuc(p, l1, l2+1, m1, m2, n1, n2, N1, Nxp2, c1, c2, Zc, Exp, Ey, Ez,R1)
                                        if l2 != 0:
                                            Exm = np.zeros(l1-1+l2+1)
                                            for t in range(0, l1+l2-1+1):
                                                Exm[t] = E(l1,l2-1,t,Ax-Bx,a,b,Px-Ax,Px-Bx,Ax-Bx)
                                            VNex += elnuc(p, l1, l2-1, m1, m2, n1, n2, N1, Nxm2, c1, c2, Zc, Exm, Ey, Ez,R1)
                                
                                else:
                                    if atomidx == atomidx_k:
                                        Exp = np.zeros(l1+1+l2+1)
                                        for t in range(0, l1+1+l2+1):
                                            Exp[t] = E(l1+1,l2,t,Ax-Bx,a,b,Px-Ax,Px-Bx,Ax-Bx)
                                        VNex += elnuc(p, l1+1, l2, m1, m2, n1, n2, Nxp1, N2, c1, c2, Zc, Exp, Ey, Ez,R1)
                                        if l1 != 0:
                                            Exm = np.zeros(l1-1+l2+1)
                                            for t in range(0, l1-1+l2+1):
                                                Exm[t] = E(l1-1,l2,t,Ax-Bx,a,b,Px-Ax,Px-Bx,Ax-Bx)
                                            VNex += elnuc(p, l1-1, l2, m1, m2, n1, n2, Nxm1, N2, c1, c2, Zc, Exm, Ey, Ez,R1)
            
                                    if atomidx == atomidx_l:

                                        Exp = np.zeros(l1+1+l2+1)
                                        for t in range(0, l1+1+l2+1):
                                            Exp[t] = E(l1,l2+1,t,Ax-Bx,a,b,Px-Ax,Px-Bx,Ax-Bx)
                                        VNex += elnuc(p, l1, l2+1, m1, m2, n1, n2, N1, Nxp2, c1, c2, Zc, Exp, Ey, Ez,R1)
                                        if l2 != 0:
                                            Exm = np.zeros(l1-1+l2+1)
                                            for t in range(0, l1+l2-1+1):
                                                Exm[t] = E(l1,l2-1,t,Ax-Bx,a,b,Px-Ax,Px-Bx,Ax-Bx)
                                            VNex += elnuc(p, l1, l2-1, m1, m2, n1, n2, N1, Nxm2, c1, c2, Zc, Exm, Ey, Ez,R1)
                        # y derivative
                                if atomidx == atomidx_k and atomidx == atomidx_l:
                                    if atom != atomidx:
                                        Eyp = np.zeros(m1+1+m2+1)
                                        for t in range(0, m1+1+m2+1):
                                            Eyp[t] = E(m1+1,m2,t,Ay-By,a,b,Py-Ay,Py-By,Ay-By)
                                        VNey += elnuc(p, l1, l2, m1+1, m2, n1, n2, Nyp1, N2, c1, c2, Zc, Ex, Eyp, Ez,R1)
                                        if m1 != 0:
                                            Eym = np.zeros(m1-1+m2+1)
                                            for t in range(0, m1-1+m2+1):
                                                Eym[t] = E(m1-1,m2,t,Ay-By,a,b,Py-Ay,Py-By,Ay-By)
                                            VNey += elnuc(p, l1, l2, m1-1, m2, n1, n2, Nym1, N2, c1, c2, Zc, Ex, Eym, Ez,R1)
                                        
                                        Eyp = np.zeros(m1+1+m2+1)
                                        for t in range(0, m1+1+m2+1):
                                            Eyp[t] = E(m1,m2+1,t,Ay-By,a,b,Py-Ay,Py-By,Ay-By)
                                        VNey += elnuc(p, l1, l2, m1, m2+1, n1, n2, N1, Nyp2, c1, c2, Zc, Ex, Eyp, Ez,R1)
                                        if m2 != 0:
                                            Eym = np.zeros(m1-1+m2+1)
                                            for t in range(0, m1+m2-1+1):
                                                Eym[t] = E(m1,m2-1,t,Ay-By,a,b,Py-Ay,Py-By,Ay-By)
                                            VNey += elnuc(p, l1, l2, m1, m2-1, n1, n2, N1, Nym2, c1, c2, Zc, Ex, Eym, Ez,R1)
                                
                                else:
                                    if atomidx == atomidx_k:
                                        Eyp = np.zeros(m1+1+m2+1)
                                        for t in range(0, m1+1+m2+1):
                                            Eyp[t] = E(m1+1,m2,t,Ay-By,a,b,Py-Ay,Py-By,Ay-By)
                                        VNey += elnuc(p, l1, l2, m1+1, m2, n1, n2, Nyp1, N2, c1, c2, Zc, Ex, Eyp, Ez,R1)
                                        if m1 != 0:
                                            Eym = np.zeros(m1-1+m2+1)
                                            for t in range(0, m1-1+m2+1):
                                                Eym[t] = E(m1-1,m2,t,Ay-By,a,b,Py-Ay,Py-By,Ay-By)
                                            VNey += elnuc(p, l1, l2, m1-1, m2, n1, n2, Nym1, N2, c1, c2, Zc, Ex, Eym, Ez,R1)
            
                                    if atomidx == atomidx_l:
                                        Eyp = np.zeros(m1+1+m2+1)
                                        for t in range(0, m1+1+m2+1):
                                            Eyp[t] = E(m1,m2+1,t,Ay-By,a,b,Py-Ay,Py-By,Ay-By)
                                        VNey += elnuc(p, l1, l2, m1, m2+1, n1, n2, N1, Nyp2, c1, c2, Zc, Ex, Eyp, Ez,R1)
                                        if m2 != 0:
                                            Eym = np.zeros(m1-1+m2+1)
                                            for t in range(0, m1+m2-1+1):
                                                Eym[t] = E(m1,m2-1,t,Ay-By,a,b,Py-Ay,Py-By,Ay-By)
                                            VNey += elnuc(p, l1, l2, m1, m2-1, n1, n2, N1, Nym2, c1, c2, Zc, Ex, Eym, Ez,R1)
                                    
                        # z derivative
                                if atomidx == atomidx_k and atomidx == atomidx_l:
                                    if atom != atomidx:
                                        Ezp = np.zeros(n1+1+n2+1)
                                        for t in range(0, n1+1+n2+1):
                                            Ezp[t] = E(n1+1,n2,t,Az-Bz,a,b,Pz-Az,Pz-Bz,Az-Bz)
                                        VNez += elnuc(p, l1, l2, m1, m2, n1+1, n2, Nzp1, N2, c1, c2, Zc, Ex, Ey, Ezp,R1)
                                        if n1 != 0:
                                            Ezm = np.zeros(n1-1+n2+1)
                                            for t in range(0, n1-1+n2+1):
                                                Ezm[t] = E(n1-1,n2,t,Az-Bz,a,b,Pz-Az,Pz-Bz,Az-Bz)
                                            VNez += elnuc(p, l1, l2, m1, m2, n1-1, n2, Nzm1, N2, c1, c2, Zc, Ex, Ey, Ezm,R1)
                                        
                                        Ezp = np.zeros(n1+1+n2+1)
                                        for t in range(0, n1+1+n2+1):
                                            Ezp[t] = E(n1,n2+1,t,Az-Bz,a,b,Pz-Az,Pz-Bz,Az-Bz)
                                        VNez += elnuc(p, l1, l2, m1, m2, n1, n2+1, N1, Nzp2, c1, c2, Zc, Ex, Ey, Ezp,R1)
                                        if n2 != 0:
                                            Ezm = np.zeros(n1-1+n2+1)
                                            for t in range(0, n1+n2-1+1):
                                                Ezm[t] = E(n1,n2-1,t,Az-Bz,a,b,Pz-Az,Pz-Bz,Az-Bz)
                                            VNez += elnuc(p, l1, l2, m1, m2, n1, n2-1, N1, Nzm2, c1, c2, Zc, Ex, Ey, Ezm,R1)
                                
                                else:
                                    if atomidx == atomidx_k:
                                        Ezp = np.zeros(n1+1+n2+1)
                                        for t in range(0, n1+1+n2+1):
                                            Ezp[t] = E(n1+1,n2,t,Az-Bz,a,b,Pz-Az,Pz-Bz,Az-Bz)
                                        VNez += elnuc(p, l1, l2, m1, m2, n1+1, n2, Nzp1, N2, c1, c2, Zc, Ex, Ey, Ezp,R1)
                                        if n1 != 0:
                                            Ezm = np.zeros(n1-1+n2+1)
                                            for t in range(0, n1-1+n2+1):
                                                Ezm[t] = E(n1-1,n2,t,Az-Bz,a,b,Pz-Az,Pz-Bz,Az-Bz)
                                            VNez += elnuc(p, l1, l2, m1, m2, n1-1, n2, Nzm1, N2, c1, c2, Zc, Ex, Ey, Ezm,R1)
            
                                    if atomidx == atomidx_l:
                                        Ezp = np.zeros(n1+1+n2+1)
                                        for t in range(0, n1+1+n2+1):
                                            Ezp[t] = E(n1,n2+1,t,Az-Bz,a,b,Pz-Az,Pz-Bz,Az-Bz)
                                        VNez += elnuc(p, l1, l2, m1, m2, n1, n2+1, N1, Nzp2, c1, c2, Zc, Ex, Ey, Ezp,R1)
                                        if n2 != 0:
                                            Ezm = np.zeros(n1-1+n2+1)
                                            for t in range(0, n1+n2-1+1):
                                                Ezm[t] = E(n1,n2-1,t,Az-Bz,a,b,Pz-Az,Pz-Bz,Az-Bz)
                                            VNez += elnuc(p, l1, l2, m1, m2, n1, n2-1, N1, Nzm2, c1, c2, Zc, Ex, Ey, Ezm,R1)
                        
                        # Electricfield contribution
                        if atomidx == atomidx_k and atomidx == atomidx_l:
                            None

                        else:
                            Zc = input[atomidx,0]
                            Cx = input[atomidx,1]
                            Cy = input[atomidx,2]
                            Cz = input[atomidx,3]
                            RPC = ((Px-Cx)**2+(Py-Cy)**2+(Pz-Cz)**2)**0.5
                            R1 = runR(l1+l2, m1+m2, n1+n2, Cx, Cy, Cz, Px, Py, Pz, p, check=1)
                            VNex += electricfield(p,Ex,Ey,Ez,Zc, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, 1,R1)
                            VNey += electricfield(p,Ex,Ey,Ez,Zc, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, 2,R1)
                            VNez += electricfield(p,Ex,Ey,Ez,Zc, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, 3,R1)
                        
                Sxarr[k,l] = Sxarr[l,k] = Sx
                Syarr[k,l] = Syarr[l,k] = Sy
                Szarr[k,l] = Szarr[l,k] = Sz
                Txarr[k,l] = Txarr[l,k] = Tx
                Tyarr[k,l] = Tyarr[l,k] = Ty
                Tzarr[k,l] = Tzarr[l,k] = Tz
                VNexarr[k,l] = VNexarr[l,k] = VNex
                VNeyarr[k,l] = VNeyarr[l,k] = VNey
                VNezarr[k,l] = VNezarr[l,k] = VNez
        
    
    # ERI integrals
    for mu in range(0, len(basisidx)):
        for nu in range(0, len(basisidx)):
            if mu >= nu:
                for lam in range(0, len(basisidx)):
                    for sig in range(0, len(basisidx)):
                        munu = mu*(mu+1)/2+nu
                        lamsig = lam*(lam+1)/2+sig
                        if lam >= sig and munu >= lamsig:
                            calcx = 0.0
                            calcy = 0.0
                            calcz = 0.0
                            
                            # All primitives have identical coords if the same atom
                            # can just choose the first one to get atomix
                            atomidx_mu = basisint[basisidx[mu,1],3]
                            atomidx_nu = basisint[basisidx[nu,1],3]
                            atomidx_lam= basisint[basisidx[lam,1],3]
                            atomidx_sig = basisint[basisidx[sig,1],3]
                            
                            if atomidx == atomidx_mu and atomidx == atomidx_nu and atomidx == atomidx_lam and atomidx == atomidx_sig:
                                None
                            else:
                                for i in range(basisidx[mu,1],basisidx[mu,1]+basisidx[mu,0]):
                                    for j in range(basisidx[nu,1],basisidx[nu,1]+basisidx[nu,0]):
                                        N1 = basisfloat[i,0]
                                        a  = basisfloat[i,1]
                                        c1 = basisfloat[i,2]
                                        Ax = basisfloat[i,3]
                                        Ay = basisfloat[i,4]
                                        Az = basisfloat[i,5]
                                        N2 = basisfloat[j,0]
                                        b  = basisfloat[j,1]
                                        c2 = basisfloat[j,2]
                                        Bx = basisfloat[j,3]
                                        By = basisfloat[j,4]
                                        Bz = basisfloat[j,5]
                                        l1 = basisint[i,0]
                                        m1 = basisint[i,1]
                                        n1 = basisint[i,2]
                                        l2 = basisint[j,0]
                                        m2 = basisint[j,1]
                                        n2 = basisint[j,2]
    
                                        p   = a+b
                                        Px  = (a*Ax+b*Bx)/p
                                        Py  = (a*Ay+b*By)/p
                                        Pz  = (a*Az+b*Bz)/p
                                        
                                        E1 = E1arr[mu,nu,i,j]
                                        E2 = E2arr[mu,nu,i,j]
                                        E3 = E3arr[mu,nu,i,j]
                                        
                                        Nxp1=basisfloat[i,6]
                                        Nxm1=basisfloat[i,7]
                                        Nyp1=basisfloat[i,8]
                                        Nym1=basisfloat[i,9]
                                        Nzp1=basisfloat[i,10]
                                        Nzm1=basisfloat[i,11]
                                        
                                        Nxp2=basisfloat[j,6]
                                        Nxm2=basisfloat[j,7]
                                        Nyp2=basisfloat[j,8]
                                        Nym2=basisfloat[j,9]
                                        Nzp2=basisfloat[j,10]
                                        Nzm2=basisfloat[j,11]
    
                                        for k in range(basisidx[lam,1],basisidx[lam,1]+basisidx[lam,0]):
                                            for l in range(basisidx[sig,1],basisidx[sig,1]+basisidx[sig,0]):
                                                N3 = basisfloat[k,0]
                                                c  = basisfloat[k,1]
                                                c3 = basisfloat[k,2]
                                                Cx = basisfloat[k,3]
                                                Cy = basisfloat[k,4]
                                                Cz = basisfloat[k,5]
                                                N4 = basisfloat[l,0]
                                                d  = basisfloat[l,1]
                                                c4 = basisfloat[l,2]
                                                Dx = basisfloat[l,3]
                                                Dy = basisfloat[l,4]
                                                Dz = basisfloat[l,5]
                                                l3 = basisint[k,0]
                                                m3 = basisint[k,1]
                                                n3 = basisint[k,2]
                                                l4 = basisint[l,0]
                                                m4 = basisint[l,1]
                                                n4 = basisint[l,2]
                                                                                            
                                                q   = c+d
                                                Qx  = (c*Cx+d*Dx)/q
                                                Qy  = (c*Cy+d*Dy)/q
                                                Qz  = (c*Cz+d*Dz)/q
            
                                                E4 = E1arr[lam,sig,k,l]
                                                E5 = E2arr[lam,sig,k,l]
                                                E6 = E3arr[lam,sig,k,l]
                                                
                                                Nxp3=basisfloat[k,6]
                                                Nxm3=basisfloat[k,7]
                                                Nyp3=basisfloat[k,8]
                                                Nym3=basisfloat[k,9]
                                                Nzp3=basisfloat[k,10]
                                                Nzm3=basisfloat[k,11]
                                                
                                                Nxp4=basisfloat[l,6]
                                                Nxm4=basisfloat[l,7]
                                                Nyp4=basisfloat[l,8]
                                                Nym4=basisfloat[l,9]
                                                Nzp4=basisfloat[l,10]
                                                Nzm4=basisfloat[l,11]
                                                
                                                alpha = p*q/(p+q)
                                                RPQ = ((Px-Qx)**2+(Py-Qy)**2+(Pz-Qz)**2)**0.5
                                                
                                                R1 = runR(l1+l2+l3+l4, m1+m2+m3+m4, n1+n2+n3+n4, Qx, Qy, Qz, Px, Py, Pz, alpha, check=1)
                                                
                                                # Calcul1te x derivative
                                                if atomidx == atomidx_mu:
                                                    E1p = np.zeros(l1+1+l2+1)
                                                    for t in range(l1+1+l2+1):
                                                        E1p[t] = E(l1+1,l2,t,Ax-Bx,a,b,Px-Ax,Px-Bx,Ax-Bx)
                                                    calcx += elelrep(p, q, l1+1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, Nxp1, N2, N3, N4, c1, c2, c3, c4, E1p,E2,E3,E4,E5,E6,R1)
                                                    if l1 != 0:
                                                        E1m = np.zeros(l1-1+l2+1)
                                                        for t in range(l1-1+l2+1):
                                                            E1m[t] = E(l1-1,l2,t,Ax-Bx,a,b,Px-Ax,Px-Bx,Ax-Bx)
                                                        calcx += elelrep(p, q,l1-1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, Nxm1, N2, N3, N4, c1, c2, c3, c4, E1m,E2,E3,E4,E5,E6,R1)
                                                        
                                                if atomidx == atomidx_nu:
                                                    E1p = np.zeros(l1+1+l2+1)
                                                    for t in range(l1+l2+1+1):
                                                        E1p[t] = E(l1,l2+1,t,Ax-Bx,a,b,Px-Ax,Px-Bx,Ax-Bx)
                                                    calcx += elelrep(p, q, l1, l2+1, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, Nxp2, N3, N4, c1, c2, c3, c4, E1p,E2,E3,E4,E5,E6,R1)
                                                    if l2 != 0:
                                                        E1m = np.zeros(l1-1+l2+1)
                                                        for t in range(l1+l2-1+1):
                                                            E1m[t] = E(l1,l2-1,t,Ax-Bx,a,b,Px-Ax,Px-Bx,Ax-Bx)
                                                        calcx += elelrep(p, q, l1, l2-1, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, Nxm2, N3, N4, c1, c2, c3, c4, E1m,E2,E3,E4,E5,E6,R1)
                                                        
                                                if atomidx == atomidx_lam:  
                                                    E4p = np.zeros(l3+1+l4+1)
                                                    for t in range(l3+1+l4+1):
                                                        E4p[t] = E(l3+1,l4,t,Cx-Dx,c,d,Qx-Cx,Qx-Dx,Cx-Dx)
                                                    calcx += elelrep(p, q, l1, l2, l3+1, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, Nxp3, N4, c1, c2, c3, c4, E1,E2,E3,E4p,E5,E6,R1)
                                                    if l3 != 0:
                                                        E4m = np.zeros(l3-1+l4+1)
                                                        for t in range(l3-1+l4+1):
                                                            E4m[t] = E(l3-1,l4,t,Cx-Dx,c,d,Qx-Cx,Qx-Dx,Cx-Dx)
                                                        calcx += elelrep(p, q, l1, l2, l3-1, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, Nxm3, N4, c1, c2, c3, c4, E1,E2,E3,E4m,E5,E6,R1)
                                                    
                                                if atomidx == atomidx_sig:
                                                    E4p = np.zeros(l3+1+l4+1)
                                                    for t in range(l3+l4+1+1):
                                                        E4p[t] = E(l3,l4+1,t,Cx-Dx,c,d,Qx-Cx,Qx-Dx,Cx-Dx)
                                                    calcx += elelrep(p, q, l1, l2, l3, l4+1, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, N3, Nxp4, c1, c2, c3, c4, E1,E2,E3,E4p,E5,E6,R1)
                                                    if l4 != 0:
                                                        E4m = np.zeros(l3-1+l4+1)
                                                        for t in range(l3+l4-1+1):
                                                            E4m[t] = E(l3,l4-1,t,Cx-Dx,c,d,Qx-Cx,Qx-Dx,Cx-Dx)
                                                        calcx += elelrep(p, q, l1, l2, l3, l4-1, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, N3, Nxm4, c1, c2, c3, c4, E1,E2,E3,E4m,E5,E6,R1)
                                                
                                                # Calcul1te y derivative
                                                if atomidx == atomidx_mu:
                                                    E2p = np.zeros(m1+1+m2+1)
                                                    for t in range(m1+1+m2+1):
                                                        E2p[t] = E(m1+1,m2,t,Ay-By,a,b,Py-Ay,Py-By,Ay-By)
                                                    calcy += elelrep(p, q, l1, l2, l3, l4, m1+1, m2, m3, m4, n1, n2, n3, n4, Nyp1, N2, N3, N4, c1, c2, c3, c4, E1,E2p,E3,E4,E5,E6,R1)
                                                    if m1 != 0:
                                                        E2m = np.zeros(m1-1+m2+1)
                                                        for t in range(m1-1+m2+1):
                                                            E2m[t] = E(m1-1,m2,t,Ay-By,a,b,Py-Ay,Py-By,Ay-By)
                                                        calcy += elelrep(p, q, l1, l2, l3, l4, m1-1, m2, m3, m4, n1, n2, n3, n4, Nym1, N2, N3, N4, c1, c2, c3, c4, E1,E2m,E3,E4,E5,E6,R1)
                                                        
                                                if atomidx == atomidx_nu:
                                                    E2p = np.zeros(m1+1+m2+1)
                                                    for t in range(m1+m2+1+1):
                                                        E2p[t] = E(m1,m2+1,t,Ay-By,a,b,Py-Ay,Py-By,Ay-By)
                                                    calcy += elelrep(p, q, l1, l2, l3, l4, m1, m2+1, m3, m4, n1, n2, n3, n4, N1, Nyp2, N3, N4, c1, c2, c3, c4, E1,E2p,E3,E4,E5,E6,R1)
                                                    if m2 != 0:
                                                        E2m = np.zeros(m1-1+m2+1)
                                                        for t in range(m1+m2-1+1):
                                                            E2m[t] = E(m1,m2-1,t,Ay-By,a,b,Py-Ay,Py-By,Ay-By)
                                                        calcy += elelrep(p, q, l1, l2, l3, l4, m1, m2-1, m3, m4, n1, n2, n3, n4, N1, Nym2, N3, N4, c1, c2, c3, c4,E1,E2m,E3,E4,E5,E6,R1)
                                                        
                                                if atomidx == atomidx_lam:  
                                                    E5p = np.zeros(m3+1+m4+1)
                                                    for t in range(m3+1+m4+1):
                                                        E5p[t] = E(m3+1,m4,t,Cy-Dy,c,d,Qy-Cy,Qy-Dy,Cy-Dy)
                                                    calcy += elelrep(p, q, l1, l2, l3, l4, m1, m2, m3+1, m4, n1, n2, n3, n4, N1, N2, Nyp3, N4, c1, c2, c3, c4, E1,E2,E3,E4,E5p,E6,R1)
                                                    if m3 != 0:
                                                        E5m = np.zeros(m3-1+m4+1)
                                                        for t in range(m3-1+m4+1):
                                                            E5m[t] = E(m3-1,m4,t,Cy-Dy,c,d,Qy-Cy,Qy-Dy,Cy-Dy)
                                                        calcy += elelrep(p, q, l1, l2, l3, l4, m1, m2, m3-1, m4, n1, n2, n3, n4, N1, N2, Nym3, N4, c1, c2, c3, c4,E1,E2,E3,E4,E5m,E6,R1)
                                                    
                                                if atomidx == atomidx_sig:
                                                    E5p = np.zeros(m3+1+m4+1)
                                                    for t in range(m3+m4+1+1):
                                                        E5p[t] = E(m3,m4+1,t,Cy-Dy,c,d,Qy-Cy,Qy-Dy,Cy-Dy)
                                                    calcy += elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4+1, n1, n2, n3, n4, N1, N2, N3, Nyp4, c1, c2, c3, c4,E1,E2,E3,E4,E5p,E6,R1)
                                                    if m4 != 0:
                                                        E5m = np.zeros(m3-1+m4+1)
                                                        for t in range(m3+m4-1+1):
                                                            E5m[t] = E(m3,m4-1,t,Cy-Dy,c,d,Qy-Cy,Qy-Dy,Cy-Dy)
                                                        calcy += elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4-1, n1, n2, n3, n4, N1, N2, N3, Nym4, c1, c2, c3, c4,E1,E2,E3,E4,E5m,E6,R1)
                                                
                                                # Calcul1te z derivative
                                                if atomidx == atomidx_mu:
                                                    E3p = np.zeros(n1+1+n2+1)
                                                    for t in range(n1+1+n2+1):
                                                        E3p[t] = E(n1+1,n2,t,Az-Bz,a,b,Pz-Az,Pz-Bz,Az-Bz)
                                                    calcz += elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1+1, n2, n3, n4, Nzp1, N2, N3, N4, c1, c2, c3, c4, E1,E2,E3p,E4,E5,E6,R1)
                                                    if n1 != 0:
                                                        E3m = np.zeros(n1-1+n2+1)
                                                        for t in range(n1-1+n2+1):
                                                            E3m[t] = E(n1-1,n2,t,Az-Bz,a,b,Pz-Az,Pz-Bz,Az-Bz)
                                                        calcz += elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1-1, n2, n3, n4, Nzm1, N2, N3, N4, c1, c2, c3, c4, E1,E2,E3m,E4,E5,E6,R1)
                                                        
                                                if atomidx == atomidx_nu:
                                                    E3p = np.zeros(n1+1+n2+1)
                                                    for t in range(n1+n2+1+1):
                                                        E3p[t] = E(n1,n2+1,t,Az-Bz,a,b,Pz-Az,Pz-Bz,Az-Bz)
                                                    calcz += elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2+1, n3, n4, N1, Nzp2, N3, N4, c1, c2, c3, c4, E1,E2,E3p,E4,E5,E6,R1)
                                                    if n2 != 0:
                                                        E3m = np.zeros(n1-1+n2+1)
                                                        for t in range(n1+n2-1+1):
                                                            E3m[t] = E(n1,n2-1,t,Az-Bz,a,b,Pz-Az,Pz-Bz,Az-Bz)
                                                        calcz += elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2-1, n3, n4, N1, Nzm2, N3, N4, c1, c2, c3, c4,E1,E2,E3m,E4,E5,E6,R1)
                                                        
                                                if atomidx == atomidx_lam: 
                                                    E6p = np.zeros(n3+1+n4+1)
                                                    for t in range(n3+1+n4+1):
                                                        E6p[t] = E(n3+1,n4,t,Cz-Dz,c,d,Qz-Cz,Qz-Dz,Cz-Dz)
                                                    calcz += elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3+1, n4, N1, N2, Nzp3, N4, c1, c2, c3, c4, E1,E2,E3,E4,E5,E6p,R1)
                                                    if n3 != 0:
                                                        E6m = np.zeros(n3-1+n4+1)
                                                        for t in range(n3-1+n4+1):
                                                            E6m[t] = E(n3-1,n4,t,Cz-Dz,c,d,Qz-Cz,Qz-Dz,Cz-Dz)
                                                        calcz += elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3-1, n4, N1, N2, Nzm3, N4, c1, c2, c3, c4,E1,E2,E3,E4,E5,E6m,R1)
                                                    
                                                if atomidx == atomidx_sig:
                                                    E6p = np.zeros(n3+1+n4+1)
                                                    for t in range(n3+n4+1+1):
                                                        E6p[t] = E(n3,n4+1,t,Cz-Dz,c,d,Qz-Cz,Qz-Dz,Cz-Dz)
                                                    calcz += elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4+1, N1, N2, N3, Nzp4, c1, c2, c3, c4,E1,E2,E3,E4,E5,E6p,R1)
                                                    if n4 != 0:
                                                        E6m = np.zeros(n3-1+n4+1)
                                                        for t in range(n3+n4-1+1):
                                                            E6m[t] = E(n3,n4-1,t,Cz-Dz,c,d,Qz-Cz,Qz-Dz,Cz-Dz)
                                                        calcz += elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4-1, N1, N2, N3, Nzm4, c1, c2, c3, c4,E1,E2,E3,E4,E5,E6m,R1)
                                                                                    
                            ERIx[mu,nu,lam,sig] = ERIx[nu,mu,lam,sig] = ERIx[mu,nu,sig,lam] = ERIx[nu,mu,sig,lam] = ERIx[lam,sig,mu,nu] = ERIx[sig,lam,mu,nu] = ERIx[lam,sig,nu,mu] = ERIx[sig,lam,nu,mu] = calcx
                            ERIy[mu,nu,lam,sig] = ERIy[nu,mu,lam,sig] = ERIy[mu,nu,sig,lam] = ERIy[nu,mu,sig,lam] = ERIy[lam,sig,mu,nu] = ERIy[sig,lam,mu,nu] = ERIy[lam,sig,nu,mu] = ERIy[sig,lam,nu,mu] = calcy
                            ERIz[mu,nu,lam,sig] = ERIz[nu,mu,lam,sig] = ERIz[mu,nu,sig,lam] = ERIz[nu,mu,sig,lam] = ERIz[lam,sig,mu,nu] = ERIz[sig,lam,mu,nu] = ERIz[lam,sig,nu,mu] = ERIz[sig,lam,nu,mu] = calcz

    return Sxarr, Syarr, Szarr, Txarr, Tyarr, Tzarr, VNexarr, VNeyarr, VNezarr, ERIx, ERIy, ERIz
    

cpdef double [:,:] runQMESPcython(int [:,:] basisidx, double [:,:] basisfloat, int [:,:] basisint, double[:,:] Ve, double Zc, double Cx, double Cy, double Cz):
    cdef double calc, p, Px, Py, Pz, RPC
    cdef int k, l, i, j, t, u, v
    cdef double [:] Ex, Ey, Ez
    cdef double [:,:,:] R1
    
    for k in range(0, len(basisidx)):
        for l in range(0, len(basisidx)):
            if k >= l:
                calc  = 0.0
                for i in range(basisidx[k,1],basisidx[k,1]+basisidx[k,0]):
                    for j in range(basisidx[l,1],basisidx[l,1]+basisidx[l,0]):
                        a  = basisfloat[i,1]
                        b  = basisfloat[j,1]
                        Ax = basisfloat[i,3]
                        Ay = basisfloat[i,4]
                        Az = basisfloat[i,5]
                        Bx = basisfloat[j,3]
                        By = basisfloat[j,4]
                        Bz = basisfloat[j,5]
                        l1 = basisint[i,0]
                        l2 = basisint[j,0]
                        m1 = basisint[i,1]
                        m2 = basisint[j,1]
                        n1 = basisint[i,2]
                        n2 = basisint[j,2]
                        N1 = basisfloat[i,0]
                        N2 = basisfloat[j,0]
                        c1 = basisfloat[i,2]
                        c2 = basisfloat[j,2]
                        
                        p   = a+b
                        Px  = (a*Ax+b*Bx)/p
                        Py  = (a*Ay+b*By)/p
                        Pz  = (a*Az+b*Bz)/p
                        
                        Ex = np.zeros(l1+l2+1)
                        Ey = np.zeros(m1+m2+1)
                        Ez = np.zeros(n1+n2+1)
                        
                        for t in range(l1+l2+1):
                            Ex[t] = E(l1,l2,t,Ax-Bx,a,b,Px-Ax,Px-Bx,Ax-Bx)
                        for u in range(m1+m2+1):
                            Ey[u] = E(m1,m2,u,Ay-By,a,b,Py-Ay,Py-By,Ay-By)
                        for v in range(n1+n2+1):
                            Ez[v] = E(n1,n2,v,Az-Bz,a,b,Pz-Az,Pz-Bz,Az-Bz)

                        RPC = ((Px-Cx)**2+(Py-Cy)**2+(Pz-Cz)**2)**0.5
                        R1 = runR(l1+l2, m1+m2, n1+n2, Cx, Cy, Cz, Px, Py, Pz, p)

                        calc += elnuc(p, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, Zc, Ex, Ey, Ez, R1)

                Ve[k,l] = Ve[l,k] = calc
    return Ve
    
cpdef double boysPrun(double m,double T):
    # boys_Python_run, used in tests.py to test boys function
    return boys(m, T)