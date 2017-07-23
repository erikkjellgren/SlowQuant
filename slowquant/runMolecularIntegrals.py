import numpy as np
import time as time
import copy
from slowquant.molecularintegrals.MolecularIntegrals import E, Overlap, Kin, elnuc, nucdiff, u_ObaraSaika, electricfield, Ndiff1, Ndiff2, Nrun, Eprecalculation, nucrep
from slowquant.molecularintegrals.MIcython import elelrep, runR

##CALC OF INTEGRALS
def runIntegrals(input, basis, settings):
    # Nuclear-nuclear repulsion
    VNN = np.zeros(1)
    VNN[0] = nucrep(input)
    np.save('slowquant/temp/enuc.npy',VNN)
    #END OF nuclear-nuclear repulsion
    
    # Precalul1tions
    Edict, GPdict, pdict = Eprecalculation(basis)
    
    # One electron integrals
    start = time.time()
    Na = np.zeros((len(basis),len(basis)))
    S = np.zeros((len(basis),len(basis)))
    T = np.zeros((len(basis),len(basis)))
    for k in range(0, len(basis)):
        for l in range(0, len(basis)):
            if k >= l:
                idx = np.zeros(2)
                idx[0] = k
                idx[1] = l
                idx = idx.astype(int)
                calc = 0
                calc2 = 0
                calc3 = 0
                for i in range(basis[idx[0]][4]):
                    for j in range(basis[idx[1]][4]):
                        a = basis[idx[0]][5][i][1]
                        b = basis[idx[1]][5][j][1]
                        Ax = basis[idx[0]][1]
                        Ay = basis[idx[0]][2]
                        Az = basis[idx[0]][3]
                        Bx = basis[idx[1]][1]
                        By = basis[idx[1]][2]
                        Bz = basis[idx[1]][3]
                        l1 = basis[idx[0]][5][i][3]
                        l2 = basis[idx[1]][5][j][3]
                        m1 = basis[idx[0]][5][i][4]
                        m2 = basis[idx[1]][5][j][4]
                        n1 = basis[idx[0]][5][i][5]
                        n2 = basis[idx[1]][5][j][5]
                        N1 = basis[idx[0]][5][i][0]
                        N2 = basis[idx[1]][5][j][0]
                        c1 = basis[idx[0]][5][i][2]
                        c2 = basis[idx[1]][5][j][2]
                        Ex = Edict[str(k)+str(l)+str(i)+str(j)+'E1']
                        Ey = Edict[str(k)+str(l)+str(i)+str(j)+'E2']
                        Ez = Edict[str(k)+str(l)+str(i)+str(j)+'E3']
                        P  = GPdict[str(k)+str(l)+str(i)+str(j)]
                        p  = pdict[str(k)+str(l)+str(i)+str(j)]

                        for atom in range(1, len(input)):
                            Zc = input[atom][0]
                            C = np.array([input[atom][1],input[atom][2],input[atom][3]])
                            RPC = ((P[0]-C[0])**2+(P[1]-C[1])**2+(P[2]-C[2])**2)**0.5
                            R1 = runR(l1+l2, m1+m2, n1+n2, C, P, p)

                            calc += elnuc(P, p, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, Zc, Ex, Ey, Ez, R1)
                        calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2)
                        calc2 += calct
                        calc3 += calct2
                        
                Na[k,l] = calc
                Na[l,k] = calc
                S[k,l]  = calc3
                S[l,k]  = calc3
                T[k,l]  = calc2
                T[l,k]  = calc2
    np.save('slowquant/temp/nucatt.npy',Na)
    np.save('slowquant/temp/overlap.npy',S)
    np.save('slowquant/temp/Ekin.npy',T)
    print(time.time()-start, 'One electron integral')
    #END OF one electron integrals
    
    # Two electron integrals
    ScreenTHR = 10**-float(settings['Cauchy-Schwarz Threshold'])
    start = time.time()
    # First run the diagonal elements, this is used in the screening
    ERI = np.zeros((len(basis),len(basis),len(basis),len(basis)))
    Gab = np.zeros((len(basis),len(basis)))
    for mu in range(0, len(basis)):
        for nu in range(0, len(basis)):
            if mu >= nu:
                l1m = mu
                sig = nu
                idx = np.zeros(4)
                idx[0] = mu
                idx[1] = nu
                idx[2] = l1m
                idx[3] = sig
                idx = idx.astype(int)
                calc = 0
                for i in range(basis[idx[0]][4]):
                    for j in range(basis[idx[1]][4]):
                        E1 = Edict[str(mu)+str(nu)+str(i)+str(j)+'E1']
                        E2 = Edict[str(mu)+str(nu)+str(i)+str(j)+'E2']
                        E3 = Edict[str(mu)+str(nu)+str(i)+str(j)+'E3']
                        P  = GPdict[str(mu)+str(nu)+str(i)+str(j)]
                        p  = pdict[str(mu)+str(nu)+str(i)+str(j)]
                        l1=basis[idx[0]][5][i][3]
                        l2=basis[idx[1]][5][j][3]
                        m1=basis[idx[0]][5][i][4]
                        m2=basis[idx[1]][5][j][4]
                        n1=basis[idx[0]][5][i][5]
                        n2=basis[idx[1]][5][j][5]
                        N1=basis[idx[0]][5][i][0]
                        N2=basis[idx[1]][5][j][0]
                        c1=basis[idx[0]][5][i][2]
                        c2=basis[idx[1]][5][j][2]
                        for k in range(basis[idx[2]][4]):
                            for l in range(basis[idx[3]][4]):
                                E4 = Edict[str(l1m)+str(sig)+str(k)+str(l)+'E1']
                                E5 = Edict[str(l1m)+str(sig)+str(k)+str(l)+'E2']
                                E6 = Edict[str(l1m)+str(sig)+str(k)+str(l)+'E3']
                                Q  = GPdict[str(l1m)+str(sig)+str(k)+str(l)]
                                q  = pdict[str(l1m)+str(sig)+str(k)+str(l)]
                                l3=basis[idx[2]][5][k][3]
                                l4=basis[idx[3]][5][l][3] 
                                m3=basis[idx[2]][5][k][4]
                                m4=basis[idx[3]][5][l][4] 
                                n3=basis[idx[2]][5][k][5]
                                n4=basis[idx[3]][5][l][5] 
                                N3=basis[idx[2]][5][k][0]
                                N4=basis[idx[3]][5][l][0]
                                c3=basis[idx[2]][5][k][2]
                                c4=basis[idx[3]][5][l][2]
                                
                                alpha = p*q/(p+q)
                                RPQ = ((P[0]-Q[0])**2+(P[1]-Q[1])**2+(P[2]-Q[2])**2)**0.5
                                
                                R1 = runR(l1+l2+l3+l4, m1+m2+m3+m4, n1+n2+n3+n4, Q, P, alpha)
                                calc += elelrep(p,q,l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, N3, N4, c1, c2, c3, c4, E1,E2,E3,E4,E5,E6,R1)

                ERI[mu,nu,l1m,sig] = calc
                ERI[nu,mu,l1m,sig] = calc
                ERI[mu,nu,sig,l1m] = calc
                ERI[nu,mu,sig,l1m] = calc
                ERI[l1m,sig,mu,nu] = calc
                ERI[sig,l1m,mu,nu] = calc
                ERI[l1m,sig,nu,mu] = calc
                ERI[sig,l1m,nu,mu] = calc
                Gab[mu,nu] = (calc)**0.5
                
    # Run all the off diagonal elements
    for mu in range(0, len(basis)):
        for nu in range(0, len(basis)):
            if mu >= nu:
                for l1m in range(0, len(basis)):
                    for sig in range(0, len(basis)):
                        # Cauchy-Schwarz inequality
                        if Gab[mu,nu]*Gab[l1m,sig] > ScreenTHR:
                            munu = mu*(mu+1)/2+nu
                            l1msig = l1m*(l1m+1)/2+sig
                            if l1m >= sig and munu > l1msig:
                                idx = np.zeros(4)
                                idx[0] = mu
                                idx[1] = nu
                                idx[2] = l1m
                                idx[3] = sig
                                idx = idx.astype(int)
                                calc = 0
                                for i in range(basis[idx[0]][4]):
                                    for j in range(basis[idx[1]][4]):
                                        E1 = Edict[str(mu)+str(nu)+str(i)+str(j)+'E1']
                                        E2 = Edict[str(mu)+str(nu)+str(i)+str(j)+'E2']
                                        E3 = Edict[str(mu)+str(nu)+str(i)+str(j)+'E3']
                                        P  = GPdict[str(mu)+str(nu)+str(i)+str(j)]
                                        p  = pdict[str(mu)+str(nu)+str(i)+str(j)]
                                        l1=basis[idx[0]][5][i][3]
                                        l2=basis[idx[1]][5][j][3]
                                        m1=basis[idx[0]][5][i][4]
                                        m2=basis[idx[1]][5][j][4]
                                        n1=basis[idx[0]][5][i][5]
                                        n2=basis[idx[1]][5][j][5]
                                        N1=basis[idx[0]][5][i][0]
                                        N2=basis[idx[1]][5][j][0]
                                        c1=basis[idx[0]][5][i][2]
                                        c2=basis[idx[1]][5][j][2]
                                        for k in range(basis[idx[2]][4]):
                                            for l in range(basis[idx[3]][4]):
                                                E4 = Edict[str(l1m)+str(sig)+str(k)+str(l)+'E1']
                                                E5 = Edict[str(l1m)+str(sig)+str(k)+str(l)+'E2']
                                                E6 = Edict[str(l1m)+str(sig)+str(k)+str(l)+'E3']
                                                Q  = GPdict[str(l1m)+str(sig)+str(k)+str(l)]
                                                q  = pdict[str(l1m)+str(sig)+str(k)+str(l)]
                                                l3=basis[idx[2]][5][k][3]
                                                l4=basis[idx[3]][5][l][3] 
                                                m3=basis[idx[2]][5][k][4]
                                                m4=basis[idx[3]][5][l][4] 
                                                n3=basis[idx[2]][5][k][5]
                                                n4=basis[idx[3]][5][l][5] 
                                                N3=basis[idx[2]][5][k][0]
                                                N4=basis[idx[3]][5][l][0]
                                                c3=basis[idx[2]][5][k][2]
                                                c4=basis[idx[3]][5][l][2]
                                                
                                                alpha = p*q/(p+q)
                                                RPQ = ((P[0]-Q[0])**2+(P[1]-Q[1])**2+(P[2]-Q[2])**2)**0.5
                                                
                                                R1 = runR(l1+l2+l3+l4, m1+m2+m3+m4, n1+n2+n3+n4, Q, P, alpha)
                                                calc += elelrep(p,q,l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, N3, N4, c1, c2, c3, c4, E1,E2,E3,E4,E5,E6,R1)
                                
                                ERI[mu,nu,l1m,sig] = calc
                                ERI[nu,mu,l1m,sig] = calc
                                ERI[mu,nu,sig,l1m] = calc
                                ERI[nu,mu,sig,l1m] = calc
                                ERI[l1m,sig,mu,nu] = calc
                                ERI[sig,l1m,mu,nu] = calc
                                ERI[l1m,sig,nu,mu] = calc
                                ERI[sig,l1m,nu,mu] = calc

    np.save('slowquant/temp/twoint.npy',ERI)
    print(time.time()-start, 'ERI')
    #END OF two electron integrals

    
def run_dipole_int(basis, input):
    X = np.zeros((len(basis),len(basis)))
    Y = np.zeros((len(basis),len(basis)))
    Z = np.zeros((len(basis),len(basis)))
    for k in range(0, len(basis)):
        for l in range(0, len(basis)):
            if k >= l:
                idx = np.zeros(2)
                idx[0] = k
                idx[1] = l
                idx = idx.astype(int)
                calcx = 0
                calcy = 0
                calcz = 0
                for i in range(basis[idx[0]][4]):
                    for j in range(basis[idx[1]][4]):
                        x, y, z = u_ObaraSaika(basis[idx[0]][5][i][1], basis[idx[1]][5][j][1], basis[idx[0]][1], basis[idx[0]][2], basis[idx[0]][3], basis[idx[1]][1], basis[idx[1]][2], basis[idx[1]][3], basis[idx[0]][5][i][3], basis[idx[1]][5][j][3], basis[idx[0]][5][i][4], basis[idx[1]][5][j][4],basis[idx[0]][5][i][5], basis[idx[1]][5][j][5], basis[idx[0]][5][i][0], basis[idx[1]][5][j][0], basis[idx[0]][5][i][2], basis[idx[1]][5][j][2], input)
                        calcx += x
                        calcy += y
                        calcz += z
                X[k,l] = calcx
                X[l,k] = calcx
                Y[k,l] = calcy
                Y[l,k] = calcy
                Z[k,l] = calcz
                Z[l,k] = calcz
    np.save('slowquant/temp/mux.npy',X)
    np.save('slowquant/temp/muy.npy',Y)
    np.save('slowquant/temp/muz.npy',Z)

def runQMESP(basis, input, rcx, rcy ,rcz):
    # Set up indexes for integrals
    Ve = np.zeros((len(basis),len(basis)))
    Edict, GPdict, pdict = Eprecalculation(basis)
    for k in range(0, len(basis)):
        for l in range(0, len(basis)):
            if k >= l:
                idx = np.zeros(2)
                idx[0] = k
                idx[1] = l
                idx = idx.astype(int)
                calc = 0
                for i in range(basis[idx[0]][4]):
                    for j in range(basis[idx[1]][4]):
                        l1 = basis[idx[0]][5][i][3]
                        l2 = basis[idx[1]][5][j][3]
                        m1 = basis[idx[0]][5][i][4]
                        m2 = basis[idx[1]][5][j][4]
                        n1 = basis[idx[0]][5][i][5]
                        n2 = basis[idx[1]][5][j][5]
                        N1 = basis[idx[0]][5][i][0]
                        N2 = basis[idx[1]][5][j][0]
                        c1 = basis[idx[0]][5][i][2]
                        c2 = basis[idx[1]][5][j][2]
                        Ex = Edict[str(k)+str(l)+str(i)+str(j)+'E1']
                        Ey = Edict[str(k)+str(l)+str(i)+str(j)+'E2']
                        Ez = Edict[str(k)+str(l)+str(i)+str(j)+'E3']
                        P  = GPdict[str(k)+str(l)+str(i)+str(j)]
                        p  = pdict[str(k)+str(l)+str(i)+str(j)]

                        Zc = -1.0
                        C = np.array([rcx, rcy ,rcz])
                        RPC = ((P[0]-C[0])**2+(P[1]-C[1])**2+(P[2]-C[2])**2)**0.5
                        R1 = runR(l1+l2, m1+m2, n1+n2, C, P, p)

                        calc += elnuc(P, p, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, Zc, Ex, Ey, Ez,R1)
            Ve[k,l] = calc
            Ve[l,k] = calc
        
    return Ve

def rungeometric_derivatives(input, basis):
    # Calcul1ting the norm1lization of the derivatives. For now only used in ERI
    Nxplus  = copy.deepcopy(basis)
    Nxminus = copy.deepcopy(basis)
    Nyplus  = copy.deepcopy(basis)
    Nyminus = copy.deepcopy(basis)
    Nzplus  = copy.deepcopy(basis)
    Nzminus = copy.deepcopy(basis)
    for i in range(len(basis)):
        for j in range(len(basis[i][5])):
            Nxplus[i][5][j][3] += 1
            if Nxminus[i][5][j][3] != 0:
                Nxminus[i][5][j][3] -= 1
            
            Nyplus[i][5][j][4] += 1
            if Nyminus[i][5][j][4] != 0:
                Nyminus[i][5][j][4] -= 1
            
            Nzplus[i][5][j][5] += 1    
            if Nzminus[i][5][j][5] != 0:
                Nzminus[i][5][j][5] -= 1
                
    Nxplus = Nrun(Nxplus)
    Nxminus = Nrun(Nxminus)
    Nyplus = Nrun(Nyplus)
    Nyminus = Nrun(Nyminus)
    Nzplus = Nrun(Nzplus)
    Nzminus = Nrun(Nzminus)
    
    # Precalul1tions
    Edict, GPdict, pdict = Eprecalculation(basis)
    
    for atomidx in range(1, len(input)):
        # Nuclear-nuclear repulsion
        VNN = np.zeros(1)
        VNN[0] = nucdiff(input, atomidx, 1)
        np.save('slowquant/temp/'+str(atomidx)+'dxenuc.npy',VNN)
        
        VNN = np.zeros(1)
        VNN[0] = nucdiff(input, atomidx, 2)
        np.save('slowquant/temp/'+str(atomidx)+'dyenuc.npy',VNN)
        
        VNN = np.zeros(1)
        VNN[0] = nucdiff(input, atomidx, 3)
        np.save('slowquant/temp/'+str(atomidx)+'dzenuc.npy',VNN)
        #END OF nuclear-nuclear repulsion
        
        # Two electron integrals x diff
        start = time.time()
        ERIx = np.zeros((len(basis),len(basis),len(basis),len(basis)))
        ERIy = np.zeros((len(basis),len(basis),len(basis),len(basis)))
        ERIz = np.zeros((len(basis),len(basis),len(basis),len(basis)))
        for mu in range(0, len(basis)):
            for nu in range(0, len(basis)):
                if mu >= nu:
                    for l1m in range(0, len(basis)):
                        for sig in range(0, len(basis)):
                            munu = mu*(mu+1)/2+nu
                            l1msig = l1m*(l1m+1)/2+sig
                            if l1m >= sig and munu >= l1msig:
                                idx = np.zeros(4)
                                idx[0] = mu
                                idx[1] = nu
                                idx[2] = l1m
                                idx[3] = sig
                                idx = idx.astype(int)
                                calcx = 0
                                calcy = 0
                                calcz = 0
                                if atomidx == basis[idx[0]][6] and atomidx == basis[idx[1]][6] and atomidx == basis[idx[2]][6] and atomidx == basis[idx[3]][6]:
                                    calcx = 0
                                    calcy = 0
                                    calcz = 0
                                else:
                                    for i in range(basis[idx[0]][4]):
                                        for j in range(basis[idx[1]][4]):
                                            E1 = Edict[str(mu)+str(nu)+str(i)+str(j)+'E1']
                                            E2 = Edict[str(mu)+str(nu)+str(i)+str(j)+'E2']
                                            E3 = Edict[str(mu)+str(nu)+str(i)+str(j)+'E3']
                                            P  = GPdict[str(mu)+str(nu)+str(i)+str(j)]
                                            p  = pdict[str(mu)+str(nu)+str(i)+str(j)]
                                            a=basis[idx[0]][5][i][1]
                                            b=basis[idx[1]][5][j][1]
                                            Ax=basis[idx[0]][1]
                                            Bx=basis[idx[1]][1]
                                            Ay=basis[idx[0]][2]
                                            By=basis[idx[1]][2]
                                            Az=basis[idx[0]][3] 
                                            Bz=basis[idx[1]][3]
                                            l1=basis[idx[0]][5][i][3]
                                            l2=basis[idx[1]][5][j][3]
                                            m1=basis[idx[0]][5][i][4]
                                            m2=basis[idx[1]][5][j][4]
                                            n1=basis[idx[0]][5][i][5]
                                            n2=basis[idx[1]][5][j][5]
                                            N1=basis[idx[0]][5][i][0]
                                            N2=basis[idx[1]][5][j][0]
                                            c1=basis[idx[0]][5][i][2]
                                            c2=basis[idx[1]][5][j][2]
                                            Nxp1=Nxplus[idx[0]][5][i][0]
                                            Nxp2=Nxplus[idx[1]][5][j][0]
                                            Nxm1=Nxminus[idx[0]][5][i][0]
                                            Nxm2=Nxminus[idx[1]][5][j][0]
                                            Nyp1=Nyplus[idx[0]][5][i][0]
                                            Nyp2=Nyplus[idx[1]][5][j][0]
                                            Nym1=Nyminus[idx[0]][5][i][0]
                                            Nym2=Nyminus[idx[1]][5][j][0]
                                            Nzp1=Nzplus[idx[0]][5][i][0]
                                            Nzp2=Nzplus[idx[1]][5][j][0]
                                            Nzm1=Nzminus[idx[0]][5][i][0]
                                            Nzm2=Nzminus[idx[1]][5][j][0]
                                            for k in range(basis[idx[2]][4]):
                                                for l in range(basis[idx[3]][4]):
                                                    E4 = Edict[str(l1m)+str(sig)+str(k)+str(l)+'E1']
                                                    E5 = Edict[str(l1m)+str(sig)+str(k)+str(l)+'E2']
                                                    E6 = Edict[str(l1m)+str(sig)+str(k)+str(l)+'E3']
                                                    Q  = GPdict[str(l1m)+str(sig)+str(k)+str(l)]
                                                    q  = pdict[str(l1m)+str(sig)+str(k)+str(l)]
                                                    c=basis[idx[2]][5][k][1]
                                                    d=basis[idx[3]][5][l][1]
                                                    Cx=basis[idx[2]][1]
                                                    Dx=basis[idx[3]][1]
                                                    Cy=basis[idx[2]][2]
                                                    Dy=basis[idx[3]][2]
                                                    Cz=basis[idx[2]][3]
                                                    Dz=basis[idx[3]][3]
                                                    l3=basis[idx[2]][5][k][3]
                                                    l4=basis[idx[3]][5][l][3] 
                                                    m3=basis[idx[2]][5][k][4]
                                                    m4=basis[idx[3]][5][l][4] 
                                                    n3=basis[idx[2]][5][k][5]
                                                    n4=basis[idx[3]][5][l][5] 
                                                    N3=basis[idx[2]][5][k][0]
                                                    N4=basis[idx[3]][5][l][0]
                                                    c3=basis[idx[2]][5][k][2]
                                                    c4=basis[idx[3]][5][l][2]    
                                                    Nxp3=Nxplus[idx[2]][5][k][0]
                                                    Nxp4=Nxplus[idx[3]][5][l][0]
                                                    Nxm3=Nxminus[idx[2]][5][k][0]
                                                    Nxm4=Nxminus[idx[3]][5][l][0]  
                                                    Nyp3=Nyplus[idx[2]][5][k][0]
                                                    Nyp4=Nyplus[idx[3]][5][l][0]
                                                    Nym3=Nyminus[idx[2]][5][k][0]
                                                    Nym4=Nyminus[idx[3]][5][l][0]
                                                    Nzp3=Nzplus[idx[2]][5][k][0]
                                                    Nzp4=Nzplus[idx[3]][5][l][0]
                                                    Nzm3=Nzminus[idx[2]][5][k][0]
                                                    Nzm4=Nzminus[idx[3]][5][l][0]
                                                    
                                                    alpha = p*q/(p+q)
                                                    RPQ = ((P[0]-Q[0])**2+(P[1]-Q[1])**2+(P[2]-Q[2])**2)**0.5
                                                    
                                                    R1 = runR(l1+l2+l3+l4, m1+m2+m3+m4, n1+n2+n3+n4, Q, P, alpha, check=1)
                                                    
                                                    # Calcul1te x derivative
                                                    if atomidx == basis[idx[0]][6]:
                                                        E1p = np.zeros(l1+1+l2+1)
                                                        for t in range(l1+1+l2+1):
                                                            E1p[t] = E(l1+1,l2,t,Ax-Bx,a,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                                        calcx += Ndiff1(l1, a)*elelrep(p, q, l1+1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, Nxp1, N2, N3, N4, c1, c2, c3, c4, E1p,E2,E3,E4,E5,E6,R1)
                                                        if l1 != 0:
                                                            E1m = np.zeros(l1-1+l2+1)
                                                            for t in range(l1-1+l2+1):
                                                                E1m[t] = E(l1-1,l2,t,Ax-Bx,a,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                                            calcx += Ndiff2(l1, a)*elelrep(p, q,l1-1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, Nxm1, N2, N3, N4, c1, c2, c3, c4, E1m,E2,E3,E4,E5,E6,R1)
                                                            
                                                    if atomidx == basis[idx[1]][6]:
                                                        E1p = np.zeros(l1+1+l2+1)
                                                        for t in range(l1+l2+1+1):
                                                            E1p[t] = E(l1,l2+1,t,Ax-Bx,a,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                                        calcx += Ndiff1(l2, b)*elelrep(p, q, l1, l2+1, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, Nxp2, N3, N4, c1, c2, c3, c4, E1p,E2,E3,E4,E5,E6,R1)
                                                        if l2 != 0:
                                                            E1m = np.zeros(l1-1+l2+1)
                                                            for t in range(l1+l2-1+1):
                                                                E1m[t] = E(l1,l2-1,t,Ax-Bx,a,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                                            calcx += Ndiff2(l2, b)*elelrep(p, q, l1, l2-1, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, Nxm2, N3, N4, c1, c2, c3, c4, E1m,E2,E3,E4,E5,E6,R1)
                                                            
                                                    if atomidx == basis[idx[2]][6]:  
                                                        E4p = np.zeros(l3+1+l4+1)
                                                        for t in range(l3+1+l4+1):
                                                            E4p[t] = E(l3+1,l4,t,Cx-Dx,c,d,Q[0]-Cx,Q[0]-Dx,Cx-Dx)
                                                        calcx += Ndiff1(l3, c)*elelrep(p, q, l1, l2, l3+1, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, Nxp3, N4, c1, c2, c3, c4, E1,E2,E3,E4p,E5,E6,R1)
                                                        if l3 != 0:
                                                            E4m = np.zeros(l3-1+l4+1)
                                                            for t in range(l3-1+l4+1):
                                                                E4m[t] = E(l3-1,l4,t,Cx-Dx,c,d,Q[0]-Cx,Q[0]-Dx,Cx-Dx)
                                                            calcx += Ndiff2(l3, c)*elelrep(p, q, l1, l2, l3-1, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, Nxm3, N4, c1, c2, c3, c4, E1,E2,E3,E4m,E5,E6,R1)
                                                        
                                                    if atomidx == basis[idx[3]][6]:
                                                        E4p = np.zeros(l3+1+l4+1)
                                                        for t in range(l3+l4+1+1):
                                                            E4p[t] = E(l3,l4+1,t,Cx-Dx,c,d,Q[0]-Cx,Q[0]-Dx,Cx-Dx)
                                                        calcx += Ndiff1(l4, d)*elelrep(p, q, l1, l2, l3, l4+1, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, N3, Nxp4, c1, c2, c3, c4, E1,E2,E3,E4p,E5,E6,R1)
                                                        if l4 != 0:
                                                            E4m = np.zeros(l3-1+l4+1)
                                                            for t in range(l3+l4-1+1):
                                                                E4m[t] = E(l3,l4-1,t,Cx-Dx,c,d,Q[0]-Cx,Q[0]-Dx,Cx-Dx)
                                                            calcx += Ndiff2(l4, d)*elelrep(p, q, l1, l2, l3, l4-1, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, N3, Nxm4, c1, c2, c3, c4, E1,E2,E3,E4m,E5,E6,R1)
                                                    
                                                    # Calcul1te y derivative
                                                    if atomidx == basis[idx[0]][6]:
                                                        E2p = np.zeros(m1+1+m2+1)
                                                        for t in range(m1+1+m2+1):
                                                            E2p[t] = E(m1+1,m2,t,Ay-By,a,b,P[1]-Ay,P[1]-By,Ay-By)
                                                        calcy += Ndiff1(m1, a)*elelrep(p, q, l1, l2, l3, l4, m1+1, m2, m3, m4, n1, n2, n3, n4, Nyp1, N2, N3, N4, c1, c2, c3, c4, E1,E2p,E3,E4,E5,E6,R1)
                                                        if m1 != 0:
                                                            E2m = np.zeros(m1-1+m2+1)
                                                            for t in range(m1-1+m2+1):
                                                                E2m[t] = E(m1-1,m2,t,Ay-By,a,b,P[1]-Ay,P[1]-By,Ay-By)
                                                            calcy += Ndiff2(m1, a)*elelrep(p, q, l1, l2, l3, l4, m1-1, m2, m3, m4, n1, n2, n3, n4, Nym1, N2, N3, N4, c1, c2, c3, c4, E1,E2m,E3,E4,E5,E6,R1)
                                                            
                                                    if atomidx == basis[idx[1]][6]:
                                                        E2p = np.zeros(m1+1+m2+1)
                                                        for t in range(m1+m2+1+1):
                                                            E2p[t] = E(m1,m2+1,t,Ay-By,a,b,P[1]-Ay,P[1]-By,Ay-By)
                                                        calcy += Ndiff1(m2, b)*elelrep(p, q, l1, l2, l3, l4, m1, m2+1, m3, m4, n1, n2, n3, n4, N1, Nyp2, N3, N4, c1, c2, c3, c4, E1,E2p,E3,E4,E5,E6,R1)
                                                        if m2 != 0:
                                                            E2m = np.zeros(m1-1+m2+1)
                                                            for t in range(m1+m2-1+1):
                                                                E2m[t] = E(m1,m2-1,t,Ay-By,a,b,P[1]-Ay,P[1]-By,Ay-By)
                                                            calcy += Ndiff2(m2, b)*elelrep(p, q, l1, l2, l3, l4, m1, m2-1, m3, m4, n1, n2, n3, n4, N1, Nym2, N3, N4, c1, c2, c3, c4,E1,E2m,E3,E4,E5,E6,R1)
                                                            
                                                    if atomidx == basis[idx[2]][6]:  
                                                        E5p = np.zeros(m3+1+m4+1)
                                                        for t in range(m3+1+m4+1):
                                                            E5p[t] = E(m3+1,m4,t,Cy-Dy,c,d,Q[1]-Cy,Q[1]-Dy,Cy-Dy)
                                                        calcy += Ndiff1(m3, c)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3+1, m4, n1, n2, n3, n4, N1, N2, Nyp3, N4, c1, c2, c3, c4, E1,E2,E3,E4,E5p,E6,R1)
                                                        if m3 != 0:
                                                            E5m = np.zeros(m3-1+m4+1)
                                                            for t in range(m3-1+m4+1):
                                                                E5m[t] = E(m3-1,m4,t,Cy-Dy,c,d,Q[1]-Cy,Q[1]-Dy,Cy-Dy)
                                                            calcy += Ndiff2(m3, c)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3-1, m4, n1, n2, n3, n4, N1, N2, Nym3, N4, c1, c2, c3, c4,E1,E2,E3,E4,E5m,E6,R1)
                                                        
                                                    if atomidx == basis[idx[3]][6]:
                                                        E5p = np.zeros(m3+1+m4+1)
                                                        for t in range(m3+m4+1+1):
                                                            E5p[t] = E(m3,m4+1,t,Cy-Dy,c,d,Q[1]-Cy,Q[1]-Dy,Cy-Dy)
                                                        calcy += Ndiff1(m4, d)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4+1, n1, n2, n3, n4, N1, N2, N3, Nyp4, c1, c2, c3, c4,E1,E2,E3,E4,E5p,E6,R1)
                                                        if m4 != 0:
                                                            E5m = np.zeros(m3-1+m4+1)
                                                            for t in range(m3+m4-1+1):
                                                                E5m[t] = E(m3,m4-1,t,Cy-Dy,c,d,Q[1]-Cy,Q[1]-Dy,Cy-Dy)
                                                            calcy += Ndiff2(m4, d)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4-1, n1, n2, n3, n4, N1, N2, N3, Nym4, c1, c2, c3, c4,E1,E2,E3,E4,E5m,E6,R1)
                                                    
                                                    # Calcul1te z derivative
                                                    if atomidx == basis[idx[0]][6]:
                                                        E3p = np.zeros(n1+1+n2+1)
                                                        for t in range(n1+1+n2+1):
                                                            E3p[t] = E(n1+1,n2,t,Az-Bz,a,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                                        calcz += Ndiff1(n1, a)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1+1, n2, n3, n4, Nzp1, N2, N3, N4, c1, c2, c3, c4, E1,E2,E3p,E4,E5,E6,R1)
                                                        if n1 != 0:
                                                            E3m = np.zeros(n1-1+n2+1)
                                                            for t in range(n1-1+n2+1):
                                                                E3m[t] = E(n1-1,n2,t,Az-Bz,a,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                                            calcz += Ndiff2(n1, a)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1-1, n2, n3, n4, Nzm1, N2, N3, N4, c1, c2, c3, c4, E1,E2,E3m,E4,E5,E6,R1)
                                                            
                                                    if atomidx == basis[idx[1]][6]:
                                                        E3p = np.zeros(n1+1+n2+1)
                                                        for t in range(n1+n2+1+1):
                                                            E3p[t] = E(n1,n2+1,t,Az-Bz,a,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                                        calcz += Ndiff1(n2, b)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2+1, n3, n4, N1, Nzp2, N3, N4, c1, c2, c3, c4, E1,E2,E3p,E4,E5,E6,R1)
                                                        if n2 != 0:
                                                            E3m = np.zeros(n1-1+n2+1)
                                                            for t in range(n1+n2-1+1):
                                                                E3m[t] = E(n1,n2-1,t,Az-Bz,a,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                                            calcz += Ndiff2(n2, b)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2-1, n3, n4, N1, Nzm2, N3, N4, c1, c2, c3, c4,E1,E2,E3m,E4,E5,E6,R1)
                                                            
                                                    if atomidx == basis[idx[2]][6]: 
                                                        E6p = np.zeros(n3+1+n4+1)
                                                        for t in range(n3+1+n4+1):
                                                            E6p[t] = E(n3+1,n4,t,Cz-Dz,c,d,Q[2]-Cz,Q[2]-Dz,Cz-Dz)
                                                        calcz += Ndiff1(n3, c)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3+1, n4, N1, N2, Nzp3, N4, c1, c2, c3, c4, E1,E2,E3,E4,E5,E6p,R1)
                                                        if n3 != 0:
                                                            E6m = np.zeros(n3-1+n4+1)
                                                            for t in range(n3-1+n4+1):
                                                                E6m[t] = E(n3-1,n4,t,Cz-Dz,c,d,Q[2]-Cz,Q[2]-Dz,Cz-Dz)
                                                            calcz += Ndiff2(n3, c)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3-1, n4, N1, N2, Nzm3, N4, c1, c2, c3, c4,E1,E2,E3,E4,E5,E6m,R1)
                                                        
                                                    if atomidx == basis[idx[3]][6]:
                                                        E6p = np.zeros(n3+1+n4+1)
                                                        for t in range(n3+n4+1+1):
                                                            E6p[t] = E(n3,n4+1,t,Cz-Dz,c,d,Q[2]-Cz,Q[2]-Dz,Cz-Dz)
                                                        calcz += Ndiff1(n4, d)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4+1, N1, N2, N3, Nzp4, c1, c2, c3, c4,E1,E2,E3,E4,E5,E6p,R1)
                                                        if n4 != 0:
                                                            E6m = np.zeros(n3-1+n4+1)
                                                            for t in range(n3+n4-1+1):
                                                                E6m[t] = E(n3,n4-1,t,Cz-Dz,c,d,Q[2]-Cz,Q[2]-Dz,Cz-Dz)
                                                            calcz += Ndiff2(n4, d)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4-1, N1, N2, N3, Nzm4, c1, c2, c3, c4,E1,E2,E3,E4,E5,E6m,R1)
                                                                                        
                                ERIx[mu,nu,l1m,sig] = calcx
                                ERIx[nu,mu,l1m,sig] = calcx
                                ERIx[mu,nu,sig,l1m] = calcx
                                ERIx[nu,mu,sig,l1m] = calcx
                                ERIx[l1m,sig,mu,nu] = calcx
                                ERIx[sig,l1m,mu,nu] = calcx
                                ERIx[l1m,sig,nu,mu] = calcx
                                ERIx[sig,l1m,nu,mu] = calcx
                                ERIy[mu,nu,l1m,sig] = calcy
                                ERIy[nu,mu,l1m,sig] = calcy
                                ERIy[mu,nu,sig,l1m] = calcy
                                ERIy[nu,mu,sig,l1m] = calcy
                                ERIy[l1m,sig,mu,nu] = calcy
                                ERIy[sig,l1m,mu,nu] = calcy
                                ERIy[l1m,sig,nu,mu] = calcy
                                ERIy[sig,l1m,nu,mu] = calcy
                                ERIz[mu,nu,l1m,sig] = calcz
                                ERIz[nu,mu,l1m,sig] = calcz
                                ERIz[mu,nu,sig,l1m] = calcz
                                ERIz[nu,mu,sig,l1m] = calcz
                                ERIz[l1m,sig,mu,nu] = calcz
                                ERIz[sig,l1m,mu,nu] = calcz
                                ERIz[l1m,sig,nu,mu] = calcz
                                ERIz[sig,l1m,nu,mu] = calcz
        np.save('slowquant/temp/'+str(atomidx)+'dxtwoint.npy',ERIx)
        np.save('slowquant/temp/'+str(atomidx)+'dytwoint.npy',ERIy)
        np.save('slowquant/temp/'+str(atomidx)+'dztwoint.npy',ERIz)
        print(time.time()-start, 'ERI diff: atom'+str(atomidx))
        #END OF two electron integrals x diff
        
        # One electron integrals
        start = time.time()
        Sxarr= np.zeros((len(basis),len(basis)))
        Syarr= np.zeros((len(basis),len(basis)))
        Szarr= np.zeros((len(basis),len(basis)))
        Txarr = np.zeros((len(basis),len(basis)))
        Tyarr = np.zeros((len(basis),len(basis)))
        Tzarr = np.zeros((len(basis),len(basis)))
        VNexarr = np.zeros((len(basis),len(basis)))
        VNeyarr = np.zeros((len(basis),len(basis)))
        VNezarr = np.zeros((len(basis),len(basis)))
        for k in range(0, len(basis)):
            for l in range(0, len(basis)):
                if k >= l:
                    idx = np.zeros(2)
                    idx[0] = k
                    idx[1] = l
                    idx = idx.astype(int)
                    Tx = 0
                    Ty = 0
                    Tz = 0
                    Sx = 0
                    Sy = 0
                    Sz = 0
                    VNex = 0
                    VNey = 0
                    VNez = 0
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
                            l1=basis[idx[0]][5][i][3]
                            l2=basis[idx[1]][5][j][3]
                            m1=basis[idx[0]][5][i][4]
                            m2=basis[idx[1]][5][j][4]
                            n1=basis[idx[0]][5][i][5]
                            n2=basis[idx[1]][5][j][5]
                            N1=basis[idx[0]][5][i][0]
                            N2=basis[idx[1]][5][j][0]
                            c1=basis[idx[0]][5][i][2]
                            c2=basis[idx[1]][5][j][2]
                            
                            Nxp1=Nxplus[idx[0]][5][i][0]
                            Nxp2=Nxplus[idx[1]][5][j][0]
                            Nxm1=Nxminus[idx[0]][5][i][0]
                            Nxm2=Nxminus[idx[1]][5][j][0]
                            
                            Nyp1=Nyplus[idx[0]][5][i][0]
                            Nyp2=Nyplus[idx[1]][5][j][0]
                            Nym1=Nyminus[idx[0]][5][i][0]
                            Nym2=Nyminus[idx[1]][5][j][0]
                            
                            Nzp1=Nzplus[idx[0]][5][i][0]
                            Nzp2=Nzplus[idx[1]][5][j][0]
                            Nzm1=Nzminus[idx[0]][5][i][0]
                            Nzm2=Nzminus[idx[1]][5][j][0]
                            
                            Ex = Edict[str(k)+str(l)+str(i)+str(j)+'E1']
                            Ey = Edict[str(k)+str(l)+str(i)+str(j)+'E2']
                            Ez = Edict[str(k)+str(l)+str(i)+str(j)+'E3']
                            P  = GPdict[str(k)+str(l)+str(i)+str(j)]
                            p  = pdict[str(k)+str(l)+str(i)+str(j)]
                                
                            if atomidx == basis[idx[0]][6] and atomidx == basis[idx[1]][6]:
                                Tx += 0
                                Ty += 0
                                Tz += 0
                                Sx += 0
                                Sy += 0
                                Sz += 0
                            else:
                                # OVERLAP AND KINETIC ENERGY
                                # x derivative
                                if atomidx == basis[idx[0]][6]:
                                    calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1+1, l2, m1, m2, n1, n2, Nxp1, N2, c1, c2)
                                    Tx += calct*Ndiff1(l1,a)
                                    Sx += calct2*Ndiff1(l1,a)
                                    if l1 != 0:
                                        calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1-1, l2, m1, m2, n1, n2, Nxm2, N2, c1, c2)
                                        Tx += calct*Ndiff2(l1,a)
                                        Sx += calct2*Ndiff2(l1,a)
                                        
                                if atomidx == basis[idx[1]][6]:
                                    calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2+1, m1, m2, n1, n2, N1, Nxp2, c1, c2)
                                    Tx += calct*Ndiff1(l2,b)
                                    Sx += calct2*Ndiff1(l2,b)
                                    if l2 != 0:
                                        calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2-1, m1, m2, n1, n2, N1, Nxm2, c1, c2)
                                        Tx += calct*Ndiff2(l2,b)
                                        Sx += calct2*Ndiff2(l2,b)
                                
                                # y derivative
                                if atomidx == basis[idx[0]][6]:
                                    calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1+1, m2, n1, n2, Nyp1, N2, c1, c2)
                                    Ty += calct*Ndiff1(m1,a)
                                    Sy += calct2*Ndiff1(m1,a)
                                    if m1 != 0:
                                        calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1-1, m2, n1, n2, Nym1, N2, c1, c2)
                                        Ty += calct*Ndiff2(m1,a)
                                        Sy += calct2*Ndiff2(m1,a)
                                        
                                if atomidx == basis[idx[1]][6]:
                                    calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2+1, n1, n2, N1, Nyp2, c1, c2)
                                    Ty += calct*Ndiff1(m2,b)
                                    Sy += calct2*Ndiff1(m2,b)
                                    if m2 != 0:
                                        calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2-1, n1, n2, N1, Nym2, c1, c2)
                                        Ty += calct*Ndiff2(m2,b)
                                        Sy += calct2*Ndiff2(m2,b)
                                
                                # z derivative
                                if atomidx == basis[idx[0]][6]:
                                    calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1+1, n2, Nzp1, N2, c1, c2)
                                    Tz += calct*Ndiff1(n1,a)
                                    Sz += calct2*Ndiff1(n1,a)
                                    if n1 != 0:
                                        calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1-1, n2, Nzm1, N2, c1, c2)
                                        Tz += calct*Ndiff2(n1,a)
                                        Sz += calct2*Ndiff2(n1,a)
                                        
                                if atomidx == basis[idx[1]][6]:
                                    calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2+1, N1, Nzp2, c1, c2)
                                    Tz += calct*Ndiff1(n2,b)
                                    Sz += calct2*Ndiff1(n2,b)
                                    if n2 != 0:
                                        calct, calct2 = Kin(a, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2-1, N1, Nzm2, c1, c2)
                                        Tz += calct*Ndiff2(n2,b)
                                        Sz += calct2*Ndiff2(n2,b)
                            
                            # NUCLEUS NUCLEUS ENERGY
                            for atom in range(1, len(input)):
                                if atomidx == basis[idx[0]][6] or atomidx == basis[idx[1]][6]:
                                    Zc = input[atom][0]
                                    C = np.array([input[atom][1],input[atom][2],input[atom][3]])
                                    RPC = ((P[0]-C[0])**2+(P[1]-C[1])**2+(P[2]-C[2])**2)**0.5
                                    R1 = runR(l1+l2, m1+m2, n1+n2, C, P, p, check=1)
                            # x derivative
                                    if atomidx == basis[idx[0]][6] and atomidx == basis[idx[1]][6]:
                                        if atom != atomidx:
                                            Exp = np.zeros(l1+1+l2+1)
                                            for t in range(0, l1+1+l2+1):
                                                Exp[t] = E(l1+1,l2,t,Ax-Bx,a,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                            VNex += Ndiff1(l1,a)*elnuc(P, p, l1+1, l2, m1, m2, n1, n2, Nxp1, N2, c1, c2, Zc, Exp, Ey, Ez,R1)
                                            if l1 != 0:
                                                Exm = np.zeros(l1-1+l2+1)
                                                for t in range(0, l1-1+l2+1):
                                                    Exm[t] = E(l1-1,l2,t,Ax-Bx,a,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                                VNex += Ndiff2(l1,a)*elnuc(P, p, l1-1, l2, m1, m2, n1, n2, Nxm1, N2, c1, c2, Zc, Exm, Ey, Ez,R1)
                                            
                                            Exp = np.zeros(l1+1+l2+1)
                                            for t in range(0, l1+1+l2+1):
                                                Exp[t] = E(l1,l2+1,t,Ax-Bx,a,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                            VNex += Ndiff1(l2,b)*elnuc(P, p, l1, l2+1, m1, m2, n1, n2, N1, Nxp2, c1, c2, Zc, Exp, Ey, Ez,R1)
                                            if l2 != 0:
                                                Exm = np.zeros(l1-1+l2+1)
                                                for t in range(0, l1+l2-1+1):
                                                    Exm[t] = E(l1,l2-1,t,Ax-Bx,a,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                                VNex += Ndiff2(l2,b)*elnuc(P, p, l1, l2-1, m1, m2, n1, n2, N1, Nxm2, c1, c2, Zc, Exm, Ey, Ez,R1)
                                    
                                    else:
                                        if atomidx == basis[idx[0]][6]:
                                            Exp = np.zeros(l1+1+l2+1)
                                            for t in range(0, l1+1+l2+1):
                                                Exp[t] = E(l1+1,l2,t,Ax-Bx,a,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                            VNex += Ndiff1(l1,a)*elnuc(P, p, l1+1, l2, m1, m2, n1, n2, Nxp1, N2, c1, c2, Zc, Exp, Ey, Ez,R1)
                                            if l1 != 0:
                                                Exm = np.zeros(l1-1+l2+1)
                                                for t in range(0, l1-1+l2+1):
                                                    Exm[t] = E(l1-1,l2,t,Ax-Bx,a,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                                VNex += Ndiff2(l1,a)*elnuc(P, p, l1-1, l2, m1, m2, n1, n2, Nxm1, N2, c1, c2, Zc, Exm, Ey, Ez,R1)
                
                                        if atomidx == basis[idx[1]][6]:

                                            Exp = np.zeros(l1+1+l2+1)
                                            for t in range(0, l1+1+l2+1):
                                                Exp[t] = E(l1,l2+1,t,Ax-Bx,a,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                            VNex += Ndiff1(l2,b)*elnuc(P, p, l1, l2+1, m1, m2, n1, n2, N1, Nxp2, c1, c2, Zc, Exp, Ey, Ez,R1)
                                            if l2 != 0:
                                                Exm = np.zeros(l1-1+l2+1)
                                                for t in range(0, l1+l2-1+1):
                                                    Exm[t] = E(l1,l2-1,t,Ax-Bx,a,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                                VNex += Ndiff2(l2,b)*elnuc(P, p, l1, l2-1, m1, m2, n1, n2, N1, Nxm2, c1, c2, Zc, Exm, Ey, Ez,R1)
                            # y derivative
                                    if atomidx == basis[idx[0]][6] and atomidx == basis[idx[1]][6]:
                                        if atom != atomidx:
                                            Eyp = np.zeros(m1+1+m2+1)
                                            for t in range(0, m1+1+m2+1):
                                                Eyp[t] = E(m1+1,m2,t,Ay-By,a,b,P[1]-Ay,P[1]-By,Ay-By)
                                            VNey += Ndiff1(m1,a)*elnuc(P, p, l1, l2, m1+1, m2, n1, n2, Nyp1, N2, c1, c2, Zc, Ex, Eyp, Ez,R1)
                                            if m1 != 0:
                                                Eym = np.zeros(m1-1+m2+1)
                                                for t in range(0, m1-1+m2+1):
                                                    Eym[t] = E(m1-1,m2,t,Ay-By,a,b,P[1]-Ay,P[1]-By,Ay-By)
                                                VNey += Ndiff2(m1,a)*elnuc(P, p, l1, l2, m1-1, m2, n1, n2, Nym1, N2, c1, c2, Zc, Ex, Eym, Ez,R1)
                                            
                                            Eyp = np.zeros(m1+1+m2+1)
                                            for t in range(0, m1+1+m2+1):
                                                Eyp[t] = E(m1,m2+1,t,Ay-By,a,b,P[1]-Ay,P[1]-By,Ay-By)
                                            VNey += Ndiff1(m2,b)*elnuc(P, p, l1, l2, m1, m2+1, n1, n2, N1, Nyp2, c1, c2, Zc, Ex, Eyp, Ez,R1)
                                            if m2 != 0:
                                                Eym = np.zeros(m1-1+m2+1)
                                                for t in range(0, m1+m2-1+1):
                                                    Eym[t] = E(m1,m2-1,t,Ay-By,a,b,P[1]-Ay,P[1]-By,Ay-By)
                                                VNey += Ndiff2(m2,b)*elnuc(P, p, l1, l2, m1, m2-1, n1, n2, N1, Nym2, c1, c2, Zc, Ex, Eym, Ez,R1)
                                    
                                    else:
                                        if atomidx == basis[idx[0]][6]:
                                            Eyp = np.zeros(m1+1+m2+1)
                                            for t in range(0, m1+1+m2+1):
                                                Eyp[t] = E(m1+1,m2,t,Ay-By,a,b,P[1]-Ay,P[1]-By,Ay-By)
                                            VNey += Ndiff1(m1,a)*elnuc(P, p, l1, l2, m1+1, m2, n1, n2, Nyp1, N2, c1, c2, Zc, Ex, Eyp, Ez,R1)
                                            if m1 != 0:
                                                Eym = np.zeros(m1-1+m2+1)
                                                for t in range(0, m1-1+m2+1):
                                                    Eym[t] = E(m1-1,m2,t,Ay-By,a,b,P[1]-Ay,P[1]-By,Ay-By)
                                                VNey += Ndiff2(m1,a)*elnuc(P, p, l1, l2, m1-1, m2, n1, n2, Nym1, N2, c1, c2, Zc, Ex, Eym, Ez,R1)
                
                                        if atomidx == basis[idx[1]][6]:
                                            Eyp = np.zeros(m1+1+m2+1)
                                            for t in range(0, m1+1+m2+1):
                                                Eyp[t] = E(m1,m2+1,t,Ay-By,a,b,P[1]-Ay,P[1]-By,Ay-By)
                                            VNey += Ndiff1(m2,b)*elnuc(P, p, l1, l2, m1, m2+1, n1, n2, N1, Nyp2, c1, c2, Zc, Ex, Eyp, Ez,R1)
                                            if m2 != 0:
                                                Eym = np.zeros(m1-1+m2+1)
                                                for t in range(0, m1+m2-1+1):
                                                    Eym[t] = E(m1,m2-1,t,Ay-By,a,b,P[1]-Ay,P[1]-By,Ay-By)
                                                VNey += Ndiff2(m2,b)*elnuc(P, p, l1, l2, m1, m2-1, n1, n2, N1, Nym2, c1, c2, Zc, Ex, Eym, Ez,R1)
                                        
                            # z derivative
                                    if atomidx == basis[idx[0]][6] and atomidx == basis[idx[1]][6]:
                                        if atom != atomidx:
                                            Ezp = np.zeros(n1+1+n2+1)
                                            for t in range(0, n1+1+n2+1):
                                                Ezp[t] = E(n1+1,n2,t,Az-Bz,a,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                            VNez += Ndiff1(n1,a)*elnuc(P, p, l1, l2, m1, m2, n1+1, n2, Nzp1, N2, c1, c2, Zc, Ex, Ey, Ezp,R1)
                                            if n1 != 0:
                                                Ezm = np.zeros(n1-1+n2+1)
                                                for t in range(0, n1-1+n2+1):
                                                    Ezm[t] = E(n1-1,n2,t,Az-Bz,a,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                                VNez += Ndiff2(n1,a)*elnuc(P, p, l1, l2, m1, m2, n1-1, n2, Nzm1, N2, c1, c2, Zc, Ex, Ey, Ezm,R1)
                                            
                                            Ezp = np.zeros(n1+1+n2+1)
                                            for t in range(0, n1+1+n2+1):
                                                Ezp[t] = E(n1,n2+1,t,Az-Bz,a,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                            VNez += Ndiff1(n2,b)*elnuc(P, p, l1, l2, m1, m2, n1, n2+1, N1, Nzp2, c1, c2, Zc, Ex, Ey, Ezp,R1)
                                            if n2 != 0:
                                                Ezm = np.zeros(n1-1+n2+1)
                                                for t in range(0, n1+n2-1+1):
                                                    Ezm[t] = E(n1,n2-1,t,Az-Bz,a,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                                VNez += Ndiff2(n2,b)*elnuc(P, p, l1, l2, m1, m2, n1, n2-1, N1, Nzm2, c1, c2, Zc, Ex, Ey, Ezm,R1)
                                    
                                    else:
                                        if atomidx == basis[idx[0]][6]:
                                            Ezp = np.zeros(n1+1+n2+1)
                                            for t in range(0, n1+1+n2+1):
                                                Ezp[t] = E(n1+1,n2,t,Az-Bz,a,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                            VNez += Ndiff1(n1,a)*elnuc(P, p, l1, l2, m1, m2, n1+1, n2, Nzp1, N2, c1, c2, Zc, Ex, Ey, Ezp,R1)
                                            if n1 != 0:
                                                Ezm = np.zeros(n1-1+n2+1)
                                                for t in range(0, n1-1+n2+1):
                                                    Ezm[t] = E(n1-1,n2,t,Az-Bz,a,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                                VNez += Ndiff2(n1,a)*elnuc(P, p, l1, l2, m1, m2, n1-1, n2, Nzm1, N2, c1, c2, Zc, Ex, Ey, Ezm,R1)
                
                                        if atomidx == basis[idx[1]][6]:
                                            Ezp = np.zeros(n1+1+n2+1)
                                            for t in range(0, n1+1+n2+1):
                                                Ezp[t] = E(n1,n2+1,t,Az-Bz,a,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                            VNez += Ndiff1(n2,b)*elnuc(P, p, l1, l2, m1, m2, n1, n2+1, N1, Nzp2, c1, c2, Zc, Ex, Ey, Ezp,R1)
                                            if n2 != 0:
                                                Ezm = np.zeros(n1-1+n2+1)
                                                for t in range(0, n1+n2-1+1):
                                                    Ezm[t] = E(n1,n2-1,t,Az-Bz,a,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                                VNez += Ndiff2(n2,b)*elnuc(P, p, l1, l2, m1, m2, n1, n2-1, N1, Nzm2, c1, c2, Zc, Ex, Ey, Ezm,R1)
                            
                            # Electricfield contribution
                            if atomidx == basis[idx[0]][6] and atomidx == basis[idx[1]][6]:
                                None

                            else:
                                Zc = input[atomidx][0]
                                C = np.array([input[atomidx][1],input[atomidx][2],input[atomidx][3]])
                                RPC = ((P[0]-C[0])**2+(P[1]-C[1])**2+(P[2]-C[2])**2)**0.5
                                R1 = runR(l1+l2, m1+m2, n1+n2, C, P, p, check=1)
                                VNex += electricfield(p,Ex,Ey,Ez,Zc, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, 'x',R1)
                                VNey += electricfield(p,Ex,Ey,Ez,Zc, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, 'y',R1)
                                VNez += electricfield(p,Ex,Ey,Ez,Zc, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, 'z',R1)
                        

                    Sxarr[k,l] = Sx
                    Sxarr[l,k] = Sx
                    Syarr[k,l] = Sy
                    Syarr[l,k] = Sy
                    Szarr[k,l] = Sz
                    Szarr[l,k] = Sz
                    Txarr[k,l] = Tx
                    Txarr[l,k] = Tx
                    Tyarr[k,l] = Ty
                    Tyarr[l,k] = Ty
                    Tzarr[k,l] = Tz
                    Tzarr[l,k] = Tz
                    VNexarr[k,l] = VNex
                    VNexarr[l,k] = VNex
                    VNeyarr[k,l] = VNey
                    VNeyarr[l,k] = VNey
                    VNezarr[k,l] = VNez
                    VNezarr[l,k] = VNez
        np.save('slowquant/temp/'+str(atomidx)+'dxoverlap.npy',Sxarr)
        np.save('slowquant/temp/'+str(atomidx)+'dyoverlap.npy',Syarr)
        np.save('slowquant/temp/'+str(atomidx)+'dzoverlap.npy',Szarr)
        np.save('slowquant/temp/'+str(atomidx)+'dxEkin.npy',Txarr)
        np.save('slowquant/temp/'+str(atomidx)+'dyEkin.npy',Tyarr)
        np.save('slowquant/temp/'+str(atomidx)+'dzEkin.npy',Tzarr)
        np.save('slowquant/temp/'+str(atomidx)+'dxnucatt.npy',VNexarr)
        np.save('slowquant/temp/'+str(atomidx)+'dynucatt.npy',VNeyarr)
        np.save('slowquant/temp/'+str(atomidx)+'dznucatt.npy',VNezarr)
        print(time.time()-start, 'One electron integral diff atom: '+str(atomidx))
        #END OF one electron integrals
        

