import numpy as np
import time as time
import copy
from slowquant.molecularintegrals.MolecularIntegrals import E, Overlap, Kin, elnuc, elelrep, nucdiff, u_ObaraSaika, electricfield, Ndiff1, Ndiff2, Nrun, Eprecalculation, nucrep, R

##CALC OF INTEGRALS
def runIntegrals(input, basis, settings):
    # Nuclear-nuclear repulsion
    E = np.zeros(1)
    E[0] = nucrep(input)
    np.save('slowquant/temp/enuc.npy',E)
    #END OF nuclear-nuclear repulsion
    
    # Precalulations
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
                            calc += elnuc(P, p, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, Zc, C, Ex, Ey, Ez)
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
                lam = mu
                sig = nu
                a = np.zeros(4)
                a[0] = mu
                a[1] = nu
                a[2] = lam
                a[3] = sig
                a = a.astype(int)
                calc = 0
                for i in range(basis[a[0]][4]):
                    for j in range(basis[a[1]][4]):
                        E1 = Edict[str(mu)+str(nu)+str(i)+str(j)+'E1']
                        E2 = Edict[str(mu)+str(nu)+str(i)+str(j)+'E2']
                        E3 = Edict[str(mu)+str(nu)+str(i)+str(j)+'E3']
                        P  = GPdict[str(mu)+str(nu)+str(i)+str(j)]
                        p  = pdict[str(mu)+str(nu)+str(i)+str(j)]
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
                        for k in range(basis[a[2]][4]):
                            for l in range(basis[a[3]][4]):
                                E4 = Edict[str(lam)+str(sig)+str(k)+str(l)+'E1']
                                E5 = Edict[str(lam)+str(sig)+str(k)+str(l)+'E2']
                                E6 = Edict[str(lam)+str(sig)+str(k)+str(l)+'E3']
                                Q  = GPdict[str(lam)+str(sig)+str(k)+str(l)]
                                q  = pdict[str(lam)+str(sig)+str(k)+str(l)]
                                l3=basis[a[2]][5][k][3]
                                l4=basis[a[3]][5][l][3] 
                                m3=basis[a[2]][5][k][4]
                                m4=basis[a[3]][5][l][4] 
                                n3=basis[a[2]][5][k][5]
                                n4=basis[a[3]][5][l][5] 
                                N3=basis[a[2]][5][k][0]
                                N4=basis[a[3]][5][l][0]
                                c3=basis[a[2]][5][k][2]
                                c4=basis[a[3]][5][l][2]
                                
                                alpha = p*q/(p+q)
                                RPQ = ((P[0]-Q[0])**2+(P[1]-Q[1])**2+(P[2]-Q[2])**2)**0.5
                                Rpre = np.ones((l1+l2+l3+l4+1,m1+m2+m3+m4+1,n1+n2+n3+n4+1))
                                Rtemp = np.ones((l1+l2+l3+l4+1,m1+m2+m3+m4+1,n1+n2+n3+n4+1,l1+l2+l3+l4+m1+m2+m3+m4+n1+n2+n3+n4+1))
                                
                                for t_tau in range(l1+l2+l3+l4+1):
                                    for u_nu in range(m1+m2+m3+m4+1):
                                        for v_phi in range(n1+n2+n3+n4+1):
                                            Rpre[t_tau,u_nu,v_phi], Rtemp = R(t_tau,u_nu,v_phi,0,alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ, Rtemp)

                                calc += elelrep(p,q,l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, N3, N4, c1, c2, c3, c4, E1,E2,E3,E4,E5,E6,Rpre)

                ERI[mu,nu,lam,sig] = calc
                ERI[nu,mu,lam,sig] = calc
                ERI[mu,nu,sig,lam] = calc
                ERI[nu,mu,sig,lam] = calc
                ERI[lam,sig,mu,nu] = calc
                ERI[sig,lam,mu,nu] = calc
                ERI[lam,sig,nu,mu] = calc
                ERI[sig,lam,nu,mu] = calc
                Gab[mu,nu] = (calc)**0.5
                
    # Run all the off diagonal elements
    for mu in range(0, len(basis)):
        for nu in range(0, len(basis)):
            if mu >= nu:
                for lam in range(0, len(basis)):
                    for sig in range(0, len(basis)):
                        # Cauchy-Schwarz inequality
                        if Gab[mu,nu]*Gab[lam,sig] > ScreenTHR:
                            munu = mu*(mu+1)/2+nu
                            lamsig = lam*(lam+1)/2+sig
                            if lam >= sig and munu > lamsig:
                                a = np.zeros(4)
                                a[0] = mu
                                a[1] = nu
                                a[2] = lam
                                a[3] = sig
                                a = a.astype(int)
                                calc = 0
                                for i in range(basis[a[0]][4]):
                                    for j in range(basis[a[1]][4]):
                                        E1 = Edict[str(mu)+str(nu)+str(i)+str(j)+'E1']
                                        E2 = Edict[str(mu)+str(nu)+str(i)+str(j)+'E2']
                                        E3 = Edict[str(mu)+str(nu)+str(i)+str(j)+'E3']
                                        P  = GPdict[str(mu)+str(nu)+str(i)+str(j)]
                                        p  = pdict[str(mu)+str(nu)+str(i)+str(j)]
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
                                        for k in range(basis[a[2]][4]):
                                            for l in range(basis[a[3]][4]):
                                                E4 = Edict[str(lam)+str(sig)+str(k)+str(l)+'E1']
                                                E5 = Edict[str(lam)+str(sig)+str(k)+str(l)+'E2']
                                                E6 = Edict[str(lam)+str(sig)+str(k)+str(l)+'E3']
                                                Q  = GPdict[str(lam)+str(sig)+str(k)+str(l)]
                                                q  = pdict[str(lam)+str(sig)+str(k)+str(l)]
                                                l3=basis[a[2]][5][k][3]
                                                l4=basis[a[3]][5][l][3] 
                                                m3=basis[a[2]][5][k][4]
                                                m4=basis[a[3]][5][l][4] 
                                                n3=basis[a[2]][5][k][5]
                                                n4=basis[a[3]][5][l][5] 
                                                N3=basis[a[2]][5][k][0]
                                                N4=basis[a[3]][5][l][0]
                                                c3=basis[a[2]][5][k][2]
                                                c4=basis[a[3]][5][l][2]
                                                
                                                alpha = p*q/(p+q)
                                                RPQ = ((P[0]-Q[0])**2+(P[1]-Q[1])**2+(P[2]-Q[2])**2)**0.5
                                                Rpre = np.ones((l1+l2+l3+l4+1,m1+m2+m3+m4+1,n1+n2+n3+n4+1))
                                                Rtemp = np.ones((l1+l2+l3+l4+1,m1+m2+m3+m4+1,n1+n2+n3+n4+1,l1+l2+l3+l4+m1+m2+m3+m4+n1+n2+n3+n4+1))
                                                
                                                for t_tau in range(l1+l2+l3+l4+1):
                                                    for u_nu in range(m1+m2+m3+m4+1):
                                                        for v_phi in range(n1+n2+n3+n4+1):
                                                            Rpre[t_tau,u_nu,v_phi], Rtemp = R(t_tau,u_nu,v_phi,0,alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ, Rtemp)
                
                                                calc += elelrep(p,q,l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, N3, N4, c1, c2, c3, c4, E1,E2,E3,E4,E5,E6,Rpre)
                                
                                ERI[mu,nu,lam,sig] = calc
                                ERI[nu,mu,lam,sig] = calc
                                ERI[mu,nu,sig,lam] = calc
                                ERI[nu,mu,sig,lam] = calc
                                ERI[lam,sig,mu,nu] = calc
                                ERI[sig,lam,mu,nu] = calc
                                ERI[lam,sig,nu,mu] = calc
                                ERI[sig,lam,nu,mu] = calc

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

                        Zc = -1
                        C = np.array([rcx, rcy ,rcz])
                        calc += elnuc(P, p, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, Zc, C, Ex, Ey, Ez)
            Ve[k,l] = calc
            Ve[l,k] = calc
        
    return Ve

def rungeometric_derivatives(input, basis):
    # Calculating the normalization of the derivatives. For now only used in ERI
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
    
    # Precalulations
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
                                calcx = 0
                                calcy = 0
                                calcz = 0
                                if atomidx == basis[a[0]][6] and atomidx == basis[a[1]][6] and atomidx == basis[a[2]][6] and atomidx == basis[a[3]][6]:
                                    calcx = 0
                                    calcy = 0
                                    calcz = 0
                                else:
                                    for i in range(basis[a[0]][4]):
                                        for j in range(basis[a[1]][4]):
                                            E1 = Edict[str(mu)+str(nu)+str(i)+str(j)+'E1']
                                            E2 = Edict[str(mu)+str(nu)+str(i)+str(j)+'E2']
                                            E3 = Edict[str(mu)+str(nu)+str(i)+str(j)+'E3']
                                            P  = GPdict[str(mu)+str(nu)+str(i)+str(j)]
                                            p  = pdict[str(mu)+str(nu)+str(i)+str(j)]
                                            a2=basis[a[0]][5][i][1]
                                            b=basis[a[1]][5][j][1]
                                            Ax=basis[a[0]][1]
                                            Bx=basis[a[1]][1]
                                            Ay=basis[a[0]][2]
                                            By=basis[a[1]][2]
                                            Az=basis[a[0]][3] 
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
                                            Nxp1=Nxplus[a[0]][5][i][0]
                                            Nxp2=Nxplus[a[1]][5][j][0]
                                            Nxm1=Nxminus[a[0]][5][i][0]
                                            Nxm2=Nxminus[a[1]][5][j][0]
                                            Nyp1=Nyplus[a[0]][5][i][0]
                                            Nyp2=Nyplus[a[1]][5][j][0]
                                            Nym1=Nyminus[a[0]][5][i][0]
                                            Nym2=Nyminus[a[1]][5][j][0]
                                            Nzp1=Nzplus[a[0]][5][i][0]
                                            Nzp2=Nzplus[a[1]][5][j][0]
                                            Nzm1=Nzminus[a[0]][5][i][0]
                                            Nzm2=Nzminus[a[1]][5][j][0]
                                            for k in range(basis[a[2]][4]):
                                                for l in range(basis[a[3]][4]):
                                                    E4 = Edict[str(lam)+str(sig)+str(k)+str(l)+'E1']
                                                    E5 = Edict[str(lam)+str(sig)+str(k)+str(l)+'E2']
                                                    E6 = Edict[str(lam)+str(sig)+str(k)+str(l)+'E3']
                                                    Q  = GPdict[str(lam)+str(sig)+str(k)+str(l)]
                                                    q  = pdict[str(lam)+str(sig)+str(k)+str(l)]
                                                    c=basis[a[2]][5][k][1]
                                                    d=basis[a[3]][5][l][1]
                                                    Cx=basis[a[2]][1]
                                                    Dx=basis[a[3]][1]
                                                    Cy=basis[a[2]][2]
                                                    Dy=basis[a[3]][2]
                                                    Cz=basis[a[2]][3]
                                                    Dz=basis[a[3]][3]
                                                    l3=basis[a[2]][5][k][3]
                                                    l4=basis[a[3]][5][l][3] 
                                                    m3=basis[a[2]][5][k][4]
                                                    m4=basis[a[3]][5][l][4] 
                                                    n3=basis[a[2]][5][k][5]
                                                    n4=basis[a[3]][5][l][5] 
                                                    N3=basis[a[2]][5][k][0]
                                                    N4=basis[a[3]][5][l][0]
                                                    c3=basis[a[2]][5][k][2]
                                                    c4=basis[a[3]][5][l][2]    
                                                    Nxp3=Nxplus[a[2]][5][k][0]
                                                    Nxp4=Nxplus[a[3]][5][l][0]
                                                    Nxm3=Nxminus[a[2]][5][k][0]
                                                    Nxm4=Nxminus[a[3]][5][l][0]  
                                                    Nyp3=Nyplus[a[2]][5][k][0]
                                                    Nyp4=Nyplus[a[3]][5][l][0]
                                                    Nym3=Nyminus[a[2]][5][k][0]
                                                    Nym4=Nyminus[a[3]][5][l][0]
                                                    Nzp3=Nzplus[a[2]][5][k][0]
                                                    Nzp4=Nzplus[a[3]][5][l][0]
                                                    Nzm3=Nzminus[a[2]][5][k][0]
                                                    Nzm4=Nzminus[a[3]][5][l][0]
                                                    
                                                    alpha = p*q/(p+q)
                                                    RPQ = ((P[0]-Q[0])**2+(P[1]-Q[1])**2+(P[2]-Q[2])**2)**0.5
                                                    Rpre = np.ones((l1+l2+l3+l4+2,m1+m2+m3+m4+2,n1+n2+n3+n4+2))
                                                    Rtemp = np.ones((l1+l2+l3+l4+2,m1+m2+m3+m4+2,n1+n2+n3+n4+2,l1+l2+l3+l4+m1+m2+m3+m4+n1+n2+n3+n4+4))
                                                    
                                                    for t_tau in range(l1+l2+l3+l4+2):
                                                        for u_nu in range(m1+m2+m3+m4+2):
                                                            for v_phi in range(n1+n2+n3+n4+2):
                                                                Rpre[t_tau,u_nu,v_phi], Rtemp = R(t_tau,u_nu,v_phi,0,alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ, Rtemp)
                                                    
                                                    # Calculate x derivative
                                                    if atomidx == basis[a[0]][6]:
                                                        E1p = np.zeros(l1+1+l2+1)
                                                        for t in range(l1+1+l2+1):
                                                            E1p[t] = E(l1+1,l2,t,Ax-Bx,a2,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                                        calcx += Ndiff1(l1, a2)*elelrep(p, q, l1+1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, Nxp1, N2, N3, N4, c1, c2, c3, c4, E1p,E2,E3,E4,E5,E6,Rpre)
                                                        if l1 != 0:
                                                            E1m = np.zeros(l1-1+l2+1)
                                                            for t in range(l1-1+l2+1):
                                                                E1m[t] = E(l1-1,l2,t,Ax-Bx,a2,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                                            calcx += Ndiff2(l1, a2)*elelrep(p, q,l1-1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, Nxm1, N2, N3, N4, c1, c2, c3, c4, E1m,E2,E3,E4,E5,E6,Rpre)
                                                            
                                                    if atomidx == basis[a[1]][6]:
                                                        E1p = np.zeros(l1+1+l2+1)
                                                        for t in range(l1+l2+1+1):
                                                            E1p[t] = E(l1,l2+1,t,Ax-Bx,a2,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                                        calcx += Ndiff1(l2, b)*elelrep(p, q, l1, l2+1, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, Nxp2, N3, N4, c1, c2, c3, c4, E1p,E2,E3,E4,E5,E6,Rpre)
                                                        if l2 != 0:
                                                            E1m = np.zeros(l1-1+l2+1)
                                                            for t in range(l1+l2-1+1):
                                                                E1m[t] = E(l1,l2-1,t,Ax-Bx,a2,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                                            calcx += Ndiff2(l2, b)*elelrep(p, q, l1, l2-1, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, Nxm2, N3, N4, c1, c2, c3, c4, E1m,E2,E3,E4,E5,E6,Rpre)
                                                            
                                                    if atomidx == basis[a[2]][6]:  
                                                        E4p = np.zeros(l3+1+l4+1)
                                                        for t in range(l3+1+l4+1):
                                                            E4p[t] = E(l3+1,l4,t,Cx-Dx,c,d,Q[0]-Cx,Q[0]-Dx,Cx-Dx)
                                                        calcx += Ndiff1(l3, c)*elelrep(p, q, l1, l2, l3+1, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, Nxp3, N4, c1, c2, c3, c4, E1,E2,E3,E4p,E5,E6,Rpre)
                                                        if l3 != 0:
                                                            E4m = np.zeros(l3-1+l4+1)
                                                            for t in range(l3-1+l4+1):
                                                                E4m[t] = E(l3-1,l4,t,Cx-Dx,c,d,Q[0]-Cx,Q[0]-Dx,Cx-Dx)
                                                            calcx += Ndiff2(l3, c)*elelrep(p, q, l1, l2, l3-1, l4, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, Nxm3, N4, c1, c2, c3, c4, E1,E2,E3,E4m,E5,E6,Rpre)
                                                        
                                                    if atomidx == basis[a[3]][6]:
                                                        E4p = np.zeros(l3+1+l4+1)
                                                        for t in range(l3+l4+1+1):
                                                            E4p[t] = E(l3,l4+1,t,Cx-Dx,c,d,Q[0]-Cx,Q[0]-Dx,Cx-Dx)
                                                        calcx += Ndiff1(l4, d)*elelrep(p, q, l1, l2, l3, l4+1, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, N3, Nxp4, c1, c2, c3, c4, E1,E2,E3,E4p,E5,E6,Rpre)
                                                        if l4 != 0:
                                                            E4m = np.zeros(l3-1+l4+1)
                                                            for t in range(l3+l4-1+1):
                                                                E4m[t] = E(l3,l4-1,t,Cx-Dx,c,d,Q[0]-Cx,Q[0]-Dx,Cx-Dx)
                                                            calcx += Ndiff2(l4, d)*elelrep(p, q, l1, l2, l3, l4-1, m1, m2, m3, m4, n1, n2, n3, n4, N1, N2, N3, Nxm4, c1, c2, c3, c4, E1,E2,E3,E4m,E5,E6,Rpre)
                                                    
                                                    # Calculate y derivative
                                                    if atomidx == basis[a[0]][6]:
                                                        E2p = np.zeros(m1+1+m2+1)
                                                        for t in range(m1+1+m2+1):
                                                            E2p[t] = E(m1+1,m2,t,Ay-By,a2,b,P[1]-Ay,P[1]-By,Ay-By)
                                                        calcy += Ndiff1(m1, a2)*elelrep(p, q, l1, l2, l3, l4, m1+1, m2, m3, m4, n1, n2, n3, n4, Nyp1, N2, N3, N4, c1, c2, c3, c4, E1,E2p,E3,E4,E5,E6,Rpre)
                                                        if m1 != 0:
                                                            E2m = np.zeros(m1-1+m2+1)
                                                            for t in range(m1-1+m2+1):
                                                                E2m[t] = E(m1-1,m2,t,Ay-By,a2,b,P[1]-Ay,P[1]-By,Ay-By)
                                                            calcy += Ndiff2(m1, a2)*elelrep(p, q, l1, l2, l3, l4, m1-1, m2, m3, m4, n1, n2, n3, n4, Nym1, N2, N3, N4, c1, c2, c3, c4, E1,E2m,E3,E4,E5,E6,Rpre)
                                                            
                                                    if atomidx == basis[a[1]][6]:
                                                        E2p = np.zeros(m1+1+m2+1)
                                                        for t in range(m1+m2+1+1):
                                                            E2p[t] = E(m1,m2+1,t,Ay-By,a2,b,P[1]-Ay,P[1]-By,Ay-By)
                                                        calcy += Ndiff1(m2, b)*elelrep(p, q, l1, l2, l3, l4, m1, m2+1, m3, m4, n1, n2, n3, n4, N1, Nyp2, N3, N4, c1, c2, c3, c4, E1,E2p,E3,E4,E5,E6,Rpre)
                                                        if m2 != 0:
                                                            E2m = np.zeros(m1-1+m2+1)
                                                            for t in range(m1+m2-1+1):
                                                                E2m[t] = E(m1,m2-1,t,Ay-By,a2,b,P[1]-Ay,P[1]-By,Ay-By)
                                                            calcy += Ndiff2(m2, b)*elelrep(p, q, l1, l2, l3, l4, m1, m2-1, m3, m4, n1, n2, n3, n4, N1, Nym2, N3, N4, c1, c2, c3, c4,E1,E2m,E3,E4,E5,E6,Rpre)
                                                            
                                                    if atomidx == basis[a[2]][6]:  
                                                        E5p = np.zeros(m3+1+m4+1)
                                                        for t in range(m3+1+m4+1):
                                                            E5p[t] = E(m3+1,m4,t,Cy-Dy,c,d,Q[1]-Cy,Q[1]-Dy,Cy-Dy)
                                                        calcy += Ndiff1(m3, c)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3+1, m4, n1, n2, n3, n4, N1, N2, Nyp3, N4, c1, c2, c3, c4, E1,E2,E3,E4,E5p,E6,Rpre)
                                                        if m3 != 0:
                                                            E5m = np.zeros(m3-1+m4+1)
                                                            for t in range(m3-1+m4+1):
                                                                E5m[t] = E(m3-1,m4,t,Cy-Dy,c,d,Q[1]-Cy,Q[1]-Dy,Cy-Dy)
                                                            calcy += Ndiff2(m3, c)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3-1, m4, n1, n2, n3, n4, N1, N2, Nym3, N4, c1, c2, c3, c4,E1,E2,E3,E4,E5m,E6,Rpre)
                                                        
                                                    if atomidx == basis[a[3]][6]:
                                                        E5p = np.zeros(m3+1+m4+1)
                                                        for t in range(m3+m4+1+1):
                                                            E5p[t] = E(m3,m4+1,t,Cy-Dy,c,d,Q[1]-Cy,Q[1]-Dy,Cy-Dy)
                                                        calcy += Ndiff1(m4, d)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4+1, n1, n2, n3, n4, N1, N2, N3, Nyp4, c1, c2, c3, c4,E1,E2,E3,E4,E5p,E6,Rpre)
                                                        if m4 != 0:
                                                            E5m = np.zeros(m3-1+m4+1)
                                                            for t in range(m3+m4-1+1):
                                                                E5m[t] = E(m3,m4-1,t,Cy-Dy,c,d,Q[1]-Cy,Q[1]-Dy,Cy-Dy)
                                                            calcy += Ndiff2(m4, d)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4-1, n1, n2, n3, n4, N1, N2, N3, Nym4, c1, c2, c3, c4,E1,E2,E3,E4,E5m,E6,Rpre)
                                                    
                                                    # Calculate z derivative
                                                    if atomidx == basis[a[0]][6]:
                                                        E3p = np.zeros(n1+1+n2+1)
                                                        for t in range(n1+1+n2+1):
                                                            E3p[t] = E(n1+1,n2,t,Az-Bz,a2,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                                        calcz += Ndiff1(n1, a2)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1+1, n2, n3, n4, Nzp1, N2, N3, N4, c1, c2, c3, c4, E1,E2,E3p,E4,E5,E6,Rpre)
                                                        if n1 != 0:
                                                            E3m = np.zeros(n1-1+n2+1)
                                                            for t in range(n1-1+n2+1):
                                                                E3m[t] = E(n1-1,n2,t,Az-Bz,a2,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                                            calcz += Ndiff2(n1, a2)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1-1, n2, n3, n4, Nzm1, N2, N3, N4, c1, c2, c3, c4, E1,E2,E3m,E4,E5,E6,Rpre)
                                                            
                                                    if atomidx == basis[a[1]][6]:
                                                        E3p = np.zeros(n1+1+n2+1)
                                                        for t in range(n1+n2+1+1):
                                                            E3p[t] = E(n1,n2+1,t,Az-Bz,a2,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                                        calcz += Ndiff1(n2, b)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2+1, n3, n4, N1, Nzp2, N3, N4, c1, c2, c3, c4, E1,E2,E3p,E4,E5,E6,Rpre)
                                                        if n2 != 0:
                                                            E3m = np.zeros(n1-1+n2+1)
                                                            for t in range(n1+n2-1+1):
                                                                E3m[t] = E(n1,n2-1,t,Az-Bz,a2,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                                            calcz += Ndiff2(n2, b)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2-1, n3, n4, N1, Nzm2, N3, N4, c1, c2, c3, c4,E1,E2,E3m,E4,E5,E6,Rpre)
                                                            
                                                    if atomidx == basis[a[2]][6]: 
                                                        E6p = np.zeros(n3+1+n4+1)
                                                        for t in range(n3+1+n4+1):
                                                            E6p[t] = E(n3+1,n4,t,Cz-Dz,c,d,Q[2]-Cz,Q[2]-Dz,Cz-Dz)
                                                        calcz += Ndiff1(n3, c)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3+1, n4, N1, N2, Nzp3, N4, c1, c2, c3, c4, E1,E2,E3,E4,E5,E6p,Rpre)
                                                        if n3 != 0:
                                                            E6m = np.zeros(n3-1+n4+1)
                                                            for t in range(n3-1+n4+1):
                                                                E6m[t] = E(n3-1,n4,t,Cz-Dz,c,d,Q[2]-Cz,Q[2]-Dz,Cz-Dz)
                                                            calcz += Ndiff2(n3, c)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3-1, n4, N1, N2, Nzm3, N4, c1, c2, c3, c4,E1,E2,E3,E4,E5,E6m,Rpre)
                                                        
                                                    if atomidx == basis[a[3]][6]:
                                                        E6p = np.zeros(n3+1+n4+1)
                                                        for t in range(n3+n4+1+1):
                                                            E6p[t] = E(n3,n4+1,t,Cz-Dz,c,d,Q[2]-Cz,Q[2]-Dz,Cz-Dz)
                                                        calcz += Ndiff1(n4, d)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4+1, N1, N2, N3, Nzp4, c1, c2, c3, c4,E1,E2,E3,E4,E5,E6p,Rpre)
                                                        if n4 != 0:
                                                            E6m = np.zeros(n3-1+n4+1)
                                                            for t in range(n3+n4-1+1):
                                                                E6m[t] = E(n3,n4-1,t,Cz-Dz,c,d,Q[2]-Cz,Q[2]-Dz,Cz-Dz)
                                                            calcz += Ndiff2(n4, d)*elelrep(p, q, l1, l2, l3, l4, m1, m2, m3, m4, n1, n2, n3, n4-1, N1, N2, N3, Nzm4, c1, c2, c3, c4,E1,E2,E3,E4,E5,E6m,Rpre)
                                                                                        
                                ERIx[mu,nu,lam,sig] = calcx
                                ERIx[nu,mu,lam,sig] = calcx
                                ERIx[mu,nu,sig,lam] = calcx
                                ERIx[nu,mu,sig,lam] = calcx
                                ERIx[lam,sig,mu,nu] = calcx
                                ERIx[sig,lam,mu,nu] = calcx
                                ERIx[lam,sig,nu,mu] = calcx
                                ERIx[sig,lam,nu,mu] = calcx
                                ERIy[mu,nu,lam,sig] = calcy
                                ERIy[nu,mu,lam,sig] = calcy
                                ERIy[mu,nu,sig,lam] = calcy
                                ERIy[nu,mu,sig,lam] = calcy
                                ERIy[lam,sig,mu,nu] = calcy
                                ERIy[sig,lam,mu,nu] = calcy
                                ERIy[lam,sig,nu,mu] = calcy
                                ERIy[sig,lam,nu,mu] = calcy
                                ERIz[mu,nu,lam,sig] = calcz
                                ERIz[nu,mu,lam,sig] = calcz
                                ERIz[mu,nu,sig,lam] = calcz
                                ERIz[nu,mu,sig,lam] = calcz
                                ERIz[lam,sig,mu,nu] = calcz
                                ERIz[sig,lam,mu,nu] = calcz
                                ERIz[lam,sig,nu,mu] = calcz
                                ERIz[sig,lam,nu,mu] = calcz
        np.save('slowquant/temp/'+str(atomidx)+'dxtwoint.npy',ERIx)
        np.save('slowquant/temp/'+str(atomidx)+'dytwoint.npy',ERIy)
        np.save('slowquant/temp/'+str(atomidx)+'dztwoint.npy',ERIz)
        print(time.time()-start, 'ERI diff: atom'+str(atomidx))
        #END OF two electron integrals x diff
        
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
                                N1=basis[a[0]][5][i][0]
                                N2=basis[a[1]][5][j][0]
                                c1=basis[a[0]][5][i][2]
                                c2=basis[a[1]][5][j][2]
                                
                                N1p=Nxplus[a[0]][5][i][0]
                                N2p=Nxplus[a[1]][5][j][0]
                                
                                N1m=Nxminus[a[0]][5][i][0]
                                N2m=Nxminus[a[1]][5][j][0]
                                
                                if atomidx == basis[a[0]][6]:
                                    calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la+1, lb, ma, mb, na, nb, N1p, N2, c1, c2)
                                    calc += calct*Ndiff1(la,a2)
                                    calc2 += calct2*Ndiff1(la,a2)
                                    if la != 0:
                                        calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la-1, lb, ma, mb, na, nb, N1m, N2, c1, c2)
                                        calc += calct*Ndiff2(la,a2)
                                        calc2 += calct2*Ndiff2(la,a2)
                                        
                                if atomidx == basis[a[1]][6]:
                                    calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb+1, ma, mb, na, nb, N1, N2p, c1, c2)
                                    calc += calct*Ndiff1(lb,b)
                                    calc2 += calct2*Ndiff1(lb,b)
                                    if lb != 0:
                                        calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb-1, ma, mb, na, nb, N1, N2m, c1, c2)
                                        calc += calct*Ndiff2(lb,b)
                                        calc2 += calct2*Ndiff2(lb,b)
                    S[k,l] = calc2
                    S[l,k] = calc2
                    T[k,l] = calc
                    T[l,k] = calc
        np.save('slowquant/temp/'+str(atomidx)+'dxoverlap.npy',S)
        np.save('slowquant/temp/'+str(atomidx)+'dxEkin.npy',T)
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
                                N1=basis[a[0]][5][i][0]
                                N2=basis[a[1]][5][j][0]
                                c1=basis[a[0]][5][i][2]
                                c2=basis[a[1]][5][j][2]
                                
                                N1p=Nyplus[a[0]][5][i][0]
                                N2p=Nyplus[a[1]][5][j][0]
                                
                                N1m=Nyminus[a[0]][5][i][0]
                                N2m=Nyminus[a[1]][5][j][0]
                                
                                if atomidx == basis[a[0]][6]:
                                    calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma+1, mb, na, nb, N1p, N2, c1, c2)
                                    calc += calct*Ndiff1(ma,a2)
                                    calc2 += calct2*Ndiff1(ma,a2)
                                    if ma != 0:
                                        calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma-1, mb, na, nb, N1m, N2, c1, c2)
                                        calc += calct*Ndiff2(ma,a2)
                                        calc2 += calct2*Ndiff2(ma,a2)
                                        
                                if atomidx == basis[a[1]][6]:
                                    calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma, mb+1, na, nb, N1, N2p, c1, c2)
                                    calc += calct*Ndiff1(mb,b)
                                    calc2 += calct2*Ndiff1(mb,b)
                                    if mb != 0:
                                        calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma, mb-1, na, nb, N1, N2m, c1, c2)
                                        calc += calct*Ndiff2(mb,b)
                                        calc2 += calct2*Ndiff2(mb,b)
                    S[k,l] = calc2
                    S[l,k] = calc2
                    T[k,l] = calc
                    T[l,k] = calc
        np.save('slowquant/temp/'+str(atomidx)+'dyoverlap.npy',S)
        np.save('slowquant/temp/'+str(atomidx)+'dyEkin.npy',T)
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
                                N1=basis[a[0]][5][i][0]
                                N2=basis[a[1]][5][j][0]
                                c1=basis[a[0]][5][i][2]
                                c2=basis[a[1]][5][j][2]
                                
                                N1p=Nzplus[a[0]][5][i][0]
                                N2p=Nzplus[a[1]][5][j][0]
                                
                                N1m=Nzminus[a[0]][5][i][0]
                                N2m=Nzminus[a[1]][5][j][0]
                                
                                if atomidx == basis[a[0]][6]:
                                    calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma, mb, na+1, nb, N1p, N2, c1, c2)
                                    calc += calct*Ndiff1(na,a2)
                                    calc2 += calct2*Ndiff1(na,a2)
                                    if na != 0:
                                        calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma, mb, na-1, nb, N1m, N2, c1, c2)
                                        calc += calct*Ndiff2(na,a2)
                                        calc2 += calct2*Ndiff2(na,a2)
                                        
                                if atomidx == basis[a[1]][6]:
                                    calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma, mb, na, nb+1, N1, N2p, c1, c2)
                                    calc += calct*Ndiff1(nb,b)
                                    calc2 += calct2*Ndiff1(nb,b)
                                    if nb != 0:
                                        calct, calct2 = Kin(a2, b, Ax, Ay, Az, Bx, By, Bz, la, lb, ma, mb, na, nb-1, N1, N2m, c1, c2)
                                        calc += calct*Ndiff2(nb,b)
                                        calc2 += calct2*Ndiff2(nb,b)
                    S[k,l] = calc2
                    S[l,k] = calc2
                    T[k,l] = calc
                    T[l,k] = calc
        np.save('slowquant/temp/'+str(atomidx)+'dzoverlap.npy',S)
        np.save('slowquant/temp/'+str(atomidx)+'dzEkin.npy',T)
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
                            Ex = Edict[str(k)+str(l)+str(i)+str(j)+'E1']
                            Ey = Edict[str(k)+str(l)+str(i)+str(j)+'E2']
                            Ez = Edict[str(k)+str(l)+str(i)+str(j)+'E3']
                            P  = GPdict[str(k)+str(l)+str(i)+str(j)]
                            p  = pdict[str(k)+str(l)+str(i)+str(j)]
                        
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
                              
                            N1p=Nxplus[a[0]][5][i][0]
                            N2p=Nxplus[a[1]][5][j][0]
                            
                            N1m=Nxminus[a[0]][5][i][0]
                            N2m=Nxminus[a[1]][5][j][0]
                            
                            if atomidx == basis[a[0]][6] and atomidx == basis[a[1]][6]:
                                for atom in range(1, len(input)):
                                    Zc = input[atom][0]
                                    C = np.array([input[atom][1],input[atom][2],input[atom][3]])
                                    
                                    if atom != atomidx:
                                        Exp = np.zeros(l1+1+l2+1)
                                        for t in range(0, l1+1+l2+1):
                                            Exp[t] = E(l1+1,l2,t,Ax-Bx,a2,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                        calc += Ndiff1(l1,a2)*elnuc(P, p, l1+1, l2, m1, m2, n1, n2, N1p, N2, c1, c2, Zc, C, Exp, Ey, Ez)
                                        if l1 != 0:
                                            Exm = np.zeros(l1-1+l2+1)
                                            for t in range(0, l1-1+l2+1):
                                                Exm[t] = E(l1-1,l2,t,Ax-Bx,a2,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                            calc += Ndiff2(l1,a2)*elnuc(P, p, l1-1, l2, m1, m2, n1, n2, N1m, N2, c1, c2, Zc, C, Exm, Ey, Ez)
                                        
                                        Exp = np.zeros(l1+1+l2+1)
                                        for t in range(0, l1+1+l2+1):
                                            Exp[t] = E(l1,l2+1,t,Ax-Bx,a2,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                        calc += Ndiff1(l2,b)*elnuc(P, p, l1, l2+1, m1, m2, n1, n2, N1, N2p, c1, c2, Zc, C, Exp, Ey, Ez)
                                        if l2 != 0:
                                            Exm = np.zeros(l1-1+l2+1)
                                            for t in range(0, l1+l2-1+1):
                                                Exm[t] = E(l1,l2-1,t,Ax-Bx,a2,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                            calc += Ndiff2(l2,b)*elnuc(P, p, l1, l2-1, m1, m2, n1, n2, N1, N2m, c1, c2, Zc, C, Exm, Ey, Ez)
                            
                            else:
                                calc += electricfield(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, input, 'x', atomidx)
                                
                                if atomidx == basis[a[0]][6]:
                                    for atom in range(1, len(input)):
                                        Zc = input[atom][0]
                                        C = np.array([input[atom][1],input[atom][2],input[atom][3]])
                                        Exp = np.zeros(l1+1+l2+1)
                                        for t in range(0, l1+1+l2+1):
                                            Exp[t] = E(l1+1,l2,t,Ax-Bx,a2,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                        calc += Ndiff1(l1,a2)*elnuc(P, p, l1+1, l2, m1, m2, n1, n2, N1p, N2, c1, c2, Zc, C, Exp, Ey, Ez)
                                        if l1 != 0:
                                            Exm = np.zeros(l1-1+l2+1)
                                            for t in range(0, l1-1+l2+1):
                                                Exm[t] = E(l1-1,l2,t,Ax-Bx,a2,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                            calc += Ndiff2(l1,a2)*elnuc(P, p, l1-1, l2, m1, m2, n1, n2, N1m, N2, c1, c2, Zc, C, Exm, Ey, Ez)
        
                                if atomidx == basis[a[1]][6]:
                                    for atom in range(1, len(input)):
                                        Zc = input[atom][0]
                                        C = np.array([input[atom][1],input[atom][2],input[atom][3]])
                                        Exp = np.zeros(l1+1+l2+1)
                                        for t in range(0, l1+1+l2+1):
                                            Exp[t] = E(l1,l2+1,t,Ax-Bx,a2,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                        calc += Ndiff1(l2,b)*elnuc(P, p, l1, l2+1, m1, m2, n1, n2, N1, N2p, c1, c2, Zc, C, Exp, Ey, Ez)
                                        if l2 != 0:
                                            Exm = np.zeros(l1-1+l2+1)
                                            for t in range(0, l1+l2-1+1):
                                                Exm[t] = E(l1,l2-1,t,Ax-Bx,a2,b,P[0]-Ax,P[0]-Bx,Ax-Bx)
                                            calc += Ndiff2(l2,b)*elnuc(P, p, l1, l2-1, m1, m2, n1, n2, N1, N2m, c1, c2, Zc, C, Exm, Ey, Ez)

                    Na[k,l] = calc
                    Na[l,k] = calc
        np.save('slowquant/temp/'+str(atomidx)+'dxnucatt.npy',Na)
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
                            Ex = Edict[str(k)+str(l)+str(i)+str(j)+'E1']
                            Ey = Edict[str(k)+str(l)+str(i)+str(j)+'E2']
                            Ez = Edict[str(k)+str(l)+str(i)+str(j)+'E3']
                            P  = GPdict[str(k)+str(l)+str(i)+str(j)]
                            p  = pdict[str(k)+str(l)+str(i)+str(j)]
                            
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
                              
                            N1p=Nyplus[a[0]][5][i][0]
                            N2p=Nyplus[a[1]][5][j][0]
                            
                            N1m=Nyminus[a[0]][5][i][0]
                            N2m=Nyminus[a[1]][5][j][0]
                            
                            if atomidx == basis[a[0]][6] and atomidx == basis[a[1]][6]:
                                for atom in range(1, len(input)):
                                    Zc = input[atom][0]
                                    C = np.array([input[atom][1],input[atom][2],input[atom][3]])
                                    
                                    if atom != atomidx:
                                        Eyp = np.zeros(m1+1+m2+1)
                                        for t in range(0, m1+1+m2+1):
                                            Eyp[t] = E(m1+1,m2,t,Ay-By,a2,b,P[1]-Ay,P[1]-By,Ay-By)
                                        calc += Ndiff1(m1,a2)*elnuc(P, p, l1, l2, m1+1, m2, n1, n2, N1p, N2, c1, c2, Zc, C, Ex, Eyp, Ez)
                                        if m1 != 0:
                                            Eym = np.zeros(m1-1+m2+1)
                                            for t in range(0, m1-1+m2+1):
                                                Eym[t] = E(m1-1,m2,t,Ay-By,a2,b,P[1]-Ay,P[1]-By,Ay-By)
                                            calc += Ndiff2(m1,a2)*elnuc(P, p, l1, l2, m1-1, m2, n1, n2, N1m, N2, c1, c2, Zc, C, Ex, Eym, Ez)
                                        
                                        Eyp = np.zeros(m1+1+m2+1)
                                        for t in range(0, m1+1+m2+1):
                                            Eyp[t] = E(m1,m2+1,t,Ay-By,a2,b,P[1]-Ay,P[1]-By,Ay-By)
                                        calc += Ndiff1(m2,b)*elnuc(P, p, l1, l2, m1, m2+1, n1, n2, N1, N2p, c1, c2, Zc, C, Ex, Eyp, Ez)
                                        if m2 != 0:
                                            Eym = np.zeros(m1-1+m2+1)
                                            for t in range(0, m1+m2-1+1):
                                                Eym[t] = E(m1,m2-1,t,Ay-By,a2,b,P[1]-Ay,P[1]-By,Ay-By)
                                            calc += Ndiff2(m2,b)*elnuc(P, p, l1, l2, m1, m2-1, n1, n2, N1, N2m, c1, c2, Zc, C, Ex, Eym, Ez)
                            
                            else:
                                calc += electricfield(a2, b, Ax, Ay, Az, Bx, By, Bz, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, input, 'y', atomidx)
                                
                                if atomidx == basis[a[0]][6]:
                                    for atom in range(1, len(input)):
                                        Zc = input[atom][0]
                                        C = np.array([input[atom][1],input[atom][2],input[atom][3]])
                                        Eyp = np.zeros(m1+1+m2+1)
                                        for t in range(0, m1+1+m2+1):
                                            Eyp[t] = E(m1+1,m2,t,Ay-By,a2,b,P[1]-Ay,P[1]-By,Ay-By)
                                        calc += Ndiff1(m1,a2)*elnuc(P, p, l1, l2, m1+1, m2, n1, n2, N1p, N2, c1, c2, Zc, C, Ex, Eyp, Ez)
                                        if m1 != 0:
                                            Eym = np.zeros(m1-1+m2+1)
                                            for t in range(0, m1-1+m2+1):
                                                Eym[t] = E(m1-1,m2,t,Ay-By,a2,b,P[1]-Ay,P[1]-By,Ay-By)
                                            calc += Ndiff2(m1,a2)*elnuc(P, p, l1, l2, m1-1, m2, n1, n2, N1m, N2, c1, c2, Zc, C, Ex, Eym, Ez)
        
                                if atomidx == basis[a[1]][6]:
                                    for atom in range(1, len(input)):
                                        Zc = input[atom][0]
                                        C = np.array([input[atom][1],input[atom][2],input[atom][3]])
                                        Eyp = np.zeros(m1+1+m2+1)
                                        for t in range(0, m1+1+m2+1):
                                            Eyp[t] = E(m1,m2+1,t,Ay-By,a2,b,P[1]-Ay,P[1]-By,Ay-By)
                                        calc += Ndiff1(m2,b)*elnuc(P, p, l1, l2, m1, m2+1, n1, n2, N1, N2p, c1, c2, Zc, C, Ex, Eyp, Ez)
                                        if m2 != 0:
                                            Eym = np.zeros(m1-1+m2+1)
                                            for t in range(0, m1+m2-1+1):
                                                Eym[t] = E(m1,m2-1,t,Ay-By,a2,b,P[1]-Ay,P[1]-By,Ay-By)
                                            calc += Ndiff2(m2,b)*elnuc(P, p, l1, l2, m1, m2-1, n1, n2, N1, N2m, c1, c2, Zc, C, Ex, Eym, Ez)
                        
                    Na[k,l] = calc
                    Na[l,k] = calc
        np.save('slowquant/temp/'+str(atomidx)+'dynucatt.npy',Na)
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
                            Ex = Edict[str(k)+str(l)+str(i)+str(j)+'E1']
                            Ey = Edict[str(k)+str(l)+str(i)+str(j)+'E2']
                            Ez = Edict[str(k)+str(l)+str(i)+str(j)+'E3']
                            P  = GPdict[str(k)+str(l)+str(i)+str(j)]
                            p  = pdict[str(k)+str(l)+str(i)+str(j)]
                            
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
                            
                            N1p=Nzplus[a[0]][5][i][0]
                            N2p=Nzplus[a[1]][5][j][0]
                            
                            N1m=Nzminus[a[0]][5][i][0]
                            N2m=Nzminus[a[1]][5][j][0]
                            
                            if atomidx == basis[a[0]][6] and atomidx == basis[a[1]][6]:
                                for atom in range(1, len(input)):
                                    Zc = input[atom][0]
                                    C = np.array([input[atom][1],input[atom][2],input[atom][3]])
                                    
                                    if atom != atomidx:
                                        Ezp = np.zeros(n1+1+n2+1)
                                        for t in range(0, n1+1+n2+1):
                                            Ezp[t] = E(n1+1,n2,t,Az-Bz,a2,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                        calc += Ndiff1(n1,a2)*elnuc(P, p, l1, l2, m1, m2, n1+1, n2, N1p, N2, c1, c2, Zc, C, Ex, Ey, Ezp)
                                        if n1 != 0:
                                            Ezm = np.zeros(n1-1+n2+1)
                                            for t in range(0, n1-1+n2+1):
                                                Ezm[t] = E(n1-1,n2,t,Az-Bz,a2,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                            calc += Ndiff2(n1,a2)*elnuc(P, p, l1, l2, m1, m2, n1-1, n2, N1m, N2, c1, c2, Zc, C, Ex, Ey, Ezm)
                                        
                                        Ezp = np.zeros(n1+1+n2+1)
                                        for t in range(0, n1+1+n2+1):
                                            Ezp[t] = E(n1,n2+1,t,Az-Bz,a2,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                        calc += Ndiff1(n2,b)*elnuc(P, p, l1, l2, m1, m2, n1, n2+1, N1, N2p, c1, c2, Zc, C, Ex, Ey, Ezp)
                                        if n2 != 0:
                                            Ezm = np.zeros(n1-1+n2+1)
                                            for t in range(0, n1+n2-1+1):
                                                Ezm[t] = E(n1,n2-1,t,Az-Bz,a2,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                            calc += Ndiff2(n2,b)*elnuc(P, p, l1, l2, m1, m2, n1, n2-1, N1, N2m, c1, c2, Zc, C, Ex, Ey, Ezm)
                            
                            else:
                                calc += electricfield(a2, b, Ax, Az, Az, Bx, Bz, Bz, l1, l2, m1, m2, n1, n2, N1, N2, c1, c2, input, 'z', atomidx)
                                
                                if atomidx == basis[a[0]][6]:
                                    for atom in range(1, len(input)):
                                        Zc = input[atom][0]
                                        C = np.array([input[atom][1],input[atom][2],input[atom][3]])
                                        Ezp = np.zeros(n1+1+n2+1)
                                        for t in range(0, n1+1+n2+1):
                                            Ezp[t] = E(n1+1,n2,t,Az-Bz,a2,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                        calc += Ndiff1(n1,a2)*elnuc(P, p, l1, l2, m1, m2, n1+1, n2, N1p, N2, c1, c2, Zc, C, Ex, Ey, Ezp)
                                        if n1 != 0:
                                            Ezm = np.zeros(n1-1+n2+1)
                                            for t in range(0, n1-1+n2+1):
                                                Ezm[t] = E(n1-1,n2,t,Az-Bz,a2,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                            calc += Ndiff2(n1,a2)*elnuc(P, p, l1, l2, m1, m2, n1-1, n2, N1m, N2, c1, c2, Zc, C, Ex, Ey, Ezm)
        
                                if atomidx == basis[a[1]][6]:
                                    for atom in range(1, len(input)):
                                        Zc = input[atom][0]
                                        C = np.array([input[atom][1],input[atom][2],input[atom][3]])
                                        Ezp = np.zeros(n1+1+n2+1)
                                        for t in range(0, n1+1+n2+1):
                                            Ezp[t] = E(n1,n2+1,t,Az-Bz,a2,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                        calc += Ndiff1(n2,b)*elnuc(P, p, l1, l2, m1, m2, n1, n2+1, N1, N2p, c1, c2, Zc, C, Ex, Ey, Ezp)
                                        if n2 != 0:
                                            Ezm = np.zeros(n1-1+n2+1)
                                            for t in range(0, n1+n2-1+1):
                                                Ezm[t] = E(n1,n2-1,t,Az-Bz,a2,b,P[2]-Az,P[2]-Bz,Az-Bz)
                                            calc += Ndiff2(n2,b)*elnuc(P, p, l1, l2, m1, m2, n1, n2-1, N1, N2m, c1, c2, Zc, C, Ex, Ey, Ezm)
        
                    Na[k,l] = calc
                    Na[l,k] = calc
        np.save('slowquant/temp/'+str(atomidx)+'dznucatt.npy',Na)
        print(time.time()-start, 'Nuc att diff z atom: '+str(atomidx))
        #END OF nucleus electron attraction z diff