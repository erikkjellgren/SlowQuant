import cython
import numpy as np
cimport numpy as np

cdef double doublebar(int i, int j, int k, int l, double [:,:,:,:] Vee):
    return Vee[i,j,k,l] - Vee[i,j,l,k]
    
cdef double tautilde(int i, int j, int a, int b, double [:,:,:,:] tdouble, double [:,:] tsingle):
    return tdouble[i,j,a,b] + 0.5*(tsingle[i,a]*tsingle[j,b]-tsingle[i,b]*tsingle[j,a])

cdef double tau(int i, int j, int a, int b, double [:,:,:,:] tdouble, double [:,:] tsingle):
    return tdouble[i,j,a,b] + tsingle[i,a]*tsingle[j,b]-tsingle[i,b]*tsingle[j,a]

cdef double ttripledis(int a, int b, int c, int i, int j, int k, double [:,:] tsingle, double [:,:,:,:] Vee):
    #Used in CCSD(T)
    return tsingle[i,a]*doublebar(j,k,b,c,Vee)

cdef double ttriplecon1(int a, int b, int c, int i, int j, int k, int occ, int dim, double [:,:,:,:] tdouble, double [:,:,:,:] Vee):
    cdef int e
    cdef double sum
    # Sum over virtuel, used in CCSD(T)
    sum = 0.0
    for e in range(occ, dim):
        sum += tdouble[j,k,a,e]*doublebar(e,i,b,c,Vee)
    return sum

cdef double ttriplecon2(int a, int b, int c, int i, int j, int k, int occ, double [:,:,:,:] tdouble, double [:,:,:,:] Vee):
    cdef int m
    cdef double sum
    # Sum over occupied, used in CCSD(T)
    sum = 0.0
    for m in range(0, occ):
        sum -= tdouble[i,m,b,c]*doublebar(m,a,j,k,Vee)
    return sum

cpdef runCythonCCSD(double [:,:] FMOspin, double [:,:,:,:] VeeMOspin, double [:,:] tsingle, double [:,:,:,:] tdouble, int dimension, int occ):
    cdef int a, e, m, n, i, j, f, b
    cdef double ECCSD
    cdef double [:,:] Fae, Fmi, Fme, tsingle_new, tsingle_old
    cdef double [:,:,:,:] Wmnij, Wabef, tdouble_new, tdouble_old
    # Build F and W
    Fae = np.zeros((dimension,dimension))
    for a in range(occ, dimension):
        for e in range(occ, dimension):
            if a != e:
                Fae[a,e] += FMOspin[a,e]
            for m in range(0, occ):
                Fae[a,e] -= 0.5*FMOspin[m,e]*tsingle[m,a]
                for f in range(occ, dimension):
                    Fae[a,e] += doublebar(m,a,f,e,VeeMOspin)*tsingle[m,f]
                    for n in range(0, occ):
                        Fae[a,e] -= 0.5*tautilde(m,n,a,f,tdouble,tsingle)*doublebar(m,n,e,f,VeeMOspin)

    Fmi = np.zeros((dimension,dimension))
    for m in range(0, occ):
        for i in range(0, occ):
            if m != i:
                Fmi[m,i] += FMOspin[m,i]
            for e in range(occ, dimension):
                Fmi[m,i] += 0.5*FMOspin[m,e]*tsingle[i,e]
                for n in range(0, occ):
                    Fmi[m,i] += doublebar(m,n,i,e,VeeMOspin)*tsingle[n,e]
                    for f in range(occ, dimension):
                        Fmi[m,i] += 0.5*doublebar(m,n,e,f,VeeMOspin)*tautilde(i,n,e,f,tdouble,tsingle)
    
    Fme = np.zeros((dimension,dimension))
    for m in range(0, occ):
        for e in range(occ, dimension):
            Fme[m,e] += FMOspin[m,e]
            for n in range(0, occ):
                for f in range(occ, dimension):
                    Fme[m,e] += doublebar(m,n,e,f,VeeMOspin)*tsingle[n,f]
    
    Wmnij = np.zeros((dimension,dimension,dimension,dimension))
    for m in range(0, occ):
        for n in range(0, occ):
            for i in range(0, occ):
                for j in range(0, occ):
                    Wmnij[m,n,i,j] += doublebar(m,n,i,j,VeeMOspin)
                    for e in range(occ, dimension):
                        Wmnij[m,n,i,j] += doublebar(m,n,i,e,VeeMOspin)*tsingle[j,e] - doublebar(m,n,j,e,VeeMOspin)*tsingle[i,e]
                        for f in range(occ, dimension):
                            Wmnij[m,n,i,j] += 0.25*doublebar(m,n,e,f,VeeMOspin)*tau(i,j,e,f,tdouble,tsingle)

    Wabef = np.zeros((dimension,dimension,dimension,dimension))
    for a in range(occ, dimension):
        for b in range(occ, dimension):
            for e in range(occ, dimension):
                for f in range(occ, dimension):
                    Wabef[a,b,e,f] += doublebar(a,b,e,f,VeeMOspin)
                    for m in range(0, occ):
                        Wabef[a,b,e,f] -= doublebar(a,m,e,f,VeeMOspin)*tsingle[m,b]
                        Wabef[a,b,e,f] += doublebar(b,m,e,f,VeeMOspin)*tsingle[m,a]
                        for n in range(0, occ):
                            Wabef[a,b,e,f] += 0.25*doublebar(m,n,e,f,VeeMOspin)*tau(m,n,a,b,tdouble,tsingle)
                            
            
    Wmbej = np.zeros((dimension,dimension,dimension,dimension))
    for m in range(0, occ):
        for b in range(occ, dimension):
            for e in range(occ, dimension):
                for j in range(0, occ):
                    Wmbej[m,b,e,j] += doublebar(m,b,e,j,VeeMOspin)
                    for f in range(occ, dimension):
                        Wmbej[m,b,e,j] += doublebar(m,b,e,f,VeeMOspin)*tsingle[j,f]
                    for n in range(0, occ):
                        Wmbej[m,b,e,j] -= doublebar(m,n,e,j,VeeMOspin)*tsingle[n,b]
                        for f in range(occ, dimension):
                            Wmbej[m,b,e,j] -= doublebar(m,n,e,f,VeeMOspin) * (0.5*tdouble[j,n,f,b]+tsingle[j,f]*tsingle[n,b])

    # Construct new T1 and new T2
    tsingle_new = np.zeros((dimension,dimension))    
    for i in range(0, occ):
        for a in range(occ, dimension):
            tsingle_new[i,a] += FMOspin[i,a]
            for e in range(occ, dimension):
                tsingle_new[i,a] += tsingle[i,e]*Fae[a,e]
            for m in range(0, occ):
                tsingle_new[i,a] -= tsingle[m,a]*Fmi[m,i]
                for e in range(occ, dimension):
                    tsingle_new[i,a] += tdouble[i,m,a,e]*Fme[m,e]
                    for f in range(occ, dimension):
                        tsingle_new[i,a] -= 0.5*tdouble[i,m,e,f]*doublebar(m,a,e,f,VeeMOspin)
                    for n in range(0, occ):
                        tsingle_new[i,a] -= 0.5*tdouble[m,n,a,e]*doublebar(n,m,e,i,VeeMOspin)
            for n in range(0, occ):
                for f in range(occ, dimension):
                    tsingle_new[i,a] -= tsingle[n,f]*doublebar(n,a,i,f,VeeMOspin)
            tsingle_new[i,a] = tsingle_new[i,a]/(FMOspin[i,i]-FMOspin[a,a])
    
    tsingle_old = tsingle
    tsingle = tsingle_new
    tdouble_new = np.zeros((dimension,dimension,dimension,dimension))
    for i in range(0, occ):
        for j in range(0, occ):
            for a in range(occ, dimension):
                for b in range(occ, dimension):
                    tdouble_new[i,j,a,b] += doublebar(i,j,a,b,VeeMOspin)
                    
                    for e in range(occ, dimension):
                        tdouble_new[i,j,a,b] += tdouble[i,j,a,e]*Fae[b,e]
                        tdouble_new[i,j,a,b] -= tdouble[i,j,b,e]*Fae[a,e]
                        for m in range(0, occ):
                            tdouble_new[i,j,a,b] += -0.5*tdouble[i,j,a,e]*tsingle[m,b]*Fme[m,e]
                            tdouble_new[i,j,a,b] -= -0.5*tdouble[i,j,b,e]*tsingle[m,a]*Fme[m,e]
                    
                        for f in range(occ, dimension):
                            tdouble_new[i,j,a,b] += 0.5*Wabef[a,b,e,f]*tau(i,j,e,f,tdouble,tsingle)
                    
                    for m in range(0, occ):
                        tdouble_new[i,j,a,b] -= tdouble[i,m,a,b]*Fmi[m,j]
                        tdouble_new[i,j,a,b] += tdouble[j,m,a,b]*Fmi[m,i]
                        for e in range(occ, dimension):
                            tdouble_new[i,j,a,b] -= 0.5*tdouble[i,m,a,b]*tsingle[j,e]*Fme[m,e]
                            tdouble_new[i,j,a,b] += 0.5*tdouble[j,m,a,b]*tsingle[i,e]*Fme[m,e]
                    
                        for n in range(0, occ):
                            tdouble_new[i,j,a,b] += 0.5*tau(m,n,a,b,tdouble,tsingle)*Wmnij[m,n,i,j]

                    for m in range(0, occ):
                        for e in range(occ, dimension):
                            tdouble_new[i,j,a,b] += Wmbej[m,b,e,j]*tdouble[i,m,a,e] - doublebar(m,b,e,j,VeeMOspin)*tsingle[i,e]*tsingle[m,a]
                            tdouble_new[i,j,a,b] -= Wmbej[m,b,e,i]*tdouble[j,m,a,e] - doublebar(m,b,e,i,VeeMOspin)*tsingle[j,e]*tsingle[m,a]
                            tdouble_new[i,j,a,b] -= Wmbej[m,a,e,j]*tdouble[i,m,b,e] - doublebar(m,a,e,j,VeeMOspin)*tsingle[i,e]*tsingle[m,b]
                            tdouble_new[i,j,a,b] += Wmbej[m,a,e,i]*tdouble[j,m,b,e] - doublebar(m,a,e,i,VeeMOspin)*tsingle[j,e]*tsingle[m,b]

                    for e in range(occ, dimension): 
                        tdouble_new[i,j,a,b] += doublebar(a,b,e,j,VeeMOspin)*tsingle[i,e]
                        tdouble_new[i,j,a,b] -= doublebar(a,b,e,i,VeeMOspin)*tsingle[j,e]
                    
                    for m in range(0, occ):
                        tdouble_new[i,j,a,b] -= tsingle[m,a]*doublebar(m,b,i,j,VeeMOspin)
                        tdouble_new[i,j,a,b] += tsingle[m,b]*doublebar(m,a,i,j,VeeMOspin)

                    tdouble_new[i,j,a,b] = tdouble_new[i,j,a,b]/(FMOspin[i,i]+FMOspin[j,j]-FMOspin[a,a]-FMOspin[b,b])

    tdouble_old = tdouble
    tdouble = tdouble_new
    
    # Calculate ECCSD
    ECCSD = 0.0
    for i in range(0, occ):
        for a in range(occ, dimension):
            ECCSD += FMOspin[i,a]*tsingle[i,a]
            for j in range(0, occ):
                for b in range(occ, dimension):
                    ECCSD += 0.25*doublebar(i,j,a,b,VeeMOspin)*tdouble[i,j,a,b]+0.5*doublebar(i,j,a,b,VeeMOspin)*tsingle[i,a]*tsingle[j,b]
    
    return ECCSD, tsingle, tdouble, tsingle_old, tdouble_old


cpdef double runCythonPerturbativeT(double [:,:] FMOspin, double [:,:,:,:] VeeMOspin, double [:,:] tsingle, double [:,:,:,:] tdouble, int dimension, int occ):
    cdef int i,j,k,a,b,c
    cdef double ET, ttripledisconnected, ttripleconnected, Dijkabc
    ET = 0.0
    for i in range(0, occ):
        for j in range(0, occ):
            for k in range(0, occ):
                for a in range(occ, dimension):
                    for b in range(occ, dimension):
                        for c in range(occ, dimension):
                            ttripledisconnected = 0.0
                            ttripleconnected = 0.0
                            Dijkabc = FMOspin[i,i]+FMOspin[j,j]+FMOspin[k,k]-FMOspin[a,a]-FMOspin[b,b]-FMOspin[c,c]
                            
                            # Construct disconnected triples
                            ttripledisconnected += ttripledis(a,b,c,i,j,k,tsingle,VeeMOspin)
                            ttripledisconnected -= ttripledis(a,b,c,j,i,k,tsingle,VeeMOspin)
                            ttripledisconnected -= ttripledis(a,b,c,k,j,i,tsingle,VeeMOspin)
                            
                            ttripledisconnected -= ttripledis(b,a,c,i,j,k,tsingle,VeeMOspin)
                            ttripledisconnected += ttripledis(b,a,c,j,i,k,tsingle,VeeMOspin)
                            ttripledisconnected += ttripledis(b,a,c,k,j,i,tsingle,VeeMOspin)
                            
                            ttripledisconnected -= ttripledis(c,b,a,i,j,k,tsingle,VeeMOspin)
                            ttripledisconnected += ttripledis(c,b,a,j,i,k,tsingle,VeeMOspin)
                            ttripledisconnected += ttripledis(c,b,a,k,j,i,tsingle,VeeMOspin)
                            
                            ttripledisconnected = ttripledisconnected/Dijkabc
                            
                            # Construct connected triples
                            ttripleconnected += ttriplecon1(a,b,c,i,j,k,occ,dimension,tdouble,VeeMOspin)
                            ttripleconnected -= ttriplecon1(a,b,c,j,i,k,occ,dimension,tdouble,VeeMOspin)
                            ttripleconnected -= ttriplecon1(a,b,c,k,j,i,occ,dimension,tdouble,VeeMOspin)
                            
                            ttripleconnected -= ttriplecon1(b,a,c,i,j,k,occ,dimension,tdouble,VeeMOspin)
                            ttripleconnected += ttriplecon1(b,a,c,j,i,k,occ,dimension,tdouble,VeeMOspin)
                            ttripleconnected += ttriplecon1(b,a,c,k,j,i,occ,dimension,tdouble,VeeMOspin)
                            
                            ttripleconnected -= ttriplecon1(c,b,a,i,j,k,occ,dimension,tdouble,VeeMOspin)
                            ttripleconnected += ttriplecon1(c,b,a,j,i,k,occ,dimension,tdouble,VeeMOspin)
                            ttripleconnected += ttriplecon1(c,b,a,k,j,i,occ,dimension,tdouble,VeeMOspin)

                            ttripleconnected += ttriplecon2(a,b,c,i,j,k,occ,tdouble,VeeMOspin)
                            ttripleconnected -= ttriplecon2(a,b,c,j,i,k,occ,tdouble,VeeMOspin)
                            ttripleconnected -= ttriplecon2(a,b,c,k,j,i,occ,tdouble,VeeMOspin)
                            
                            ttripleconnected -= ttriplecon2(b,a,c,i,j,k,occ,tdouble,VeeMOspin)
                            ttripleconnected += ttriplecon2(b,a,c,j,i,k,occ,tdouble,VeeMOspin)
                            ttripleconnected += ttriplecon2(b,a,c,k,j,i,occ,tdouble,VeeMOspin)
                            
                            ttripleconnected -= ttriplecon2(c,b,a,i,j,k,occ,tdouble,VeeMOspin)
                            ttripleconnected += ttriplecon2(c,b,a,j,i,k,occ,tdouble,VeeMOspin)
                            ttripleconnected += ttriplecon2(c,b,a,k,j,i,occ,tdouble,VeeMOspin)
                            
                            ttripleconnected = ttripleconnected/Dijkabc
                            
                            ET += ttripleconnected*Dijkabc*(ttripleconnected+ttripledisconnected)
    
    ET = ET/36.0
    return ET