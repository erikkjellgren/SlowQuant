import numpy as np
import math
import scipy.misc as scm

def N(a, l, m, n):
    part1 = (2.0/math.pi)**(3.0/4.0)
    part2 = 2.0**(l+m+n) * a**((2.0*l+2.0*m+2.0*n+3.0)/(4.0))
    part3 = math.sqrt(scm.factorial2(int(2*l-1))*scm.factorial2(int(2*m-1))*scm.factorial2(int(2*n-1)))
    N = part1 * ((part2)/(part3))
    return N

def bassiset(input, set):
    basis_out = []
    idx = 1
    for i in range(1, len(input)):
        if set['basisset'] == 'STO3G':
            if input[i][0] == 1:
                basis_func = [[0,3.42525091,0.15432897,0,0,0],
                                [0,0.62391373,0.53532814,0,0,0],
                                [0,0.16885540,0.44463454,0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],3,basis_func, i])
                idx += 1
            elif input[i][0] == 6:
                basis_func = [[0, 71.6168370,0.15432897,0,0,0],
                                [0,13.0450960 ,0.53532814,0,0,0],
                                [0,3.5305122,0.44463454,0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],3,basis_func, i])
                idx += 1
                basis_func = [[0,2.9412494,-0.09996723,0,0,0],
                                [0,0.6834831,0.39951283,0,0,0],
                                [0,0.2222899,0.70011547,0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],3,basis_func, i])
                idx += 1
                basis_func = [[0,2.9412494,0.15591627,1,0,0],
                                [0,0.6834831,0.60768372,1,0,0],
                                [0,0.2222899,0.39195739,1,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],3,basis_func, i])
                idx += 1
                basis_func = [[0,2.9412494,0.15591627,0,1,0],
                                [0,0.6834831,0.60768372,0,1,0],
                                [0,0.2222899,0.39195739,0,1,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],3,basis_func, i])
                idx += 1
                basis_func = [[0,2.9412494,0.15591627,0,0,1],
                                [0,0.6834831,0.60768372,0,0,1],
                                [0,0.2222899,0.39195739,0,0,1]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],3,basis_func, i])
                idx += 1
            elif input[i][0] == 7:
                basis_func = [[0, 99.1061690,0.15432897,0,0,0],
                                [0,18.0523120 ,0.53532814,0,0,0],
                                [0,4.8856602,0.44463454,0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],3,basis_func, i])
                idx += 1
                basis_func = [[0,3.7804559,-0.09996723,0,0,0],
                                [0,0.8784966,0.39951283,0,0,0],
                                [0,0.2857144,0.70011547,0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],3,basis_func, i])
                idx += 1
                basis_func = [[0,3.7804559,0.15591627,1,0,0],
                                [0,0.8784966,0.60768372,1,0,0],
                                [0,0.2857144,0.39195739,1,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],3,basis_func, i])
                idx += 1
                basis_func = [[0,3.7804559,0.15591627,0,1,0],
                                [0,0.8784966,0.60768372,0,1,0],
                                [0,0.2857144,0.39195739,0,1,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],3,basis_func, i])
                idx += 1
                basis_func = [[0,3.7804559,0.15591627,0,0,1],
                                [0,0.8784966,0.60768372,0,0,1],
                                [0,0.2857144,0.39195739,0,0,1]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],3,basis_func, i])
                idx += 1
            elif input[i][0] == 8:
                basis_func = [[0,130.7093200,0.15432897,0,0,0],
                                [0,23.8088610,0.53532814,0,0,0],
                                [0,6.4436083,0.44463454,0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],3,basis_func, i])
                idx += 1
                basis_func = [[0,5.0331513,-0.09996723,0,0,0],
                                [0,1.1695961,0.39951283,0,0,0],
                                [0,0.3803890,0.70011547,0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],3,basis_func, i])
                idx += 1
                basis_func = [[0,5.0331513,0.15591627,1,0,0],
                                [0,1.1695961,0.60768372,1,0,0],
                                [0,0.3803890,0.39195739,1,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],3,basis_func, i])
                idx += 1
                basis_func = [[0,5.0331513,0.15591627,0,1,0],
                                [0,1.1695961,0.60768372,0,1,0],
                                [0,0.3803890,0.39195739,0,1,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],3,basis_func, i])
                idx += 1
                basis_func = [[0,5.0331513,0.15591627,0,0,1],
                                [0,1.1695961,0.60768372,0,0,1],
                                [0,0.3803890,0.39195739,0,0,1]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],3,basis_func, i])
                idx += 1
        
        elif set['basisset'] == 'DZ':
            if input[i][0] == 1:
                basis_func = [[0,19.2406000,0.0328280,0,0,0],
                                [0,2.8992000,0.2312080,0,0,0],
                                [0,0.6534000,0.8172380,0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],3,basis_func, i])
                idx += 1
                basis_func = [[0, 0.1776000, 1.0000000, 0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
            elif input[i][0] == 8:
                basis_func = [[0,7816.5400000,0.0020310,0,0,0],
                                [0,1175.8200000,0.0154360,0,0,0],
                                [0,273.1880000,0.0737710,0,0,0],
                                [0,81.1696000,0.2476060,0,0,0],
                                [0,27.1836000,0.6118320,0,0,0],
                                [0,3.4136000,0.2412050,0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],6,basis_func, i])
                idx += 1
                basis_func = [[0, 9.5322000, 1.0000000, 0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0, 0.9398000, 1.0000000, 0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0, 0.2846000, 1.0000000, 0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0,35.1832000,0.0195800,1,0,0],
                                [0,7.9040000,0.1241890,1,0,0],
                                [0,2.3051000,0.3947270,1,0,0],
                                [0,0.7171000,0.6273750,1,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],4,basis_func, i])
                idx += 1
                basis_func = [[0,35.1832000,0.0195800,0,1,0],
                                [0,7.9040000,0.1241890,0,1,0],
                                [0,2.3051000,0.3947270,0,1,0],
                                [0,0.7171000,0.6273750,0,1,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],4,basis_func, i])
                idx += 1
                basis_func = [[0,35.1832000,0.0195800,0,0,1],
                                [0,7.9040000,0.1241890,0,0,1],
                                [0,2.3051000,0.3947270,0,0,1],
                                [0,0.7171000,0.6273750,0,0,1]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],4,basis_func, i])
                idx += 1
                basis_func = [[0, 0.2137000, 1.0000000, 1,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0, 0.2137000, 1.0000000, 0,1,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0, 0.2137000, 1.0000000, 0,0,1]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                
        elif set['basisset'] == 'DZP':
            if input[i][0] == 1:
                basis_func = [[0,19.2406000,0.0328280,0,0,0],
                                [0,2.8992000,0.2312080,0,0,0],
                                [0,0.6534000,0.8172380,0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],3,basis_func, i])
                idx += 1
                basis_func = [[0, 0.1776000, 1.0000000, 0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0, 1.0000000, 1.0000000, 1,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0, 1.0000000, 1.0000000, 0,1,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0, 1.0000000, 1.0000000, 0,0,1]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
            elif input[i][0] == 8:
                basis_func = [[0,7816.5400000,0.0020310,0,0,0],
                                [0,1175.8200000,0.0154360,0,0,0],
                                [0,273.1880000,0.0737710,0,0,0],
                                [0,81.1696000,0.2476060,0,0,0],
                                [0,27.1836000,0.6118320,0,0,0],
                                [0,3.4136000,0.2412050,0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],6,basis_func, i])
                idx += 1
                basis_func = [[0, 9.5322000, 1.0000000, 0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0, 0.9398000, 1.0000000, 0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0, 0.2846000, 1.0000000, 0,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0,35.1832000,0.0195800,1,0,0],
                                [0,7.9040000,0.1241890,1,0,0],
                                [0,2.3051000,0.3947270,1,0,0],
                                [0,0.7171000,0.6273750,1,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],4,basis_func, i])
                idx += 1
                basis_func = [[0,35.1832000,0.0195800,0,1,0],
                                [0,7.9040000,0.1241890,0,1,0],
                                [0,2.3051000,0.3947270,0,1,0],
                                [0,0.7171000,0.6273750,0,1,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],4,basis_func, i])
                idx += 1
                basis_func = [[0,35.1832000,0.0195800,0,0,1],
                                [0,7.9040000,0.1241890,0,0,1],
                                [0,2.3051000,0.3947270,0,0,1],
                                [0,0.7171000,0.6273750,0,0,1]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],4,basis_func, i])
                idx += 1
                basis_func = [[0, 0.2137000, 1.0000000, 1,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0, 0.2137000, 1.0000000, 0,1,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0, 0.2137000, 1.0000000, 0,0,1]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0, 0.8500000, 1.0000000, 2,0,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0, 0.8500000, 1.0000000, 0,2,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0, 0.8500000, 1.0000000, 0,0,2]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0, 0.8500000, 1.0000000, 1,1,0]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0, 0.8500000, 1.0000000, 1,0,1]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1
                basis_func = [[0, 0.8500000, 1.0000000, 0,1,1]]
                basis_out.append([idx,input[i][1],input[i][2],input[i][3],1,basis_func, i])
                idx += 1

    for i in range(len(basis_out)):
        for j in range(len(basis_out[i][5])):
            basis_out[i][5][j][0] = N(basis_out[i][5][j][1], basis_out[i][5][j][3], basis_out[i][5][j][4], basis_out[i][5][j][5])
    
    return basis_out
