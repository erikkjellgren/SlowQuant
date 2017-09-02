import slowquant.hartreefock.HartreeFock as HF
import slowquant.hartreefock.UHF as UHF
import slowquant.basissets.BasisSet as BS
import slowquant.molecularintegrals.runMolecularIntegrals as MI
import slowquant.integraltransformation.IntegralTransform as IT
import numpy as np

def HartreeFock(input, VNN, Te, S, VeN, Vee, deTHR=10**-6,rmsTHR=10**-6,Maxiter=100, DO_DIIS='Yes', DIIS_steps=6, print_SCF='Yes'):
    return HF.HartreeFock(input, VNN, Te, S, VeN, Vee, deTHR=10**-6,rmsTHR=10**-6,Maxiter=100, DO_DIIS='Yes', DIIS_steps=6, print_SCF='Yes')

def UnrestrictedHartreeFock(input, VNN, Te, S, VeN, Vee, deTHR=10**-6,rmsTHR=10**-6,Maxiter=100, UHF_mix=0.15, print_SCF='Yes'):
    return UHF.UnrestrictedHartreeFock(input, VNN, Te, S, VeN, Vee, deTHR=10**-6,rmsTHR=10**-6,Maxiter=100, UHF_mix=0.15, print_SCF='Yes')

def Integrals(input, set, pathtobasis):
    basis = BS.bassiset(input, set, path=pathtobasis)
    results = {}
    results = MI.runIntegrals(input, basis, set, results)
    return results['VNN'], results['VNe'], results['S'], results['Te'], results['Vee']

def basisset(input, set, pathtobasis):
    return BS.bassiset(input, set, path=pathtobasis)

def Transform1eMO(S, C):
    return IT.Transform1eMO(S, C)

def Transform2eMO(Vee, C):
    return IT.Transform2eMO(C, Vee)

def Transform1eSPIN(S):
    return IT.Transform1eSPIN(S)
    
def Transform2eSPIN(Vee):
    return IT.Transform2eSPIN(Vee)

def dipoleintegral(input, set, pathtobasis):
    basis = BS.bassiset(input, set, path=pathtobasis)
    results = {}
    results = MI.run_dipole_int(basis, input, results)
    return results['mu_x'], results['mu_y'], results['mu_z']