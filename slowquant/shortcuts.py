import slowquant.hartreefock.HartreeFock as HF
import slowquant.basissets.BasisSet as BS
import slowquant.molecularintegrals.runMolecularIntegrals as MI
import numpy as np

def HartreeFock(input, VNN, Te, S, VeN, Vee, deTHR=10**-6,rmsTHR=10**-6,Maxiter=100, DO_DIIS='Yes', DIIS_steps=6, print_SCF='Yes'):
    return HF.HartreeFock(input, VNN, Te, S, VeN, Vee, deTHR=10**-6,rmsTHR=10**-6,Maxiter=100, DO_DIIS='Yes', DIIS_steps=6, print_SCF='Yes')

def Integrals(input, set, pathtobasis):
    basis = BS.bassiset(input, set, path=pathtobasis)
    results = {}
    results = MI.runIntegrals(input, basis, set, results)
    return results['VNN'], results['VNe'], results['S'], results['Te'], results['Vee']