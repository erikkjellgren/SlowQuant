import slowquant.hartreefock.HartreeFock as HF
import slowquant.hartreefock.UHF as UHF
import slowquant.basissets.BasisSet as BS
import slowquant.molecularintegrals.runMolecularIntegrals as MI
import slowquant.integraltransformation.IntegralTransform as IT
import slowquant.configurationinteraction.CI as CI
import slowquant.properties as prop
import slowquant.mollerplesset.MPn as MP
import slowquant.coupledcluster.PythonCC as CC
import slowquant.bomd.BOMD as BO
import numpy as np

def HartreeFock(molecule, VNN, Te, S, VeN, Vee, dethr=10**-6, rmsthr=10**-6, maxiter=100, do_diis='Yes', diis_steps=6, print_scf='Yes'):
    return HF.HartreeFock(molecule, VNN, Te, S, VeN, Vee, deTHR=dethr, rmsTHR=rmsthr, Maxiter=maxiter, DO_DIIS=do_diis, DIIS_steps=diis_steps, print_SCF=print_scf')

def UnrestrictedHartreeFock(molecule, VNN, Te, S, VeN, Vee, dethr=10**-6, rmsthr=10**-6, maxiter=100, uhf_mix=0.15, print_scf='Yes'):
    return UHF.UnrestrictedHartreeFock(molecule, VNN, Te, S, VeN, Vee, deTHR=dethr,rmsTHR=rmsthr, Maxiter=maxiter, UHF_mix=uhf_mix, print_SCF=print_scf)

def Integrals(molecule, basisset):
    basis = BS.bassiset(molecule, basisset)
    results = {}
    results = MI.runIntegrals(molecule, basis, set, results)
    return results['VNN'], results['VNe'], results['S'], results['Te'], results['Vee']

def Transform1eMO(S, C):
    return IT.Transform1eMO(S, C)

def Transform2eMO(Vee, C):
    return IT.Transform2eMO(C, Vee)

def Transform1eSPIN(S):
    return IT.Transform1eSPIN(S)
    
def Transform2eSPIN(Vee):
    return IT.Transform2eSPIN(Vee)

def dipoleintegral(molecule, basisset):
    basis = BS.bassiset(molecule, basisset)
    results = {}
    results = MI.run_dipole_int(basis, molecule, results)
    return results['mu_x'], results['mu_y'], results['mu_z']
    
def CIS(occ, F, C, VeeMOspin):
    return CI.CIS(occ, F, C, VeeMOspin)
    
def Mulliken_charges(basisset, molecule, D, S):
    basis = BS.bassiset(molecule, basisset)
    return prop.MulCharge(basis, molecule, D, S)
    
def Lowdin_charges(basisset, molecule, D, S):
    basis = BS.bassiset(molecule, basisset)
    return prop.LowdinCharge(basis, molecule, D, S)   
    
def dipolemoment(molecule, D, mux_integral, muy_integral, muz_integral):
    return prop.dipolemoment(molecule, D, mux_integral, muy_integral, muz_integral)
    
def RPA(occ, F, C, VeeMOspin):
    return prop.RPA(occ, F, C, VeeMOspin)
    
def MP2(occ, F, C, VeeMO):
    return MP.MP2(occ, F, C, VeeMO)
    
def MP3(occ, F, C, VeeMOspin):
    return MP.MP3(occ, F, C, VeeMOspin)
    
def DCPT2(occ, F, C, VeeMO):
    return MP.DCPT2(occ, F, C, VeeMO)
    
def CCSD(occ, F, C, VeeMOspin, Maxiter=100, dethr=1e-10, rmsthr=1e-10, runccsdt=0):
    return CC.CCSD(occ, F, C, VeeMOspin, maxiter=Maxiter, deTHR=dethr, rmsTHR=rmsthr, runCCSDT=runccsdt)
