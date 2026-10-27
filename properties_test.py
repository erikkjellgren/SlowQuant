import numpy as np
import pyscf
from pyscf import mcscf, scf, gto, x2c
from scipy.stats import unitary_group
from scipy.linalg import solve
from pyscf.x2c import sfx2c1e
from pyscf import cc
import scipy.linalg
from scipy.linalg import expm
from pathlib import Path
import os

from slowquant.unitary_coupled_cluster.generalized_ups_wavefunction import GeneralizedWaveFunctionUPS
# from slowquant.unitary_coupled_cluster.generalized_ucc_wavefunction import GeneralizedWaveFunctionUCC
from slowquant.unitary_coupled_cluster.generalized_properties import GeneralizedProperties

def NR(geometry, basis, active_space, unit="bohr", charge=0, spin=0, c=137.036):
    """.........."""
    print("active space:", {active_space})
    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin, nucmod=1)
    # mol = pyscf.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)

    mol.build()

    # mf = scf.GHF(mol).sfx2c1e() #spinfree
    # mf = scf.GHF(mol).x2c1e()
    mf = scf.GHF(mol).x2c()
    # mf = scf.GHF(mol)


    mf.conv_tol = 1e-10 #energy tolerance from PYSCF
    mf.conv_tol_grad = 1e-8 #gradient tolerance form PYSCF

    mf.max_cycle = 50000
    
    mf.kernel()
    print("PySCF SCF energy:", mf.e_tot)
    print("PySCF electronic:", mf.e_tot - mf.energy_nuc())
    coeff=np.array(mf.mo_coeff, dtype=complex)
    e_nuc=mf.energy_nuc()



    # # load data
    # data = np.load("data_HBr_UCCSD_00.npz")
    # c_mo, thetas_real, thetas_imag = data["c_mo"], data["thetas_real"], data["thetas_imag"]
    # coeff=np.array(c_mo, dtype=complex)


    WF =GeneralizedWaveFunctionUPS(
        active_space,
        coeff,
        mol,
        "fuccsd",
        True, #Do x2c "sf" for spinfree
        {"n_layers": 1, "is_spin_conserving" : False},
        include_active_kappa=False,
    )

    ny_theta_real = np.random.uniform(-0.05, 0.05, len(WF.thetas))
    # ny_theta_imag = np.random.uniform(-0.05, 0.05, len(WF.thetas))

    ny_theta_imag = [0.0] * len(WF.thetas)

    WF.set_thetas(ny_theta_real, ny_theta_imag)

    WF.run_wf_optimization_2step("l-bfgs-b", orbital_optimization=False, tol=1e-16, maxiter = 20000)
    
    print("E_opt: (+nuc!)", WF._energy_elec + e_nuc)

    props = GeneralizedProperties(WF, mol, mf, x2c=True, picture_change=True, spin_free=False, c=c)

    "Electric field gradients"
    efg = props.electric_field_gradient()
    # exi=props.excitation_energies()
    # os=props.oscillator_strengths()
    # dip_mol=props.dipole_moment()

    print(efg)

from pyscf.gto.basis import load
import pyscf.gto as gto

#MOLEKYLE

def HF():
    geometry = """F  0.0   0.0  0.0;
        H  0.0  0.0  0.91680 """
    basis = {'H': gto.uncontract(load('x2c-SV(P)all.nw', 'H')),
                'F': gto.uncontract(load('x2c-SV(P)all.nw', 'F'))}
    # basis = "x2c-SVPall.nw"
    active_space = ((3,3), 8) #spin orbitaler or spinor basis
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

# HF()

def h2():
    geometry = """H  0.0   0.0  0.0;
        H  0.0  0.0  0.74"""
    basis = "STO-3g"
    active_space = ((1, 1), 4) #spin orbitaler or spinor basis
    charge = 0
    spin = 0

    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

# h2()

def HBr():
    geometry = """H  0.0   0.0  1.41443;
        Br  0.0  0.0  0.0 """
    basis = {'H': gto.uncontract(load('x2c-SV(P)all.nw', 'H')),
                'Br': gto.uncontract(load('x2c-SV(P)all.nw', 'Br'))}
    active_space = ((3,3), 8) #spin orbitaler or spinor basis
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )

# HBr()

def HRb():
    geometry = """H  0.0   0.0  2.367;
        Rb  0.0  0.0  0.0 """
    basis = {'H': gto.uncontract(load('x2c-SV(P)all.nw', 'H')),
                'Rb': gto.uncontract(load('x2c-SV(P)all.nw', 'Rb'))}
    active_space = ((1,1), 4) #spin orbitaler or spinor basis
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
# HRb()

def HCl():
    geometry = """H  0.0   0.0  1.27455;
        Cl  0.0  0.0  0.0 """
    basis='dyallv2z'
    # basis = {'H':'sto-3g','Cl': 'x2c-SVPall.nw'}
    # basis = {'H': gto.uncontract(load('x2c-SVPall_2c.nw', 'H')),
    #             'Cl': gto.uncontract(load('x2c-SVPall_2c.nw', 'Cl'))}
    active_space = ((3,3), 8) #spin orbitaler or spinor basis
    charge = 0
    spin = 0
    NR(
        geometry=geometry, basis=basis, active_space=active_space, charge=charge, spin=spin, unit="angstrom"
    )
HCl()