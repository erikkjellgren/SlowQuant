import pyscf
from pyscf import fci, mcscf, scf

mol = pyscf.M(atom="O 0 0 0; H 0.0  0.0  0.9697", basis="6-31G", unit="angstrom", spin=1)
mol.build()
mf = scf.UHF(mol)
mf.kernel()


mc = mcscf.UCASCI(mf, 3, (2, 1))
res = mc.kernel(mf.mo_coeff)


ci_coeff = mc.ci  # CI coefficients
norb = mc.ncas  # Number of active orbitals
nelec = mc.nelecas  # Number of active electrons

# Compute the 1,2-RDMs
rdms = fci.direct_uhf.make_rdm12s(ci_coeff, norb, nelec)
