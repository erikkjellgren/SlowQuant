

Hartree-Fock methods
====================

In this section the Hartree-Fock methods will be described.

Restricted Hartree-Fock
-----------------------

For the Roothan-Hartee-Fock equations an orthogonal basis is needed, first the first orthogonalization matrix is constructed from the overlap matrix. First by a diagonalization:

.. math::
   SL_{S}=L_{S}\Lambda_{S}

and then the orthogonalization matrix is constructed:

.. math::
   S_{\mathrm{ortho}}=L_{S}\Lambda_{S}^{-1/2}L_{S}^{T}

The SCF iterations requires an initial Fock matrix given as:

.. math::
   F_{0}=H^{\mathrm{core}}

The SCF procedure is calulated as the following equations. First the fock matrix is constructed:

.. math::
   \left.F_{n,ij}\right|_{n\neq0}=H_{ij}^{\mathrm{core}}+\sum_{kl}^{\mathrm{AO}}D_{n-1,kl}\left(2\left(ij\left|\right|kl\right)-\left(ik\left|\right|jl\right)\right)

Then the Fock matrix is brought into the orthogonal basis:

.. math::
   F_{n}^{'}=S_{\mathrm{ortho}}^{T}H^{\mathrm{core}}S_{\mathrm{ortho}}

The F' is then diagonalized:

.. math::
   F_{n}^{'}C_{n}^{'}=C_{n}^{'}\epsilon_{n}

The coefficients are transformed back:

.. math::
   C_{n}=S_{\mathrm{ortho}}C_{n}^{'}

A density matrix can be made from the coefficients:

.. math::
   D_{n,ij}=\sum_{k}^{\mathrm{occ}}C_{n,ki}C_{n,kj}

The electronic energy of system can be found as:

.. math::
   E_{n,\mathrm{elec}}=\sum_{ij}^{\mathrm{AO}}D_{0,ij}\left(H_{ij}^{\mathrm{core}}+F_{n,ij}\right)

The above SCF procedure will is stopped at certain tresholds. The change in energy and the RMSD of the density matrix can be found as:

.. math::
   \Delta E_{n}=E_{n,\mathrm{elec}}-E_{n-1,\mathrm{elec}}

.. math::
   \mathrm{RMSD}_{n}=\sqrt{\sum_{ij}D_{n,ij}-D_{n-1,ij}}

FUNCTION:

- HartreeFock.HartreeFock(input, VNN, Te, S, VeN, Vee, deTHR=10**-6,rmsTHR=10**-6,Maxiter=100, DO_DIIS='Yes', DIIS_steps=6, print_SCF='Yes')
- return EHF, C, F, D, iter

Input:

- input, inputfile object
- VNN, nuclear-nuclear repulsion
- Te, electronic energy matrix
- S, overlap matrix
- VeN, nuclear-electron attraction matrix
- Vee, ERI matrix
- deTHR, energy change threshold for convergence
- rmsTHR, RMS threshold for convergence
- Maxiter, max SCF iterations
- DO_DIIS, enable DIIS, if = 'Yes'
- DIIS_steps, how many steps are stored in DIIS
- print_SCF, wether to print SCF to output

Output:

- EHF, Total Hartree-Fock energy
- C, MO coefficients
- F, Fock matrix
- D, density matrix
- iter, SCF iterations used

References:

- http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project3


Unrestricted Hartee-Fock
------------------------

The unrestricted Hartee-Fock method uses the same SCF procedure as as the restricted Hartree-Fock, but with the Fock matrix coupling the alpha and beta spins:

.. math::
   F_{n,\alpha,ij}=H_{ij}^{\mathrm{core}}+\sum_{kl}^{\mathrm{AO}}D_{n-1,\alpha,kl}\left(\left(ij\left|\right|kl\right)-\left(ik\left|\right|jl\right)\right)+\sum_{kl}^{\mathrm{AO}}D_{n-1,\beta,kl}\left(ij\left|\right|kl\right)

FUNCTION:

- UHF.HartreeFock(input, VNN, Te, S, VeN, Vee, deTHR=10**-6,rmsTHR=10**-6,Maxiter=100, UHF_mix=0.15, print_SCF='Yes')
- return EUHF, C_alpha, F_alpha, D_alpha, C_beta, F_beta, D_beta, iter

Input:

- input, inputfile object
- VNN, nuclear-nuclear repulsion
- Te, electronic energy matrix
- S, overlap matrix
- VeN, nuclear-electron attraction matrix
- Vee, ERI matrix
- deTHR, energy change threshold for convergence
- rmsTHR, RMS threshold for convergence
- Maxiter, max SCF iterations
- UHF_mix, how much beta and alpha are mixed to break symmetry
- print_SCF, wether to print SCF to output

Output:

- EUHF, total UHF energy
- C_alpha, MO coefficients
- F_alpha, Fock matrix
- D_alpha, density matrix
- C_beta, MO coefficients
- F_beta, Fock matrix
- D_beta, density matrix


References:

- Szabo and Ostlund, Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory

In unrestricted Hartree-Fock for a closed shell system the spin-symmetry needs to be broken else restricted Hartree-Fock is restored. This is done by the following method, after the first MO coefficients have been made:

.. math::
   C_{i,\mathrm{HOMO}}^{\mathrm{new}}=\frac{1}{\sqrt{1+k^{2}}}\left(C_{i,\mathrm{HOMO}}^{\mathrm{old}}+kC_{i,\mathrm{LUMO}}^{\mathrm{old}}\right)

.. math::
   C_{i,\mathrm{LUMO}}^{\mathrm{new}}=\frac{1}{\sqrt{1+k^{2}}}\left(-kC_{i,\mathrm{HOMO}}^{\mathrm{old}}+C_{i,\mathrm{LUMO}}^{\mathrm{old}}\right)



