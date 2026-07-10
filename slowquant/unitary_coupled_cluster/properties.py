import numpy as np
import scipy
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor
from typing import Any

from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS

from slowquant.molecularintegrals.integralfunctions import one_electron_integral_transform
from slowquant.unitary_coupled_cluster.operators import one_elec_op_0i_0a
from slowquant.unitary_coupled_cluster.operator_state_algebra import expectation_value
from slowquant.unitary_coupled_cluster.linear_response import naive

class properties():
    def __init__(
        self,
        wave_function: WaveFunctionUCC | WaveFunctionUPS,
        property_options: dict[str, Any] = {},
    ) -> None:
        """Initialize property calculations.

        Args:
            wave_function: Wave function object.
        """
        self.wf = wave_function
        if isinstance(self.wf, WaveFunctionUCC):
            self.index_info = (
                self.wf.ci_info,
                self.wf.thetas,
                self.wf.ucc_layout,
            )
        elif isinstance(self.wf, WaveFunctionUPS):
            self.index_info = (
                self.wf.ci_info,
                self.wf.thetas,
                self.wf.ups_layout,
            )
        else:
            raise ValueError(f"Got incompatible wave function type, {type(self.wf)}")

        self.property_options = property_options
        self._LR_singlet = None
        self._LR_triplet = None

    @property
    def LR_singlet(self) -> np.ndarray:
        """Calculate singlet linear response.
        
        Returns:
            singlet spin-adapted linear response object
        """
        if self._LR_singlet is None:
            if "excitations" not in self.property_options.keys():
                # default option
                self.property_options["excitations"] = "SD"
            if "lr_formulation" not in self.property_options.keys():
                # default option
                self.property_options["lr_formulation"] = naive
            self._LR_singlet = self.property_options["lr_formulation"].LinearResponse(
                self.wf, 
                excitations=self.property_options["excitations"], 
                triplet=False
                )
        return self._LR_singlet
    
    @property
    def LR_triplet(self) -> np.ndarray:
        """Calculate triplet linear response.
        
        Returns:
            triplet spin-adapted linear response object
        """
        if self._LR_triplet is None:
            if "excitations" not in self.property_options.keys():
                # default option
                self.property_options["excitations"] = "SD"
            if "lr_formulation" not in self.property_options.keys():
                # default option
                self.property_options["lr_formulation"] = naive
            self._LR_triplet = self.property_options["lr_formulation"].LinearResponse(
                self.wf, 
                excitations=self.property_options["excitations"], 
                triplet=True
                )
        return self._LR_triplet

    def get_polarisability(self, freq=0) -> np.ndarray:
        """Calculate the frequency dependent polarisability tensor.

        Returns:
            Polarisability tensor (in au).
        """
        prop_grad = self.LR_singlet.get_property_gradient(self.wf.int_gen.electric_dipole)
        
        if freq == 0:
            response = scipy.linalg.solve(self.LR_singlet.hessian, prop_grad)
        else:
            response = scipy.linalg.solve(self.LR_singlet.hessian - freq * self.LR_singlet.metric, prop_grad)
        
        alpha = np.einsum('ix,iy->xy', prop_grad, response)

        print(f'Polarisabilities:\n \t xx: {alpha[0,0]:.4f} \t yy: {alpha[1,1]:.4f} \t zz: {alpha[2,2]:.4f}')

        return alpha

    def get_nuclear_shielding_tensor(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the shielding tensor of each nuclei.

        Returns:
            Diamagnetic and paramagnetic shielding tensor for each nuclei (in ppm).
        """
        atoms = self.wf.int_gen.atom_coordinates
        dia_shield = np.zeros((len(atoms), 3, 3))
        para_shield = np.zeros((len(atoms), 3, 3))

        for i in range(len(atoms)):
            dia_i = []
            origin = atoms[i,:]

            # Diamagnetic term
            dia_ao = self.wf.int_gen.diamagnetic_shielding(common_orig=origin, atom_coord=origin)

            for comp in dia_ao:
                dia_mo = one_electron_integral_transform(self.wf.c_mo, comp)
                dia_op = one_elec_op_0i_0a(dia_mo, self.wf.num_inactive_orbs, self.wf.num_active_orbs)
                dia_i.append(expectation_value(self.wf.ci_coeffs, 
                                               [dia_op], 
                                               self.wf.ci_coeffs, 
                                               *self.index_info))
            
            dia_i = np.array(dia_i).reshape((3,3))
            dia_shield[i,:,:] = dia_i - dia_i.trace() * np.eye(3)
            
            # PSO
            property_gradient = self.LR_singlet.get_property_gradient(
                self.wf.int_gen.orbital_paramagnetic(origin)
                )
            response_vector = scipy.linalg.solve(self.LR_singlet.hessian, property_gradient)

            # Anguar Momentum
            property_gradient = self.LR_singlet.get_property_gradient(
                self.wf.int_gen.angular_momentum(origin)
                )
            
            # Paramagnetic shielding tensor
            para_shield[i,:,:] -= np.einsum('ix,iy->xy', response_vector, property_gradient)


        dia_shield *= nist.ALPHA**2 * 1e6
        para_shield *= nist.ALPHA**2 * 1e6

        print('Shielding (in ppm):')
        for i in range(len(atoms)):
            print(f'{i}: \tTotal={np.trace(dia_shield[i,:,:] + para_shield[i,:,:]) / 3:.4f} \tDia={np.trace(dia_shield[i,:,:]) / 3:.4f} \tPara={np.trace(para_shield[i,:,:]) / 3:.4f}')

        return dia_shield, para_shield
    
    def get_spin_spin_coupling_constant(self) -> np.ndarray:
        """Calculate the spin-spin coupling constant tensor of each nuclei.

        Returns:
            DSO, PSO, FC and SD coupling tensor for each nuclei (in Hz).
        """
        atoms = self.wf.int_gen.atom_coordinates
        nuc_pair = [(i,j) for i in range(len(atoms)) for j in range(i)]

        # DSO term
        dso = np.zeros((len(nuc_pair), 3, 3))
        dso_k = []
        for k, (i,j) in enumerate(nuc_pair):
            dso_ao = self.wf.int_gen.orbital_diamagnetic(atom1_coord=atoms[i,:], atom2_coord=atoms[j,:])
            for comp in dso_ao:
                dso_mo = one_electron_integral_transform(self.wf.c_mo, comp)
                dso_op = one_elec_op_0i_0a(dso_mo, self.wf.num_inactive_orbs, self.wf.num_active_orbs)
                dso_k.append(expectation_value(self.wf.ci_coeffs, 
                                               [dso_op], 
                                               self.wf.ci_coeffs, 
                                               *self.index_info))
            
            dso_k = - np.array(dso_k).reshape((3,3))
            dso[k,:,:] = dso_k - dso_k.trace() * np.eye(3)
        
        # PSO term
        property_gradient = []
        response_vector = []
        for i in range(len(atoms)):
            property_gradient.append(
                self.LR_singlet.get_property_gradient(
                self.wf.int_gen.orbital_paramagnetic(atoms[i,:])
                )
            )
            response_vector.append(
                scipy.linalg.solve(
                    self.LR_singlet.hessian, 
                    property_gradient[i])
            )
        
        pso = np.zeros_like(dso)
        for k, (i,j) in enumerate(nuc_pair):
            pso[k,:,:] -= np.einsum(
                'ix,iy->xy', 
                response_vector[i], 
                property_gradient[j]
                )

        # FC term
        property_gradient = []
        response_vector = []
        for i in range(len(atoms)):
            property_gradient.append(
                self.LR_triplet.get_property_gradient(
                self.wf.int_gen.fermi_contact(atoms[i,:])
                )
            )
            response_vector.append(
                scipy.linalg.solve(
                    self.LR_triplet.hessian, 
                    property_gradient[i])
            )
        
        fc = np.zeros_like(dso)
        for k, (i,j) in enumerate(nuc_pair):
            fc[k,:,:] -= (
                np.ones((3,3)) * np.einsum(
                    'ix,iy->xy',
                    response_vector[i], 
                    property_gradient[j]
                )
            )

        # SD+FC term
        property_gradient = []
        response_vector = []
        for i in range(len(atoms)):
            property_gradient.append(
                self.LR_triplet.get_property_gradient(
                self.wf.int_gen.spin_dipolar_fermi_contact(atoms[i,:])
                ).reshape(-1, 3, 3)
            )
            response_vector.append(
                scipy.linalg.solve(
                    self.LR_triplet.hessian, 
                    property_gradient[i])
            )
        
        sdfc = np.zeros_like(dso)
        for k, (i,j) in enumerate(nuc_pair):
            sdfc[k,:,:] -= (
                np.einsum(
                    'ixw,iyw->xy',
                    response_vector[i], 
                    property_gradient[j]
                )
            )

        def convert_unit(sscc_tensor):
            nuc_magneton = .5 * (nist.E_MASS/nist.PROTON_MASS)
            au2Hz = nist.HARTREE2J / nist.PLANCK
            unit = au2Hz * nuc_magneton ** 2 * nist.ALPHA**4
            sscc_tensor *= unit

            charges = self.wf.int_gen.atom_charges
            gyro = []
            for i in range(len(atoms)):
                gyro.append(get_nuc_g_factor(charges[i]))
            
            for k, (i,j) in enumerate(nuc_pair):
                sscc_tensor[k,:,:] *= gyro[i] * gyro[j]
            return sscc_tensor
        
        jtensor_dso  = convert_unit(dso)
        jtensor_pso  = convert_unit(pso)
        jtensor_fc   = convert_unit(fc)
        jtensor_sdfc = convert_unit(sdfc)
        jtensor_total = jtensor_dso + jtensor_pso + jtensor_sdfc

        print('SSCC (in Hz):')
        for k, (i,j) in enumerate(nuc_pair):
            print(f'{i} {j}: \tDSO={np.trace(jtensor_dso[k,:,:])/3:.4f} \tPSO={np.trace(jtensor_pso[k,:,:])/3:.4f} \tSD={np.trace(jtensor_sdfc[k,:,:]-jtensor_fc[k,:,:])/3:.4f} \tFC={np.trace(jtensor_fc[k,:,:])/3:.4f} \tTotal={np.trace(jtensor_total[k,:,:])/3:.4f}')

        return jtensor_dso, jtensor_pso, jtensor_fc, (jtensor_sdfc - jtensor_fc)