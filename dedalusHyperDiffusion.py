#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:37:36 2022

@author: bradleydavy
"""

from dedalus.core.basis import Fourier
from dedalus.core.field import Operand
from dedalus.core.operators import Separable, FutureField
from dedalus.tools.array import reshape_vector
import dedalus.public as de
import numpy as np

class hyperDiffusion(Separable, FutureField):
    
    '''
    Parameters
    ----------
    cls : TYPE
        DESCRIPTION.
    arg0 : TYPE
        DESCRIPTION.
    *args : TYPE
        DESCRIPTION.
    **kw : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    def __new__(cls, arg0, *args, **kw):

        # Cast to operand
        arg0 = Operand.cast(arg0)

        # Check all bases are Fourier
        for basis in arg0.domain.bases:
            if not isinstance(basis, Fourier):
                raise NotImplementedError("Operator only implemented for full-Fourier domains. ")
        
        # Check for scalars
        if arg0.domain.dim == 0:
            return 0
        else:
            return object.__new__(cls)

    def __init__(self, arg, q = 1.1, s = 1, i_0 = 120, **kw):
        arg = Operand.cast(arg)
        super().__init__(arg, **kw)
        self.kw = {'s': s}
        self.s = s
        self.q = q
        self.i_0 = i_0
        self.name = 'Lap[%s]' % self.s
        self.axis = None
        
        # Build operator symbol array
        slices = self.domain.dist.coeff_layout.slices(self.domain.dealias)
        local_wavenumbers = [self.domain.elements(axis) for axis in range(self.domain.dim)] # These are my wave numbers
        local_k2 = np.sum([ki**2 for ki in local_wavenumbers], axis=0) ## This is squaring them like a typical laplacian
        hyper_diffusion_array = self.generate_hyperdifussion_array(np.shape(local_k2), q, i_0)
        self.local_symbols = local_k2*hyper_diffusion_array ## This raises them to the power s for the fractional laplacian

    def meta_constant(self, axis):
        # Preserve constancy
        return self.args[0].meta[axis]['constant']

    def check_conditions(self):
        arg0, = self.args
        # Must be in coeff layout
        is_coeff = np.any(arg0.layout.grid_space)
        return is_coeff

    def operator_form(self, index):
        # Get local index, special casing zero mode for NCC preconstruction pass
        if any(index):
            local_index = index - self.domain.dist.coeff_layout.start(scales=None)
        else:
            local_index = index
        return self.local_symbols[tuple(local_index)]
    
    def generate_hyperdifussion_array(self, shape, q, i_0):
        '''
        Parameters
        ----------
        shape : Shape of the output array.
        q : Hyperdiffusion parameter, defined below.
        i_0 : Cut off wavenumber.

        Returns
        -------
        hyper_diffusion_array : Array containing the functional form of the hyperdiffusion.
        
        
        Notes
        -----
        
        Implements hyperdiffusion of the form \nu = \nu_0 (q**(i - i_0)) if i > i_0 else \nu_0.
        '''

        hyper_diffusion_array = np.ones(shape, dtype=float)
        if len(np.shape(hyper_diffusion_array)) == 1:
            ## 1d problem requires 1d array.
            for i in range(len(hyper_diffusion_array)):
                if i > i_0:
                    hyper_diffusion_array[i] = q**(i-i_0)
            return hyper_diffusion_array
        
        elif len(np.shape(hyper_diffusion_array)) == 2:
            ## 2d problem requires 2d array.
            for i in range(np.shape(hyper_diffusion_array)[0]):
                for j in range(np.shape(hyper_diffusion_array)[1]):
                        k = (i**2 + j**2)**0.5
                        if k > i_0:
                            hyper_diffusion_array[i][j] = q**(k - i_0)
                            
            return hyper_diffusion_array

    def operate(self, out):
        arg0, = self.args
        # Require coeff layout
        arg0.require_coeff_space()
        out.layout = arg0.layout
        # Apply symbol array to coefficients
        np.multiply(arg0.data, self.local_symbols, out=out.data)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        