#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 16:19:10 2021

@author: tufan
"""

import openmdao.api as om
import numpy as np

class T_corner(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('disc_SiC', types=int,default=10,desc='SiC-Discretization in r-direction')
        self.bound1 = self.options['disc_SiC']+2;
        
        # print('T_corner.initialize')
    
    def setup(self):
        
        self.add_input('T_side',val=293.0, shape=(self.options['disc_SiC'],1), desc='Temperature of the verical surf of SiC', units='K')
        self.add_input('Ac_SiC', shape=(self.options['disc_SiC'],1), units='m**2', desc='Cross sectional area of SiC elements')
        
        self.add_output('T_corner', units='K', desc='Corner temperature')
        
        self.declare_partials(of='*', wrt='*',method='fd')
        self.linear_solver = om.ScipyKrylov()
        
        # print('T_corner.setup')

    def compute(self, inputs, outputs):

        # outputs['T_corner']=sum((np.multiply(inputs['Ac_SiC'],inputs['T_side'][:,0])).diagonal())/sum(inputs['Ac_SiC'])
        outputs['T_corner']=sum((np.multiply(inputs['Ac_SiC'],inputs['T_side'])))/sum(inputs['Ac_SiC'])
        
        # print('T_corner.compute')
        
# if __name__ == '__main__':
    
#     pass