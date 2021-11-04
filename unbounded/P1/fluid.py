#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 00:47:08 2021

@author: tufan
"""

import openmdao.api as om
import numpy as np

class fluid(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('disc_z', types=int, default=20,desc='Discretization in z-direction')
        self.options.declare('disc_RPC', types=int, default=20,desc='RPC-Discretization in r-direction')
        
        # print('fluid.initialize')

    def setup(self):
        
        disc_z = self.options['disc_z']
        disc_RPC = self.options['disc_RPC']
        
        self.add_input('h_',desc='Heat transfer coefficient inside RPC',units='W/(m**2*K)')
        self.add_input('A_spec',val=500,desc='Specific Surface of the RPC',units='m**-1')
        self.add_input('V_RPC', shape=(disc_RPC,1), desc='Volume of each RPC element', units='m**3')
        self.add_input('m', shape=(disc_RPC,1), desc='mass flow rate passing inside each RPC element', units='kg/s')
        self.add_input('cp',val=1005.,desc='Specific Heat Capacity',units='J/(kg*K)')
        
        self.add_input('T_RPC', shape=(disc_RPC,disc_z), desc='Temperature distribution of the PRC', units='K')
        self.add_input('T_fluid_in', val=293.,desc='Inlet temperature of air',units='K')
        
        self.add_output('T_fluid',val=293.0, shape=(disc_RPC,disc_z), desc='Temperature of the air', units='K')
        self.add_output('T_fluid_out', val=293.0, desc='Outlet temperature of air',units='K')
        
        # for efficiency calculations
        self.add_input('Q_Solar_In',val=1000., desc='Total Solar Power Input',units='W')
        self.add_output('eff_S2G', desc='Efficiency solar input to outlet',units='K')
        
        self.declare_partials('*', '*', method='fd')
        self.linear_solver = om.ScipyKrylov()
        
        # print('fluid.setup')
    
    def compute(self, inputs, outputs):
        
        disc_z = self.options['disc_z']
        disc_RPC = self.options['disc_RPC']
        
        h_ = inputs['h_']
        A_spec = inputs['A_spec']
        V_RPC = inputs['V_RPC']
        m = inputs['m']
        cp = inputs['cp']
        
        T_RPC = inputs['T_RPC']
        T_IN = np.ones((disc_RPC,1), dtype=float)*inputs['T_fluid_in']
        T_OUT  = np.zeros((disc_RPC,1), dtype=float)
        Tf = np.zeros((disc_RPC,disc_z), dtype=float)
        
        for i in range(1,disc_z+1):
            for j in range(1,disc_RPC+1):
                T_OUT[j-1,0]=(2*h_*A_spec*T_RPC[j-1,i-1]*V_RPC[j-1,0]-T_IN[j-1,0]*(h_*A_spec*V_RPC[j-1,0]-2*m[j-1,0]*cp))/(h_*A_spec*V_RPC[j-1,0]+2*m[j-1,0]*cp);
                Tf[j-1,i-1]=(T_IN[j-1,0]+T_OUT[j-1,0])/2;
            
            T_IN=np.copy(T_OUT);
        
        outputs['T_fluid'] = Tf
        outputs['T_fluid_out'] = sum(np.multiply(m,T_OUT)/np.sum(m));
        # print(outputs['T_fluid_out'])
        
        Q_Fluid=cp*sum(np.multiply(m, (T_OUT-inputs['T_fluid_in'])));
        
        Q_Solar_In = inputs['Q_Solar_In']
        outputs['eff_S2G'] = (Q_Fluid/Q_Solar_In)
        
        # print('fluid.compute')