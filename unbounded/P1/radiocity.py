#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Aug 24 13:28:10 2021

@author: tufan
This model for cavitiy radiocity eqn. solution for receiver model
"""

import openmdao.api as om
import numpy as np
from math import pi
import time

class radiocity(om.ExplicitComponent):

    
    def initialize(self):
        
        self.options.declare('disc_z', types=int, default=20, desc='Discretization in z-direction')
        
        disc_z = self.options['disc_z']
        self.Nz=disc_z+2;
        
        # print('radiocity.initialize')
        
    def setup(self):
        
        disc_z = self.options['disc_z']
        
        self.add_input('Q_Solar_In',val=1000, desc='Total Solar Power Input',units='W')
        
        self.add_input('F_I_1', shape=(1,disc_z), desc='View factor aperture to cavity')
        self.add_input('F_I_BP', desc='View factor aperture to back plate')
        self.add_input('F_1_BP', shape=(1,disc_z), desc='View factor cavity to BP')
        self.add_input('F_BP_1', shape=(1,disc_z), desc='View factor BP to cavity')
        self.add_input('F', shape=(disc_z,disc_z), desc='View factor cavity to cavity')
        
        self.add_input('sigma',val=5.67*10**(-8),desc='Stefan-Boltzman Const.',units='W/(m**2*K**4)')
        
        self.add_input('r_1', val = 0.015,desc='Inner radius SiC', units='m')
        self.add_input('dL', desc='Axial mesh length', units='m')
        
        self.add_input('B', shape=(disc_z+1,disc_z+1), desc='Coeff. matrix of the radiocity')
        
        "CAV surface Temperature + BP"
        self.add_input('T_cav',val=293.0, shape=(1,self.Nz), desc='Temperature of the inner surf of CAV', units='K')
        self.add_input('T_BP',val=293.0, desc='Temperature of the BP of CAV', units='K')
        self.add_output('Q',shape=(disc_z+1,1), desc='Heat input to the surface of CAV', units='W')
        
        self.declare_partials(of='*', wrt='*',method='fd')
        self.linear_solver = om.ScipyKrylov()
        
        # print('radiocity.setup')
    
    def compute(self, inputs, outputs):
        
        # in fucntion param.s
        disc_z = self.options['disc_z']
        Nz = self.Nz
        
        Q_Solar_In = inputs['Q_Solar_In']
        
        F_I_1 = inputs['F_I_1']
        F_I_BP = inputs['F_I_BP']
        F_1_BP = inputs['F_1_BP']
        F_BP_1 = inputs['F_BP_1']
        F = inputs['F']
        
        sigma = inputs['sigma']
        
        r_1 = inputs['r_1']
        dL = inputs['dL']
        
        B = inputs['B']
        
        T = inputs['T_cav']
        T_BP = inputs['T_BP']
        
        K = np.zeros((disc_z+1,1),dtype=float)
        for i in range(1,disc_z+1):
            K[i-1,0] = -Q_Solar_In*F_I_1[0,i-1]/(2*pi*r_1*dL)+sigma*T[0,i]**4-sum(sigma*np.multiply(np.power(T[0,1:Nz-1],4),F[i-1,:]))-sigma*T_BP**4*F_1_BP[0,i-1]                
        
        K[disc_z,0] = -Q_Solar_In*F_I_BP/(pi*r_1**2)+sigma*T_BP**4-sum(sigma*np.multiply(np.power(T[0,1:Nz-1],4),F_BP_1[0,:]))
        
        Q_Solar_Net_new_1=np.matmul(np.linalg.inv(B),(-K))
        
        outputs['Q'][-1,0] = Q_Solar_BP = ((pi*r_1**2*Q_Solar_Net_new_1[-1,0])**2)**0.5;
        outputs['Q'][0:-1,0] = Q_Solar_Net_new = (Q_Solar_Net_new_1[0:-1,0]*2*pi*r_1*dL).reshape(1,len(Q_Solar_Net_new_1[0:-1,0]))
        
        # if Q_Solar_BP<0:
        #     outputs['Q'][-1,0] = 1273^4*pi*r_1**2*sigma*0.9
        
        # print('radiocity.compute')
        # print(f'radiocity.compute\nQ:{Q_Solar_Net_new}')
        
if __name__ =='__main__':
    
    p = om.Problem()
    p.model.add_subsystem('radiocity', radiocity(disc_z = 20))
    p.setup()
    
    p.set_val('radiocity.F_I_1', np.load('F_I_1.npy'))
    p.set_val('radiocity.F_I_BP', np.load('F_I_BP.npy'))
    p.set_val('radiocity.F_1_BP', np.load('F_1_BP.npy'))
    p.set_val('radiocity.F_BP_1', np.load('F_BP_1.npy'))
    p.set_val('radiocity.F', np.load('F.npy'))
    p.set_val('radiocity.B', np.load('B.npy'))
    p.set_val('radiocity.dL', np.load('dL.npy'))
    
    p.run_model()
    
    Q=p.get_val('radiocity.Q')