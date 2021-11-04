#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 22:47:55 2021

@author: tufanakba
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 12:42:37 2021

@author: tufanakba
P1 irrad. class using MDAO api
"""

import openmdao.api as om
import numpy as np
from math import log

class I_Rad(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('disc_z', types=int,default=20,desc='Discretization in z-direction')
        self.options.declare('disc_SiC', types=int,default=10,desc='SiC-Discretization in r-direction')
        self.options.declare('disc_RPC', types=int,default=20,desc='RPC-Discretization in r-direction')
        self.options.declare('disc_INS', types=int,default=10,desc='INS-Discretization in r-direction')
        
        # disc_z = self.options['disc_z']
        # disc_SiC = self.options['disc_SiC']
        # disc_RPC = self.options['disc_RPC']
        # disc_INS = self.options['disc_INS']
        
        self.disc_r=self.options['disc_SiC']+self.options['disc_RPC']+self.options['disc_INS'];
        self.Nr=self.disc_r+4;
        
        self.a = np.zeros((self.options['disc_RPC']+2,1),dtype=float)
        self.b = np.zeros((self.options['disc_RPC']+2,1),dtype=float)
        self.c = np.zeros((self.options['disc_RPC']+2,1),dtype=float)
        self.d = np.zeros((self.options['disc_RPC']+2,self.options['disc_z']),dtype=float)
        # self.G = np.zeros((self.options['disc_RPC']+2,self.options['disc_z']),dtype=float)
        
        # print('fluid.initialize')

    def setup(self):
        
        self.add_input('E',val=0.9,desc='Emissivity')
        self.add_input('E_INS',val=0.5,desc='Emissivity of Insulation')
        self.add_input('omega', val=0.2, desc='scattering albedo')
        self.add_input('K_ex',val=200,desc='Extinction Coefficient',units='m**-1')
        
        self.add_input('sigma',val=5.67*10**(-8),desc='Stefan-Boltzman Const.',units='W/(m**2*K**4)')
        
        self.add_input('r_n', shape=(self.Nr,1), units='m', desc='radial node coordinates')
        
        self.add_input('T_RPC',val=293.0, shape=(self.options['disc_RPC'],self.options['disc_z']), desc='Temperature distribution of the RPC', units='K')
        self.add_input('T_RPC_CAV',val=293.0, shape=(1,self.options['disc_z']), desc='Temperature distribution of RPC-CAV', units='K')
        self.add_input('T_RPC_INS',val=293.0, shape=(1,self.options['disc_z']), desc='Temperature distribution of RPC-CAV', units='K')
        
        self.add_output('G', shape=(self.options['disc_RPC']+2,self.options['disc_z']), desc='incident radiation of the PRC', units='W/m**2')#val=G_ini???
        
        self.declare_partials('*', '*', method='fd')
        self.linear_solver = om.LinearRunOnce()
        
        # print('fluid.setup')
    
    def compute(self, inputs, outputs):
        
        disc_z = self.options['disc_z']
        disc_RPC = self.options['disc_RPC']
        
        E = inputs['E']
        E_INS = inputs['E_INS']
        # omega = inputs['omega']
        K_ex = inputs['K_ex']
        C_C = 3 * inputs['K_ex']**2*(1-inputs['omega'])
        
        # sigma = 4 * inputs['sigma']
        
        r_n = inputs['r_n'][self.options['disc_SiC']+1:self.options['disc_SiC']+disc_RPC+3,0]
        
        T_CAV = 4 * inputs['sigma'] * inputs['T_RPC_CAV']**4
        T_INS = 4 * inputs['sigma'] * inputs['T_RPC_INS']**4
        T_RPC = 4 * inputs['sigma'] * inputs['T_RPC']**4
        
        # G = np.zeros((disc_RPC+2,disc_z))
        
        self.b[0,0] = B_cav = -2/3*(2-E)/E/K_ex
        self.a[-1,0] = A_ins = -2/3*(2-E_INS)/E_INS/K_ex
        c2_cav = r_n[0]*log(r_n[1]/r_n[0])
        c2_ins = r_n[-1]*log(r_n[-1]/r_n[-2])
        self.c[0,0] = C_cav = c2_cav - self.b[0,0]
        self.c[-1,0] = C_ins = c2_ins - self.a[-1,0]
        
            
        self.d[0,:] = D_cav = c2_cav*T_CAV[0,:]
        self.d[-1,:] = D_ins = c2_ins*T_INS[0,:]
        
        c3_RPC = np.zeros((disc_RPC+2),dtype=float)
        for j in range(1,disc_RPC+1):
            
            self.a[j,0] = 1/log(r_n[j]/r_n[j-1])
            self.b[j,0] = 1/log(r_n[j+1]/r_n[j])
            c3_RPC[j] =  r_n[j]**2*log((r_n[j]+r_n[j+1])/(r_n[j]+r_n[j-1]))*C_C
            self.c[j,0] = -self.a[j,0]-self.b[j,0]-c3_RPC[j]
            
        for j in range(1,disc_RPC+1):
            for i in range(0,disc_z):
                
                self.d[j,i] = -c3_RPC[j]*T_RPC[j-1,i]
                
        # flat the arrays as vectors
        # v_a=self.a.reshape(-1,1)
        # v_b=self.b.reshape(-1,1)
        # v_c=self.c.reshape(-1,1)
        # v_d=self.d.reshape(-1,1)
        # v_e=self.e.reshape(-1,1)
        # v_f=self.f.reshape(-1,1)
        
        # Left Hand Side Matrix
        # LHS = np.diagflat(v_c) + np.diagflat(v_a[disc_z:(disc_z*(disc_RPC+2)),0],-(disc_z)) + np.diagflat(v_b[0:(disc_z*(disc_RPC+2))-(disc_z),0],disc_z)
        # LHS = np.linalg.inv(np.diagflat(self.c) + np.diagflat(self.a[1:,0],-1) +np.diagflat(self.b[:-1,0],1))
        LHS = np.diagflat(self.c) + np.diagflat(self.a[1:,0],-1) +np.diagflat(self.b[:-1,0],1)
        
        # np.save('a.npy',self.a)
        # np.save('b.npy',self.b)
        # np.save('c.npy',self.c)
        # np.save('d.npy',self.d)
        # np.save('LHS.npy',LHS)
        # np.save('v_d.npy',v_d)
        # Matrix Inversion
        # v_G = np.matmul(np.linalg.inv(LHS),v_d)
        for i in range(0,disc_z):
            # self.G[:,i] =  np.matmul(LHS,self.d[:,i])
            outputs['G'][:,i] = np.linalg.solve(LHS,self.d[:,i])
            
            
            
        # self.G = G_new = np.resize(v_G,(disc_RPC+2,disc_z))
        # outputs['G'] = self.G
        
        # print('fluid.compute')


if __name__ == '__main__':
    
    p = om.Problem()
    p.model.add_subsystem('I_Rad', I_Rad())
    
    p.setup()

    p.set_val('I_Rad.r_n', np.load('r_n.npy'))
    
    p.run_model()
    G =  p.get_val('I_Rad.G')
    print(f'G:\n{G}')
    # a = np.load('a.npy')
    # b = np.load('b.npy')
    # c = np.load('c.npy')
    # d = np.load('d.npy')