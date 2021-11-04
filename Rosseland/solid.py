#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:40:50 2021

@author: tufan
finite different solver for Rosseland approx.
"""

import openmdao.api as om
import numpy as np
from math import pi, log

class solid(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('disc_z', types=int,default=20,desc='Discretization in z-direction')
        self.options.declare('disc_SiC', types=int,default=10,desc='SiC-Discretization in r-direction')
        self.options.declare('disc_RPC', types=int,default=20,desc='RPC-Discretization in r-direction')
        self.options.declare('disc_INS', types=int,default=10,desc='INS-Discretization in r-direction')
        
        disc_z = self.options['disc_z']
        disc_SiC = self.options['disc_SiC']
        disc_RPC = self.options['disc_RPC']
        disc_INS = self.options['disc_INS']
        
        self.disc_r=disc_SiC+disc_RPC+disc_INS;
        self.Nr = Nr =self.disc_r+4;
        self.Nz = Nz = disc_z+2;
        self.NN = self.Nz*self.Nr;
        self.bound1 = bound1 = disc_SiC+2;
        self.bound2 = bound2 = disc_SiC+2+disc_RPC+1;
        
        #In. Val.s
        self.a = np.zeros((self.Nr,self.Nz),dtype=float)
        self.b = np.zeros((self.Nr,self.Nz),dtype=float)
        self.c = np.zeros((self.Nr,self.Nz),dtype=float)
        self.d = np.zeros((self.Nr,self.Nz),dtype=float)
        self.e = np.zeros((self.Nr,self.Nz),dtype=float)
        self.f = np.zeros((self.Nr,self.Nz),dtype=float)
        
        # Coefficient Matrices (Edges)
        self.a[0,0]=1;
        self.c[0,0]=-0.5;
        self.e[0,0]=-0.5;
        self.a[0,Nz-1]=1;
        self.c[0,Nz-1]=-0.5;
        self.d[0,Nz-1]=-0.5;
        self.a[Nr-1,0]=1;
        self.b[Nr-1,0]=-0.5;
        self.e[Nr-1,0]=-0.5;
        self.a[Nr-1,Nz-1]=1;
        self.b[Nr-1,Nz-1]=-0.5;
        self.d[Nr-1,Nz-1]=-0.5;
        
        #Coefficient Matrices (Boundary Edges)
        self.a[bound1-1,0]=1;
        self.a[bound2-1,0]=1;
        self.a[bound1-1,Nz-1]=1;
        self.a[bound2-1,Nz-1]=1;
        self.b[bound1-1,0]=-0.5;
        self.b[bound2-1,0]=-0.5;
        self.b[bound1-1,Nz-1]=-0.5;
        self.b[bound2-1,Nz-1]=-0.5;
        self.c[bound1-1,0]=-0.5;
        self.c[bound2-1,0]=-0.5;
        self.c[bound1-1,Nz-1]=-0.5;
        self.c[bound2-1,Nz-1]=-0.5;
        
        self.T_ini = 293
        self.v_T = np.ones((self.NN,1),dtype=float)*self.T_ini
        self.T = np.ones((Nr,Nz),dtype=float)*self.T_ini
        
        # print('solid.initialize')
        
    def setup(self):
        
        disc_z = self.options['disc_z']
        disc_SiC = self.options['disc_SiC']
        disc_RPC = self.options['disc_RPC']
        disc_INS = self.options['disc_INS']
        rows = disc_z+1
        
        self.add_input('Q', val = 0.85*1000/disc_z, shape=(rows,1), desc='Heat input to the surface of CAV', units='W')
        self.add_input('E',val=0.9,desc='Emissivity')
        self.add_input('sigma',val=5.67*10**(-8),desc='Stefan-Boltzman Const.',units='W/(m**2*K**4)')
        
        self.add_input('r_1', val = 0.015,desc='Inner radius SiC', units='m')
        self.add_input('dL', desc='Axial mesh length', units='m')
        
        self.add_input('k_SiC',val=33., desc='Conductivity SiC',units='W/(m*K)')
        
        self.add_input('r_n', shape=(self.Nr,1), units='m', desc='radial node coordinates')
        self.add_input('r_g', shape=(self.Nr-3,1), units='m', desc='radial grid coordinates')
        self.add_input('z_n', shape=(1,disc_z+2), units='m', desc='axial grid coordinates')
        
        self.add_input('h_loss_cav', val=0., units='W/(m**2*K)', desc='heat transfer coefficient from cavity')
        self.add_input('h_loss',val=15,desc='Heat transfer coefficient to ambient',units='W/(m**2*K)')
        self.add_input('h_loss_z', units='W/(m**2*K)', desc='heat transfer coefficient on vertical surfaces')
        self.add_input('h_',desc='Heat transfer coefficient inside RPC',units='W/(m**2*K)')

        self.add_input('Tamb',val=293,desc='Ambient Temperature', units='K')
        self.add_input('T_corner', val=925, units='K', desc='Corner temperature')
        
        self.add_input('k_RPC', desc='Conductivity RPC',units='W/(m*K)')
        self.add_input('k_INS',val=0.3,desc='Conductivity insulation',units='W/(m*K)')
        self.add_input('K_ex',val=200,desc='Extinction Coefficient',units='m**-1')
        
        self.add_input('Ac_SiC', shape=(disc_SiC,1), units='m**2', desc='Cross sectional area of SiC elements')
        self.add_input('Ac_RPC', shape=(disc_RPC,1), units='m**2', desc='Cross sectional area of RPC elements')
        self.add_input('Ac_INS', shape=(disc_INS,1), units='m**2', desc='Cross sectional area of INS elements')
        
        self.add_input('A_spec',val=500,desc='Specific Surface of the RPC',units='m**-1')
        self.add_input('V_RPC', shape=(disc_RPC,1), desc='Volume of each RPC element', units='m**3')
        
        self.add_input('T_fluid',val=293.0, shape=(disc_RPC,disc_z), desc='Temperature of the air', units='K')
        
        self.add_output('T', shape=(self.Nr*self.Nz,1), desc='Temperature distribution of the receiver', units='K')
        self.add_output('T_RPC', shape=(disc_RPC,disc_z), desc='Temperature distribution of the PRC', units='K')
        self.add_output('T_cav', shape=(self.Nz,1), desc='Temperature of the inner surf of CAV', units='K')
        self.add_output('T_outer', shape=(self.Nz,1), desc='Temperature of the outer surf of INS', units='K')
        self.add_output('T_BP', units='K', desc='Temperature of back plate')
        self.add_output('T_side', shape=(self.bound1-2,1), units='K', desc='Temperature of the inlet SiC face side')
        
        self.declare_partials(of='*', wrt='*',method='fd')
        self.linear_solver = om.ScipyKrylov()
        
        # print('solid.setup')
    
    def compute(self, inputs, outputs):
        
        disc_z = self.options['disc_z']
        disc_SiC = self.options['disc_SiC']
        disc_RPC = self.options['disc_RPC']
        disc_INS = self.options['disc_INS']
        bound1 = self.bound1
        bound2 = self.bound2
        Nr = self.Nr
        Nz = self.Nz
        NN = self.NN
        
        E = inputs['E']
        Q_Solar_Net_new_1 = inputs['Q']
        sigma = inputs['sigma']
        
        r_1 = inputs['r_1']
        dL = inputs['dL']
        
        k_SiC = inputs['k_SiC']
        
        r_n = inputs['r_n']
        r_g = inputs['r_g']
        z_n = inputs['z_n']
        
        h_loss_cav = inputs['h_loss_cav']
        h_loss = inputs['h_loss']
        h_loss_z = inputs['h_loss_z']
        h_ = inputs['h_']

        Tamb = inputs['Tamb']
        T_corner = inputs['T_corner']
        
        k_RPC = inputs['k_RPC']
        k_INS = inputs['k_INS']
        K_ex = inputs['K_ex']
        
        Ac_SiC = inputs['Ac_SiC']
        Ac_RPC = inputs['Ac_RPC']
        Ac_INS = inputs['Ac_INS']
        
        A_spec = inputs['A_spec']
        V_RPC = inputs['V_RPC']
        
        Tf = inputs['T_fluid']
        
        #  Back plate temperature and total radiative heat input
        outputs['T_BP'] = (Q_Solar_Net_new_1[-1,0]/(pi*r_1**2*sigma*E))**0.25
        Q_Solar_Net = Q_Solar_Net_new_1[0:-1]
        
        # Horizontal Boundary Conditions
        for i in range(1,disc_z+1):
            # Inner wall SiC
            self.a[0,i]=2*pi*dL*k_SiC*1/log(r_n[1,0]/r_n[0,0])+h_loss_cav*2*pi*r_1*dL;
            self.c[0,i]=-k_SiC*2*pi*dL*1/log(r_n[1,0]/r_n[0,0]);
            self.f[0,i]=Q_Solar_Net[i-1,0]+2*pi*r_1*dL*h_loss_cav*Tamb;
            # Boundary SiC<->RPC
            self.a[bound1-1,i]=-k_SiC*2*pi*dL*1/log(r_n[bound1-1,0]/r_n[bound1-2,0])-(k_RPC+16/(3*K_ex)*sigma*((self.T[bound1-1,i]+self.T[bound1,i])/2)**3)*2*pi*dL*1/log(r_n[bound1,0]/r_n[bound1-1,0]);
            self.b[bound1-1,i]=k_SiC*2*pi*dL*1/log(r_n[bound1-1,0]/r_n[bound1-2,0]);
            self.c[bound1-1,i]=(k_RPC+16/(3*K_ex)*sigma*((self.T[bound1-1,i]+self.T[bound1,i])/2)**3)*2*pi*dL*1/log(r_n[bound1,0]/r_n[bound1-1,0]);
            # Boundary RPC<->INS
            self.a[bound2-1,i]=-(k_RPC+16/(3*K_ex)*sigma*(((self.T[bound2-1,i]+self.T[bound2-2,i])/2)**3))*2*pi*dL*1/log(r_n[bound2-1,0]/r_n[bound2-2,0])-k_INS*2*pi*dL*1/log(r_n[bound2,0]/r_n[bound2-1,0]);
            self.b[bound2-1,i]=(k_RPC+16/(3*K_ex)*sigma*(((self.T[bound2-1,i]+self.T[bound2-2,i])/2)**3))*2*pi*dL*1/log(r_n[bound2-1,0]/r_n[bound2-2,0])
            self.c[bound2-1,i]=k_INS*2*pi*dL*1/log(r_n[bound2,0]/r_n[bound2-1,0]);
            # Outer wall INS
            self.a[Nr-1,i]=-k_INS*2*pi*dL*1/log(r_n[Nr-1,0]/r_n[Nr-2,0])-h_loss*2*pi*r_n[Nr-1,0]*dL;
            self.b[Nr-1,i]=k_INS*2*pi*dL*1/log(r_n[Nr-1,0]/r_n[Nr-2,0]);
            self.f[Nr-1,i]=-2*pi*r_n[Nr-1,0]*dL*h_loss*Tamb;

        # Vertical Boundary Conditions SiC
        count=0;
        for j in range(1,bound1-1):
            self.a[j,0] = -Ac_SiC[count,0]*k_SiC*1/(z_n[0,1]-z_n[0,0])-Ac_SiC[count,0]*k_SiC*1/0.01;
            self.a[j,Nz-1] = -Ac_SiC[count,0]*k_SiC*1/(z_n[0,Nz-1]-z_n[0,Nz-2])-h_loss_z*Ac_SiC[count,0];
            self.d[j,Nz-1] = Ac_SiC[count,0]*k_SiC*1/(z_n[0,Nz-1]-z_n[0,Nz-2]);
            self.e[j,0]=Ac_SiC[count,0]*k_SiC*1/(z_n[0,Nz-1]-z_n[0,Nz-2]);
            self.f[j,0]=-Ac_SiC[count,0]*k_SiC*1/0.01*T_corner;
            self.f[j,Nz-1]=-Ac_SiC[count,0]*h_loss_z*Tamb;
            count=count+1;
        
        # Vertical Boundary Conditions RPC
        count=0;
        for j in range(bound1,bound2-1):
            self.a[j,0]=-Ac_RPC[count,0]*k_RPC*1/(z_n[0,1]-z_n[0,0])-h_loss_z*Ac_RPC[count,0];
            self.a[j,Nz-1]=-Ac_RPC[count,0]*k_RPC*1/(z_n[0,Nz-1]-z_n[0,Nz-2])-h_loss_z*Ac_RPC[count,0];
            self.d[j,Nz-1]=Ac_RPC[count,0]*k_RPC*1/(z_n[0,Nz-1]-z_n[0,Nz-2]);
            self.e[j,0]=Ac_RPC[count,0]*k_RPC*1/(z_n[0,1]-z_n[0,0]);
            self.f[j,0]=-Ac_RPC[count,0]*h_loss_z*Tamb;
            self.f[j,Nz-1]=-Ac_RPC[count,0]*h_loss_z*Tamb;
            count=count+1;

        # Vertical Boundary Conditions INS
        count=0;
        for j in range(bound2,Nr-1):
            self.a[j,0] = -Ac_INS[count,0]*k_INS*1/(z_n[0,1]-z_n[0,0])-h_loss_z*Ac_INS[count,0];
            self.a[j,Nz-1] = -Ac_INS[count,0]*k_INS*1/(z_n[0,Nz-1]-z_n[0,Nz-2])-h_loss_z*Ac_INS[count,0];
            self.d[j,Nz-1] = Ac_INS[count,0]*k_INS*1/(z_n[0,Nz-1]-z_n[0,Nz-2]);
            self.e[j,0] = Ac_INS[count,0]*k_INS*1/(z_n[0,1]-z_n[0,0]);
            self.f[j,0] = -Ac_INS[count,0]*h_loss_z*Tamb;
            self.f[j,Nz-1] = -Ac_INS[count,0]*h_loss_z*Tamb;
            count=count+1;

        # Internal nodes SiC
        for i in range(2,disc_z+2):
            count=0;
            for j in range(2,disc_SiC+2):
                k1=k_SiC*2*pi*dL*1/log(r_n[j-1,0]/r_n[j-2,0]);
                k2=k_SiC*2*pi*dL*1/log(r_n[j,0]/r_n[j-1,0]);
                k3=k_SiC*Ac_SiC[count,0]*1/(z_n[0,i-1]-z_n[0,i-2]);
                k4=k_SiC*Ac_SiC[count,0]*1/(z_n[0,i]-z_n[0,i-1]);
                self.a[j-1,i-1]=-k1-k2-k3-k4;
                self.b[j-1,i-1]=k1;
                self.c[j-1,i-1]=k2;
                self.d[j-1,i-1]=k3;
                self.e[j-1,i-1]=k4;
                count=count+1;
                
        # Internal nodes RPC
        for i in range(2,disc_z+2):
            count=0;
            for j in range(bound1+1,bound2):
                k1=(k_RPC+16/(3*K_ex)*sigma*(((self.T[j-1,i-1]+self.T[j-2,i-1])/2)**3))*2*pi*dL*1/log(r_n[j-1,0]/r_n[j-2,0]);
                k2=(k_RPC+16/(3*K_ex)*sigma*(((self.T[j-1,i-1]+self.T[j,i-1])/2)**3))*2*pi*dL*1/log(r_n[j,0]/r_n[j-1,0]);
                k3=k_RPC*Ac_RPC[count,0]*1/(z_n[0,i-1]-z_n[0,i-2]);
                k4=k_RPC*Ac_RPC[count,0]*1/(z_n[0,i]-z_n[0,i-1]);
                k5=h_*A_spec*V_RPC[j-bound1-1,0];
                self.a[j-1,i-1]=-k1-k2-k3-k4-k5;
                self.b[j-1,i-1]=k1;
                self.c[j-1,i-1]=k2;
                self.d[j-1,i-1]=k3;
                self.e[j-1,i-1]=k4;
                self.f[j-1,i-1]=-k5*Tf[j-bound1-1,i-2];
                count=count+1;        

        # Internal nodes INS
        for i in range(2,disc_z+2):
            count=0;
            for j in range(bound2+1,Nr):
                k1=k_INS*2*pi*dL*1/log(r_n[j-1,0]/r_n[j-2,0]);
                k2=k_INS*2*pi*dL*1/log(r_n[j,0]/r_n[j-1,0]);
                k3=k_INS*Ac_INS[count,0]*1/(z_n[0,i-1]-z_n[0,i-2]);
                k4=k_INS*Ac_INS[count,0]*1/(z_n[0,i]-z_n[0,i-1]);
                self.a[j-1,i-1]=-k1-k2-k3-k4;
                self.b[j-1,i-1]=k1;
                self.c[j-1,i-1]=k2;
                self.d[j-1,i-1]=k3;
                self.e[j-1,i-1]=k4;
                count=count+1;
                
        # flat the arrays as vectors
        v_a=self.a.reshape(-1,1)
        v_b=self.b.reshape(-1,1)
        v_c=self.c.reshape(-1,1)
        v_d=self.d.reshape(-1,1)
        v_e=self.e.reshape(-1,1)
        v_f=self.f.reshape(-1,1)
        
        # Left Hand Side Matrix
        LHS = np.diagflat(v_a) + np.diagflat(v_b[disc_z+2:NN,0],-(disc_z+2)) + np.diagflat(v_c[0:NN-(disc_z+2),0],disc_z+2) + np.diagflat(v_d[1:NN],-1) + np.diagflat(v_e[0:NN-1],1)

        # Matrix Inversion
        v_T = np.matmul(np.linalg.inv(LHS),v_f)
        self.T = T_new = np.resize(v_T,(Nr,Nz))
        outputs['T'] = T_new
        # T_new = np.resize(v_T,(Nr,Nz))
        outputs['T_outer'] = T_new[-1,:]
        outputs['T_cav'] = T_new[0,:]
        outputs['T_RPC'] = T_RPC = T_new[bound1:bound2-1,1:Nz-1] ##### Burası T[bound1-1:bound2,1:Nz-1]
        outputs['T_side'] = T_new[1:bound1-1,0]
        
        # np.save('v_a.npy',v_a)
        # np.save('v_b.npy',v_b)
        # np.save('v_c.npy',v_c)
        # np.save('v_d.npy',v_d)
        # np.save('v_e.npy',v_e)
        # np.save('v_f.npy',v_f)
        # np.save('v_T.npy',v_T)    
        
        # print(f'solid.compute\nQ={Q_Solar_Net_new_1}')
        
        
if __name__ == '__main__':
    
    p = om.Problem()
    p.model.add_subsystem('solid', solid())
    p.setup()
    
    p.set_val('solid.dL', np.load('dL.npy'))
    p.set_val('solid.z_n', np.load('z_n.npy'))
    p.set_val('solid.r_n', np.load('r_n.npy'))
    p.set_val('solid.r_g', np.load('r_g.npy'))
    p.set_val('solid.h_loss_z', np.load('h_loss_z.npy'))
    p.set_val('solid.h_', np.load('h_.npy'))
    p.set_val('solid.k_RPC', np.load('k_RPC.npy'))
    p.set_val('solid.Ac_RPC', np.load('Ac_RPC.npy'))
    p.set_val('solid.Ac_INS', np.load('Ac_INS.npy'))
    p.set_val('solid.Ac_SiC', np.load('Ac_SiC.npy'))
    p.set_val('solid.V_RPC', np.load('V_RPC.npy'))
    
    p.run_model()