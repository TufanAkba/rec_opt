#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:51:14 2021

@author: tufan

Promotes all inputs and outputs,
Basic calculator for initial parameters
No iteration exists or solver required
"""

import openmdao.api as om
from math import pi, log10
import numpy as np
from Radn_fncs import MoCa_3D
import time

class initialization(om.ExplicitComponent):
    
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
        self.Nr=self.disc_r+4;
        self.Nz=disc_z+2;
        self.NN=self.Nz*self.Nr;
        
        self.bound1=disc_SiC+2;
        self.bound2=disc_SiC+2+disc_RPC+1;
        
        # print('initialization.initialize')
        
    def setup(self):
        
        disc_z = self.options['disc_z']
        disc_SiC = self.options['disc_SiC']
        disc_RPC = self.options['disc_RPC']
        disc_INS = self.options['disc_INS']
        
        self.add_input('L',val=0.065,desc='Length of the SiC tube', units='m')
        self.add_input('r_1', val = 0.015,desc='Inner radius SiC', units='m')
        self.add_input('s_SiC',val=0.005,desc='Thikness of SiC tube', units='m')
        self.add_input('s_RPC',val=0.015,desc='Thikness of RPC tube', units='m')
        self.add_input('s_INS',val=0.1,desc='Thickness Insulation', units='m')
        
        self.add_input('h_loss',val=15,desc='Heat transfer coefficient to ambient',units='W/(m**2*K)')
        
        self.add_input('k_INS',val=0.3,desc='Conductivity insulation',units='W/(m*K)')
        self.add_input('k_SiC',val=33,desc='Conductivity SiC',units='W/(m*K)')
        self.add_input('k_Air',val=0.08,desc='Conductivity air',units='W/(m*K)')
        
        self.add_input('porosity',val=0.81,desc='Porosity RPC')
        
        self.add_input('p',val=10,desc='Pressure',units='bar')
        
        self.add_input('Mass_Flow', val=0.00068,desc='Mass flow rate', units='kg/s')
        
        self.add_input('D_nom',val=0.00254,desc='Nominal pore size', units='m')
        
        self.add_input('Tamb',val=293,desc='Ambient Temperature', units='K')
        
        self.add_input('E',val=0.9,desc='Emissivity')
        
        self.add_output('dL', desc='Axial mesh length', units='m')
        
        # for radiocity
        self.add_output('B', shape=(disc_z+1,disc_z+1), desc='Coeff. matrix of the radiocity')
        
        self.add_output('F', shape=(disc_z,disc_z), desc='View factor cavity to cavity')
        self.add_output('F_I_1', shape=(1,disc_z), desc='View factor aperture to cavity')
        self.add_output('F_I_BP', desc='View factor aperture to BP')
        self.add_output('F_1_BP', shape=(1,disc_z), desc='View factor cavity to BP')
        self.add_output('F_BP_1', shape=(1,disc_z), desc='View factor BP to cavity')
        
        # for fluid
        self.add_output('h_',desc='Heat transfer coefficient inside RPC',units='W/(m**2*K)')
        self.add_output('V_RPC', shape=(disc_RPC,1), desc='Volume of each RPC element', units='m**3')
        self.add_output('m', shape=(disc_RPC,1), desc='mass flow rate passing inside each RPC element', units='kg/s')
        
        #for solid
        self.add_output('r_n', shape=(self.Nr,1), units='m', desc='radial node coordinates')
        self.add_output('r_g', shape=(self.Nr-3,1), units='m', desc='radial grid coordinates')
        self.add_output('z_n', shape=(1,disc_z+2), units='m', desc='axial grid coordinates')
        
        self.add_output('h_loss_cav', units='W/(m**2*K)', desc='heat transfer coefficient from cavity')
        self.add_output('h_loss_z', units='W/(m**2*K)', desc='heat transfer coefficient on vertical surfaces')
        
        self.add_output('k_RPC', desc='Conductivity RPC',units='W/(m*K)')
        
        self.add_output('Ac_SiC', shape=(disc_SiC,1), units='m**2', desc='Cross sectional area of SiC elements')
        self.add_output('Ac_RPC', shape=(disc_RPC,1), units='m**2', desc='Cross sectional area of RPC elements')
        self.add_output('Ac_INS', shape=(disc_INS,1), units='m**2', desc='Cross sectional area of INS elements')
        self.add_output('Volume', val=0.0037216091972588094, units='m**3')
        
        self.declare_partials(of='*', wrt='*',method='fd')
        self.linear_solver = om.LinearRunOnce()
        
        # print('initialization.setup')
        
    def compute(self, inputs, outputs):
        
        # in fucntion param.s
        disc_z = self.options['disc_z']
        disc_SiC = self.options['disc_SiC']
        disc_RPC = self.options['disc_RPC']
        disc_INS = self.options['disc_INS']
        
        disc_r = self.disc_r
        Nr = self.Nr
        # Nz = self.Nz
        # NN = self.NN
        
        r_1 = inputs['r_1']
        s_SiC = inputs['s_SiC']
        s_RPC = inputs['s_RPC']
        s_INS = inputs['s_INS']
        L = inputs['L']
        outputs['Volume'] =  pi*(r_1+s_SiC+s_RPC+s_INS)**2*L
        
        h_loss = inputs['h_loss']
        
        k_INS = inputs['k_INS']
        k_SiC = inputs['k_SiC']
        k_Air = inputs['k_Air']
        
        porosity = inputs['porosity']
        
        p = inputs['p']
        
        Mass_Flow =abs(inputs['Mass_Flow'])
        
        D_nom = inputs['D_nom']
        
        Tamb = inputs['Tamb']
        
        E = inputs['E']
        
        # Grid limits
        r_2=r_1+s_SiC;
        r_3=r_2+s_RPC;
        r_4=r_3+s_INS;
        outputs['dL'] = dL = L/disc_z;
        dr_SiC=(r_2-r_1)/disc_SiC;
        dr_RPC=(r_3-r_2)/disc_RPC;
        dr_INS=(r_4-r_3)/disc_INS;
        # bound1=self.bound1
        # bound2=self.bound2
        
        # vertical convection loss
        outputs['h_loss_z'] = 1/(1/h_loss+s_INS/k_INS);
        
        # Effective Heat Conduction Coefficient
        v=k_Air/k_SiC;
        f=0.3823;
        mu=f*(porosity*v+(1-porosity))+(1-f)*(porosity/v+1-porosity)**(-1);
        outputs['k_RPC'] = k_SiC*mu;        
        
        # Average Heat Transfer Coefficient in RPC
        d0=1.5590;
        d1=0.5954;
        d2=0.5626;
        d3=0.4720;
        Rho_IN=p*10**5/(287*298);
        u_IN=Mass_Flow/(Rho_IN*pi*(r_3**2-r_2**2));
        Re_IN=u_IN*D_nom/0.00001595283450644;
        Nu_IN=d0+d1*Re_IN**d2*0.67**d3;
        h_IN=Nu_IN*k_Air/D_nom;
        Rho_OUT=p*10**5/(287*1473);
        u_OUT=Mass_Flow/(Rho_OUT*pi*(r_3**2-r_2**2));
        Re_OUT=u_OUT*D_nom/0.00020511715999999998;
        Nu_OUT=d0+d1*Re_OUT**d2*0.67**d3;
        h_OUT=Nu_OUT*k_Air/D_nom;
        outputs['h_'] = (h_IN+h_OUT)/2;
        
        # Calculation of the internal loss coefficient
        T_ave=(1473+Tamb)/2; #Burayı Kontrol Etmek Lazım!
        Pr=0.68;
        dvis=1.458*10**-6*T_ave**1.5/(T_ave+110.4);
        dens=10**6/(287*T_ave);  #Burayı Kontrol Et
        kvis=dvis/dens;
        Gr=9.81*1/T_ave*r_1**3*(1473-293)/kvis**2;
        Ra=Gr*Pr;
        T0t=r_1*Ra/L;
        Nu_loss_cav=(10**(1.256*log10(T0t)-0.343))/(T0t);
        h_loss_cav=Nu_loss_cav*k_Air/(r_1);
        # !!! #
        h_loss_cav=0; #h_loss_cav is set to zero!
        outputs['h_loss_cav'] = h_loss_cav
        
        #Calculation of z-coordinates
        z_n = np.empty((1,disc_z+2),dtype=float)
        z_n[0,0]=0;
        
        for i in range(1,disc_z+1):
            z_n[0,i]=z_n[0,0]+dL/2+(i-1)*dL;
        
        z_n[0,disc_z+1]=L;
        
        #Calculation of r-coordinates (n=nodal, g=grid)
        r_n = np.ones((Nr,1),dtype=float)
        r_g = np.ones((Nr-3,1),dtype=float)
        r_n[0,0]=r_1;
        
        for i in range(1,disc_SiC+1):
            r_n[i,0] = r_n[0]+dr_SiC/2+(i-1)*dr_SiC;
            r_g[i-1,0]=r_1+(i-1)*dr_SiC;
        
        r_n[disc_SiC+1,0]=r_2;
        r_g[disc_SiC,0]=r_2;
        
        for i in range(disc_SiC+2,(disc_SiC+2+disc_RPC)):
            r_n[i,0]=r_2+dr_RPC/2+(i-(disc_SiC+2))*dr_RPC;
            r_g[i-1,0]=r_2+(i-(disc_SiC+1))*dr_RPC;
        
        r_n[disc_SiC+2+disc_RPC,0]=r_3;
        
        for i in range (disc_SiC+2+disc_RPC+1,Nr-1):
            r_n[i,0]=r_3+dr_INS/2+(i-(disc_SiC+2+disc_RPC+1))*dr_INS;
            if i-1<=disc_r+1:
                r_g[i-2,0]=r_3+(i-(disc_SiC+disc_RPC+2))*dr_INS;
        
        r_n[Nr-1,0]=r_4;
        
        outputs['z_n'] = z_n
        outputs['r_n'] = r_n
        outputs['r_g'] = r_g
        
        # Calculation of cross-sectional areas
        Ac_SiC = np.empty((disc_SiC,1),dtype=float)
        for i in range (1,disc_SiC+1):
            Ac_SiC[i-1,0]=pi*(r_g[i]**2-r_g[i-1]**2);
        
        Ac_RPC = np.empty((disc_RPC,1),dtype=float)
        V_RPC = np.empty((disc_RPC,1),dtype=float)
        m = np.empty((disc_RPC,1),dtype=float)
        for i in range(1,disc_RPC+1):
            Ac_RPC[i-1,0]=pi*(r_g[disc_SiC+i]**2-r_g[disc_SiC+i-1]**2);
            V_RPC[i-1,0]=dL*Ac_RPC[i-1,0];
            m[i-1,0]=Mass_Flow*Ac_RPC[i-1,0]/(pi*(r_3**2-r_2**2));
            
        outputs['V_RPC'] = V_RPC
        outputs['m'] = m
            
        Ac_INS = np.empty((disc_INS,1),dtype=float)
        for i in range(1,disc_INS+1):
            Ac_INS[i-1,0]=pi*(r_g[disc_SiC+disc_RPC+i]**2-r_g[disc_SiC+disc_RPC+i-1]**2);
            
        outputs['Ac_SiC'] = Ac_SiC
        outputs['Ac_RPC'] = Ac_RPC
        outputs['Ac_INS'] = Ac_INS
        
        # Calculation of the Configuration Factors
        F_temp = np.empty((1,disc_z-1),dtype=float)
        
        for i in range(1,disc_z):
            L1=dL/r_1;
            L2=(dL+(i-1)*dL)/r_1;
            L3=(2*dL+(i-1)*dL)/r_1;
            X1=((L3-L1)**2+4)**0.5;
            X2=((L2-L1)**2+4)**0.5;
            X3=((L3)**2+4)**0.5;
            X4=((L2)**2+4)**0.5;
            F_temp[0,i-1]=1/(4*(L3-L2))*(2*L1*(L3-L2)+(L3-L1)*X1-(L2-L1)*X2-L3*X3+L2*X4);
        
        F_1_1=(1+dL/(2*r_1))-(1+(dL/(2*r_1))**2)**(0.5);
        F=np.zeros((disc_z,disc_z),dtype=float)
        np.fill_diagonal(F,F_1_1)
        
        for i in range(1,disc_z+1):
            w=i+1;
            e=1;
            neg=1;
            for j in range(1,disc_z):
                if w>disc_z:
                    w=1;
                    e=disc_z-j;
                    neg=-1;
                F[i-1,w-1]=F_temp[0,e-1];
                w=w+1;
                e=e+1*neg;
        
        del neg, e, w, F_1_1, F_temp
        
        F_1_I = np.empty((1,disc_z),dtype=float)
        F_1_BP = np.empty((1,disc_z),dtype=float)
        for i in range(1,disc_z+1):
            j=disc_z-i+1;
            H1=dL/r_1;
            H2=(i-1)*dL/r_1;
            F_1_I[0,i-1]=0.25*((1+H2/H1)*(4+(H1+H2)**2)**0.5-(H1+2*H2)-H2/H1*(4+H2**2)**0.5);
            F_1_BP[0,j-1]=0.25*((1+H2/H1)*(4+(H1+H2)**2)**0.5-(H1+2*H2)-H2/H1*(4+H2**2)**0.5);
        
        F_BP_1=2*pi*r_1*dL/(pi*r_1**2)*np.copy(F_1_BP);
        F_1_1=F[0,0];
        
        # Radiosity Matrix Calculation
        B = np.ones((disc_z,disc_z),dtype=float)*-F*(1-E)/E;
        np.fill_diagonal(B,1/E-F_1_1*(1-E)/E)
        B = np.append(B,(-F_BP_1*(1-E)/E),axis=0)
        B = np.append(B,np.zeros((disc_z+1,1),dtype=float),axis=1)
        B[0:disc_z,-1] = (-F_1_BP*(1-E)/E)
        B[-1,-1] = 1/E
        
        outputs['B'] = B
        
        # print('Monte Carlo Calculation running...')
        F_I_1, F_I_BP = MoCa_3D(disc_z=disc_z,r_1=r_1,L=L,dL=dL)
        # print('Finished with Monte Carlo!')

        outputs['F'] = F
        outputs['F_1_BP'] = F_1_BP  
        outputs['F_BP_1'] = F_BP_1
        outputs['F_I_1'] = F_I_1
        outputs['F_I_BP'] = F_I_BP

        # print('initialization.compute')

if __name__ == "__main__":
    
    tic = time.time()
    
    p = om.Problem()
    p.model.add_subsystem('init', initialization(disc_z = 20, disc_SiC = 10, disc_RPC=20, disc_INS = 10))
    p.setup()
    
    p.run_model()
    
    # dL=p.get_val('init.dL')
    # B=p.get_val('init.B')
    # F=p.get_val('init.F')
    # F_I_1=p.get_val('init.F_I_1')
    # F_I_BP=p.get_val('init.F_I_BP')
    # F_1_BP=p.get_val('init.F_1_BP')
    # F_BP_1=p.get_val('init.F_BP_1')
    # h_ = p.get_val('init.h_')
    # V_RPC = p.get_val('init.V_RPC')
    # m = p.get_val('init.m')
    # r_n = p.get_val('init.r_n')
    # r_g = p.get_val('init.r_g')
    # z_n = p.get_val('init.z_n')
    # h_loss_cav = p.get_val('init.h_loss_cav')
    # h_loss_z = p.get_val('init.h_loss_z')
    # k_RPC = p.get_val('init.k_RPC')
    # Ac_SiC = p.get_val('init.Ac_SiC')
    # Ac_RPC = p.get_val('init.Ac_RPC')
    # Ac_INS = p.get_val('init.Ac_INS')
    
    
    print('Elapsed time is', time.time()-tic, 'seconds', sep=None)