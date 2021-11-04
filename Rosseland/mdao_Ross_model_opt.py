#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:45:32 2021

Recreate version of the Rosseland approximation receiver model for OpenMDAO

@author: tufan
"""

import numpy as np
import openmdao.api as om
from math import pi, log, exp, log10
from Radn_fncs import MoCa_3D
from draw_contour import draw_contour

import time



class mdao_Ross_model(om.ExplicitComponent):
    
    
    """
    Revised version of the Hess's Rosseland model for OpenMDAO models.
    For the final version of the model, solver will be a seperate file.
    
    L=0.065;                                %Length of the SiC tube [m]
    disc_z=20;                              %Discretization in z-direction
    disc_SiC=10;                            %SiC-Discretization in r-direction
    disc_RPC=20;                            %RPC-Discretization in r-direction
    disc_INS=10;                            %INS-Discretization in r-direction
    Tamb=20;                                %Ambient Temperature [�C]
    sigma=5.67*10^(-8);                     %Boltzmann Constant [W/(m^2*K^4)]
    Mass_Flow=0.00068;                      %Mass flow rate [kg/s]
    cp=1005;                                %Specific Heat Capacity [J/(kg*K)]
    T_Inlet=20;                             %Inlet Temperature [�C]
    h_loss=15;                              %Heat transfer coefficient to ambient [W/(m^2*K)]
    Q_Solar_In=1000;                        %Total Solar Power Input [W]
    p=10;                                   %Pressure [bar]
    k_Air=0.08;                             %Conductivity air [W/(m*K)]
    T_corner=925; %T_corner disabled!                          %Corner Temperature [�C]
    
    % SiC
    r_1=0.015;                              %Inner radius SiC [m]
    s_SiC=0.005;                            %Thikness of SiC tube [m]
    k_SiC=33;                               %Conductivity SiC [W/(m*K)]
    E=0.9;                                  %Emissivity []
    
    % RPC
    s_RPC=0.015;                            %Thickness RPC [m]
    K_ex=200;                               %Extinction Coefficient [m^-1]
    porosity=0.81;                          %Porosity RPC []
    A_spec=500;                             %Specific Surface of the RPC [m^2/m^3]
    D_nom=0.00254;                          %Nominal pore size [m]
    
    % Insulation
    s_INS=0.1;                              %Thickness Insulation [m]
    k_INS=0.3;                              %Conductivity insulation [W/(m*K)]
    
    
    
    """
    
    def initialize(self):
        
        
        "Discretization param.s"
        self.options.declare('disc_z', types=int,default=20,desc='Discretization in z-direction')
        self.options.declare('disc_SiC', types=int,default=10,desc='SiC-Discretization in r-direction')
        self.options.declare('disc_RPC', types=int,default=20,desc='RPC-Discretization in r-direction')
        self.options.declare('disc_INS', types=int,default=10,desc='INS-Discretization in r-direction')
        
        
    def setup(self):
        
        # self.add_input('L',val=0.065,desc='Length of the SiC tube', units='m')
        self.add_input('Mass_Flow', val=0.00068 ,desc='Mass flow rate', units='kg/s')
        # self.add_input('Q_Solar_In',val=1000,desc='Total Solar Power Input',units='W')
        # self.add_input('T_corner',val=1198,desc='Corner Temperature',units='K')
        
        # SiC
        # self.add_input('r_1',desc='Inner radius SiC', units='m')
        # self.add_input('s_SiC',val=0.005,desc='Thikness of SiC tube', units='m')
        
        # RPC
        # self.add_input('s_RPC',val=0.015,desc='Thikness of RPC tube', units='m')
        
        # Insulation
        # self.add_input('s_INS',val=0.1,desc='Thickness Insulation', units='m')
        
        # self.add_input('omega',val=0.2,desc='Scattering Albedo')
        # self.add_input('porosity',val=0.81,desc='Porosity RPC')
        
        # self.add_input('sigma',val=5.67*10**(-8),desc='Stefan-Boltzman Const.',units='W/(m**2*K**4)')
        # self.add_input('cp',val=1005,desc='Specific Heat Capacity',units='J/(kg*K)')
        # self.add_input('k_Air',val=0.08,desc='Conductivity air',units='W/(m*K)')
        
        # self.add_input('Tamb',val=293,desc='Ambient Temperature', units='K')
        # self.add_input('p',val=10,desc='Pressure',units='bar')
        # self.add_input('T_Inlet',val=293,desc='Inlet Temperature',units='K')
        
        # self.add_input('h_loss',val=15,desc='Heat transfer coefficient to ambient',units='W/(m**2*K)')
        # self.add_input('k_SiC',val=33,desc='Conductivity SiC',units='W/(m*K)')
        # self.add_input('E',val=0.9,desc='Emissivity')
        # self.add_input('K_ex',val=200,desc='Extinction Coefficient',units='m**-1')
        # self.add_input('A_spec',val=500,desc='Specific Surface of the RPC',units='m**-1')
        # self.add_input('D_nom',val=0.00254,desc='Nominal pore size', units='m')
        # self.add_input('k_INS',val=0.3,desc='Conductivity insulation',units='W/(m*K)')
        # self.add_input('E_INS',val=0.5,desc='Emissivity of the Insulation')
        
        #  define outputs

        self.add_output('T_OUT_m', val=0, desc='Temperature of outlet fluid', units='K')
        # self.add_output('eff_abs', val=0, desc='Efficiency of absorbed solar energy')
        self.add_output('eff_S2G', val=0, desc='Efficiency of solar to gas')
        self.add_output('T_Ins', val=0, desc='Min. Temp. of Insulator', units='K')
        
        # self.add_output('P_out', val=0, desc='Outlet Power',units='W')

        # sizing outputs
        # self.add_output('V', val=0, desc='Volume of the receiver',units='m**3')        
        # disc_z = self.options['disc_z']
        # disc_SiC = self.options['disc_SiC']
        # disc_RPC = self.options['disc_RPC']
        # disc_INS = self.options['disc_INS']
        # self.add_output('T', shape=(disc_SiC+disc_RPC+disc_INS+4,disc_z+2))
        
        self.declare_partials(of='*', wrt='*',method='fd')
        self.linear_solver = om.ScipyKrylov()
        
    # def setup_partials(self):
    # # Finite difference all partials.
    #     # self.declare_partials('*', '*', method='fd')
    # # Complex Step all partials.
    #     self.declare_partials('*', '*', method='cs')
    #     # self.declare_coloring(wrt='*',method='cs')#,show_sparsity=True,show_summary=True)
        
        
    def compute(self, inputs, outputs):
        
        disc_z = self.options['disc_z']
        disc_SiC = self.options['disc_SiC']
        disc_RPC = self.options['disc_RPC']
        disc_INS = self.options['disc_INS']
        
        
        disc_r=disc_SiC+disc_RPC+disc_INS;
        Nr=disc_r+4;
        Nz=disc_z+2;
        NN=Nz*Nr;
        
        # porosity = inputs['porosity']
        porosity = 0.81
        
        
        # Tamb = inputs['Tamb'].real
        Tamb = 293        
        
        
        # Inputs
        # L = inputs['L'].real
        L = 0.065
        
        # sigma = inputs['sigma']
        sigma = 5.67*10**(-8)
        Mass_Flow = inputs['Mass_Flow'].real
        # Mass_Flow = 0.00068
        # cp = inputs['cp']
        cp = 1005
        # T_Inlet = inputs['T_Inlet'].real
        T_Inlet = 293
        
        
        
        # h_loss = inputs['h_loss']
        h_loss = 15
        # p = inputs['p']
        p = 10
        # k_Air = inputs['k_Air']
        k_Air = 0.08
        # T_corner = inputs['T_corner']
        T_corner = 925;
        
        # SiC
        # r_1 = inputs['r_1'].real
        r_1 = 0.015
        
        # Q_Solar_In = (inputs['Q_Solar_In'].real)*(r_1/0.015)**2#initial r_1
        Q_Solar_In = 1000
        
        
        # s_SiC = inputs['s_SiC'].real
        s_SiC = 0.005
        # k_SiC = inputs['k_SiC']
        k_SiC = 33
        # E = inputs['E']
        E = 0.9
        
        # RPC
        # s_RPC = inputs['s_RPC'].real
        s_RPC = 0.015
        # K_ex = inputs['K_ex']
        K_ex = 200
        # A_spec = inputs['A_spec']
        A_spec = 500
        # D_nom = inputs['D_nom']
        D_nom = 0.00254
        
        # Insulation
        # s_INS = inputs['s_INS'].real
        s_INS = 0.1
        # k_INS = inputs['k_INS']
        # E_INS = inputs['E_INS']
        k_INS = 0.3
        # E_INS = 0.5
        
        
        
        r_2=r_1+s_SiC;
        r_3=r_2+s_RPC;
        r_4=r_3+s_INS;
        dL=L/disc_z;
        dr_SiC=(r_2-r_1)/disc_SiC;
        dr_RPC=(r_3-r_2)/disc_RPC;
        dr_INS=(r_4-r_3)/disc_INS;
        h_loss_z=1/(1/h_loss+s_INS/k_INS);
        bound1=disc_SiC+2;
        bound2=disc_SiC+2+disc_RPC+1;
        
        # Effective Heat Conduction Coefficient
        v=k_Air/k_SiC;
        f=0.3823;
        mu=f*(porosity*v+(1-porosity))+(1-f)*(porosity/v+1-porosity)**(-1);
        k_RPC=k_SiC*mu;
        # k_RPC=0.75*k_RPC;
        
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
        h_=(h_IN+h_OUT)/2;
        
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
        
        h_loss_cav=0; #h_loss_cav is temporary set to zero!
        
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
        
        print('Monte Carlo Calculation running...')
        F_I_1, F_I_BP = MoCa_3D(disc_z=disc_z,r_1=r_1,L=L,dL=dL)
        
        print('Finished with Monte Carlo!')
        print('Iteration started...')
        
        z_n = np.empty((1,disc_z+2),dtype=float)
        #Calculation of z-coordinates
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
            
        

        
        Ac_INS = np.empty((disc_INS,1),dtype=float)
        for i in range(1,disc_INS+1):
            Ac_INS[i-1,0]=pi*(r_g[disc_SiC+disc_RPC+i]**2-r_g[disc_SiC+disc_RPC+i-1]**2);
        
        #In. Val.s
        a = np.zeros((Nr,Nz),dtype=float)
        b = np.zeros((Nr,Nz),dtype=float)
        c = np.zeros((Nr,Nz),dtype=float)
        d = np.zeros((Nr,Nz),dtype=float)
        e = np.zeros((Nr,Nz),dtype=float)
        f = np.zeros((Nr,Nz),dtype=float)
        Q_Solar_Net = 0.85*Q_Solar_In/disc_z*np.ones((disc_z,1),dtype=float)
        T_ini = 20+273
        v_T = np.ones((NN,1),dtype=float)*T_ini
        T = np.ones((Nr,Nz),dtype=float)*T_ini
        Tf = np.ones((disc_RPC,disc_z),dtype=float)*T_ini
        T_OUT = np.ones((disc_RPC,1),dtype=float)*(273+T_Inlet)
        T_IN = np.ones((disc_RPC,1),dtype=float)*(273+T_Inlet)
        T_RPC = np.ones((disc_RPC+2,disc_z),dtype=float)*T_ini
        T_BP = T_ini



        # Radiosity Matrix Calculation
        B = np.ones((disc_z,disc_z),dtype=float)*-F*(1-E)/E;
        np.fill_diagonal(B,1/E-F_1_1*(1-E)/E)
        # for i in range(1,disc_z):
        #     for j in range(1,disc_z):
        #         if i==j:
        #             B[i-1,j-1]=1/E-F_1_1*(1-E)/E;
        B = np.append(B,(-F_BP_1*(1-E)/E),axis=0)
        B = np.append(B,np.zeros((disc_z+1,1),dtype=float),axis=1)
        B[0:disc_z,-1] = (-F_1_BP*(1-E)/E)
        B[-1,-1] = 1/E

    
        # Coefficient Matrices (Edges)
        a[0,0]=1;
        c[0,0]=-0.5;
        e[0,0]=-0.5;
        a[0,Nz-1]=1;
        c[0,Nz-1]=-0.5;
        d[0,Nz-1]=-0.5;
        a[Nr-1,0]=1;
        b[Nr-1,0]=-0.5;
        e[Nr-1,0]=-0.5;
        a[Nr-1,Nz-1]=1;
        b[Nr-1,Nz-1]=-0.5;
        d[Nr-1,Nz-1]=-0.5;
    
        #Coefficient Matrices (Boundary Edges)
        a[bound1-1,0]=1;
        a[bound2-1,0]=1;
        a[bound1-1,Nz-1]=1;
        a[bound2-1,Nz-1]=1;
        b[bound1-1,0]=-0.5;
        b[bound2-1,0]=-0.5;
        b[bound1-1,Nz-1]=-0.5;
        b[bound2-1,Nz-1]=-0.5;
        c[bound1-1,0]=-0.5;
        c[bound2-1,0]=-0.5;
        c[bound1-1,Nz-1]=-0.5;
        c[bound2-1,Nz-1]=-0.5;
    
    
        #Iter.s
        itnumb=1;
        dQ=100;
        residual=1
        
        
        # while itnumb<=2:
        while abs(dQ)>=0.1:        
        
            K = np.zeros((disc_z+1,1),dtype=float)
            
            

            for i in range(1,disc_z+1):
            #     # K(i,1)=(-Q_Solar_In*F_I_1(i)/(2*pi*r_1*dL)+sigma*T(1,i+1)^4-sum(sigma*T(1,2:Nz-1).^4.*F(i,:))-sigma*T_BP^4*F_1_BP(i));
                K[i-1,0] = -Q_Solar_In*F_I_1[0,i-1]/(2*pi*r_1*dL)+sigma*T[0,i]**4-sum(sigma*np.multiply(np.power(T[0,1:Nz-1],4),F[i-1,:]))-sigma*T_BP**4*F_1_BP[0,i-1]
                
                
                
            K[disc_z,0] = -Q_Solar_In*F_I_BP/(pi*r_1**2)+sigma*T_BP**4-sum(sigma*np.multiply(np.power(T[0,1:Nz-1],4),F_BP_1[0,:]))
            Q_Solar_Net_new_1=np.matmul(np.linalg.inv(B),(-K))
            
            Q_Solar_BP = pi*r_1**2*Q_Solar_Net_new_1[-1,0];
            # Q_Solar_Net_new = np.empty((len(Q_Solar_Net_new_1[0:-1,0]),1),dtype=float)
            Q_Solar_Net_new = (Q_Solar_Net_new_1[0:-1,0]*2*pi*r_1*dL).reshape(1,len(Q_Solar_Net_new_1[0:-1,0]))
            
            # Backplate temperature
            T_BP_new=(Q_Solar_BP/(pi*r_1**2*sigma*E))**0.25;
            
            # Horizontal Boundary Conditions
            for i in range(1,disc_z+1):
                # Inner wall SiC
                a[0,i]=2*pi*dL*k_SiC*1/log(r_n[1,0]/r_n[0,0])+h_loss_cav*2*pi*r_1*dL;
                c[0,i]=-k_SiC*2*pi*dL*1/log(r_n[1,0]/r_n[0,0]);
                f[0,i]=Q_Solar_Net[i-1,0]+2*pi*r_1*dL*h_loss_cav*Tamb;
                # Boundary SiC<->RPC
                a[bound1-1,i]=-k_SiC*2*pi*dL*1/log(r_n[bound1-1,0]/r_n[bound1-2,0])-(k_RPC+16/(3*K_ex)*sigma*((T[bound1-1,i]+T[bound1,i])/2)**3)*2*pi*dL*1/log(r_n[bound1,0]/r_n[bound1-1,0]);
                b[bound1-1,i]=k_SiC*2*pi*dL*1/log(r_n[bound1-1,0]/r_n[bound1-2,0]);
                c[bound1-1,i]=(k_RPC+16/(3*K_ex)*sigma*((T[bound1-1,i]+T[bound1,i])/2)**3)*2*pi*dL*1/log(r_n[bound1,0]/r_n[bound1-1,0]);
                # f[bound1-1,i]=-1/(3*K_ex)*(G[1,i-1]-G[0,i-1])/log(r_n[bound1,0]/r_n[bound1-1,0])*2*pi*dL;
                # Boundary RPC<->INS
                a[bound2-1,i]=-(k_RPC+16/(3*K_ex)*sigma*(((T[bound2-1,i]+T[bound2-2,i])/2)**3))*2*pi*dL*1/log(r_n[bound2-1,0]/r_n[bound2-2,0])-k_INS*2*pi*dL*1/log(r_n[bound2,0]/r_n[bound2-1,0]);
                b[bound2-1,i]=(k_RPC+16/(3*K_ex)*sigma*(((T[bound2-1,i]+T[bound2-2,i])/2)**3))*2*pi*dL*1/log(r_n[bound2-1,0]/r_n[bound2-2,0])
                c[bound2-1,i]=k_INS*2*pi*dL*1/log(r_n[bound2,0]/r_n[bound2-1,0]);
                # f[bound2-1,i]=1/(3*K_ex)*(G[disc_RPC+1,i-1]-G[disc_RPC,i-1])/log(r_n[bound2-1,0]/r_n[bound2-2,0])*2*pi*dL;
                # Outer wall INS
                a[Nr-1,i]=-k_INS*2*pi*dL*1/log(r_n[Nr-1,0]/r_n[Nr-2,0])-h_loss*2*pi*r_n[Nr-1,0]*dL;
                b[Nr-1,i]=k_INS*2*pi*dL*1/log(r_n[Nr-1,0]/r_n[Nr-2,0]);
                f[Nr-1,i]=-2*pi*r_n[Nr-1,0]*dL*h_loss*Tamb;
        
        
            # Vertical Boundary Conditions SiC
            count=0;
            for j in range(1,bound1-1):
                a[j,0] = -Ac_SiC[count,0]*k_SiC*1/(z_n[0,1]-z_n[0,0])-Ac_SiC[count,0]*k_SiC*1/0.01;
                a[j,Nz-1] = -Ac_SiC[count,0]*k_SiC*1/(z_n[0,Nz-1]-z_n[0,Nz-2])-h_loss_z*Ac_SiC[count,0];
                d[j,Nz-1] = Ac_SiC[count,0]*k_SiC*1/(z_n[0,Nz-1]-z_n[0,Nz-2]);
                e[j,0]=Ac_SiC[count,0]*k_SiC*1/(z_n[0,Nz-1]-z_n[0,Nz-2]);
                f[j,0]=-Ac_SiC[count,0]*k_SiC*1/0.01*T_corner;
                f[j,Nz-1]=-Ac_SiC[count,0]*h_loss_z*Tamb;
                count=count+1;
                
            # Vertical Boundary Conditions RPC
            count=0;
            for j in range(bound1,bound2-1):
                a[j,0]=-Ac_RPC[count,0]*k_RPC*1/(z_n[0,1]-z_n[0,0])-h_loss_z*Ac_RPC[count,0];
                a[j,Nz-1]=-Ac_RPC[count,0]*k_RPC*1/(z_n[0,Nz-1]-z_n[0,Nz-2])-h_loss_z*Ac_RPC[count,0];
                d[j,Nz-1]=Ac_RPC[count,0]*k_RPC*1/(z_n[0,Nz-1]-z_n[0,Nz-2]);
                e[j,0]=Ac_RPC[count,0]*k_RPC*1/(z_n[0,1]-z_n[0,0]);
                f[j,0]=-Ac_RPC[count,0]*h_loss_z*Tamb;
                f[j,Nz-1]=-Ac_RPC[count,0]*h_loss_z*Tamb;
                count=count+1;
                

                
            # Vertical Boundary Conditions INS
            count=0;
            for j in range(bound2,Nr-1):
                a[j,0] = -Ac_INS[count,0]*k_INS*1/(z_n[0,1]-z_n[0,0])-h_loss_z*Ac_INS[count,0];
                a[j,Nz-1] = -Ac_INS[count,0]*k_INS*1/(z_n[0,Nz-1]-z_n[0,Nz-2])-h_loss_z*Ac_INS[count,0];
                d[j,Nz-1] = Ac_INS[count,0]*k_INS*1/(z_n[0,Nz-1]-z_n[0,Nz-2]);
                e[j,0] = Ac_INS[count,0]*k_INS*1/(z_n[0,1]-z_n[0,0]);
                f[j,0] = -Ac_INS[count,0]*h_loss_z*Tamb;
                f[j,Nz-1] = -Ac_INS[count,0]*h_loss_z*Tamb;
                count=count+1;
    
            # Internal nodes SiC
            for i in range(2,disc_z+2):
                count=0;
                for j in range(2,disc_SiC+2):
                    k1=k_SiC*2*pi*dL*1/log(r_n[j-1,0]/r_n[j-2,0]);
                    k2=k_SiC*2*pi*dL*1/log(r_n[j,0]/r_n[j-1,0]);
                    k3=k_SiC*Ac_SiC[count,0]*1/(z_n[0,i-1]-z_n[0,i-2]);
                    k4=k_SiC*Ac_SiC[count,0]*1/(z_n[0,i]-z_n[0,i-1]);
                    a[j-1,i-1]=-k1-k2-k3-k4;
                    b[j-1,i-1]=k1;
                    c[j-1,i-1]=k2;
                    d[j-1,i-1]=k3;
                    e[j-1,i-1]=k4;
                    count=count+1;
                    
            # Internal nodes RPC
            for i in range(2,disc_z+2):
                count=0;
                for j in range(bound1+1,bound2):
                    k1=(k_RPC+16/(3*K_ex)*sigma*(((T[j-1,i-1]+T[j-2,i-1])/2)**3))*2*pi*dL*1/log(r_n[j-1,0]/r_n[j-2,0]);
                    k2=(k_RPC+16/(3*K_ex)*sigma*(((T[j-1,i-1]+T[j,i-1])/2)**3))*2*pi*dL*1/log(r_n[j,0]/r_n[j-1,0]);
                    k3=k_RPC*Ac_RPC[count,0]*1/(z_n[0,i-1]-z_n[0,i-2]);
                    k4=k_RPC*Ac_RPC[count,0]*1/(z_n[0,i]-z_n[0,i-1]);
                    k5=h_*A_spec*V_RPC[j-bound1-1,0];
                    a[j-1,i-1]=-k1-k2-k3-k4-k5;
                    b[j-1,i-1]=k1;
                    c[j-1,i-1]=k2;
                    d[j-1,i-1]=k3;
                    e[j-1,i-1]=k4;
                    f[j-1,i-1]=-k5*Tf[j-bound1-1,i-2];
                    count=count+1;        

            # Internal nodes INS
            for i in range(2,disc_z+2):
                count=0;
                for j in range(bound2+1,Nr):
                    k1=k_INS*2*pi*dL*1/log(r_n[j-1,0]/r_n[j-2,0]);
                    k2=k_INS*2*pi*dL*1/log(r_n[j,0]/r_n[j-1,0]);
                    k3=k_INS*Ac_INS[count,0]*1/(z_n[0,i-1]-z_n[0,i-2]);
                    k4=k_INS*Ac_INS[count,0]*1/(z_n[0,i]-z_n[0,i-1]);
                    a[j-1,i-1]=-k1-k2-k3-k4;
                    b[j-1,i-1]=k1;
                    c[j-1,i-1]=k2;
                    d[j-1,i-1]=k3;
                    e[j-1,i-1]=k4;
                    count=count+1;            
        
        
            # flat the arrays as vectors
            v_a=a.reshape(-1,1)
            v_b=b.reshape(-1,1)
            v_c=c.reshape(-1,1)
            v_d=d.reshape(-1,1)
            v_e=e.reshape(-1,1)
            v_f=f.reshape(-1,1)
            
            

    
            # Left Hand Side Matrix
            LHS = np.diagflat(v_a) + np.diagflat(v_b[disc_z+2:NN,0],-(disc_z+2)) + np.diagflat(v_c[0:NN-(disc_z+2),0],disc_z+2) + np.diagflat(v_d[1:NN],-1) + np.diagflat(v_e[0:NN-1],1)

            # Matrix Inversion
            v_T = np.matmul(np.linalg.inv(LHS),v_f)
            T_new = np.resize(v_T,(Nr,Nz))
            T_RPC = T[bound1:bound2-1,1:Nz-1]
            
            for i in range(1,disc_z+1):
                for j in range(1,disc_RPC+1):
                    T_OUT[j-1,0]=(2*h_*A_spec*T_RPC[j-1,i-1]*V_RPC[j-1,0]-T_IN[j-1,0]*(h_*A_spec*V_RPC[j-1,0]-2*m[j-1,0]*cp))/(h_*A_spec*V_RPC[j-1,0]+2*m[j-1,0]*cp);
                    Tf[j-1,i-1]=(T_IN[j-1,0]+T_OUT[j-1,0])/2;
                
                T_IN=np.copy(T_OUT);
            
            # Previous .m file format
            # T_RPC = T[bound1-1:bound2,1:Nz-1]
            
            # for i in range(1,disc_z+1):
            #     for j in range(1,disc_RPC+1):
            #         T_OUT[j-1,0]=(2*h_*A_spec*T_RPC[j,i-1]*V_RPC[j-1,0]-T_IN[j-1,0]*(h_*A_spec*V_RPC[j-1,0]-2*m[j-1,0]*cp))/(h_*A_spec*V_RPC[j-1,0]+2*m[j-1,0]*cp);
            #         Tf[j-1,i-1]=(T_IN[j-1,0]+T_OUT[j-1,0])/2;
                
            #     T_IN=np.copy(T_OUT);
        
            #Relation
            SOR=0.9*exp(0.01*(dQ-100)); #Underrelaxation Coefficient
            residual=np.max(abs(np.true_divide((T_new-T),(T_new))));
            T=SOR*T_new+(1-SOR)*T;
            T_BP=SOR*T_BP_new+(1-SOR)*T_BP;
            
            Q_Solar_Net=SOR*Q_Solar_Net_new.T+(1-SOR)*Q_Solar_Net;
            T_IN[0:disc_RPC,0]=T_Inlet;
            T_corner=sum((np.multiply(Ac_SiC,T[1:bound1-1,0])).diagonal())/sum(Ac_SiC);
        
            Q_In=sum(Q_Solar_Net);
            Q_Loss=sum(2*pi*r_n[-1,0]*dL*h_loss*(T[-1,1:Nz-1]-Tamb))
            Q_Loss+=sum(k_SiC*(np.multiply(Ac_SiC,(T[1:bound1-1,0]-T_corner)/0.01).diagonal()))
            Q_Loss+=sum(h_loss_z*(np.multiply(Ac_SiC,(T[1:bound1-1,-1]-Tamb)).diagonal()))
            Q_Loss+=sum(h_loss_z*(np.multiply(Ac_RPC,(T[bound1:bound2-1,0]-Tamb))).diagonal())
            Q_Loss+=sum(h_loss_z*(np.multiply(Ac_RPC,(T[bound1:bound2-1,-1]-Tamb))).diagonal())
            Q_Loss+=sum(h_loss_z*(np.multiply(Ac_INS,(T[bound2:-1,0]-Tamb))).diagonal())
            Q_Loss+=sum(h_loss_z*(np.multiply(Ac_INS,(T[bound2:-1,-1]-Tamb))).diagonal())
            Q_Loss+=sum(2*pi*r_1*dL*h_loss_cav*(T[0,1:-2]-Tamb));        
        
            Q_Fluid=sum(cp*np.multiply(m, (T_OUT-T_IN)));
            dQ=(Q_In-Q_Loss-Q_Fluid)/Q_In*100;
        
            # dQ = 0.000001
            # print('Iteration Loop',str(itnumb),' => Max Residual = ',str(residual),' => Q-Imbalance [%] = ', dQ, sep=None)
            itnumb=itnumb+1;
        
        # np.save('v_a_ross.npy',v_a)
        # np.save('v_b_ross.npy',v_b)
        # np.save('v_c_ross.npy',v_c)
        # np.save('v_d_ross.npy',v_d)
        # np.save('v_e_ross.npy',v_e)
        # np.save('v_f_ross.npy',v_f)
        # np.save('v_T_ross.npy',v_T)        
        
        
        # Out of the loop
        print('Iteration Loop',str(itnumb-1),' => Max Residual = ',str(residual),' => Q-Imbalance [%] = ', dQ, sep=None)
        # print('s_RPC:',str(s_RPC))
        # print('s_INS:',str(s_INS))
        # print('m_dot:',str(Mass_Flow))
        
        # T=T-273;
        # Tf=Tf-273;
        # T_OUT=T_OUT-273;
        T_OUT_m=sum(np.multiply(m,T_OUT)/Mass_Flow);
        # eff_abs = sum(Q_Solar_Net) / Q_Solar_In;
        eff_S2G = Q_Fluid/Q_Solar_In;

        outputs['T_OUT_m'] = T_OUT_m
        outputs['T_Ins'] = np.amax(T[-1,0:Nz])
        #efficienies are coded as -eff for minimization
        # outputs['eff_abs'] = eff_abs **-1
        outputs['eff_S2G'] = eff_S2G *-100
        # outputs['P_out'] = Mass_Flow*cp*(T_OUT_m-T_Inlet)
        # outputs['V'] = pi*(r_4**2-r_1**2)*L
        
        print('Outlet Fluid Temperature', str(T_OUT_m),'K')
        print('Ins Surf. Temperature', str(outputs['T_Ins']),'K')
        
        # print("size", str(np.shape(Q_Solar_Net_new_1)))
        
        # print('Power Outlet (W)=',str(outputs['P_out']))
        print('Solar to Gas Eff:',str(eff_S2G),'%')
        
        draw_contour(z_n[0,:], r_n[:,0], T-273, r_2, r_3, Mass_Flow, 10)
        # outputs['T'] = T
        # save the temperature values
        # np.save('T_MDAO.npy',T-273)
        # np.save('Tf_MDAO.npy',Tf-273)
        print('')       
        
if __name__ =='__main__':
    # build the model
    tic = time.time()

    p = om.Problem()
    p.model.add_subsystem('collector', mdao_Ross_model(disc_z=20,disc_SiC=10,disc_RPC=20,disc_INS=10)) 
    
    # this part for optimization
    # p.driver = om.ScipyOptimizeDriver()#optimizer='differential_evolution')
    p.driver = om.pyOptSparseDriver()
    # p.driver = om.DOEDriver(om.UniformGenerator(num_samples=5))
    
    # this part for optimization
    Mass_Flow = 0.00068
    offset  = 0.2
    # p.model.set_input_defaults('receiver.Mass_Flow', Mass_Flow, units='kg/s')
    
    p.model.add_design_var('collector.Mass_Flow', upper=Mass_Flow*(1+offset), lower=Mass_Flow*(1-offset), units='kg/s')
    p.model.add_constraint('collector.T_OUT_m', lower=1273, units='K')
    p.model.add_objective('collector.eff_S2G')
    
    p.setup()
    p.run_driver()

    # Original
    # disc_z=20,disc_SiC=10,disc_RPC=20,disc_INS=10
    # Modified
    # disc_z=20,disc_SiC=5,disc_RPC=10,disc_INS=5

    # p.setup()
    # p.set_val('collector.r_1', 0.015,units='m')
    # p.set_val('collector.Mass_Flow', 0.00068, units='kg/s')
    # p.run_model()

    # check_partials_data = p.check_partials(compact_print=True)
        # plot with defaults
    # om.partial_deriv_plot('V', 'L', check_partials_data)
    # om.partial_deriv_plot('V', 's_INS', check_partials_data)
    # p.check_partials(compact_print=True)#, method='fd')

    print('Elapsed time is', time.time()-tic, 'seconds', sep=None)

    # om.view_connections(p, outfile= "p1_solver.html", show_browser=False)
    # om.n2(p)
        