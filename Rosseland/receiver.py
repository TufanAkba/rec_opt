#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 14:09:51 2021

@author: tufan


For connecting solid T - radiocity Q - fluid Tf calculations.

"""

import openmdao.api as om
# import numpy as np
import time

from initialization import initialization
from radiocity import radiocity
from solid import solid
from T_corner import T_corner
from fluid import fluid
from draw_contour import draw_contour

class Receiver(om.Group):
    
    def initialize(self):
        
        self.options.declare('disc_z', types=int,default=20,desc='Discretization in z-direction')
        self.options.declare('disc_SiC', types=int,default=10,desc='SiC-Discretization in r-direction')
        self.options.declare('disc_RPC', types=int,default=20,desc='RPC-Discretization in r-direction')
        self.options.declare('disc_INS', types=int,default=10,desc='INS-Discretization in r-direction')
        
        # print('Receiver.initialize')

    def setup(self):
        
        disc_z = self.options['disc_z']
        disc_SiC = self.options['disc_SiC']
        disc_RPC = self.options['disc_RPC']
        disc_INS = self.options['disc_INS']
        
        subsys = initialization(disc_z = disc_z, disc_SiC = disc_SiC, disc_INS = disc_INS, disc_RPC = disc_RPC)
        self.add_subsystem('init', subsys, 
                           promotes_inputs=['*'], 
                           promotes_outputs=['*'])
        
        subsys = radiocity(disc_z = disc_z)
        self.add_subsystem('radiocity', subsys,
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
        subsys = solid(disc_z = disc_z, disc_SiC = disc_SiC, disc_INS = disc_INS, disc_RPC = disc_RPC)
        self.add_subsystem('solid', subsys, 
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
        subsys = T_corner(disc_SiC = disc_SiC)
        self.add_subsystem('T_corner', subsys,
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
        subsys = fluid(disc_z = disc_z, disc_RPC = disc_RPC)
        self. add_subsystem('fluid', subsys,
                            promotes=['*'])
        
        self.set_order(['init','radiocity','solid','fluid','T_corner'])
        
        # self.nonlinear_solver =  om.NewtonSolver()
        # self.nonlinear_solver.options['solve_subsystems'] = True
        # # self.nonlinear_solver.linesearch = om.BroydenSolver()
        # self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        # self.nonlinear_solver.linesearch.options['maxiter'] = 10
        # self.nonlinear_solver.linesearch.options['iprint'] = 2
        # self.nonlinear_solver.linesearch.options['debug_print'] = True
        
        # self.nonlinear_solver = om.NonlinearRunOnce()
        
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options['use_aitken'] = True
        self.nonlinear_solver.options['aitken_max_factor'] = 1.0
        
        # self.nonlinear_solver = om.NonlinearBlockJac()
        self.nonlinear_solver.options['err_on_non_converge'] = True
        
        self.nonlinear_solver.options['iprint'] = 1
        self.nonlinear_solver.options['maxiter'] = 1000
        rtol = 2.5e-05;#print(rtol)
        self.nonlinear_solver.options['rtol'] = rtol
        
        self.linear_solver = om.ScipyKrylov()
        
        # print('Receiver.setup')
    
if __name__ == "__main__":
    

    Mass_Flow = 0.00068
# i=10
# M=np.arange(-i,i)*0.000001+Mass_Flow

# for Mass_Flow in M:
    tic = time.time()
    p = om.Problem()
    p.model.add_subsystem('receiver', Receiver(disc_z=20,disc_SiC=10,disc_RPC=20,disc_INS=10))
    p.setup()    
    
    # this oart for initialize
    p.set_val('receiver.L', 0.065, units='m')
    
    r_1 = 0.015
    s_SiC = 0.005
    s_RPC = 0.015
    s_INS  = 0.1
    
    # L = 0.05877593
    # Mass_Flow = 0.00050082
    # s_INS = 0.01601926
    # s_RPC = 0.03041568
    # s_INS = 0.05413648
    
    p.set_val('receiver.r_1', r_1, units='m')
    p.set_val('receiver.s_SiC',s_SiC, units='m')
    p.set_val('receiver.s_RPC',s_RPC, units='m')
    p.set_val('receiver.s_INS',s_INS, units='m')
    
    p.set_val('receiver.E', 0.9)
    
    p.set_val('receiver.h_loss', 15., units='W/(m**2*K)')
    
    p.set_val('receiver.k_INS', 0.3, units='W/(m*K)')
    p.set_val('receiver.k_SiC', 33., units='W/(m*K)')
    p.set_val('receiver.k_Air', 0.08, units='W/(m*K)')
    
    p.set_val('receiver.porosity', 0.81)
    
    p.set_val('receiver.p', 10., units='bar')
    
    # Mass_Flow = 0.00068
    p.set_val('receiver.Mass_Flow', Mass_Flow, units='kg/s')

    p.set_val('receiver.D_nom', 0.00254, units='m')
    p.set_val('receiver.A_spec', 500.0, units='m**-1')
    p.set_val('receiver.K_ex', 200., units='m**-1')
    
    p.set_val('receiver.cp',1005.,units='J/(kg*K)')
    
    p.set_val('receiver.T_fluid_in',293, units='K')
    
    p.set_val('receiver.Tamb', 293., units='K')    
    
    # this part for radiocity
    p.set_val('receiver.Q_Solar_In', 1000., units='W')
    p.set_val('receiver.sigma', 5.67*10**(-8), units='W/(m**2*K**4)')
    
    p.run_model()
    print('Elapsed time is', time.time()-tic, 'seconds', sep=None)
    
    om.view_connections(p, outfile= "receiver.html", show_browser=False)
    om.n2(p, outfile="receiver_n2.html", show_browser=False)
    
    z_n = p.get_val('receiver.z_n')
    r_n = p.get_val('receiver.r_n')
    T = p.get_val('receiver.T').reshape(44,22)
    
    draw_contour(z_n[0,:], r_n[:,0], T-273, r_1+s_SiC, r_1+s_SiC+s_RPC, Mass_Flow, 10)
    
    Tf_out= p.get_val('receiver.T_fluid_out')
    if Tf_out<1273:
        print('failed')
    print(Mass_Flow)
    print(f'Tf_out: {Tf_out}K')
    print(p.get_val('receiver.eff_S2G'))
    # Q = p.get_val('receiver.radiocity.Q')
    # print(Q)
    # T = p.get_val('receiver.T')
    # print(T)