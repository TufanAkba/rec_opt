#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 14:46:47 2021

@author: tufan
"""
import openmdao.api as om
from math import pi
#import numpy as np
import time

from draw_contour import draw_contour

from receiver import Receiver
# TODO: recorder ekle


"""
optimization notes:
    
scaler=-1000
t-c: get neg. mass flow opt.n crashes
slsqp: limits to upper bound
pyopt: limits to upper bound

scaler=-500
t-c:
slsqp: limits to lower bound
pyopt: limits to lower bound
t-c: corse opt.

scaler:-250
slsqp: mass flow:[0.00065281]
t-c-: mass flow:[0.00068242]
pyopt: tries to get 0 change this code

scaler:-200
t-c: 

t-c -1,-10,-100,-1000,-10000 optimizasyonları
-1 : 193 iter.s ==> xtol satisfied
Elapsed time is 9158.4 seconds
Tf_out:[1289.10453279]
mass flow:[0.00066815]
eff:[66.88766228]%

{('receiver.fluid.eff_S2G', '_auto_ivc.v11'): array([[972.07816453]])}
{('receiver.fluid.T_fluid_out', '_auto_ivc.v11'): array([[-1456740.8258236]])}

-10: 224 iter.s ==> xtol satisfied
Elapsed time is 12526.7 seconds
Tf_out:[1295.77316439]
mass flow:[0.00066004]
eff:[66.51771588]%

{('receiver.fluid.eff_S2G', '_auto_ivc.v11'): array([[8061.0306187]])}
{('receiver.fluid.T_fluid_out', '_auto_ivc.v11'): array([[-1224627.06775017]])}

-100: 126.0 s 
Tf_out:[1278.58972674]
mass flow:[0.00068106]
eff:[67.46001936]%

{('receiver.fluid.eff_S2G', 'receiver.Mass_Flow'): array([[-242.56314709]])}
{('receiver.fluid.T_fluid_out', 'receiver.Mass_Flow'): array([[3191.96812077]])}

-750 :  2 iter
Tf_out:[1273.84611971]
mass flow:[0.00068693]
eff:[67.71399135]%

{('receiver.fluid.eff_S2G', 'receiver.Mass_Flow'): array([[-1819.22360315]])}
{('receiver.fluid.T_fluid_out', 'receiver.Mass_Flow'): array([[3191.96812077]])}

-900: 8176.9 s ==> display closed for this optimization
Tf_out:[1272.6895284]
mass flow:[0.00068836]
eff:[67.77539782]%

-1000: 131 iter.s ==> xtol satisfied
Tf_out:[1279.10550399]
mass flow:[0.00068042]
eff:[67.43223515]%

{('receiver.fluid.eff_S2G', 'receiver.Mass_Flow'): array([[-3086.99734133]])}
{('receiver.fluid.T_fluid_out', 'receiver.Mass_Flow'): array([[3949.75101767]])}

-10000: 136 iter.s ==>

Tf_out:[1257.65048528]
mass flow:[0.0007072]
eff:[68.56122354]%

{('receiver.fluid.eff_S2G', 'receiver.Mass_Flow'): array([[-19245.05658972]])}
{('receiver.fluid.T_fluid_out', 'receiver.Mass_Flow'): array([[2495.39471216]])}

slsqp -1,-10,-100,-1000,-10000 optimizasyonları
tol: 1e-8

-900: 22 iter.s
upper bound

-1000: {'receiver.Mass_Flow': array([0.00070488])}' de sabitleniyor
    
-750: 8 iter.s 1e-7 tol
Tf_out:[1301.71752652]
mass flow:[0.00065285]
eff:[66.18328499]%

{('receiver.fluid.eff_S2G', 'receiver.Mass_Flow'): array([[-6471.56105526]])}
{('receiver.fluid.T_fluid_out', 'receiver.Mass_Flow'): array([[7607.09723644]])}

-850 trust-constr with 10**-7 tol gives the best sol.n
"""



if __name__ == '__main__':

    
    # Opt.n param.s got best soln.
    scaler = -(850)
    optimizer='trust-constr'
    tol = 1e-7
    print(f'\n{optimizer} optimizater with scaler: {scaler} ({tol} tolerance)')
    print('------------------------------------------------------\n')
    
    r_1 = 0.015
    s_SiC = 0.005
    s_RPC = 0.015
    s_INS  = 0.1
    L = 0.065
    vol = pi * (r_1+s_SiC+s_RPC+s_INS)**2 * L
    
    
    tic = time.time()
    
    p = om.Problem()
    p.model.add_subsystem('receiver', Receiver(disc_z=20,disc_SiC=10,disc_RPC=20,disc_INS=10))
    
    # this part for optimization
    debug_print = ['desvars','objs','nl_cons','totals']
    # p.driver = om.ScipyOptimizeDriver(debug_print = debug_print, optimizer='SLSQP', tol=1e-7)#,optimizer='differential_evolution')
    p.driver = om.ScipyOptimizeDriver(debug_print = debug_print, optimizer=optimizer, tol=tol)#,optimizer='differential_evolution')#scaler 900 ip.driver = om.ScipyOptimizeDriver(optimizer='shgo',debug_print = debug_print)#,optimizer='differential_evolution')#scaler 900 is optimum gain!..
    # TODO: SLSQP ile trust-constr'yi -1000 scaler'da karşılaştır!
    # TODO: trust-constr ve SLSQP olunca scaler'in 100'den büyük olması lazım aksi halde gtol'den optimizasyonu tamamlıyor
    #p.driver = om.pyOptSparseDriver(debug_print = debug_print)
    # p.driver = om.DOEDriver(om.UniformGenerator(num_samples=10),debug_print = debug_print)
    
    # p.driver = om.SimpleGADriver(debug_print = debug_print)
    # p.driver.options['penalty_parameter'] = 5.
    # p.driver.options['penalty_exponent'] = 8.
    

    # this part for optimization
    Mass_Flow = 0.00065
    offset  = 0.04
    p.model.set_input_defaults('receiver.Mass_Flow', Mass_Flow, units='kg/s')
    
    p.model.set_input_defaults('receiver.s_SiC',s_SiC, units='m')
    p.model.set_input_defaults('receiver.s_RPC',s_RPC, units='m')
    p.model.set_input_defaults('receiver.s_INS', s_INS, units='m')
    p.model.set_input_defaults('receiver.L',L, units='m')
    
    p.model.add_design_var('receiver.s_SiC',lower=s_SiC*(1-offset),upper=s_SiC*(1+offset), units='m',scaler= 1000)
    p.model.add_design_var('receiver.s_RPC',lower=s_RPC*(1-offset),upper=s_RPC*(1+offset), units='m',scaler= 750)
    p.model.add_design_var('receiver.s_INS',lower=s_INS*(1-offset),upper=s_INS*(1+offset), units='m',scaler= 750)
    p.model.add_design_var('receiver.L',lower=L*(1-offset),upper=L*(1+offset), units='m',scaler= 100)
    p.model.add_design_var('receiver.Mass_Flow', upper=Mass_Flow*(1+offset), lower=Mass_Flow*(1-offset), units='kg/s')
    
    p.model.add_constraint('receiver.T_outer', upper=373, units='K')
    p.model.add_constraint('receiver.T_fluid_out', lower=1000, upper=1100, units='degC')
    p.model.add_constraint('receiver.Volume', upper=vol,units='m**3')
    
    p.model.add_objective('receiver.eff_S2G',scaler=scaler, adder=0)#, adder=-1,scaler=100)#scaling should be increased
    
    # Here we show how to attach recorders to each of the four objects:
    #   problem, driver, solver, and system
    
    # Create a recorder
    recorder = om.SqliteRecorder('cases.sql')
    
    # Attach recorder to the problem
    p.add_recorder(recorder)
    
    # Attach recorder to the driver
    p.driver.add_recorder(recorder)
    
    
    p.setup()    
    
    # Attach recorder to a subsystem
    p.model.receiver.add_recorder(recorder)
    
    # Attach recorder to a solver
    p.model.receiver.nonlinear_solver.add_recorder(recorder)

    
    p.set_solver_print(1)
    
    # this part for initialize
    # p.set_val('receiver.L', 0.065, units='m')

    p.set_val('receiver.r_1', r_1, units='m')
    # p.set_val('receiver.s_SiC',s_SiC, units='m')
    # p.set_val('receiver.s_RPC',s_RPC, units='m')
    # p.set_val('receiver.s_INS',s_INS, units='m')
    
    p.set_val('receiver.E', 0.9)
    
    p.set_val('receiver.h_loss', 15., units='W/(m**2*K)')
    
    p.set_val('receiver.k_INS', 0.3, units='W/(m*K)')
    p.set_val('receiver.k_SiC', 33., units='W/(m*K)')
    p.set_val('receiver.k_Air', 0.08, units='W/(m*K)')
    
    p.set_val('receiver.porosity', 0.81)
    
    p.set_val('receiver.p', 10., units='bar')
    
    # Mass_Flow = 0.00068
    # p.set_val('receiver.Mass_Flow', Mass_Flow, units='kg/s')

    p.set_val('receiver.D_nom', 0.00254, units='m')
    p.set_val('receiver.A_spec', 500.0, units='m**-1')
    p.set_val('receiver.K_ex', 200., units='m**-1')
    
    p.set_val('receiver.cp',1005.,units='J/(kg*K)')
    
    p.set_val('receiver.T_fluid_in',293, units='K')
    
    p.set_val('receiver.Tamb', 293., units='K')    
    
    # this part for radiocity
    p.set_val('receiver.Q_Solar_In', 1000., units='W')
    p.set_val('receiver.sigma', 5.67*10**(-8), units='W/(m**2*K**4)')

    p.final_setup()
    # p.run_model()  #just for single iteration, solves not optimizes
    p.run_driver()
    p.record("final_state")
    print('Elapsed time is', time.time()-tic, 'seconds', sep=None)
    
    om.view_connections(p, outfile= "receiver.html", show_browser=False)
    om.n2(p, outfile="receiver_n2.html", show_browser=False)
    
    z_n = p.get_val('receiver.z_n')
    r_n = p.get_val('receiver.r_n')
    T = p.get_val('receiver.T').reshape(44,22)
    Mass_Flow = p.get_val('receiver.Mass_Flow')
    draw_contour(z_n[0,:], r_n[:,0], T-273, r_1+s_SiC, r_1+s_SiC+s_RPC, Mass_Flow, 10)
    
    Tf_out= p.get_val('receiver.T_fluid_out')
    print(f'Tf_out:{Tf_out}')
    m = p.get_val('receiver.Mass_Flow')
    print(f'mass flow:{m}')
    eff_S2G = p.get_val('receiver.eff_S2G')
    print(f'eff:{(eff_S2G)*100}%')
    s_SiC = p.get_val('receiver.s_SiC')
    print(f's_SiC: {s_SiC}  m')

    p.cleanup()
    
    # Q = p.get_val('receiver.radiocity.Q')
    # print(Q)
    # T = p.get_val('receiver.T')
    # print(T)