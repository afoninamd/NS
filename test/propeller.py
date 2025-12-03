#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 17:54:55 2025

@author: afoninamd
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main.evolution import gett
from main.constants import G, c, h, k_B, M_NS, R_NS, R0, kpc, sigma_B, galaxy_age, Gyr, m_p, year
from main.model import Object
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from astropy.io import fits
from time import time
from scipy.integrate import quad
import matplotlib as mpl


mpl.rcParams['text.usetex'] = True

def propeller_duration_2d():
    num = 50
    v_arr = np.linspace(0,1000,num)*1e5
    B_arr = 10**np.linspace(8,15,num)
    rho = 0.1 * m_p
    delta_t_arr = np.zeros([num, num])
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    
    for k in range(4):
        ax = axes.flatten()[k]
        case = ['A', 'B', 'C', 'D'][k]
        for i in range(num):
            for j in range(num):
                # case = 'A'#['A', 'B', 'C', 'D'][i]
                v = v_arr[i]
                B = B_arr[j]
                Mdot = 4*np.pi*(G*M_NS)**2*rho*v**(-3)
                NS = Object(B=B, v=v, Mdot=Mdot, case=case, Omega=None)
                # print(NS.dP_dt_propeller(0, NS.P_PA(0)))
                def integrand(P):
                    dPdt = NS.dP_dt_propeller(0, P)
                    return 1.0 / dPdt
                
                lower = NS.P_EP(0)
                upper = NS.P_PA(0)
                
                delta_t, error = quad(integrand, lower, upper)
                delta_t_arr[i,j] = np.log10(delta_t/year)
            
        
            cms = ax.pcolormesh((v_arr)/1e5, np.log10(B_arr), delta_t_arr, cmap='magma', shading='auto', rasterized=True, vmin=4, vmax=20)
        cbar_ax = fig.add_axes([0.92, 0.108, 0.02, 0.77])
        fig.colorbar(cms, cax=cbar_ax) #, label='Fraction of the accretor stage')


def propeller_duration_1d():
    num = 10
    v_arr = np.linspace(0,10,num//2,endpoint=False)*1e5
    v_arr = np.append(v_arr, np.linspace(10,1000,num//2)*1e5)
    # v_arr[1] = 1e-20
    
    delta_t_arr = np.zeros(num)
    delta_t_arr1 = np.zeros(num)
    delta_t_arr2 = np.zeros(num)
    
    fig, ax = plt.subplots()
    colors = ['C3', 'C2', 'C1', 'C0']
    lss = [(0, (3, 5, 1, 5, 1, 5)), '-.', '--', '-']
    for k in range(4):
        case = ['A', 'B', 'C', 'D'][k]
        for i in range(num):
            # case = 'A'#['A', 'B', 'C', 'D'][i]
            v = v_arr[i]
            
            def get_delta_t(rho, B):
                Mdot = 4*np.pi*(G*M_NS)**2*rho*v**(-3)
                NS = Object(B=B, v=v, Mdot=Mdot, case=case, Omega=None)
                # print(NS.dP_dt_propeller(0, NS.P_PA(0)))
                def integrand(P):
                    dPdt = NS.dP_dt_propeller(0, P)
                    return 1.0 / dPdt
                
                lower = NS.P_EP(0)
                upper = NS.P_PA(0)
                
                delta_t, error = quad(integrand, lower, upper)
                return (delta_t/year)
            
            delta_t_arr1[i] = get_delta_t(rho=0.1*m_p, B=1e12)
            delta_t_arr2[i] = get_delta_t(rho=10*m_p, B=1e14)
            # delta_t_arr2[i] = get_delta_t(rho=0.01*m_p, B=1e10)
        
        # ax.plot(v_arr/1e5, delta_t_arr, ls=lss[k], color=colors[k])
        ax.plot(v_arr/1e5, delta_t_arr1, ls=lss[k], color=colors[k], label=case)
        ax.plot(v_arr/1e5, delta_t_arr2, ls=lss[k], color=colors[k])
        print(delta_t_arr2)
        ax.fill_between(v_arr/1e5, delta_t_arr1, delta_t_arr2, color=colors[k], alpha=0.1)
    ax.hlines(y=[(galaxy_age/year), 10**6, 10**9], xmin=-1, xmax=v_arr[-1]/1e5, ls=':', color='k', alpha=0.5)
    # ax.set_xlim([-1, v_arr[-1]/1e5])
    # ax.set_ylim([1,10**14])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Propeller stage duration, yr', fontsize=20)
    ax.set_xlabel('$v$, km s$^{-1}$', fontsize=20)
    leg = ax.legend(title='Propeller\nmodel')
    leg.get_title().set_fontsize(16)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    fig.savefig(f'/home/afoninamd/Documents/NS/project/pops/figures/propeller.pdf', bbox_inches='tight')


# propeller_duration_1d()
# propeller_duration_2d()



# for j in range(2):
#     v = [10e5, 100e5][j]
#     rho = [1, 0.01][j] * m_p
#     Mdot = 4*np.pi*(G*M_NS)**2*rho*v**(-3)
#     for i in range(4):
#         case = ['A', 'B', 'C', 'D'][i]
#         NS = Object(B=1e12, v=v, Mdot=Mdot, case=case, Omega=None)
#         # print(NS.dP_dt_propeller(0, NS.P_PA(0)))
#         def integrand(P):
#             dPdt = NS.dP_dt_propeller(0, P)
#             return 1.0 / dPdt
        
#         lower = NS.P_EP(0)
#         upper = NS.P_PA(0)
        
#         delta_t, error = quad(integrand, lower, upper)
        
#         print("Delta t =", np.log10(delta_t/year))
#         print("Estimated numerical error =", error)
#     print('\n')