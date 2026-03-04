#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:53:24 2026

@author: afoninamd
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from main.constants import year, AU, Myr, day, M_sun

import matplotlib.pyplot as plt
import numpy as np

import cosmic
from cosmic.plotting import evolve_and_plot
from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve
print(cosmic.__file__)

def orbit_evo(m2, porb, ecc, omega_spin_2=2*np.pi/(4*day/year)):
    """
    COSMIC code
    INPUT:
    m2 -- mass of the second component in g [then converted to M_sun]
    porb -- orbital period in s [then converted to days]
    ecc -- initial eccentricity of the binary MS+NS
    omega_spin_2 - spin period of the second component IN DAYS
    OUTPUT:
    numpy array of times in seconds
    numpy array of spin period of the second component in seconds
    numpy array of orbital periods in seconds
    numpy array of eccentricities
    """
    
    m2 = m2 / M_sun
    porb = porb / day
    
    single_binary = InitialBinaryTable.InitialBinaries(
                    m1=1.4,
                    m2=m2,
                    porb=porb, # days
                    ecc=ecc,
                    tphysf=10000,  # 10000 is 10 Gyr evolution time
                    kstar1=13,
                    kstar2=1,  # 0 for <0.7 M_sun, 1 for >0.7 M_sun
                    metallicity=0.014  # 0.02 solar)
                    )
    
    BSEDict = {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.001, 'pts3': 0.02, 'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 'tflag': 1, 'acc2': 1.5, 'grflag' : 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.25, 'ecsn_mlow' : 1.6, 'aic' : 1, 'ussn' : 0, 'sigmadiv' :-20.0, 'qcflag' : 1, 'eddlimflag' : 0, 'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 1, 'ST_tide' : 1, 'bdecayfac' : 1, 'rembar_massloss' : 0.5, 'kickflag' : 1, 'zsun' : 0.014, 'bhms_coll_flag' : 0, 'don_lim' : -1, 'acc_lim' : -1, 'rtmsflag' : 0, 'wd_mass_lim': 1,
               'fryer_mass_limit': 0, 'ppi_co_shift': 0, 'ppi_extra_ml':0, 'maltsev_mode':0, 'maltsev_fallback':0, 'maltsev_pf_prob':0, 'mm_mu_ns':1.4, 'mm_mu_bh':1.0,
               'omega_spin_2': omega_spin_2}
    
    bpp, bcm, initC, kick_info = Evolve.evolve(initialbinarytable=single_binary, BSEDict=BSEDict,
                                               dtp=0.0)
    
    Pspin= 2*np.pi / np.array(bcm['omega_spin_2']) * year
    return np.array(bcm['tphys']) * Myr, Pspin, np.array(bcm['porb']) * day, np.array(bcm['ecc'])


def test_orbit_evo():
    a_arr = 0.5 * AU
    m2 = 1 * M_sun
    porb = (a_arr/AU)**1.5 * (1/(1.4+m2/M_sun))**0.5 * year
    print(porb) # seconds
    ecc = 0.8
    
    t, pspin, p_orb, e = orbit_evo(m2, porb, ecc)
    
    t = t / Myr
    pspin = pspin / day
    p_orb = p_orb / day
    
    fig, axes = plt.subplots(nrows=2, figsize=(6, 8))
    
    axes[0].set_title("e = {:.1f}, p_orb = {:.2f} d".format(ecc, porb))
    
    ax = axes[0]
    ax.plot(t, p_orb, label='Orbital Period', color='k', ls='--', marker='o', markersize=1)
    ax.plot(t, pspin, label='Star Spin Period', marker='o', markersize=1)
    ax.set_ylabel("Spin Period [day]")
    ax.set_yscale('log')
    ax.legend()
    
    ax = axes[1]
    ax.plot(t, e, color='k', ls='--', marker='o', markersize=1)
    # ax.legend()
    ax.set_ylabel('Eccentricity')
    ax.set_xlabel("Time [Myr]")
    
# test_orbit_evo()

def M_dot


def popsynthesis_1d(num=100):
    M_star = M_sun
    M_dot_0 = 1
    
    a_arr = 10**np.linspace(-1, 0, num)*AU
    M_dot_arr = M_dot(a_arr, wind_velocity(a_arr), M_star, M_dot_0)
    v_arr = ((G * (M_NS+M_star) / a_arr) + wind_velocity(a_arr)**2)**0.5
    
    orb_p = (a_arr/AU)**1.5 * (1/(2.4))**0.5 * year
    Omega_arr = 2*np.pi/orb_p
    
    L_max_arr = np.zeros(len(Omega_arr))
    t_accr_arr = np.zeros(len(Omega_arr))
    
    for i in range(len(Omega_arr)):
        t = gett(t_end=1e10*year, n=1000)
        B = getB(t, B0=1e12, field='CF')
        
        v = t*0 + v_arr[i]
        
        _, M_dot_from = wp(t)
        Mdot = M_dot_arr[i]/M_dot_0*M_dot_from
        
        # time0 = time() # mark the time
        t, P, stages = evolution(t=t, P0=1e-1, B=B, Omega=Omega_arr[i], case='A',
                                 Mdot=Mdot, v=v, plot=0)
        
        if len(stages[stages==3]) > 0:
            Mdot_max = Mdot[stages==3][0]
            # Mdot_max = np.amax(Mdot[stages==3])
            L_max_arr[i] = G * M_NS * Mdot_max / R_NS
            t_accr_arr[i] = t[stages==3][0]
    
    fig, axes = plt.subplots(nrows=2)
    ax = axes[0]
    ax.plot(a_arr/AU, L_max_arr)
    ax.set_yscale("log")
    ax.set_ylabel("L_max, erg/s")
    
    ax = axes[1]
    ax.plot(a_arr/AU, t_accr_arr/year)
    ax.set_yscale("log")
    ax.set_ylabel("t_accr, yr")
    ax.set_xlabel("a, AU")

# popsynthesis(num=10)

def popsynthesis_2d(num=4):
    M_star = M_sun
    M_dot_0 = 1
    
    a_arr = 10**np.linspace(-1, 0, num)*AU # R MIN IS A ROCHE LOBE OVERFLOW
    M_star_arr = np.linspace(0.4, 2, num)*M_sun
    
    M_dot_arr = np.zeros([num, num])
    
    v_arr = ((G * (M_NS+M_star) / a_arr) + wind_velocity(a_arr)**2)**0.5 # v_inf on a circular orbit
    
    orb_p = (a_arr/AU)**1.5 * (1/(2.4))**0.5 * year
    Omega_arr = 2*np.pi/orb_p
    
    L_max_arr = np.zeros([num, num])
    t_accr_arr = np.zeros([num, num])
    
    for j in range(len(M_star_arr)):
        M_dot_arr[:,j] = M_dot(a_arr, wind_velocity(a_arr), M_star_arr[j], M_dot_0)
        for i in range(len(Omega_arr)):
            t = gett(t_end=1e10*year, n=1000)
            B = getB(t, B0=1e12, field='CF')
            
            v = t*0 + v_arr[i]
            
            _, M_dot_from = wp(t) # mass loss rate of the star
            Mdot = M_dot_arr[i,j]/M_dot_0*M_dot_from
            
            # time0 = time() # mark the time
            t, P, stages = evolution(t=t, P0=1e-1, B=B, Omega=Omega_arr[i], case='A',
                                     Mdot=Mdot, v=v, plot=0)
            
            if len(stages[stages==3]) > 0:
                Mdot_max = Mdot[stages==3][0]
                # Mdot_max = np.amax(Mdot[stages==3])
                L_max_arr[i,j] = G * M_NS * Mdot_max / R_NS
                t_accr_arr[i,j] = t[stages==3][0] / year
    
    
    a_AU = a_arr / AU
    M_star_sun = M_star_arr / M_sun
    
    # Create meshgrid for pcolormesh
    X, Y = np.meshgrid(a_AU, M_star_sun)
    
    fig, axes = plt.subplots(nrows=2) #, figsize=(16, 10))
    # gs = gridspec.GridSpec(2, width_ratios=[1, 1, 0.05], height_ratios=[1, 1])
    
    ax2 = axes[0]
    im2 = ax2.pcolormesh(X, Y, L_max_arr.T, norm=LogNorm(), cmap='plasma', shading='auto')
    ax2.set_xlabel('Orbital Separation [AU]')
    ax2.set_ylabel('Stellar Mass [M☉]')
    ax2.set_title('Maximum Luminosity [erg/s]')
    ax2.set_xscale('log')
    ax2.set_yscale("log")
    ax2.set_ylabel("L_max, erg/s")
    plt.colorbar(im2, ax=ax2)
    
    ax = axes[1]
    # ax.plot(a_arr/AU, t_accr_arr/year)
    ax.set_yscale("log")
    ax.set_ylabel("t_accr, yr")
    ax.set_xlabel("a, AU")
    
    ax3 = axes[1]
    im3 = ax3.pcolormesh(X, Y, t_accr_arr.T, norm=LogNorm(), cmap='inferno', shading='auto')
    ax3.set_xlabel('Orbital Separation [AU]')
    ax3.set_ylabel('Stellar Mass [M☉]')
    ax3.set_title('Time to Max Accretion [yr]')
    ax3.set_xscale('log')
    plt.colorbar(im3, ax=ax3)
    

popsynthesis_2d(num=10)