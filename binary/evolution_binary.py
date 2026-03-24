#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:53:24 2026

@author: afoninamd
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from main.constants import year, AU, Myr, day, M_sun, R_sun, Gyr, G, M_NS,c, R_NS, m_p
from main.model import Object
from main.evolution import evolution, find_stage, getB, increase_time_resolution
from binary.wind import wind_parameters, wind_parameters_isolated_sun, Omega_sun, M_dot_sun, Omega_initial, wind_velocity
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy as sp
from time import time

import cosmic
from cosmic.plotting import evolve_and_plot
from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve
print(cosmic.__file__)

def orbit_evo(m2, porb, ecc): #, omega_spin_2=Omega_initial/100): # 2*np.pi/(4*day/year)
    """
    COSMIC code
    INPUT:
    m2 -- mass of the second component in g [then converted to M_sun]
    porb -- orbital period in s [then converted to days]
    ecc -- initial eccentricity of the binary MS+NS
    omega_spin_2 - spin period of the second component IN DAYS
    OUTPUT:
    numpy array of times in seconds
    numpy array of spin omega of the second component in seconds
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
               'fryer_mass_limit': 0, 'ppi_co_shift': 0, 'ppi_extra_ml':0, 'maltsev_mode':0, 'maltsev_fallback':0, 'maltsev_pf_prob':0, 'mm_mu_ns':1.4, 'mm_mu_bh':1.0}
               # 'omega_spin_2': omega_spin_2}
    
    bpp, bcm, initC, kick_info = Evolve.evolve(initialbinarytable=single_binary, BSEDict=BSEDict,
                                               dtp=0.0)
    
    # Pspin= 2*np.pi / np.array(bcm['omega_spin_2']) * year
    omega_spin = np.array(bcm['omega_spin_2']) / year
    
    ev = np.array(bpp['tphys'])[1]  # second_event_time
    t = np.array(bcm['tphys'])
    return t[t<ev] * Myr, omega_spin[t<ev], np.array(bcm['porb'])[t<ev] * day, np.array(bcm['ecc'])[t<ev], np.array(bcm['rad_2'])[t<ev] * R_sun


def gett_binary(num=100):
    """ For the evolution with a sun-like companion """
    t0 = 1e6*year
    t1 = 10**np.linspace(np.log10(t0), np.log10(0.1*Gyr), endpoint=False, num=num)
    t2 = 10**np.linspace(np.log10(0.1*Gyr), np.log10(1.0*Gyr), endpoint=False, num=num)
    # t3 = 10**np.linspace(np.log10(1.0*Gyr), np.log10(1.1e10*year), endpoint=True, num=num)
    t3 = np.linspace((1.0*Gyr), (11*Gyr), endpoint=True, num=3*num)
    t = np.append(t1, t2)
    t = np.append(t, t3)
    return t


def evolution_binary(t: np.array, P0:float, B: np.array,
              Omega: float, case: str, M_dot_apo: np.array, v_apo: np.array,
              M_dot_per: np.array, v_per: np.array, plot: bool):
    """ A copy of the evolution function but for the binaries """
    """
    Calculates the spin period evolution and the evolutionary stages,
    the arrays t, B, v, Mdot are of the same length
    """
    leng = len(t)
    
    P = np.zeros(leng) # an empty array for the spin period
    P[0] = P0 # the first element of the spin period array is P0

    stages = np.zeros(leng, dtype=np.int8) # an empty array for stages
    
    NS_per_0 = Object(B=B[0], v=v_per[0], Mdot=M_dot_per[0], case=case,
                Omega=Omega) # for the first stage in per
    stages[0] = find_stage(t_cur=t[0], P_cur=P[0], NS=NS_per_0, previous_stage=1)
    
    NS_apo = Object(B=B[0], v=v_apo[0], Mdot=M_dot_apo[0], case=case, Omega=Omega)
    NS_per = Object(B=B[0], v=v_per[0], Mdot=M_dot_per[0], case=case, Omega=Omega)
    
    
    dP_dt = [NS_apo.dP_dt_ejector,
             NS_apo.dP_dt_propeller,
             NS_apo.dP_dt_accretor,
             NS_apo.dP_dt_georotator]
        
    for i in (range(1, leng)):
        
        """ Define the stage and find dP """ #!!!
        prev_stage = stages[i-1]
        
        if prev_stage == 3:
            """ Starting with the first accretor point the stage = accretion """
            for j in range(i, leng):
                P[j] = NS_per.P_eq(t[j-1])
            stages[i:] = 3
            break

        dt = t[i] - t[i-1]

        P_prev = P[i-1]
        t_prev = t[i-1]

        P[i] = P_prev + dt * dP_dt[prev_stage-1](t_prev, P_prev)
        
        NS_per.B0 = B[i]
        NS_per.v0 = v_per[i]
        NS_per.Mdot0 = M_dot_per[i]
        
        NS_apo.B0 = B[i]
        NS_apo.v0 = v_apo[i]
        NS_apo.Mdot0 = M_dot_apo[i]
        
        stages[i] = find_stage(t_cur=t[i], P_cur=P[i], NS=NS_per,
                           previous_stage=stages[i-1])
    
    if plot:
        P_EP = np.zeros(leng)
        P_PE = np.zeros(leng)
        P_PA = np.zeros(leng)
        P_AP = np.zeros(leng)
        for i in (range(0, leng)):
            NS = Object(B=B[i], v=v_per[i], Mdot=M_dot_per[i], case=case, Omega=Omega)
            P_EP[i] = NS.P_EP(t[i])
            P_PE[i] = NS.P_PE(t[i])
            P_PA[i] = NS.P_PA(t[i])
            P_AP[i] = NS.P_AP(t[i])
        
        
        fig = plt.figure(figsize=(6, 10))
        
        # Define GridSpec with relative heights: 1/2 for top, 1/6 for each of the three bottom plots
        gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 1, 1])  # total sum=6; top=3/6=1/2
        
        # Create subplots with shared x-axis
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        ax2 = fig.add_subplot(gs[2], sharex=ax0)
        ax3 = fig.add_subplot(gs[3], sharex=ax0)
        for ax in [ax0, ax1, ax2]:
            ax.tick_params(labelbottom=False)
        
        # fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(6,8))
        plt.subplots_adjust(hspace=0)
        
        ax = ax0
        ax.set_ylabel('P, s')
        ax.fill_between(t/Gyr, 0, P_EP, color='C0')
        ax.plot(t/Gyr, P_EP, color='white', ls='-')
        ax.plot(t/Gyr, P_PE, color='white', ls='--')
        ax.fill_between(t/Gyr, P_PA, t*0+1.5*max(P), color='C1')
        ax.fill_between(t/Gyr, P_EP, P_PA, color='#aec7e8')
        ax.plot(t/Gyr, P_PA, color='white', ls='-')
        ax.plot(t/Gyr, P_AP, color='white', ls='--')
        ax.plot(t/Gyr, P, 'k', marker='o', markersize=7)
        
        ax.set_title(case)
        # Define colors for each stage
        colors = {
            1: 'C0',
            2: '#aec7e8',
            3: 'C1',
            4: '#ffbb78'
        }
        # Plot dotted line with different colors for each stage
        stages_np = np.array(stages)
        for stage in range(1, 5):
            mask = stages_np == stage
            ax.plot(t[mask]/Gyr, P[mask], 'o', markersize=4,
                    ls='none', color=colors[stage])
        
        ax.set_yscale('log')
        ax.set_xlim([t[0]/Gyr, t[-1]/Gyr])
        ax.set_ylim(0, 1.5*max(P))
        
        ax = ax1
        ax.set_ylabel('B, G')
        ax.plot(t/Gyr, B)
        ax.set_yscale('log')
        
        ax = ax2
        ax.set_ylabel('M_dot, g/s')
        ax.plot(t/Gyr, M_dot_apo)
        ax.set_yscale('log')
        
        ax = ax3
        ax.set_ylabel('v, km/s')
        ax.set_xlabel('t, Gyr')
        ax.plot(t/Gyr, v_apo/1e5)
        ax.set_yscale('log')
    return t, P, stages


def RocheLobeRadius(q):
    """ Returns R_L/a, lower q gives higher R_L/a"""
    return 0.49 * q**(2/3) / (0.6*q**(2/3) + np.log(1+q**(1/3)))


def get_current_wind_properties(r, v_cur, Mdotw):
    v_wind = wind_velocity(r)
    v = (v_wind**2 + v_cur**2)**0.5
    rho = Mdotw / (4*np.pi*r**2*v_wind)
    Mdot = 4 * np.pi * G**2 * M_NS**2 * rho / v**3
    
    R_G = 2 * G * M_NS / v**2
    factor = 3
    Mdot[r<factor*R_G] = r[r<factor*R_G]**2/4/R_sun**2 * Mdotw[r<factor*R_G]
    return v, Mdot


def getvMdot(axis, ecc, Mdotw, Mstar):
    
    a_apo = axis * (1 + ecc)
    a_per = axis * (1 - ecc)
    
    v_orb = (G*(Mstar+M_NS)/axis)**0.5
    v_orb_apo = v_orb * ((1-ecc)/(1+ecc))**0.5
    v_orb_per = v_orb * ((1+ecc)/(1-ecc))**0.5
    
    v_apo, M_dot_apo = get_current_wind_properties(a_apo, v_orb_apo, Mdotw)
    v_per, M_dot_per = get_current_wind_properties(a_per, v_orb_per, Mdotw)
    
    return v_apo, M_dot_apo, v_per, M_dot_per


def one_binary_evolution(P0, B0, case, field, Mstar, ecc0, Porb0, plot):
    """
    Put everything in CGS
    """
    t_binary, omega_spin, p_orb, ecc, r_star = orbit_evo(Mstar, Porb0, ecc0) # from the BSE code
    M_dot_wind = wind_parameters(Mstar, r_star, omega_spin) # mass-loss rate of the second component
    axis = ((p_orb/year)**2 / (1/(1.4+Mstar/M_sun)))**(1/3) * AU # semi-major axis
    """ Roche lobe overflow """
    # RL = axis * RocheLobeRadius(q=Mstar/M_NS) # for the sun-like component
    if len(t_binary) < 2:
    # if len(t_binary[RL<r_star]) > 0:
        print("There is a Roche lobe overflow in P0 = {} s, B0 = 10^{} G, Mstar = {} Msun, case {}, e = {}, Porb = {} d".format(P0, np.log10(B0), case, Mstar/M_sun, ecc0, Porb0/day))
        return None # Roche lobe overflow is detected
    
    if plot:
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(6,6))
        ax = axes[0,0]
        ax.plot(t_binary/Gyr, p_orb/day, label='p_orb')
        ax.plot(t_binary/Gyr, 2*np.pi/omega_spin/day, label='p_spin')
        ax.legend()
        ax.set_ylabel('$P$, d')
        ax = axes[1,0]
        ax.plot(t_binary/Gyr, ecc, label='eccentricity')
        ax.set_ylabel('eccentricity')
        ax = axes[2,0]
        ax.plot(t_binary/Gyr, axis/R_sun, label='a')
        ax.set_ylabel('$a$, $R_\odot$')
        ax = axes[3,0]
        ax.plot(t_binary/Gyr, M_dot_wind, label='M_dot')
        ax.set_ylabel('$\dot{M}_w$, g/s')
        ax.set_xlabel('$t$, Gyr')
    
    """ Inputs for the NS evolution """
    t = gett_binary()
    M_dot_wind_func = sp.interpolate.interp1d(t_binary, M_dot_wind, kind='linear', bounds_error=False, fill_value=np.nan, assume_sorted=False)
    ecc_func = sp.interpolate.interp1d(t_binary, ecc, kind='linear', bounds_error=False, fill_value=np.nan, assume_sorted=False)
    axis_func = sp.interpolate.interp1d(t_binary, axis, kind='linear', bounds_error=False, fill_value=np.nan, assume_sorted=False)
    p_orb_func = sp.interpolate.interp1d(t_binary, p_orb, kind='linear', bounds_error=False, fill_value=np.nan, assume_sorted=False)
    
    M_dot_wind = M_dot_wind_func(t)
    ecc = ecc_func(t)
    axis = axis_func(t)
    p_orb = p_orb_func(t)
    
    Omega = 2 * np.pi / p_orb
    
    if plot:
        ax = axes[0,1]
        ax.plot(t/Gyr, p_orb/day, label='p_orb')
        ax.legend()
        ax.set_ylabel('$P$, d')
        ax = axes[1,1]
        ax.plot(t/Gyr, ecc, label='eccentricity')
        ax.set_ylabel('eccentricity')
        ax = axes[2,1]
        ax.plot(t/Gyr, axis/R_sun, label='a')
        ax.set_ylabel('$a$, $R_\odot$')
        ax = axes[3,1]
        ax.plot(t/Gyr, M_dot_wind, label='M_dot')
        ax.set_ylabel('$\dot{M}_w$, g/s')
        ax.set_xlabel('$t$, Gyr')
    
    """ Finding nans """
    if len(Omega[np.isnan(Omega)]) > 0:# or len(M_dot_wind[np.isnan(M_dot_wind)]) > 0:
        first_nan_idx = np.where(np.isnan(Omega))[0][0]
        # first_nan_idx_mdot = np.where(np.isnan(M_dot_wind))[0][0]
        # first_nan_idx = np.min([first_nan_idx_omega, first_nan_idx_mdot])
        M_dot_wind = M_dot_wind[:first_nan_idx]
        ecc        = ecc[:first_nan_idx]
        axis       = axis[:first_nan_idx]
        p_orb      = p_orb[:first_nan_idx]
        Omega      = Omega[:first_nan_idx]
        t          = t[:first_nan_idx]
    
    v_apo, M_dot_apo, v_per, M_dot_per = getvMdot(axis, ecc, M_dot_wind, Mstar)
    B = getB(t=t, B0=B0, field=field)
    Omega_func = sp.interpolate.interp1d(t, Omega, kind='linear', bounds_error=False, fill_value=np.nan, assume_sorted=False)
    
    
    t, P, stages = evolution_binary(t=t, P0=1e-1, B=B, Omega=Omega_func, case=case,
                             M_dot_apo=M_dot_apo, v_apo=v_apo, 
                             M_dot_per=M_dot_per, v_per=v_per,
                             plot=plot)
    
    """ Second iteration """
    t_old = t
    t = increase_time_resolution(t_old, P, stages)
    
    M_dot_wind = M_dot_wind_func(t)
    ecc = ecc_func(t)
    axis = axis_func(t)
    p_orb = p_orb_func(t)
    Omega = 2 * np.pi / p_orb
    v_apo, M_dot_apo, v_per, M_dot_per = getvMdot(axis, ecc, M_dot_wind, Mstar)
    B = getB(t=t, B0=B0, field=field)
    t, P, stages = evolution_binary(t=t, P0=1e-1, B=B, Omega=Omega_func, case=case,
                             M_dot_apo=M_dot_apo, v_apo=v_apo, 
                             M_dot_per=M_dot_per, v_per=v_per,
                             plot=plot)
    
    return t, P, stages

# time1 = time()
def axis_to_Porb(a, Mstar):
    return (a/AU)**1.5 * (1/(1.4 + Mstar/M_sun))**0.5 * year

""" HERE """ #!!!
Mstar = 1*M_sun
a = AU * 10**(-0.75) #(-0.75)
for a in[AU * 10**(-0.75), AU * 10**(-1)]:
    one_binary_evolution(P0=0.01, B0=1e12, case='A', field='CF', Mstar=1*M_sun, ecc0=0.5, Porb0=axis_to_Porb(a, Mstar), plot=1)

# print(10**(-0.25)*AU/R_sun * (0.1))
# for a in[AU * 10**(-0.25), AU * 10**(-0.3)]:
#     one_binary_evolution(P0=0.01, B0=1e12, case='A', field='CF', Mstar=1*M_sun, ecc0=0.9, Porb0=axis_to_Porb(a, Mstar), plot=1)

# print(time()-time1)
# string = 'Рентгеновские наблюдения кандидата в аккрецирующие нейтронные звезды в широкой двойной системе'
# print(string.upper())

def test_orbit_evo():
    a_arr = 10000 * AU #0 * R_sun#0.03 * AU
    m2 = 1.0 * M_sun
    porb = (a_arr/AU)**1.5 * (1/(1.4+m2/M_sun))**0.5 * year
    print(porb/day) # seconds
    ecc = 0.
    
    """ Isolated sun's evolution """
    t_isolated, omega_spin_isolated, _, _, _ = orbit_evo(m2, (10000)**1.5 * (1/(1.4+m2/M_sun))**0.5 * year, 0.0)
    t, omega_spin, p_orb, e, r_star = orbit_evo(m2, porb, ecc)
    pspin = 2*np.pi/omega_spin
    print(omega_spin[0]/day)
    print(pspin[0]/day)
    
    # domega = -np.diff(omega_spin)# -omega_spin[-1]+omega_spin[1]
    # dt = np.diff(t) #t[-1] - t[1]
    # fig, ax = plt.subplots()
    # ax.plot(np.log10(domega)/np.log10(dt))
    # print(np.log10(domega)/np.log10(dt))
    # omega_spin_isolated_0 = omega_spin_0[t0<100*Myr][-1]
    # print(len(t), len(omega_spin), len(p_orb), len(e), len(r_star))
    
    fig, axes = plt.subplots(nrows=2, figsize=(6, 8))
    
    axes[0].set_title("e = {:.1f}, p_orb = {:.2f} d".format(ecc, porb))
    
    ax = axes[0]
    ax.plot(t/Myr, p_orb/day, label='Orbital Period', color='k', ls='--', marker='o', markersize=1)
    # ax.plot(t/Myr, 27, label='Orbital Period', color='k', ls='--', marker='o', markersize=1)
    
    omega_johnstone = omega_spin[-1] * (t/t[-1])**(-0.566)
    pspin_johnstone = 2*np.pi/omega_johnstone
    ax.plot(t/Myr, pspin_johnstone/day, label='Johnstone', marker='o', markersize=1)
    ax.plot(t/Myr, pspin/day, label='Star Spin Period', marker='o', markersize=1)
    
    ax.set_ylabel("Spin Period [day]")
    ax.set_yscale('log')
    ax.legend()
    
    ax = axes[1]
    ax.plot(t, e, color='k', ls='--', marker='o', markersize=1)
    # ax.legend()
    ax.set_ylabel('Eccentricity')
    ax.set_xlabel("Time [Myr]")
    
    Rstar0 = r_star[0]
    R_sun_0 = 0.8882494502975121 * R_sun
    Mdotsun = M_dot_sun * np.abs(omega_spin/Omega_sun)**(4/3) * (m2/M_sun)**(-3.36) * (Rstar0/R_sun_0)**(2)
    # omega, Mdotsun = wind_parameters(t, Mstar=m2, Rstar=r_star, omega_spin_binary=omega_spin, t_isolated=t_isolated, omega_spin_isolated=omega_spin_isolated)
    _, Mdotsunold_analytic = wind_parameters_isolated_sun(t)
    Mdotsunold = M_dot_sun * np.abs(omega_johnstone/Omega_sun)**(4/3) * (m2/M_sun)**(-3.36) * (Rstar0/R_sun_0)**(2)
    
    fig, ax = plt.subplots()
    ax.plot(t/Myr, Mdotsunold/M_sun*year)
    ax.plot(t/Myr, Mdotsun/M_sun*year, marker='o')
    ax.plot(t/Myr, Mdotsunold_analytic/M_sun*year, '--')
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_ylabel('M_dot_wind, g/s')
    ax.set_xlabel('age, Myr')
    
    # fig, ax = plt.subplots()
    # ax.plot(t/Gyr, r_star**0.5)#pspin)

# m2 = 1*M_sun
# t_isolated, omega_spin_isolated, p_orb, e, r_star = orbit_evo(m2, (10000)**1.5 * (1/(1.4+m2/M_sun))**0.5 * year, 0.0)
# Mdot = wind_parameters(m2, r_star, omega_spin_isolated)
# plt.plot(t_isolated/Gyr, Mdot)

# test_orbit_evo()
# NS = Object(B=1e12, v=100e5, Mdot=1e12, case='A', Omega=None)
# print(NS.P_PA(0))

def test_initial_spin():
    a_arr = 10000 * AU #0 * R_sun#0.03 * AU
    m2 = 1.0 * M_sun
    porb = (a_arr/AU)**1.5 * (1/(1.4+m2/M_sun))**0.5 * year
    print(porb/day) # seconds
    ecc = 0.
    arr = np.zeros(1000)
    """ Isolated sun's evolution """
    for i in tqdm(range(len(arr))):
        t_isolated, omega_spin_isolated, _, _, _ = orbit_evo(m2, (10000)**1.5 * (1/(1.4+m2/M_sun))**0.5 * year, 0.0, omega_spin_2=Omega_initial)
        # rnd = np.random.rand()
        # pspinif = rnd * 4.95 + 0.05
        # ospin = 1/pspinif
        # arr[i] = ospin
        arr[i] = 2*np.pi/omega_spin_isolated[0]/day
        # arr[i] = (2*np.pi/omega_spin_isolated[0]/day) #1/(np.random.rand() * 4.95 + 0.05)
        
    plt.hist(arr)
    print(arr)
    print(np.amax(arr), np.amin(arr))
# test_initial_spin()
# print(16.930359717316872*1.16, 0.17426283216826438*1.16)
"""call random_number(rnd)
pspinif = rnd * 19.8d0 + 0.2d0
ospin(k) = 45.35d0*vrotf(mass(k), ST_tide)/rm*pspinif/1.16"""
# rnd = np.random.rand()
# ospinif = rnd * 4.95 + 0.05
# print(1/ospinif)


def P_PA(B, v, Mdot):
    mu = B * R_NS**3
    R_G = 2 * G * M_NS / v**2
    rho = 1 * m_p
    Mdot = np.pi * R_G**2 * rho * v
    # rho = Mdot / 4 / np.pi**2 / (G*M_NS)**2 * v**3
    # p = 2**(5/6) * mu**(2/3) / ((G*M_NS)**(1/3)*v**(2/3)*Mdot**(1/3)) / 0.87**(1.5)
    p = 2**(5/6)*np.pi/0.87**1.5 * (mu**2/(Mdot*v**2*G*M_NS))**(1/3)
    # p = 2**(1/6)*np.pi**(1/3)/0.87**(3/2) * (v*mu**2/(rho))**(1/3) / G / M_NS
    
    p = 1.3e4 *(v/10e5)**(1/3)*(B/1e12)**(2/3)* (1)**(-1/3)
    return p

# print(P_PA(B=1e12, v=10e5, Mdot=0 ))#1e12))

# test_orbit_evo()
# v = 10e5
# print(G*M_NS*2/v**3/year)

# R_NS = 10e5
# r_G = G*M_NS/c**2
# print((1-2*r_G/R_NS)**0.5)

# R_NS = 9e5
# r_G = G*M_NS/c**2
# print((1-2*r_G/R_NS)**0.5)

# R_NS = 14e5
# r_G = G*M_NS/c**2
# print((1-2*r_G/R_NS)**0.5)
# # a = (-0.75+0.566*4/3)/2
# # print(1/a)
# # print(0.566*4/3)


