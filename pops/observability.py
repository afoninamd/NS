#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 10:38:44 2025

@author: afoninamd
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main.evolution import gett
from main.constants import G, c, h, k_B, M_NS, R_NS, R0, kpc, sigma_B, galaxy_age, Gyr
from pops.track import get_coordinates_velocities, evolution_galaxy_iterations
from pops.density import neutral_density, molecular_density  # , density_cold, density_hot,  halo_density, ionized_density
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from astropy.io import fits
from time import time

path_data = 'data/'

""" Tabulated data """
range_kev = np.array([
    -np.inf, 0.1, 0.284, 0.4, 0.532, 0.707,
    0.867, 1.303, 1.840, 2.471, 3.21,
    4.038, 7.111, 8.331, np.inf
    ])

c0 = np.array([17.3, 34.6, 78.1, 71.4,
               95.5, 308.9, 120.6, 141.3,
               202.7, 342.7, 352.2, 433.9,
               629.0, 701.2
               ])
c1 = np.array([608.1, 267.9, 18.8, 66.8,
               145.8, -380.6, 169.3, 146.8,
               104.7, 18.7, 18.7, -2.4, 30.9, 25.2])
c2 = np.array([-2150, -476.1, 4.3, -51.4, -61.1,
               294, -47.7, -31.5, -17, 0, 0, 0.75, 0, 0])
    

def cross_section(nu: np.array) -> np.array:
    """ Returns cross section corresponding to the nu array """
    cross = np.zeros(len(nu))
    kev = h * nu / 1.60218e-9
    for i in range(len(range_kev)-1):
        cond = np.logical_and(kev>range_kev[i], kev<=range_kev[i+1])
        E = kev[cond]
        cross[cond] = (c0[i] + c1[i]*E + c2[i]*E**2) / E**3 * 1e-24 # cm^2
    return cross


def erosita_effective_area(plot=False) -> sp.interpolate:
    fits_file = 'arf01_200nmAl_sdtq.fits'
    with fits.open(path_data + fits_file) as hdul:
        specresp_hdu = hdul[1]
        if specresp_hdu.data is not None:
            df = pd.DataFrame(specresp_hdu.data)
            energ_lo = df['ENERG_LO']
            energ_hi = df['ENERG_HI']
            specresp = df['SPECRESP']
            energ_mid = (energ_lo + energ_hi) / 2
            specresp[energ_mid<0.2] = 0
             
            """ Plot eRosita S_eff """
            if plot:
                fig, ax = plt.subplots()
                ax.tick_params(axis='both', which='major', labelsize=16)
                plt.plot(energ_mid, specresp, linestyle='-')
                plt.xlabel('$E$, кэВ', fontsize=16)
                plt.ylabel('$S_{eff}$, см$^{2}$', fontsize=16)
                # plt.title('Spectral Response vs Energy')
                plt.grid()
                plt.xscale('log')  # Log scale for energy axis
                plt.yscale('log')  # Log scale for spectral response axis#
                plt.savefig('/home/afoninamd/Documents/NS/pictures/diplom/eRosita.pdf', bbox_inches='tight')
            # [kev], [cm^2]
            E, Seff = np.array(energ_mid), np.array(specresp)
            return sp.interpolate.interp1d(E, Seff, bounds_error=False, fill_value=0)


def column_density(pos) -> float:
    """ Requies two dots in the Galaxy [kpc], returns number of m_p on the line """
    """ Create x, y, z arrays """
    xi = np.linspace(0, 1, 1000)
    pos1 = np.array([0, R0, 0]) # x, y, z
    pos2 = np.array(pos) #x, y, z
    
    """ Slow integration """
    xi1 = np.append(np.array([0.]), xi[0:len(xi)-1])
    deltaxi = xi - xi1
    
    x = pos1[0] + xi*(pos2[0]-pos1[0])
    y = pos1[1] + xi*(pos2[1]-pos1[1])
    z = pos1[2] + xi*(pos2[2]-pos1[2])
    
    """ Only hydrogen atoms are needed """
    n = neutral_density(x, y, z) + molecular_density(x, y, z) # + halo_density(x, y, z) + ionized_density(z)
    N = np.sum(n*deltaxi) * kpc * np.sum((pos2-pos1)**2)**0.5
    return N


def plank(nu, T) -> np.array:
    r_s = 2 * G * M_NS / c**2
    g_00 = 1 - r_s/R_NS
    nu = nu / (g_00)**0.5
    B = 2*h*nu**3/c**2 / (np.exp(h*nu/(k_B*T))-1)
    return B


def flux_to_counts_constants(num=1000):
    """ All necessary constants for the flux_to_counts function """
    nu = 10**np.linspace(13, 19, num)
    kev = h * nu / 1.60218e-9
    Seff = np.array(erosita_effective_area()(kev))
    cross = np.array(cross_section(nu))
    deltanu = np.zeros(len(nu))
    deltanu[1:] = np.diff(nu)
    return nu, deltanu, cross, Seff


def flux_to_counts(T, N_H, F0, nu, deltanu, cross, Seff):
    """ F0 = L / 4 pi r^2 """
    spectrum = plank(nu, T)
    N0 = np.sum(deltanu * spectrum)  # normalization
    N_ph = spectrum / (h*nu) / N0 * F0  # the number of photons
    absorption = np.exp(-cross * N_H)
    
    counts_no_absorption_array = np.array(Seff*N_ph)
    counts_no_absorption_array[np.isnan(counts_no_absorption_array)] = 0
    counts_no_absorption = sp.integrate.simpson(counts_no_absorption_array, x=nu)
    
    counts_with_absorption_array = np.array(Seff*N_ph*absorption)
    counts_with_absorption_array[np.isnan(counts_with_absorption_array)] = 0
    counts_with_absorption = sp.integrate.simpson(counts_with_absorption_array, x=nu)
    
    flux_no_absorption_array = N_ph*h*nu
    flux_no_absorption = sp.integrate.simpson(flux_no_absorption_array, x=nu)  # = F0
    
    flux_with_absorption_array = N_ph*h*nu*absorption
    flux_with_absorption = sp.integrate.simpson(flux_with_absorption_array, x=nu)
    
    # plt.plot(nu, counts_no_absorption_array, alpha=0.2, c='C0')
    # plt.plot(nu, counts_with_absorption_array, alpha=0.2, c='C1')
    # plt.xscale('log')
    
    # plt.plot(nu, flux_no_absorption_array, alpha=0.2, c='C0')
    # plt.plot(nu, flux_with_absorption_array, alpha=0.2, c='C1')
    # plt.xscale('log')
    # plt.yscale('log')
    
    return counts_no_absorption, counts_with_absorption, flux_no_absorption, flux_with_absorption


def one_observability(t1, stages1, x1, y1, z1, B1, Mdot1,
                      nu, deltanu, cross, Seff):
    """ Observability calculaton for one track """
    leng = len(t1) # the number of dots
    T = np.zeros(leng)  # the T_eff of polar caps
    f0 = np.zeros(leng)  # flux no absorption
    f1 = np.zeros(leng)  # flux with absorption
    c0 = np.zeros(leng)  # counts no absorption
    c1 = np.zeros(leng)  # counts with absorption
    
    Mdot1[stages1!=3] = 0
    L_array = G * M_NS / R_NS * Mdot1
    r = kpc * ((x1-0)**2 + (y1-R0)**2 + (z1-0)**2)**0.5
    F0_array = L_array / (4*np.pi*r**2)
    
    idx = np.logical_and(stages1==3, F0_array>1e-15)  # temporary arrays corresponding to accretion
    lenga = len(t1[idx])
    Tt = np.zeros(lenga)  # effective temperature
    f0t = np.zeros(lenga)  # flux no absorption
    f1t = np.zeros(lenga)  # flux with absorption
    c0t = np.zeros(lenga)  # counts no absorption
    c1t = np.zeros(lenga)  # counts with absorption
    
    x2 = x1[idx]  # the coordinates for the accretor stage
    y2 = y1[idx]
    z2 = z1[idx]
    B2 = B1[idx]  # for the Alfven radius
    Mdot2 = Mdot1[idx]  # for the Alfven radius
    F0_array = F0_array[idx]
    L_array = L_array[idx]
    
    mu = B2 * R_NS**3
    R_A = (mu**2 / (2 * Mdot2 * (2 * G * M_NS)**0.5))**(2/7)
    R_cap = R_NS * (R_NS/R_A)**0.5
    S_cap = 2 * np.pi * R_cap**2
    Tt = (L_array / S_cap / sigma_B)**0.25
    
    for i in range(lenga):
        N_H = column_density(np.array([x2[i], y2[i], z2[i]]))
        F0 = F0_array[i]
        c0t[i], c1t[i], f0t[i], f1t[i] = flux_to_counts(Tt[i], N_H, F0, nu,
                                                        deltanu, cross, Seff)
    
    T[idx], c0[idx], c1[idx], f0[idx], f1[idx] = Tt, c0t, c1t, f0t, f1t
    return T, f0, f1, c0, c1


def flux_to_counts_test():
    # fig, ax = plt.subplots()
    N_H = column_density([0,5,0.5])
    
    F0 = 1
    T = 1e7
    
    T = 10**np.linspace(4,8)
    f0, f1 = np.zeros(len(T)), np.zeros(len(T))
    c0, c1 = np.zeros(len(T)), np.zeros(len(T))
    
    time0 = time()
    num = [30, 100, 300, 1000, 3000, 10000]
    time1 = np.zeros(len(num))
    fig, ax = plt.subplots()
    
    for j in range(len(num)):
        for i in range(len(T)):
            nu, deltanu, cross, Seff = flux_to_counts_constants(num[j])
            c0[i], c1[i], f0[i], f1[i] = flux_to_counts(T[i], N_H, F0, nu, deltanu, cross, Seff)
        time1[j] = time() - time0
    
        # ax.plot(T, c1, alpha=0.5, label=num[j]) # it is F0
    # ax.plot(T, f1) # with absorption, less than f0
    ax.legend()
    ax.set_xscale('log')
    print(time() - time0, 's')
    
    fig, ax = plt.subplots()
    ax.plot(num, time1)


def one_observability_test():
    """
    Creates one evolutionary track of a neutron star and calculates
    and plots fluxes, count rates and effective temperatures
    """
    pos = np.array([0, 8, 0])
    vel = np.array([0, 0, 30])
    
    res_trajectory = get_coordinates_velocities(pos=pos, vel=vel,
                                                t_end=galaxy_age,
                                                plot=0)
    t0, num, xyz, v_xyz = res_trajectory
    
    t = gett(t_end=galaxy_age, n=num)
    P0 = 1e-1
    B0 = 1e12
    field = 'CF'
    case = 'A'
    galaxy_type = 'simple'  #'two_phase'
    
    res = evolution_galaxy_iterations(P0, t, xyz, v_xyz, B0, field, case,
                                      plot=1, iterations=1 , # 3
                                      galaxy_type=galaxy_type)
    t1, P1, B1, stages1, v1, Mdot1, ph1, x1, y1, z1, vx1, vy1, vz1 = res
    
    nu, deltanu, cross, Seff = flux_to_counts_constants()
    
    time0 = time()
    T, f0, f1, c0, c1 = one_observability(t1, stages1, x1, y1, z1, B1, Mdot1,
                                          nu, deltanu, cross, Seff)
    
    print(time()-time0, 's')
    
    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(6, 10))
    
    t1 = t1 / Gyr
    
    ax = axes[0]
    ax.set_title('{} galaxy, case {}, field {}'.format(galaxy_type, case, field))
    ax.plot(t1, T)
    ax.set_ylabel('Effective Temperature')
    ax.set_yscale('log')
    ax.set_ylim([1e5, 1e7])
    
    ax = axes[1]
    ax.plot(t1, f0, label='no absorption')
    ax.plot(t1, f1, label='with absorption')
    ax.set_ylabel('Flux')
    ax.set_yscale('log')
    
    ax = axes[2]
    ax.plot(t1, c0, label='no absorption')
    ax.plot(t1, c1, label='with absorption')
    ax.set_ylabel('Count Rate')
    ax.set_xlabel('t, Gyr')
    ax.set_yscale('log')


# one_observability_test()
# flux_to_counts_test()