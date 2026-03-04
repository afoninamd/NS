#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:20:05 2026

@author: afoninamd
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp
from scipy.optimize import root_scalar
from main.model import Object
from main.evolution import gett, getB, evolution
from main.constants import m_p, G, M_NS, R_NS, AU, R_sun, k_B, M_sun, year, Myr, Gyr
from time import time

from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec


""" Wind profiles v(r) """

def wind_velocity(r):
    """
    The dotted line in the middle panel of Fig.3 from Johnstone I (2015)
    shows the radial speed profile of the slow wind
    estimated by Sheeley et al. (1997).
    r in cm
    v in cm/s
    """
    v_a = 418.7e5
    r1 = 4.5 * R_sun
    r_a = 15.2 * R_sun
    beta = 0.5
    return v_a * (1 - np.exp(-(r-r1)/r_a))**beta


def wind_velocity_beta_law(r):
    """ Beta-law for the wind speed, better for giants """
    beta = 3
    v_inf = 400e5
    v = v_inf * (1 - R_sun/r)**beta
    return v


def parker_velocity_profile(M, R, Tcor, r_grid):
    """
    Compute Parker wind velocity profile.

    Parameters:
        M     : stellar mass (kg)
        R     : stellar radius (m)
        Tcor  : coronal temperature (K)
        r_grid: array of radii (m)

    Returns:
        v(r) in m/s
    """
    def sound_speed(T):
        # Mean molecular weight (fully ionized plasma)
        mu = 0.6
        return np.sqrt(k_B*T/(mu*m_p))
    
    def sonic_radius(M, cs):
        return G*M/(2*cs**2)
    
    def parker_equation(v, r, cs, rs):
        return (v/cs)**2 - np.log((v/cs)**2) - (4*np.log(r/rs) + 4*rs/r - 3)

    cs = sound_speed(Tcor)
    rs = sonic_radius(M, cs)

    v = np.zeros_like(r_grid)

    for i, r in enumerate(r_grid):

        # Subsonic branch inside sonic point
        if r < rs:
            bracket = (1e-6*cs, cs)
        else:
            bracket = (cs, 10*cs)

        sol = root_scalar(
            parker_equation,
            args=(r, cs, rs),
            bracket=bracket,
            method='brentq'
        )

        v[i] = sol.root

    return v, cs, rs


def plot_wind_velocity_profiles():
    """ An illustration of v_wind(r) for the Sun """
    
    def orbital_velocity(r):
        return (G*2.4*M_sun/r)**0.5
    
    M_star = M_sun
    R_star = R_sun
    T_cor  = 1.5e6  # solar coronal temperature
    
    r = np.linspace(1.01*R_star, 1*AU, 500)
    
    v, cs, rs = parker_velocity_profile(M_star, R_star, T_cor, r)
    
    plt.figure()
    plt.plot(r/AU, v/1e5, label="Parker solar wind model")
    plt.plot(r/AU, wind_velocity(r)/1e5, label="Sheeley et al. (1997)")
    plt.plot(r/AU, orbital_velocity(r)/1e5, label="Orbital velocity")
    
    
    plt.plot(r/AU, (v**2+orbital_velocity(r)**2)**0.5/1e5, label="Parker solar wind model", lw=3)
    plt.plot(r/AU, (wind_velocity(r)**2+orbital_velocity(r)**2)**0.5/1e5, label="Sheeley et al. (1997)", lw=3)
    plt.axvline(R_sun/AU, ls=':', color='k')
    plt.legend()
    plt.xlabel("r (AU)")
    plt.ylabel("v (km/s)")
    # plt.title("Parker Solar Wind Model")
    plt.grid()
    plt.show()

# print(G*2.4*M_sun/400e5**2/R_sun)

# plot_wind_velocity_profiles()

def plot_M_dot_profile():
    """ An illustration of M_dot for an NS at the circular orbit around the Sun """
    
    def R_G(v_inf, M_star=M_sun):
        return (G * (M_NS+M_star)) / v_inf**2

    def local_density(r, v_w, M_dot_0=1):
        return M_dot_0 / (4 * np.pi * r**2 * v_w)

    def M_dot(r, v_w, M_star=M_sun, M_dot_0=1e12):
        v_inf = ((G * (M_NS+M_star) / r) + v_w**2)**0.5
        return np.pi * R_G(v_inf)**2 * local_density(r, v_w, M_dot_0) * v_inf
    
    M_star = M_sun
    R_star = R_sun
    M_dot_0 = 10**12
    T_cor  = 1.5e6  # solar coronal temperature
    
    r = np.linspace(2*R_star, 1*AU, 500)
    
    plt.figure()
    
    v_w, cs, rs = parker_velocity_profile(M_star, R_star, T_cor, r)
    plt.plot(r/AU, M_dot(r, v_w, M_star)/M_dot_0, label="Parker solar wind model")
    plt.plot(r/AU, M_dot(r, wind_velocity(r), M_star)/M_dot_0, label="Sheeley et al. (1997)")
    plt.axvline(R_sun/AU, ls=':', color='k')
    plt.legend()
    plt.xlabel("r (AU)")
    plt.ylabel("M_dot/M_dot_wind")
    plt.yscale("log")
    # plt.title("Parker Solar Wind Model")
    plt.grid()
    plt.show()


def wind_parameters(t, pspin):
    """
    The solar wind parameters for a given t 
    pspin: the spin period of the stars cgs
    """
    
    def Mdot_t(t):
        """ The sun's mass loss rate """
        s = 0.75
        Mdot= 2e-14*M_sun/year * (4.6*Gyr/t)**s
        # print("{:.2e}".format(2e-14*(4.6*Gyr/0.1/Gyr)**0.75))
        return Mdot
    
    def T_t(t):
        """ The temperature at the base r=r0 """
        T0 = 1.5e6 * (4.6*Gyr/t)**0.05
        return T0
        
    if isinstance(t, float) or isinstance(t, int):
        if t < 0.1 * Gyr:
            Mdot = Mdot_t(0.1*Gyr)
            T0 = T_t(1.0*Gyr)
        else:
            Mdot = Mdot_t(t)
            T0 = T_t(t)
    else:
        Mdot = Mdot_t(t)
        Mdot[t<=0.1*Gyr] = Mdot_t(0.1*Gyr)

        T0 = T_t(t)
        T0[t<1.0*Gyr] = T_t(1.0*Gyr)
    
    
    
    
    return T0, Mdot
