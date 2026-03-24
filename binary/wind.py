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
from main.constants import m_p, G, M_NS, R_NS, AU, R_sun, k_B, M_sun, year, Myr, Gyr, day
from time import time

from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

M_dot_sun= 1.4e-14*M_sun/year
Omega_sun = 2 * np.pi / 27 / day
# print(Omega_sun)
Omega_sun = 2.67e-6 # Carrington rotation
Omega_initial = Omega_sun #* (100 * Myr/4.6/Gyr)**(-0.5) # 4 days Johnstone 2015 II Omega ~ t^-0.5 if t > 100 Myr
# print(2*np.pi/Omega_sun/15/day)# print(2*np.pi/Omega_initial / day)

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
    v = v_a * (1 - np.exp(-(r-r1)/r_a))**beta
    v[np.isnan(v)] = 0
    # print(v)
    return v

r = np.linspace(0, 1)*AU
plt.plot(r, wind_velocity(r))


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


def Omega_saturation(Mstar):
    """ Saturated regime (Johnstone 2015 II) """
    Omega_sun_sat = 15 * Omega_sun
    Omega_sat = Omega_sun_sat * (Mstar/M_sun)**2.3
    return Omega_sat


def wind_parameters(Mstar, Rstar, Omega):
    """ 
    Omega is the spin of the sun-like star
    """
    Omega_sat = Omega_saturation(Mstar)
    Omega[Omega>Omega_sat] = Omega_sat
    M_dot_w = M_dot_sun * (Rstar/R_sun)**2 * (Omega/Omega_sun)**1.33 * (Mstar/M_sun)**(-3.36)
    
    return M_dot_w


""" Outdated wind parameters functions """

def wind_parameters_isolated_sun(t):
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


def wind_parameters_binary_only(Mstar, Rstar, Pspin):
    """
    Mstar in g
    Radius is in cm
    Pspin in seconds
    """
    Omega = 2 * np.pi / Pspin
    Omega_sat = Omega_saturation(Mstar)
    Omega[Omega>Omega_sat] = Omega_sat
    """ Unsaturated regime (Johnstone 2015 II) """
    M_dot_w = M_dot_sun * (Rstar/R_sun)**2 * (Omega/Omega_sun)**1.33 * (Mstar/M_sun)**(-3.36)
    
    return M_dot_w


def wind_parameters_old(t, Mstar, Rstar, omega_spin_binary, t_isolated, omega_spin_isolated): #omega_spin_isolated, 
    """ Johnstone II """
    """ Omega ~ t^-0.566 from the observations of the Sun """
    t0 = 100*Myr
    a = 0.566
    omega_0 = Omega_initial # the omega of an isolated star at t = 0 and t = t0
    
    omega_isolated_johnstone = omega_0 *(t/t0)**(-a)
    omega_isolated_johnstone[t<t0] = omega_0
    
    t_isolated[-1] = t[-1]
    omega_isolated_bse = np.array(sp.interpolate.interp1d(t_isolated, omega_spin_isolated)(t))
    omega_isolated_bse = omega_isolated_bse / omega_isolated_bse[0] * omega_0
    # d_omega_isolated = np.zeros(len(t))
    # d_omega_isolated[t>t0] = -omega_0 * a * (t[t>t0]/t0)**(-a-1)/t0
    
    """ d_omega for the same star, but in a binary """
    omega_binary = omega_spin_binary / omega_spin_binary[0] * omega_0
    # d_omega_binary = np.gradient(omega_spin_binary, t)
    
    """ Combined """
    omega_isolated = omega_isolated_johnstone
    omega_binary = - omega_binary + omega_isolated_bse
    # d_omega_spin = np.zeros(len(t))
    # d_omega_spin[t>t0] = d_omega_isolated[t>t0] + d_omega_binary[t>t0]
    
    # """ creating an array of the final omega """
    # omega = np.zeros(len(t))
    # omega[t>t0] = omega_0 + sp.integrate.cumulative_trapezoid(d_omega_spin[t>t0], t[t>t0], initial=0)
    # first_val = omega[omega!=0][0]
    # omega[omega==0] = first_val
    # omega[omega<0] = 0
    
    omega = omega_isolated + omega_binary
    
    """ Omega function has a maximum """
    Omega_sat = Omega_saturation(Mstar)
    omega[omega>Omega_sat] = Omega_sat
    
    Rstar0 = Rstar[0]
    R_sun_0 = 0.8882494502975121 * R_sun
    Mdot = M_dot_sun * np.abs(omega/Omega_sun)**(4/3) * (Mstar/M_sun)**(-3.36) * (Rstar0/R_sun_0)**(2)
    
    # plt.plot(omega)
    # plt.plot(t, omega_binary, ls='--')
    # plt.plot(t, omega_isolated, ls='--')
    # plt.plot(t, d_omega_spin, ls='--')
    # print(d_omega_binary)
    return omega, Mdot


def wind_parameters_try1(t, Mstar, Rstar, omega_spin_binary, omega_spin_isolated_0): #omega_spin_isolated, 
    """ Johnstone II """
    """ Omega ~ t^-0.566 from the observations of the Sun """
    t0 = 100*Myr
    a = 0.566
    omega_0 = omega_spin_isolated_0 # the omega of an isolated star at t0
    
    d_omega_isolated = np.zeros(len(t))
    d_omega_isolated[t>t0] = -omega_0 * a * (t[t>t0]/t0)**(-a-1)/t0
    
    """ d_omega for the same star, but in a binary """
    d_omega_binary = np.gradient(omega_spin_binary, t)
    
    """ Combined """
    d_omega_spin = np.zeros(len(t))
    d_omega_spin[t>t0] = d_omega_isolated[t>t0] + d_omega_binary[t>t0]
    
    """ creating an array of the final omega """
    omega = np.zeros(len(t))
    omega[t>t0] = omega_0 + sp.integrate.cumulative_trapezoid(d_omega_spin[t>t0], t[t>t0], initial=0)
    first_val = omega[omega!=0][0]
    omega[omega==0] = first_val
    # omega[omega<0] = 0
    
    """ Omega function has a maximum """
    Omega_sat = Omega_saturation(Mstar)
    omega[omega>Omega_sat] = Omega_sat
    
    Rstar0 = Rstar[0]
    R_sun_0 = 0.8882494502975121 * R_sun
    Mdot = M_dot_sun * np.abs(omega/Omega_sun)**(4/3) * (Mstar/M_sun)**(-3.36) * (Rstar0/R_sun_0)**(2)
    
    # plt.plot(omega)
    # plt.plot(t, d_omega_binary, ls='--')
    # plt.plot(t, d_omega_isolated, ls='--')
    # plt.plot(t, d_omega_spin, ls='--')
    # print(d_omega_binary)
    return omega, Mdot

def wind_parameters_outdated(t, Mstar, Rstar, Pspin, Pspin0):
    """
    Includes the evolution of an isolated star and a star in a binary
    Pspin0 - evolution, when the NS is really far away a = 1000 AU, e = 0
    """
    # M_dot_binary = wind_parameters_binary_only(Mstar, Rstar, Pspin)
    Rstar0 = Rstar[0]
    R_sun_0 = 0.8882494502975121 * R_sun
    """if it is alone, it evolves like the Sun but with a different radius and mass"""
    _, M_dot_isolated = wind_parameters_isolated_sun(t)
    M_dot_isolated = M_dot_isolated * (Rstar0/R_sun_0)**2 * (Mstar/M_sun)**(-3.36)
    M_dot = M_dot_isolated
    # print(M_dot[0])
    # print(Rstar[0], Pspin[0])
    
    Omega_evo_isolated_mult = (t/4.6/Gyr)**(-0.75) * (Mstar/M_sun)**(3.36) * (Rstar/R_sun)**(-2)
    Omega_evo_isolated_mult[t<100*Myr] = (100*Myr/4.6/Gyr)**(-0.75) * (Mstar/M_sun)**(3.36) * (Rstar[t<100*Myr]/R_sun)**(-2)
    Omega_evo_isolated = Omega_sun * Omega_evo_isolated_mult**(3/4)
    # Omega_evo_isolated[0] = Omega_evo_isolated[1]
    Omega_evo_isolated = Omega_evo_isolated / Omega_evo_isolated[0] * 2 * np.pi / Pspin[0] #Omega_initial # normalization
    
    """
    If the star spins up, then the Omega difference starts to be important
    This mode starts with the last true "isolated" value of M_dot
    taking into account further evolution of the Radius
    """
    dP = np.diff(Pspin)
    print(dP)
    idxs = np.where(dP < 0)[0] # starts to spin up
    if len(idxs) > 0:
        idx = idxs[0]
        print("idx pf the dPspin > 0 =", idx)
        M_dot[idx:] = M_dot_isolated[idx] * (Rstar[idx:]/Rstar[idx])**2 * (2 * np.pi / Pspin[idx:]/Omega_sun)**1.33
    
    # M_dot = M_dot_isolated * (2 * np.pi / Pspin / Omega_evo_isolated)**1.33
    """ If the Saturation takes place, then it should be topped by it """
    Omega_sat = Omega_saturation(Mstar)
    M_dot_sat = M_dot_sun * (Rstar/R_sun)**2 * (Omega_sat/Omega_sun)**1.33 * (Mstar/M_sun)**(-3.36)

    M_dot[M_dot>M_dot_sat] = M_dot_sat[M_dot>M_dot_sat]

    return M_dot
