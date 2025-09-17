#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:37:55 2023

@author: afoninamd
"""


from astropy import constants as const
import gala.potential as gp
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import pi
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.cosmology import default_cosmology

# mw = gp.BovyMWPotential2014() # from Bovy 2015 (getpot is in the bottom)
mw = gp.MilkyWayPotential()

N = 100  # the number of tracks overall
output_dir = 'result/realistic/'

""" Physical constants """
c = const.c.cgs.value                  # speed of light in vacuum,   [cm / s]
G = const.G.cgs.value                  # gravitational constant,     [cm**3 / (g * s**2)]
M_sun = const.M_sun.cgs.value          # solar mass,                 [g]
M_NS = 1.4 * M_sun
R_sun = const.R_sun.cgs.value
k_B = const.k_B.cgs.value
e = 4,8032067991251e-10
h = const.h.cgs.value
m_p = const.m_p.cgs.value              # proton mass,                [g]
AU = 1.496e+13                         # astronomical unit,          [cm]
day = 24 * 60 * 60                     # [s]
year = 365.24219878 * day              # year,                       [s]
Myr = 1e6 * year
Gyr = 1e9 * year
birth_age = 5.                         # [s]
galaxy_age = 1.361e10 * year           # [s]
giant_age = 30e6 * year                # [s]
R_NS = 1e6                             # [cm]
I = 1e45                               # [g cm**2]
sunPos = 8.0                           # [kpc] from Yao2017 8.3
pc = 3.08e18                           # [cm]
rgal = 20                              # [kpc]
rsun = 8                               # [kpc]
L_sun = 3.827 * 10**33                 # [erg / s]
T_eff_sun = 5780                       # [K]
sigma_B = const.sigma_sb.cgs.value        # Stefan-Boltzmann constant sigma
sigma_T = 6.652e-29 * 1e4 # [cm**2]
kpc = 1e3 * pc
R0 = 8  # [kpc]

MdotEdd = 4 * pi * m_p * c / sigma_T * R_NS #/M_sun * year#/ (1.4 * M_sun)
# print('MdotEdd {:.3e}'.format(MdotEdd))
Time = 2 * pi * 1000 * R_sun / (G * 15.5 * M_sun / 1000 / R_sun)**0.5
# print('{:.3e}'.format(Time/year))
""" Other """
maxTrackTime = 100 # s
fontsize = 24
giant_age0 = 7e6 * year

""" For the halo profile """
cosmo = default_cosmology.get()
H_0 = cosmo.H(0.0).value * 1e5 / 1e6 / pc

""" ISM """
# c_s_hot = (k_B*2e6/m_p)**0.5
# c_s_cold = (k_B*8000/m_p)**0.5
c_s_hot = 100e5
c_s_cold = 10e5
# kpc = 1e3*pc
# print(c_s_hot/1e5, c_s_cold/1e5)


# def D():
    # """ Turbulent accretor """
#     rho = m_p * 1
#     v = 100e5
#     R_G = 2*G*M_NS/v**2
#     M_dot = np.pi * R_G**2 * rho * v
#     v_t = 10e5
#     R_t = 2e22
#     D = 1/6 * (M_dot * v_t * R_t / I)**2 * R_G / v
#     B = 1e12
#     mu = B * R_NS**3
#     R_A = (mu**2 / (2 * M_dot * (2 * G*M_NS)**0.5))**(2/7)
#     Gamma = 8 * M_dot * R_A**2
#     P = 2*np.pi*(Gamma/D)**0.5
#     print("{:.2e}".format(P))


#print("{:.2e}".format(8e9 * 0.01**(2/3)))
# print(np.log10(85000/26000) / np.log10(3e-4/1e-3))
# print(np.log10(313/108) / np.log10(4.1e-2/1e-1))


# rho = 1.22e-27
# h = 3.3
# z = 0.09
# M = 4*3.14*rho*h**2*z*kpc**3
# rho = 1.51e-25
# h = 5
# z = 0.1
# M = M +4*3.14*rho*h**2*z*kpc**3
# print('{:.2e}'.format(M/M_sun))

# print(10**(-1.2)*4 / (10**(-1.2)*4+1))
# print(10**(-0.8)*4 / (10**(-0.8)*4+1))
# print(10**(-12+11.05)*4 / (10**(-12+11.05)*4+1))
# print(10**(-1.55)*4 / (10**(-1.55)*4+1))

# beta = 1000e5/c
# gamma = (1/(1-beta**2)**0.5)
# print(beta, gamma*100-100, gamma**3)
# print((1+beta)/(1-beta))
# print(G*1.4*M_sun/R_NS/c**2)
# r_g = 2*G*1.4*M_sun/c**2

# P = 3e4
# tau = 1.5e9*year
# B_array = 10**np.linspace(11,14)
# mu0=B_array*1e18
# k0 = 2*mu0**2*(2*np.pi)**2/c**3/I
# # print(k0)
# t_CF = P**2/2/k0
# plt.xscale('log')
# plt.yscale('log')


# t0 = 3*tau/2 * np.log(P**2/tau/k0 - 1)
# print(t0/Gyr)
# fig, ax = plt.subplots()
# ax.plot(B_array, t_CF/Gyr)
# ax.plot(B_array, t0/Gyr)
# ax.set_xscale('log')
# ax.set_yscale('log')


# omega0 = 1 # 318 
# omega = omega0 / (1-r_g/R_NS)**0.25
# print(omega)

""" Design """
# plt.rcParams['font.family'] = "serif"
# plt.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'
# plt.rcParams['text.usetex'] = True

# v = 100e5
# M_dot = 4 * (G*M_NS)**2 * 1 / v**3
# P_PA = 100000
# P = 20
# R_m = 1e12
# t_A = I / M_dot / R_m**2 * np.log(P_PA/P)
# t_B = 2 * pi * I / (M_dot *(2*G*M_NS*R_m)**0.5) * (1/P - 1/P_PA)
# print(t_B/year)

# print(1.5e-2**(-2/3))
# print("{:.2e}".format(1e5*(2e20/1e18)**(1/3)))

tmult = 1.0
# mpl.rcParams['figure.figsize'] = [5*tmult, 3.5*tmult]
# mpl.rcParams['figure.figsize'] = [10*tmult, 15*tmult]
# mpl.rcParams['lines.linestyle'] = '-'
# mpl.rcParams['axes.titlesize'] = 18 * tmult
# lw0 = 1.0
# mpl.rcParams['lines.linewidth'] = lw0 * tmult
# mpl.rcParams['lines.markersize'] = 4 * tmult
# # tmult = 0.1
mpl.rcParams['axes.labelsize'] = 14 * tmult
mpl.rcParams['xtick.labelsize'] = 14 * tmult
mpl.rcParams['ytick.labelsize'] = 14 * tmult
mpl.rcParams['legend.fontsize'] = 14 * tmult
# fontsize = 16 * tmult
# mpl.rcParams['legend.frameon'] = True
# mpl.rcParams["legend.fancybox"] = False
# mpl.rcParams['legend.facecolor'] = 'white'
# plt.rcParams["xtick.minor.visible"] =  True
# plt.rcParams["ytick.minor.visible"] =  True
# plt.rcParams['font.family'] = "serif"
# plt.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'
# plt.rcParams['text.usetex'] = True
# print(167781/3 + 37656/3)

# print(100*pc/100e5/year/1e6)

# print(6e7*M_sun/14e63/m_p)

""" Custom Milky Way potential """
# def GetPotential(log_M_h, log_r_s, log_M_n, log_a):
#     mw_potential = gp.CCompositePotential()
#     # mw_potential['bulge'] = gp.HernquistPotential(m=5E9, c=1., units=galactic)
#     mw_potential['disk'] = gp.MiyamotoNagaiPotential(m=0.1*6.8E10*u.Msun,
#                                                       a=3*u.kpc, b=280*u.pc,
#                                                       units=galactic)
#     # mw_potential['nucl'] = gp.HernquistPotential(m=np.exp(log_M_n),
#     #                                                c=np.exp(log_a)*u.pc,
#     #                                                units=galactic)
#     # mw_potential['halo'] = gp.NFWPotential(m=np.exp(log_M_h),
#     #       r_s=np.exp(log_r_s), units=galactic)
#     return mw_potential