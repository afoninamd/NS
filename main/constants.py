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

mw = gp.BovyMWPotential2014() # from Bovy 2015

""" Physical constants """
c = const.c.cgs.value                  # speed of light in vacuum,   [cm / s]
G = const.G.cgs.value                  # gravitational constant,     [cm**3 / (g * s**2)]
M_sun = const.M_sun.cgs.value          # solar mass,                 [g]
R_sun = const.R_sun.cgs.value
k_B = const.k_B.cgs.value
e = 4,8032067991251e-10
h = const.h.cgs.value
m_p = const.m_p.cgs.value              # proton mass,                [g]
AU = 1.496e+13                         # astronomical unit,          [cm]
day = 24 * 60 * 60                     # [s]
year = 365.24219878 * day              # year,                       [s]
birth_age = 5                          # [s]
galaxy_age = 1.361e10 * year           # [s]
giant_age = 30e6 * year                # [s]
R_NS = 1e6                             # [cm]
I = 1e45                               # [g cm**2]
sunPos = 8.0                           # [kpc] from Yao2017 8.3
pc = 3.08e18                           # [cm]
rgal = 20                              # [kpc]
L_sun = 3.827 * 10**33                 # [erg / s]
T_eff_sun = 5780                       # [K]
sigma_B = const.sigma_sb.cgs.value        # Stefan-Boltzmann constant sigma
sigma_T = 6.652e-29 * 1e4 # [cm**2]
MdotEdd = 4 * pi * m_p * c / sigma_T * R_NS #/M_sun * year#/ (1.4 * M_sun)
# print('MdotEdd {:.3e}'.format(MdotEdd))
Time = 2 * pi * 1000 * R_sun / (G * 15.5 * M_sun / 1000 / R_sun)**0.5
# print('{:.3e}'.format(Time/year))
""" Other """
maxTrackTime = 100 # s
fontsize = 24
giant_age0 = 7e6 * year
# print('{:.2e}'.format((k_B*1e4/m_p)**0.5/1e5))
# print(10**(-0.12))
# print(R_sun/AU)
# omega = 2 * pi /30/3600/24

# print((2*G*M_sun/R_sun)**0.5/1e5*0.329/5**0.5)
# print(30*R_sun/AU)
# print(0.2*(1-0.4)*AU/R_sun)
# T = np.array([189, 1046])
# a = (2.4*(T/365.425)**2)**(1/3)
# T = (1/2.4)**0.5
# print(a, T*365.2425)
# print("{:.2e}".format(8e13/4**(28/5)))

# B_gal = 1e-6
# p_ram = 1 * m_p * 1e6**2
# p_B_gal = B_gal**2 / 8/ pi
# print(p_B_gal/p_ram)
# print((G*M_sun/R_sun**2)/100)
# print(0.15*AU/R_sun)
# print("{:.2e}".format(1e36/G*R_NS/1.4/M_sun))
# amin = (22/365.2425)**(2/3) * (2.4)**(1/3)
# print(amin)
# print(50**0.75)

""" Design """
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'
plt.rcParams['text.usetex'] = True

tmult = 1
mpl.rcParams['figure.figsize'] = [10*tmult, 7*tmult]
# mpl.rcParams['figure.figsize'] = [10*tmult, 15*tmult]
mpl.rcParams['lines.linestyle'] = '-'
mpl.rcParams['axes.titlesize'] = 16 * tmult
lw0 = 1.0
mpl.rcParams['lines.linewidth'] = lw0 * tmult
mpl.rcParams['lines.markersize'] = 4 * tmult
mpl.rcParams['axes.labelsize'] = 16 * tmult
mpl.rcParams['xtick.labelsize'] = 16 * tmult
mpl.rcParams['ytick.labelsize'] = 16 * tmult
mpl.rcParams['legend.fontsize'] = 14 * tmult
mpl.rcParams['legend.frameon'] = True
mpl.rcParams["legend.fancybox"] = False
mpl.rcParams['legend.facecolor'] = 'white'
plt.rcParams["xtick.minor.visible"] =  True
plt.rcParams["ytick.minor.visible"] =  True
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'
plt.rcParams['text.usetex'] = True