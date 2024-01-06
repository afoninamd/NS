#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:37:55 2023

@author: afoninamd
"""


from astropy import constants as const

""" Physical constants """
c = const.c.cgs.value                  # speed of light in vacuum,   [cm / s]
G = const.G.cgs.value                  # gravitational constant,     [cm**3 / (g * s**2)]
M_sun = const.M_sun.cgs.value          # solar mass,                 [g]
m_p = const.m_p.cgs.value              # proton mass,                [g]
year = 365.24219878 * 24 * 60 * 60     # year,                       [s]
birth_age = 5                          # [s]
galaxy_age = 1.361e10 * year           # [s]
R_NS = 1e6                             # [cm]
I = 1e45                               # [g cm**2]
sunPos = 8.0                           # [kpc] from Yao2017 8.3
pc = 3.08e18                           # [cm]
rgal = 20                              # [kpc]

""" Other """
fontsize = 16
# number_of_dots = 1000