#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 22:00:05 2024

@author: afoninamd
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main.constants import m_p, k_B, e, h, pi, c
from main.constants import G, M_sun, year
import numpy as np

Z = 1

def g_ff(v, w):
    """ Gaunt factor, p. 171 RybLightman """
    b_min = 4*Z*e**2 / (pi*m_p*v**2) # if mv**2/2<<Z**2*Ry
    b_max = v / w
    return 3**0.5/pi * np.log(b_max/b_min)

def a_ff(T, n, v, w):
    nu = w/2/pi
    return 0.018*T**(-1.5)*Z**2*n**2*nu**(-2)*g_ff(v, w)

v = 400e5
R_G = 2*G*1.4*M_sun/v**2
print(R_G/v/3600)
M_dot = 1e12
dE = 2/15 * M_dot * R_G * v # delta in grav energy of the 3/2 hot envelope
rho = M_dot / (4*R_G**2*v)
B = 1e9
P = 1e-1
mu = B*1e6**3
omega = 2*pi/P
L = 2*mu**2*omega**4/c**3
print(dE/L/3600)

# def tau_L():
    