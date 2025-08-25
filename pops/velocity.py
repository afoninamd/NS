#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 18:02:52 2024

@author: afoninamd
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
from main.constants import mw

def v_circ_1d(x, y):
    """ Requires kpc, returns km/s, one-dimensional """
    R = (x**2 + y**2)**0.5 # [kpc]
    if np.isscalar(R):
        num = 1
        ze = 0
    else:
        num = len(R)
        ze = np.zeros(num)
    pos = np.array([R, ze, ze])
    v_circ = mw.circular_velocity(pos).value # [km/s]
    return v_circ # [km/s]

# print(v_circ_1d(0.4, 1.6))

def MW_circular_plot():
    x = np.linspace(0.01, 20)
    v = np.zeros(len(x))
    for i in range(len(x)):
        v[i] = v_circ_1d(x[i], 0)
    fig, ax = plt.subplots()
    plt.plot(x, v)


def v_circ_2d(v, x, y):
    """ Counter-clockwise, requires km/s and kpc, returns km/s, two-dimensional """
    v_circ = v # [km/s]
    R = (x**2 + y**2)**0.5 # [kpc]
    sin_phi = y / R
    cos_phi = x / R
    return -sin_phi*v_circ, cos_phi*v_circ # v_x, v_y [km/s]


def v_ISM(x, y, z):
    """ counter-clockwise """
    grad = 15 # [km/s / kpc]
    v_halo = 180 # [km/s]
    v = v_circ_1d(x, y)
    v_ISM_module = v
    
    cond1 = np.logical_and(z>1, v>v_halo)
    v_ISM_module[cond1] = np.maximum(v_halo,
                                     v_circ_1d(x, y)[cond1] - z[cond1]*grad) # [km/s]
    v_ISM = v_circ_2d(v_ISM_module, x, y)
    return v_ISM # [v_x, v_y] km/s

def v_relative_to_ISM(v_x, v_y, x, y, z):
    """ Array of velocities v -> relative to ISM """
    v_x_ISM, v_y_ISM = v_ISM(x, y, z) # [km/s]
    v_x -= v_x_ISM
    v_y -= v_y_ISM
    return v_x, v_y


def v_test():
    """ vector-diagram """
    # fig, ax = plt.subplots()
    # phi = np.linspace(0, 2*np.pi)
    # rho = 8
    # x = rho * np.cos(phi)
    # y = rho * np.sin(phi)
    # z = np.zeros(len(phi)) + 10
    # v_x, v_y = v_circ_2d(v_circ_1d(x, y), x, y)
    # plt.quiver(x, y, v_x, v_y) 
    
    """ module """
    fig, ax = plt.subplots()
    x = np.linspace(0.01,20, 100) # kpc
    y = x #np.zeros(len(x))
    for z0 in [0, 1, 1.5, 2, 4]:
        z = np.zeros(len(x))+z0
    
        v_x, v_y = v_ISM(x, y, z)
        v = (v_x**2 + v_y**2)**0.5
        
        # plt.plot(x, v_circ_1d(x, y)) # z = 0
        plt.plot(x, v) # z != 0
# v_test()