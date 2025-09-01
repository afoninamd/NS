#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 15:29:05 2025

@author: afoninamd
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from main.constants import rgal, mw, N, output_dir

"""  CREATING THE DISTRIBUTION  """

def B_pulsar(N: int) -> np.array:
    B_power = np.random.uniform(low=10, high=15, size=N)
    return 10**B_power


def P_pulsar(N: int) -> np.array:
    P_power = np.random.uniform(low=-2, high=3, size=N)
    return 10**P_power


def sample_spherical(N, ndim=3):
    """ Uniform spaced vectors """
    vec = np.random.randn(ndim, N)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def V(N: int):
    """ Arrays of Vx, Vy, Vz in cm/s """
    
    xi, yi, zi = sample_spherical(N)
    
    V = np.random.uniform(low=0, high=500e5, size=N)
    Vx, Vy, Vz = V*xi, V*yi, V*zi
    
    return Vx, Vy, Vz  #  cm/s


def Vpec(x: np.array, y: np.array, N: np.array,
         rlim=rgal):
    """ One set of peculiar velocities """
    D = 6e-7 # (km/s)^2 / year
    tau = 0 # year, progenitor star age
    A = 0.7
    c_0 = 10 # km/s
    
    """ Defining sigmas for distributions """
    pot = mw
    r = (x**2 + y**2)**0.5
    dr = rlim / N
    dx = dr / r * x
    dy = dr / r * y
    
    """ Reshaped position array """
    
    pos_1 = np.array([x+dx, y+dy, np.zeros(N)])
    pos_2 = np.array([x-dx, y-dy, np.zeros(N)])
    
    cvel_1 = pot.circular_velocity(pos_1)[0].value 
    cvel_2 = pot.circular_velocity(pos_2)[0].value
    Omega = pot.circular_velocity([x, y, np.zeros(N)])[0].value
    diffOmega = (cvel_1 - cvel_2) / (2 * dr) / r
    kappa = 2 * Omega * (1 + r / (2 * Omega) * diffOmega)**0.5
    B = kappa / (2 * Omega)
    c_r = (c_0**2 + D * tau) / (1 + A**2 + B**2)
    c_phi = B * c_r
    c_z = A * c_r
    
    """ Velocities generation """
    v_r = np.random.normal(0, scale=c_r)
    v_phi = np.random.normal(0, scale=c_phi)
    v_z = np.random.normal(0, scale=c_z)
    
    """ To cartesian coordiates """
    sin_phi = y / r
    cos_phi = x / r
    v_x = v_r * cos_phi - v_phi * sin_phi
    v_y = v_r * sin_phi + v_phi * cos_phi
    
    """ To cm/s """
    v_x = v_x * 1e5
    v_y = v_y * 1e5
    v_z = v_z * 1e5
    
    return v_x, v_y, v_z


def XYZ(N=1, rlim=rgal, arms=False, Yao2017=True):
    """ Initial coordinates distribution and pecular velocity distribution """

    """ integrated PDF """
    # i = 10000
    # sum_rho = np.zeros(i)
    # r = np.linspace(0, 30, i)
    # for j in range(1, i):
    #     sum_rho[j] = sum_rho[j-1] + (r[j] - r[j-1]) * 2 * pi * r[j] * PDF(r[j])
    # plt.plot(r, sum_rho/sum_rho[-1])
    # return 0

    phi = np.random.uniform(low=0, high=2*np.pi, size=N)
    r = np.random.uniform(low=0, high=20, size=N)  # kpc
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    # x_list = []
    # y_list = []
    
    # while len(x_list) < N:
    #     # Generate candidate points
    #     x_candidate = np.random.uniform(-rgal, rgal, size=N)
    #     y_candidate = np.random.uniform(-rgal, rgal, size=N)
        
    #     # Compute the condition
    #     mask = x_candidate**2 + y_candidate**2 <= rgal**2
        
    #     # Select points that satisfy the condition
    #     x_filtered = x_candidate[mask]
    #     y_filtered = y_candidate[mask]
        
    #     # Append to lists
    #     x_list.extend(x_filtered)
    #     y_list.extend(y_filtered)
        
    #     # If we've gathered enough points, truncate
    #     if len(x_list) > N:
    #         x_list = x_list[:N]
    #         y_list = y_list[:N]
    
    # # Convert to numpy arrays
    # x = np.array(x_list)
    # y = np.array(y_list)
    z = np.random.uniform(-0.5, 0.5, size=N)
    
    v_x, v_y, v_z = Vpec(x, y, N)
    
    return x, y, z, v_x, v_y, v_z # kpc and cm/s


def distr(f, Nmax):
    """ Generates distribution of the value from the function f """
    print(str(f)[10] + " distribution")
    if f == XYZ:
        x, y, z, v_x, v_y, v_z = XYZ(Nmax)
        array = np.zeros((6, Nmax))
        array[0] = x
        array[1] = y
        array[2] = z
        array[3] = v_x
        array[4] = v_y
        array[5] = v_z
    elif f == V:
        array = np.zeros((3, Nmax))
        x, y, z = V(Nmax)
        array[0] = x
        array[1] = y
        array[2] = z
    else:
        array = f(Nmax)
    
    return array


def distribution(N=100):
    """ Creating arrays of p, B, M, V """
    distrV = distr(V, N)
    distrX = distr(XYZ, N) # XYZ and peculiar VxVyVz
    
    P_array = distr(P_pulsar, N)
    B_array = distr(B_pulsar, N)
    
    df = pd.DataFrame({'P': P_array, 'B': B_array,
                       'x': distrX[0], 'y': distrX[1], 'z': distrX[2],
                       'Vx': distrV[0], 'Vy': distrV[1], 'Vz': distrV[2],
                       'Vxpec': distrX[3], 'Vypec': distrX[4],
                       'Vzpec': distrX[5]})
    
    pd.DataFrame.to_csv(df, 'distribution_{}.csv'.format(N), sep=';')


def test_distribution(N):
    """ Pictures of generated distribution """
    fig, ax = plt.subplots()
    data = pd.read_csv('distribution_{}.csv'.format(N), sep=';')
    # Array = (data['Vx']**2 + data['Vy']**2 + data['Vz']**2)**0.5
    Array = data['z']
    # Array = np.log10(Array)
    # print(np.mean(Array), np.std(Array))
    # Array = (data['x']**2 + data['y']**2)**0.5
    
    plt.hist(Array, bins=10, alpha=0.5)
    # plt.scatter(data['x'], data['y'])


if not os.path.exists(f'distribution_{N}.csv'):
    print('new distribution is in process')
    distribution(N)