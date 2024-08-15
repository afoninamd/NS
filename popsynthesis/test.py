#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:13:11 2023

@author: afoninamd
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from model import Simulation
import scipy as sp
from main.constants import year
# from main import model
#from main.constants import birth_age, galaxy_age
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from numpy import pi
import pandas as pd
from scipy.optimize import brentq
from potential import Trajectory

from potential import DensityMapPlot
from track import Track, TrackPlot
from potential import Orbit
from tqdm import tqdm
from time import time

to_time = time()

# N = 100
# # Popsynthesis(N, "CF")

# # help(model.Simulation(1,1,1,1))

# # Trajectory(pos=[0,10,0.], vel=[0,0,0], t_end=galaxy_age, plotOrbit=True,
# #                 realisticMap=False) # initial pos, vel


# # TrackPlot(0.1, 1e12, pos0=[0,0,0], vel0=[0,0,150],
# #           field='CF', t_start = birth_age, t_end=galaxy_age, plotOrbit=True,
# #           realisticMap=False)

# # TrackPlot(0.5, 1e12, pos0=[0,1,0], vel0=[30,0,0],
# #       field='CF', t_start = birth_age, t_end=galaxy_age, plot=True, Misiriotis=True)


# # TrackPlot(0.1, 1e14, pos0=[10,0,0], vel0=[0,100,20],
# #       field='ED', t_start = birth_age, t_end=galaxy_age, plot=True, Misiriotis=1, n=0.3, v=30e5)
# """ Const propeller """
# # TrackPlot(0.1, 1e14, pos0=[10,0,0], vel0=[1000,0,0],
# #       field='HA', t_start = birth_age, t_end=galaxy_age, plot=True, Misiriotis=1)
# """ WRONG PROPELLER"""
# # TrackPlot(0.1, 1e14, pos0=[0,0,2], vel0=[100,0,0],
# #       field='HA', t_start = birth_age, t_end=galaxy_age, plot=False, Misiriotis=1)
# """ Accretor / Georotator = 50 / 50 """
# # TrackPlot(0.1, 1e14, pos0=[0,0,2], vel0=[100,0,0],
# #       field='HA', t_start = birth_age, t_end=galaxy_age, plot=True, Misiriotis=1)
# """ Accretor / Propeller = 50 / 50 """
# # TrackPlot(0.1, 1e14, pos0=[0,0,2], vel0=[100,0,10],
# #       field=0, t_start = birth_age, t_end=galaxy_age, plot=True, Misiriotis=True)

# """ 100 NS time, realistic density map """
# # for _ in tqdm(range(100)): # 23:35 100%|██████████| 1000/1000 [01:29<00:00, 11.17it/s]
# #     Track(0.1, 1e14, pos0=[0,0,2], vel0=[100,0,10],
# #           field=0, t_start = birth_age, t_end=galaxy_age, plot=False, Misiriotis=False)

# """ 100 NS time, Misiriotis density model """
# # for _ in tqdm(range(100)): # 04:12
# #     Track(0.1, 1e14, pos0=[0,0,2], vel0=[100,0,10],
# #           field=0, t_start = birth_age, t_end=galaxy_age, plot=False, Misiriotis=True)
# # DensityMapPlot(H2=1)

# def checkR_m():
#     B = 10**np.linspace(9, 30, 5)
#     v = 10e5
#     n = 0.001
#     field = 'CF'
#     t = 5.
    
#     a0 = np.zeros(len(B))
#     a1 = np.zeros(len(B))
#     a2 = np.zeros(len(B))
#     for i in range (len(B)):
#         NS = model.Simulation(B=B[i], v=v, n=n, field=field)
#         a0[i] = NS.R_G(t)
#         a1[i] = NS.R_m(t)
#         a2[i] = NS.R_m_geo(t)
#     plt.plot(B, a0)
#     plt.plot(B, a1)
#     plt.plot(B, a2)
#     plt.yscale('log')
#     plt.xscale('log')


# from_time = time()
# print("\nIt took {:.1f} s".format(from_time-to_time))


# # help(model.Simulation(1,1,1,1))

# # Trajectory(pos=[0,20,0.], vel=[150,0,0], t_end=galaxy_age, plot=True) # initial pos, vel


# # import numpy as np
# # import matplotlib.pyplot as plt
# # def func(x, a, b):
# #     return a * x**6 + b * x**3.5
# # x = np.linspace(-10, 10, 100)
# # a = 1e+6;b = 2;y = func(x, a, b)
# # # plt.figure()
# # plt.plot(x, y)
# # plt.xlabel('x')
# # plt.ylabel('y')
# # plt.grid(True)
# # plt.show()

# def func1():
#     def n(t):
#         T = 1e8
#         return np.sin(t/T)**2
    
#     NS = Simulation(n=n)
#     t = 10**np.linspace(0, 10)*year
#     R_A = np.zeros(len(t))
#     for i in range(len(t)):
#         R_A[i] = NS.R_A(t[i])
#     plt.plot(t, R_A)

# TrackPlot(P0=1e-1, B0=1e12, pos0=[0,3,0.], vel0=[150,0,0], field="CF", t_start=5.,
#                 t_end=galaxy_age, plotOrbit=True, realisticMap=False, v=None,
#                 n=None, t=None, number_of_dots=100, case="A1", Omega=None, c_s=10e5)

# # T, P, B, v, n, t_stages, stages, NS_0, EPAG = Track(P0=1e-1, B0=1e12, pos0=[0,3,0.], vel0=[150,0,0], field="CF", t_start=5.,
# #                t_end=galaxy_age, plotOrbit=True, realisticMap=False, v=None,
# #                n=None, t=None, number_of_dots=10000, case="A", Omega=None)
# # print(t_stages, stages)
# # string = '[2e1 2.00e2 200e6]'
# # string = string.replace('[', '')
# # string = string.replace(']', '')

# # string = string.split(' ')
# # for i in range(len(string)):
# #     string[i] = float(string[i])
    
# # print(string)
# # print(str(string))

# # TrackPlot(0.5, 1e12, pos0=[0,5,0], vel0=[0,0,0],
# #       field='HA', t_start = birth_age, t_end=galaxy_age, plot=True, Misiriotis=True)

# # TrackPlot(0.5, 1e12, pos0=[0,1,0], vel0=[30,0,0],
# #       field='CF', t_start = birth_age, t_end=galaxy_age, plot=True, Misiriotis=True)


# # TrackPlot(0.1, 1e14, pos0=[10,0,0], vel0=[0,100,20],
# #       field='ED', t_start = birth_age, t_end=galaxy_age, plot=True, Misiriotis=1, n=0.3, v=30e5)
# """ Const propeller """
# # TrackPlot(0.1, 1e14, pos0=[10,0,0], vel0=[1000,0,0],
# #       field='HA', t_start = birth_age, t_end=galaxy_age, plot=True, Misiriotis=1)
# """ WRONG PROPELLER"""
# # TrackPlot(0.1, 1e14, pos0=[0,0,2], vel0=[100,0,0],
# #       field='HA', t_start = birth_age, t_end=galaxy_age, plot=False, Misiriotis=1)
# """ Accretor / Georotator = 50 / 50 """
# # TrackPlot(0.1, 1e14, pos0=[0,0,2], vel0=[100,0,0],
# #       field='HA', t_start = birth_age, t_end=galaxy_age, plot=True, Misiriotis=1)
# """ Accretor / Propeller = 50 / 50 """
# # TrackPlot(0.1, 1e14, pos0=[0,0,2], vel0=[100,0,10],
# #       field=0, t_start = birth_age, t_end=galaxy_age, plot=True, Misiriotis=True)

# """ 100 NS time, realistic density map """
# # for _ in tqdm(range(100)): # 23:35 100%|██████████| 1000/1000 [01:29<00:00, 11.17it/s]
# #     Track(0.1, 1e14, pos0=[0,0,2], vel0=[100,0,10],
# #           field=0, t_start = birth_age, t_end=galaxy_age, plot=False, Misiriotis=False)

# """ 100 NS time, Misiriotis density model """
# # for _ in tqdm(range(100)): # 04:12
# #     Track(0.1, 1e14, pos0=[0,0,2], vel0=[100,0,10],
# #           field=0, t_start = birth_age, t_end=galaxy_age, plot=False, Misiriotis=True)
# # DensityMapPlot(H2=1)

# def checkR_m():
#     B = 10**np.linspace(9, 30, 5)
#     v = 10e5
#     n = 0.001
#     field = 'CF'
#     t = 5.
    
#     a0 = np.zeros(len(B))
#     a1 = np.zeros(len(B))
#     a2 = np.zeros(len(B))
#     for i in range (len(B)):
#         NS = model.Simulation(B=B[i], v=v, n=n, field=field)
#         a0[i] = NS.R_G(t)
#         a1[i] = NS.R_m(t)
#         a2[i] = NS.R_m_geo(t)
#     plt.plot(B, a0)
#     plt.plot(B, a1)
#     plt.plot(B, a2)
#     plt.yscale('log')
#     plt.xscale('log')


# from_time = time()
# print("\nIt took {:.1f} s".format(from_time-to_time))



















