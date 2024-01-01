#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:13:11 2023

@author: afoninamd
"""

# from model import Simulation
import scipy as sp
import model
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from numpy import pi
import pandas as pd
from scipy.optimize import brentq
from potential import Trajectory
from constants import birth_age, galaxy_age
from potential import DensityMapPlot
from track import Track, TrackPlot, TrackEjectorPropeller
from potential import Orbit
from tqdm import tqdm
from time import time
from constants import year
to_time = time()


# help(model.Simulation(1,1,1,1))

Trajectory(pos=[0,20,0.], vel=[150,0,0], t_end=galaxy_age, plot=True,
               Misiriotis=False) # initial pos, vel

# TrackPlot(0.5, 1e12, pos0=[0,5,0], vel0=[0,0,0],
#       field='HA', t_start = birth_age, t_end=galaxy_age, plot=True, Misiriotis=True)

# TrackPlot(0.5, 1e12, pos0=[0,1,0], vel0=[30,0,0],
#       field='CF', t_start = birth_age, t_end=galaxy_age, plot=True, Misiriotis=True)


# TrackPlot(0.1, 1e14, pos0=[10,0,0], vel0=[0,100,20],
#       field='ED', t_start = birth_age, t_end=galaxy_age, plot=True, Misiriotis=1, n=0.3, v=30e5)
""" Const propeller """
# TrackPlot(0.1, 1e14, pos0=[10,0,0], vel0=[1000,0,0],
#       field='HA', t_start = birth_age, t_end=galaxy_age, plot=True, Misiriotis=1)
""" WRONG PROPELLER"""
# TrackPlot(0.1, 1e14, pos0=[0,0,2], vel0=[100,0,0],
#       field='HA', t_start = birth_age, t_end=galaxy_age, plot=False, Misiriotis=1)
""" Accretor / Georotator = 50 / 50 """
# TrackPlot(0.1, 1e14, pos0=[0,0,2], vel0=[100,0,0],
#       field='HA', t_start = birth_age, t_end=galaxy_age, plot=True, Misiriotis=1)
""" Accretor / Propeller = 50 / 50 """
# TrackPlot(0.1, 1e14, pos0=[0,0,2], vel0=[100,0,10],
#       field=0, t_start = birth_age, t_end=galaxy_age, plot=True, Misiriotis=True)

""" 100 NS time, realistic density map """
# for _ in tqdm(range(100)): # 23:35 100%|██████████| 1000/1000 [01:29<00:00, 11.17it/s]
#     Track(0.1, 1e14, pos0=[0,0,2], vel0=[100,0,10],
#           field=0, t_start = birth_age, t_end=galaxy_age, plot=False, Misiriotis=False)

""" 100 NS time, Misiriotis density model """
# for _ in tqdm(range(100)): # 04:12
#     Track(0.1, 1e14, pos0=[0,0,2], vel0=[100,0,10],
#           field=0, t_start = birth_age, t_end=galaxy_age, plot=False, Misiriotis=True)
# DensityMapPlot(H2=1)

def checkR_m():
    B = 10**np.linspace(9, 30, 5)
    v = 10e5
    n = 0.001
    field = 'CF'
    t = 5.
    
    a0 = np.zeros(len(B))
    a1 = np.zeros(len(B))
    a2 = np.zeros(len(B))
    for i in range (len(B)):
        NS = model.Simulation(B=B[i], v=v, n=n, field=field)
        a0[i] = NS.R_G(t)
        a1[i] = NS.R_m(t)
        a2[i] = NS.R_m_geo(t)
    plt.plot(B, a0)
    plt.plot(B, a1)
    plt.plot(B, a2)
    plt.yscale('log')
    plt.xscale('log')


from_time = time()
print("\nIt took {:.1f} s".format(from_time-to_time))



















