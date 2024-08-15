# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:19:03 2023

@author: AMD
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main.model import Simulation, Ejector, Propeller, Accretor
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, exp
import pandas as pd
from scipy.optimize import brentq
from tqdm import tqdm
from main.constants import c, G, M_sun, m_p, year, birth_age, galaxy_age, R_NS, I, R_sun
from main.constants import fontsize, maxTrackTime
# from SyXRB import func, path, metallicity
from time import time

def GetB(B0, field, t):
    """ Returns B(t) or float """
    
    if field == "CF" or not field:
        B = B0
        # def B(t):
        #     return B0
    
    elif field == "ED":
        t_Ohm = galaxy_age / (4 * np.log(10))
        def B(t):
            return B0 * (exp(-t / t_Ohm))

    elif field == "HA":
        t_Ohm = 1e6 * year
        t_Hall = 1e4 / (B0 / 1e15) * year
        B_ref = B0 * np.exp(-3)
        def B(t):
            B = ((B0 * exp(-t / t_Ohm) /
                  (1 + t_Ohm / t_Hall * (1 - exp(-t / t_Ohm)))))
            if isinstance(t, int) or isinstance(t, float):
                B = min(B, B_ref)
            else:
                B[B < B_ref] = B_ref
            return B
        # t = np.append(t, 2*t[-1])
        # B = B(t)
        # B = sp.interpolate.interp1d(t, B, bounds_error=False,
        #                                 kind='linear', fill_value=B0)
        if isinstance(t, int) or isinstance(t, float):
            B = np.max(B_ref, B(t))
        else:
            t = np.append(t, 2*t[-1])
            B = B(t)
            B = sp.interpolate.interp1d(t, B, bounds_error=False,
                                        kind='linear', fill_value=B0)

    else:
        print("Field model {} does not exist. There are only CF, ED and HA".format(field))

    return B
# t = year*10**np.linspace(1, 10)
# plt.plot(t, GetB(B0=1e15, field='HA', t=t)(t))
# plt.yscale('log')
# plt.xscale('log')

def Evolution(P0, B0, t, v, n, field, max_step, case, Omega, p_out):
    # B=1e12, v=100e5, n=1., field='CF', case='A', Omega=None, p_out=0.
    """ First stage """
    t_start = t[0]
    B = GetB(B0, field, t)
    NS = Simulation(B, v, n, field, case, Omega, p_out)
    NS_0 = NS
    NS = NS.first_stage(t_start, P0)
    
    """ Arrays """
    stages = np.array([])
    t_stages = np.array([])
    T = np.array([t_start])
    P = np.array([P0])
    
    """ All stages """
    EPAG = np.zeros(4)
    EPAG_name = ['E', 'P', 'A', 'G']
    
    to_time = time()
    error_set = [0], [0], [0], [0], NS_0, np.zeros(4)
    """ Stages calculation """
    while T[-1] < t[-1]:#0.9:
        from_time = time()
        if from_time-to_time > maxTrackTime:
            break
        
        stage_name = NS.title()
        print(stage_name)

        if len(stages) > 100:
            break
    
        stages = np.append(stages, stage_name)
        t_stages = np.append(t_stages, T[-1])
        t = t[t>=T[-1]]
        
        # print(T[-1], np.array([P[-1]]), np.array(t), max_step)
        message, solution, stage = NS.solution(T[-1], np.array([P[-1]]),
                                               np.array(t), max_step)
        # if len(stages) > 2:
        #     max_step = min(max_step, abs(t_stages[-1]-t_stages[-2]))
        if stage == "settle":
            T_cur, P_cur = solution
        elif message == "Georotator":
            T_cur, P_cur = solution
        else:
            T_cur =  np.array(solution.t)
            P_cur = np.array(solution.y)[0]
        
        T = np.append(T, np.array(T_cur))
        P = np.append(P, np.array(P_cur))
        
        if P[-1] == np.inf:
            idx = np.where(P==np.inf)[0][0]
            P = P[1:idx]
            T = T[1:idx]
            break
        
        
        duration = T_cur[-1] - T_cur[0]
        
        for i in range(len(EPAG)):
            if EPAG_name[i] == stage_name:
                EPAG[i] += duration
        # else:
        #     print("There is no such stage as {}".format(stage_name))
        
        NS = stage
        if NS == None:
            # print(solution.message)
            break

    # """ Only for popsynthesis-like calculations """
    # if solution.status != 0:
    #     print("\nError in track.Evolution\n")
    #     print(solution.message)
    #     return error_set

    EPAG = EPAG / (T[-1] - T[0])
    
    T = T[1:len(T)]
    P = P[1:len(P)]
    t_stages = np.append(t_stages, T[-1])
    
    return T, P, t_stages, stages, NS_0, EPAG
