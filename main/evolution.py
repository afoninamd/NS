#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 10:06:32 2025

@author: afoninamd
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from time import time
from main.model import Object
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from main.constants import galaxy_age, year, G, M_NS, Gyr
t_Ohm = 1e6 * year


def star_formation_history(t: np.array):
    """ interpolation from Haywood 2016, Fig.4, fine blue line """
    t = t / Gyr
    SFR = np.zeros(len(t))
    SFR[t<=7] = 0.1
    cond1 = np.logical_and(t>8.5, t<10)
    SFR[cond1] = 0.5 * 2 / 3 * (t[cond1]-8.5)
    cond2 = np.logical_and(t>10, t<=12.5)
    SFR[cond2] = 0.5
    SFR[t>12.5] = 0.5 - 0.5/(13.8-12.5) * (t[t>12.5]-12.5)
    SFR[SFR<0] = 0
    t = t * Gyr
    return SFR  # / np.sum(SFR) # 2.65 --analytical coefficient # M_sun per second


def gett(t_end: float, n: int) -> np.array:
    """
    Returns an array of t suitable for an ejector stage,
    it mean that since P is proportional to t**0.5,
    the time steps are distributed as t(i+1) = t(i) * 2
    n -- the number of step or the lenth of t array
    """
    t_max = t_end / n # original constant step
    t_min = 3_000 # find it from P/dP_dt for a large field
    leng = int(np.log2(t_max/t_min))
    t1 = 2**np.linspace(0, np.log2(t_max/t_min), leng) * t_min
    t1 = np.append(np.array([0]), t1)
    # t2 = t2[t2>t1[-1]]
    # t1 = np.linspace(0, t2[0], 20, endpoint=False)
    t2 = np.linspace(t1[-1], t_end, n-leng, endpoint=True)
    t2 = t2[1:len(t2)]
    t = np.append(t1, t2)
    return t


def gettHall(B0: float) -> float:
    return 1e4 / (B0 / 1e15) * year


def gettAttractor(B0: float) -> float:
    t_Hall = gettHall(B0)
    A = t_Ohm / t_Hall
    x = (1 + A) / (np.exp(3) + A)
    t = -np.log(x) * t_Ohm
    return t


def getB(t: np.array, B0: float, field: str) -> np.array:
    """
    Returns an array of B of length of the time array
    """
    if field == 'CF':  # constant field
        B = B0 + np.zeros(len(t))
    
    elif field == 'ED':  # exponential decay
        t_char = galaxy_age / (4 * np.log(10))
        B = B0 * (np.exp(-t / t_char))
    
    elif field == 'HA':
        t_Hall = gettHall(B0)
        B = ((B0 * np.exp(-t / t_Ohm) /
              (1 + t_Ohm / t_Hall * (1 - np.exp(-t / t_Ohm)))))
        B_ref = B0 * np.exp(-3) # approximately 3 e-foldings
        B[B<B_ref] = B_ref
    else:
        print("There are only 'CF', 'ED' and 'HA' field models")
    
    return B


def getMdot(rho: np.array, v: np.array, t: np.array) -> np.array:
    """
    Returns an array of accretion rate values of length of the time array
    """
    R_G = G * M_NS / v**2
    Mdot = np.pi * R_G**2 * rho * v
    return Mdot


def find_stage(t_cur: float, P_cur: float, NS: Object,
               previous_stage: int) -> int:
    """
    Returns one of four numbers, each one corresponds to an evolutionary stage:
        1 - ejector
        2 - propeller
        3 - accretor
        4 - georotator
    To find the first stage, assume that the previous stage is ejector,
    i.e. previous_stage = 1,
    """
    if P_cur < NS.P_PE(t_cur):
       stage = 1 # Ejector for sure
    elif P_cur > NS.P_PA(t_cur):
       stage = 3 # Accretor for sure
    elif P_cur > NS.P_AP(t_cur):
        if previous_stage < 3:
            stage = 2 # Propeller if there are no accretors in the past
        else:
            stage = 3 # Accretor othewise
    elif P_cur > NS.P_EP(t_cur):
        stage = 2 # Propeller for sure
    elif P_cur > NS.P_PE(t_cur):
        if previous_stage > 1:
            stage = 2 # Propeller if there is a propeller in the past
        else:
            stage = 1 # Ejector otherwise
    """ Georotator """
    if stage == 3:
        if NS.R_A(t_cur) > NS.R_G(t_cur): # if magnetosphere is large
            stage = 4 # Georotator
    return stage # a number from 1 to 4


def evolution(t: np.array, P0:float, B: np.array,
              Omega: float, case: str, Mdot: np.array,
              v: np.array, plot: bool):
    """
    Calculates the spin period evolution and the evolutionary stages,
    the arrays t, B, v, Mdot are of the same length
    """
    leng = len(t)
    
    P = np.zeros(leng) # an empty array for the spin period
    P[0] = P0 # the first element of the spin period array is P0

    stages = np.zeros(leng, dtype=int) # an empty array for stages
    
    NS = Object(B=B[0], v=v[0], Mdot=Mdot[0], case=case,
                Omega=Omega) # for the first stage
    stages[0] = find_stage(t_cur=t[0], P_cur=P[0], NS=NS, previous_stage=1)
    
    for i in (range(1, leng)):
        """ Define the stage and find dP """
        dP_dt = [NS.dP_dt_ejector, NS.dP_dt_propeller, NS.dP_dt_accretor,
                 NS.dP_dt_georotator]
        P[i] = P[i-1] + (t[i]-t[i-1]) * dP_dt[stages[i-1]-1](t[i-1], P[i-1])
        NS = Object(B=B[i], v=v[i], Mdot=Mdot[i], case=case,
                    Omega=Omega)  # define the properties of the NS at t[i]
        stages[i] = find_stage(t_cur=t[i], P_cur=P[i], NS=NS,
                           previous_stage=stages[i-1])
        
    if plot:
        P_EP = np.zeros(leng)
        P_PE = np.zeros(leng)
        P_PA = np.zeros(leng)
        P_AP = np.zeros(leng)
        for i in (range(0, leng)):
            NS = Object(B=B[i], v=v[i], Mdot=Mdot[i], case=case, Omega=Omega)
            P_EP[i] = NS.P_EP(t[i])
            P_PE[i] = NS.P_PE(t[i])
            P_PA[i] = NS.P_PA(t[i])
            P_AP[i] = NS.P_AP(t[i])
        
        
        fig = plt.figure(figsize=(6, 10))
        
        # Define GridSpec with relative heights: 1/2 for top, 1/6 for each of the three bottom plots
        gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 1, 1])  # total sum=6; top=3/6=1/2
        
        # Create subplots with shared x-axis
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        ax2 = fig.add_subplot(gs[2], sharex=ax0)
        ax3 = fig.add_subplot(gs[3], sharex=ax0)
        for ax in [ax0, ax1, ax2]:
            ax.tick_params(labelbottom=False)
        
        # fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(6,8))
        plt.subplots_adjust(hspace=0)
        
        ax = ax0
        ax.set_ylabel('P, s')
        ax.fill_between(t/Gyr, 0, P_EP, color='C0')
        ax.plot(t/Gyr, P_EP, color='white', ls='-')
        ax.plot(t/Gyr, P_PE, color='white', ls='--')
        ax.fill_between(t/Gyr, P_PA, t*0+1.5*max(P), color='C1')
        ax.fill_between(t/Gyr, P_EP, P_PA, color='#aec7e8')
        ax.plot(t/Gyr, P_PA, color='white', ls='-')
        ax.plot(t/Gyr, P_AP, color='white', ls='--')
        ax.plot(t/Gyr, P, 'k', marker='o', markersize=7)
        
        ax.set_title(case)
        # Define colors for each stage
        colors = {
            1: 'C0',
            2: '#aec7e8',
            3: 'C1',
            4: '#ffbb78'
        }
        # Plot dotted line with different colors for each stage
        stages_np = np.array(stages)
        for stage in range(1, 5):
            mask = stages_np == stage
            ax.plot(t[mask]/Gyr, P[mask], 'o', markersize=4,
                    ls='none', color=colors[stage])
        
        ax.set_yscale('log')
        ax.set_xlim([t[0]/Gyr, t[-1]/Gyr])
        ax.set_ylim(0, 1.5*max(P))
        
        ax = ax1
        ax.set_ylabel('B, G')
        ax.plot(t/Gyr, B)
        ax.set_yscale('log')
        
        ax = ax2
        ax.set_ylabel('M_dot, g/s')
        ax.plot(t/Gyr, Mdot)
        ax.set_yscale('log')
        
        ax = ax3
        ax.set_ylabel('v, km/s')
        ax.set_xlabel('t, Gyr')
        ax.plot(t/Gyr, v/1e5)
        ax.set_yscale('log')
        
    return t, P, stages


def increase_time_resolution(t, P, stages):
    """
    Increase time resolution by inserting points where P jumps significantly.
    """
    t = np.array(t)
    P = np.array(P)
    new_t = [t[0]]  # start with first time point

    for i in range(1, len(t)):
        P_step = abs(P[i] - P[i-1]) / P[i-1]
        if P_step > 2:  # threshold for jump
            num_t = max(1, int(P_step))
            # Insert points between t[i-1] and t[i]
            inserted_points = np.linspace(t[i-1], t[i], 2 * num_t + 1,
                                          endpoint=False)
            new_t.extend(inserted_points)
        else:
            # No jump; just keep previous point
            new_t.append(t[i])

    return np.array(new_t)


def test_evolution():
    """
    The plot "time versus number of steps" is linear:
        1_000_000 steps - 14 s for CF and ED, 18 s for HA
    """
    def one_try(n):
        """ Return the time needed for one calculation of the evolution """
    
        t = gett(t_end=galaxy_age, n=n)
        B = getB(t, B0=1e13, field='ED')
        v = t*0 + 100e5
        Mdot = t*0 + 1e12
        
        time0 = time() # mark the time
        t, P, stages = evolution(t=t, P0=1e-1, B=B, Omega=None, case='A',
                                 Mdot=Mdot, v=v, plot=1)
        
        return time()-time0 # the time needed for n steps
    
    N = [1000]
    # N = [100, 300, 1000, 3000, 10_000, 30_000, 100_000] #, 300_000, 1_000_000]
    time_array = np.zeros(len(N))
    for i in range(len(N)):
        time_array[i] = one_try(N[i])
    
    plt.plot(N, time_array)

