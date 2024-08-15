#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:23:41 2024

@author: afoninamd
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from main.model import Simulation, Ejector, Propeller, Accretor
from main.evolution import GetB, Evolution
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from numpy import pi, exp
import pandas as pd
from scipy.optimize import brentq
from potential import Trajectory, TrajectoryPlot
from tqdm import tqdm
from main.constants import c, G, M_sun, m_p, year, birth_age, galaxy_age, R_NS, I, R_sun
from main.constants import fontsize, maxTrackTime
# from SyXRB import func, path, metallicity
from time import time


def Track(P0, B0, pos0=None, vel0=None, field="CF", t_start=False, t_end=False,
          plotOrbit=False, realisticMap=False, v=None, n=None, t=None,
          number_of_dots=10000, case="A", Omega=None, c_s=0, p_out=0):
    """ Returns arrays t, P, B, v, n, evolutionary stages """
    try:
        p_out = sp.interpolate.interp1d(t, p_out, bounds_error=False,
                                    kind='linear', fill_value=np.mean(p_out))
    except Exception as e:
        pass
    if (v is None) or (n is None):
        """ Initial parameter arrays """
        t, v, n, orbPeriod = Trajectory(pos0, vel0, t_end, plotOrbit,
                                        realisticMap)
        max_step = orbPeriod / 12
        # print('max_step = {:.3e} s'.format(max_step))
        """ Filling space between t[0]=0 and t[1]~1e13 s """
        t = t[t < t_end]
        t = np.append(t, t_end)
        if t[1] / t_start > 10:
            add_t = np.linspace(np.log10(t_start), np.log10(t[1]), num=100,
                                endpoint=False)
            add_t = 10**add_t #np.delete(add_t, 0) # without t_star
            t = np.delete(t, 0) # excluding the first zero
            t = np.append(add_t, t)
        else:
            t[0] = t_start
    elif t is None:
        t = 10**np.linspace(np.log10(t_start), np.log10(t_end), number_of_dots)
        max_step = t_end / (number_of_dots // 10)
    elif not (t is None):
        t_start = t[0]
        t_end = t[-1]
        max_step = t_end / number_of_dots
    else:
        t = np.linspace(0., t_end, number_of_dots)
        v = np.zeros(len(t)) + v
        n = np.zeros(len(t)) + n
    
    """ Making functions out of arrays """
    v = (v**2 + c_s**2)**0.5
    # t_ext = np.append(t, np.array([t[-1]*2]))
    v = sp.interpolate.interp1d(t, v, bounds_error=False, kind='linear',
                                fill_value=np.mean(v))
    n = sp.interpolate.interp1d(t, n, bounds_error=False, kind='linear',
                                fill_value=0)
    
    B = GetB(B0, field, t)
    res = Evolution(P0, B0, t, v, n, field, max_step, case, Omega, p_out)
    T, P, t_stages, stages, NS_0, EPAG = res
        
    """ Making arrays from scipy functions """
    if np.isscalar(B):
        B = np.zeros(len(T)) + B
    else:
        B = np.array(B(T))
    if np.isscalar(v):
        v = np.zeros(len(T)) + v
    else:
        v = np.array(v(T))
    if np.isscalar(n):
        n = np.zeros(len(T)) + n
    else:
        n = n(T)
    
    return T, P, B, v, n, t_stages, stages, NS_0, EPAG


def TrackPlot(P0, B0, pos0=None, vel0=None, field="CF", t_start=False,
               t_end=False, plotOrbit=False, realisticMap=False, v=None,
               n=None, t=None, number_of_dots=10000, case="A", Omega=None,
               c_s=0, p_out=0):
    
    res = Track(P0, B0, pos0, vel0,
                                                      field, t_start, t_end,
                                                      plotOrbit, realisticMap,
                                                      v, n, t,
                                                      number_of_dots, case,
                                                      Omega, c_s, p_out)
    t, P, B, v, n, t_stages, stages, NS, popt = res
    t_stages = t_stages / year
    # print("number of stages and t in yrs:", len(t_stages), stages, t_stages, NS)
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_axes([0.1, 0.1, 0.9, 0.9], aspect=1)
    # fig, ax = plt.subplots(facecolor='white', layout='constrained')
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    t = t / year
    
    axs[0].plot(t, P, marker='o')
    axs[0].set_ylabel("P, s", fontsize=fontsize)
    # axs[0].plot(t, NS.dP_dt(t, P))
    axs[1].plot(t, B)
    axs[1].set_ylabel("B, G", fontsize=fontsize)
    axs[2].plot(t, v)
    axs[2].set_ylabel("v, cm/s", fontsize=fontsize)
    axs[3].plot(t, n)#, marker='o')
    axs[3].set_ylabel("n, cm$^{-3}$", fontsize=fontsize)
    
    # axs[0].set_title("Track", fontsize=20, verticalalignment='bottom')
    axs[3].set_xlabel("time, yrs", fontsize=fontsize) # time in [s]
    
    """ Lines and letters """
    arrs = [P, B, v, n]
    axs[0].set_yscale('log')
    
    t_end = t[-1]
    t_stages = np.append(t_stages, t_end)
    for i in range(len(axs)):
        # axs[i].set_xscale('log') 
        top = np.max(arrs[i])
        bottom = np.min(arrs[i])
        min_hight = 2
        if top and bottom:
            if top / bottom < min_hight:
                top = top * min_hight
                bottom = bottom / min_hight
        if not i:
            letter_height = (top * bottom)**0.5
        else:
            letter_height = (top + bottom) / 1.5
        for k in range(1, len(stages)):
            axs[i].vlines(t_stages[k], bottom, top, linestyles ="dotted", 
                        colors ="k")
        if not i:
            for j in range(len(stages)):
                letter_position = (t_stages[j] * t_stages[j+1])**0.5
                name = stages[j][0]
                axs[i].text(letter_position, letter_height, name,
                            fontsize = fontsize-2)
        axs[i].tick_params(which='major', width=1.0, length=10,
                            labelsize=fontsize)
        axs[i].tick_params(which='minor', width=1.0, length=5,
                            labelsize=fontsize-2, labelcolor='0.25')
    """ Critical radii """
    
    # B = sp.interpolate.interp1d(t, B, bounds_error=True, kind='linear',
    #                             fill_value=np.mean(B))
    # v = sp.interpolate.interp1d(t, v, bounds_error=True, kind='linear',
    #                             fill_value=np.mean(v))
    # n = sp.interpolate.interp1d(t, n, bounds_error=True, kind='linear',
    #                             fill_value=1e-6)
    # NS = Simulation(B, v, n, field, t_start=t_start, t_stop=t_end)
    
    """ Find Ejector """
    tEstart, tEend = 0., 0.
    for j in range(len(stages)):
        name = stages[j][0]
        if name == 'E':
            tEstart, tEend = t_stages[j], t_stages[j+1]
    
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    idxEstart, idxEend = find_nearest(t, tEstart), find_nearest(t, tEend)
    
    t = t * year
    # R_l = NS.R_l(P)
    # R_c = NS.R_c(P)
    # R_Sh = NS.R_Sh(t, P)
    # R_G = NS.R_G(t)
    
    # if isinstance(R_G, float):
    #     R_G = np.zeros(len(t)) + R_G
    # if NS.case == "A1":
    #     R_m = NS.R_m_A1(t, P)
    #     R_m_geo = NS.R_m_A1_geo(t, P)
    #     R_m[R_m>R_G] = R_m_geo[R_m>R_G]
    #     R_stop = NS.R_m_A1(t, P)
    # else:
    #     R_m = NS.R_m(t)
    #     R_m_geo = NS.R_m_geo(t)
    #     R_m[R_m>R_G] = R_m_geo[R_m>R_G]
    #     R_stop = NS.R_m(t)
    # if np.isscalar(R_stop):
    #     R_stop = np.zeros(len(t)) + R_stop
    # if tEstart and tEend:
    #     R_stop[idxEstart:idxEend+1] = R_Sh[idxEstart:idxEend+1]
    
    """ We plot every 1000th point in (t, P) """
    # t = t[::1000]
    # P = P[::1000]
    R_l = np.zeros(len(t))
    R_c = np.zeros(len(t))
    R_Sh = np.zeros(len(t))
    R_G = np.zeros(len(t))
    R_m = np.zeros(len(t))
    R_A = np.zeros(len(t))
    R_stop = np.zeros(len(t))
    
    for i in tqdm(range(len(t))):
        R_l[i] = NS.R_l(P[i])
        R_c[i] = NS.R_c(P[i])
        R_Sh[i] = NS.R_Sh(t[i], P[i])
        R_G[i] = NS.R_G(t[i])
        R_m[i] = NS.R_m_cur(t[i], P[i])
        R_A[i] = NS.R_A(t[i])
        if tEstart < t[i] < tEend:
            R_stop[i] = R_Sh[i]
        else:
            R_stop[i] = R_m[i]
    
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # disk1 = np.zeros(len(t))
    # disk2 = np.zeros(len(t))
    # for i in range(len(t)):
    #     disk1[i], disk2[i] = NS.isDisk(t[i], P[i])
    
    try:
        # print('in track.py P[-1] = {:.2e} s'.format(P[-1]))
        # R_A = NS.R_A(t)
        
        t = t / year
        ax.plot(t, R_G, label='$R_{G}$', ls=':')
        # ax.plot(t, R_stop, label='$R_{stop}$', ls='--')
        
        ax.plot(t, R_m, label='$R_{m}$', ls='--')
        ax.plot(t, R_c, label='$R_c$', ls='-.')
        ax.plot(t, R_l, label='$R_l$', ls=':')
        ax.plot(t, R_Sh, label='$R_{Sh}$', ls='--')
        ax.plot(t, R_A, label='$R_{A}$', ls='-', color='k')
        # ax.set_xlabel("time, yrs", fontsize=fontsize)
        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.plot(t, disk1, ls='-', color='blue', label='disk1', alpha=0.5)
        # print('min and max of disk1', np.min(disk1), np.max(disk1))
        # ax.plot(t, disk2, ls='--', color='k', label='1', alpha=0.5)
        # ax.plot(t, abs(disk2-disk1), ls='-', color='red', label='disk3', alpha=0.5)
        
        ax.legend(fontsize=fontsize)
        
    except BaseException as e:
        """ Exception in TrackPlot in track.py """
        print(e)
        print(len(t))
        print(len(R_G))
        print(len(R_stop))
        print(len(R_c))
        print(len(R_l))
    
    Rs = np.array([])
    for Rs_cur in [R_G, R_c, R_l]:
        Rs = np.append(Rs, Rs_cur)
    top = np.min(Rs)
    # top = 1e-1
    bottom = np.max(Rs) #Rs !!!
    letter_height = (top * bottom)**0.5
    for j in range(len(stages)):
        letter_position = (t_stages[j] * t_stages[j+1])**0.5
        name = stages[j][0]
        ax.text(letter_position, letter_height, name, fontsize=fontsize-2)
        
    for k in range(1, len(stages)):
        ax.vlines(t_stages[k], bottom, top, linestyles="dotted", 
                    colors="k")
    ax.set_title('B = {:.2e} G, case {}'.format(B[0], case))
    ax.tick_params(which='major', width=1.0, length=10, labelsize=fontsize)
    ax.tick_params(which='minor', width=1.0, length=5, labelsize=fontsize-2,
               labelcolor='0.25')
    return res