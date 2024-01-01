# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:19:03 2023

@author: AMD
"""

from model import Simulation, Ejector, Propeller, Accretor
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from numpy import pi, exp
import pandas as pd
from scipy.optimize import brentq
from potential import Trajectory, TrajectoryPlot
from tqdm import tqdm
from constants import c, G, M_sun, m_p, year, birth_age, galaxy_age, R_NS, I
from constants import fontsize


def GetB(B0, field, t):
    """ Returns B(t) or float """
    
    if field == "CF" or not field:
        B = B0
    
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


def IllustrationB():
    t = np.linspace(1000, galaxy_age, 1000000)
    for rr in [11,12,13,14]:
        plt.plot(t/year, GetB(10**rr, 'HA', t)(t))
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('time, yrs')
    plt.ylabel('B, G')


def Track(P0, B0, pos0, vel0, field, t_start, t_end, plot, Misiriotis,
          v=None, n=None):
    """ Returns arrays t, P, B, v, n, evolutionary stages """
    
    if not v or not n:
        """ Initial parameter arrays """
        t, v, n, orbPeriod = Trajectory(pos0, vel0, t_end, plot, Misiriotis)

        v = sp.interpolate.interp1d(t, v, bounds_error=False, kind='linear',
                                    fill_value=np.mean(v))
        n = sp.interpolate.interp1d(t, n, bounds_error=False, kind='linear',
                                    fill_value=1e-6)

    else:
        t, v_1, n_1, orbPeriod = Trajectory(pos0, vel0, t_end, plot,
                                            Misiriotis)
    max_step = orbPeriod / 12
    print('max_step = {:.3e} s'.format(max_step))
    
    """ Filling space between t[0]=0 and t[1]~1e13 s """
    t = t[t < t_end]
    t = np.append(t, t_end)
    if t[1] / t_start > 10:
        add_t = np.linspace(np.log10(t_start), np.log10(t[1]), num=100,
                            endpoint=False)
        add_t = 10**add_t #np.delete(add_t, 0) # without t_start  
        t = np.delete(t, 0) # excluding the first zero
        t = np.append(add_t, t)
    else:
        t[0] = t_start
    
    B = GetB(B0, field, t)
    
    """ First stage """
    t_start=t_start
    NS = Simulation(B, v, n, field, t_start=t_start, t_stop=t_end)
    NS = NS.first_stage(t_start, P0)
    NS_0 = NS
    
    """ Arrays """
    stages = np.array([])
    t_stages = np.array([])
    T = np.array([t_start])
    P = np.array([P0])
    
    AccretorDuration = 0
    PropellerDuration = 0
    EjectorDuration = 0
    GeorotatorDuration = 0
    
    """ Stages calculation """
    while T[-1] < t_end:
        
        stages = np.append(stages, str(NS))
        t_stages = np.append(t_stages, T[-1])
        t = t[t >= T[-1]]
        
        solution, stage = NS.solution(T[-1], np.array([P[-1]]), np.array(t),
                                      max_step)
        
        T = np.append(T, np.array(solution.t))
        P = np.append(P, np.array(solution.y)[0])
        
        duration = np.array(solution.t)[-1] - np.array(solution.t)[0]
        if str(NS)[0] == "A":
            AccretorDuration += duration
        elif str(NS)[0] == "P":
            PropellerDuration += duration
        elif str(NS)[0] == "E":
            EjectorDuration += duration
        elif str(NS)[0] == "G":
            GeorotatorDuration += duration
        else:
            print("There is no such stage as {}".format(str(NS)[0]))
        
        NS = stage
        if NS == None:
            # print(solution.message)
            break
        
    """ Resulting arrays """
    if isinstance(B, float) or isinstance(B, int):
        B = np.zeros(len(T)) + B
    else:
        B = np.array(B(T))
    if isinstance(v, float) or isinstance(v, int):
        v = np.zeros(len(T)) + v
    else:
        v = np.array(v(T))
    if isinstance(n, float) or isinstance(n, int):
        n = np.zeros(len(T)) + n
    else:
        n = n(T)
    
    duration = AccretorDuration, PropellerDuration, EjectorDuration, GeorotatorDuration
    duration = duration / (T[-1] - T[0])
    return T, P, B, v, n, t_stages, stages, NS_0, duration


def TrackPlot(P0, B0, pos0, vel0, field, t_start, t_end, plot, Misiriotis,
              v=None, n=None):
    
    t, P, B, v, n, t_stages, stages, NS, popt = Track(P0, B0, pos0, vel0, field,
                                                t_start, t_end, plot,
                                                Misiriotis, v, n)
    print("number of stages:", len(t_stages), stages, t_stages)
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_axes([0.1, 0.1, 0.9, 0.9], aspect=1)
    # fig, ax = plt.subplots(facecolor='white', layout='constrained')
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    
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
    axs[3].set_xlabel("time, s", fontsize=fontsize) # time in [s]
    
    """ Lines and letters """
    arrs = [P, B, v, n]
    axs[0].set_yscale('log')
    
    t_stages = np.append(t_stages, t_end)
    for i in range(len(axs)):
        axs[i].set_xscale('log')
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
    
    # R_l = np.zeros(len(t))
    # R_c = np.zeros(len(t))
    # R_Sh = np.zeros(len(t))
    # R_G = np.zeros(len(t))
    # R_m = np.zeros(len(t))
    # R_stop = np.zeros(len(t))
    
    R_l = NS.R_l(P)
    R_c = NS.R_c(P)
    R_Sh = NS.R_Sh(t, P)
    R_G = NS.R_G(t)
    if isinstance(R_G, float):
        R_G = np.zeros(len(t)) + R_G
    R_m = NS.R_m(t)
    R_stop = R_m
    if tEstart and tEend:
        R_stop[idxEstart:idxEend] = R_Sh[idxEstart:idxEend]

    # for i in range(len(t)):
    #     R_l[i] = NS.R_l(P[i])
    #     R_c[i] = NS.R_c(P[i])
    #     R_Sh[i] = NS.R_Sh(t[i], P[i])
    #     R_G[i] = NS.R_G(t[i])
    #     R_m[i] = NS.R_m(t[i])
    #     if tEstart < t[i] < tEend:
    #         R_stop[i] = R_Sh[i]
    #     else:
    #         R_stop[i] = R_m[i]
    
    fig, ax = plt.subplots(figsize=(15, 12))
    
    try:
        ax.plot(t, R_G, label='$R_{G}$', ls=':')
        ax.plot(t, R_stop, label='$R_{stop}$', ls='--')
        ax.plot(t, R_c, label='$R_c$', ls='-.')
        ax.plot(t, R_l, label='$R_{l}$', ls=':')
        # ax.plot(t, R_Sh, label='$R_{Sh}$', ls='--')
        # ax.plot(t, R_m, label='$R_{m}$', ls='--')
        ax.set_xlabel("time, s", fontsize=fontsize)
        ax.legend(fontsize=fontsize)
        ax.set_xscale('log')
        ax.set_yscale('log')
    except BaseException:
        print(len(t))
        print(len(R_G))
        print(len(R_stop))
        print(len(R_c))
        print(len(R_l))
    
    Rs = np.array([])
    for Rs_cur in [R_G, R_c, R_l]:
        Rs = np.append(Rs, Rs_cur)
    top = np.min(Rs)
    bottom = np.max(Rs)
    letter_height = (top * bottom)**0.5
    for j in range(len(stages)):
        letter_position = (t_stages[j] * t_stages[j+1])**0.5
        name = stages[j][0]
        ax.text(letter_position, letter_height, name, fontsize=fontsize-2)
        
    for k in range(1, len(stages)):
        ax.vlines(t_stages[k], bottom, top, linestyles="dotted", 
                    colors="k")
    
    ax.tick_params(which='major', width=1.0, length=10, labelsize=fontsize)
    ax.tick_params(which='minor', width=1.0, length=5, labelsize=fontsize-2,
               labelcolor='0.25')


def TrackEjectorPropeller(B, v, n, field, case, P0=None):
    """ time for the first two stages, Ejector and Propeller
    of the NS with the initial period = 0 s"""
    B0 = B
    num = 10000
    t_start = birth_age
    
    NS0 = Simulation(B, v=v, n=n, field=0)
    if field == 'CF' or not field:
        t_end = 10 * (NS0.t_E(0) + NS0.t_P(case))
    elif field == 'HA':
        t_end = 40 * (NS0.t_E(0) + NS0.t_P(case))
    elif field == 'ED':
        t_end = 10 * (NS0.t_E(0) + NS0.t_P(case))
    
    if P0 == None:
        P0 = 1e-3
    t = 10**np.linspace(np.log10(t_start), np.log10(t_end), num)
    B = GetB(B0, field, t)
    NS = Simulation(B=B, v=v, n=n, field=field, t_start=t_start, t_stop=t_end,
                    case=case)
    NS = NS.first_stage(t_start, P0)
    
    """ Arrays """
    # stages = np.array([NS])
    t_stages = np.array([t_start])
    T = np.array([t_start])
    P = np.array([P0])
    P_E = 0
    
    max_step = t[-1]
    """ Stages calculation """
    if str(NS) == "Ejector()":
        # max_step = NS0.t_E(0)
        solution, stage = NS.solution(T[-1], np.array([P0]), np.array(t),
                                      max_step)
        t_stages = np.append(t_stages, np.array(solution.t[-1]))
        
        T = np.append(T, np.array(solution.t))
        P = np.append(P, np.array(solution.y)[0])
        t_E = np.array(solution.t[-1])
        P_E = solution.y[0][-1]
        # B_Array = NS.B(T)
        print("t_E = {:.2e}".format( t_E/year))
    elif str(NS) == "Propeller()":
        # max_step = NS0.t_P(case)
        solution, stage = NS.solution(t_start, np.array([P0]), np.array(t),
                                      max_step)
        NS = stage
        if str(NS) == "Accretor()":
            t_stages = np.append(t_stages, np.array(solution.t[-1]))
            t_E = 0
            t_P = np.array(solution.t[-1])
            print("t_P =", t_P)
            
            T = np.append(T, np.array(solution.t))
            P = np.append(P, np.array(solution.y)[0])
            # B_Array = np.append(NS.B(np.array(solution.t)))
            # return 0, t_P
        else:
            print("Last stage is not Accretor, it is", str(NS))
            return 0, 0
        
    else:
        print("First stage is not Ejector or Propeller, it is", str(NS))
        return 0, 0
    
    
    if str(NS) != 'Accretor()':
        NS = stage
        
        t = 10**np.linspace(0, np.log10(40*NS0.t_P(case)), 100*num)
        if field == "ED":
            t = 10**np.linspace(0, 40*np.log10(NS0.t_P(case)), 100*num)
            t_end = t[-1]
            
        # B = GetB(B0, field, t)
        NS = Simulation(B=B, v=v, n=n, field=field, t_start=t_E, t_stop=t_end,
                        case=case)
        P0 = NS.P_EP(t_E)
        
        NS = NS.first_stage(t_E, 1.01*P0)

        if str(NS) == "Propeller()":
            # max_step = NS0.t_P(case)
            solution, stage = NS.solution(t_E, np.array([P0]), np.array(t),
                                          max_step)
            NS = stage
            if str(NS) == "Accretor()":
                t_stages = np.append(t_stages, np.array(solution.t[-1]))
                t_P = np.array(solution.t[-1])
                print("t_P =", t_P)
                
                T = np.append(T, np.array(solution.t) + t_E)
                P = np.append(P, np.array(solution.y)[0])
                # B_Array = np.append(B_Array, NS.B(np.array(solution.t) + t_E))
            else:
                print("Last stage is not Accretor, it is", str(NS))
                return 0, 0
        else:
            print("Second stage is not Propeller, it is", str(NS))
            return 0, 0
    
    # plt.plot(T, P, ".")
    # plt.plot(t_E, P_E, 'ro')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.plot(T, B_Array, 'black')
    return t_E, t_P

    
# TrackEjectorPropeller(1e14, 30e5, 0.3, 'HA', "B", 100)
# TrackEjectorPropeller(1e15, 30e5, 0.3, 'HA', "B")

# t = 2*1e10*year
# B0 = 1
# t_Ohm = galaxy_age / (4 * np.log(10))
# B=B0 * (exp(-t / t_Ohm))
# print(B)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    