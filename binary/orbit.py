#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 14:30:27 2026

@author: afoninamd
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main.model import Object
from main.evolution import getB, getMdot, find_stage # they return np.arrays
from main.constants import m_p, AU, year, Gyr, R_sun, G, M_sun
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def gett_binary(num=10000):
    """ For the evolution with a sun-like companion """
    t0 = 1e6*year
    t1 = 10**np.linspace(np.log10(t0), np.log10(0.1*Gyr), endpoint=False, num=num)
    t2 = 10**np.linspace(np.log10(0.1*Gyr), np.log10(1.0*Gyr), endpoint=False, num=num)
    # t3 = 10**np.linspace(np.log10(1.0*Gyr), np.log10(1.1e10*year), endpoint=True, num=num)
    t3 = np.linspace((1.0*Gyr), (11*Gyr), endpoint=True, num=3*num)
    t = np.append(t1, t2)
    t = np.append(t, t3)
    return t


def wp(t):
    """ The solar wind parameters for a given t """
    
    def Mdot_t(t):
        """ The sun's mass loss rate """
        s = 0.75
        Mdot= 2e-14*M_sun/year * (4.6*Gyr/t)**s
        # print("{:.2e}".format(2e-14*(4.6*Gyr/0.1/Gyr)**0.75))
        return Mdot
    
    def T_t(t):
        """ The temperature at the base r=r0 """
        T0 = 1.5e6 * (4.6*Gyr/t)**0.05
        return T0
        
    if isinstance(t, float) or isinstance(t, int):
        if t < 0.1 * Gyr:
            Mdot = Mdot_t(0.1*Gyr)
            T0 = T_t(1.0*Gyr)
        else:
            Mdot = Mdot_t(t)
            T0 = T_t(t)
    else:
        Mdot = Mdot_t(t)
        Mdot[t<=0.1*Gyr] = Mdot_t(0.1*Gyr)

        T0 = T_t(t)
        T0[t<1.0*Gyr] = T_t(1.0*Gyr)
    
    return T0, Mdot

def v_r(r):
    """ Beta-law for the wind speed """
    beta = 3
    v_inf = 400e5
    v = v_inf * (1 - R_sun/r)**beta
    return v

def wta(r=1*AU, t=None, plot=False, age=None):
    """
    Turns the wind parameters to the accretion parameters
    It works correctly only for r > 25 R_sun
    """
    if t is None:
        t = gett_binary()
    
    """ The boundary conditions """
    if age:
        T0, Mdot = wp(age)
        T0 = np.zeros(len(t)) + T0
        Mdotw = np.zeros(len(t)) + Mdot
    else:
        T0, Mdotw = wp(t) # we do not need T0
    # gamma = 5/3
    # alpha = 1.51
    # kg = 1e3 # g
    rho = Mdotw / (4*np.pi*r**2*v_r(r)) + np.zeros(len(t))
    # K = 1e-12
    # c_s = (gamma*K*R_sun**(3*alpha-1)*kg**(1-alpha)*rho**(alpha-1))**0.5
    
    v_orb = (G*2.4*M_sun/r)**0.5
    v = (v_r(r)**2 + v_orb**2)**0.5 + np.zeros(len(t))
    n = rho / m_p
    
    if plot:
        # plt.plot(t/Gyr, v/1e5)
        # plt.plot(t/Gyr, c_s/1e5)
        plt.plot(t/Gyr, Mdotw, lw=3)
        plt.ylabel('$\dot{M}_w,$ g/s')
        plt.xlabel('$t,$ Gyr')
        plt.xlim([-0.03, 1.005*t[-1]/Gyr])
        # plt.plot(t, T0)
        # plt.plot(t, (5/3*k_B*1.8e6/(0.6*m_p))**0.5+np.zeros(len(t)))
        plt.yscale('log')
    return t, v, n


def evolution_binary(t: np.array, P0:float, B: np.array,
              Omega: float, a: float, e: float, case: str, plot: bool):
    """
    Calculates the spin period evolution and the evolutionary stages,
    the arrays t, B, v, Mdot are of the same length
    """
    apo = a * (1+e)
    per = a * (1-e)

    _, _, n_apo = wta(r=apo, t=t, plot=False, age=None)
    _, _, n_per = wta(r=per, t=t, plot=False, age=None)
    rho_apo = m_p * n_apo + np.zeros(len(t))
    rho_per = m_p * n_per + np.zeros(len(t))
    v_apo = ((G*2.4*M_sun/a)*((1-e)/(1+e)) + v_r(apo)**2)**0.5 + np.zeros(len(t))
    v_per = ((G*2.4*M_sun/a)*((1+e)/(1-e)) + v_r(per)**2)**0.5 + np.zeros(len(t))
    Mdot_apo = getMdot(rho=rho_apo, v=v_apo, t=t)
    Mdot_per = getMdot(rho=rho_per, v=v_per, t=t)
    
    leng = len(t)
    
    P = np.zeros(leng) # an empty array for the spin period
    P[0] = P0 # the first element of the spin period array is P0

    stages = np.zeros(leng, dtype=int) # an empty array for stages
    
    NS_per = Object(B=B[0], v=v_per[0], Mdot=Mdot_per[0], case=case,
                    Omega=Omega) # for the first stage
    NS_apo = Object(B=B[0], v=v_apo[0], Mdot=Mdot_apo[0], case=case,
                    Omega=Omega) # for the evolution
    
    stages[0] = find_stage(t_cur=t[0], P_cur=P[0], NS=NS_per, previous_stage=1)
    
    for i in (range(1, leng)):
        """ Define the stage and find dP """
        dP_dt = [NS_apo.dP_dt_ejector, NS_apo.dP_dt_propeller, NS_apo.dP_dt_accretor,
                 NS_apo.dP_dt_georotator]
        P[i] = P[i-1] + (t[i]-t[i-1]) * dP_dt[stages[i-1]-1](t[i-1], P[i-1])
        NS_apo = Object(B=B[i], v=v_apo[i], Mdot=Mdot_apo[i], case=case,
                        Omega=Omega) # for the evolution # define the properties of the NS at t[i]
        NS_per = Object(B=B[i], v=v_per[i], Mdot=Mdot_per[i], case=case,
                        Omega=Omega) # for the first stage
        
        stages[i] = find_stage(t_cur=t[i], P_cur=P[i], NS=NS_per,
                           previous_stage=stages[i-1])
        
    if plot:
        P_EP = np.zeros(leng)
        P_PE = np.zeros(leng)
        P_PA = np.zeros(leng)
        P_AP = np.zeros(leng)
        for i in (range(0, leng)):
            NS_apo = Object(B=B[i], v=v_apo[i], Mdot=Mdot_apo[i], case=case,
                            Omega=Omega)
            P_EP[i] = NS_apo.P_EP(t[i])
            P_PE[i] = NS_apo.P_PE(t[i])
            P_PA[i] = NS_apo.P_PA(t[i])
            P_AP[i] = NS_apo.P_AP(t[i])
        
        
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
        ax.plot(t/Gyr, Mdot_apo)
        ax.plot(t/Gyr, Mdot_per, ls='--')
        ax.set_yscale('log')
        
        ax = ax3
        ax.set_ylabel('v, km/s')
        ax.set_xlabel('t, Gyr')
        ax.plot(t/Gyr, v_apo/1e5)
        ax.plot(t/Gyr, v_per/1e5, ls='--')
        ax.set_yscale('log')
        
    return t, P, stages

e = 0.8
a = 1 * AU

orb_p = (a/AU)**1.5 * (1/(2.4))**0.5 * year
Omega = 2*np.pi/orb_p
t = gett_binary(10_000)
B = getB(t=t, B0=1e12, field='CF')

t, P, stages = evolution_binary(t=t, P0=0.1, B=B,
              Omega=Omega, case='A', a=a, e=e, plot=True)
