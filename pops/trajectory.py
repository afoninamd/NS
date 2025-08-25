#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 18:11:27 2025

@author: afoninamd
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import gala.dynamics as gd
import gala.potential as gp
from time import time
from main.constants import year, galaxy_age, pc, rgal, mw
from pops.velocity import v_circ_1d, v_circ_2d
from scipy.signal import find_peaks, correlate


def orbit(pos=[10,0,0.], vel=[0,175,0], t_end=galaxy_age, plot=False,
          number_of_dots=10_000):
    n_steps_max = number_of_dots
    """
    Returns coordinates and velocities in absolute frame of reference
    """
    
    """ Rotational curve for initial velocities """
    pot = mw
    pos = np.array(pos)
    vel = np.array(vel)
    x0 = pos[0]
    y0 = pos[1]
    v_x, v_y = v_circ_2d(v_circ_1d(x0, y0), x0, y0)
    vel[0] = vel[0] + v_x
    vel[1] = vel[1] + v_y
    
    """ Initial position and velocity """
    ics = gd.PhaseSpacePosition(pos=pos * u.kpc, vel=vel * u.km/u.s)
    
    """ Orbit integrating """
    v_max = (vel[0]**2 + vel[1]**2 + vel[2]**2)**0.5
    dt = 4 * pc / (v_max * 1e5) # [s]
    dt = max(dt, t_end/n_steps_max) # min step is 10 kyr 10e3*year
    n_steps = int(1.1 + t_end / dt)
    # print('n_steps = {}, t_end = {} Gyr, dt = {:.2e} Gyr'.format(n_steps, t_end/year/1e9, dt/year/1e9))
    dt = dt / 1e6 / year # dt must be in Myr
    orbit = gp.Hamiltonian(pot).integrate_orbit(ics, dt=dt, n_steps=n_steps)
                                                # 0.5*u.Myr,
                                                # n_steps=1+int(2*t_end/year/1e6))

    t = np.array(orbit.t) # Myr
    xyz = np.array(orbit.pos.xyz) # [x, y, z] kpc
    v_xyz = np.array(orbit.vel.d_xyz) # [v_x, v_y, v_z] kpc / Myr
    v_xyz = v_xyz * 1000 * pc / (1e6 * year) * 1e-5 # [km/s]
    
    if plot:
        """ Custom plot """
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]
        fontsize=20
        alpha = 0.7
        labelpad = 0
        lw = 0.1
        
    
        fig11, axs = plt.subplots(1, 3, figsize=(16, 4.4))
        
        axs[0].plot(x, y, alpha=alpha, lw=lw)
        r = 1.1*np.max((x**2+y**2)**0.5)
        axs[0].set_xlim(-r, r)
        axs[0].set_ylim(-r, r)
        axs[0].set_ylabel("$y$, kpc", fontsize=fontsize, labelpad=labelpad)
        axs[0].set_xlabel("$x$, kpc", fontsize=fontsize, labelpad=labelpad)
        axs[1].plot(x, z, alpha=alpha, lw=lw)
        r = 1.1*np.max((x**2+z**2)**0.5)
        axs[1].set_xlim(-r, r)
        # axs[1].set_ylim(-r, r)
        axs[1].set_ylabel("$z$, kpc", fontsize=fontsize, labelpad=labelpad)
        axs[1].set_xlabel("$x$, kpc", fontsize=fontsize, labelpad=labelpad)
        axs[2].plot(y, z, alpha=alpha, lw=lw)
        r = 1.1*np.max((y**2+z**2)**0.5)
        axs[2].set_xlim(-r, r)
        # axs[2].set_ylim(-r, r)
        axs[2].set_ylabel("$z$, kpc", fontsize=fontsize, labelpad=labelpad)
        axs[2].set_xlabel("$y$, kpc", fontsize=fontsize, labelpad=labelpad)
        
        for i in range(len(axs)):
            axs[i].set_box_aspect(1)
            axs[i].tick_params(which='major', width=1.0, length=10,
                               labelsize=fontsize)
            axs[i].tick_params(which='minor', width=1.0, length=5,
                               labelsize=fontsize-2, labelcolor='0.25')
        
        if np.max(np.append(x, y)) > rgal:
            circ = plt.Circle((0, 0), rgal, fill=False)
            axs[0].add_patch(circ)
        if np.max(z) > 2:
            axs[1].add_patch(plt.Rectangle((-rgal, -2), 2*rgal, 4, fill=False))
            axs[2].add_patch(plt.Rectangle((-rgal, -2), 2*rgal, 4, fill=False))
            # axs[0].patches.Circle((0, 0), rgal, fill=False)
        
        for i0 in range(3):
            axs[i0].tick_params(axis='both', which='major', labelsize=16)
            axs[i0].set_xticks([-8,-4,0,4,8])
        axs[0].set_yticks([-8,-4,0,4,8])
        axs[1].set_yticks([-2,-1,0,1,2])
        axs[2].set_yticks([-2,-1,0,1,2])
        fig11.savefig('/home/afoninamd/Documents/NS/pictures/diplom/traj.pdf', bbox_inches='tight')

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 12), sharex=True)
        for i in range(3):
            ax[0,i].plot(t, xyz[i])
            ax[1,i].plot(t, v_xyz[i])
        for axes in ax.flatten():
            axes.set_xlabel('t, Myr')
    
    return t*1e6*year, xyz, v_xyz # s, kpc, km/s


def correlation(dt: float, signal: np.array):
    # the signal without shift is signal - mean
    corr = correlate(signal - np.mean(signal), signal - np.mean(signal),
                     mode='full')
    # correlation function > 0 -- the second part of the correlation array
    corr = corr[len(corr)//2:]
    peaks, _ = find_peaks(corr, height=0) # peaks are possible periods
    if len(peaks) > 1:
        period_idx = peaks[1]
        period_time = period_idx * dt
        return period_time
    else:
        return np.nan


def find_orbital_period(t, xyz, v_xyz):
    """
    Uses trajectory data to find a possible orbital period through vorrelation
    """
    possible_period = np.zeros(6) + np.nan
    
    for i in range(6):
        signal = [xyz[0], xyz[1], xyz[2], v_xyz[0], v_xyz[1], v_xyz[2]][i]
        possible_period[i] = correlation(t[1]-t[0], signal)
        if not np.isnan(possible_period[i]):
            return possible_period[i]  # [s]
    else:
        return np.nan


def test_orbit():
    """
    time - n_steps is linear
    1_000_000 - 0.3-0.4 s
    1_000_000 - 0.6 s wih correlation
    """
    N = [100, 1000, 10_000, 100_000, 1_000_000]
    time_array = np.zeros(len(N))
    for i in range(len(N)):
        n = N[i]
        time0 = time()
        t, xyz, v_xyz = orbit(pos=[10,0,0.], vel=[0,175,0], t_end=galaxy_age,
                              plot=False, number_of_dots=n)
        (find_orbital_period(t, xyz, v_xyz))
        time_array[i] = time() - time0
    plt.plot(N, time_array)
    
# test_orbit()