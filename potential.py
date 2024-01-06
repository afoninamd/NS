#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:58:15 2023

@author: afoninamd
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic
from gala.units import UnitSystem
from tqdm import tqdm
import pandas as pd
import scipy as sp
from astropy.io import fits
from constants import year, galaxy_age, pc, rgal
from constants import fontsize
from density import DensityMap

""" Defining cgs unit system """
usys = UnitSystem(u.cm, u.second, u.radian, u.gram) # cgs
# """ Checking """
# dens = gp.NFWPotential(m=6E11, r_s=20., units=galactic).density([0, 5, 0.], 0.)
# dens = pot.density([0, 5, 0], 0.)
# dens = usys.decompose(dens).value / m_p

def Orbit(pos=[10,0,0.], vel=[0,175,0], t_end=galaxy_age, plotOrbit=False):

    """ Defining the Milky Way potential """
    mw = gp.MilkyWayPotential() # pot.units = (kpc, Myr, solMass, rad)
    # Om_bar = 42. * u.km/u.s/u.kpc
    # frame = gp.ConstantRotatingFrame(Omega=[0,0,Om_bar.value]*Om_bar.unit,
    #                              units=galactic)
    pot = mw
    
    """ Rotational curve for initial velocities """
    cvel = pot.circular_velocity(pos)[0].value
    R = (pos[0]**2 + pos[1]**2)**0.5
    vel[0] += cvel * pos[1] / R
    vel[1] -= cvel * pos[0] / R
    
    """ Min orbital period """ #!!! check, REDO
    acc = pot.acceleration(pos, t=0.0).value
    acceleration = (acc[0]**2 + acc[1]**2 + acc[2]**2)**0.5
    acc = acceleration[0] # kpc / Myr
    maxR = (pos[0]**2 + pos[1]**2 + pos[2]**2)**0.5
    v = (maxR * acc)**0.5
    orbPeriod = 2 * np.pi * maxR / v # Myr
    orbPeriod = orbPeriod * year * 1e6 # s
    
    """ Initial position and velocity """
    ics = gd.PhaseSpacePosition(pos=pos * u.kpc, vel=vel * u.km/u.s)
    
    """ Orbit integrating """
    v_max = (vel[0]**2 + vel[1]**2 + vel[2]**2)**0.5
    dt = 4 * pc / (v_max * 1e5) / year / 1e6 # [Myr]
    if dt < 10_000 / 1e6:
        dt = 10_000

    n_steps = int(1.1 + t_end / year / 1e6 / dt)
    orbit = gp.Hamiltonian(pot).integrate_orbit(ics, dt=dt, n_steps=n_steps)
                                                # 0.5*u.Myr,
                                                # n_steps=1+int(2*t_end/year/1e6))

    t = np.array(orbit.t) # Myr
    xyz = np.array(orbit.pos.xyz) # [x, y, z] kpc
    v_xyz = np.array(orbit.vel.d_xyz) # [v_x, v_y, v_z] kpc / Myr
    
    if plotOrbit:
        """ Custom plot """
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]
        
        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        
        axs[0].plot(x, y)
        r = 1.1*np.max((x**2+y**2)**0.5)
        axs[0].set_xlim(-r, r)
        axs[0].set_ylim(-r, r)
        axs[0].set_ylabel("y, kpc", fontsize=fontsize)
        axs[0].set_xlabel("x, kpc", fontsize=fontsize)
        axs[1].plot(x, z)
        r = 1.1*np.max((x**2+z**2)**0.5)
        axs[1].set_xlim(-r, r)
        axs[1].set_ylim(-r, r)
        axs[1].set_ylabel("z, kpc", fontsize=fontsize)
        axs[1].set_xlabel("x, kpc", fontsize=fontsize)
        axs[2].plot(y, z)
        r = 1.1*np.max((y**2+z**2)**0.5)
        axs[2].set_xlim(-r, r)
        axs[2].set_ylim(-r, r)
        axs[2].set_ylabel("z, kpc", fontsize=fontsize)
        axs[2].set_xlabel("y, kpc", fontsize=fontsize)
        
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
            

        """ Gala plot """
        # fig = orbit.plot() # shows xy, xz, yz planes
        # fig.show()
    
    return t, xyz, v_xyz, orbPeriod


def Trajectory(pos=[10,0,0.], vel=[0,175,0], t_end=galaxy_age, plotOrbit=False,
               realisticMap=False): # initial pos, vel

    """ Obtaining time, position and velocity arrays [v] = 1 kpc / Myr """
    t, xyz, v_xyz, orbPeriod = Orbit(pos=pos, vel=vel, t_end=t_end,
                                     plotOrbit=plotOrbit)
    t = t * 1e6 * year # [s]
    v_xyz = v_xyz * 1000 * pc / (1e6 * year) * 1e-5 # [km/s]
    
    """ Array of velocities v -> relative to ISM """
    mw = gp.MilkyWayPotential() # pot.units = (kpc, Myr, solMass, rad)
    pot = mw
    
    cvel = pot.circular_velocity(pos).value # [km/s]
    R2 = xyz[0]**2 + xyz[1]**2
    cosTeta = (R2 / (R2 + xyz[2]**2))**0.5

    v_xyz[0] = v_xyz[0] - cvel * cosTeta * xyz[1] / R2**0.5
    v_xyz[1] = v_xyz[1] + cvel * cosTeta * xyz[0] / R2**0.5

    v2 = np.zeros(len(t))
    for i in range(len(v_xyz)):
        v2 = v2 + v_xyz[i]**2
    v = v2**0.5
    
    v = v * 1e5 # [cm/s]
    
    if not realisticMap:
        """ Number density array from Misiriotis et al. model """
        n = DensityMap(xyz[0], xyz[1], xyz[2])
        # n = np.zeros(len(t))
        # for i in range(len(t)):
        #     n[i] = DensityMap(xyz[0][i], xyz[1][i], xyz[2][i])
        
    else:
        """ Number density array from realistic map """
        n = DensityMap(xyz[0], xyz[1], xyz[2], realisticMap=True)

    """ Defining 0.1 stellar disk potential """
    # x0 = [np.log(6E11), np.log(20.), np.log(2E9), np.log(100.)]
    # pot = get_potential(*x0)
    
    """ Array of number densities n """
    # dens = pot.density(xyz, 0.0) # [solMass / kpc3]
    # dens = usys.decompose(dens) # [g / cm3]
    # n = np.array(dens.value) / m_p # [cm-3]
    
    return t, v, n, orbPeriod


def TrajectoryPlot(pos=[1,0,0.], vel=[0,175,0], t_end=galaxy_age):
    """ Plotting trajectory and corresponding v(t) or n(t) """
    # t, xyz, v_xyz = Orbit(pos=pos, plot=True)
    t, v, n, orbPeriod = Trajectory(pos=pos, vel=vel, t_end=t_end, plot=True)
    plt.figure(figsize=(20, 20))
    plt.plot(t, n, marker='o')
    plt.plot(t, v)
