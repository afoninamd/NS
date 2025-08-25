#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 13:01:18 2025

@author: afoninamd
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import scipy as sp
from main.evolution import evolution, getB, gett, getMdot, increase_time_resolution
from main.constants import galaxy_age, Gyr, c_s_cold, c_s_hot, m_p
from time import time
import matplotlib.pyplot as plt
from pops.trajectory import orbit, find_orbital_period
from pops.velocity import v_relative_to_ISM
from pops.density import density_cold, density_hot, galaxy_phase


def get_coordinates_velocities(pos: np.array, vel: np.array, t_end: float,
                               plot: bool):
    """
    Returns a huge time array, the recommended number of steps 
    and coordinates and velocities corresponding to this time array
    """
    # s, kpc, km/s
    t, xyz, v_xyz = orbit(pos, vel, t_end, plot, number_of_dots=100_000)
    orbital_period = find_orbital_period(t, xyz, v_xyz)
    
    if np.isnan(orbital_period):
        num = 1361
    else:
        num = max(1361, int(22*t_end/orbital_period)) # the number of steps
    
    x = sp.interpolate.interp1d(t, xyz[0])
    y = sp.interpolate.interp1d(t, xyz[1])
    z = sp.interpolate.interp1d(t, xyz[2])
    
    vx = sp.interpolate.interp1d(t, v_xyz[0])
    vy = sp.interpolate.interp1d(t, v_xyz[1])
    vz = sp.interpolate.interp1d(t, v_xyz[2])
    
    return t, num, [x, y, z], [vx, vy, vz]


def get_accretion_parameters(xyz: np.array, v_xyz: np.array, galaxy_type: str):
    
    v_x = v_xyz[0]
    v_y = v_xyz[1]
    v_z = v_xyz[2]
    
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    
    v_x, v_y = v_relative_to_ISM(v_x, v_y, x, y, z)  # km/s
    
    v = 1e5 * (v_x**2 + v_y**2 + v_z**2)**0.5
    
    leng = len(x)
    
    if galaxy_type == 'simple':
        
        n = density_cold(x, y, z) + density_hot(x, y, z)
        v = (v**2 + c_s_cold**2)**0.5
        filling_factor = np.zeros(leng) + 1
        phase = np.zeros(leng) + 1
        
    elif galaxy_type == 'two_phase':
        
        filling_factor = galaxy_phase(x, y, z)
        
        w = np.random.uniform(0, 1, leng)
        phase = np.zeros(leng, dtype=int)
        phase[w < filling_factor] = 1
        
        n = density_cold(x, y, z)
        n[phase==0] = density_hot(x, y, z)[phase==0]
        
        v[phase==0] = (v[phase==0]**2 + c_s_hot**2)**0.5
        v[phase==1] = (v[phase==1]**2 + c_s_cold**2)**0.5
        
    else:
        print("There are only 'simple' and 'two_phase' galaxy types")
    
    return n, v, filling_factor, phase


def evolution_galaxy_iterations(P0: float, t: np.array, xyz: sp.interpolate,
                                v_xyz: sp.interpolate, B0: float, field: str,
                                case: str, plot: bool,
                                iterations: int, galaxy_type: str):
    """
    For the evolution in the Galaxy
    Uses evolution function several times with different time arrays
    until the spin period evolution has no jumps according to function
    increase_time_resolution
    Returns t, P, stages as the evolution function itself
    Use only if iterations >=2, otherwise use evolution function
    """
    
    x, y, z = xyz[0], xyz[1], xyz[2] # scipy.interpolates
    vx, vy, vz = v_xyz[0], v_xyz[1], v_xyz[2] # scipy.interpolates
    
    for k in range(iterations):
        
        x1, y1, z1 = x(t), y(t), z(t)
        vx1, vy1, vz1 = vx(t), vy(t), vz(t)
        
        n, v, f, ph = get_accretion_parameters(xyz=[x1, y1, z1],
                                               v_xyz=[vx1, vy1, vz1],
                                               galaxy_type=galaxy_type)
        Mdot = getMdot(m_p*n, v, t)
        
        B = getB(t=t, B0=B0, field=field)
        t, P, stages = evolution(t=t, P0=P0, B=B, Omega=None, case=case,
                                 Mdot=Mdot, v=v, plot=plot)
        
        new_t = increase_time_resolution(t, P, stages)
        # t = new_t
        if len(new_t) == len(t):
            break
        elif k < iterations-1:
            t = new_t
    # print(len(t), len(P), len(B), len(stages), len(f), len(x1))
    return t, P, B, stages, v, Mdot, ph, x1, y1, z1, vx1, vy1, vz1


def test_evolution_galaxy_iterations():
    """
    The plot "time versus number of steps" tends to be linear:
        100_000 steps - 7 s for CF - 2 iterations
    """
    def one_try(n):
        """ Return the time needed for one calculation of the evolution """
    
        t = gett(galaxy_age, n=n)
        v = sp.interpolate.interp1d(t, t*0 + 100e5*(1.001+np.sin(t/Gyr)))
        Mdot = sp.interpolate.interp1d(t, t*0 + 1e15)
        
        time0 = time()
        t, P, _, _, _ = evolution_galaxy_iterations(1e-1, t, Mdot, v, 1e12,
                                                   'CF', None, 'B', 1,
                                                   iterations=2)
        
        return time()-time0 # the time needed for n steps
    
    N = [1000]
    # N = [100, 300, 1000, 3000, 10_000, 30_000, 100_000]#, 300_000, 1_000_000]
    time_array = np.zeros(len(N))
    for i in range(len(N)):
        time_array[i] = one_try(N[i])
    
    plt.plot(N, time_array)
