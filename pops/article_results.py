#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 19:40:08 2025

@author: afoninamd
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
# import pyarrow.feather as feather

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pops.track import get_coordinates_velocities, evolution_galaxy_iterations
from pops.observability import flux_to_counts_constants
from main.evolution import gett
from main.constants import galaxy_age, N, G, M_NS, m_p
from main.model import Object
from pops.trajectory import orbit

import warnings
import multiprocessing
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
n_cores = multiprocessing.cpu_count()-1

plt.rcdefaults()


def left_the_galaxy():
    """
    Performs calculations for distribution_{}.csv initial parameters,
    where {} is the number of tracks
    """
                # df = pd.DataFrame({'i': np.array([]), 'accretor_part': np.array([])})
    N = 10000
    data = pd.read_csv('/home/afoninamd/Documents/NS/project/pops/result/realistic/distr/distribution_pulsar_10000_0.csv'.format(N), sep=';')
    # data = pd.read_csv('distribution_{}.csv'.format(N), sep=';')
    Vx = data['Vx']
    Vy = data['Vy']
    Vz = data['Vz']
    x = data['x']
    y = data['y']
    z = data['z']
    vx = Vx / 1e5
    vy = Vy / 1e5
    vz = Vz / 1e5
    i_array = np.array(range(N))
    def func(i): #for i in tqdm(range(N)): #tqdm
    
        pos = [x[i], y[i], z[i]]  # kpc
        vel = [vx[i], vy[i], vz[i]]  # km/s
        
        res_trajectory = get_coordinates_velocities(pos=pos, vel=vel,
                                                    t_end=galaxy_age,
                                                    plot=0)
        t0, num, xyz, v_xyz = res_trajectory
        
        final_distance = (xyz[0](t0[-1])**2+xyz[1](t0[-1])**2+xyz[2](t0[-1])**2)**0.5
        if final_distance > 100: # kpc
            return 0
        else:
            return 1
    
    status = Parallel(n_jobs=n_cores)(delayed(func)(i) for i in tqdm(i_array))
    df = pd.DataFrame({'i': i_array, 'status': status})
    df.to_csv('left_the_galaxy.csv', sep=';')


def check_the_left_objects():
    # left_the_galaxy()
    
    # df = pd.read_csv('left_the_galaxy.csv', sep=';')
    df = pd.read_csv('left_the_galaxy.csv', sep=';')
    status = np.array(df['status'])
    print(len(status[status>0])/10000)
    
    df = pd.read_csv('/home/afoninamd/Documents/NS/project/pops/result/realistic/distr/distribution_pulsar_10000_0.csv'.format(N), sep=';')
    # df = pd.read_csv('distribution_{}.csv'.format(N), sep=';')
    V = (df['Vx']**2 + df['Vy']**2 + df['Vz']**2)**0.5
    V1 = V[status>0]
    V2 = V[status==0]
    bins = np.linspace(0, 500e5, 51)
    plt.hist(V, bins, alpha=0.2, color='k')
    plt.hist(V1, bins, alpha=0.5, color='C0')
    plt.hist(V2, bins, alpha=0.5, color='C1')

# check_the_left_objects()

def is_150_a_threshold(thres=150, N=1000):
    V = thres
    R = 20
    i_array = np.array(range(N))
    # def sample_spherical(npoints, ndim=3):
    #     vec = np.random.randn(ndim, npoints)
    #     vec /= np.linalg.norm(vec, axis=0)
    #     return vec
    # xi, yi, zi = sample_spherical(N)
    # Vx, Vy, Vz = xi*V, yi*V, zi*V
    
    df = pd.read_csv('left_the_galaxy_{}_{}.csv'.format(thres, R), sep=';')
    Vx, Vy, Vz = df['Vx'], df['Vy'], df['Vz']
    i = 41 
    
    
    def func(i): #for i in tqdm(range(N)): #tqdm
    
        pos = [R, 0, 0]  # kpc
        vel = [Vx[i], Vy[i], Vz[i]]  # km/s
        
        res_trajectory = get_coordinates_velocities(pos=pos, vel=vel,
                                                    t_end=galaxy_age,
                                                    plot=0)
        t0, num, xyz, v_xyz = res_trajectory
        
        final_distance = (xyz[0](t0[-1])**2+xyz[1](t0[-1])**2+xyz[2](t0[-1])**2)**0.5
        
        orbit(pos=pos, vel=vel, t_end=galaxy_age, plot=True)
        
        # year = np.pi*1e7
        # Gyr = year*1e9
        # plt.plot(t0/13.8/Gyr, xyz[0](t0), alpha=0.5)
        # plt.plot(t0/13.8/Gyr, xyz[1](t0), alpha=0.5)
        if final_distance > 100: # kpc
            return 0
        else:
            return 1
    
    status = func(0)
    print(status)
    return 0 

    status = Parallel(n_jobs=n_cores)(delayed(func)(i) for i in tqdm(i_array))
    df = pd.DataFrame({'i': i_array, 'Vx':Vx, 'Vy':Vy, 'Vz': Vz, 'status': status})
    df.to_csv('left_the_galaxy_{}_{}.csv'.format(thres, R), sep=';')
    
    df = pd.read_csv('left_the_galaxy_{}_{}.csv'.format(thres, R), sep=';')
    status = np.array(df['status'])
    V = (df['Vx']**2 + df['Vy']**2 + df['Vz']**2)**0.5
    V1 = V[status>0]
    V2 = V[status==0]
    print(len(V1), len(V2))
    
is_150_a_threshold()



def R_G_R_L():
    B = 10**np.linspace(10, 15, 10000)
    v = 100e5
    numdens = 0.1
    R_G = 2 * G * M_NS / v**2
    Mdot = np.pi * R_G**2 * m_p * numdens * v
    Pl = np.zeros(len(B))
    PG = np.zeros(len(B))
    for i in range(len(B)):
        NS = Object(B=B[i], v=100e5, Mdot=Mdot, case='A', Omega=None)
        Pl[i] = NS.P_EP_l(0)
        PG[i] = NS.P_EP_G(0)
    plt.plot(B, Pl)
    plt.plot(B, PG)
    
    # P = 10**np.linspace(-1, 3, 1000)
    # R_L = NS.R_l(P)
    # R_G = NS.R_G(0) + np.zeros(len(P))
    # plt.plot(P, R_L)
    # plt.plot(P, R_G)
    plt.xscale('log')
    plt.yscale('log')
    
    v = 10e5
    numdens = 0.1
    R_G = 2 * G * M_NS / v**2
    Mdot = np.pi * R_G**2 * m_p * numdens * v
    NS1 = Object(1e12, v, Mdot=Mdot, case='A', Omega=None)
    
    v = 100e5 
    # numdens = 1
    R_G = 2 * G * M_NS / v**2
    Mdot = np.pi * R_G**2 * m_p * numdens * v
    NS2 = Object(1e12, v, Mdot=Mdot, case='A', Omega=None)
    print(np.log10(NS2.P_EP_l(0)/NS1.P_EP_l(0))*9)
    print(NS2.P_EP_l(0))
    

# R_G_R_L()