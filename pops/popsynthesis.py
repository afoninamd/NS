#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 19:06:33 2025

@author: afoninamd
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pops.track import get_coordinates_velocities, evolution_galaxy_iterations
from pops.observability import column_density, flux_to_counts, flux_to_counts_constants, one_observability
from main.evolution import gett
from main.constants import galaxy_age, Gyr, year, G, M_NS, R_NS, kpc, R0
import matplotlib.pyplot as plt
from time import time
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import multiprocessing
import pyarrow.feather as feather
warnings.filterwarnings("ignore")
n_cores = multiprocessing.cpu_count()-1

path_dir = '/home/afoninamd/Documents/project/pops'
path_result = path_dir + '/result/'


def popsynthesis(star_type, start, N):

    data = pd.read_csv(path_dir+'/distribution_{}.csv'.format(star_type),
                       sep=';')
    P = data['P']
    B = data['B']
    Vx = data['Vx']
    Vy = data['Vy']
    Vz = data['Vz']
    x = data['x']
    y = data['y']
    z = data['z']
    Vxpec = data['Vxpec']
    Vypec = data['Vypec']
    Vzpec = data['Vzpec']
    vx = (Vx+Vxpec) / 1e5
    vy = (Vy+Vypec) / 1e5
    vz = (Vz+Vzpec) / 1e5
    
    def one_star(i: int):
        pos = [x[i], y[i], z[i]] # kpc
        vel = [vx[i], vy[i], vz[i]]# km/s
        P0 = P[i]
        B0 = B[i]
        
        res_trajectory = get_coordinates_velocities(pos=pos, vel=vel,
                                                    t_end=galaxy_age,
                                                    plot=0)
        t0, num, xyz, v_xyz = res_trajectory
        
        final_distance = (xyz[0](t0[-1])**2+xyz[1](t0[-1])**2+xyz[2](t0[-1])**2)**0.5
        if final_distance > 100: # kpc
            return 0
        t = gett(t_end=galaxy_age, n=num)
        
        nu, deltanu, cross, Seff = flux_to_counts_constants()  # for observability

        for galaxy_type in ['simple', 'two_phase']:
            for field in ['CF', 'ED']:
                for case in ['A', 'B', 'C']: #, 'D']:
                    res = evolution_galaxy_iterations(P0, t, xyz, v_xyz, B0, field, case,
                                                      plot=0, iterations=3 , # 3
                                                      galaxy_type=galaxy_type)
                    t1, P1, B1, stages1, v1, Mdot1, ph1, x1, y1, z1, vx1, vy1, vz1 = res
                    if len(stages1[stages1>1]) > 0:
                        if len(stages1==3) == 0:
                            df = pd.DataFrame({'t': t1, 'P': P1, 'B': B1, 'stages': stages1,
                                           'v': v1, 'Mdot': Mdot1, 'phase': ph1,
                                           'x': x1, 'y': y1, 'z': z1})
                            # leng = len(t1)
                            # T = np.zeros(leng)
                            # f0 = np.zeros(leng)
                            # f1 = np.zeros(leng)
                            # c0 = np.zeros(leng)
                            # c1 = np.zeros(leng)
                        else:
                            T, f0, f1, c0, c1 = one_observability(t1, stages1, x1, y1, z1, B1, Mdot1,
                                                                  nu, deltanu, cross, Seff)
                         
                            df = pd.DataFrame({'t': t1, 'P': P1, 'B': B1, 'stages': stages1,
                                           'v': v1, 'Mdot': Mdot1, 'phase': ph1,
                                           'x': x1, 'y': y1, 'z': z1,
                                           'T': T, 'f0': f0, 'c0': c0,
                                           'f1': f1, 'c1': c1})
                        
                        # reducing the size
                        df['stages'] = df['stages'].astype('int8')
                        df['phase'] = df['phase'].astype('int8')
                        float_cols = df.select_dtypes(include=['float64']).columns
                        df[float_cols] = df[float_cols].astype('float32')
                        
                        name = '{}/{}/{}/{}/{}'.format(galaxy_type, field,
                                                        star_type, case, i)
                        path = path_result + name + '.feather'
                        
                        # pd.DataFrame.to_csv(df, path, sep=';', mode='a')
                        feather.write_feather(df, path)
                            # pd.DataFrame.to_hdf(df, path, mode='a', key=name)
        return res # the result of the evolution
    
    # for i in range(7):
    #     one_star(i)
    # one_star(i=1)
    Parallel(n_jobs=n_cores)(delayed(one_star)(i)
                                    for i in tqdm(range(start, start + N)))


def calculations(star_type, start=0, N=10_000):
    """
    Performs calculations for distribution_{}.csv initial parameters,
    where {} can be any star_type in 'pulsar', 'magnetar'
    """
    for galaxy_type in ['simple', 'two_phase']:
        for field in ['CF', 'ED']:
            for case in ['A', 'B', 'C']:
                path = path_result + '{}/{}/{}/{}/'.format(galaxy_type, field,
                                                                  star_type, case)
                directory = os.path.dirname(path)
                os.makedirs(directory, exist_ok=True)
    
    popsynthesis(star_type, start, N)

# for i in range(10):
#     print(i)
#     calculations('magnetar', start=i*10000, N=10_000)
#     calculations('pulsar', start=i*10000, N=10_000)
# for i in range(10, 90):
#     print(i)
#     calculations('pulsar', start=i*10000, N=10_000)


def check_star(i):
    star_type = 'pulsar'
    for galaxy_type in ['simple', 'two_phase']:
        for field in ['CF', 'ED']:
            for case in ['A']: #, 'B', 'C', 'D']:
                path = path_result + '{}/{}/{}/{}/{}.feather'.format(galaxy_type, field,
                                                                  star_type, case, i)
                df = pd.read_feather(path)
                # arr = df['P']df['x']**2+
                arr = (df['x']**2+df['y']**2)**0.5
                arr = df['P']
                plt.plot(df['t']/Gyr, arr, alpha=0.5, marker='o', label='{}_{}_{}'.format(galaxy_type, field, case))
                plt.legend()
                plt.yscale('log')
# check_star(10)