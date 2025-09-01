#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 10:30:52 2025

@author: afoninamd
"""

from mpi4py import MPI
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import pyarrow.feather as feather

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pops.track import get_coordinates_velocities, evolution_galaxy_iterations
from pops.observability import flux_to_counts_constants
from main.evolution import gett
from main.constants import galaxy_age, Gyr, N, output_dir

comm = MPI.COMM_WORLD
crank = comm.Get_rank()
csize = comm.Get_size()

vals_per_core = N // csize
remainder = N % csize

start_idx = crank * vals_per_core + min(crank, remainder)
end_idx = start_idx + vals_per_core
if crank < remainder:
    end_idx += 1

if crank == 0:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


"""  POPSYNTHESIS STARTS HERE  """

def calculations():
    """
    Performs calculations for distribution_{}.csv initial parameters,
    where {} is the number of tracks
    """

    data = pd.read_csv('distribution_{}.csv'.format(N), sep=';')
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

    for i in (range(start_idx, end_idx)): #tqdm
        # print(f"Process {crank} handling index {i} (start = {start_idx}, end = {end_idx})")
        # print(i)
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
            continue
        t = gett(t_end=galaxy_age, n=num)
        
        nu, deltanu, cross, Seff = flux_to_counts_constants()  # for observability
    
        for galaxy_type in ['simple', 'two_phase']:
            for field in ['CF', 'ED']:
                for case in ['A', 'B', 'C', 'D']:
                    file_name = output_dir + '{}_{}_{}_{}.txt'.format(crank, galaxy_type, field, case)
                    with open(file_name, 'a') as file:
                        res = evolution_galaxy_iterations(P0, t, xyz, v_xyz, B0, field, case,
                                                          plot=0, iterations=3 , # 3
                                                          galaxy_type=galaxy_type)
                        t1, P1, B1, stages1, v1, Mdot1, ph1, x1, y1, z1, vx1, vy1, vz1 = res
                        # if len(stages1[stages1==3]) > 0:
                            # for sfr in [True, False]:
                            #     name = output_dir + file_name + '_sfr{}'.format(sfr)
                        
                        """ count weights """
                        weight = t1[1:] - t1[:-1]
                        # if sfr:
                        #     weight = weight * star_formation_history(t1[1:])
                        weight = weight / np.sum(weight)
                        weight[stages1[1:] != 3] = 0
                        
                        file.write('{}\t{}\n'.format(i, np.sum(weight)))
                            
                        # df = pd.DataFrame({'t': t1, 'stages': stages1})
                        #                 # 'P': P1, 'B': B1, 'v': v1, 'Mdot': Mdot1, 'phase': ph1,
                        #                 # 'x': x1, 'y': y1, 'z': z1})
                        # # reducing the size
                        # df['stages'] = df['stages'].astype('int8')
                        # # df['phase'] = df['phase'].astype('int8')
                        # float_cols = df.select_dtypes(include=['float64']).columns
                        # df[float_cols] = df[float_cols].astype('float32')
                        
                        # name = '{}/{}/{}/{}'.format(galaxy_type, field, case, i)
                        # path = output_dir + name + '.feather'
                        
                        # # pd.DataFrame.to_csv(df, path, sep=';', mode='a')
                        # feather.write_feather(df, path)
                        
    # with open(os.path.join(output_dir, f"result_{crank}.txt"), "w") as f:
    #     for ii in range(local_vals_count):
    #         f.write(f"{start_idx + ii}\t{local_data[ii]}\n")


def check_star(i):
    for galaxy_type in ['simple', 'two_phase']:
        for field in ['CF', 'ED']:
            for case in ['A']: #, 'B', 'C', 'D']:
                path = output_dir + '{}/{}/{}/{}.feather'.format(galaxy_type, field, case, i)
                df = pd.read_feather(path)
                # arr = df['P']df['x']**2+
                arr = (df['x']**2+df['y']**2)**0.5
                arr = df['P']
                plt.plot(df['t']/Gyr, arr, alpha=0.5, marker='o', label='{}_{}_{}'.format(galaxy_type, field, case))
                plt.legend()
                plt.yscale('log')
# check_star(10)

calculations()

# for galaxy_type in ['simple', 'two_phase']:
#     for field in ['CF', 'ED']:
#         for case in ['A', 'B', 'C', 'D']:
#             path = output_dir + '{}/{}/{}/'.format(galaxy_type, field, case)
#             directory = os.path.dirname(path)
#             os.makedirs(directory, exist_ok=True)