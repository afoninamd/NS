#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 15:14:35 2025

@author: afoninamd
"""

from mpi4py import MPI
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.feather as feather

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pops.track import get_coordinates_velocities, evolution_galaxy_iterations
from pops.observability import flux_to_counts_constants, one_observability
from main.evolution import gett, star_formation_history
from main.constants import galaxy_age, N, output_dir, R0, arr_size

comm = MPI.COMM_WORLD
crank = comm.Get_rank()
csize = comm.Get_size()

def illustration_roman():
    n = 1000000
    R0 = 8
    Rbuldge = 2
    x, y, z = np.random.normal(0, 20, n), np.random.normal(0, 20, n), np.random.normal(0, 5, n)
    r0 = np.array([[0]*n, [R0]*n, [0]*n])
    r1 = np.array([x, y, z])
    r2 = r0 - r1
    
    def module(array):
        return (array[0]**2 + array[1]**2 + array[2]**2)**0.5
    
    mr0 = R0
    mr1 = module(r1)
    mr2 = module(r2)
    cosa = r2[1] / (mr2)
    cosa0 = R0 / (R0**2 + Rbuldge**2)**0.5
    print(cosa)
    tana = (1/cosa**2 - 1)**0.5
    print((tana))
    cond = np.logical_and(tana<0.25, mr2 < mr0 / cosa0)
    cond = np.logical_and(cond, mr1 < R0)
    cond = np.logical_or(cond, mr1<2)
    x = x[cond]
    y = y[cond]
    z = z[cond]
    fig, ax = plt.subplots(ncols=3)
    ax[0].scatter(x, y)
    ax[0].set_ylabel('y')
    ax[0].set_xlabel('x')
    ax[1].scatter(x, z)
    ax[1].set_ylabel('z')
    ax[1].set_xlabel('x')
    ax[2].scatter(y, z)
    ax[2].set_ylabel('z')
    ax[2].set_xlabel('y')


"""  POPSYNTHESIS STARTS HERE  """

Rbins = np.linspace(0, 20, arr_size+1) # from the GC
rbins = np.linspace(0, 20, arr_size+1) # from the Sun better for 0 to 20
zbins = np.linspace(0, 5, arr_size+1)
fbins = 10**np.linspace(-15, -5, arr_size+1)
cbins = 10**np.linspace(-4, 6, arr_size+1)
vbins = np.linspace(0, 500, arr_size+1) * 1e5
Tbins = 10**np.linspace(5, 9, arr_size+1) # better from 5 to 8

def calculations(star_type):
   
    """
    Performs calculations for distribution_{}.csv initial parameters,
    where {} is the number of tracks
    """
                # df = pd.DataFrame({'i': np.array([]), 'accretor_part': np.array([])})
    if star_type == 'pulsar':
        N1 = 9*N//10
    elif star_type == 'magnetar':
        N1 = N//10
    
    vals_per_core = N1 // csize
    remainder = N1 % csize
    data = pd.read_csv(output_dir + 'distribution_{}_{}.csv'.format(star_type, N1), sep=';')

    start_idx = crank * vals_per_core + min(crank, remainder)
    end_idx = start_idx + vals_per_core
    if crank < remainder:
        end_idx += 1

    if crank == 0:
        os.makedirs(output_dir + 'feather', exist_ok=True)
        os.makedirs(output_dir + 'temp', exist_ok=True)
    comm.Barrier()
        
    P = data['P']
    B = data['B']
    Vx = data['Vx']
    Vy = data['Vy']
    Vz = data['Vz']
    x = data['x']
    y = data['y']
    z = data['z']
    # Vxpec = data['Vxpec']
    # Vypec = data['Vypec']
    # Vzpec = data['Vzpec']
    vx = Vx / 1e5
    vy = Vy / 1e5
    vz = Vz / 1e5
    # vx = (Vx+Vxpec) / 1e5
    # vy = (Vy+Vypec) / 1e5
    # vz = (Vz+Vzpec) / 1e5
    """ for the Roman telescope """
    def module(array):
        return (array[0]**2 + array[1]**2 + array[2]**2)**0.5
    Rbuldge = 2
    mr0 = R0
    cosa0 = R0 / (R0**2 + Rbuldge**2)**0.5
    
    for galaxy_type in ['simple', 'two_phase']:
        for field in ['CF', 'ED']:
            for case in ['A', 'B', 'C', 'D']:
                for add_string in ['', '_roman']:
                    # df0 = pd.DataFrame({'R': Rcounts, 'r': rcounts, 'z': zcounts,
                    #                     'v': vcounts, 'T': Tcounts,
                    #                     'f0': f0counts, 'f1': f1counts,
                    #                     'c0': c0counts, 'c1': c1counts})
                    df0 = pd.DataFrame({'R': np.zeros(arr_size),
                                        'R-2': np.zeros(arr_size),
                                        'R-1': np.zeros(arr_size),
                                        'r': np.zeros(arr_size),
                                        'r-2': np.zeros(arr_size),
                                        'r-1': np.zeros(arr_size),
                                        'z': np.zeros(arr_size),
                                        'z-2': np.zeros(arr_size),
                                        'z-1': np.zeros(arr_size),
                                        'v': np.zeros(arr_size),
                                        'v-2': np.zeros(arr_size),
                                        'v-1': np.zeros(arr_size),
                                        'T': np.zeros(arr_size),
                                        'T-2': np.zeros(arr_size),
                                        'T-1': np.zeros(arr_size),
                                        'f0': np.zeros(arr_size),
                                        'f1': np.zeros(arr_size),
                                        'c0': np.zeros(arr_size),
                                        'c1': np.zeros(arr_size)
                                    })
                    float_cols = df0.select_dtypes(include=['float64']).columns
                    df0[float_cols] = df0[float_cols].astype('float32')
                    
                    name = 'feather/{}_{}_{}_{}_{}_erosita{}'.format(crank, galaxy_type, field,
                                                    case, star_type, add_string)
                    feather.write_feather(df0, output_dir + name + '.feather')
    
    for i in (range(start_idx, end_idx)): #tqdm
    
        # print(f"Process {crank} handling index {i} (start = {start_idx}, end = {end_idx})")
        
        pos = [x[i], y[i], z[i]]  # kpc
        vel = [vx[i], vy[i], vz[i]]  # km/s
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
                    res = evolution_galaxy_iterations(P0, t, xyz, v_xyz, B0, field, case,
                                                      plot=0, iterations=3,
                                                      galaxy_type=galaxy_type)
                    t1, P1, B1, stages1, v1, Mdot1, ph1, x1, y1, z1, vx1, vy1, vz1 = res
                    
                        # for sfr in [True, False]:
                        #     name = output_dir + file_name + '_sfr{}'.format(sfr)
                    
                    weight = t1[1:] - t1[:-1]
                    weight = weight * star_formation_history(t1[1:])
                    weight = weight / np.sum(weight)
                    
                    wE = weight[stages1[1:] == 1]
                    wP = weight[stages1[1:] == 2]
                    wA = weight[stages1[1:] == 3]
                    wG = weight[stages1[1:] == 4] 
                    
                    # new_row = pd.DataFrame({'i': [i], 'accretor_part': [np.sum(weight)]})
                    # df = pd.concat([df, new_row], ignore_index=True)
                    file_name = output_dir + 'temp/{}_{}_{}_{}_{}'.format(crank, galaxy_type, field, case, star_type)
                    with open(file_name + '.txt', 'a') as file:
                        file.write('{}\t{}\t{}\t{}\t{}\n'.format(i, np.float32(np.sum(wE)), np.float32(np.sum(wP)), np.float32(np.sum(wA)), np.float32(np.sum(wG))))
                        
                    """ Does Roman see it? """
                    r1 = np.array([x1[1:], y1[1:], z1[1:]])
                    n = len(x1[1:])
                    r0 = np.array([[0]*n, [R0]*n, [0]*n])
                    r2 = r0 - r1
                    
                    mr1 = module(r1)
                    mr2 = module(r2)
                    cosa = r2[1] / (mr2)
                    tana = (1/cosa**2 - 1)**0.5
                    
                    cond = np.logical_and(tana<0.25, mr2 < mr0 / cosa0)
                    cond = np.logical_and(cond, mr1 < R0)
                    cond = np.logical_or(cond, mr1<2)
                    
                    stages2 = stages1[1:]
                    w_roman = weight[cond]
                    if np.sum(w_roman) > 0:
                        wE = weight[np.logical_and(cond, stages2 == 1)]
                        wP = weight[np.logical_and(cond, stages2 == 2)]
                        wA = weight[np.logical_and(cond, stages2 == 3)]
                        wG = weight[np.logical_and(cond, stages2 == 4)]
                        
                        with open(file_name + '_roman.txt', 'a') as file:
                            file.write('{}\t{}\t{}\t{}\t{}\n'.format(i, np.float32(np.sum(wE)), np.float32(np.sum(wP)), np.float32(np.sum(wA)), np.float32(np.sum(wG))))
                    
                    """ Does Erosita see it? """
                    if len(stages1[stages1==3]) > 0:
                        T, f0, f1, c0, c1 = one_observability(t1, stages1, x1, y1, z1, B1, Mdot1,
                                                              nu, deltanu, cross, Seff)
                        if len(c0[c0>1e-4]) > 0:
                            
                            weight1 = np.zeros(len(weight))
                            weight2 = np.zeros(len(weight))
                            weight1[::] = weight[::]
                            weight2[::] = weight[::]
                            weight1[c1[1:]<1e-1] = 0
                            weight2[c1[1:]<1e-2] = 0
                            
                            R = (x1[1:]**2+y1[1:]**2+z1[1:]**2)**0.5
                            Rcounts, _ = np.histogram(R, bins=Rbins, weights=weight)
                            R2counts, _ = np.histogram(R, bins=Rbins, weights=weight2)
                            R1counts, _ = np.histogram(R, bins=Rbins, weights=weight1)
                            
                            r = (x1[1:]**2+(y1[1:]-R0)**2+z1[1:]**2)**0.5
                            rcounts, _ = np.histogram(r, bins=rbins, weights=weight)
                            r2counts, _ = np.histogram(r, bins=rbins, weights=weight2)
                            r1counts, _ = np.histogram(r, bins=rbins, weights=weight1)
                            
                            ztemp = z1[1:]
                            zcounts, _ = np.histogram(ztemp, bins=zbins, weights=weight)
                            z2counts, _ = np.histogram(ztemp, bins=zbins, weights=weight2)
                            z1counts, _ = np.histogram(ztemp, bins=zbins, weights=weight1)
                            
                            vtemp = v1[1:]
                            vcounts, _ = np.histogram(vtemp, bins=vbins, weights=weight)
                            v2counts, _ = np.histogram(vtemp, bins=vbins, weights=weight2)
                            v1counts, _ = np.histogram(vtemp, bins=vbins, weights=weight1)
                            
                            Ttemp = T[1:]
                            Tcounts, _ = np.histogram(Ttemp, bins=Tbins, weights=weight)
                            T2counts, _ = np.histogram(Ttemp, bins=Tbins, weights=weight2)
                            T1counts, _ = np.histogram(Ttemp, bins=Tbins, weights=weight1)
                            
                            f0counts, _ = np.histogram(f0[1:], bins=fbins, weights=weight)
                            f1counts, _ = np.histogram(f1[1:], bins=fbins, weights=weight)
                            c0counts, _ = np.histogram(c0[1:], bins=cbins, weights=weight)
                            c1counts, _ = np.histogram(c1[1:], bins=cbins, weights=weight)
                            
                            df = pd.DataFrame({'R': Rcounts, 'R-2': R2counts, 'R-1': R1counts,
                                               'r': rcounts, 'r-2': r2counts, 'r-1': r1counts,
                                               'z': zcounts, 'z-2': z2counts, 'z-1': z1counts,
                                               'v': vcounts, 'v-2': v2counts, 'v-1': v1counts,
                                               'T': Tcounts, 'T-2': T2counts, 'T-1': T1counts,
                                               'f0': f0counts, 'f1': f1counts,
                                               'c0': c0counts, 'c1': c1counts})
                            float_cols = df.select_dtypes(include=['float64']).columns
                            df[float_cols] = df[float_cols].astype('float32')
                            
                            name = 'feather/{}_{}_{}_{}_{}_erosita'.format(crank, galaxy_type, field,
                                                            case, star_type)
                            df0 = pd.read_feather(output_dir + name + '.feather')
                            feather.write_feather(df+df0, output_dir + name + '.feather')
                            
                            """ For Erosita + Roman """
                            roman = np.zeros(len(x1))
                            roman[1:][cond] = 1
                            weight = weight * roman[1:]
                            weight1 = weight1 * roman[1:]
                            weight2 = weight2 * roman[1:]
                            
                            Rcounts, _ = np.histogram(R, bins=Rbins, weights=weight)
                            R2counts, _ = np.histogram(R, bins=Rbins, weights=weight2)
                            R1counts, _ = np.histogram(R, bins=Rbins, weights=weight1)
                            
                            rcounts, _ = np.histogram(r, bins=rbins, weights=weight)
                            r2counts, _ = np.histogram(r, bins=rbins, weights=weight2)
                            r1counts, _ = np.histogram(r, bins=rbins, weights=weight1)
                            
                            zcounts, _ = np.histogram(ztemp, bins=zbins, weights=weight)
                            z2counts, _ = np.histogram(ztemp, bins=zbins, weights=weight2)
                            z1counts, _ = np.histogram(ztemp, bins=zbins, weights=weight1)
                            
                            vcounts, _ = np.histogram(vtemp, bins=vbins, weights=weight)
                            v2counts, _ = np.histogram(vtemp, bins=vbins, weights=weight2)
                            v1counts, _ = np.histogram(vtemp, bins=vbins, weights=weight1)
                            
                            Tcounts, _ = np.histogram(Ttemp, bins=Tbins, weights=weight)
                            T2counts, _ = np.histogram(Ttemp, bins=Tbins, weights=weight2)
                            T1counts, _ = np.histogram(Ttemp, bins=Tbins, weights=weight1)
                            
                            f0counts, _ = np.histogram(f0[1:], bins=fbins, weights=weight)
                            f1counts, _ = np.histogram(f1[1:], bins=fbins, weights=weight)
                            c0counts, _ = np.histogram(c0[1:], bins=cbins, weights=weight)
                            c1counts, _ = np.histogram(c1[1:], bins=cbins, weights=weight)
                            
                            df = pd.DataFrame({'R': Rcounts, 'R-2': R2counts, 'R-1': R1counts,
                                               'r': rcounts, 'r-2': r2counts, 'r-1': r1counts,
                                               'z': zcounts, 'z-2': z2counts, 'z-1': z1counts,
                                               'v': vcounts, 'v-2': v2counts, 'v-1': v1counts,
                                               'T': Tcounts, 'T-2': T2counts, 'T-1': T1counts,
                                               'f0': f0counts, 'f1': f1counts,
                                               'c0': c0counts, 'c1': c1counts})
                            float_cols = df.select_dtypes(include=['float64']).columns
                            df[float_cols] = df[float_cols].astype('float32')
                            
                            name = 'feather/{}_{}_{}_{}_{}_erosita_roman'.format(crank, galaxy_type, field,
                                                            case, star_type)
                            df0 = pd.read_feather(output_dir + name + '.feather')
                            feather.write_feather(df+df0, output_dir + name + '.feather')
                            # df = pd.DataFrame({'B': B1[c1>1e-4], #'t': t1[c1>1e-4], 'P': P1[c1>1e-4], 'stages': stages1[c1>1e-4],
                            #                 'v': v1[c1>1e-4], #'Mdot': Mdot1[c1>1e-4], #'phase': ph1[c1>1e-4],
                            #                 'x': x1[c1>1e-4], 'y': y1[c1>1e-4], 'z': z1[c1>1e-4],
                            #                 'T': T[c1>1e-4], 'f0': f0[c1>1e-4], 'c0': c0[c1>1e-4],
                            #                 'f1': f1[c1>1e-4], 'c1': c1[c1>1e-4], 
                            #                 'weight': weight[c1[1:]>1e-4], 'roman': weight[c1[1:]>1e-4]*roman[c1>1e-4]})
                            # reducing the size
                            # df['stages'] = df['stages'].astype('int8')
                            # df['phase'] = df['phase'].astype('int8')
                            

calculations('magnetar')
calculations('pulsar')

# df = pd.read_feather('result/realistic/simple_ED_A_pulsar_erosita' + '.feather')
# plt.plot(Tbins[1:], df['T'])


# a = np.linspace(0, 4, 5)
# b = np.linspace(1,5,5)
# df = pd.DataFrame({'a':a, 'b':b})
# # df1 = pd.DataFrame({a, b})
# # df1 = df * np.array([1,1,1,1,10])
# df = df.multiply(np.array([1,1,1,1,10]), axis=0)
# print(df)

# for galaxy_type in ['simple', 'two_phase']:
#     for field in ['CF', 'ED']:
#         for case in ['A', 'B', 'C', 'D']:
#             path = output_dir + '{}/{}/{}/'.format(galaxy_type, field, case)
#             directory = os.path.dirname(path)
#             os.makedirs(directory, exist_ok=True)