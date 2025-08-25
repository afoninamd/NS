#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 16:53:29 2025

@author: afoninamd
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main.constants import c, G, I, R_NS, M_NS, pc, kpc, galaxy_age
from pops.track import get_coordinates_velocities, evolution_galaxy_iterations
from main.evolution import gett
from pops.distribution import sample_spherical
import pyarrow.feather as feather
import corner
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import multiprocessing
import pyarrow.feather as feather
warnings.filterwarnings("ignore")
n_cores = multiprocessing.cpu_count()-1

plt.rcParams['axes.labelsize'] = 16  # or a specific number like 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

z_bins = np.linspace(-250, 250, 6, endpoint=True) / 1000
z = np.array((z_bins[1:] + z_bins[:-1]) / 2) # kpc

R_bins = np.linspace(0, 20, 11, endpoint=True)
R = np.array((R_bins[1:] + R_bins[:-1]) / 2) # kpc

V_bins = np.linspace(0, 300, 11, endpoint=True)
V = np.array((V_bins[1:] + V_bins[:-1]) / 2) # km/s

B_bins = np.linspace(10.5, 15.5, 6, endpoint=True)
B = np.array((B_bins[1:] + B_bins[:-1]) / 2) # logscaled

P_bins = np.linspace(-3.5, 1.5, 6, endpoint=True)
P = np.array((P_bins[1:] + P_bins[:-1]) / 2) # logscaled

# data = pd.read_csv('distribution_pulsar.csv', sep=';')
# module = (data['Vx']**2 + data['Vy']**2 + data['Vz']**2)**0.5
# ux = data['Vx'] / module
# uy = data['Vy'] / module
# uz = data['Vz'] / module

field = 'CF'
case = 'A'
galaxy_type = 'simple'

# ux, uy, uz = sample_spherical(2)[:, 0]
# print(ux, uy, uz)
# print(ux**2+uz**2+uy**2)

def func(zi):
    # counter = zi*len(R)*len(V)*len(P)*len(B)
# for zi in tqdm(range(len(z))):
    for Ri in tqdm(range(len(R))):
        for Vi in range(len(V)):
            for Pi in range(len(P)):
                for Bi in (range(len(B))):
                    for di in range(10):
                        # print(di) #(zi+1)*(Ri+1)*(Vi+1)*(Pi+1)*(Bi+1)
                        pos = [R[Ri], 0, z[zi]] # kpc
                        ux, uy, uz = sample_spherical(2)[:, 0]
                        vel = [V[Vi]*ux, V[Vi]*uy, V[Vi]*uz]# km/s
                
                        res_trajectory = get_coordinates_velocities(pos=pos, vel=vel,
                                                                    t_end=galaxy_age,
                                                                    plot=0)
                        t0, num, xyz, v_xyz = res_trajectory
                        
                        final_distance = (xyz[0](t0[-1])**2+xyz[1](t0[-1])**2+xyz[2](t0[-1])**2)**0.5
                        if final_distance > 100: # kpc
                            # counter += 100
                            continue
                        t = gett(t_end=galaxy_age, n=num)
                        
                        P0 = 10**P[Pi]
                        B0 = 10**B[Bi]
                        res = evolution_galaxy_iterations(P0, t, xyz, v_xyz, B0, field, case,
                                                          plot=0, iterations=2 , # 3
                                                          galaxy_type=galaxy_type)
                        t1, P1, B1, stages1, v1, Mdot1, ph1, x1, y1, z1, vx1, vy1, vz1 = res
                        if len(stages1[stages1==3]) > 0:
                            df = pd.DataFrame({'t': t1, 'P': P1, 'B': B1, 'stages': stages1,
                                            'v': v1, 'Mdot': Mdot1, 'phase': ph1,
                                            'x': x1, 'y': y1, 'z': z1})
                            df['stages'] = df['stages'].astype('int8')
                            df['phase'] = df['phase'].astype('int8')
                            float_cols = df.select_dtypes(include=['float64']).columns
                            df[float_cols] = df[float_cols].astype('float32')
                            
                            name = 'result/uniform_grid/{}/{}/Z{}R{}V{}B{}_{}'.format(galaxy_type, case, zi, Ri, Vi, Bi, di)
                            path = name + '.feather'
                            
                            feather.write_feather(df, path)
                    # counter += 10


# Parallel(n_jobs=n_cores)(delayed(func)(i) for i in tqdm(range(len(z))))


def create_array():
    shape = (len(z), len(R), len(V), len(P), len(B))
    array = np.zeros(shape)
    for zi in tqdm(range(len(z))):
        for Ri in tqdm(range(len(R))):
            for Vi in range(len(V)):
                for Pi in range(len(P)):
                    for Bi in range(len(B)):
                        for di in range(10):
                            try:
                                name = 'result/uniform_grid/{}/{}/Z{}R{}V{}B{}_{}'.format(galaxy_type, case, zi, Ri, Vi, Bi, di)
                                path = name + '.feather'
                                df = pd.read_feather(path)
                                t = df['t'].values
                                arr = df['stages'].values
                                
                                delta_t = (t[1:] - t[:-1]).astype(np.float32)
                                weights = delta_t
                                
                                arr_for_weights = arr[1:].astype(np.float32)
                                arr_for_weights[arr_for_weights!=3] = 0
                                arr_for_weights[arr_for_weights==3] = 1
                                
                                stage_part = arr_for_weights * delta_t / np.sum(weights)
                                
                                array[zi, Ri, Vi, Pi, Bi] += np.sum(stage_part)
                            except:
                                pass
    np.save('uniform_grid.npy', array)




# create_array()
loaded_array = np.load('uniform_grid.npy')
# loaded_array = np.log10(loaded_array)


# n_samples = 1000
# data_points = np.random.randn(n_samples, 5)
print(np.shape(loaded_array))
# print(np.shape(data_points))

labels = [r'$z_0$, kpc', r'$R_0$, kpc', '$v_0$, km s$^{-1}$', r'log$_{10}P_0$, s', r'log$_{10}B_0$, G']
bins_list = [z_bins, R_bins, V_bins, P_bins, B_bins]

z_counts = np.sum(loaded_array, axis=(1,2,3,4))
R_counts = np.sum(loaded_array, axis=(0,2,3,4))
V_counts = np.sum(loaded_array, axis=(0,1,3,4))
P_counts = np.sum(loaded_array, axis=(0,1,2,4))
B_counts = np.sum(loaded_array, axis=(0,1,2,3))
# print(z_counts)

counts_list = [z_counts, R_counts, V_counts, P_counts, B_counts]
centers_list = [z, R, V, P, B]
# n_vars = data_points.shape[1]
n_vars = 5


fig, axes = plt.subplots(n_vars, n_vars, figsize=(15, 12))
plt.subplots_adjust(wspace=0.20, hspace=0.1)

# vmax = 300*np.max(loaded_array)
vmax = 3000
vmin = np.min(loaded_array)

for i in range(n_vars):
    for j in range(n_vars):
        ax = axes[i, j]
        if j == 0 and i > 0:
            ax.set_ylabel(labels[i])
        else:
            ax.axes.yaxis.set_ticklabels([])
        if i == n_vars - 1: # and j < n_vars - 1:
            ax.set_xlabel(labels[j])
        else:
            ax.axes.xaxis.set_ticklabels([])
        if i == j:
            ax.hist(centers_list[i], bins_list[i], weights=counts_list[i], color='grey', density=True)#, orientation='horizontal') #histtype='stepfilled', 
            # ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            # ax.stairs(counts_list[i], bins_list[i], color='grey', histtype='stepfilled')
            # ax.bar(counts_list[i], centers_list[i], align='center', color='grey')
        elif i > j:
            
            # ax.set_xlabel(labels[i])
            
            
            x_edges = centers_list[j]
            y_edges = centers_list[i]
            
            axes_to_sum = tuple(ax for ax in range(loaded_array.ndim) if ax not in (i, j))
            counts = np.sum(loaded_array, axis=axes_to_sum)
            
            # ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
            # print(np.max(counts))
            cms = ax.pcolormesh(x_edges, y_edges, counts.T, cmap='viridis', shading='auto')
                          # vmin=vmin, vmax=vmax)
        else:
            ax.axis('off')

fig.savefig('triangle.pdf', bbox_inches='tight')