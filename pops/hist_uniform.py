#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 13:11:24 2025

@author: afoninamd
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from main.constants import N, output_dir
from main.evolution import star_formation_history


cur_dir = output_dir + 'npy/'
if not os.path.exists(cur_dir):
    os.mkdir(cur_dir)

""" Bins of the five distributions """
Vb = np.linspace(0, 500, 51, endpoint=True)
Pb = np.linspace(-2, 3, 51, endpoint=True)
Bb = np.linspace(10, 15, 51, endpoint=True)
Zb = np.linspace(-0.5, 0.5, 51, endpoint=True)
Rb = np.linspace(0, 20, 51, endpoint=True)

cur_stage = 3

def create_uniform():
    
    """ Reading the distribution data """
    df = pd.read_csv('distribution_{}.csv'.format(N), sep=';')
    V = np.array((df['Vx']+df['Vxpec'])**2 + (df['Vy']+df['Vypec'])**2 + (df['Vz']+df['Vzpec'])**2)**0.5 / 1e5
    P = np.log10(df['P'])  # s
    B = np.log10(df['B'])  # G
    Z = np.array(df['z'])  # kpc
    R = np.array(df['x']**2 + df['y']**2)**0.5 # kpc
    
    """ find the bin """
    Vi = np.searchsorted(Vb, V) - 1
    Pi = np.searchsorted(Pb, P) - 1
    Bi = np.searchsorted(Bb, B) - 1
    Zi = np.searchsorted(Zb, Z) - 1
    Ri = np.searchsorted(Rb, R) - 1
    
    Vi[Vi>=len(Vb)-1] = len(Vb) - 2  # so there is no indexes > len(array)
    Pi[Pi>=len(Pb)-1] = len(Pb) - 2
    Bi[Bi>=len(Bb)-1] = len(Bb) - 2
    Zi[Zi>=len(Zb)-1] = len(Zb) - 2
    Ri[Ri>=len(Rb)-1] = len(Rb) - 2
    
    galaxy_type, field, case = 'simple', 'CF', 'A'
    sfh = True
    for galaxy_type in ['simple', 'two_phase']:
        for field in ['CF', 'ED']:
            for case in ['A', 'B', 'C']:
                for sfh in [False]:
                    Array = np.zeros([len(Vb)-1, len(Pb)-1, len(Bb)-1, len(Zb)-1, len(Rb)-1], dtype=np.float32)
                    for i in range(N):
                        path_i = output_dir + '{}/{}/{}/{}.feather'.format(galaxy_type, field, case, i)
                        if os.path.exists(path_i):
                            """ read one track """
                            f = pd.read_feather(path_i)
                            t = np.array(f['t'])
                            stage = np.array(f['stages'])
                            
                            """ count weights """
                            weight = t[1:] - t[:-1]
                            if sfh:
                                weight = weight * star_formation_history(t[1:])
                            weight = weight / np.sum(weight)
                            weight[stage[1:] != cur_stage] = 0
                        
                            """ add weights to the bin """
                            Array[Vi[i], Pi[i], Bi[i], Zi[i], Ri[i]] += np.sum(weight)
                    np.save(cur_dir+f'{galaxy_type}_{field}_{case}_sfh{sfh}_stage{cur_stage}', Array)


def plot_uniform(galaxy_type='simple', field='CF', case='A', sfh=True, stage=cur_stage):
    loaded_array = np.load(cur_dir+f'{galaxy_type}_{field}_{case}_sfh{sfh}_stage{cur_stage}.npy')
    # loaded_array = np.log10(loaded_array)
    
    print(np.shape(loaded_array))
    labels = [r'$z_0$, kpc', r'$R_0$, kpc', '$v_0$, km s$^{-1}$', r'log$_{10}P_0$, s', r'log$_{10}B_0$, G']
    bins_list = [Zb, Rb, Vb, Pb, Bb]
    
    V_counts = np.sum(loaded_array, axis=(1,2,3,4))
    P_counts = np.sum(loaded_array, axis=(0,2,3,4))
    B_counts = np.sum(loaded_array, axis=(0,1,3,4))
    Z_counts = np.sum(loaded_array, axis=(0,1,2,4))
    R_counts = np.sum(loaded_array, axis=(0,1,2,3))
    
    V_center = (Vb[1:] + Vb[:-1]) / 2
    P_center = (Pb[1:] + Pb[:-1]) / 2
    B_center = (Bb[1:] + Bb[:-1]) / 2
    Z_center = (Zb[1:] + Zb[:-1]) / 2
    R_center = (Rb[1:] + Rb[:-1]) / 2
    
    counts_list = [Z_counts, R_counts, V_counts, P_counts, B_counts]
    centers_list = [Z_center, R_center, V_center, P_center, B_center]
    # n_vars = data_points.shape[1]
    n_vars = 5
    
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(15, 12))
    plt.subplots_adjust(wspace=0.20, hspace=0.1)
    
    # vmax = 300*np.max(loaded_array)
    # vmax = 3000
    # vmin = np.min(loaded_array)
    
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
    
    if not os.path.exists(output_dir + 'figures/'):
        os.mkdir(output_dir + 'figures/')
    fig.savefig(output_dir + 'figures/triangle_{galaxy_type}_{field}_{case}_sfh{sfh}_stage{cur_stage}.pdf', bbox_inches='tight')

plot_uniform()