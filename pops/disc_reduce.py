#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 18:32:40 2026

@author: afoninamd
"""
from mpi4py import MPI
import sys
import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import pyarrow.feather as feather

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from main.evolution import star_formation_history #, gett, j_if_disc
from main.constants import galaxy_age, G, M_NS, R_t #, N, R0, Gyr, G, M_sun, R_NS, m_e, m_p, e

comm = MPI.COMM_WORLD
crank = comm.Get_rank()
csize = comm.Get_size()

output_dir = 'result/realistic/'

if crank == 0:
    os.makedirs(output_dir + 'disc_reduced', exist_ok=True)
comm.Barrier()

""" Bins of the five distributions """
arr_size = 501
Vb = np.linspace(0, 500, arr_size, endpoint=True) * 1e5
Mb = 10**np.linspace(4, 24, arr_size, endpoint=True) # log Mdot
Bb = 10**np.linspace(6, 16, arr_size, endpoint=True)
Zb = np.linspace(-10, 10, arr_size, endpoint=True)
Rb = np.linspace(0, 20, arr_size, endpoint=True)
tb = np.linspace(0, galaxy_age, arr_size, endpoint=True)
Tb = 10**np.linspace(4, 10, arr_size, endpoint=True)
c0b = 10**np.linspace(-10, 10, arr_size, endpoint=True)
c1b = 10**np.linspace(-10, 10, arr_size, endpoint=True)
f0b = 10**np.linspace(-30, 20, arr_size, endpoint=True)
f1b = 10**np.linspace(-30, 20, arr_size, endpoint=True)
Tb[0] = 0
c0b[0] = 0
c1b[0] = 0
f0b[0] = 0
f1b[0] = 0
jb = np.linspace(0, 100, arr_size, endpoint=True)


bins = {
'v':    Vb,
'Mdot': Mb,      # log10(Mdot)
'B':    Bb,      # log10(B)
'R':    Rb,      # distance from the GC in the Galactic Plane
'z':    Zb,      # distance from Galactic plane (kpc)
't':    tb,      # age (s)
'T':    Tb,      # log10(T_eff)
'c0':   c0b,
'c1':   c1b,
'f0':   f0b,
'f1':   f1b,
'j':    jb}

if crank == 0:
    pd.DataFrame(bins).to_csv(output_dir + 'disc_reduced/bins.csv', sep=';')


def get_all_i(field, case, galaxy_type):
    disc_dir = os.path.join(output_dir + 'disc/{}_{}_{}/'.format(field,
                                                                 case,
                                                                 galaxy_type))
    i_pulsar = []
    i_magnetar = []

    def scan_dir(path):
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith('.csv'):
                    
                    # num_str = entry.name.split("_", 1)[0] #
                    # 
                    if 'magnetar' in entry.name:
                        num_str = entry.name[:-13]
                        i = int(num_str)
                        i_magnetar.append(i)
                    else:
                        num_str = entry.name[:-4]
                        i = int(num_str)
                        i_pulsar.append(i)

    scan_dir(disc_dir)
    return sorted(i_pulsar), sorted(i_magnetar)


def create_histogram_data(star_type):
    
    if crank == 0:
        for galaxy_type in ['simple', 'two_phase']:
            for field in ['CF', 'ED']:
                for case in ['A', 'B', 'C', 'D']:
                    os.makedirs(output_dir + f'disc_reduced/magnetar/{field}_{case}_{galaxy_type}', exist_ok=True)
                    os.makedirs(output_dir + f'disc_reduced/pulsar/{field}_{case}_{galaxy_type}', exist_ok=True)
    comm.Barrier()
        
    for galaxy_type in ['simple', 'two_phase']:
        for field in ['CF', 'ED']:
            for case in ['A', 'B', 'C', 'D']:
                if crank == 0:
                    print('reading {}_{}_{}_pulsar.txt'.format(field, case, galaxy_type))
                if star_type == '':
                    i_arr = np.loadtxt(output_dir + 'disc_reduced/i_{}_{}_{}_pulsar.txt'.format(field, case, galaxy_type), dtype=int)
                elif star_type == '_magnetar':
                    i_arr = np.loadtxt(output_dir + 'disc_reduced/i_{}_{}_{}_magnetar.txt'.format(field, case, galaxy_type), dtype=int)
                accum = {}
                inside = {}
                outside = {}
            
                for col, edges in bins.items():
                    accum[col] = np.zeros(len(edges) - 1, dtype=np.float64)
                    inside[col] = 0.0
                    outside[col] = 0.0
                # Also for 'R'
                accum['R'] = np.zeros(len(Rb) - 1, dtype=np.float64)
                inside['R'] = 0.0
                outside['R'] = 0.0
                
                """ Assigning files to nodes """
                try:
                    vals_per_core = len(i_arr) // csize
                    remainder = len(i_arr) % csize
                except TypeError:
                    
                    continue
                start_idx = crank * vals_per_core + min(crank, remainder)
                end_idx = start_idx + vals_per_core
                if crank < remainder:
                    end_idx += 1
                my_i_arr = i_arr[start_idx:end_idx]
                
                """ Main loop for histogram creation """
                for i in my_i_arr:
                    fname = os.path.join(output_dir, 'disc',
                                         f'{field}_{case}_{galaxy_type}',
                                         f'{i}{star_type}.csv')
                    if not os.path.isfile(fname):
                        print(f'{i}: not os.path.isfile(fname)')
                        print(f'no {field}_{case}_{galaxy_type}/{i}{star_type}.csv')
                        continue
                    
                    df = pd.read_csv(fname, sep=';')
                    
                    """ Calculating weights """
                    t = np.array(df['t'])
                    j = np.array(df['j'])
                    """ A different turbulence model -> different j """
                    v = np.array(df['v'])
                    R_G = 2 * G * M_NS/ v**2
                    j = j * (R_G/R_t)**(1/6)
                    
                    weight = np.zeros(len(t))
                    weight[1:] = t[1:] - t[:-1]
                    weight = weight * star_formation_history(t)
                    weight = weight / np.sum(weight)
                    weight[j<1] = 0
                    
                    # Update histograms
                    for col, edges in bins.items():
                        # skip if column is missing (but it shouldn't be)
                        if col not in df.columns:
                            continue
                        vals = np.array(df[col])
                        # Use numpy histogram, add to accumulator
                        cnt, _ = np.histogram(vals, bins=np.array(edges), weights=weight)
                        accum[col] += cnt
                        
                        lo, hi = edges[0], edges[-1]
                        mask = (vals >= lo) & (vals <= hi)
                        inside[col] += weight[mask].sum()
                        outside[col] += weight[~mask].sum()
                    
                    # Radial distance histogram
                    
                    R = np.sqrt(np.array(df['x'])**2 + np.array(df['y'])**2)
                    cnt_R, _ = np.histogram(R, bins=Rb, weights=weight)
                    accum['R'] += cnt_R
                    
                    lo, hi = Rb[0], Rb[-1]
                    mask = (vals >= lo) & (vals <= hi)
                    inside['R'] += weight[mask].sum()
                    outside['R'] += weight[~mask].sum()
                
                if star_type == '':
                    star_name = 'pulsar'
                elif star_type == '_magnetar':
                    star_name = 'magnetar'
                    
                pd.DataFrame(accum).to_csv(output_dir + f'disc_reduced/{star_name}/{field}_{case}_{galaxy_type}/{crank}.csv', sep=';')
                pd.DataFrame(inside, index=[0]).to_csv(output_dir + f'disc_reduced/{star_name}/{field}_{case}_{galaxy_type}/{crank}_inside.csv', sep=';')
                pd.DataFrame(outside, index=[0]).to_csv(output_dir + f'disc_reduced/{star_name}/{field}_{case}_{galaxy_type}/{crank}_outside.csv', sep=';')


def create_txt_i_arr():
    """ Write down the number corresponding to discs in initial distributions """
    if crank == 0:
        galaxy_type_arr = ['simple', 'simple', 'simple', 'simple',
                            'simple', 'simple', 'simple', 'simple',
                            'two_phase', 'two_phase', 'two_phase', 'two_phase',
                            'two_phase', 'two_phase', 'two_phase', 'two_phase']
        field_arr =  ['CF', 'CF', 'CF', 'CF', 'ED', 'ED', 'ED', 'ED',
                      'CF', 'CF', 'CF', 'CF', 'ED', 'ED', 'ED', 'ED']
        case_arr = ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D',
                    'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D']
        for j in range(16):
            galaxy_type = galaxy_type_arr[j]
            field = field_arr[j]
            case = case_arr[j]
            i_p, i_m = get_all_i(field, case, galaxy_type)
            
            np.savetxt(output_dir + 'disc_reduced/i_{}_{}_{}_pulsar.txt'.format(field, case, galaxy_type),
                        i_p, fmt='%d') # to check initial conditions if needed
            np.savetxt(output_dir + 'disc_reduced/i_{}_{}_{}_magnetar.txt'.format(field, case, galaxy_type),
                        i_m, fmt='%d') # to check initial conditions if needed
    comm.Barrier()



create_txt_i_arr()

create_histogram_data(star_type='_magnetar')
create_histogram_data(star_type='')


# galaxy_type = galaxy_type_arr[crank]
# field = field_arr[crank]
# case = case_arr[crank]

# npy_dir = output_dir + 'disc_reduced/npy/'
# if not os.path.exists(npy_dir):
#     os.makedirs(npy_dir)

# df = pd.read_csv('/home/afoninamd/Documents/NS/project/pops/result/realistic/disc_reduced/magnetar/CF_A_simple/0.csv', sep=';')
# T = np.array(df['T'])
# plt.scatter(Tb[1:], T)


