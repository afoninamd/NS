#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 15:13:44 2025

@author: afoninamd
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import glob
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from time import time
from main.constants import Gyr, galaxy_age
from pops.popsynthesis import path_dir, path_result


def star_formation_rate(t: np.array) -> np.array:
    """ array in s, interpolation from Haywood 2016, Fig.4, fine blue line """
    t = t / Gyr
    SFR = np.zeros(len(t))
    SFR[t<=7] = 0.1
    cond1 = np.logical_and(t>8.5, t<10)
    SFR[cond1] = 0.5 * 2 / 3 * (t[cond1]-8.5)
    cond2 = np.logical_and(t>10, t<=12.5)
    SFR[cond2] = 0.5
    SFR[t>12.5] = 0.5 - 0.5/(13.8-12.5) * (t[t>12.5]-12.5)
    SFR[SFR<0] = 0
    t = t * Gyr
    return SFR #/ np.sum(SFR) # 2.65 -- analytical coefficient # M_sun per second


def hist_T(galaxy_type: str, star_type: str, field: str, case: str, N: int):
    bins = np.linspace(3.5, 7, 401)
    cts = np.zeros(len(bins)-1)
    for i in tqdm(range(N)):
        try:
            filename = path_result + '{}/{}/{}/{}/{}.feather'.format(galaxy_type, field,
                                                     star_type, case, i)
            df = pd.read_feather(filename)
            t = df['t'].values
            arr = np.log10(df['T'].values)
            # arr = np.log10(arr)
            
            delta_t = np.zeros(len(t))
            delta_t[1:] = (t[1:] - t[:-1]).astype(np.float32)
            weight = delta_t
            weight *= star_formation_rate(t)
            weight = weight / np.sum(weight)
            
            counts, _ = np.histogram(arr, bins=bins, weights=weight)
            cts += counts
        except FileNotFoundError:
            pass
    return bins, cts

# for galaxy_type in ['simple', 'two_phase']:
#     case = 'A'
#     for star_type in ['magnetar', 'pulsar']:
#         if star_type == 'magnetar':
#             N = 10_000
#         else:
#             N = 90_000
#         for field in ['CF', 'ED']:
#             i, a = hist_T(galaxy_type, star_type, field, case, N)
#             i = (i[1:] + i[:-1])/2
#             if star_type != 'magnetar':
#                 a = a / 9
#             if field == 'CF':
#                 ls = '-'
#             else:
#                 ls = '--'
#             plt.plot(i, a, alpha=0.8, ls=ls, label='{}_{}_{}'.format(galaxy_type, star_type, field))
# plt.legend()


def logN_logS_table(galaxy_type: str, star_type: str, field: str, case: str,
             N: int, is_sfr: bool):
    i_array = np.arange(N)
    f_edges = np.linspace(-40, -10, 301, endpoint=True)
    c_edges = np.linspace(-30, 0, 301, endpoint=True)
    f0_array = np.zeros(len(f_edges))
    f1_array = np.zeros(len(f_edges))
    c0_array = np.zeros(len(c_edges))
    c1_array = np.zeros(len(c_edges))
    for i in tqdm(i_array):
        try:
            """ Reading the data """
            filename = path_result + '{}/{}/{}/{}/{}.feather'.format(galaxy_type, field,
                                                     star_type, case, i)
            df = pd.read_feather(filename)
            stages = df['stages'].values
            if len(stages[stages==3]) == 0: # skip if no accretors
                continue
            t = df['t'].values
            f0 = df['f0'].values
            f1 = df['f1'].values
            c0 = df['c0'].values
            c1 = df['c1'].values
            # x = df['x'].values
            # y = df['y'].values
            # z = df['z'].values
            
            """ Normalization """
            delta_t = np.zeros(len(t))
            delta_t[1:] = (t[1:] - t[:-1]).astype(np.float32)
            weight = delta_t
            if is_sfr:
                weight *= star_formation_rate(t)
            weight = weight / np.sum(weight)
            
            f0 = np.log10(f0)
            f1 = np.log10(f1)
            c0 = np.log10(c0)
            c1 = np.log10(c1)
            
            for j in range(len(f_edges)):
                f0_array[j] += np.sum(weight[f0>f_edges[j]])
                f1_array[j] += np.sum(weight[f1>f_edges[j]])
                
                c0_array[j] += np.sum(weight[c0>c_edges[j]])
                c1_array[j] += np.sum(weight[c1>c_edges[j]])
                
        except FileNotFoundError:
            pass
        
    path = path_result + 'table/logNlogS_sfr{}_{}_{}_{}_{}_{}.csv'.format(is_sfr, galaxy_type, field,
                                             star_type, case, N)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
        
    data = pd.DataFrame({'f_edges': f_edges, 'c_edges': c_edges,
                         'f0': f0_array, 'f1': f1_array,
                         'c0': c0_array, 'c1': c1_array})
    data.to_csv(path)
    # return i_array, A_array


def six_figures(options=''):
        
    figc, axesc = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(12, 15))
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    
    figf, axesf = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(12, 15))
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    
    alpha = 0.8
    weight = 1000
    is_sfr = True
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['text.usetex'] = True
    colors = ['C0', 'C1']
    fields = ['CF', 'ED']
    cases = ['A', 'B', 'C']
    for j in range(2):
        field = fields[j]
        for i in range(3): #
            case = cases[i]
            for k in [0, 1]:
                galaxy_type = ['simple', 'two_phase'][k]
                ls2 = '-' # ['-', ':'][k]
                ls1 = ':' # ['-', ':'][k]
                galaxy_type_str = ['one phase', 'two phases'][k]
                """ Calculation """
                # logN_logS_table(galaxy_type, 'magnetar', field, case, N=10_000, is_sfr)
                # logN_logS_table(galaxy_type, 'pelsar', field, case, N=90_000, is_sfr)
                
                """ Figures """
                path = path_result + 'table/logNlogS_sfr{}_{}_{}_pulsar_{}_90000.csv'.format(is_sfr, galaxy_type, field, case)
                data = pd.read_csv(path)
                f = 10**data['f_edges']
                c = 10**data['c_edges']
                f0 = data['f0'] * weight
                f1 = data['f1'] * weight
                c0 = data['c0'] * weight
                c1 = data['c1'] * weight
                
                # axesc[i, j].plot(c, c0, c='C0', ls=ls2, alpha=alpha)
                axesc[i, j].plot(c, c1, c=colors[k], ls=ls1, alpha=alpha, label='pulsars, {}'.format(galaxy_type_str))
                
                # axesf[i, j].plot(f, f0, c='C0', ls=ls2,  alpha=alpha)
                axesf[i, j].plot(f, f1, c=colors[k], ls=ls1, alpha=alpha, label='pulsars, {}'.format(galaxy_type_str))
                
                
                path = path_result + 'table/logNlogS_sfr{}_{}_{}_magnetar_{}_10000.csv'.format(is_sfr, galaxy_type, field, case)
                data = pd.read_csv(path)
                
                f0 += data['f0'] * weight
                f1 += data['f1'] * weight
                c0 += data['c0'] * weight
                c1 += data['c1'] * weight
                
                # axesc[i, j].plot(c, c0, c='C1', ls=ls2, alpha=alpha)
                axesc[i, j].plot(c, c1, c=colors[k], ls=ls2, alpha=alpha, label='+ magnetars, {}'.format(galaxy_type_str))
                
                # axesf[i, j].plot(f, f0, c='C1', ls=ls2, alpha=alpha)
                axesf[i, j].plot(f, f1, c=colors[k], ls=ls2, alpha=alpha, label='+ magnetars, {}'.format(galaxy_type_str))
                
            for axes in [axesc, axesf]:
                axes[i, j].set_yscale('log')
                axes[i, j].set_xscale('log')
                axes[i, j].tick_params(axis='both', labelsize=10)
                axes[i, j].grid(alpha=0.5)
                
        for i in range(2):
            axesc[2, i].set_xlabel(r'CR$_0$, s$^{-1}$')
            axesf[2, i].set_xlabel(r'$F_0$, erg cm$^{-2}$ s$^{-1}$')
        for i in range(3):
            axesc[i, 0].set_ylabel(r'N(CR$>$CR$_0$)')
            axesf[i, 0].set_ylabel(r'N($F>F_0$)')
        
    if options == '_zoomed':
        for i in range(3):
            for j in range(2):
                axesf[i, j].set_xlim([1e-15, 3e-11])
                axesc[i, j].set_xlim([1e-4, 3.])
                
                axesf[i, j].set_ylim([1., 3e4])
                axesc[i, j].set_ylim([1., 3e4])
                
                axesf[i, j].text(1.5e-15, 1.2e1, fields[j], fontsize=14)
                axesc[i, j].text(1.5e-4, 1.2e1, fields[j], fontsize=14)
                
                axesf[i, j].text(1.5e-15, 4, 'case {}'.format(cases[i]), fontsize=14)
                axesc[i, j].text(1.5e-4, 4, 'case {}'.format(cases[i]), fontsize=14)
    else:
        for i in range(3):
            for j in range(2):
                axesf[i, j].set_ylim([1., 3e7])
                axesc[i, j].set_ylim([1., 3e7])
                
                axesf[i, j].text(1e-13, 2e6, fields[j], fontsize=16)
                axesc[i, j].text(1e-3, 2e6, fields[j], fontsize=16)
                
                axesf[i, j].text(1e-13, 3e5, 'case {}'.format(cases[i]), fontsize=14)
                axesc[i, j].text(1e-3, 3e5, 'case {}'.format(cases[i]), fontsize=14)
    # else:
    for axes in [axesc, axesf]:
        axes[0, 0].legend(fontsize=12)

    figc.savefig('counts{}.pdf'.format(options), bbox_inches='tight')
    figf.savefig('flux{}.pdf'.format(options), bbox_inches='tight')

# for options in ['', '_zoomed']:
#     six_figures(options)
""" Random functions ahead """




def accretor_part(galaxy_type: str, star_type: str, field: str, case: str,
                  N: int):
    i_array = np.arange(N)
    A_array = np.zeros(N)
    for i in tqdm(i_array):
        try:
            filename = path_result + '{}/{}/{}/{}/{}.feather'.format(galaxy_type, field,
                                                     star_type, case, i)
            df = pd.read_feather(filename)
            t = df['t'].values
            arr = df['stages'].values
            # arr = np.log10(arr)
            
            delta_t = (t[1:] - t[:-1]).astype(np.float32) / galaxy_age
            arr_for_weights = arr[1:].astype(np.float32)
            
            arr_for_weights[arr_for_weights!=3] = 0
            arr_for_weights[arr_for_weights==3] = 1
            
            stage_part = arr_for_weights * delta_t

            A_array[i] = np.sum(stage_part)
        except FileNotFoundError:
            pass
    return i_array, A_array


def just_histogram():
    fig, ax = plt.subplots()
    for galaxy_type in ['simple', 'two_phase']:
        star_type = 'pulsar'
        field = 'CF'
        case = 'C'
        i, A = accretor_part(galaxy_type, star_type, field, case,
                      N=10000)
        A = np.log10(A)
        bins = np.linspace(-4, 0, 30)
        plt.hist(A, alpha=0.5, bins=bins)


def weighted(star_type, case, galaxy_type, field):
    # Define your file path pattern (adjust accordingly)
    # Example: 'path/to/your/files/galaxy_type/field/star_type/case/*.feather'
    path_pattern = path_result + '{}/{}/{}/{}/*.feather'.format(galaxy_type, field,
                                                     star_type, case)
    
    # Define your bin edges before reading files
    start_bin = 1  # example min log10(P) -2 12
    end_bin = 4     # example max log10(B) 7 15
    num_bins = 4
    bin_edges = np.linspace(start_bin, end_bin, num_bins + 1, endpoint=True)
    
    binned_counts = np.zeros(num_bins)
    
    all_arr = []
    all_weights = []
    for filename in tqdm(glob.glob(path_pattern, recursive=False)):
        df = pd.read_feather(filename)
        t = df['t'].values
        arr = df['stages'].values
        # arr = np.log10(arr)
        
        delta_t = (t[1:] - t[:-1]).astype(np.float32)
        arr_for_weights = arr[1:].astype(np.float32)
        
        # Append to the overall lists
        # all_arr.extend(arr_for_weights)
        # all_weights.extend(delta_t)
        
        # Compute histogram for current file
        counts, _ = np.histogram(arr_for_weights, bins=bin_edges, weights=delta_t)
        binned_counts += counts

    return bin_edges, binned_counts
    return all_arr, all_weights
# Plot histogram with combined data


def two_dimensional_histogram():
    galaxy_type = 'two_phase'
    star_type = 'pulsar'
    field = 'CF'
    case='A'

    data = pd.read_csv('distribution_{}.csv'.format(star_type), sep=';')
    # B0 = (data['z']) #np.log10
    B0 = (data['x']**2 + data['y']**2)**0.5
    v0 = ((data['Vx']+data['Vxpec'])**2+(data['Vy']+data['Vypec'])**2+(data['Vz']+data['Vzpec'])**2)**0.5/1e5
    # v0 = np.log10(data['P']) #
    # v0 = (data['x']**2 + data['y']**2)**0.5
    
    # B0 = B0[:10000]
    # v0 = v0[:10000]

    i_array, A_array = accretor_part(galaxy_type, star_type, field, case,
                      N=150000)
    A = (A_array)#[A_array>0]
    # plt.plot(i_array, A_array)
    
    Bw = np.zeros(len(B0)) # weight
    Bw[i_array] = A_array
    vw = np.zeros(len(v0)) # weight
    vw[i_array] = A_array
    
    # ax.hist2d(B0, v0, bins=30, cmap='viridis') #, weights=A)
    weights = Bw * vw
    
    
    # Define number of bins
    bins_x = 20
    bins_y = 100
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    plt.hist2d(B0, v0, bins=50, weights=weights, cmap='viridis')
    # hist_unweighted, xedges, yedges = np.histogram2d(B0, v0, bins=[bins_x, bins_y])
    # hist_weighted, _, _ = np.histogram2d(B0, v0, bins=[xedges, yedges], weights=weights)
    # # To avoid division by zero, set zeros in unweighted histogram to NaN or a small number
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     ratio = np.true_divide(hist_weighted, hist_unweighted)
    #     ratio[hist_unweighted == 0] = 0  # or set to zero if preferred
    # plt.imshow(ratio.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    #        aspect='auto', cmap='viridis') #, interpolation='bilinear')
    
    plt.colorbar(label='Fraction of the accretor stage')
    plt.xlabel('$R_0$')
    plt.ylabel('$v_0$')
    # plt.ylim(0, 500)
    # plt.title('Ratio of Weighted to Unweighted Histogram Counts')
    
    
    
    return 0 
    

    # plt.title('Weighted 2D Histogram of $B$ vs $v$')
    
    # Add colorbar
    plt.colorbar(label='Weighted Counts')
    
    
    

    # Unique categories in A
    # categories = np.unique(A)
    
    # Plot each category separately
    # for cat in categories:
    #     mask = A == cat
    #     plt.scatter(B0[mask], v0[mask], label=f'A={cat}', alpha=0.5)
    
    # ax.set_xlabel('B0')
    # ax.set_ylabel('P0')
    # # plt.legend()
    # # plt.title('Scatter plot colored by A')
    
    
    # num=10
    # fig, ax = plt.subplots(nrows=2)
    # countsB01, binsB0 = np.histogram(B0, weights=A, bins=num)
    # countsv01, binsv0 = np.histogram(v0, weights=A, bins=num)
    
    # countsB0, _ = np.histogram(B0, bins=num)
    # countsv0, _ = np.histogram(v0, bins=num)
    
    # ax[0].plot(binsB0[:-1], countsB01/countsB0)
    # ax[1].plot(binsv0[:-1], countsv01/countsv0)
    
    # # ax[0].hist(B0, weights=A, bins=30)
    # # ax[1].hist(v0, weights=A, bins=30)
    
    # # plt.hist2d(x, y, bins=30, cmap='viridis')  # You can change the colormap
    
    # # # Add colorbar to show the scale
    # # plt.colorbar(label='Counts')
    # fig, ax = plt.subplots(figsize=(10,6))
    # ax.hist2d(B0, v0, bins=30, cmap='viridis')


two_dimensional_histogram()



# for galaxy_type in ['simple','two_phase']:
#     for case in ['A']: #, 'B', 'C', 'D']:
#         for field in ['CF']:
#             star_type = 'magnetar'
#             # star_type='pulsar'
#             # case='A'
#             # all_arr, all_weights = weighted(star_type, case, galaxy_type, field)
#             # plt.hist(all_arr, weights=all_weights, bins=50)
#             bin_edges, binned_counts = weighted(star_type, case, galaxy_type, field)
#             bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#             # bin_centers = bin_edges[1:]
#             bin_width = np.diff(bin_edges)
#             plt.bar(bin_centers, binned_counts, width=bin_width, align='center', alpha=0.5)
#             plt.xlabel('Array')
#             plt.ylabel('Weighted count (by Δt)')
#             plt.title('Weighted Histogram of Array across all files of '+ star_type + ' in a ' + galaxy_type + ' galaxy')
# time0 = time()
# print('it took {:.0f} s'.format(time()-time0))

# print(100000*3*1e18/13.8/Gyr/1e5)