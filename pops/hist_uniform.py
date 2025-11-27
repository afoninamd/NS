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
from PyPDF2 import PdfMerger
import scipy as sp
import matplotlib as mpl
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from main.constants import N, output_dir, G, M_NS, m_p, galaxy_age
from main.model import Object

mpl.rcParams["text.usetex"] = True

csize = 480

output_dir=''
npy_dir = output_dir + 'npy/'
if not os.path.exists(npy_dir):
    os.makedirs(npy_dir)

""" Bins of the five distributions """
arr_size = 51
Vb = np.linspace(0, 500, arr_size, endpoint=True)
Pb = np.linspace(-2, 3, arr_size, endpoint=True)
Bb = np.linspace(10, 15, arr_size, endpoint=True)
Zb = np.linspace(-0.5, 0.5, arr_size, endpoint=True)
Rb = np.linspace(0, 20, arr_size, endpoint=True)

cur_stage = 3

def create_uniform(N, res_name):
    # res_name = 'result1'
    # if N == 9_000_000:
    #     res_name = 'result2'
    # i_array = []
    # i_array = np.zeros(arr_size)
    """ Reading the distribution data """
    df = pd.read_csv(res_name+'/distribution_{}.csv'.format(N), sep=';')
    # V = np.array((df['Vx']+df['Vxpec'])**2 + (df['Vy']+df['Vypec'])**2 + (df['Vz']+df['Vzpec'])**2)**0.5 / 1e5
    V = np.array((df['Vx'])**2 + (df['Vy'])**2 + (df['Vz'])**2)**0.5 / 1e5
    P = np.log10(df['P'])  # s
    B = np.log10(df['B'])  # G
    Z = np.array(df['z'])  # kpc
    R = (np.array(df['x'])**2 + np.array(df['y'])**2)**0.5 # kpc
    
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
    
    galaxy_type_arr = ['simple', 'two_phase']
    field_arr = ['CF', 'ED']
    case_arr = ['A', 'B', 'C', 'D']
    if res_name == 'result2':
        galaxy_type_arr = ['simple']
        field_arr = ['CF']
        case_arr = ['A', 'C']
    if res_name == 'result3':
        galaxy_type_arr = ['two_phase']
        field_arr = ['ED']
        case_arr = ['B']
        
    
    for galaxy_type in galaxy_type_arr:
        for field in field_arr:
            for case in case_arr:
                Array = np.zeros([len(Vb)-1, len(Pb)-1, len(Bb)-1, len(Zb)-1, len(Rb)-1], dtype=np.float32)
                for crank in range(csize):
                    file_name = res_name+'/result/uniform/{}_{}_{}_{}'.format(crank, galaxy_type, field, case)
                    data = np.loadtxt(file_name+'.txt', dtype=float)
                    data = np.array(data)
                    try:
                        index = np.array(data[:, 0], dtype=int)
                        accretor_part = np.array(data[:, 1])
                        for j in range(len(index)):
                            i = index[j]
                            Array[Vi[i], Bi[i], Pi[i], Ri[i], Zi[i]] += accretor_part[j]
                    except BaseException as e:
                        pass
                    # folder_path = output_dir + '{}/{}/{}/'.format(galaxy_type, field, case)
                    # file_list = [entry.name for entry in os.scandir(folder_path) if entry.is_file()]
                    # for i in range(N):
                    # # for file in file_list:
                    #     path_i = output_dir + '{}/{}/{}/{}.feather'.format(galaxy_type, field, case, i)
                    #     # if os.path.exists(path_i):
                    #     try:
                    #         """ read one track """
                    #         f = pd.read_feather(path_i)
                    #         # f = pd.read_feather(file)
                    #         t = np.array(f['t'])
                    #         stage = np.array(f['stages'])
                            
                    #         """ count weights """
                    #         weight = t[1:] - t[:-1]
                    #         if sfh:
                    #             weight = weight * star_formation_history(t[1:])
                    #         weight = weight / np.sum(weight)
                    #         weight[stage[1:] != cur_stage] = 0
                    #         """ add weights to the bin """
                        # except FileNotFoundError:
                        #     pass
                    # if field == 'CF' and galaxy_type == 'simple':
                    #     for i in range(arr_size):
                    #         if Array[Vi[i], Bi[i], Ri[i], Zi[i], Pi[i]] != 0:
                    #             i_array = i_array.append(i)
                            # print(i)
                np.save(npy_dir+'{}_{}_{}_{}'.format(galaxy_type, field, case, N), Array)
    # print(i_array)


def plot_uniform(galaxy_type='simple', field='ED', case='A', res_name = 'result3'):
    # loaded_array = np.load(npy_dir+'{}_{}_{}_{}.npy'.format(galaxy_type, field, case, 1_000_000))
    # N_tracks = '1 million'
    # if (case == 'A' or case == 'C') and galaxy_type == 'simple' and field == 'CF':
    #     loaded_array += np.load(npy_dir+'{}_{}_{}_{}.npy'.format(galaxy_type, field, case, 9_000_000))
    #     N_tracks = '10 million'
    # if res_name == 'result3':
    #     loaded_array += np.load(npy_dir+'{}_{}_{}_{}.npy'.format(galaxy_type, field, case, 9_000_000))
    
    
    loaded_array = np.load(npy_dir+'{}_{}_{}_{}.npy'.format(galaxy_type, field, case, 1_000_000))
    loaded_array += np.load(npy_dir+'{}_{}_{}_{}.npy'.format(galaxy_type, field, case, 9_000_000))
    
    
    # title = 'Galaxy is {}, field {}, propeller {}, {} tracks'.format(galaxy_type, field, case, N_tracks)
    # i_array = np.numpy(range(51))
    # print(loaded_array[loaded_array!=0])
    # print(loaded_array!=0)
    loaded_array = loaded_array / 10e6 * arr_size**2
    
    labels = ['$v_{kick}$, km s$^{-1}$', r'log$_{10}B_0$, G', r'log$_{10}P_0$, s',  r'$R_0$, kpc', r'$z_0$, kpc']
    bins_list = [Vb, Bb, Pb, Rb, Zb]
    
    V_counts = np.sum(loaded_array, axis=(1,2,3,4))
    B_counts = np.sum(loaded_array, axis=(0,2,3,4))
    P_counts = np.sum(loaded_array, axis=(0,1,3,4))
    R_counts = np.sum(loaded_array, axis=(0,1,2,4))
    Z_counts = np.sum(loaded_array, axis=(0,1,2,3))
    
    V_center = (Vb[1:] + Vb[:-1]) / 2
    P_center = (Pb[1:] + Pb[:-1]) / 2
    B_center = (Bb[1:] + Bb[:-1]) / 2
    Z_center = (Zb[1:] + Zb[:-1]) / 2
    R_center = (Rb[1:] + Rb[:-1]) / 2
    
    counts_list = [V_counts, B_counts, P_counts, R_counts, Z_counts]
    centers_list = [V_center, B_center, P_center, R_center, Z_center]
    # n_vars = data_points.shape[1]
    n_vars = 5
    
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(14, 12))
    plt.subplots_adjust(wspace=0.35, hspace=0.15)
    
    # vmax = 300*np.max(loaded_array)
    vmax = 1
    vmin = 0
    
    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            # else:
                # ax.axes.yaxis.set_ticklabels([])
            if i == n_vars - 1: # and j < n_vars - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.axes.xaxis.set_ticklabels([])
            if i == j:
                ax.hist(centers_list[i], bins_list[i], weights=counts_list[i]/arr_size, color='grey', density=False, rasterized=True)#, orientation='horizontal') #histtype='stepfilled', 
                # ax.axes.xaxis.set_ticklabels([])
                # ax.axes.yaxis.set_ticklabels([])
                
                ax.set_xlim([bins_list[i][0], bins_list[i][-1]])
                # ax.stairs(counts_list[i], bins_list[i], color='grey', histtype='stepfilled')
                # ax.bar(counts_list[i], centers_list[i], align='center', color='grey')
            elif i > j:
                
                # ax.set_xlabel(labels[i])
                x_edges = centers_list[j]
                y_edges = centers_list[i]
                
                ax.set_xlim([bins_list[j][0], bins_list[j][-1]])
                
                axes_to_sum = tuple(ax for ax in range(loaded_array.ndim) if ax not in (i, j))
                counts = np.sum(loaded_array, axis=axes_to_sum)
                
                levels = np.array([1e-3, 0.01, 0.1, 0.5]) #, 0.5, 0.8])
                # if j > 0:
                #     levels = []
                # if j == 1 and i == 4    :
                #     levels = np.array([1e-3, 0.01, 0.1])
                # levels = np.array([1e-3, 0.01, 0.1, 0.2]) #, 0.5, 0.8])
                
                cms = ax.pcolormesh(x_edges, y_edges, counts.T, cmap='viridis', shading='auto', rasterized=True, vmin=vmin, vmax=vmax)
                
                smoothed_counts = sp.ndimage.gaussian_filter(counts.T, sigma=2.5)
                cnt = ax.contour(x_edges, y_edges, smoothed_counts, levels, colors='white', rasterized=True, linewidths=0.5,
                                    vmin=vmin, vmax=vmax)
                mpl.rcParams["text.usetex"] = False
                ax.clabel(cnt, levels, fontsize=10, fmt='%r') #, manual=True) #, fmt='%.1f')
                mpl.rcParams["text.usetex"] = True
                
                # ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
                # print(np.max(counts))
                # cms = ax.pcolormesh(x_edges, y_edges, counts.T, cmap='viridis', shading='auto', rasterized=True)
                                    # vmin=vmin, vmax=vmax)
            else:
                ax.axis('off')
    cbar_ax = fig.add_axes([0.92, 0.108, 0.02, 0.77])  # [left, bottom, width, height]
    fig.colorbar(cms, cax=cbar_ax, label='Fraction of the accretor stage')
    
    axes[4, 2].set_xticks([-2, 0, 2])
    
    axes[2, 1].text(14, 2, r'III', color='white', fontsize=20)#, ha='right')
    axes[2, 0].text(400, 2, r'II', color='white', fontsize=20)#, ha='right')
    axes[1, 0].text(400, 14, r'I', color='white', fontsize=20)#, ha='right')
    
    if not os.path.exists(output_dir + 'figures/'):
        os.mkdir(output_dir + 'figures/')
    # fig.suptitle(title, fontsize=20)
    fig.savefig(output_dir + f'figures/triangle_{galaxy_type}_{field}_{case}.pdf', bbox_inches='tight')


def plot_uniform_article(N=9_000_000, case='A', galaxy_type='simple', field='CF'):
    loaded_array = np.load(npy_dir+'{}_{}_{}_{}.npy'.format(galaxy_type, field, case, 1_000_000))
    loaded_array += np.load(npy_dir+'{}_{}_{}_{}.npy'.format(galaxy_type, field, case, 9_000_000))
    loaded_array = loaded_array / 10e6 * arr_size**2
    # norm_const = np.sum(loaded_array)
    # N_tracks = '10 million'
    # title = 'Galaxy is {}, field {}, propeller {}, {} tracks'.format(galaxy_type, field, case, N_tracks)
    # i_array = np.numpy(range(51))
    # print(loaded_array[loaded_array!=0])
    # print(loaded_array!=0)
    
    labels = ['$v_{kick}$, km s$^{-1}$', r'log$_{10}B_0$, G', r'log$_{10}P_0$, s',  r'$R_0$, kpc', r'$z_0$, kpc']
    bins_list = [Vb, Bb, Pb, Rb, Zb]
    
    V_counts = np.sum(loaded_array, axis=(1,2,3,4))
    B_counts = np.sum(loaded_array, axis=(0,2,3,4))
    P_counts = np.sum(loaded_array, axis=(0,1,3,4))
    R_counts = np.sum(loaded_array, axis=(0,1,2,4))
    Z_counts = np.sum(loaded_array, axis=(0,1,2,3))
    
    V_center = (Vb[1:] + Vb[:-1]) / 2
    P_center = (Pb[1:] + Pb[:-1]) / 2
    B_center = (Bb[1:] + Bb[:-1]) / 2
    Z_center = (Zb[1:] + Zb[:-1]) / 2
    R_center = (Rb[1:] + Rb[:-1]) / 2
    
    counts_list = [V_counts, B_counts, P_counts, R_counts, Z_counts]
    centers_list = [V_center, B_center, P_center, R_center, Z_center]

    # n_vars = data_points.shape[1]
    n_vars = 3
    
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(14, 12))
    plt.subplots_adjust(wspace=0.20, hspace=0.1)
    
    # vmax = 300*np.max(loaded_array)
    vmax = 1
    vmin = 0
    
    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            else:
                pass
                # ax.axes.yaxis.set_ticklabels([])
            if i == n_vars - 1: # and j < n_vars - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.axes.xaxis.set_ticklabels([])
            if i == j:
                # if i == 2: i = 4
                ax.hist(centers_list[i], bins_list[i], weights=counts_list[i]/arr_size, color='grey', density=False, rasterized=True)#, orientation='horizontal') #histtype='stepfilled', 
                # ax.axes.xaxis.set_ticklabels([])
                # ax.axes.yaxis.set_ticklabels([])
                
                ax.set_xlim([bins_list[i][0], bins_list[i][-1]])
                # ax.stairs(counts_list[i], bins_list[i], color='grey', histtype='stepfilled')
                # ax.bar(counts_list[i], centers_list[i], align='center', color='grey')
            elif i > j:
                # if i == 2: i = 4
                # if j == 2: j = 4
                # ax.set_xlabel(labels[i])
                x_edges = centers_list[j]
                y_edges = centers_list[i]
                
                ax.set_xlim([bins_list[j][0], bins_list[j][-1]])
                
                axes_to_sum = tuple(ax for ax in range(loaded_array.ndim) if ax not in (i, j))
                counts = np.sum(loaded_array, axis=axes_to_sum)
                
                # ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
                # print(np.max(counts)) 
                # pcolormesh contourf
                
                levels = np.array([1e-3, 0.01, 0.1, 0.2, 0.5, 0.8])
                
                cms = ax.pcolormesh(x_edges, y_edges, counts.T, cmap='viridis', shading='auto', rasterized=True, vmin=vmin, vmax=vmax)
                
                smoothed_counts = sp.ndimage.gaussian_filter(counts.T, sigma=2)
                cnt = ax.contour(x_edges, y_edges, smoothed_counts, levels, colors='white', rasterized=True, linewidths=0.5,
                                    vmin=vmin, vmax=vmax)
                mpl.rcParams["text.usetex"] = False
                ax.clabel(cnt, levels, fontsize=10, fmt='%r') #, manual=True) #, fmt='%.1f')
                mpl.rcParams["text.usetex"] = True
            else:
                ax.axis('off')
            # if i == 4: i = 2
            # if j == 4: j = 2
    cbar_ax = fig.add_axes([0.92, 0.108, 0.02, 0.77])  # [left, bottom, width, height]
    fig.colorbar(cms, cax=cbar_ax, label='Fraction of the accretor stage')
    # cbar_ax.set_ylabel('Fraction of the accretor stage') #, va='bottom', ha='center')
    
    """ Lines on the BP diagram """
    for numdens in [0.1]:
        for v in [100e5]:
        # v = 30e5
            R_G = 2 * G * M_NS / v**2
            Mdot = np.pi * R_G**2 * m_p * numdens * v
            Pl = np.zeros(len(Bb))
            for k in range(len(Bb)):
                NS = Object(10**Bb[k], v, Mdot, case='B', Omega=None)
                Pl[k] = NS.P_EP(0)
            # print(Bb, np.log10(Pl))
            axes[2, 1].plot(Bb, np.log10(Pl), label='{:.0f}, {}'.format(v/1e5, numdens), color='white', ls=':')
    axes[2, 1].set_ylim([-2, 3])
    # axes[2, 1].legend()
    numdens = 0.1
    Mdot = (2 * G * M_NS)**2 * np.pi * m_p * numdens
    NS = Object(1, 1, Mdot, case='B', Omega=None)
    A = (NS.R_G(0) / NS.R_A(0))**(7/4)
    print("{:.2e}".format(A/(100*1e5)**5))
    axes[1, 0].plot(Vb, np.log10(A/(Vb*1e5)**5), color='white', ls='-.', lw=1)
    axes[1, 0].set_ylim([10, 15])
    
    axes[2, 1].text(14, 2, r'III', color='white', fontsize=30)
    axes[2, 0].text(400, 2, r'II', color='white', fontsize=30)
    axes[1, 0].text(400, 14, r'I', color='white', fontsize=30)
    
    """ Lines on the Bv diagram """
    # for numdens in [0.1, 0.3, 1]:
    #     # v = 30e5
    #     R_G = 2 * G * M_NS / v**2
    #     Mdot = np.pi * R_G**2 * m_p * numdens * v
    #     Pl = np.zeros(len(Bb))
    #     for k in range(len(Bb)):
    #         NS = Object(10**Bb[k], v, Mdot, case='B', Omega=None)
    #         vcrit[k] = NS.P_EP(0)
    #     # print(Bb, np.log10(Pl))
    #     axes[1, 0].plot(vcrit, Bb, label='{:.0f}, {}'.format(v/1e5, numdens), alpha=0.7) #color='white')
    # axes[1, 0].legend()
    
    if not os.path.exists(output_dir + 'figures/'):
        os.mkdir(output_dir + 'figures/')
    # fig.suptitle(title, fontsize=20)
    fig.savefig(output_dir + f'figures/triangle{case}.pdf', bbox_inches='tight')


# create_uniform(9_000_000, res_name='result2')
# create_uniform(1_000_000, res_name='result1')
# create_uniform(9_000_000, res_name='result3')
# for galaxy_type in ['simple', 'two_phase']:
#     for field in ['CF', 'ED']:
#         for case in ['A', 'B', 'C', 'D']:
#             plot_uniform(galaxy_type, field, case)

plot_uniform_article(case='A')
# plot_uniform_article(case='C')
plot_uniform(case='B', field='ED', galaxy_type='two_phase')

def ejector_propeller_condition():
    num = 1000
    B_array = 10**np.linspace(10, 15, num)
    # v_array = np.linspace(10, 500, num) * 1e5
    # R_G = 2 * G * M_NS / v_array**2
    # n = 0.1
    # Mdot_array = np.pi * R_G**2 * m_p * n * v_array
    P = np.zeros(len(B_array))
    PG = np.zeros(len(B_array))
    Pl = np.zeros(len(B_array))
    t_E = np.zeros(len(B_array))
    for n in [0.1, 1, 10]:
        for v in [10e5, 100e5, 300e5]:
            for i in range(len(B_array)):
            # v = v_array[-1]
            # Mdot = Mdot_array[-1]
                B = B_array[i]
                R_G = 2 * G * M_NS / v**2
                Mdot = np.pi * R_G**2 * m_p * n * v
                NS = Object(B, v, Mdot, case='A', Omega=None)
                P[i] = NS.P_EP(0)
                PG[i] = NS.P_EP_G(0)
                Pl[i] = NS.P_EP_l(0)
                t_E[i] = NS.t_ejector(0.01, P[i])
            
            plt.plot(B_array, t_E, label='v={:.0f}, n={:.1f}'.format(v/1e5, n), alpha=0.7)
    # plt.plot(B_array, P, alpha=0.5, color='C0')
    # plt.plot(B_array, Pl, alpha=0.5, color='C1', ls='--')
    # plt.plot(B_array, PG, alpha=0.5, color='C2', ls=':')
    plt.plot(B_array, galaxy_age + B_array * 0, color='black')

# fig, ax = plt.subplots()
# ax.set_yscale('log')
# ax.set_xscale('log')

# ejector_propeller_condition()
# plt.legend()

def merge_pdfs():
    merger = PdfMerger()
    folder_path = 'result/uniform/figures/'
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    pdf_files.sort()

    for filename in pdf_files:
        file_path = os.path.join(folder_path, filename)
        merger.append(file_path)
    
    output_path = os.path.join(folder_path, 'triangle.pdf')
    merger.write(output_path)
    merger.close()

# merge_pdfs()

def table():
    galaxy_type = ['simple', 'simple', 'two_phase', 'two_phase']
    field = ['CF', 'ED']*2
    print('  & ' + ' & '.join(field) + ' \\\\')
    N = 1_000_000
    num_parts = 4
    
    def split_array_into_five_parts(array):
        array = array.flatten()
        np.random.shuffle(array)
        length = len(array)
        part_size = length // num_parts
        parts = []
        for i in range(num_parts):
            start_index = i * part_size
            end_index = (i + 1) * part_size if i < num_parts else length
            parts.append(np.sum(array[start_index:end_index]))
        return np.array(parts)

    for case in ['A', 'B', 'C', 'D']:
        row = case
        for j in range(4):
            loaded_array = np.load(npy_dir+'{}_{}_{}_{}.npy'.format(galaxy_type[j], field[j], case, N))
            sum_array = split_array_into_five_parts(loaded_array) / (1_000_000 / num_parts) * 100
            if case == 'D':
                row += ' & ${:.4f}'.format(np.mean(sum_array))+' \\pm '+'{:.4f}$'.format(np.std(sum_array))
            else:
                row += ' & ${:.3f}'.format(np.mean(sum_array))+' \\pm '+'{:.3f}$'.format(np.std(sum_array))
        row += ' \\\\'
        print(row)
# table()
