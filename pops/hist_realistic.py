#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 23:23:05 2025

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
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main.constants import arr_size

path = '/home/afoninamd/Downloads/'
# res_name = 'realistic1/'
# arr_size = 100
# Rbins = np.linspace(0, 20, 101) # from the GC
# rbins = np.linspace(0, 10, 101) # from the Sun
# zbins = np.linspace(0, 5, 101)
# fbins = 10**np.linspace(-15, -5, 101)
# cbins = 10**np.linspace(-4, 6, 101)
# vbins = np.linspace(0, 500, 101) * 1e5
# Tbins = 10**np.linspace(4, 9, 101)


path = '/home/afoninamd/Documents/project/pops/result/realistic/'
res_name = ''
Rbins = np.linspace(0, 20, arr_size+1) # from the GC
rbins = np.linspace(0, 20, arr_size+1) # from the Sun better for 0 to 20
zbins = np.linspace(0, 5, arr_size+1)
fbins = 10**np.linspace(-15, -5, arr_size+1)
cbins = 10**np.linspace(-4, 6, arr_size+1)
vbins = np.linspace(0, 500, arr_size+1) * 1e5
Tbins = 10**np.linspace(5, 9, arr_size+1) # better from 5 to 8

# for galaxy_type in ['simple']:#, 'two_phase']:
#     for field in ['CF']:#, 'ED']:
#         for case in ['A']:#, 'B', 'C', 'D']:
#             for add_string in ['_roman']:#, '_roman']:
#                 data_arr = np.zeros(arr_size)
#                 for rand_i in range(10):
                    
#                     file_name = '{}_{}_{}_{}_erosita{}'.format(galaxy_type, field,
#                                                     case, 'pulsar', add_string, rand_i)
#                     df = pd.read_feather(path + res_name + file_name + '.feather')
#                     file_name = '{}_{}_{}_{}_erosita{}'.format(galaxy_type, field,
#                                                     case, 'magnetar', add_string, rand_i)
#                     data = df+pd.read_feather(path + res_name + file_name + '.feather')
#                     arr = cbins
#                     data_arr += data['c0']
#                     plt.plot((arr[1:]+arr[:-1])/2, data['c0'], alpha=0.3)
#                 plt.plot((arr[1:]+arr[:-1])/2, data_arr, 'red')
#                     # plt.yscale('log')
#                     # plt.xscale('log')

# print(cbins[20])

def pdfs():
    for galaxy_type in ['simple','two_phase']:#, 'two_phase']:
        for field in ['CF', 'ED']:#, 'ED']:
            for case in ['A', 'B', 'C']: #, 'D']:
                fig, ax = plt.subplots()
                ax.vlines(x=1e-2, ymin=1e-2, ymax=1e7, color='black', alpha=0.5, ls='--')
                
                for add_string in ['', '_roman']:
                    file_name = '{}_{}_{}_{}_erosita{}'.format(galaxy_type, field,
                                                    case, 'pulsar', add_string)
                    df = pd.read_feather(path + res_name + file_name + '.feather')
                    file_name = '{}_{}_{}_{}_erosita{}'.format(galaxy_type, field,
                                                    case, 'magnetar', add_string)
                    data = df+pd.read_feather(path + res_name + file_name + '.feather')
                    
                    arr = cbins
                    data_c1 = np.array(data['c1'])
                    data_c1 = np.cumsum(data_c1[::-1])[::-1]
                    
                    data_c0 = np.array(data['c0'])
                    data_c0 = np.cumsum(data_c0[::-1])[::-1]
                    
                    alpha = 1
                    norm = 300
                    if add_string == '':
                        ax.plot(arr[1:], norm*data_c0, color='C0', alpha=alpha, label='erosita')
                        ax.plot(arr[1:], norm*data_c1, color='C0', alpha=alpha, ls=':')
                        ax.plot([arr[20]], [norm*data_c1[19]], color='C0', markersize=5, marker='o')
                        ax.text(1.2e-2, 1e5, '{:.0f} NSs with CR > 1e-2'.format(norm*data_c1[20]), color='C0', fontsize=14)
                    else:
                        ax.plot(arr[1:], norm*data_c0, color='C1', alpha=alpha, label='erosita and roman')
                        ax.plot(arr[1:], norm*data_c1, color='C1', alpha=alpha, ls=':')
                        ax.plot([arr[20]], [norm*data_c1[19]], color='C1', markersize=5, marker='o')
                        ax.text(1.2e-2, 1e4, '{:.0f} NSs with CR > 1e-2 \nobservable by roman'.format(norm*data_c1[20]), color='C1', fontsize=14)
                    
                ax.set_yscale('log')
                ax.legend()
                ax.set_xscale('log')
                ax.set_xlim([1e-4, 10])
                ax.set_ylim([1e-1, 1e7])
                ax.set_ylabel(r'$N(CR > CR_0)$')
                ax.set_xlabel(r'$CR_0$, cts s$^{-1}$')
                fig.tight_layout()
                
                ax.set_title('galaxy is {}, field {}, case {}'.format(galaxy_type, field, case))
                fig.savefig('counts/counts_{}_{}_{}.pdf'.format(galaxy_type, field, case), format='pdf')



# pdfs()
# 
def merge_pdfs():
    merger = PdfMerger()
    folder_path = 'counts/'
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    pdf_files.sort()

    for filename in pdf_files:
        file_path = os.path.join(folder_path, filename)
        merger.append(file_path)
    
    output_path = os.path.join(folder_path, 'counts1.pdf')
    merger.write(output_path)
    merger.close()


# bims = [0, 1, 2, 3]
# data_c0 = [50, 40, 10]

# print(np.cumsum(data_c0[::-1])[::-1])

def readtxt():
    file_name = '/home/afoninamd/Downloads/realistic0/all.txt'
    data = pd.read_csv(file_name, delimiter='\t', header=None)
    # row = '{}_{}_{}_{}{}'.format(galaxy_type, field, case, star_type, addstring)
    string_array = data[0]
    w = np.array(data[1])
    wE = np.array(data[2])
    wP = np.array(data[3])
    wA = np.array(data[4])
    wG = np.array(data[5])
    n = len(wE) // 2
    # print(wE[:n]/wE[n:])
    arr0 = wE + wP + wA + wG
    
    for w1 in [arr0, wE, wP, wA, wG]:
        w1[:n][w1[:n]==0] = 1
        # arr = w1[n:]/w1[:n] * 100
        arr = w1[n:]/arr0[n:] / (w1[:n]/arr0[:n])
        
        print(np.mean(arr), np.std(arr))
        
    
    

# readtxt()


def pdfs1():
    for galaxy_type in ['simple','two_phase']:#, 'two_phase']:
        for field in ['CF', 'ED']:#, 'ED']:
            for case in ['A', 'B', 'C']: #, 'D']:
                fig, ax = plt.subplots()
                ax.vlines(x=1e-2, ymin=1e-2, ymax=1e7, color='black', alpha=0.5, ls='--')
                
                for add_string in ['', '_roman']:
                    file_name = '{}_{}_{}_erosita{}_std'.format(galaxy_type, field,
                                                    case, add_string)
                    
                    print(path + res_name + file_name + '.feather')
                    
                    data_std = pd.read_feather(path + res_name + file_name + '.feather')
                    data_c1_std = data_std['c1']
                    std_c1 = data_c1_std #np.cumsum(data_c1_std[::-1])[::-1]
                    
                    file_name = '{}_{}_{}_erosita{}_sum'.format(galaxy_type, field,
                                                    case, add_string)
                    data = pd.read_feather(path + res_name + file_name + '.feather')
                    
                    arr = cbins
                    data_c1 = np.array(data['c1'])
                    data_c1 = np.cumsum(data_c1[::-1])[::-1]
                    
                    data_c0 = np.array(data['c0'])
                    data_c0 = np.cumsum(data_c0[::-1])[::-1]
                    
                    alpha = 1
                    N = 300
                    norm = 300_000_000 // N
                    idx = 2 # 399 for 2000
                    if add_string == '':
                        color = 'C0'
                        label = 'erosita'
                        ax.text(1.2e-2, 1e5, '{:.0f} NSs with CR > 1e-2'.format(norm*data_c1[idx]), color='C0', fontsize=14)
                    else:
                        color = 'C1'
                        label = 'erosita and roman'
                        ax.text(1.2e-2, 1e4, '{:.0f} NSs with CR > 1e-2 \nobservable by roman'.format(norm*data_c1[idx]), color='C1', fontsize=14)
                    
                    ax.plot(arr[1:], norm*data_c0, color=color, alpha=alpha, label=label)
                    ax.plot(arr[1:], norm*data_c1, color=color, alpha=alpha, ls=':')
                    ax.plot([arr[idx+1]], [norm*data_c1[idx]], color=color, markersize=5, marker='o')
                    
                    ax.fill_between(arr[1:], y1= norm*(data_c1-std_c1), y2 = norm*(data_c1), color=color, alpha=0.1)
                    ax.fill_between(arr[1:], y2 = norm*(data_c1+std_c1), y1 = norm*(data_c1), color=color, alpha=0.1)
                    
                ax.set_yscale('log')
                ax.legend()
                ax.set_xscale('log')
                ax.set_xlim([1e-4, 10e5])
                ax.set_ylim([1e-1, 1e7])
                ax.set_ylabel(r'$N(CR > CR_0)$')
                ax.set_xlabel(r'$CR_0$, cts s$^{-1}$')
                fig.tight_layout()
                
                ax.set_title('galaxy is {}, field {}, case {}'.format(galaxy_type, field, case))
                # break
                # print(len(std_c1[std_c1!=0]))
                fig.savefig('counts/counts_{}_{}_{}.pdf'.format(galaxy_type, field, case), format='pdf')



def plotr():
    for galaxy_type in ['simple','two_phase']:#, 'two_phase']:
        for field in ['CF', 'ED']:#, 'ED']:
            for case in ['A', 'B', 'C']: #, 'D']:
                fig, ax = plt.subplots()
                
                for add_string in ['']:#, '_roman']:
                    
                    file_name = '{}_{}_{}_erosita{}_sum'.format(galaxy_type, field,
                                                    case, add_string)
                    data = pd.read_feather(path + res_name + file_name + '.feather')
                    
                    arr = Rbins
                    data_2 = np.array(data['r-2'])
                    data_1 = np.array(data['r-1'])
                    data_0 = np.array(data['r'])
                    
                    alpha = 1
                    N = 300
                    norm = 300_000_000 // N
                    idx = 2 # 399 for 2000
                    
                    arr = rbins
                    ax.plot(arr[1:], norm*data_0, 'k')
                    ax.plot(arr[1:], norm*data_1, 'C0')
                    ax.plot(arr[1:], norm*data_2, 'C1')
                    ax.set_yscale('log')
                    ax.set_xlim()
                    
                    
                    file_name = '{}_{}_{}_erosita{}_std'.format(galaxy_type, field,
                                                    case, add_string)
                    data = pd.read_feather(path + res_name + file_name + '.feather')
                    std_2 = np.array(data['r-2'])
                    std_1 = np.array(data['r-1'])
                    std_0 = np.array(data['r'])
                    
                    ax.fill_between(arr[1:], y1= norm*(data_0-std_0), y2 = norm*(data_0), color='k', alpha=0.1)
                    ax.fill_between(arr[1:], y1= norm*(data_1-std_1), y2 = norm*(data_1), color='C0', alpha=0.1)
                    ax.fill_between(arr[1:], y1= norm*(data_2-std_2), y2 = norm*(data_2), color='C1', alpha=0.1)
                    
                    ax.fill_between(arr[1:], y2= norm*(data_0+std_0), y1 = norm*(data_0), color='k', alpha=0.1)
                    ax.fill_between(arr[1:], y2= norm*(data_1+std_1), y1 = norm*(data_1), color='C0', alpha=0.1)
                    ax.fill_between(arr[1:], y2= norm*(data_2+std_2), y1 = norm*(data_2), color='C1', alpha=0.1)

# plotr()