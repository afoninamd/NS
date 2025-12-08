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

from main.constants import arr_size, galaxy_age, Gyr, Myr, year
full_path = '/home/afoninamd/Downloads/test_new_10k/'
# full_path = '/home/afoninamd/Downloads/realistic2/'
path = '/home/afoninamd/Downloads/'
res_name = 'realistic2/'
# path = '/home/afoninamd/Documents/project/pops/result/realistic/'
# res_name = ''
Rbins = np.linspace(0, 20, arr_size+1) # from the GC
rbins = np.linspace(0, 20, arr_size+1) # from the Sun better for 0 to 20
zbins = np.linspace(0, 5, arr_size+1)
fbins = 10**np.linspace(-15, -5, arr_size+1)
cbins = 10**np.linspace(-4, 6, arr_size+1)
vbins = np.linspace(0, 500, arr_size+1) * 1e5
Tbins = 10**np.linspace(5, 9, arr_size+1) # better from 5 to 8
tbins = np.linspace(0, galaxy_age, arr_size+1)

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
    os.makedirs('counts/', exist_ok=True)
    for galaxy_type in ['simple','two_phase']:#, 'two_phase']:
        for field in ['CF', 'ED']:#, 'ED']:
            for case in ['A', 'B', 'C', 'D']:
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
    os.makedirs('counts/', exist_ok=True)
    for galaxy_type in ['simple','two_phase']:#, 'two_phase']:
        for field in ['CF', 'ED']:#, 'ED']:
            for case in ['A', 'B', 'C', 'D']:
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
                    N = 10_000_000
                    norm = 300_000_000 // N
                    idx = 399 # 399 for 2000
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
                    # data = pd.read_feather(path + res_name + file_name + '.feather')
                    
                    data = pd.read_feather(full_path +'/' + file_name + '.feather')
                    # print(data.columns)
                    arr = tbins / Gyr #Rbins
                    # data_2 = np.array(data['r-2'])
                    # data_1 = np.array(data['r-1'])
                    # data_0 = np.array(data['r'])
                    data_2 = np.array(data['t-2'])
                    data_1 = np.array(data['t-1'])
                    data_0 = np.array(data['t'])
                    
                    alpha = 1
                    N = 300
                    norm = 300_000_000 // N
                    idx = 2 # 399 for 2000
                    
                    # arr = rbins
                    ax.plot(arr[1:], norm*data_0, 'k')
                    ax.plot(arr[1:], norm*data_1, 'C0')
                    ax.plot(arr[1:], norm*data_2, 'C1')
                    ax.set_yscale('log')
                    ax.set_xlim()
                    
                    file_name = '{}_{}_{}_erosita{}_std'.format(galaxy_type, field,
                                                    case, add_string)
                    data = pd.read_feather(path + res_name + file_name + '.feather')
                    # std_2 = np.array(data['r-2'])
                    # std_1 = np.array(data['r-1'])
                    # std_0 = np.array(data['r'])
                    print(data.columns)
                    std_2 = np.array(data['t-2'])
                    std_1 = np.array(data['t-1'])
                    std_0 = np.array(data['t'])
                    
                    ax.fill_between(arr[1:], y1= norm*(data_0-std_0), y2 = norm*(data_0), color='k', alpha=0.1)
                    ax.fill_between(arr[1:], y1= norm*(data_1-std_1), y2 = norm*(data_1), color='C0', alpha=0.1)
                    ax.fill_between(arr[1:], y1= norm*(data_2-std_2), y2 = norm*(data_2), color='C1', alpha=0.1)
                    
                    ax.fill_between(arr[1:], y2= norm*(data_0+std_0), y1 = norm*(data_0), color='k', alpha=0.1)
                    ax.fill_between(arr[1:], y2= norm*(data_1+std_1), y1 = norm*(data_1), color='C0', alpha=0.1)
                    ax.fill_between(arr[1:], y2= norm*(data_2+std_2), y1 = norm*(data_2), color='C1', alpha=0.1)


def rebin(nrebin=40): # new number of bins
    plt.rcParams['font.size'] = 16
    plt.rcParams['text.usetex'] = True
    os.makedirs('counts_rebinned/', exist_ok=True)
     
    fig, axes = plt.subplots(figsize=(12 , 10), nrows=2, ncols=2)
    
    axes[0,0].set_title(r'One phase ISM, constant field')
    axes[0,1].set_title(r'One phase ISM, exponentially decaying field')
    axes[1,0].set_title(r'Two phase ISM, constant field')
    axes[1,1].set_title(r'Two phase ISM, exponentially decaying field')
                
    for i in range(2):
        galaxy_type = ['simple','two_phase'][i]
        lines = ['A', 'B', 'C', 'D'] # for the table
        for j in range(2):
            field = ['CF', 'ED'][j]
            lss = ['-', '--', '-.']
            cases = ['A', 'B', 'C', 'D']
            ax = axes[i, j]
            axes[i, 0].set_ylabel(r'$N($CR$ > $CR$_0)$', fontsize=20)
            axes[1, j].set_xlabel(r'CR$_0$, cts s$^{-1}$', fontsize=20)
            ax.vlines(x=1e-2, ymin=1e-2, ymax=1e7, color='grey', alpha=0.2, ls='-', lw=2)
            # ax.set_title('galaxy is {}, field {}'.format(galaxy_type, field))
            ax.tick_params(axis='both', labelsize=16)
            for k in range(3):
                ls = lss[k]
                case = cases[k]
                file_name = '{}_{}_{}_erosita{}_std'.format(galaxy_type, field,
                                                case, '')
                data_std = pd.read_feather(path + res_name + file_name + '.feather')
                data_c1_std = np.array(data_std['c1'])
                std_c1 = data_c1_std #np.cumsum(data_c1_std[::-1])[::-1]
                file_name = '{}_{}_{}_erosita{}_sum'.format(galaxy_type, field,
                                                case, '')
                data = pd.read_feather(path + res_name + file_name + '.feather')
                
                bins = cbins #arr[1:]
                
                data_c1 = np.array(data['c1'])
                data_c1 = np.cumsum(data_c1[::-1])[::-1]
                
                n_in_bin = len(data_c1)//nrebin
                data_c1_temp = np.zeros(nrebin)
                std_c1_temp = np.zeros(nrebin)
                
                std_c1_squared = std_c1**2
                for itemp in range(n_in_bin):
                    data_c1_temp += data_c1[itemp::n_in_bin]
                    std_c1_temp += std_c1_squared[itemp::n_in_bin]
                data_c1 = data_c1_temp / n_in_bin
                bins = bins[::n_in_bin]
                bins = bins[1::]
                
                std_c1 = std_c1_temp**0.5
                
                alpha = 0.5
                N = 10_000_000
                norm = 300_000_000 // N
                idx = nrebin // 5 - 1 # 400 for 2000
                color = ['C3', 'C2', 'C0', 'C3'][k]
                # ax.text(1.2e-2, 1e5, '{:.0f} NSs with CR > 1e-2'.format(norm*data_c1[idx]), color='C0', fontsize=14)
                idx1 = nrebin // 10 - 1
                if case == 'C':
                    diagonal = bins**(-1) * data_c1[idx1] * bins[idx1]
                    ax.plot(bins[:idx+1], norm*diagonal[:idx+1], ls='--', color='grey')
                if (case == 'A' and (i == 0 or j == 0)) or (case == 'B' and i == 1 and j == 1):# or (case == 'C' and k == 3):
                    diagonal = bins**(-3/2) * data_c1[19] * bins[19]**(3/2)
                    # print(norm*data_c1[19], bins[19])
                    ax.plot(bins[15:], norm*diagonal[15:], ls='--', color='grey')
                    
             
                print(bins[7])
                
                def format_num(n):
                    # numbers < 100 -> 1 significant digit
                    # numbers >=100 -> 2 significant digits
                    sig = 1 if n < 100 else 2
                    return n#float(f"{n:.{sig}g}")

                lines[k] = lines[k] + ' & ${:.0f} \pm {:.0f}$'.format(format_num(norm*data_c1[7]), format_num(norm*std_c1[7]))
                
                ax.plot(bins, norm*data_c1, color=color, alpha=alpha, ls=ls, lw=1, markersize=3, marker='o', label=case)
                
                # ax.plot(arr[1:], norm*data_c0, color=color, alpha=alpha)
                # ax.plot(bins, norm*data_c1, color=color, alpha=alpha, ls=ls, lw=1, markersize=3, marker='o', label=case)
                # ax.plot([bins[idx]], [norm*data_c1[idx]], color=color, markersize=5, marker='o', label=case)
                
                ax.fill_between(bins, y1 = norm*(data_c1-std_c1), y2 = norm*(data_c1), color=color, alpha=0.1)
                ax.fill_between(bins, y2 = norm*(data_c1+std_c1), y1 = norm*(data_c1), color=color, alpha=0.1)
                    
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.set_xlim([1e-3, 1e1])
                ax.set_ylim([1e-1, 1e5])
            ax.legend(title=r'Propeller model')
        for line in lines:
            print(line + ' \\\\')
        
    fig.tight_layout()
        # break
        # print(len(std_c1[std_c1!=0]))
    plt.subplots_adjust(wspace=0.4, hspace=0.4, left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.tight_layout()

    fig.savefig('counts_rebinned/counts.pdf', format='pdf')

# rebin(nrebin=40)

def plotrrebin(nrebin=100,field = 'ED',  case = 'B', galaxy_type = 'two_phase'): # new number of bins
    plt.rcParams['font.size'] = 16
    plt.rcParams['text.usetex'] = True
    os.makedirs('counts_rebinned/', exist_ok=True)
    fig, axes = plt.subplots(figsize=(12 , 15), nrows=3, ncols=2)
    # fig, ax = plt.subplots()
    axes = axes.flatten()
    names = ['r', 'R', 'z', 'v', 't', 'T']
    
    dim = [', kpc', ', kpc', ', kpc', ', km s$^{-1}$', ', Gyr', ', K']
    dim1 = [', kpc$^{-1}$', ', kpc$^{-1}$', ', kpc$^{-1}$', ', (km s$^{-1}$)$^{-1}$', ', Gyr$^{-1}$', ', K$^{-1}$']
    
    N = 10_000_000
    norm = 300_000_000 // N
    
    for i in range(6):
        # if i != 3:
        #     continue
        ax = axes[i]
        name = names[i]
        if name == 'T':
            ax.set_xscale('log')
        
        # if i == 3:
        #     ax.set_xlim([0, 100])
        # elif i == 2:
        #     ax.set_xlim([0, 0.4])
        # elif i == 1:
        #     ax.set_xlim([0, 10])
        # else:
        #     ax.set_xlim([0, 1.5])
        bins = [rbins, Rbins, zbins, vbins/1e5, tbins/Gyr, Tbins][i]
        file_name = '{}_{}_{}_erosita{}_std'.format(galaxy_type, field, case, '')
        data_std = pd.read_feather(full_path + file_name + '.feather') #path + res_name
        
        file_name = '{}_{}_{}_erosita{}_sum'.format(galaxy_type, field,
                                        case, '')
        data = pd.read_feather(full_path + file_name + '.feather')
        
        # x_arr = (bins[1:] + bins[:1]) / 2
        
        
        ax.tick_params(axis='both', labelsize=16)
        ax.set_yscale('log')
        
        for j in range(3):
            cts_num = ['', '-1', '-2'][j]
            try:
                std_0 = np.array(data_std[name+cts_num])
            except:
                continue
            data_0 = np.array(data[name+cts_num])
            
            data_0 = data_0 * bins[-1] / len(bins) # new dim
            std_0 = std_0 * bins[-1] / len(bins) # new dim
            
            color = ['k', 'C0', 'C1'][j]
            
            n_in_bin = len(data_0)//nrebin
            data_0_temp = np.zeros(nrebin)
            std_0_temp = np.zeros(nrebin)
            
            std_0_squared = std_0**2
            for itemp in range(n_in_bin):
                data_0_temp += data_0[itemp::n_in_bin]
                std_0_temp += std_0_squared[itemp::n_in_bin]
            
            data_0 = data_0_temp / n_in_bin
            x_arr = bins[::n_in_bin]
            x_arr = (x_arr[1:] + x_arr[:-1]) / 2
            
            std_0 = std_0_temp**0.5
            
            
            # reg = 1000 # from kpc to pc
            # if i != 3:
                
                
            # bin_width = bins[-1] / (len(bins) / n_in_bin)
            # data_0 = data_0 / np.sum(data_0 * bin_width) * norm
            # std_0 = std_0 / np.sum(std_0 * bin_width) * norm
            
            # data_0 = moving_average(data_0, window_size=2)
            # std_0 = moving_average(std_0, window_size=2)
            
            ax.plot(x_arr, norm*data_0, color)
            
            
            """ The 99% line from the start """
            if False: # name == 'z' and cts_num == '-2':
                for value_a in [0.5, 0.99]:
                    y = norm * data_0
                    x = x_arr
                    cum_int = sp.integrate.cumulative_trapezoid(y, x, initial=0)
                    total = cum_int[-1]
                    threshold_value = value_a * total
                    idx_99 = np.searchsorted(cum_int, threshold_value)
                    x_99 = x[idx_99]
                    ax.axvline(x_99, linestyle=':', label='99% integral', color=color)
                    print('{}_{}_{} z for {}: {:.3f}'.format(galaxy_type, field, case, value_a, x_99))
            
            if name == 't' and cts_num == '':
                y = norm * data_0
                x = x_arr
                cum_int = sp.integrate.cumulative_trapezoid(y, x, initial=0)
                total = cum_int[-1]
                threshold_value = 12.8
                y1 = y[x>threshold_value]
                x1 = x[x>threshold_value]
                cum_int_last_Gyr = sp.integrate.cumulative_trapezoid(y1, x1, initial=0) 
                print('{}_{}_{}: {:.0f} %'.format(galaxy_type, field, case, 100*cum_int_last_Gyr[-1]/total))
                
            if False: #name == 'T' and cts_num == '-2':
                imax = np.argmax(y)
                x_centers = 0.5 * (x_arr[:-1] + x_arr[1:])
                x_at_max = x_centers[imax]
                
                print('{}_{}_{} T of maximum: {:.3e}'.format(galaxy_type, field, case, x_at_max))
                      
            
            # if cts_num == '-2':
            #     print(case, field, x_99)
            
            """ 3_sigma-interval """
            # num_of_sigma = 1
            # x = x_arr
            # pdf = y / np.trapz(y, x)
            # mu = np.trapz(x * pdf, x)
            # sigma = np.sqrt(np.trapz((x - mu)**2 * pdf, x))
            # low_3s = mu - num_of_sigma * sigma
            # high_3s = mu + num_of_sigma * sigma
            # plt.axvline(low_3s,  color=color, linestyle='--', label='μ − {}σ'.format(num_of_sigma))
            # plt.axvline(high_3s, color=color, linestyle='--', label='μ + {}σ'.format(num_of_sigma))
            
        
            ax.fill_between(x_arr, y1=norm*(data_0-std_0), y2=norm*(data_0), color=color, alpha=0.1)
            ax.fill_between(x_arr, y2=norm*(data_0+std_0), y1=norm*(data_0), color=color, alpha=0.1)
        
        ax.set_xlabel("$"+name+"$"+dim[i], fontsize=14)
        ax.set_ylabel('$N$' + dim1[i], fontsize=14)

        # ax.set_ylim([1., 3e5])
        
    # print(np.sum(np.array(data['r']))*norm)
    # print(np.sum(np.array(data['r-1']))*norm)
    # print(np.sum(np.array(data['r-2']))*norm)
    
    fig.tight_layout()
    # break
    # print(len(std_c1[std_c1!=0]))
    plt.subplots_adjust(wspace=0.4, hspace=0.4, left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.tight_layout()
    
    fig.suptitle('{}_{}_{}'.format(galaxy_type, field, case))
    fig.savefig('counts_rebinned/{}_{}_{}.pdf'.format(galaxy_type, field, case), format='pdf')

""" HERE!!!! """
# plotr()
plt.close()
for galaxy_type in ['simple', 'two_phase']:
    for case in ['A', 'B', 'C']:
        for field in ['CF', 'ED']:
            pass
            plotrrebin(case=case, field=field, galaxy_type=galaxy_type)
            # break

    # for i in range(2):
    #     galaxy_type = ['simple','two_phase'][i]
    #     for j in range(2):
    #         field = ['CF', 'ED'][j]
    #         lss = ['-', '--', '-.']
    #         cases = ['A', 'B', 'C', 'D']
    #         ax = axes[i, j]
    #         axes[i, 0].set_ylabel(r'$N(CR > CR_0)$', fontsize=20)
    #         axes[1, j].set_xlabel(r'$CR_0$, cts s$^{-1}$', fontsize=20)
    #         ax.vlines(x=1e-2, ymin=1e-2, ymax=1e7, color='black', alpha=0.5, ls='--', lw=0.5)
    #         # ax.set_title('galaxy is {}, field {}'.format(galaxy_type, field))
    #         ax.tick_params(axis='both', labelsize=16)
    #         for k in range(3):
    #             ls = lss[k]
    #             case = cases[k]
    #             file_name = '{}_{}_{}_erosita{}_std'.format(galaxy_type, field,
    #                                             case, '')
    #             data_std = pd.read_feather(path + res_name + file_name + '.feather')
    #             data_c1_std = np.array(data_std['c1'])
    #             std_c1 = data_c1_std #np.cumsum(data_c1_std[::-1])[::-1]
    #             file_name = '{}_{}_{}_erosita{}_sum'.format(galaxy_type, field,
    #                                             case, '')
    #             data = pd.read_feather(path + res_name + file_name + '.feather')
                
    #             bins = cbins #arr[1:]
                
    #             data_c1 = np.array(data['r'])
    #             # data_c1 = np.cumsum(data_c1[::-1])[::-1]
                
    #             n_in_bin = len(data_c1)//nrebin
    #             data_c1_temp = np.zeros(nrebin)
    #             std_c1_temp = np.zeros(nrebin)
                
    #             std_c1_squared = std_c1**2
    #             for itemp in range(n_in_bin):
    #                 data_c1_temp += data_c1[itemp::n_in_bin]
    #                 std_c1_temp += std_c1_squared[itemp::n_in_bin]
    #             data_c1 = data_c1_temp / n_in_bin
    #             bins = bins[::n_in_bin]
    #             bins = bins[1::]
                
    #             std_c1 = std_c1_temp**0.5
                
    #             alpha = 0.5
    #             N = 10_000_000
    #             norm = 300_000_000 // N
    #             idx = nrebin // 5 - 1 # 400 for 2000
    #             color = ['C3', 'C2', 'C0', 'C3'][k]
    #             # ax.text(1.2e-2, 1e5, '{:.0f} NSs with CR > 1e-2'.format(norm*data_c1[idx]), color='C0', fontsize=14)
                
    #             # ax.plot(arr[1:], norm*data_c0, color=color, alpha=alpha)
    #             ax.plot(bins, norm*data_c1, color=color, alpha=alpha, ls=ls, lw=1, markersize=3, marker='o', label=case)
    #             # ax.plot([bins[idx]], [norm*data_c1[idx]], color=color, markersize=5, marker='o', label=case)
                
    #             ax.fill_between(bins, y1 = norm*(data_c1-std_c1), y2 = norm*(data_c1), color=color, alpha=0.1)
    #             ax.fill_between(bins, y2 = norm*(data_c1+std_c1), y1 = norm*(data_c1), color=color, alpha=0.1)
                    
    #             ax.set_yscale('log')
    #             ax.legend(title=r'Propeller model')
    #             ax.set_xscale('log')
    #             ax.set_xlim([1e-3, 1e1])
    #             ax.set_ylim([1e-1, 1e5])
    #             fig.tight_layout()
    #             # break
    #             # print(len(std_c1[std_c1!=0]))
    #             plt.subplots_adjust(wspace=0.4, hspace=0.4, left=0.1, right=0.9, top=0.9, bottom=0.1)
    #             plt.tight_layout()
    
    #             fig.savefig('counts_rebinned/counts_{}_{}_{}.pdf'.format(galaxy_type, field, case), format='pdf')


# data_c1 = std_c1 = np.array([0,0,1,4])
# nrebin = 1
# n_in_bin = 4
# data_c1_temp = np.zexros(nrebin)
# std_c1_temp = np.zeros(nrebin)
# for itemp in range(n_in_bin):
#     data_c1_temp += data_c1[itemp::n_in_bin]
#     std_c1_temp += (std_c1[itemp::n_in_bin])**2
# print(data_c1_temp, std_c1_temp**0.5)

# pdfs1()
# merge_pdfs()
# plotrrebin(100)

# rebin(10)
# rebin(100)
# rebin(40)
# plotrrebin(40)
# plotrrebin(20)

# a = np.array([0,1,2,4])
# print(a[::2])
# print(a[::2]+a[])


def table_stages_old():
    df = pd.read_csv('/home/afoninamd/Downloads/realistic_fixed/all.txt', sep='\s+', header=None)
    df_p = df[:16]
    df_m = df[16:32]
    df_pr = df[32:48]
    df_mr = df[48:]
    # df_m = np.zeros((10,16))
    # df_p = pd.read_csv('/home/afoninamd/Downloads/realistic_fixed/all_pulsar.txt', sep='\s+', header=None)
    # df_m = pd.read_csv('/home/afoninamd/Downloads/realistic_fixed/all_magnetar.txt', sep='\s+', header=None)
    # df_pr = pd.read_csv('/home/afoninamd/Downloads/realistic_fixed/all_pulsar_roman.txt', sep='\s+', header=None)
    # df_mr = pd.read_csv('/home/afoninamd/Downloads/realistic_fixed/all_magnetar_roman.txt', sep='\s+', header=None)
    
    norm = (np.array(df_p[1])+np.array(df_m[1]))[0]
    norm_p = np.array(df_p[1])[0]
    norm_m = np.array(df_m[1])[0]
    
    ejector = (np.array(df_p[2]) + np.array(df_m[2])) / norm
    propeller = (np.array(df_p[3]) + np.array(df_m[3])) / norm
    accretor = (np.array(df_p[4]) + np.array(df_m[4])) / norm
    georotator = (np.array(df_p[5]) + np.array(df_m[5])) / norm
    
    
    s_ejector = (np.array(df_p[6])**2*norm_p**2 + np.array(df_m[6])**2*norm_m**2)**0.5 / norm
    s_propeller = (np.array(df_p[7])**2*norm_p**2 + np.array(df_m[7])**2*norm_m**2)**0.5 / norm
    s_accretor = (np.array(df_p[8])**2*norm_p**2 + np.array(df_m[8])**2*norm_m**2)**0.5 / norm
    s_georotator = (np.array(df_p[9])**2*norm_p**2 + np.array(df_m[9])**2*norm_m**2)**0.5 / norm
    
    for i in range(len(ejector)):
        all_norm = 100
        print('{} & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ & ${:.2e} \pm {:.2e}$ & ${:.2e} \pm {:.2e}$ \\\\'.format(df_p[0][i], ejector[i]*all_norm, s_ejector[i]*all_norm, propeller[i]*all_norm, s_propeller[i]*all_norm, accretor[i]*all_norm, s_accretor[i]*all_norm, georotator[i]*all_norm, s_georotator[i]*all_norm))
        

# table_stages_old()
# print(4736752+526368)
# print(4734626+525733)


def table_stages():
    df = pd.read_csv('/home/afoninamd/Documents/NS/project/pops/result/all.txt', sep='\s+', header=None)
    df_p = df[:16]
    df_m = np.zeros((10,16))
    # df_pr = df[16:32]
    # df_mr = np.zeros(16)
    
    # df_p = pd.read_csv('/home/afoninamd/Downloads/realistic_fixed/all_pulsar.txt', sep='\s+', header=None)
    # df_m = pd.read_csv('/home/afoninamd/Downloads/realistic_fixed/all_magnetar.txt', sep='\s+', header=None)
    # df_pr = pd.read_csv('/home/afoninamd/Downloads/realistic_fixed/all_pulsar_roman.txt', sep='\s+', header=None)
    # df_mr = pd.read_csv('/home/afoninamd/Downloads/realistic_fixed/all_magnetar_roman.txt', sep='\s+', header=None)
    
    norm = (np.array(df_p[1])+np.array(df_m[1]))[0]
    norm_p = np.array(df_p[1])[0]
    norm_m = 0 #np.array(df_m[1])[0]
    
    ejector = (np.array(df_p[2]) + np.array(df_m[2])) / norm
    propeller = (np.array(df_p[3]) + np.array(df_m[3])) / norm
    accretor = (np.array(df_p[4]) + np.array(df_m[4])) / norm
    georotator = (np.array(df_p[5]) + np.array(df_m[5])) / norm
    
    s_ejector = (np.array(df_p[6])**2*norm_p**2 + np.array(df_m[6])**2*norm_m**2)**0.5 / norm
    s_propeller = (np.array(df_p[7])**2*norm_p**2 + np.array(df_m[7])**2*norm_m**2)**0.5 / norm
    s_accretor = (np.array(df_p[8])**2*norm_p**2 + np.array(df_m[8])**2*norm_m**2)**0.5 / norm
    s_georotator = (np.array(df_p[9])**2*norm_p**2 + np.array(df_m[9])**2*norm_m**2)**0.5 / norm
    
    for i in range(len(ejector)):
        all_norm = 100
        print('{} & ${:.2f} \pm {:.2f}$ & ${:.2f} \pm {:.2f}$ & ${:.2e} \pm {:.2e}$ & ${:.2e} \pm {:.2e}$ \\\\'.format(df_p[0][i], ejector[i]*all_norm, s_ejector[i]*all_norm, propeller[i]*all_norm, s_propeller[i]*all_norm, accretor[i]*all_norm, s_accretor[i]*all_norm, georotator[i]*all_norm, s_georotator[i]*all_norm))
        
# table_stages()


def test_distr():
    df = pd.read_csv('/home/afoninamd/Downloads/realistic_fixed_distribs/distribution_magnetar_1000000.csv', sep=';')
    Vxpec = np.array(df['Vxpec'])/1e5
    Vypec = np.array(df['Vypec'])/1e5
    Vzpec = np.array(df['Vzpec'])/1e5
    plt.hist(Vxpec, alpha=0.5)
    plt.hist(Vypec, alpha=0.5)
    plt.hist(Vzpec, alpha=0.5)
    plt.hist((Vxpec**2+Vypec**2+Vzpec**2)**0.5)