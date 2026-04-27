#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:32:56 2026

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
import glob
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main.constants import arr_size, galaxy_age, Gyr, Myr, year, G, m_p, M_NS, R_NS
from main.evolution import j_if_disc


"""
Table 1 from  arXiv:2601.10393 (astro-ph)
Long Period Transients (LPTs): a comprehensive review
"""

lpt_name = ['GCRT J1745−3009', 'GLEAM-X J1627−5235', 'GPM J1839−10', 'ASKAP J1935+2148',
            'CHIME J0630+25', 'ASKAP/DART J1832−0911', 'ILT J1101+5521',
            'GLEAM−X J0704−36', 'ASKAP J1839−0756', 'ASKAP J1448−6856', 'CHIME/ILT J1634+44',
            'ASKAP J1755−2527', 'AR Sco', 'eRASSU J1912−4410']

lpt_l_deg = np.array([358.8911, 332.4646, 22.1526, 57.1901, 187.9709, 22.6406, 150.4551, 247.8955,
             24.5450, 313.1644, 70.1692, 4.116664, 0.11599, 353.3432])

lpt_b_deg = np.array([-0.5409, -2.6009, -2.0629, 0.7453, 7.0606, -0.0839, 55.5199,
             -13.6352, -1.0557, -8.4338, 42.5754, -0.122707, 353.52, -22.0613])

is_NS = np.array([0.5, 1, 1, 1,
            1, 0, 0,
            0, 1, 0.3, 0,
            1, 0, 0])

for i in range(len(lpt_name)):
    if is_NS[i] == 1:
        print(lpt_name[i])

# lpt-b_deg_real = lpt_b_deg[is_NS==1]

""" abs(lpt_b_deg) > 13 -> WD, 0.5 - there is some evidence that it is a WD"""

def rebin_2d(arr, factor_x, factor_y):
    sx, sy = arr.shape
    return arr.reshape(sx//factor_x, factor_x, sy//factor_y, factor_y).sum(axis=(1,3))

def rebin_edges(edges, factor):
    return edges[::factor]

def MdotMap(galaxy_type='simple', field='CF', case='A'):
    name2d = '{}_{}_{}'.format(galaxy_type, field, case)
    output_dir = '/home/afoninamd/Documents/NS/project/pops/result/disks1/'
    number_file = output_dir + name2d + '_Number_Rz.npy'
    mdot_file   = output_dir + name2d + '_Mdot_Rz.npy'
    B_file = output_dir + name2d + '_B_Rz.npy'
    v_file = output_dir + name2d + '_v_Rz.npy'
    j_file = output_dir + name2d + '_j_Rz.npy'
    
    Number_Rz = (np.load(number_file))
    
    
    Mdot_Rz = (np.load(mdot_file)) / Number_Rz
    B_Rz = (np.load(B_file)) / Number_Rz
    v_Rz = (np.load(v_file)) / Number_Rz
        
    fig, ax = plt.subplots(figsize=(8,6))
    
    J_Rz = j_if_disc(B=B_Rz, v=v_Rz, M_dot=Mdot_Rz)
    # J_Rz[J_Rz<1] = np.nan
    # counts = J_Rz / Number_Rz
    
    # J_1d = J_Rz.reshape(-1)
    # print(len(J_1d[J_1d>1]))
    
    j_Rz = J_Rz * Number_Rz#
    j_Rz = np.load(j_file)
    
    norm = np.sum(j_Rz)
    
    
    disc_percentage = np.sum(j_Rz)/np.sum(Number_Rz)
    
    
    print("{}\t{}\t{}:\t{:.2f}".format(field, case, galaxy_type, disc_percentage*100))
    
    R2bins = np.linspace(0, 20, 200+1)
    z2bins = np.linspace(0, 2, 200+1)
    """ Binning """
    factor = 1  # укрупнение в 4 раза
    Number_rebinned = rebin_2d(Number_Rz, factor, factor)
    j_rebinned = rebin_2d(j_Rz, factor, factor)
    counts = j_rebinned / Number_rebinned
    R2bins = rebin_edges(R2bins, factor)
    z2bins = rebin_edges(z2bins, factor)
    
    """ Do not show the bins where the statistics is bad """
    # mask = Number_rebinned < 10
    # counts[mask] = np.nan
    
    # counts = v_Rz.T / 1e5
    # counts = np.log10(Mdot_Rz.T)
    # plt.pcolormesh(R2bins, z2bins, Number_Rz.T, shading='auto') # Number_Rz also an interesting thing
    cms = plt.pcolormesh(R2bins, z2bins, counts.T, shading='auto') # Number_Rz also an interesting thing
    plt.xlabel('$R$, kpc')
    plt.ylabel('$z$, kpc')
    # plt.title('Mdot(R,z), [g/s]')
    
    cbar = plt.colorbar()
    # cbar.set_label('log$_{10}\dot{M}$, [g/s]')
    # cbar.set_label('$j_t(v) / j_K(B, v, \dot{M})$')
    cbar.set_label('fraction of accretors with discs')
    
    # smoothed_counts = sp.ndimage.gaussian_filter(counts, sigma=1)
    # levels = np.linspace(8,15, 20)
    # cnt = plt.contour((R2bins[1:]+R2bins[:1])/2, (z2bins[1:]+z2bins[:1])/2, smoothed_counts, levels, colors='white', rasterized=True, linewidths=0.5)
    #                     # vmin=vmin, vmax=vmax)
    # mpl.rcParams["text.usetex"] = False
    # try:
    #     plt.clabel(cnt, levels, fontsize=10, fmt='%r')
    # except:
    #     pass

    ax.set_title('ISM: {}, Magnetic field: {}, Propeller model: {}\nTotal ratio of disc accretors to accretors: {:.4f}'.format(galaxy_type, field, case, disc_percentage))

    disc_path = '/home/afoninamd/Documents/NS/project/pops/result/disc_figures/'
    fig.savefig(disc_path+'MdotMap_{}_{}_{}.pdf'.format(galaxy_type, field, case), format='pdf')


def PeqMap(galaxy_type='simple', field='CF', case='A', get_tau_flag=False):
    name2d = '{}_{}_{}'.format(galaxy_type, field, case)
    output_dir = '/home/afoninamd/Documents/NS/project/pops/result/disks1/'
    number_file = output_dir + name2d + '_Number_Rz.npy'
    mdot_file   = output_dir + name2d + '_Mdot_Rz.npy'
    B_file = output_dir + name2d + '_B_Rz.npy'
    v_file = output_dir + name2d + '_v_Rz.npy'
    j_file = output_dir + name2d + '_j_Rz.npy'
    
    Number_Rz = (np.load(number_file))
    
    Mdot_Rz = (np.load(mdot_file)) / Number_Rz
    B_Rz = (np.load(B_file)) / Number_Rz
    v_Rz = (np.load(v_file)) / Number_Rz
        
    
    
    def get_P_eq(B, v, M_dot):
        """
        https://articles.adsabs.harvard.edu/pdf/1973ApJ...179..585D 
        https://academic.oup.com/mnras/article/533/1/386/7720517
        """
        # R_t = 2e20
        # v_t = 10e5
        # R_G = 2*G*M_NS/v**2
        # n = M_dot / (np.pi*R_G**2*v*m_p)
        mu = B*R_NS**3
        # v[v>60e5] = np.nan
        R_A = (mu**2 / (2 * M_dot * (2 * G*M_NS)**0.5))**(2/7)
        # P_eq = 2.6e3 * (v_t/10e5)**(-2/3)*(mu/1e30)**(2/3)*(n)**(-2/3)*(v/10e5)**(13/3) # wrong
        # L = G*M_NS*M_dot/R_NS
        # P_eq = (L/1e37)**(-3/7)*(mu/1e30)**(6/7) # with lower B
        
        # print(R_A)
        P_eq = ((R_A/0.87)**3 * (2*np.pi)**2 / (G*M_NS))**0.5
        
        # print(P_eq/P_eq0) # it is 3
        return mu**2/(G*M_NS*P_eq**2/4/np.pi**2) / (M_dot*(G*M_NS*R_A)**0.5)
    
    def get_tau(B, v, M_dot):
        mu = B*R_NS**3
        R_A = (mu**2 / (2 * M_dot * (2 * G*M_NS)**0.5))**(2/7)
        P = get_P_eq(B, v, M_dot)
        tau = 2*np.pi*1e45 / (M_dot * (G*M_NS*R_A)**0.5) / P#tau = P / P_dot
        return tau
    
    fig, ax = plt.subplots(figsize=(8,6))
    if get_tau_flag == True:
        # counts = v_Rz
        counts = np.log10(get_tau(B=B_Rz, v=v_Rz, M_dot=Mdot_Rz))
        print(np.min(counts[counts>0]))
    else:
        counts = np.log10(get_P_eq(B=B_Rz, v=v_Rz, M_dot=Mdot_Rz))
        
    
    
    R2bins = np.linspace(0, 20, 200+1)
    z2bins = np.linspace(0, 2, 200+1)
    
    
    """ Binning """
    # factor = 1  # укрупнение в 4 раза
    # Number_rebinned = rebin_2d(Number_Rz, factor, factor)
    # j_rebinned = rebin_2d(j_Rz, factor, factor)
    # counts = j_rebinned / Number_rebinned
    # R2bins = rebin_edges(R2bins, factor)
    # z2bins = rebin_edges(z2bins, factor)
    
    # """ Do not show the bins where the statistics is bad """
    # # mask = Number_rebinned < 10
    # # counts[mask] = np.nan
    
    plt.pcolormesh(R2bins, z2bins, counts.T, shading='auto', cmap='viridis_r') # Number_Rz also an interesting thing
    plt.xlabel('$R$, kpc')
    plt.ylabel('$z$, kpc')
    # plt.title('Mdot(R,z), [g/s]')
    
    cbar = plt.colorbar()
    # cbar.set_label('log$_{10}\dot{M}$, [g/s]')
    # cbar.set_label('$j_t(v) / j_K(B, v, \dot{M})$')
    if get_tau:
        cbar.set_label('np.log10(tau = P/P_dot)')
    else:
        cbar.set_label('np.log10 Equillibrium Period')
    
    # smoothed_counts = sp.ndimage.gaussian_filter(counts, sigma=1)
    # levels = np.linspace(8,15, 20)
    # cnt = plt.contour((R2bins[1:]+R2bins[:1])/2, (z2bins[1:]+z2bins[:1])/2, smoothed_counts, levels, colors='white', rasterized=True, linewidths=0.5)
    #                     # vmin=vmin, vmax=vmax)
    # mpl.rcParams["text.usetex"] = False
    # try:
    #     plt.clabel(cnt, levels, fontsize=10, fmt='%r')
    # except:
    #     pass

    # ax.set_title('ISM: {}, Magnetic field: {}, Propeller model: {}\nTotal ratio of disc accretors to accretors: {:.4f}'.format(galaxy_type, field, case, disc_percentage))

    # disc_path = '/home/afoninamd/Documents/NS/project/pops/result/disc_figures/'
    # fig.savefig(disc_path+'EqP_{}_{}_{}.pdf'.format(galaxy_type, field, case), format='pdf')
# PeqMap(galaxy_type='simple', field='ED', case='A')


            
def get_h(galaxy_type='two_phase', field='ED', case='B'):
    """ half-width of a disc with disc-accerors """
    name2d = '{}_{}_{}'.format(galaxy_type, field, case)
    output_dir = '/home/afoninamd/Documents/NS/project/pops/result/disks1/'
    number_file = output_dir + name2d + '_Number_Rz.npy'
    mdot_file   = output_dir + name2d + '_Mdot_Rz.npy'
    B_file = output_dir + name2d + '_B_Rz.npy'
    v_file = output_dir + name2d + '_v_Rz.npy'
    j_file = output_dir + name2d + '_j_Rz.npy'
    
    Number_Rz = (np.load(number_file))
    
    
    Mdot_Rz = (np.load(mdot_file)) / Number_Rz
    B_Rz = (np.load(B_file)) / Number_Rz
    v_Rz = (np.load(v_file)) / Number_Rz
    
    R2bins = np.linspace(0, 20, 200+1)
    z2bins = np.linspace(0, 2, 200+1)
    
    R = (R2bins[1:] + R2bins[:-1])/2
    z = (z2bins[1:] + z2bins[:-1])/2
    # fig, ax = plt.subplots(figsize=(8,6))
    
    # J_Rz = j_if_disc(B=B_Rz, v=v_Rz, M_dot=Mdot_Rz)
    # J_Rz[J_Rz<1] = np.nan
    # counts = J_Rz / Number_Rz
    
    # J_1d = J_Rz.reshape(-1)
    # print(len(J_1d[J_1d>1]))
    
    # j_Rz = J_Rz * Number_Rz#
    j_Rz = np.load(j_file)
    
    norm = np.sum(j_Rz)
    
    j = np.sum(j_Rz, axis=0) # sun along r axis
    j_sum = np.sum(j)
    
    j_cur = j[0]
    j_rec = 68.27/100 * j_sum
    j_rec = 95.45/100 * j_sum
    for i in range(1, len(j)):
        if j_cur >= j_rec:
            k = i
            break
        else:
            j_cur += j[i]
            
    
    fig, ax = plt.subplots()
    ax.plot(z, j)
    ax.axvline(z[k], color='k', ls='--')
    ax.set_ylabel('Number of discs')
    ax.set_xlabel('z, kpc')
    
    return k, z[k] # it is the half-height for the accretors (1 sigma)


def get_accretor_number_in_h(galaxy_type='two_phase', field='ED', case='B'):
    """ half-width of a disc with disc-accerors """
    name2d = '{}_{}_{}'.format(galaxy_type, field, case)
    output_dir = '/home/afoninamd/Documents/NS/project/pops/result/disks1/'
    number_file = output_dir + name2d + '_Number_Rz.npy'
    mdot_file   = output_dir + name2d + '_Mdot_Rz.npy'
    B_file = output_dir + name2d + '_B_Rz.npy'
    v_file = output_dir + name2d + '_v_Rz.npy'
    j_file = output_dir + name2d + '_j_Rz.npy'
    
    Number_Rz = (np.load(number_file))
    
    R2bins = np.linspace(0, 20, 200+1)
    z2bins = np.linspace(0, 2, 200+1)
    
    R = (R2bins[1:] + R2bins[:-1])/2
    z = (z2bins[1:] + z2bins[:-1])/2
    # fig, ax = plt.subplots(figsize=(8,6))
    
    # J_Rz = j_if_disc(B=B_Rz, v=v_Rz, M_dot=Mdot_Rz)
    # J_Rz[J_Rz<1] = np.nan
    # counts = J_Rz / Number_Rz
    
    # J_1d = J_Rz.reshape(-1)
    # print(len(J_1d[J_1d>1]))
    
    # j_Rz = J_Rz * Number_Rz#
    k, h = get_h(galaxy_type, field, case)
    
    j = np.sum(Number_Rz, axis=0) # sun along r axis
    j_sum = np.sum(j)
    
    j_in_h = np.sum(j[:k])
    
    
    
    fig, ax = plt.subplots()
    ax.plot(z, j, color='red')
    ax.axvline(z[k], color='k', ls='--')
    ax.set_ylabel('Number of accretors')
    ax.set_xlabel('z, kpc')
    
    
    print(j_in_h, '/', j_sum, '=')
    print(j_in_h/j_sum) # it is the half-height for the accretors (1 sigma)


def get_accretor_number_in_r(galaxy_type='two_phase', field='ED', case='B', Radius=1, Zadius=1, accretor_part=0.1):
    """ Number of accretors and accretors with discs near the sun (1 kpc) """
    """ half-width of a disc with disc-accerors """
    name2d = '{}_{}_{}'.format(galaxy_type, field, case)
    output_dir = '/home/afoninamd/Documents/NS/project/pops/result/disks1/'
    number_file = output_dir + name2d + '_Number_Rz.npy'
    mdot_file   = output_dir + name2d + '_Mdot_Rz.npy'
    B_file = output_dir + name2d + '_B_Rz.npy'
    v_file = output_dir + name2d + '_v_Rz.npy'
    j_file = output_dir + name2d + '_j_Rz.npy'
    
    Number_Rz = (np.load(number_file)).T
    norm = np.sum(Number_Rz)
    j_Rz = (np.load(j_file)).T
    
    R2bins = np.linspace(0, 20, 200+1)
    z2bins = np.linspace(0, 2, 200+1)
    
    R = (R2bins[1:] + R2bins[:-1])/2
    z = (z2bins[1:] + z2bins[:-1])/2

    Rgrid, zgrid = np.meshgrid(R, z)

    # Create a mask where the conditions are met
    mask = np.logical_or(abs(Rgrid-8)>Radius, zgrid > Zadius)
    
    V = (2*Radius)**2 *(2*Zadius) # be careful with the volume you choose!!!
    Num_in_gal = 3e8 * accretor_part
    Num = (2*Radius/8/2/np.pi) * Num_in_gal
    # norm = norm / 1e9 # to kpc
    
    # print(mask)
    
    counts = Number_Rz
    counts[mask] = 0
    
    j_Rz[mask] = 0
    print("Part of accretors in the 1 kpc is {:.2e}".format(np.sum(counts)/norm*Num))
    print("Part of discs to accretors in the 1 kpc is {:.2e}".format(np.sum(j_Rz)/norm*Num))
    
    print("Number of accretors in the 1 pc**3 is {:.2e}".format(np.sum(counts)/norm*Num/V/1e9))
    print("Number of discs to accretors in the 1 pc**3 is {:.2e}".format(np.sum(j_Rz)/norm*Num/V/1e9))

    fig, ax = plt.subplots()
    plt.pcolormesh(Rgrid, zgrid, counts, shading='auto', cmap='viridis') # Number_Rz also an interesting thing
    plt.xlabel('$R$, kpc')
    plt.ylabel('$z$, kpc')
    
    cbar = plt.colorbar()
    # cbar.set_label('log$_{10}\dot{M}$, [g/s]')
    # cbar.set_label('$j_t(v) / j_K(B, v, \dot{M})$')
    # cbar.set_label('fraction of accretors with discs')
    
    # Set counts to zero where the mask is true
    

# get_accretor_number_in_r(galaxy_type='two_phase', field='ED', case='B')
# for field in ['CF', 'ED']:
#     for case in ['A', 'B', 'C', 'D']:
#         for galaxy_type in ['simple', 'two_phase']:
#             # print(case)
#             MdotMap(galaxy_type, field, case=case)

# get_accretor_number_in_h(galaxy_type='two_phase', field='ED', case='B')

# get_h(galaxy_type='two_phase', field='ED', case='B')

def CDF_latitude(galaxy_type='two_phase', field='ED', case='B'):
    
    output_dir = '/home/afoninamd/Documents/NS/project/pops/result/disks1/'

    # disc_percentage = disc_percentage.flatten()
    def get_latitude(R, z):
        R0 = 8
        # Raddist = R-R0 # everything flat
        phi = np.random.uniform(0, 2*np.pi) 
        # phi = np.random.uniform(0, np.pi/18)
        Raddist = (R0**2+R**2-2*R0*R*np.cos(phi))**0.5
        sinmodb = z / (z**2+(Raddist)**2)**0.5
        return np.arcsin(sinmodb)
    
    R_bins = np.linspace(0, 20, 201)
    z_bins = np.linspace(0, 2, 201)
    R, z = np.meshgrid((R_bins[1:]+R_bins[:1])/2, (z_bins[1:]+z_bins[:1])/2)
    b_arr = get_latitude(R, z)
    b_bins = np.linspace(0, np.pi, 100001)
    x_arr = (b_bins[1:]+b_bins[:1])/2*180/np.pi
    x_arr[0] = 0
    
    fig, ax = plt.subplots(figsize=(7,6))
    """ LPT part """
    lpt = np.array(lpt_b_deg)
    lpt = lpt[is_NS==1]
    lpt = np.sort(np.abs(lpt))
    
    cdf_lpt, _ = np.histogram(lpt, weights=lpt*0+1, bins=b_bins*180/np.pi)
    cdf_lpt_1 = np.zeros_like(cdf_lpt)  # Create an array of the same shape
    cdf_lpt_1[0] = cdf_lpt[0]
    for i in range(1, len(cdf_lpt)):
        cdf_lpt_1[i] = cdf_lpt_1[i-1] + cdf_lpt[i]
    ax.plot(x_arr, cdf_lpt_1/(cdf_lpt_1[-1]), color='black', label='{} LPTs (NS candidates)'.format(len(lpt)))
    
    """ Obtaining the data """
    for field in ['CF', 'ED']:
        for case in ['A', 'B', 'C']:
            for galaxy_type in ['simple', 'two_phase']:
                name2d = '{}_{}_{}'.format(galaxy_type, field, case)
                j_file = output_dir + name2d + '_j_Rz.npy'
                j_Rz = np.load(j_file)
                
                cdf_counts, _ = np.histogram(b_arr.flatten(), weights=j_Rz.flatten(), bins=b_bins)
                cdf_counts_1 = np.zeros_like(cdf_counts)  # Create an array of the same shape
                cdf_counts_1[0] = cdf_counts[0]
                for i in range(1, len(cdf_counts)):
                    cdf_counts_1[i] = cdf_counts_1[i-1] + cdf_counts[i]
                
                """ Plotting the data """
                # ax.hist(cdf_counts_1, bins=b_bins, label='CDF of number of objects')
                ax.plot(x_arr, cdf_counts_1/(cdf_counts_1[-1]), label=f'{field}, {case}, {galaxy_type}')
    
    
    
    ax.plot(x_arr, cdf_lpt_1/(cdf_lpt_1[-1]), color='black')
    
    ax.set_xlabel('Galactic Latitude b, deg')
    ax.set_ylabel('Cumulative Number of LPRTs')
    # ax.title('Cumulative Distribution of Objects by Galactic Latitude')
    # plt.legend()
    ax.set_xlim([-0.3, 5])
    plt.grid()
    plt.show()
    plt.legend(fontsize=12)
    
    disc_path = '/home/afoninamd/Documents/NS/project/pops/result/disc_figures/'
    fig.savefig(disc_path+'lpt.pdf', format='pdf')
    


CDF_latitude()
# CDF_latitude(galaxy_type='simple', field='CF', case='A')


# for case in ['A', 'B', 'C', 'D']:
#     for field in ['CF', 'ED']:
#         for galaxy_type in ['simple', 'two_phase']:
#             MdotMap(galaxy_type, field, case)
# MdotMap(galaxy_type='two_phase', field='ED', case='B')
# MdotMap(galaxy_type='two_phase', field='ED', case='A')


