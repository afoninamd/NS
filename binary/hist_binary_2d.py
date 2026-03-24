#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:32:04 2026

@author: afoninamd
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from main.constants import AU, R_sun, M_sun, year

def plot_2d_fraction(filename="popsynth_2d_ae.npz"):

    data = np.load(filename)

    a   = data["a"]
    ecc = data["ecc"]
    idx = data["idx"]

    # fraction = 1 if stage 3 reached
    fraction = (idx != -1).astype(float)

    # normalize (optional smoothing)
    smoothed = sp.ndimage.gaussian_filter(fraction, sigma=1.0)

    X, Y = np.meshgrid(ecc, np.log10(a/AU))

    plt.figure(figsize=(6, 5))

    pcm = plt.pcolormesh(
        X, Y, smoothed,
        shading='auto',
        cmap='viridis',
        vmin=0,
        vmax=1
    )

    plt.colorbar(pcm, label='Fraction reaching stage 3')

    # contours
    levels = [0.1, 0.3, 0.5, 0.7]
    try:
        plt.contour(X, Y, smoothed, levels=levels, colors='white', linewidths=0.7)
    except:
        pass

    plt.xlabel('eccentricity')
    plt.ylabel(r'log$_{10} a$ [AU]')
    plt.title('Accretor Fraction')

    plt.tight_layout()
    plt.show()


def plot_2d_time_one_file(filename="/home/afoninamd/Documents/NS/project/binary/popsynth_2d_ae.npz"):
    path = "/home/afoninamd/Documents/NS/project/binary/"
    data = np.load(filename)
    
    # for i in range(0, 10):
        # name_of_one_file = "popsynth_2d_ae_run_{}.npz".format(i)
        # data = np.load(path + name_of_one_file)

    a   = data["a"]
    ecc = data["ecc"]
    time = data["time"]

    # mask invalid
    valid = ~np.isnan(time)

    # mean time (already single value per cell, so just mask)
    time_plot = np.copy(time)
    time_plot[~valid] = np.nan

    X, Y = np.meshgrid(ecc, np.log10(a/AU))

    plt.figure(figsize=(6, 5))

    pcm = plt.pcolormesh(
        X, Y, np.log10(time_plot/year),
        shading='auto',
        cmap='inferno'
    )

    plt.colorbar(pcm, label=r'log$_{10}$ $t_accr$ [yr]')

    plt.xlabel('eccentricity')
    plt.ylabel(r'log$_{10} a$ [AU]')
    plt.title('Accretion Onset Time')

    plt.tight_layout()
    plt.show()


# plot_2d_time()

def plot_2d_time_sum_files(filename="/home/afoninamd/Documents/NS/project/binary/popsynth_2d_ae.npz"):
    path = "/home/afoninamd/Documents/NS/project/binary/results_binary/"
    
    time_list = []

    # Load all runs
    for i in range(10):
        name_of_one_file = f"popsynth_2d_ae_run_{i}.npz"
        data = np.load(path + name_of_one_file)

        time = data["time"]
        time_list.append(time)

    # Stack into 3D array: (runs, ...)
    time_stack = np.array(time_list)

    # Compute mean ignoring NaNs
    time_mean = np.nanmean(time_stack, axis=0)

    # Load grid from one file (assuming same for all runs)
    data = np.load(path + "popsynth_2d_ae_run_0.npz")
    a   = data["a"]
    ecc = data["ecc"]

    X, Y = np.meshgrid(ecc, np.log10(a / AU))

    plt.figure(figsize=(6, 5))

    pcm = plt.pcolormesh(
        X, Y, np.log10(time_mean / year),
        shading='auto',
        cmap='inferno'
    )

    plt.colorbar(pcm, label=r'log$_{10}$ $t_{accr}$ [yr]')

    plt.xlabel('eccentricity')
    plt.ylabel(r'log$_{10} a$ [AU]')
    plt.title('Accretion Onset Time (Mean over runs)')
    
    plt.plot(ecc[ecc>0.45], np.log10(12*R_sun/(1-ecc[ecc>0.45])/AU), ls='--', lw='3', color='white')
    
    plt.tight_layout()
    plt.show()

# plot_2d_time_sum_files()