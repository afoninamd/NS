#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:43:35 2026

@author: afoninamd
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from main.constants import M_sun, AU, R_sun


def plot_4d_results(filename="popsynth_4d.npz"):

    data = np.load(filename)

    a     = data["a"]
    Mstar = data["Mstar"]
    ecc   = data["ecc"]
    B0    = data["B0"]
    time  = data["time"]
    P     = data["P"]

    # -------------------------------------------------
    # Convert to physical plotting variables
    # -------------------------------------------------
    
    loga = np.log10(a/R_sun)
    logM = (Mstar/M_sun)
    logB = np.log10(B0)
    e    = ecc

    vars_list = [loga, logM, e, logB]
    labels = [
        r'log$_{10} a$, R_sun',
        r'$M_\star$, M$_\odot$',
        r'$e$',
        r'log$_{10} B_0$, G'
    ]

    n_vars = 4

    # -------------------------------------------------
    # Normalize (like your code)
    # -------------------------------------------------

    valid = ~np.isnan(time)
    total = np.product(time.shape)

    weights = np.zeros_like(time)
    weights[valid] = 1.0 / total  # fraction of systems reaching stage 3

    # -------------------------------------------------
    # Precompute 1D projections
    # -------------------------------------------------

    counts_list = []
    centers_list = []

    for i in range(n_vars):
        axes_to_sum = tuple(ax for ax in range(4) if ax != i)
        counts = np.sum(weights, axis=axes_to_sum)

        centers = vars_list[i]

        counts_list.append(counts)
        centers_list.append(centers)

    # -------------------------------------------------
    # Plot grid
    # -------------------------------------------------

    fig, axes = plt.subplots(n_vars, n_vars, figsize=(12, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.1)

    vmin = 0
    vmax = np.nanmax(weights) * 50

    for i in range(n_vars):
        for j in range(n_vars):

            ax = axes[i, j]

            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])

            if i == n_vars - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])

            # -------------------------------------------------
            # Diagonal: 1D distributions
            # -------------------------------------------------
            if i == j:

                ax.plot(centers_list[i], counts_list[i], color='black')
                ax.set_xlim([centers_list[i][0], centers_list[i][-1]])

            # -------------------------------------------------
            # Lower triangle: 2D projections
            # -------------------------------------------------
            elif i > j:

                x = vars_list[j]
                y = vars_list[i]

                axes_to_sum = tuple(ax for ax in range(4) if ax not in (i, j))
                counts = np.sum(weights, axis=axes_to_sum)

                X, Y = np.meshgrid(x, y)

                pcm = ax.pcolormesh(
                    X, Y, counts.T,
                    shading='auto',
                    cmap='viridis',
                    vmin=vmin,
                    vmax=vmax
                )

                # smoothing (like your code)
                smoothed = sp.ndimage.gaussian_filter(counts.T, sigma=1.0)

                levels = np.linspace(np.nanmin(smoothed), np.nanmax(smoothed), 4)[1:]

                try:
                    ax.contour(X, Y, smoothed, levels=levels, colors='white', linewidths=0.5)
                except:
                    pass

            else:
                ax.axis('off')

    # -------------------------------------------------
    # Colorbar
    # -------------------------------------------------

    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.75])
    fig.colorbar(pcm, cax=cbar_ax, label='Fraction reaching stage 3')

    plt.show()
    
plot_4d_results(filename="/home/afoninamd/Documents/NS/project/binary/popsynth_4d.npz")