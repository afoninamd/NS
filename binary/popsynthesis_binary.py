#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main.constants import year, AU, day, M_sun
from binary.evolution_binary import one_binary_evolution

import numpy as np
from tqdm import tqdm


# ============================================================
# Utility: convert axis → orbital period
# ============================================================

def axis_to_Porb(a, Mstar):
    """
    Kepler's 3rd law:
    P^2 ~ a^3 / (M1 + M2)
    """
    return (a/AU)**1.5 * (1/(1.4 + Mstar/M_sun))**0.5 * year


# ============================================================
# MAIN 4D POPULATION SYNTHESIS
# ============================================================

def popsynthesis_4d(num_a=5, num_M=5, num_e=5, num_B=5):

    # -------------------------------
    # Parameter grids
    # -------------------------------
    
    a_arr      = 10**np.linspace(-1, 1, num_a) * AU       # orbital separation
    Mstar_arr  = np.linspace(0.5, 10, num_M) * M_sun      # stellar mass
    ecc_arr    = np.linspace(0.0, 0.9, num_e)             # eccentricity
    B0_arr     = 10**np.linspace(9, 13, num_B)           # magnetic field

    # -------------------------------
    # Allocate result arrays
    # -------------------------------
    
    shape = (num_M, num_a, num_e, num_B)

    P_arr  = np.full(shape, np.nan)      # index of first stage 3
    time_arr = np.full(shape, np.nan)          # time of first stage 3

    # -------------------------------
    # Main loop
    # -------------------------------
    
    total = num_M * num_a * num_e * num_B
    
    with tqdm(total=total) as pbar:

        for i in range(num_M):
            for j in range(num_a):
                for k in range(num_e):
                    for l in range(num_B):

                        Mstar = Mstar_arr[i]
                        a     = a_arr[j]
                        ecc0  = ecc_arr[k]
                        B0    = B0_arr[l]

                        # convert to Porb
                        Porb0 = axis_to_Porb(a, Mstar)

                        try:
                            out = one_binary_evolution(
                                P0=0.01,
                                B0=B0,
                                case='A',
                                field='CF',
                                Mstar=Mstar,
                                ecc0=ecc0,
                                Porb0=Porb0,
                                plot=False
                            )

                            if out is None:
                                pbar.update(1)
                                continue

                            t, P, stages = out

                            # fast detection of first stage 3
                            mask = (stages == 3)

                            if mask.any():
                                first_idx = mask.argmax()
                                P_arr[i, j, k, l]  = P[first_idx]
                                time_arr[i, j, k, l] = t[first_idx]

                        except Exception:
                            # optional: log errors
                            pass

                        pbar.update(1)

    return {
        "a": a_arr,
        "Mstar": Mstar_arr,
        "ecc": ecc_arr,
        "B0": B0_arr,
        "P": P_arr,
        "time": time_arr
    }


# ============================================================
# SAVE / LOAD
# ============================================================

def save_results(results, filename="popsynth_4d.npz"):
    np.savez(
        filename,
        a=results["a"],
        Mstar=results["Mstar"],
        ecc=results["ecc"],
        B0=results["B0"],
        P=results["P"],
        time=results["time"]
    )


def load_results(filename="popsynth_4d.npz"):
    return dict(np.load(filename))


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    results = popsynthesis_4d(
        # num_a=5,
        # num_M=5,
        # num_e=4,
        # num_B=4
    )

    save_results(results)

    print("Done.")