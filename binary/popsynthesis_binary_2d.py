#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:29:00 2026

@author: afoninamd
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tqdm import tqdm
from main.constants import AU, M_sun, year
from binary.evolution_binary import one_binary_evolution

from concurrent.futures import ProcessPoolExecutor

def axis_to_Porb(a, Mstar):
    return (a/AU)**1.5 * (1/(1.4 + Mstar/M_sun))**0.5 * year

def popsynthesis_2d(num_a=100, num_e=100,
                      Mstar=1.0*M_sun,
                      B0=1e12):

    # -------------------------------
    # 🔹 grids
    # -------------------------------
    
    a_arr   = 10**np.linspace(-1, 1, num_a) * AU
    ecc_arr = np.linspace(0.0, 0.9, num_e)

    # -------------------------------
    # 🔹 results
    # -------------------------------
    
    P_arr  = np.full((num_a, num_e), np.nan)
    time_arr = np.full((num_a, num_e), np.nan)

    # -------------------------------
    # 🔹 loop
    # -------------------------------
    
    total = num_a * num_e

    with tqdm(total=total) as pbar:

        for i, a in enumerate(a_arr):
            for j, ecc0 in enumerate(ecc_arr):

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

                    mask = (stages == 3)
                    if mask.any():
                        idx = mask.argmax()
                        P_arr[i, j]  = P[idx]
                        time_arr[i, j] = t[idx]

                except:
                    pass

                pbar.update(1)

    return {
        "a": a_arr,
        "ecc": ecc_arr,
        "P": P_arr,
        "time": time_arr
    }


def save_2d(results, filename="popsynth_2d_ae.npz"):
    np.savez(filename,
             a=results["a"],
             ecc=results["ecc"],
             P=results["P"],
             time=results["time"])


def load_2d(filename="popsynth_2d_ae.npz"):
    return dict(np.load(filename))


def run_one_simulation(run_id):
    print(f"Starting run {run_id}")

    results = popsynthesis_2d()

    filename = f"popsynth_2d_ae_run_{run_id}.npz"
    save_2d(results, filename)

    print(f"Finished run {run_id}")

    return filename


if __name__ == "__main__":

    n_runs = 9
    n_workers = max(1, os.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        files = list(executor.map(run_one_simulation, range(n_runs)))

    print("All runs finished:")
    for f in files:
        print(f)


def load_all_runs(n_runs=10):
    all_data = []

    for i in range(n_runs):
        data = np.load(f"popsynth_2d_ae_run_{i}.npz")
        all_data.append(data["time"])

    return np.stack(all_data)
# load_all_runs()
# if __name__ == "__main__":

#     results = popsynthesis_2d(
#         # num_a=5,
#         # num_M=5,
#         # num_e=4,
#         # num_B=4
#     )

#     save_2d(results)

#     print("Done.")

