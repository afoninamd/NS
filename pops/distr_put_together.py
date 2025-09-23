#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 11:37:04 2025

@author: afoninamd
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main.constants import N, output_dir


def create_pulsar():
    N1 = N // 10 * 9
    kind = 'pulsar'
    df0 = pd.DataFrame({'P': np.array([]), 'B': np.array([]),
                       'x': np.array([]), 'y': np.array([]), 'z': np.array([]),
                       'mode': np.array([]),
                       'Vx': np.array([]), 'Vy': np.array([]), 'Vz': np.array([]),
                       'Vxpec': np.array([]), 'Vypec': np.array([]),
                       'Vzpec': np.array([])})
    
    pattern = output_dir + 'distr/distribution_{}_*_*.csv'.format(kind)
    for file_name in glob.glob(pattern):
        df = pd.read_csv(file_name, sep=';')
        df0 = df0.append(df)
    
    pd.DataFrame.to_csv(df0, output_dir + 'distribution_{}_{}.csv'.format(kind, len(df0['P'])), sep=';')


def create_magnetar():
    N1 = N // 10
    kind = 'magnetar'
    df0 = pd.DataFrame({'P': np.array([]), 'B': np.array([]),
                       'x': np.array([]), 'y': np.array([]), 'z': np.array([]),
                       'mode': np.array([]),
                       'Vx': np.array([]), 'Vy': np.array([]), 'Vz': np.array([]),
                       'Vxpec': np.array([]), 'Vypec': np.array([]),
                       'Vzpec': np.array([])})
    df0['P_in'] = np.array([])
    df0['B_in'] = np.array([])
    
    pattern = output_dir + 'distr/distribution_{}_*_*.csv'.format(kind)
    for file_name in glob.glob(pattern):
        df = pd.read_csv(file_name, sep=';')
        df0 = df0.append(df)
    
    pd.DataFrame.to_csv(df0, output_dir + 'distribution_{}_{}.csv'.format(kind, len(df0['P'])), sep=';')


create_pulsar()
create_magnetar()