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

path = '/home/afoninamd/Downloads/'
path = '/home/afoninamd/Documents/project/pops/result/realistic/'
# res_name = 'realistic0/'
# arr_size = 100
# Rbins = np.linspace(0, 20, 101) # from the GC
# rbins = np.linspace(0, 10, 101) # from the Sun
# zbins = np.linspace(0, 5, 101)
# fbins = 10**np.linspace(-15, -5, 101)
# cbins = 10**np.linspace(-4, 6, 101)
# vbins = np.linspace(0, 500, 101) * 1e5
# Tbins = 10**np.linspace(4, 9, 101)

res_name = ''
arr_size = 2000
Rbins = np.linspace(0, 20, arr_size+1) # from the GC
rbins = np.linspace(0, 20, arr_size+1) # from the Sun better for 0 to 20
zbins = np.linspace(0, 5, arr_size+1)
fbins = 10**np.linspace(-15, -5, arr_size+1)
cbins = 10**np.linspace(-4, 6, arr_size+1)
vbins = np.linspace(0, 500, arr_size+1) * 1e5
Tbins = 10**np.linspace(5, 9, arr_size+1) # better from 5 to 8

for galaxy_type in ['simple']:#, 'two_phase']:
    for field in ['CF']:#, 'ED']:
        for case in ['A']:#, 'B', 'C', 'D']:
            for add_string in ['']:#, '_roman']:
                data_arr = np.zeros(arr_size)
                for rand_i in range(10):
                    
                    file_name = '{}_{}_{}_{}_erosita{}_{}'.format(galaxy_type, field,
                                                    case, 'pulsar', add_string, rand_i)
                    df = pd.read_feather(path + res_name + file_name + '.feather')
                    file_name = '{}_{}_{}_{}_erosita{}_{}'.format(galaxy_type, field,
                                                    case, 'magnetar', add_string, rand_i)
                    data = df+pd.read_feather(path + res_name + file_name + '.feather')
                    arr = cbins
                    data_arr += data['c0']
                    plt.plot((arr[1:]+arr[:-1])/2, data['c0'], alpha=0.3)
                plt.plot((arr[1:]+arr[:-1])/2, data_arr, 'red')
                    # plt.yscale('log')
                    # plt.xscale('log')

















