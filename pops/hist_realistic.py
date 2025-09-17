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
# res_name = 'realistic0/'
# arr_size = 100
# Rbins = np.linspace(0, 20, 101) # from the GC
# rbins = np.linspace(0, 10, 101) # from the Sun
# zbins = np.linspace(0, 5, 101)
# fbins = 10**np.linspace(-15, -5, 101)
# cbins = 10**np.linspace(-4, 6, 101)
# vbins = np.linspace(0, 500, 101) * 1e5
# Tbins = 10**np.linspace(4, 9, 101)

res_name = 'realistic0/'
arr_size = 100
Rbins = np.linspace(0, 20, 101) # from the GC
rbins = np.linspace(0, 10, 101) # from the Sun
zbins = np.linspace(0, 5, 101)
fbins = 10**np.linspace(-15, -5, 101)
cbins = 10**np.linspace(-4, 6, 101)
vbins = np.linspace(0, 500, 101) * 1e5
Tbins = 10**np.linspace(4, 9, 101)

for galaxy_type in ['simple']:#, 'two_phase']:
    for field in ['CF', 'ED']:
        for case in ['A']:#, 'B', 'C', 'D']:
            for add_string in ['']:#, '_roman']:
                file_name = '{}_{}_{}_{}_erosita{}'.format(galaxy_type, field,
                                                case, 'pulsar', add_string)
                df = pd.read_feather(path + res_name + file_name + '.feather')
                file_name = '{}_{}_{}_{}_erosita{}'.format(galaxy_type, field,
                                                case, 'magnetar', add_string)
                data = df+pd.read_feather(path + res_name + file_name + '.feather')
                arr = zbins
                plt.plot((arr[1:]+arr[:-1])/2, data['z'])
                # plt.yscale('log')
                # plt.xscale('log')