#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 18:36:38 2025

@author: afoninamd
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main.constants import output_dir
import numpy as np
import glob
import pandas as pd
import pyarrow.feather as feather

def create_one_txt():
    for addstring in ['', '_roman']:
        for star_type in ['pulsar', 'magnetar']:
            for galaxy_type in ['simple', 'two_phase']:
                for field in ['CF', 'ED']:
                    for case in ['A', 'B', 'C', 'D']:
                        row = '{}_{}_{}_{}{}'.format(galaxy_type, field, case, star_type, addstring)
                        i_array = np.array([])
                        wE = np.array([])
                        wP = np.array([])
                        wA = np.array([])
                        wG = np.array([])
                        pattern = '{}temp/*_{}_{}_{}_{}{}.txt'.format(output_dir, galaxy_type, field, case, star_type, addstring)
                        for file_name in glob.glob(pattern):
                            data = np.loadtxt(file_name, delimiter='\t')
                            # i_array = i_array.append(data[:, 0].astype(int))
                            try:
                                i_array = np.append(i_array, data[:, 0])
                                wE = np.append(wE, data[:, 1])
                                wP = np.append(wP, data[:, 2])
                                wA = np.append(wA, data[:, 3])
                                wG = np.append(wG, data[:, 4])
                            except IndexError:
                                pass
                        
                        with open(output_dir + 'all.txt', 'a') as file:
                            # print('writing')
                            file.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(row, len(i_array), np.float32(np.sum(wE)), np.float32(np.sum(wP)), np.float32(np.sum(wA)), np.float32(np.sum(wG))))


def create_16_feathers():
    arr_size = 2000
    for star_type in ['pulsar', 'magnetar']:
        for galaxy_type in ['simple', 'two_phase']:
            for field in ['CF', 'ED']:
                for case in ['A', 'B', 'C', 'D']:
                    for add_string in ['', '_roman']:
                        for rand_i in range(10):
                            name = '{}_{}_{}_{}_erosita{}_{}'.format(galaxy_type, field,
                                                            case, star_type, add_string, rand_i)
                            
                            # df0 = pd.DataFrame({'R': Rcounts, 'r': rcounts, 'z': zcounts,
                            #                     'v': vcounts, 'T': Tcounts,
                            #                     'f0': f0counts, 'f1': f1counts,
                            #                     'c0': c0counts, 'c1': c1counts})
                            df0 = pd.DataFrame({'R': np.zeros(arr_size),
                                                'R-2': np.zeros(arr_size),
                                                'R-1': np.zeros(arr_size),
                                                'r': np.zeros(arr_size),
                                                'r-2': np.zeros(arr_size),
                                                'r-1': np.zeros(arr_size),
                                                'z': np.zeros(arr_size),
                                                'z-2': np.zeros(arr_size),
                                                'z-1': np.zeros(arr_size),
                                                'v': np.zeros(arr_size),
                                                'v-2': np.zeros(arr_size),
                                                'v-1': np.zeros(arr_size),
                                                'T': np.zeros(arr_size),
                                                'T-2': np.zeros(arr_size),
                                                'T-1': np.zeros(arr_size),
                                                'f0': np.zeros(arr_size),
                                                'f1': np.zeros(arr_size),
                                                'c0': np.zeros(arr_size),
                                                'c1': np.zeros(arr_size)
                                            })
                            float_cols = df0.select_dtypes(include=['float64']).columns
                            df0[float_cols] = df0[float_cols].astype('float32')
                            
                            feather.write_feather(df0, output_dir + name + '.feather')
     
    for star_type in ['pulsar', 'magnetar']:
        for galaxy_type in ['simple', 'two_phase']:
            for field in ['CF', 'ED']:
                for case in ['A', 'B', 'C', 'D']:
                    for add_string in ['', '_roman']:
                        rand_i = np.random.randint(0, 10)
                        name0 = '{}_{}_{}_{}_erosita{}_*'.format(galaxy_type, field,
                                                        case, star_type, add_string)
                        pattern = output_dir + 'feather/*_' + name0 + '.feather'
                        for file_name in glob.glob(pattern):
                            rand_i = np.random.randint(0, 10)
                            name = '{}_{}_{}_{}_erosita{}_{}'.format(galaxy_type, field,
                                                        case, star_type, add_string, rand_i)
                            df0 = pd.read_feather(output_dir + name + '.feather')
                            df = pd.read_feather(file_name)
                            feather.write_feather(df+df0, output_dir + name + '.feather')

create_one_txt()
create_16_feathers()
