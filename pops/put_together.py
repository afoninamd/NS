#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 18:36:38 2025

@author: afoninamd
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main.constants import output_dir, arr_size
import numpy as np
import glob
import pandas as pd
import pyarrow.feather as feather

def create_txt():
    for addstring in ['', '_roman']:
        for star_type in ['pulsar', 'magnetar']:
            for galaxy_type in ['simple', 'two_phase']:
                for field in ['CF', 'ED']:
                    for case in ['A', 'B', 'C', 'D']:
                        row = '{}_{}_{}_{}{}'.format(galaxy_type, field, case, star_type, addstring)
                        N_files = 100
                        # for rand_i in range(N_files):
                        #     with open(output_dir + 'all_{}.txt'.format(rand_i), 'a') as file:
                        #         file.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format('', '', '', '', '', ''))

                        i_array = np.array([])
                        wE = np.array([])
                        wP = np.array([])
                        wA = np.array([])
                        wG = np.array([])
                        pattern = '{}temp/*_{}_{}_{}_{}{}.txt'.format(output_dir, galaxy_type, field, case, star_type, addstring)
                        for file_name in glob.glob(pattern):
                            # rand_i = np.random.uniform(N_files)
                            # df0 = np.loadtxt(output_dir + 'all_{}.txt'.format(rand_i), delimiter='\t')
                            # i_array = df0[:, 0]
                            # wE = df0[:, 1]
                            # wP = df0[:, 2]
                            # wA = df0[:, 3]
                            # wG = df0[:, 4]
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
                        
                        N1 = len(i_array)
                        vals_per_iteration = N1 // N_files
                        remainder = N1 % N_files
                        
                        i_sum = np.array([])
                        wE_sum = np.array([])
                        wP_sum = np.array([])
                        wA_sum = np.array([])
                        wG_sum = np.array([])
                        for rand_i in range(N_files):
                            # with open(output_dir + 'all_{}.txt'.format(rand_i), 'a') as file:
                                # print('writing')
                                start_idx = rand_i * vals_per_iteration + min(rand_i, remainder)
                                end_idx = start_idx + vals_per_iteration
                                if rand_i < remainder:
                                    end_idx += 1
                                
                                i_sum = np.append(i_sum, len(i_array[start_idx:end_idx]))
                                wE_sum = np.append(wE_sum, np.sum(wE[start_idx:end_idx]))
                                wP_sum = np.append(wP_sum, np.sum(wP[start_idx:end_idx]))
                                wA_sum = np.append(wA_sum, np.sum(wA[start_idx:end_idx]))
                                wG_sum = np.append(wG_sum, np.sum(wG[start_idx:end_idx]))
                        
                        if len(i_sum) > 0:
                            wE_sum = wE_sum / i_sum
                            wP_sum = wP_sum / i_sum
                            wA_sum = wA_sum / i_sum
                            wG_sum = wG_sum / i_sum
                        else:
                            wE_sum = 0
                            wP_sum = 0
                            wA_sum = 0
                            wG_sum = 0
                            
                        with open(output_dir + 'all.txt', 'a') as file:
                            file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(row, len(i_array), np.float32(np.sum(wE)), np.float32(np.sum(wP)), np.float32(np.sum(wA)), np.float32(np.sum(wG)),
                                                                                                         np.float32(np.std(wE_sum)), np.float32(np.std(wP_sum)), np.float32(np.std(wA_sum)), np.float32(np.std(wG_sum))))


def create_feather():
    # for star_type in ['pulsar', 'magnetar']:
    for galaxy_type in ['simple', 'two_phase']:
        for field in ['CF', 'ED']:
            for case in ['A', 'B', 'C', 'D']:
                for add_string in ['', '_roman']:
                    # for rand_i in range(N_files):
                    # name = '{}_{}_{}_{}_erosita{}'.format(galaxy_type, field,
                    #                                 case, star_type, add_string)
                    
                    # df0 = pd.DataFrame({'R': Rcounts, 'r': rcounts, 'z': zcounts,
                    #                     'v': vcounts, 'T': Tcounts,
                    #                     'f0': f0counts, 'f1': f1counts,
                    #                     'c0': c0counts, 'c1': c1counts})
                    
                    df_sum = pd.DataFrame({'R': np.zeros(arr_size),
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
                    
                    N_files = 0
                    
                    name0 = '*{}_{}_{}_pulsar_erosita{}'.format(galaxy_type, field,
                                                    case, add_string)
                    pattern = output_dir + 'feather/*_' + name0 + '.feather'
                    for file_name in glob.glob(pattern):
                        N_files += 1
                        df_pulsar = pd.read_feather(file_name)
                        file_name_magnetar = file_name.replace('pulsar', 'magnetar')
                        df_magnetar = pd.read_feather(file_name_magnetar)
                        df = df_pulsar + df_magnetar
                        df_sum = df_sum + df
            
                    float_cols = df_sum.select_dtypes(include=['float64']).columns
                    df_sum[float_cols] = df_sum[float_cols].astype('float32')
                    
                    name = '{}_{}_{}_erosita{}_{}'.format(galaxy_type, field,
                                                case, add_string, 'sum')
                    feather.write_feather(df_sum, output_dir + name + '.feather')
                    
                    
                    df_mean = df_sum / N_files
                    
                    df_square = pd.DataFrame({'R': np.zeros(arr_size),
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
                    #0_simple_CF_A_magnetar_erosita_roman
                    
                    for file_name in glob.glob(pattern):
                        df_pulsar = pd.read_feather(file_name)
                        file_name_magnetar = file_name.replace('pulsar', 'magnetar')
                        df_magnetar = pd.read_feather(file_name_magnetar)
                        df = df_pulsar + df_magnetar
                        df_temp = df - df_mean
                        df_square = df_square + df_temp.pow(2)
                    
                    df_std = df_square.pow(0.5) / (N_files-1)**0.5 * N_files
                    # print(df_std, N_files)
                    # print(N_files)
                    # if not N_files:
                    #     print(name)
                    
                    float_cols = df_std.select_dtypes(include=['float64']).columns
                    df_std[float_cols] = df_std[float_cols].astype('float32')
                    
                    name = '{}_{}_{}_erosita{}_{}'.format(galaxy_type, field,
                                                case, add_string, 'std')
                    feather.write_feather(df_std, output_dir + name + '.feather')
                        
                        
                        # feather.write_feather(df_mean, output_dir + name + '.feather')


    # # for star_type in ['pulsar', 'magnetar']:
    # #     for galaxy_type in ['simple', 'two_phase']:
    # #         for field in ['CF', 'ED']:
    # #             for case in ['A', 'B', 'C', 'D']:
    # #                 for add_string in ['', '_roman']:
    #                     name0 = '{}_{}_{}_{}_erosita{}_*'.format(galaxy_type, field,
    #                                                     case, star_type, add_string)
    #                     pattern = output_dir + 'feather/*_' + name0 + '.feather'
    #                     # for file_name in glob.glob(pattern):
    #                         rand_i = np.random.randint(0, N_files)
    #                         name = '{}_{}_{}_{}_erosita{}_{}'.format(galaxy_type, field,
    #                                                     case, star_type, add_string, rand_i)
    #                         df0 = pd.read_feather(output_dir + name + '.feather')
    #                         df = pd.read_feather(file_name)
    #                         feather.write_feather(df+df0, output_dir + name + '.feather')



create_txt()
create_feather()



