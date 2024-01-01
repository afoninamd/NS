# -*- coding: utf-8 -*-
"""
Created on Thu May 11 22:17:08 2023

@author: AMD
"""
from track import Track, TrackPlot
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
from joblib import Parallel, delayed
import multiprocessing
from constants import birth_age, galaxy_age
import os.path

warnings.filterwarnings("ignore")
n_cores = multiprocessing.cpu_count()

def Popsynthesis(N=1, field="CF"):
    
    data = pd.read_csv('distribution.csv', sep=';')
    p = data['p']
    B = data['B']
    # M = data['M']
    Vx = data['Vx']
    Vy = data['Vy']
    Vz = data['Vz']
    x = data['x']
    y = data['y']
    z = data['z']
    V = (Vx**2 + Vy**2 + Vz**2)**0.5
    
    """ Each stage duration """
    A = 0
    P = 0
    E = 0
    G = 0
    number_All = 0
    
    A_Array = np.array([])
    P_Array = np.array([])
    E_Array = np.array([])
    G_Array = np.array([])
    p_Array = np.array([])
    B_Array = np.array([])
    V_Array = np.array([])
    
    start = np.random.randint(0, high=len(p)-N, size=None, dtype=int)
    
    def func_for_popsynthesis(i):
        
        vel0 = np.array([Vx[i], Vy[i], Vz[i]]) * 1e-5
        pos0 = np.array([x[i], y[i], z[i]])
        nonlocal p
        nonlocal B
        popt = Track(10**p[i], B[i], pos0=pos0, vel0=vel0, field=field,
                      t_start=birth_age, t_end=galaxy_age, plot=False,
                      Misiriotis=True)
        T, p_res, B_res, v_res, n_res, t_stages, stages, NS_0, duration = popt
        A, P, E, G = duration
        try:
            return A, P, E, G, p[i], B[i], V[i]
        except BaseException as e:
            print(e)
            return 0, 0, 0, 0, 0, 0, 0

    res = Parallel(n_jobs=n_cores)(delayed(func_for_popsynthesis)(i)
                                   for i in tqdm(range(start, start + N)))
    res = np.array(res)
    A_Array = np.append(A_Array, res[:, 0])
    P_Array = np.append(P_Array, res[:, 1])
    E_Array = np.append(E_Array, res[:, 2])
    G_Array = np.append(G_Array, res[:, 3])
    p_Array = np.append(p_Array, res[:, 4])
    B_Array = np.append(B_Array, res[:, 5])
    V_Array = np.append(V_Array, res[:, 6])
    print(p_Array)
    A = sum(A_Array)
    P = sum(P_Array)
    E = sum(E_Array)
    G = sum(G_Array)
    
    number_All = len(B_Array[B_Array != 0])
    if number_All == 0:
        number_All = 1
    print(f'\nnumber of tracks: {number_All}')
    print("\n Accretor Propeller Ejector Georotator\n",
          A / number_All, P / number_All,
          E / number_All, G / number_All)
    
    name = 'f'+str(N) + str(field)
    
    if os.path.isfile('results/popsynthesis.csv'):
        df = pd.DataFrame([[name, N, number_All, field,
                            A / number_All, P / number_All,
                            E / number_All, G / number_All]])
        header = None
    
    else:
        df = pd.DataFrame({'name': [name], 'N': [N],
                           'trackNumber': [number_All], 'field': [field],
                           'A': [A / number_All], 'P': [P / number_All],
                           'E': [E / number_All], 'G': [G / number_All]})
        header = True
    
    pd.DataFrame.to_csv(df, 'results/popsynthesis.csv', sep=';',
                        mode='a', columns=None, header=header)
    df = pd.DataFrame({'p': p_Array, 'B': B_Array, 'V': V_Array,
                       'A': A_Array, 'P': P_Array, 'E': E_Array, 'G': G_Array})

    pd.DataFrame.to_csv(df, 'results/tracks/'+name+'.csv', sep=';', mode='w',
                        columns=None)
    
    return p_Array, B_Array, V_Array

N = 20
Popsynthesis(N, field="CF")

""" Old popsynthesis """

# N = 200000
# for i in [0,1,2,3]:
#     Popsynthesis(N, fallback=False, hysteresis=True, field=i)
# N = 2 * N
# for i in [0,1,2,3]:
#     Popsynthesis(N, fallback=True, hysteresis=True, field=i)

# N=200000
# for field in [0,1,2,3]:
#     dots_num = 11
#     # field = 3
#     for j in range(dots_num):
#         powerM = -10 + 5 * j / (dots_num-1)
#         M_cur = 10**powerM*M_sun
#         A = 0
#         P = 0
#         E = 0
#         G = 0
#         number_All = 0
        
#         A_Array = np.array([])
#         P_Array = np.array([])
#         E_Array = np.array([])
#         G_Array = np.array([])
#         p_Array = np.array([])
#         B_Array = np.array([])
#         M_Array = np.array([])
#         V_Array = np.array([])
#         t_A_Array = np.array([])
#         num_Array = np.array([])
       
#         start = np.random.randint(0, high=len(p)-N, size=None, dtype=int)
#         def func_for_popsynthesis(i):
#             if V[i] > 1.5e7:
#                 # A, P, E, G, text, last_stage, t_Accretor = 0, 0, 0, 0, 'Velocity', 'E', 0.0
#                 """ A, P, E, G, p, B, M, V, t_A, num """
#                 return 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
#             A, P, E, G, text, last_stage, t_Accretor = Track(10**p[i], 10**B[i],
#                                                   M_cur, V[i],
#                                                   fallback=True,
#                                                   hysteresis=True,
#                                                   field=field)
#             return A, P, E, G, p[i], B[i], M[i], V[i], t_Accretor, 1
#         res = Parallel(n_jobs=n_cores)(delayed(func_for_popsynthesis)(i) for i in tqdm(range(start, start + N)))
#         res = np.array(res)
#         A_Array = np.append(A_Array, res[:, 0])
#         P_Array = np.append(P_Array, res[:, 1])
#         E_Array = np.append(E_Array, res[:, 2])
#         G_Array = np.append(G_Array, res[:, 3])    
#         p_Array = np.append(p_Array, res[:, 4])
#         B_Array = np.append(B_Array, res[:, 5])
#         M_Array = np.append(M_Array, res[:, 6])
#         V_Array = np.append(V_Array, res[:, 7])
#         t_A_Array = np.append(t_A_Array, res[:, 8])
#         num_Array = np.append(num_Array, res[:, 9])

#         A = sum(A_Array)
#         P = sum(P_Array)
#         E = sum(E_Array)
#         G = sum(G_Array)
#         number_All = len(B_Array[B_Array != 0])
#         if number_All == 0:
#             number_All = 1
#         # number_of_tracks = len(B_Array[B_Array != 0])
#         print(f'\nnumber of tracks: {number_All}')
#         print("\n Accretor Propeller Ejector Georotator\n",
#               A / number_All, P / number_All,
#               E / number_All, G / number_All)
        
#         df = pd.DataFrame([[N, number_All, True, True, field,
#                             A / number_All, P / number_All,
#                             E / number_All, G / number_All, powerM]])
        
#         pd.DataFrame.to_csv(df, 'popsynthesis\\graphM_popsynthesis.csv', sep=';', mode='a', columns=None,
#                             header=None)
#         df = pd.DataFrame({'p': p_Array, 'B': B_Array, 'M': M_Array,
#                               'V': V_Array, 'A': t_A_Array})
#         name = 'f'+str(field)
#         name = name+'fallback'
#         name = 'popsynthesis\\'+name+'forGraphM.csv'
#         pd.DataFrame.to_csv(df, name, sep=';', mode='w', columns=None)



# def func_for_M():
#     N = 200000
#     dots_num = 11
#     M_initial_Array = np.array([])
#     Aarr = np.array([])
#     for j in range(dots_num):
#         powerM = -10 + 5 * i / (dots_num-1)
#         def func_for_M(i):
            
#     res = Parallel(n_jobs=n_cores)(delayed(func_for_M)(i) for i in range(dots_num))
#     res = np.array(res)
#     M_initial_Array = np.append(M_initial_Array, res[:, 0])
#     A, P, E, G = Popsynthesis(N, fallback=True, hysteresis=True, field=0, power_M=powerM)
#     Aarr = np.append(Aarr, A)



