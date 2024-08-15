# -*- coding: utf-8 -*-
"""
Created on Thu May 11 22:17:08 2023

@author: AMD
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from track import Track, TrackPlot
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
from joblib import Parallel, delayed
import multiprocessing
from main.constants import birth_age, galaxy_age, year
import os.path

warnings.filterwarnings("ignore")
n_cores = multiprocessing.cpu_count()-1


def PopStages(N=1, field="CF", name="f", start=None):
    
    data = pd.read_csv('data//distribution.csv', sep=';')
    p = data['p']
    B = data['B']
    # M = data['M']
    Vx = data['Vx']
    Vy = data['Vy']
    Vz = data['Vz']
    x = data['x']
    y = data['y']
    z = data['z']
    Vxpec = data['Vxpec']
    Vypec = data['Vypec']
    Vzpec = data['Vzpec']
    V = ((Vx+Vxpec)**2 + (Vy+Vypec)**2 + (Vz+Vzpec)**2)**0.5
    
    if start is None:
        start = np.random.randint(0, high=len(p)-N, size=None, dtype=int)
    
    def func_for_popsynthesis(i):
        
        vel0 = np.array([Vx[i]+Vxpec[i], Vy[i]+Vypec[i], Vz[i]+Vzpec[i]]) * 1e-5
        pos0 = np.array([x[i], y[i], z[i]])
        nonlocal p
        nonlocal B
        try:
            popt = Track(p[i], B[i], pos0=pos0, vel0=vel0, field=field,
                          t_start=birth_age, t_end=galaxy_age,  plotOrbit=False,
                          realisticMap=False)
            T1, P1, B1, v1, n1, t_stages, stages, NS_0, EPAG = popt
            E = EPAG[0]
            P = EPAG[1]
            A = EPAG[2]
            G = EPAG[3]

        except Exception as e:
            pass
        try:
            return E, P, A, G, str(t_stages), str(stages), i
        except BaseException as e:
            print(e)
            return 0, 0, 0, 0, str(['ERROR']), str([0]), i

    res = Parallel(n_jobs=n_cores)(delayed(func_for_popsynthesis)(i)
                                   for i in tqdm(range(start, start + N)))
    res = np.array(res)
    
    E = res[:, 0]
    P = res[:, 1]
    A = res[:, 2]
    G = res[:, 3]
    t_stages = res[:, 4]
    stages = res[:, 5]
    i_Array = res[:, 6]
    
    name = name + str(N) + str(field)
    
    if os.path.isfile('results/{}_stages.csv'.format(name)):
        header = None
    else:
        header = True
    
    df = pd.DataFrame({'E': E, 'P': P, 'A': A, 'G': G,
                       't_stages': t_stages, 'stages': stages, 'i': i_Array})
    
    pd.DataFrame.to_csv(df, 'results/{}_stages.csv'.format(name), sep=';',
                        mode='a', columns=None, header=header)


def iPlot(i, field='CF'):
    data = pd.read_csv('data//distribution.csv', sep=';')
    p = data['p']
    B = data['B']
    # M = data['M']
    Vx = data['Vx']
    Vy = data['Vy']
    Vz = data['Vz']
    x = data['x']
    y = data['y']
    z = data['z']
    Vxpec = data['Vxpec']
    Vypec = data['Vypec']
    Vzpec = data['Vzpec']
    vel0 = np.array([Vx[i]+Vxpec[i], Vy[i]+Vypec[i], Vz[i]+Vzpec[i]]) * 1e-5
    pos0 = np.array([x[i], y[i], z[i]])
    
    TrackPlot(p[i], B[i], pos0=pos0, vel0=vel0, field=field,
                  t_start=birth_age, t_end=galaxy_age,  plotOrbit=True,
                  realisticMap=False)
    


def AverageAccretorAge(N, t_start=5.):
    
    def StrToFloat(t_stages, stages):
        """ Makes an array of floats out of a string """
        def MapString(string):
            string = string.replace('[', '')
            string = string.replace(']', '')
            string = string.replace("'", '')
            string = string.replace("\n", '')
            string = string.split(' ')
            return string
        
        stages = MapString(stages)
        t_stages = MapString(t_stages)
        for i in range(len(t_stages)):
            t_stages[i] = float(t_stages[i])
        
        return t_stages, stages
    name = 'results/f{}CF_stages.csv'.format(N)
    data = pd.read_csv(name, sep=';')
    T_STAGES = data['t_stages']
    STAGES = data['stages']
    Mean = 0
    num = 0
    for i in range(len(STAGES)):
        t = T_STAGES[i]
        st = STAGES[i]
        print(st)
        if not ('ERROR' in t):
            t, st = StrToFloat(t, st)
            
            if t[-1] >= 12e9*year:
                num += 1
                t = np.array(t)
                # t = np.append(t, t_end)
                t_end = t[-1]
                T = t_end - t_start
                parts = 0
                mean = 0
                for j in range(len(st)):
                    if st[j] == 'A':
                        part = (t[j+1]-t[j]) / T
                        parts += part
                        value = (t[j+1]+t[j]) / 2
                        mean += part * value
                if parts:
                    mean = mean / parts
                Mean += mean
    Mean = Mean / num
    print('The mean accretor age is {} Gyr'.format(mean / year / 1e9))
    return mean

# i=0
# iPlot(i)
# for k in range(100):
#     N = 100
#     start = k*N
#     PopStages(N, field="CF", start=start)
N = 100
# PopStages(100, field="CF")
AverageAccretorAge(N)
# p = galaxy_age - 1.01409219e+16
# p = p / galaxy_age
# print(p)
# m = galaxy_age + 1.01409219e+16
# m = m / 2
# print(m*p/1e9/year)
