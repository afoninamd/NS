# -*- coding: utf-8 -*- #
"""
Module for calculating the neutron star evolution

Created on Thu Aug  4 14:51:42 2022
Updated on Mon Mar 13 20:58:16 2023

@author: Marina Afonina

"""
from math import pi
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

def XYZ():
    # !!!
    
    R = np.random.uniform(low=2.0, high=10.0)
    phi = np.random.uniform(low=0, high=2*pi)
    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y
    x, y = pol2cart(R, phi)
    z = 0
    return x, y, z # kpc
    # def rho(r):
        # a = 1.64
        # b = 4.01
        # R0 = 8.5
        # R1 = 0.55
        # """ Probability Function """
        # rho = ((r + R1) / (R0 + R1))**a * np.exp(-b * (r - R0) / (R0 + R1))
        # return r

def V():
    """ Кики изотропные? """
    w = 0.1
    # w = 0.2
    sigma1 = 45e5
    sigma2 = 336e5
    w_cur = np.random.uniform(low=0.0, high=1.0)
    
    if w_cur < w:
        sigma = sigma1
    elif w_cur > w:
        sigma = sigma2
    a = (pi * sigma**2 / (3 * pi - 8))**0.5
    
    """ We are interested only in V < 150 km/s """
    Vmin = 1.5e7
    V = Vmin * 2
    while V > Vmin:
        Vx = np.random.normal(loc=0.0, scale=a, size=None)
        Vy = np.random.normal(loc=0.0, scale=a, size=None)
        Vz = np.random.normal(loc=0.0, scale=a, size=None)
        V = (Vx**2 + Vy**2 + Vz**2)**0.5
    
    return Vx, Vy, Vz


def B():
    B_mean = np.log(10**13.25)
    B_sigma = 0.6
    B_power = np.random.lognormal(mean=B_mean, sigma=B_sigma, size=None)
    return B_power


def p():
    p_mean = -1.04
    p_sigma = 0.53
    p_power = np.random.normal(loc=p_mean, scale=p_sigma, size=None)
    return p_power


def M():
    """ [10**(-5 + M_fb_ini_power)] = 1 solar mass / second """
    M_fb_ini_power = np.random.uniform(low=0.0, high=5.0)
    return (-10 + M_fb_ini_power)


def distr(f, Nmax):
    print(str(f)[10] + " distribution")
    # if f == V:
    #     array = np.zeros((3, Nmax))
    #     for i in tqdm(range(Nmax)):
    #         Vx, Vy, Vz = V()
    #         array[0][i] = Vx
    #         array[1][i] = Vy
    #         array[2][i] = Vz
    #     # print(array)
        
    if f == XYZ or f == V:
        array = np.zeros((3, Nmax))
        for i in tqdm(range(Nmax)):
            x, y, z = f()
            array[0][i] = x
            array[1][i] = y
            array[2][i] = z
    else:
        array = np.zeros(Nmax)
        for i in tqdm(range(Nmax)):
            array[i] = f()
    
    return array


""" Creating arrays of p, B, M, V """
N = 1_000_000
distrV = distr(V, N)
distrX = distr(XYZ, N)
df = pd.DataFrame({'p': distr(p, N), 'B': distr(B, N), 'M': distr(M, N),
                   'x': distrX[0], 'y': distrX[1], 'z': distrX[2],
                   'Vx': distrV[0], 'Vy': distrV[1], 'Vz': distrV[2]})
pd.DataFrame.to_csv(df, 'distribution.csv', sep=';')

""" Test """
Array = np.zeros(N)
# for i in tqdm(range(N)):
#     Array[i] = B()+5
N = 10000
res = distr(V, N)
Array = (res[0]**2 + res[1]**2 + res[2]**2)**0.5


""" Illustration """
count, bins, ignored = plt.hist(Array, 100, density=True, align='mid')
# x = np.linspace(min(bins), max(bins), 10000)
# mu = 0.6
# sigma = 13.25
# pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
#         / (x * sigma * np.sqrt(2 * np.pi)))
# plt.plot(pdf)
#plt.hist(Array, bins=1, range=[0, 8e12])#'auto')#, range=[0, 1e10], density=None, weights=None)


# def PDF(x, sigma):
#     a = (pi * sigma**2 / (3 * pi - 8))**0.5
#     return (2/pi)**0.5 * x**2 * exp(-x**2/(2*a**2)) / a**3

# v_Array = np.geomspace(1, 2000, num=1000)
# p1_Array = PDF(v_Array, 45e5)
# p2_Array = PDF(v_Array, 336e5)
# p_Array = p1_Array * 0.1 + p2_Array * 0.9
# plt.plot(v_Array, p_Array)

