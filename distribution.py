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
from constants import rgal

def XYZ(N=1, rlim=rgal, arms=True, Yao2017=True):
    """ Initial coordinates distribution, returns arrays of x, y, z """

    """ For eq.(16) from Faucher-Giguere (2006) """
    a = 1.64
    b = 4.01
    R0 = 8.5 
    R1 = 0.55
    
    def PDF(r, A=1):
        """ eq.(16) from Faucher-Giguere (2006) """
        rho = A * ((r + R1) / (R0 + R1))**a * np.exp(-b * (r - R0) / (R0 + R1))
        return rho
    
    def GenerationR(N):
        r_max = a / b * (R0 + R1) - R1 # maximum of PDF
        A = np.sum(PDF(np.linspace(0, rlim, N)))
        PDF_max = PDF(r_max, A)
        r = np.zeros(N)
        
        for i in tqdm(range(N)):
            r[i] = np.random.uniform(low=0, high=rlim)
            u = np.random.uniform(low=0, high=PDF_max)
            while PDF(r[i], A) < u:
                r[i] = np.random.uniform(low=0, high=rlim)
                u = np.random.uniform(low=0, high=PDF_max)
        return r
    
    if not arms:
        r = GenerationR(N)
        phi = np.random.uniform(low=0, high=2*pi, size=N)
    else:
        def SpiralArm(r, k, r0, Teta0):
            """ Clean spiral arm """
            Teta = k * np.log(r / r0) + Teta0
            """ Blurring """
            if np.isscalar(r):
                size = 1
            else:
                size = len(r)
            Teta_corr = np.random.uniform(low=0, high=2*np.pi, size=size)
            Teta = Teta + Teta_corr * np.exp(-0.35 * r)# r in kpc
            return Teta
        
        if Yao2017:
            """ Table 1 from Yao et al. (2017) """
            R = np.array([3.35, 3.71, 3.56, 3.67, 8.21]) # kpc
            phi = np.array([44.4, 120.0, 218.6, 330.3, 55.1]) # deg
            psi = np.array([11.43, 9.84, 10.38, 10.54, 2.77]) # deg
            """ to Faucher-Giguere form """
            k = 1 / np.tan(psi / 180 * np.pi)
            r0 = R
            Teta0 = phi
        else:
            """ Table 2 from Faucher-Giguere (2006) """
            k = np.array([4.25, 4.25, 4.89, 4.89]) # rad
            r0 = np.array([3.48, 3.48, 4.90, 4.90]) # kpc
            Teta0 = np.array([1.57, 4.71, 4.09, 0.95]) # rad
        
        """ Generation of r and phi within the spiral arms """
        N_arm = np.random.randint(0, high=len(k), size=N, dtype=int)
        r = GenerationR(N)
        phi = np.zeros(N)
        for i in range(len(k)):
            phi[N_arm==i] = SpiralArm(r[N_arm==i], k[i], r0[i], Teta0[i])

        """ Blur of r """
        r = r + np.random.normal(loc=0, scale=0.07*r)

    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y
    
    x, y = pol2cart(r, phi)
    
    """ z distribution from Faucher-Giguere (2006) """
    z0 = 50 / 1000 # kpc
    z = np.random.exponential(scale=z0, size=N)
    sign = np.random.randint(0, high=2, size=N)
    z[sign>0] = -z[sign>0] # half of z should be negative
    
    return x, y, z # kpc


def V():
    """ One set of Vx, Vy, Vz """
    w = 0.1
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
    """ Generates distribution of the value from the function f """
    print(str(f)[10] + " distribution")
    if f == XYZ:
        x, y, z = XYZ(Nmax)
        array = np.zeros((3, Nmax))
        array[0] = x
        array[1] = y
        array[2] = z
    elif f == V:
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


def InitialDistribution():
    """ Creating arrays of p, B, M, V """
    N = 1_000_00
    
    """ V array creation is long, so if you want to skip it... """
    # data = pd.read_csv('data//distribution.csv', sep=';')
    # Vx = data['Vx']
    # Vy = data['Vy']
    # Vz = data['Vz']
    # distrV = np.array([Vx, Vy, Vz])
    
    distrV = distr(V, N)
    distrX = distr(XYZ, N)
    df = pd.DataFrame({'p': distr(p, N), 'B': distr(B, N), 'M': distr(M, N),
                        'x': distrX[0], 'y': distrX[1], 'z': distrX[2],
                        'Vx': distrV[0], 'Vy': distrV[1], 'Vz': distrV[2]})
    pd.DataFrame.to_csv(df, "data//distribution.csv", sep=';')


def DistributionTheory(file_name="data//distribution.csv"):
    data = pd.read_csv(file_name, sep=';')
    p = data['p']
    B = data['B']
    M = data['M']
    Vx = data['Vx']
    Vy = data['Vy']
    Vz = data['Vz']
    x = data['x']
    y = data['y']
    z = data['z']
    plt.plot(x, y, 'ko', markersize=0.2)
    # plt.hist(z,bins=100)
    # V = (Vx**2 + Vy**2 + Vz**2)**0.5
    # plt.hist((x**2+y**2)**0.5, bins=100)
    
    
# InitialDistribution()
# DistributionTheory()

""" Test """
# Array = np.zeros(N)
# # for i in tqdm(range(N)):
# #     Array[i] = B()+5
# N = 10000
# res = distr(V, N)
# Array = (res[0]**2 + res[1]**2 + res[2]**2)**0.5


""" Illustration """
# count, bins, ignored = plt.hist(Array, 100, density=True, align='mid')

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

