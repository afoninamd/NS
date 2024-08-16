# -*- coding: utf-8 -*- #
"""
Module for calculating the neutron star evolution

Created on Thu Aug  4 14:51:42 2022
Updated on Mon Mar 13 20:58:16 2023

@author: Marina Afonina

"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from math import pi
import numpy as np
from tqdm import tqdm
# import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from main.constants import rgal, mw, fontsize
# from pathlib import Path
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'
plt.rcParams['text.usetex'] = True
plt.rc('text', usetex=True)
# plt.rc('font', **{'family': 'verdana'})
# plt.rc('text.latex', unicode=True)
# plt.rcParams['text.latex.preamble'] = [r'\usepackage[utf8x]{inputenc}',
#             r'\usepackage[english,russian]{babel}',
#             r'\usepackage{amsmath}']


def XYZ(N=1, rlim=rgal, arms=False, Yao2017=True):
    """ Initial coordinates distribution and pecular velocity distribution """

    """ For eq.(16) from Faucher-Giguere (2006) """
    # a = 1.64
    # b = 4.01
    # R0 = 8.5
    # R1 = 0.55
    """ For eq.(17) from Yusifov (2004) """
    a = 4
    b = 6.8
    R0 = 8.5
    
    def PDF(r, A=1):
        """ eq.(16) from Faucher-Giguere et al. (2006) """
        # rho = A * ((r + R1) / (R0 + R1))**a * np.exp(-b * (r - R0) / (R0 + R1))
        """ eq.(17) from Yusifov et al. (2004) """
        rho = A * (r/R0)**a * np.exp(-b*r/R0)
        return rho
    
    """ integrated PDF """
    # i = 10000
    # sum_rho = np.zeros(i)
    # r = np.linspace(0, 30, i)
    # for j in range(1, i):
    #     sum_rho[j] = sum_rho[j-1] + (r[j] - r[j-1]) * 2 * pi * r[j] * PDF(r[j])
    # plt.plot(r, sum_rho/sum_rho[-1])
    # return 0

    def GenerationR(N):
        r_max = a / b * R0 # maximum of PDF for Yusifov
        # r_max = a / b * (R0 + R1) - R1 # maximum of PDF
        A = np.sum(PDF(np.linspace(0, rlim, N)))
        PDF_max = PDF(r_max, A)
        r = np.zeros(N)
        phi = np.zeros(N)
        
        for i in tqdm(range(N)):
            r[i] = rlim * np.random.uniform(low=0, high=1)**0.5
            phi[i] = np.random.uniform(low=0, high=1) * 2 * pi
            u = np.random.uniform(low=0, high=PDF_max)
            while PDF(r[i], A) < u:
                r[i] = rlim * np.random.uniform(low=0, high=1)**0.5
                u = np.random.uniform(low=0, high=PDF_max)
        return r, phi
    
    if not arms:
        r, phi = GenerationR(N)
        # phi = np.random.uniform(low=0, high=2*pi, size=N)
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
        r, phi = GenerationR(N)
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
    
    def Vpec(x, y):
        """ One set of peculiar velocities """
        D = 6e-7 # (km/s)^2 / year
        tau = 0 # year, progenitor star age
        A = 0.7
        c_0 = 10 # km/s
        
        """ Defining sigmas for distributions """
        pot = mw
        r = (x**2 + y**2 + z**2)**0.5
        dr = rlim / N
        dx = dr / r * x
        dy = dr / r * y
        
        """ Reshaped position array """
        # pos_1 = np.zeros([N, 3])
        # pos_2 = np.zeros([N, 3])
        # pos_1[::, 0] = x + dx
        # pos_1[::, 1] = y + dy
        # pos_1[::, 2] = 0
        # pos_2[::, 0] = x - dx
        # pos_2[::, 1] = y - dy
        # pos_2[::, 2] = 0
        
        pos_1 = np.array([x+dx, y+dy, np.zeros(N)])
        pos_2 = np.array([x-dx, y-dy, np.zeros(N)])
        
        cvel_1 = pot.circular_velocity(pos_1)[0].value 
        cvel_2 = pot.circular_velocity(pos_2)[0].value
        Omega = pot.circular_velocity([x, y, np.zeros(N)])[0].value
        diffOmega = (cvel_1 - cvel_2) / (2 * dr) / r
        kappa = 2 * Omega * (1 + r / (2 * Omega) * diffOmega)**0.5
        B = kappa / (2 * Omega)
        c_r = (c_0**2 + D * tau) / (1 + A**2 + B**2)
        c_phi = B * c_r
        c_z = A * c_r
        
        """ Velocities generation """
        v_r = np.random.normal(0, scale=c_r)
        v_phi = np.random.normal(0, scale=c_phi)
        v_z = np.random.normal(0, scale=c_z)
        
        """ To cartesian coordiates """
        sin_phi = y / r
        cos_phi = x / r
        v_x = v_r * cos_phi - v_phi * sin_phi
        v_y = v_r * sin_phi + v_phi * cos_phi
        
        """ To cm/s """
        v_x = v_x * 1e5
        v_y = v_y * 1e5
        v_z = v_z * 1e5
        
        return v_x, v_y, v_z
    
    v_x, v_y, v_z = Vpec(x, y)
    
    return x, y, z, v_x, v_y, v_z # kpc and cm/s


def V(N):
    """ Arrays of Vx, Vy, Vz in cm/s """
    w = 0.1
    sigma1 = 45e5
    sigma2 = 336e5
    Vmin = 150e5

    V = np.zeros(N) + 2 * Vmin # to make it larger than Vmin initially
    Vx = np.zeros(N)
    Vy = np.zeros(N)
    Vz = np.zeros(N)
    
    while len(V[V>Vmin]) > 0:
        size=len(V[V>Vmin])
        
        w_cur = np.random.uniform(low=0.0, high=1.0, size=size)
        sigma = np.zeros(size)
        sigma[w_cur<w] = sigma1
        sigma[w_cur>w] = sigma2
        a = (pi * sigma**2 / (3 * pi - 8))**0.5
        
        Vx[V>Vmin] = np.random.normal(loc=0.0, scale=a, size=size)
        Vy[V>Vmin] = np.random.normal(loc=0.0, scale=a, size=size)
        Vz[V>Vmin] = np.random.normal(loc=0.0, scale=a, size=size)
        V = (Vx**2 + Vy**2 + Vz**2)**0.5
    
    return Vx, Vy, Vz # cm/s


def B():
    B_mean = np.log(10**13.25)
    B_sigma = 0.6
    B = np.random.lognormal(mean=B_mean, sigma=B_sigma, size=None)
    return B


def p():
    p_mean = -1.04
    p_sigma = 0.53
    p_power = np.random.normal(loc=p_mean, scale=p_sigma, size=None)
    p = 10**p_power
    return p


def M():
    """ [10**(-5 + M_fb_ini_power)] = 1 solar mass / second """
    M_fb_ini_power = np.random.uniform(low=0.0, high=5.0)
    return (-10 + M_fb_ini_power)


def distr(f, Nmax):
    """ Generates distribution of the value from the function f """
    print(str(f)[10] + " distribution")
    if f == XYZ:
        x, y, z, v_x, v_y, v_z = XYZ(Nmax)
        array = np.zeros((6, Nmax))
        array[0] = x
        array[1] = y
        array[2] = z
        array[3] = v_x
        array[4] = v_y
        array[5] = v_z
    elif f == V:
        array = np.zeros((3, Nmax))
        x, y, z = V(Nmax)
        array[0] = x
        array[1] = y
        array[2] = z
        # for i in tqdm(range(Nmax)):
        #     x, y, z = f()
        #     array[0][i] = x
        #     array[1][i] = y
        #     array[2][i] = z
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
    distrX = distr(XYZ, N) # XYZ and peculiar VxVyVz 
    df = pd.DataFrame({'p': distr(p, N), 'B': distr(B, N), 'M': distr(M, N),
                       'x': distrX[0], 'y': distrX[1], 'z': distrX[2],
                       'Vx': distrV[0], 'Vy': distrV[1], 'Vz': distrV[2],
                       'Vxpec': distrX[3], 'Vypec': distrX[4],
                       'Vzpec': distrX[5]})
    pd.DataFrame.to_csv(df, 'data//distribution.csv', sep=';')


# Eijk()
InitialDistribution()
# plotXYZ()

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

