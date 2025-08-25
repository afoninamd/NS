#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 11:57:02 2025

@author: afoninamd
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from main.evolution import gett, evolution, getB, gettAttractor
from tqdm import tqdm
from main.constants import rgal, mw


def B_pulsar(N: int) -> np.array:
    """ https://ui.adsabs.harvard.edu/abs/2022MNRAS.514.4606I/abstract """
    B_mean = 12.44
    B_sigma = 0.44
    B_power = np.random.normal(loc=B_mean, scale=B_sigma, size=N)
    return 10**B_power


def B_magnetar(N: int) -> np.array:
    """ https://www.physics.mcgill.ca/~pulsar/magnetar/main.html"""
    # use the real data and choose only high-field magnetars B > 1e13 G
    data = pd.read_csv("/home/afoninamd/Documents/NS/data/TabO1.csv")
    B_tab = np.log10(np.array(data['B']))
    B_tab = B_tab[B_tab>13]
    # derive the mean and sigma from the real data
    B_std_m = np.std(B_tab)
    B_mean_m = np.mean(B_tab)
    # create an array of B_power using the derived mean and sigma
    B_power = np.random.normal(loc=B_mean_m, scale=B_std_m, size=N)
    return 10**B_power


def P_pulsar(N: int) -> np.array:
    P_mean = -1.04
    P_sigma = 0.53
    P_power = np.random.normal(loc=P_mean, scale=P_sigma, size=N)
    return 10**P_power


def P_magnetar(P_array: np.array, B_array: np.array) -> np.array:
    """
    Creates an array of P for magnetars with the field value B
    It will be lognormal for fields > 1e14 G (mathematically)
    and lognormal for any field (empirically) with
    mean 1.25 and std 0.26
    """
    leng = len(B_array)
    P_initial = P_array
    P_final = np.zeros(leng)
    for i in tqdm(range(leng)):
        B0 = B_array[i]
        P0 = P_initial[i]
        t = gett(t_end=gettAttractor(B0), n=1000)
        B = getB(t, B0, field='HA')
        t, P, stages = evolution(t=t, P0=P0, B=B,
                      Omega=None, case='A', Mdot=1e-35+t*0,
                      v=1e35+t*0, plot=0)
        stages = np.array(stages)
        
        if len(stages[stages>1]) > 0:
            print('There is transition from the ejector stage')
        P_final[i] = P[-1]
    return P_final


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def V(N: int) -> tuple[np.array, np.array, np.array]:
    """ Arrays of Vx, Vy, Vz in cm/s """
    w = 0.2
    sigma1 = 45e5
    sigma2 = 336e5
    
    Vx = np.zeros(N)
    Vy = np.zeros(N)
    Vz = np.zeros(N)
    
    xi, yi, zi = sample_spherical(N)
    w_cur = np.random.uniform(low=0.0, high=1.0, size=N)
    
    V = sp.stats.maxwell.rvs(loc=0, scale=sigma1, size=N)
    V[w_cur>w] = sp.stats.maxwell.rvs(loc=0, scale=sigma2, size=len(V[w_cur>w]))

    Vx, Vy, Vz = xi*V, yi*V, zi*V
    
    mode = np.zeros(N) + 1
    mode[w_cur>w] = 2
    
    return Vx, Vy, Vz, mode # cm/s


def Vpec(x: np.array, y: np.array, N: np.array,
         rlim=rgal) -> tuple[np.array, np.array, np.array]:
    """ One set of peculiar velocities """
    D = 6e-7 # (km/s)^2 / year
    tau = 0 # year, progenitor star age
    A = 0.7
    c_0 = 10 # km/s
    
    """ Defining sigmas for distributions """
    pot = mw
    r = (x**2 + y**2)**0.5
    dr = rlim / N
    dx = dr / r * x
    dy = dr / r * y
    
    """ Reshaped position array """
    
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
            phi[i] = np.random.uniform(low=0, high=1) * 2 * np.pi
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
    
    v_x, v_y, v_z = Vpec(x, y, N)
    
    return x, y, z, v_x, v_y, v_z # kpc and cm/s


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
        array = np.zeros((4, Nmax))
        x, y, z, mode = V(Nmax)
        array[0] = x
        array[1] = y
        array[2] = z
        array[3] = mode
    else:
        array = f(Nmax)
    
    return array


def distribution(N=100, kind='pulsar'):
    """ Creating arrays of p, B, M, V """
    distrV = distr(V, N)
    distrX = distr(XYZ, N) # XYZ and peculiar VxVyVz
    
    if kind == 'pulsar':
        P_array = distr(P_pulsar, N)
        B_array = distr(B_pulsar, N)
    elif kind == 'magnetar':
        B_array_initial = distr(B_magnetar, N)
        P_array_initial = distr(P_pulsar, N)
        P_array = P_magnetar(P_array_initial, B_array_initial)
        B_array = B_array_initial / np.exp(3) # after the Hall attractor
    
    df = pd.DataFrame({'P': P_array, 'B': B_array,
                       'x': distrX[0], 'y': distrX[1], 'z': distrX[2],
                       'mode': distrV[3],
                       'Vx': distrV[0], 'Vy': distrV[1], 'Vz': distrV[2],
                       'Vxpec': distrX[3], 'Vypec': distrX[4],
                       'Vzpec': distrX[5]})
    if kind == 'magnetar':
        df['P_in'] = P_array_initial
        df['B_in'] = B_array_initial
    pd.DataFrame.to_csv(df, 'distribution_{}.csv'.format(kind), sep=';')


def test_distribution(kind='magnetar'):
    # for kind in ['pulsar', 'magnetar']:
    data = pd.read_csv('distribution_{}.csv'.format(kind), sep=';')
    # Array = (data['Vx']**2 + data['Vy']**2 + data['Vz']**2)**0.5
    Array = data['z']
    # Array = np.log10(Array)
    # print(np.mean(Array), np.std(Array))
    
    # data = pd.read_csv('distribution_pulsar.csv'.format(kind), sep=';')
    # Array = np.append(Array, data['B'])
    # Array = np.log10(Array)
    
    plt.hist(Array, bins=10, alpha=0.5)
    # plt.hist(np.random.normal(loc=np.mean(Array), scale=np.std(Array), size=len(Array)),
    #           30, alpha=0.5)
    # new_array = np.log10(B_magnetar(N=10000))#**(-0.5)
    # plt.hist(np.random.normal(loc=np.mean(new_array), scale=np.std(new_array), size=len(Array)),
    #           30, alpha=0.5)
    # plt.hist(new_array, 30, alpha=0.5)

# fig, ax = plt.subplots()
# test_distribution()

# distribution(N=900000, kind='pulsar')
# distribution(N=100000, kind='magnetar')
# test_distribution(kind='magnetar')
