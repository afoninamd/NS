#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:01:23 2024

@author: afoninamd
"""

import matplotlib.pyplot as plt
import numpy as np
import gala.potential as gp
from gala.units import galactic
import astropy.units as u
from tqdm import tqdm
from astropy.io import fits
from constants import m_p, pc, rgal
from constants import fontsize
import scipy as sp
import scipy.interpolate
import pandas as pd

def GetPotential(log_M_h, log_r_s, log_M_n, log_a):
    """ Custom Milky Way potential """
    mw_potential = gp.CCompositePotential()
    # mw_potential['bulge'] = gp.HernquistPotential(m=5E9, c=1., units=galactic)
    mw_potential['disk'] = gp.MiyamotoNagaiPotential(m=0.1*6.8E10*u.Msun,
                                                      a=3*u.kpc, b=280*u.pc,
                                                      units=galactic)
    # mw_potential['nucl'] = gp.HernquistPotential(m=np.exp(log_M_n),
    #                                                c=np.exp(log_a)*u.pc,
    #                                                units=galactic)
    # mw_potential['halo'] = gp.NFWPotential(m=np.exp(log_M_h),
    #       r_s=np.exp(log_r_s), units=galactic)
    return mw_potential


def RealisticDensityMapCreating():
    """ Creates map.fits from MWHI.fits and MWH2.fits """
    
    def CombineArrays(wide, small):
        """ Wide grid = 2 * small grid, new grid is small """
        l = len(wide)

        a = np.zeros(2*l-1)
        b = np.zeros(2*l-1)
        
        for i in range(l):
            a[2*i] += wide[i]
        
        for i in range(1, 2*l-1, 2):
            j = int(i/2)
            a[i] += 0.5 * (wide[j] + wide[j+1])
        
        b[l//2 : l+l//2] = small
        
        return a + b


    def CombineArraysLargeStep(wide, small):
        """ Wide grid = 2 * small grid, new grid is wide """
        a = wide
        b = small
        i2 = int(len(a)/2) # center
        i4 = int(len(a)/4) # max number from center if i_center = 0
        for i in range(1, i4):
            for j in [-i, i]:
                a[i2+j] += b[i2+2*j] + 0.5 * (b[i2+2*j-1] + b[i2+2*j+1])
        for j in [-i4, i4]:
            a[i2+j] += b[i2+2*j] + 0.5 * b[i2-2*j]
        a[i2] += b[i2] + 0.5 * (b[i2-1] + b[i2+1])
        return a
    
    """ Nakanishi data """
    
    with fits.open('data//MWHI.fits') as hdul:
        data1 = hdul[0].data[::,::,::]
        data1 = np.flip(data1, axis=0) # to make CRVAL3 < 0
        print(hdul[0].header)
    
    with fits.open('data//MWH2.fits') as hdul:
        data2 = 2 * hdul[0].data[::,::,::]
        print(hdul[0].header)
    
    z = np.shape(data1)[0]
    x = np.shape(data1)[1]
    y = np.shape(data1)[2]
    data = np.zeros((2*z-1, x, y))
    
    for k in tqdm(range(x)):
        for l in range(y):
            data[::,k,l] = CombineArrays(data1[::,k,l], data2[::,k,l])

    """ Center coordinates """
    x0 = int(np.shape(data)[1] / 2)
    y0 = int(np.shape(data)[2] / 2)
    
    """ Replacing zeros """
    f = data[::,::,::] # in kpc
    for z in tqdm(range(np.shape(data)[0])):
        for x in range(np.shape(data)[1]):
            for y in range(np.shape(data)[2]):
                if not f[z, x, y]:
                    fsum = 0
                    for xn in [x, 2*x0-x]:
                        for yn in [y, 2*y0-y]:
                            f_cur = f[z, xn, yn]
                            if f_cur:
                                data[z, x, y] += f_cur
                                fsum += 1
                    for yn in [x, 2*x0-x]:
                        for xn in [y, 2*y0-y]:
                            f_cur = f[z, xn, yn]
                            if f_cur:
                                data[z, x, y] += f_cur
                                fsum += 1
                    data[z, x, y] = data[z, x, y] / fsum
                    
                    # for f_cur in [f[z, 2*x0-x, y]]
                    #[f[z, x, 2*y0-y], f[z, y, x], f[z, 2*y0-y, x], f[z, -x, y]]:
    
    """ Creating new fits file """
    with fits.open('data//map.fits', mode='update') as hdul:
        hdul[0].data = data
    return 0


def RealisticDensityMapPlot(HI=True, H2=True, logscale=False):
    """ Shows a figure from  MWHI.fits and MWH2.fits """
    
    name = "map"
    title = "HI and H$_2$"
    if HI and not H2:
        name = "MWHI"
        title = "HI only"
    if H2 and not HI:
        name = "MWH2"
        title = "H$_2$ only"
    with fits.open(f"data//{name}.fits") as hdul:
        print(hdul[0].header)
        data = hdul[0].data[::,::,::]

    """ Setting a plot """
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.85], aspect=1)
    ax.set_title(title, fontsize=fontsize+6, verticalalignment='bottom')
    ax.set_xlabel("kpc", fontsize=fontsize)
    ax.set_ylabel("kpc", fontsize=fontsize)

    X = np.linspace(-20, 20, num=np.shape(data)[1]) # [kpc]
    Y = np.linspace(-20, 20, num=np.shape(data)[2]) # [kpc]
    Z = np.linspace(-2, 2, num=np.shape(data)[0]) # [kpc]

    # h1 = data1[::,0,0]
    # h2 = data2[::,0,0]
    # for i in range(1, int(len(X)/2-10)):
    #     # h1 += data1[::,i,i]
    #     # h2 += data2[::,i,i]
    #     plt.plot(Z, (data1[::,i,i]), 'red')
    #     plt.plot((Z/2), np.flip(data2[::,i,i]), 'blue')
    # for i in range(int(len(X)/2 + 10), len(X)):
    #     plt.plot(Z, (data1[::,i,i]), 'red')
    #     plt.plot((Z/2), np.flip(data2[::,i,i]), 'blue')
    
    NumberDensity = data[0,::,::]
    for i in range(1, len(data[::,0,0])):
        NumberDensity += data[i,::,::]
    NumberDensity = NumberDensity #* m_p * pc**2 / M_sun * pc
    NumberDensity = NumberDensity * 4000 * pc / np.shape(data)[0] # h = 4000 pc
    
    if logscale:
        NumberDensity = np.log10(NumberDensity)

    pcm = ax.pcolormesh(X, Y, (NumberDensity), cmap='Greys') # np.log10
    fig.colorbar(pcm, ax=ax)
    
    # print("mean density:", np.mean(NumberDensity) * m_p, "$g/cm^{-2}$")
    X, Y = np.meshgrid(X, Y)
    contours = np.array([0.25, 0.5, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 7, 8.0, 10,
                         11.3, 12, 13, 14, 16.0, 17, 20])
    ax.contour(X, Y, NumberDensity, np.mean(NumberDensity) * contours,
               colors='black') #np.meshgrid(X, Y), 
    ax.tick_params(which='major', width=1.0, length=10, labelsize=fontsize)
    ax.tick_params(which='minor', width=1.0, length=5, labelsize=fontsize-2,
               labelcolor='0.25')


def DensityMap(x, y, z, realisticMap=False):
    
    """ Density from coordinates from Misiriotis (2006) astro.ph/0607638 """
    R = (x**2 + y**2)**0.5
    
    """ Table 2 """
    nw0 = 1.22e-27 / m_p  # cm^-3
    hw  = 3.3             # kpc
    zw  = 0.09            # kpc
    nc0 = 1.51e-25 / m_p
    hc  = 5.0
    zc  = 0.1
    
    """ Table 4 """
    n20 = 4.06 * 2
    h2  = 2.57
    z2  = 0.08
    n10 = 0.32
    h1  = 18.24
    z1  = 0.52
    Rt  = 2.75
    
    """ The warm dust component """
    nw = nw0 * np.exp(-R / hw - abs(z) / zw)
    
    """ The cold dust component """
    nc = nc0 * np.exp(-R / hc - abs(z) / zc)
    
    if not realisticMap:

        """ The molecular hydrogen """
        n2 = n20 * np.exp(-R / h2 - abs(z) / z2)
        
        """ The atomic hydrogen """
        if np.isscalar(R):
            if R > Rt:
                n1 = n10 * np.exp(-R / h1 - abs(z) / z1)
            else:
                n1 = 0
        else:
            n1 = np.zeros(len(R))
            n1[R > Rt] = n10 * np.exp(-R[R > Rt] / h1 - abs(z[R > Rt]) / z1)

        n = nw + nc + n2 + n1
    
    if realisticMap:
        """ The molecular and atomic hydrogen from the map """
        
        with fits.open('data//map.fits', mode='update') as hdul:
            data = hdul[0].data[::,::,::]
        
        X = np.linspace(-20, 20, num=np.shape(data)[1]) # [kpc]
        Y = np.linspace(-20, 20, num=np.shape(data)[2]) # [kpc]
        Z = np.linspace(-2, 2, num=np.shape(data)[0]) # [kpc]
        
        points = (Z, X, Y)
        values = data
        point = (z, x, y)
        
        n = sp.interpolate.interpn(points, values, point, method='linear',
                                   bounds_error=False, fill_value=0.)
        
        n = nw + nc + n

    return n


def DensityMapPlot():
    """ Shows a figure of Misiriotis et al. (2006) density map """
    
    step = 40 # pc
    num  = 2 * rgal * 1000 // step
    
    """ Setting a plot """
    title = "Misiriotis et al. (2006)"
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.85], aspect=1)
    ax.set_title(title, fontsize=fontsize+6, verticalalignment='bottom')
    ax.set_xlabel("kpc", fontsize=fontsize)
    ax.set_ylabel("kpc", fontsize=fontsize)

    X = np.linspace(-rgal, rgal, num=num) # [kpc]
    Y = np.linspace(-rgal, rgal, num=num) # [kpc]
    Z = np.linspace(-10, 10, num=num) # [kpc]
    Z = np.array([2.])
    
    numberDensity = np.zeros([num, num, num])
    sumDensity = np.zeros([num, num])
    for i in tqdm(range(len(Z))):
        for j in range(len(X)):
            for k in range(len(Y)):
                z = Z[i]
                x = X[j]
                y = Y[k]
                numberDensity[i, j, k] = DensityMap(x, y, z)
        sumDensity += numberDensity[i,::,::]

    numberDensity = sumDensity
    # numberDensity = numberDensity # * m_p * pc**2 / M_sun * pc
    # numberDensity = numberDensity * np.max(Z) * 1000 * pc / num # h = 4000 pc
    
    pcm = ax.pcolormesh(X, Y, numberDensity, cmap='Greys') # np.log10
    fig.colorbar(pcm, ax=ax)
    
    # print("mean density:", np.mean(NumberDensity) * m_p, "$g/cm^{-2}$")
    X, Y = np.meshgrid(X, Y)
    contours = np.array([0.25, 0.5, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 7, 8.0, 10,
                         11.3, 12, 13, 14, 16.0, 17, 20])
    ax.contour(X, Y, numberDensity, np.mean(numberDensity) * contours,
               colors='black') #np.meshgrid(X, Y), 
    ax.tick_params(which='major', width=1.0, length=10, labelsize=fontsize)
    ax.tick_params(which='minor', width=1.0, length=5, labelsize=fontsize-2,
               labelcolor='0.25')


def DensityMap2DPlot():
    
    df = pd.read_csv('data//map.csv')
    Z = np.array(df['0'])
    num = int(len(Z)**0.5)
    X = np.linspace(-rgal, rgal, num=num) # [kpc]
    Y = np.linspace(-rgal, rgal, num=num) # [kpc]
    NumberDensity = np.reshape(Z, [num, num])
    
    """ Plotting """
    def PlottingSlice():
        f = NumberDensity[::][num // 2]
        x = X[0:len(f)]
        plt.plot(x, f, 'white')
        plt.xlabel('distance, kpc')
        plt.ylabel('$\log n (cm^{-3})$')
    
    def PlottingMap():
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_axes([0.1, 0.1, 0.9, 0.9], aspect=1)
        # fig, ax = plt.subplots(facecolor='white', layout='constrained')
        pcm = ax.pcolormesh(X, Y, NumberDensity)
        fig.colorbar(pcm, ax=ax)
        ax.set_title("Density Map", fontsize=20, verticalalignment='bottom')
        ax.set_xlabel("kpc", fontsize=14)
        ax.set_ylabel("kpc", fontsize=14)
    
    PlottingMap()
    PlottingSlice()
