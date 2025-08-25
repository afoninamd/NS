#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:01:23 2024

@author: afoninamd
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from main.constants import m_p, pc, rgal, c_s_hot, c_s_cold, M_sun
import scipy as sp
import scipy.interpolate
import pandas as pd


def ionized_density(z):
    """ Gaensler 2008 (Exponential Fit To All Data) """
    N0 = 0.014
    N = N0 * np.exp(-abs(z)/1.83) #0.710
    return N


def dust_density(x, y, z):
    """ Density from coordinates from Misiriotis (2006) astro.ph/0607638 """
    R = (x**2 + y**2)**0.5
    
    """ Table 2 """
    nw0 = 1.22e-27 / m_p  # cm^-3
    hw  = 3.3             # kpc
    zw  = 0.09            # kpc
    nc0 = 1.51e-25 / m_p
    hc  = 5.0
    zc  = 0.1
    
    """ The warm dust component """
    nw = nw0 * np.exp(-R / hw - abs(z) / zw)
    
    """ The cold dust component """
    nc = nc0 * np.exp(-R / hc - abs(z) / zc)
    
    return nw + nc


def neutral_density(x, y, z, realisticMap=False, author=None):
    R = (x**2+y**2)**0.5
    """ Table 4 """
    n10 = 0.32
    h1  = 18.24
    z1  = 0.52
    Rt  = 2.75
    
    """ The atomic hydrogen """
    def func_n1(x, y, z):
        n1 = np.zeros(shape=np.shape(x))
        n1[(x**2+y**2+z**2)**0.5>Rt] = n10 * np.exp(-R[(x**2+y**2+z**2)**0.5>Rt] / h1 - abs(z[(x**2+y**2+z**2)**0.5>Rt]) / z1)
        return n1
    n = func_n1(x, y, z)# + ExtraplanarDensity(x, y, z)
    
    if author == 'Yao':
        """ Yao 2017 """
        zw = 0 # ignore the warp
        H1 = 1.673
        def func(z):
            n = 1 / np.cosh(abs(z-zw)/H1)**2
            return n
        # integral = sp.integrate.quad(func, 0, 10)[0]
        # result = 2*np.pi*13**2*integral * m_p*(1e3*pc)**3
        # print("{:.2e}".format(result))
        result = 4.37e+10*M_sun
        result = 8.6e10*M_sun/result
        n = func(z) * result
        
        Bd = 15
        Ad = 2.5
        R = (x**2 + y**2)**0.5
        
        if np.isscalar(R):
            if R > Bd:
                n = n / np.cosh((R-Bd)/Ad)**2
        else:
            n[R>Bd] = n[R>Bd] / np.cosh((R-Bd)/Ad)**2
        # n = n/3**0.5
    elif author == 'Kalberla':
        R0 = 8.5
        Rn = 3.15
        kpc = 1e3*pc
        n0 = 0.9
        
        if np.isscalar(R):
            n_max = 10 * M_sun/m_p / pc**2 * np.exp(abs(z)/z1) / (2*z1*kpc)
            n = n0 * np.exp(abs(z)/z1) * np.exp(-(R-R0)/Rn)
            n = min(n, n_max)
                
        else:
            n = n0 * np.exp(abs(z)/z1) * np.exp(-(R-R0)/Rn) 
            n[n>n_max] = 10 * M_sun/m_p / pc**2 * np.exp(abs(z[n>n_max])/z1) / (2*z1*kpc)

    """ + cut-off from Yao 2017 """
    Bd = 15
    Ad = 2.5
    n[R>Bd] = n[R>Bd] / np.cosh((R[R>Bd]-Bd)/Ad)**2
    return n


def molecular_density(x, y, z):
    """ The molecular hydrogen """
    R = (x**2 + y**2)**0.5

    """ Table 4 """
    n20 = 4.06 * 2
    h2  = 2.57
    z2  = 0.08

    def func_n2(R, z):
        n2 = n20 * np.exp(-R / h2 - abs(z) / z2)
        return n2
    n = func_n2(R, z)
    """ Yao 2017 """
    # Bd = 15
    # Ad = 2.5
    # n[R>Bd] = n[R>Bd] / np.cosh((R[R>Bd]-Bd)/Ad)**2
    return n


def test_disc_density(x, y, z, realisticMap=False, fullMass=False):
    """ Neutral H and H2 + neutral extraplanar """
    R = (x**2 + y**2)**0.5
    
    """ Table 4 """
    n20 = 4.06 * 2
    h2  = 2.57
    z2  = 0.08
    n10 = 0.32
    h1  = 18.24
    z1  = 0.52
    Rt  = 2.75
    
    """ The molecular hydrogen """
    def func_n2(R, z):
        n2 = n20 * np.exp(-R / h2 - abs(z) / z2)
        return n2
    
    """ The atomic hydrogen """
    def func_n1(R, z):
        if np.isscalar(R):
            if (R**2+z**2)**0.5 > Rt:
                n1 = n10 * np.exp(-R / h1 - abs(z) / z1)
            else:
                n1 = 0
        else:
            n1 = np.zeros(len(R))
            n1[(R**2+z**2)**0.5>Rt] = n10 * np.exp(-R[(R**2+z**2)**0.5>Rt] / h1 - abs(z[(R**2+z**2)**0.5>Rt]) / z1)
        return n1
    
    n1 = func_n1(R, z)
    n2 = func_n2(R, z)
    n = n2 + n1
    
    """ Check full mass """
    if fullMass:
        def better_func_n2(x, y, z):
            R = (x**2 + y**2)**0.5
            n2 = func_n2(R, z)
            return n2
        z10 = 0
        x10 = y10 = -np.inf
        z20 = y20 = x20 = np.inf # kpc
        integral = sp.integrate.nquad(better_func_n2, [[x10, x20], [y10, y20], [z10, z20]])
        print("H2 is {:.2e} M_sun \tTheir formula".format(4*np.pi*m_p*n20*z2*h2**2*(1e3*pc)**3/M_sun))
        print("H2 is {:.2e} M_sun \tMy formula".format(2*2*np.pi*m_p*n20*z2*h2**2*(1e3*pc)**3/M_sun))
        print("H2 is {:.2e} +- {:.2e} M_sun \tNumerical".format(2*integral[0]*(1e3*pc)**3 / M_sun * m_p, 2*integral[1]*(1e3*pc)**3 / M_sun * m_p))
        def better_func_n1(R, z):
            return n10 * np.exp(-R / h1 - abs(z) / z1) * 2 * np.pi * R
        R10 = Rt
        R20 = 13 #np.inf
        z10 = 0
        z20 = np.inf
        integral = sp.integrate.nquad(better_func_n1, [[R10, R20], [z10, z20]])[0]
        print("HI is {:.2e} M_sun \tNumerical (inside {} kpc)".format(2*integral*(1e3*pc)**3 / M_sun * m_p, R20))
        gamma = sp.special.gammaincc(2, Rt/h1, out=None)
        result = 4*np.pi*n10*z1*h1**2*gamma
        print("HI is {:.2e} M_sun \tAnalytic".format(result*(1e3*pc)**3 / M_sun * m_p))
        print("H2 is {:.2e} M_sun \tTheir number".format(8.2e9))
        
    """ The check is over """
    
    if realisticMap:
        """ The molecular and atomic hydrogen from the map """
        
        with fits.open('/home/afoninamd/Documents/NS/data/map.fits', mode='update') as hdul:
            data = hdul[0].data[::,::,::]
        
        X = np.linspace(-20, 20, num=np.shape(data)[1]) # [kpc]
        Y = np.linspace(-20, 20, num=np.shape(data)[2]) # [kpc]
        Z = np.linspace(-2, 2, num=np.shape(data)[0]) # [kpc]
        
        points = (Z, X, Y)
        values = data
        point = (z, x, y)
        
        n = sp.interpolate.interpn(points, values, point, method='linear',
                                   bounds_error=False, fill_value=0.)
    
    return n


def density_cold(x, y, z, realisticMap=False):
    """ Cold gas with helium """
    n_disk = neutral_density(x, y, z) + molecular_density(x, y, z)
    n_ionized = ionized_density(z)
    n_dust = dust_density(x, y, z)
    return (n_disk + n_ionized) * 1.33 + n_dust # 1.33 if for Helium


def halo_density(x, y, z):
    """ Hodges-Kluck et al. 2016 """
    r = (x**2+y**2+z**2)**0.5
    
    """ Locatelli 2024 """
    beta = 0.5
    n0 = 3.2e-2
    Rh = 6.2
    zh = 1.1
    C= 4.6e-2
    r0 = (C / (n0))**(1/(3*beta)) # 1.27, not 1
    R = (x**2+y**2)**0.5
    n_beta = n0 * (1+(r/r0)**2)**(-1.5*beta)
    n_disk = n0 * np.exp(-R/Rh) * np.exp(-abs(z)/zh)

    return n_beta + n_disk


def density_hot(x, y, z):
    """ Hot gas with helium """
    n_hot = halo_density(x, y, z)
    # if np.isscalar(n_halo):
    #     if n_halo > 0.01:
    #         n_halo = 0.01
    # else:
    #     n_halo[n_halo>0.01] = 0.01
    return n_hot * 1.33 # He


def galaxy_phase(x, y, z, realisticMap=False):
    """ Returns c_s and n and requires a track in the Galaxy """
    n_disk = neutral_density(x, y, z) + molecular_density(x, y, z)
    n_ionized = ionized_density(z)
    n = n_disk + n_ionized
    n_hot = density_hot(x, y, z)
    # n_cold = 100 n_not because it is c_s_hot**2/c_s_cold**2 * n_hot # since P_hot = P_cold
    f = (n/n_hot - 1) / ((c_s_hot/c_s_cold)**2 - 1)
    f[f>1] = 1
    
    R = (x**2+y**2)**0.5
    f[R>20] = 0
    
    return f
