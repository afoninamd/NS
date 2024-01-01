#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:58:15 2023

@author: afoninamd
"""

import astropy.coordinates as coord
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
import numpy as np
import gala.integrate as gi
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic
from gala.units import UnitSystem
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import scipy as sp
from astropy.io import fits
from constants import c, G, M_sun, m_p, year, birth_age, galaxy_age, R_NS, I, sunPos, pc, rgal
from constants import fontsize

""" Defining cgs unit system """
usys = UnitSystem(u.cm, u.second, u.radian, u.gram) # cgs
# """ Checking """
# dens = gp.NFWPotential(m=6E11, r_s=20., units=galactic).density([0, 5, 0.], 0.)
# dens = pot.density([0, 5, 0], 0.)
# dens = usys.decompose(dens).value / m_p

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

def get_potential(log_M_h, log_r_s, log_M_n, log_a):
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


def DensityMap():
    
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
    

def DensityMapMisiriotis(x, y, z):
    """ Following Misiriotis (2006) astro.ph/0607638 """
    R = (x**2 + y**2)**0.5
    # R = np.array(R)
    # z = np.array(z)
    
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
    
    """ The molecular and atomic hydrogen """
    n2 = n20 * np.exp(-R / h2 - abs(z) / z2)
    
    if R > Rt:
        n1 = n10 * np.exp(-R / h1 - abs(z) / z1)
    else:
        n1 = 0
        # !!! vectorize!!!!
    # if isinstance(R, np.ndarray):
    # if not np.size(R) - 1:
    #     n1 = n10 * np.exp(-R / h1 - abs(z) / z1)
    # else:
    #     n1 = np.zeros(len(R))
    #     for i in range(len(R)):
    #         if R[i] > Rt:
    #             n1[i] = n10 * np.exp(-R / h1 - abs(z[i]) / z1)

    n = nw + nc + n2 + n1

    return n

def DensityMapMisiriotisPlot():
    
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
                numberDensity[i, j, k] = DensityMapMisiriotis(x, y, z)
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


def DensityMapPlot(HI=True, H2=True, logscale=False):
    
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


def SpiralArm(R, phi, psi):
    """ One spiral arm, [kpc, degrees, degrees] """
    phi_ai = phi / 180 * np.pi
    psi_ai = psi / 180 * np.pi
    
    # r = np.linspace(0, 20, 100)
    phi = np.linspace(phi_ai, phi_ai + 1.2 * np.pi, 100)
    r = R * np.exp((phi - phi_ai) * np.tan(psi_ai))
    
    X = r * np.cos(phi)
    Y = r * np.sin(phi)
    
    return X, Y


def YaoArms():
    """ Spiral arms from Yao et al. (2017) """
    R = [3.35, 3.71, 3.56, 3.67, 8.21] # kpc
    phi = [44.4, 120.0, 218.6, 330.3, 55.1]
    psi = [11.43, 9.84, 10.38, 10.54, 2.77]
    for i in range(len(R)):
        X, Y = SpiralArm(R[i], phi[i], psi[i])
        plt.plot(X, Y, 'white')
    # X, Y = SpiralArm(3.35, 44.4, 11.43)


def Orbit(pos=[10,0,0.], vel=[0,175,0], t_end=galaxy_age, plot=False):

    """ Defining the Milky Way potential """
    mw = gp.MilkyWayPotential() # pot.units = (kpc, Myr, solMass, rad)
    # Om_bar = 42. * u.km/u.s/u.kpc
    # frame = gp.ConstantRotatingFrame(Omega=[0,0,Om_bar.value]*Om_bar.unit,
    #                              units=galactic)
    pot = mw
    
    """ Rotational curve for initial velocities """
    cvel = pot.circular_velocity(pos)[0].value
    R = (pos[0]**2 + pos[1]**2)**0.5
    vel[0] += cvel * pos[1] / R
    vel[1] += cvel * pos[0] / R
    
    """ Min orbital period """ #!!! check, REDO
    acc = pot.acceleration(pos, t=0.0).value
    acceleration = (acc[0]**2 + acc[1]**2 + acc[2]**2)**0.5
    acc = acceleration[0] # kpc / Myr
    maxR = (pos[0]**2 + pos[1]**2 + pos[2]**2)**0.5
    v = (maxR * acc)**0.5
    orbPeriod = 2 * np.pi * maxR / v # Myr
    orbPeriod = orbPeriod * year * 1e6 # s
    
    """ Initial position and velocity """
    ics = gd.PhaseSpacePosition(pos=pos * u.kpc, vel=vel * u.km/u.s)
    
    """ Orbit integrating """
    v_max = (vel[0]**2 + vel[1]**2 + vel[2]**2)**0.5
    dt = 4 * pc / (v_max * 1e5) / year / 1e6 # [Myr]
    if dt < 10_000 / 1e6:
        dt = 10_000

    n_steps = int(1.1 + t_end / year / 1e6 / dt)
    orbit = gp.Hamiltonian(pot).integrate_orbit(ics, dt=dt, n_steps=n_steps)
                                                # 0.5*u.Myr,
                                                # n_steps=1+int(2*t_end/year/1e6))

    t = np.array(orbit.t) # Myr
    xyz = np.array(orbit.pos.xyz) # [x, y, z] kpc
    v_xyz = np.array(orbit.vel.d_xyz) # [v_x, v_y, v_z] kpc / Myr
    
    if plot:
        """ Custom plot """
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]
        
        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        
        axs[0].plot(x, y)
        r = 1.1*np.max((x**2+y**2)**0.5)
        axs[0].set_xlim(-r, r)
        axs[0].set_ylim(-r, r)
        axs[0].set_ylabel("y, kpc", fontsize=fontsize)
        axs[0].set_xlabel("x, kpc", fontsize=fontsize)
        axs[1].plot(x, z)
        r = 1.1*np.max((x**2+z**2)**0.5)
        axs[1].set_xlim(-r, r)
        axs[1].set_ylim(-r, r)
        axs[1].set_ylabel("z, kpc", fontsize=fontsize)
        axs[1].set_xlabel("x, kpc", fontsize=fontsize)
        axs[2].plot(y, z)
        r = 1.1*np.max((y**2+z**2)**0.5)
        axs[2].set_xlim(-r, r)
        axs[2].set_ylim(-r, r)
        axs[2].set_ylabel("z, kpc", fontsize=fontsize)
        axs[2].set_xlabel("y, kpc", fontsize=fontsize)
        
        for i in range(len(axs)):
            axs[i].set_box_aspect(1)
            axs[i].tick_params(which='major', width=1.0, length=10,
                               labelsize=fontsize)
            axs[i].tick_params(which='minor', width=1.0, length=5,
                               labelsize=fontsize-2, labelcolor='0.25')
        
        if np.max(np.append(x, y)) > rgal:
            circ = plt.Circle((0, 0), rgal, fill=False)
            axs[0].add_patch(circ)
        if np.max(z) > 2:
            axs[1].add_patch(plt.Rectangle((-rgal, -2), 2*rgal, 4, fill=False))
            axs[2].add_patch(plt.Rectangle((-rgal, -2), 2*rgal, 4, fill=False))
            # axs[0].patches.Circle((0, 0), rgal, fill=False)
            

        """ Gala plot """
        # fig = orbit.plot() # shows xy, xz, yz planes
        # fig.show()
    
    return t, xyz, v_xyz, orbPeriod


def Trajectory(pos=[10,0,0.], vel=[0,175,0], t_end=galaxy_age, plot=False,
               Misiriotis=False): # initial pos, vel

    """ Obtaining time, position and velocity arrays """
    t, xyz, v_xyz, orbPeriod = Orbit(pos=pos, vel=vel, t_end=t_end, plot=plot) # [kpc / Myr]
    t = t * 1e6 * year # [s]
    v_xyz = v_xyz * 1000 * pc / (1e6 * year) * 1e-5 # [km/s]
    
    """ Absolute velocities -> velocities relative to ISM """#!!! REDO
    # mw = gp.MilkyWayPotential() # pot.units = (kpc, Myr, solMass, rad)
    # pot = mw
    # pos = xyz
    # vel = v_xyz
    # print(vel[0])
    # cvel = pot.circular_velocity(pos).value
    # R = (pos[0]**2 + pos[1]**2)**0.5
    # vel[0] -= cvel * pos[1] / R
    # vel[1] -= cvel * pos[0] / R
    # print(vel[0])
    
    
    """ Array of velocities v """
    mw = gp.MilkyWayPotential() # pot.units = (kpc, Myr, solMass, rad)
    pot = mw
    v2 = np.zeros(len(t))
    for i in range(len(v_xyz)):
        v2 = v2 + v_xyz[i]**2
    v = v2**0.5
    v = abs(v - pot.circular_velocity(xyz).value)
    v = v * 1e5 # [cm/s]
    
    if Misiriotis:
        """ Number density array from Misiriotis et al. model """
        n = np.zeros(len(t))
        for i in range(len(t)):
            n[i] = DensityMapMisiriotis(xyz[0][i], xyz[1][i], xyz[2][i])
        
    else:
        """ Number density array from realistic map """
        with fits.open('map.fits', mode='update') as hdul:
            data = hdul[0].data[::,::,::]
        
        X = np.linspace(-20, 20, num=np.shape(data)[1]) # [kpc]
        Y = np.linspace(-20, 20, num=np.shape(data)[2]) # [kpc]
        Z = np.linspace(-2, 2, num=np.shape(data)[0]) # [kpc]
        
        points = (Z, X, Y)
        values = data
        point = (xyz[2], xyz[0], xyz[1])
        
        # print(xyz[0][xyz[0]>10])
        
        n = sp.interpolate.interpn(points, values, point, method='linear',
                                   bounds_error=False, fill_value=0.)
    
    """ Defining 0.1 stellar disk potential """
    # x0 = [np.log(6E11), np.log(20.), np.log(2E9), np.log(100.)]
    # pot = get_potential(*x0)
    
    """ Array of number densities n """
    # dens = pot.density(xyz, 0.0) # [solMass / kpc3]
    # dens = usys.decompose(dens) # [g / cm3]
    # n = np.array(dens.value) / m_p # [cm-3]
    
    return t, v, n, orbPeriod


def TrajectoryPlot(pos=[1,0,0.], vel=[0,175,0], t_end=galaxy_age):
    """ Plotting trajectory and corresponding v(t) or n(t) """
    # t, xyz, v_xyz = Orbit(pos=pos, plot=True)
    t, v, n, orbPeriod = Trajectory(pos=pos, vel=vel, t_end=t_end, plot=True)
    plt.figure(figsize=(20, 20))
    plt.plot(t, n, marker='o')
    plt.plot(t, v)


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

