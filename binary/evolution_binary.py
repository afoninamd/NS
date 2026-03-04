# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:08:19 2026

@author: AMD
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main.model import Object
from main.evolution import getB, getMdot, find_stage # they return np.arrays
from main.constants import m_p, AU, year, Gyr, R_sun, G, M_sun
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def gett_binary(num=10000):
    """ For the evolution with a sun-like companion """
    t0 = 1e6*year
    t1 = 10**np.linspace(np.log10(t0), np.log10(0.1*Gyr), endpoint=False, num=num)
    t2 = 10**np.linspace(np.log10(0.1*Gyr), np.log10(1.0*Gyr), endpoint=False, num=num)
    # t3 = 10**np.linspace(np.log10(1.0*Gyr), np.log10(1.1e10*year), endpoint=True, num=num)
    t3 = np.linspace((1.0*Gyr), (11*Gyr), endpoint=True, num=3*num)
    t = np.append(t1, t2)
    t = np.append(t, t3)
    return t

