#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 13:58:01 2025

@author: afoninamd
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:31:32 2023

@author: AMD
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from math import pi
from main.constants import c, G, I, R_NS, M_NS
import warnings
warnings.filterwarnings("ignore")

class Object():
    def __init__(self, B=1e12, v=100e5, Mdot=1e12, case='A', Omega=None):
        """
        B, v, n -- float or scipy.inerpolate
        choose Omega if its a binary
        """
        self.B0 = B
        self.v0 = v # cm/s
        # self.n0 = n
        self.Mdot0 = Mdot
        self.case = case
        self.R_NS = R_NS
        self.I = I
        self.GM = G * M_NS
        self.k = 2.
        """ ISM """
        # self.p_out_0 = p_out # /8/pi*(R_sun/0.03/AU)**6 # p_out
        self.R_t = 2e20
        self.v_t = 10e5
        """ Accretion """
        self.k_t = 1 / 2**1.5 * (1 / (0.96))**3
        self.eta = 1/4
        self.Omega = Omega
        """ Other """
        self.settle = 0
        # self.Bsin = True

    def B(self, t):
        """ Dipole magnetic field at the equator """
        return self.B0

    def v(self, t):
        """ Includes v_infty and c_s """
        return self.v0

    def M_dot(self, t):
        """ The amount of captured matter [g/s] """
        return self.Mdot0

    # def n(self, t):
    #     if np.isscalar(self.n0):
    #         if not np.isscalar(t) and len(t.shape):
    #             return np.zeros(len(t)) + self.n0
    #         else:
    #             return self.n0
    #     else:
    #         return self.n0(t)

    # def p_out(self, t):
    #     if np.isscalar(self.p_out_0):
    #         if not np.isscalar(t) and len(t.shape):
    #             return np.zeros(len(t)) + self.p_out_0
    #         else:
    #             return self.p_out_0
    #     else:
    #         return self.p_out_0(t)

    # def rho(self, t):
    #     """ It is the real rho = 4 * rho_infty """
    #     rho = self.n(t) * m_p
    #     return rho

    def R_G(self, t):
        R_G = 2 * self.GM / self.v(t)**2
        return R_G

    # def M_dot(self, t): # 2.5/4!!!!!!!!
    #     """ For isolated NS """
    #     M_dot = pi * self.R_G(t)**2 * self.rho(t) * self.v(t)
    #     return M_dot

    # def first_stage(self, t, P):
    #     if P <= self.P_EP(t):# and P < self.P_PE(t):
    #         stage = Ejector
    #     elif P > self.P_PA(t):
    #         if self.R_m_cur(t, P) > self.R_G(t): # from A to G condition
    #             stage = Georotator
    #         else:
    #             stage = Accretor
    #     else:
    #         stage = Propeller
    #     return stage(self.B0, self.v0, self.n0, self.field, self.case, self.Omega, self.p_out_0)


    def mu(self, t):
        mu = self.B(t) * self.R_NS**3
        return mu

    def R_l(self, P):
        omega = 2 * pi / P
        R_l = c / omega
        return R_l 

    def R_Sh(self, t, P):
        omega = 2 * pi / P
        
        # R_Sh = ((self.k * self.mu(t)**2 * omega**4 /
        #           (4 * pi * c**4 * (self.rho(t) * self.v(t)**2 + p_out)))**0.5)
        
        R_Sh = ((4*self.k * self.mu(t)**2 / self.M_dot(t) * self.GM**2 *
                  omega**4 / self.v(t)**5 / c**4)**0.5)
        return R_Sh

    def R_c(self, P):
        R_c = (self.GM * P**2 / (4 * pi**2))**(1/3)
        return 0.87 * R_c

    # def pe_R_A(self, t, x):
    #     """ Polynomial equation for R_A """
    #     return self.M_dot(t)*(2*self.GM)**0.5/(4*pi)*x**3.5 + self.p_out(t)*x**6 - self.mu(t)**2/(8*pi)
    
    def R_A(self, t):
        R_A = (self.mu(t)**2 / (2 * self.M_dot(t) * (2 * self.GM)**0.5))**(2/7)
        # def pe(x):
        #     return self.pe_R_A(t, x)
        # if self.p_out_0:
        #     solution = root(pe, R_A)
        #     R_A = solution.x[0]
        return R_A

    def R_m(self, t):
        R_m = self.R_A(t)**(7/9) * self.R_G(t)**(2/9)
        if R_m < self.R_A(t):
            R_m = self.R_A(t)
        return R_m
    
    # def pe_R_m_A1(self, t, P, x):
    #     """ Polynomial equation for R_m_A1 """
    #     omega = 2 * pi / P
    #     p_out = self.p_out(t)
    #     return (2 * self.M_dot(t) * omega**2 / (2 * self.GM)**0.5 * x**(13/2)
    #             + 8 * pi * p_out * x**6 - self.mu(t)**2)
    
    def R_m_A1(self, t, P):
        omega = 2 * pi / P
        R_m = (self.mu(t)**2 / (2 * self.M_dot(t)) * self.v(t) *
               self.R_G(t)**0.5 / omega**2)**(2/13)
        # def pe(x):
        #     return self.pe_R_m_A1(t, P, x)
        # if self.p_out_0:
        #     solution = root(pe, R_m)
        #     R_m = solution.x[0]
        return R_m
    
    # def pe_R_m_A1_geo(self, t, P, x):
        # """ Polynomial equation for R_m_A1 """
        # omega = 2 * pi / P
        # p_out = self.p_out(t)
        # return (self.M_dot(t) * omega**2 * self.v(t)**3 / (2 * self.GM**2) * x**8
        #         + 8 * pi * p_out * x**6 - self.mu(t)**2)
    
    def R_m_A1_geo(self, t, P):
        omega = 2 * pi / P
        R_m = (self.mu(t)**2 / (2 * self.M_dot(t)) * self.v(t) *
               self.R_G(t)**2 / omega**2)**(1/8)
        # def pe(x):
        #     return self.pe_R_m_A1_geo(t, P, x)
        # if self.p_out_0:
        # solution = root(pe, R_m)
        # R_m = solution.x[0]
        return R_m
    
    def R_m_geo(self, t):
        """ по аналогии с 23.III в Липунове """
        # p_out = self.p_out(t)
        # p_out = 0
        R_m = (self.mu(t)**2 / (2*self.M_dot(t)*self.v(t)) # + 8*pi*self.R_G(t)**2*p_out)
               * self.R_G(t)**2)**(1/6)
        return R_m

    def R_m_settle(self, t):
        # f = 0.1
        # K2 = 2.75**2
        # R_A = (4*gamma/(gamma-1) * f*K2/(phi*(1+gamma*m_t**2)) *
        #        self.mu(t)**2/(self.M_dot(t)*(2*self.GM)**0.5))**(2/7)
        """ reverse Compton cooling """
        R_A = 1.55e9 * (self.mu(t)/1e30)**(6/11) * (self.M_dot(t)/1e16)**(-2/11)
        """ free-free cooling  """
        # M_dot_x = 1e10*0.001*(self.mu(t)/1e30)**(2/21)*1*(self.M_dot(t)/1e10)**(9/7) # self.M_dot(t) * (t_ff/t_cool)**(1/3)
        # L_x = self.GM * M_dot_x / self.R_NS
        # R_A = 2.2e10 * (L_x/1e30)**(-2/9) * (self.mu(t)/1e30)**(16/27) # Popov, Postnov, Shakura 2015
        # R_A = self.R_A(t)
        return R_A
    
    def R_m_cur(self, t, P):
        # if self.case == "A1":
        #     R_m = self.R_m_A1(t, P)
        #     if R_m > self.R_G(t):
        #         R_m = self.R_m_A1_geo(t, P)
        # else:
        R_m = self.R_m(t)
        if R_m > self.R_G(t):
            R_m = self.R_m_geo(t)
        return R_m

    def R_Sh_geo(self, t, P):
        """ R_Sh < R_l for the reverse transition """
        omega = 2*pi/P
        A = (c**4*self.M_dot(t)*self.v(t)/(self.k*self.mu(t)**2*omega**4))**2 # for p~R^-5/2
        A = 1 / A
        # if self.case == 'A1': # return if A1 is being calculated
        #     A = 1 / A # p ~ r^-3/2
        R_Sh = self.R_G(t) * A
        return R_Sh

    def P_Ejector(self, t, P0):
        """ Analytical solution only for CF!!! """
        P2 = 8 * np.pi**2 * self.k * self.mu(t)**2 / self.I / c**3 * t
        P = (P2 + P0**2)**0.5
        return P 
    
    def age_Ejector(self, P0, P2):
        """ Analytical solution only for constants!!! """
        t = 5.
        t = self.I * c**3 / (8*pi**2*self.mu(t)**2*self.k) * (P2**2-P0**2)
        return t

    def P_EP_l(self, t):
        """ R_Sh = R_l, given R_Sh < R_G, R_Sh_in < min(R_l, R_G) """
        A = self.R_Sh_geo(t, 2*pi/1)
        B = self.R_l(2*pi/1)
        # if self.case == 'A1':
        #     """ p_out ~ R^-5/2 - free fall """
        #     omega = (A/B)**(1/7)
        # else:
        """ p_out ~ R^-3/2 """
        omega = (B/A)**(1/9)
        P1 = 2 * pi / omega
        if self.R_l(P1) > self.R_G(t):
            A = self.R_Sh_geo(t, 2*pi/1)
            # if self.case == 'A1': # p ~ r^-5/2
            #     P1 = 2*pi*(self.R_G(t)/A)**0.125
            # else:
            P1 = 2*pi*(A/self.R_G(t))**0.125
        
        """ Check is done """
        # P1 = 2*pi/c * (2*self.GM*self.k**2 * self.mu(t)**4 / (self.M_dot(t)**2*self.v(t)**4))**(1/9) 
        # if self.R_l(P1) > self.R_G(t):
        #     P1 = 2*pi/c * (self.k * self.mu(t)**2 / (self.M_dot(t)*self.v(t)))**(1/4)
        
        return P1
    
    def P_EP_G(self, t):
        """ R_Sh = R_l, given R_Sh > R_G, R_Sh = A*omega**2, R_l = B/omega 
            R_Sh_out > max(R_l, R_G) """
        """ R_Sh = R_G, given R_Sh = A*omega**2 """
        A = self.R_Sh(t, 2*pi/1)
        omega = (self.R_G(t) / A)**0.5
        P2 = 2 * pi / omega
        if self.R_l(P2) > self.R_G(t):
            A = self.R_Sh(t, 2*pi/1)
            B = self.R_l(2*pi/1)
            omega = (B/A)**(1/3)
            P2 = 2 * pi / omega
            
        """ Check is done """
        # P2 = 2*pi/c * (self.k*self.mu(t)**2 / (self.M_dot(t)*self.v(t)))**0.25
        # if self.R_l(P2) > self.R_G(t):
        #     P2 = 2*pi/c * (4*self.GM**2 * self.k*self.mu(t)**2 / (self.M_dot(t) * self.v(t)**5))**(1/6)
        
        return P2
    
    def P_EP_m(self, t):
        # if self.case == 'A1':
        #     A = self.R_m_A1(t, 2*pi/1)
        #     B = self.R_l(2*pi/1)
        #     omega = (B/A)**(13/9)
        #     if self.R_m_A1(t, 2*pi/omega) > self.R_G(t):
        #         A = self.R_m_A1_geo(t, 2*pi/1)
        #         omega = (B/A)**(4/3)
        #     P2 = 2*pi/omega
        # else:
        omega = c / self.R_m_cur(t, 1)
        P3 = 2 * pi / omega
        
        """ Check is done """
        # R_m = max(self.R_A(t), (self.R_A(t)**(7/9)*self.R_G(t)**(2/9)))
        # if R_m > self.R_G(t):
        #     R_m = (2*self.mu(t)**2*self.GM**2 / (self.M_dot(t)*self.v(t)**5))**(1/6)
        # P3 = 2*pi/c * R_m
        
        return P3

    def P_PE(self, t):
        P1 = self.P_EP_l(t)
        P2 = self.P_EP_m(t)
        # P3 = self.P_EP_G(t)
        return min(P1, P2)
    
    def P_EP(self, t):
        P1 = self.P_EP_l(t)
        # P2 = self.P_EP_m(t)
        P3 = self.P_EP_G(t)
        return max(P1, P3)

    
    def P_PA(self, t):
        """ R_m = R_c """
        if self.case == "A1":
            P_PA = (self.R_m_A1(t, 1.) / self.R_c(1.))**(39/14)
            P_PA_geo = (self.R_m_A1_geo(t, 1.) / self.R_c(1.))**(12/5)
            P_PA = min(P_PA, P_PA_geo)
        else:
            R_m = self.R_m(t)
            if R_m > self.R_G(t):
                R_m = self.R_m_geo(t)
            P_PA = (self.R_m(t) / self.R_c(1.))**(1.5)
        return P_PA

    def P_AP(self, t):
        """ R_A = R_c """
        # P_AP = 2 * pi * self.R_A(t)**1.5 / self.GM**0.5
        P_AP = (self.R_A(t) / self.R_c(1.))**1.5
        return P_AP

    def P_turb(self, t):
        v_turb = 1e5 # 1 kms
        R_t = 1e18 #2e20 # 2e20 cm
        v_t = (self.R_G(t) / R_t)**(1/3) * v_turb
        
        rho = self.M_dot(t) / (4 * self.GM**2)* self.v(t)**3
        P_LP = 500 * self.k_t**(1/3) * (self.mu(t)/1e30)**(2/3) * (rho*1e24)**(-2/3) * (self.v(t)/1e6)**(13/3) * (v_t/1e6)**(-2/3) * (1.4)**(-8/3)
        # D = 1/6 *(self.M_dot(t) * v_t * self.R_G(t) / self.I)**2 * self.R_G(t) / self.v(t)
        # P_new = 2 * pi * (self.k_t * self.mu(t)**2 / (3*self.GM*self.I*D))**(1/3)
        return P_LP
        # return np.inf #


    def P_eq(self, t):
        """ Equillibrium period at the accretor stage """
        """
        K_sd = self.mu(t)**2 / self.R_c(P)**3 * self.k_t
        if self.SyXRB:
            K_su = self.M_dot(t) * self.eta * self.Omega * self.R_G(t)**2
            R_m = max(R_m, self.R_NS)
            K_su = min(K_su, self.M_dot(t) * (self.GM * R_m)**0.5)
        """
        if self.Omega is None:
            return np.inf
        
        if self.settle:
            R_m = self.R_m_settle(t)
        else:
            R_m  = self.R_A(t)
        
        # K_su = self.M_dot(t) * (self.GM * self.R_A(t))**0.5
        K_su = self.M_dot(t) * self.eta * self.Omega * self.R_G(t)**2
        if K_su > self.M_dot(t) * (self.GM * R_m)**0.5:
            print('disc')
            print(np.log10(self.B(t)))
            print(K_su / self.M_dot(t))
            print((self.GM * R_m)**0.5)
        # if K_su > self.M_dot(t) * (self.GM * R_m)**0.5:
        #     print('DISK')
        R_m = max(R_m, self.R_NS)
        K_su = min(K_su, self.M_dot(t) * (self.GM * R_m)**0.5)
        # K_su  = self.M_dot(t) * (self.GM * R_m)**0.5
        if self.settle:
            # K_sd = 8 * self.M_dot(t) * R_m**2 * 
            rho_inf = self.M_dot(t) / (4*pi*self.R_G(t)**2*self.v(t))
            K_sd_1 = 8*pi*rho_inf*self.R_G(t)**1.5*(2*self.GM)**0.5*R_m**2
            P_eq = 2*pi/K_su*K_sd_1
        else:
            P_eq = self.mu(t) * (self.k_t / K_su / self.R_c(1.)**3)**0.5
        # P_eq1 = 2*pi*self.mu(t) * (self.k_t / self.GM / K_su)**0.5
        # print(np.log10(P_eq/P_eq1))
        return P_eq

    def j(self, t):
        j_t = self.v_t * self.R_G(t)**(4/3) / self.R_t**(1/3)
        j_K = (self.GM * self.R_A(t))**0.5
        j = min(j_K, j_t)
        return j #j_K, j_t

    def dP_dt_ejector(self, t, P):
        mu = self.mu(t)
        # if seld.Bsin:
        #     omega = 2 * pi / P
        #     mu = mu * (1 + np.sin(omega*t))
        dP_dt = 4 * self.k * pi**2 * mu**2 / (c**3 * self.I * P)
        return dP_dt
    
    def t_char_ejector(self, t=0):
        """ Characteristic time = P/dot{P}, choose the min P = P0 """
        t = 0 # corresponds to the minimum characteristic time
        P = 0.1
        return P/self.dP_dt(t, P)
    
    def t_ejector(self, P0, P):
        return self.I * c**3 / (8*np.pi**2*self.k * self.mu(0)**2) * (P**2-P0**2)

    def K_P(self, t, P):
        R_m = self.R_m_cur(t, P)
        R_m = min(R_m, self.R_l(P))
        """ If disk """
        if self.Omega:
            if (self.GM*R_m)**0.5 <= self.eta*self.Omega*self.R_G(t)**2:
                R_m = self.R_A(t)
        # if R_m > self.R_G(t):
        #     pass
            # print('geo at propeller')
        
        if self.case == "A":
            K_P = self.M_dot(t) * 2 * pi / P * R_m**2
        elif self.case == "B":
            K_P = self.M_dot(t) * (2 * self.GM * R_m)**0.5
        elif self.case == "C":
            omega = 2 * pi / P
            if R_m > self.R_G(t):
                v = self.v(t)
            else:
                v = (2 * self.GM / R_m)**0.5
            # K_P = self.GM * self.M_dot(t) / omega / R_m
            K_P = self.M_dot(t) * v**2 / (2 * omega)
        elif self.case == "D":
            omega = 2 * pi / P
            K_P = self.M_dot(t) * self.v(t)**2 / (2 * omega)
        else:
            print("There are only 5 propeller cases: A, B, C and D, no",
                  self.case)
        # print(K_P)
        return K_P

    def dP_dt_propeller(self, t, P):
        dP_dt = P**2 / (2 * pi * self.I) * self.K_P(t, P)
        return dP_dt
    
    def t_char_propeller(self, t):
        """ for case A and B t_char is min earlier (dot{M}-max, \mu-max) """
        P = self.P_AP(t)
        return P/self.dP_dt(t, P)


    def K_su(self, t):
        if self.settle:
            R_m = self.R_m_settle(t)
        else:
            R_m  = self.R_A(t)
        # print(self.R_m_settle(t)/self.R_A(t))
        if self.Omega:
            K_su = self.M_dot(t) * self.eta * self.Omega * self.R_G(t)**2
            R_m = max(R_m, self.R_NS)
            K_su = min(K_su, self.M_dot(t) * (self.GM * R_m)**0.5)
        else:
            K_su = 0
        return K_su
    
    def K_sd(self, t, P):
        if self.settle:
            R_m = self.R_m_settle(t)
            omega = 2 * np.pi / P
            # K_sd = 8 * self.M_dot(t) * R_m**2 * 
            rho_inf = self.M_dot(t) / (4*pi*self.R_G(t)**2*self.v(t))
            K_sd = omega * 8*pi*rho_inf*self.R_G(t)**1.5*(2*self.GM)**0.5*R_m**2
        else:
            K_sd = self.mu(t)**2 / self.R_c(P)**3 * self.k_t
        return K_sd
    
    def dP_dt_accretor(self, t, P):
        dP_dt = P**2 / (2 * pi * self.I) * (self.K_sd(t, P) - self.K_su(t))
        # if abs(dP_dt) > 1e-5:
        #     print(dP_dt)
        #     print(self.K_su(t))
        #     # print(self.M_dot(t) * self.eta * self.Omega * self.R_G(t)**2)
        #     print(self.Omega)
        #     print(self.v(t))
        return dP_dt
    
    def t_char_accretor(self, t):
        """ t_char is min earlier """
        P = self.P_AP(t)
        return P/self.dP_dt(t, P)


    def K_sd_georotator(self, t, P):
        """ Decelerating moment of inertia on Accretor stage """ 
        """ Магнитосфера R_m_geo теперь!!!!! """
        return 0. # 1 / 3 * self.mu(t)**2 / self.R_c(P)**3
    
    def dP_dt_georotator(self, t, P):
        """ Period change over time on Accretor stage """
        # dP_dt = P**2 * (self.K_sd(t, P)) / (2 * pi * self.I)
        return 0. # dP_dt

    # def first_stage(self, t, P):
    #     if P < self.P_EP(t): # self.R_Sh(t, P) > max(self.R_l(P), self.R_G(t)):
    #         stage = 1 #self.dP_dt_ejector(t, P), 1
    #     elif P < self.P_PA(t): #self.R_m_cur(t, P) > self.R_c(P):
    #         stage = 2 #self.dP_dt_propeller(t, P), 2
    #     elif self.R_m_cur(t, P) < self.R_G(t): # from A to G condition
    #         stage = 3 #self.dP_dt_accretor(t, P), 3
    #     else:
    #         if self.Georotators:
    #             stage = 4 #self.dP_dt_georotator(t, P), 4
    #         else:
    #             stage = 3 #self.dP_dt_accretor(t, P), 3

    #     return stage  # tuple
