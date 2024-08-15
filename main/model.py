#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:31:32 2023

@author: AMD
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from astropy import constants as const
from dataclasses import dataclass
import numpy as np
from math import pi
from numpy import exp
from scipy.integrate import solve_ivp
from scipy.optimize import root, newton
from abc import abstractmethod
import matplotlib.pyplot as plt
from main.constants import c, G, M_sun, m_p, year, birth_age, galaxy_age, I, R_NS, R_sun, AU
import warnings
warnings.filterwarnings("ignore")

class Simulation():
    """ Contains common functions and initial parameters """
    def __init__(self, B=1e12, v=100e5, n=1., field='CF', case='A', Omega=None, p_out=0):
        """
        B, v, n -- float or scipy.inerpolate
        choose Omega if its a binary
        """
        self.B0 = B
        self.v0 = v # cm/s
        self.n0 = n
        self.field = field
        self.case = case
        self.R_NS = R_NS
        self.I = I
        self.GM = G * 1.4 * M_sun
        self.k = 2.
        """ ISM """
        self.p_out_0 = p_out # /8/pi*(R_sun/0.03/AU)**6 # p_out
        self.R_t = 2e20
        self.v_t = 10e5
        """ Accretion """
        self.k_t = 1 / 2**1.5 * (1 / (0.96))**3
        self.eta = 1/4
        self.Omega = Omega
        """ Other """
        self.Georotators = 0 # False # turns off Georotator stages
        self.settle = False
        self.calcPeriod = False#True

    def B(self, t):
        if np.isscalar(self.B0):
            if not np.isscalar(t) and len(t.shape):
                return np.zeros(len(t)) + self.B0
            else:
                return self.B0
        else:
            return self.B0(t)

    def v(self, t):
        if np.isscalar(self.v0):
            if not np.isscalar(t) and len(t.shape):
                return np.zeros(len(t)) + self.v0
            else:
                return self.v0
        else:
            return self.v0(t)

    def n(self, t):
        if np.isscalar(self.n0):
            if not np.isscalar(t) and len(t.shape):
                return np.zeros(len(t)) + self.n0
            else:
                return self.n0
        else:
            return self.n0(t)

    def p_out(self, t):
        if np.isscalar(self.p_out_0):
            if not np.isscalar(t) and len(t.shape):
                return np.zeros(len(t)) + self.p_out_0
            else:
                return self.p_out_0
        else:
            return self.p_out_0(t)

    def rho(self, t):
        """ It is the real rho = 4 * rho_infty """
        rho = self.n(t) * m_p
        return rho

    def R_G(self, t):
        R_G = 2 * self.GM / self.v(t)**2
        return R_G

    def M_dot(self, t): # 2.5/4!!!!!!!!
        """ For isolated NS """
        M_dot = pi * self.R_G(t)**2 * self.rho(t) * self.v(t)
        return M_dot

    def first_stage(self, t, P):
        if self.R_Sh(t, P) > max(self.R_l(P), self.R_G(t)):
            stage = Ejector
        elif self.R_m_cur(t, P) > self.R_c(P):
            stage = Propeller
        elif self.R_m_cur(t, P) < self.R_G(t): # from A to G condition
            stage = Accretor
        else:
            stage = Georotator

        return stage(self.B0, self.v0, self.n0, self.field, self.case, self.Omega, self.p_out_0)

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

    @abstractmethod
    def title(self):
        return 'Simulation'

    @abstractmethod
    def get_term_events(self):
        print("These events must not be in Simulation")
        pass

    @abstractmethod
    def dP_dt(self, t, p):
        print("This function must not be in Simulation")
        pass

    def mu(self, t):
        mu = self.B(t) * self.R_NS**3
        return mu

    def R_l(self, P):
        omega = 2 * pi / P
        R_l = c / omega
        return R_l 

    def R_Sh(self, t, P):
        omega = 2 * pi / P
        p_out = 0
        
        R_Sh = ((self.k * self.mu(t)**2 * omega**4 /
                  (4 * pi * c**4 * (self.rho(t) * self.v(t)**2 + p_out)))**0.5)
        
        # R_Sh = ((4*self.k * self.mu(t)**2 / self.M_dot(t) * self.GM**2 *
        #           omega**4 / self.v(t)**5 / c**4)**0.5)
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
        p_out = 0
        R_m = (self.mu(t)**2 / (2*self.M_dot(t)*self.v(t) + 8*pi*self.R_G(t)**2*p_out)
               * self.R_G(t)**2)**(1/6)
        return R_m

    def R_m_cur(self, t, P):
        if self.case == "A1":
            R_m = self.R_m_A1(t, P)
            if R_m > self.R_G(t):
                R_m = self.R_m_A1_geo(t, P)
        else:
            R_m = self.R_m(t)
            if R_m > self.R_G(t):
                R_m = self.R_m_geo(t)
        return R_m

    def R_Sh_geo(self, t, P):
        """ R_Sh < R_l for the reverse transition """
        omega = 2*pi/P
        A = (c**4*self.M_dot(t)*self.v(t)/(self.k*self.mu(t)**2*omega**4))**2 # for p~R^-5/2
        if not self.case == 'A1':
            A = 1 / A # p ~ r^-3/2
        R_Sh = self.R_G(t) * A
        return R_Sh

    def P_EP_l(self, t):
        """ R_Sh = R_l, given R_Sh < R_G, R_Sh_in < min(R_l, R_G) """
        A = self.R_Sh_geo(t, 2*pi/1)
        B = self.R_l(2*pi/1)
        if self.case == 'A1':
            """ p_out ~ R^-5/2 - free fall """
            omega = (A/B)**(1/7)
        else:
            """ p_out ~ R^-3/2 """
            omega = (B/A)**(1/9)
        P1 = 2 * pi / omega
        if self.R_l(P1) > self.R_G(t):
            A = self.R_Sh_geo(t, 2*pi/1)
            if self.case == 'A1': # p ~ r^-5/2
                P1 = 2*pi*(self.R_G(t)/A)**0.125
            else:
                P1 = 2*pi*(A/self.R_G(t))**0.125
        return P1
    
    def P_EP_G(self, t):
        """ R_Sh = R_l, given R_Sh > R_G, R_Sh = A*omega**2, R_l = B/omega 
            R_Sh_out > max(R_l, R_G) """
        A = self.R_Sh(t, 2*pi/1)
        B = self.R_l(2*pi/1)
        omega = (B/A)**(1/3)
        P2 = 2 * pi / omega
        if self.R_l(P2) < self.R_G(t):
            """ R_Sh = R_G, given R_Sh = A*omega**2 """
            A = self.R_Sh(t, 2*pi/1)
            omega = (self.R_G(t) / A)**0.5
            P2 = 2 * pi / omega
        return P2
    
    def P_EP_m(self, t):
        if self.case == 'A1':
            A = self.R_m_A1(t, 2*pi/1)
            B = self.R_l(2*pi/1)
            omega = (B/A)**(13/9)
            if self.R_m_A1(t, 2*pi/omega) > self.R_G(t):
                A = self.R_m_A1_geo(t, 2*pi/1)
                omega = (B/A)**(4/3)
            P2 = 2*pi/omega
        else:
            omega = c / self.R_m_cur(t, 1)
            P2 = 2 * pi / omega
        return P2

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

    def P_eq(self, t):
        """ Equillibrium period at the accretor stage """
        """
        K_sd = self.mu(t)**2 / self.R_c(P)**3 * self.k_t
        if self.SyXRB:
            K_su = self.M_dot(t) * self.eta * self.Omega * self.R_G(t)**2
            R_m = max(R_m, self.R_NS)
            K_su = min(K_su, self.M_dot(t) * (self.GM * R_m)**0.5)
        """
        K_su = self.M_dot(t) * (self.GM * self.R_A(t))**0.5
        P_eq = self.mu(t) * (self.k_t / K_su / self.R_c(1.)**3)**0.5
        # P_eq1 = 2*pi*self.mu(t) * (self.k_t / self.GM / K_su)**0.5
        # print(np.log10(P_eq/P_eq1))
        return P_eq

    def j(self, t):
        j_t = self.v_t * self.R_G(t)**(4/3) / self.R_t**(1/3)
        j_K = (self.GM * self.R_A(t))**0.5
        j = min(j_K, j_t)
        return j_K, j_t

    def E_to_P(self, t, P):
        return P - self.P_EP(t)
        # return max(self.R_l(P), self.R_G(t)) - self.R_Sh(t, P)
    
    def P_to_E(self, t, P):
        # return 1000
        return self.P_PE(t) - P
        # return self.R_Sh(t, P) - self.R_l(P)
    
    def P_to_A(self, t, P):
        # if self.R_c(P) > 2 * self.R_m_cur(t, P):
        #     return 0
        # else:
        return self.R_c(P) - self.R_m_cur(t, P)
    
    def A_to_G(self, t, P):
        if not self.Georotators:
            return 1000
        if self.case == "A1":
            return self.R_m_A1(t, P) - self.R_G(t)
        return self.R_m(t) - self.R_G(t)
        
    def A_to_P(self, t, P):
        # return 1000
        return self.R_A(t) - self.R_c(P)

    # def G_to_P(self, t, P):
    #     if self.case == "A1":
    #         R_m = self.R_m_A1(t, P)
    #     else:
    #         R_m = self.R_m(t)
    #     if R_m > self.R_c(P):
    #         return 0.
    #     else:
    #         return 1.
        # return self.R_m(t) - self.R_c(P)
        # return -self.P_to_G(t, P)
    
    # def G_to_A(self, t, P):
    #     return -self.A_to_G(t, P)
    
    def A_to_S(self, t, P):
        return P - self.P_turb(t)

    
    E_to_P.terminal = True
    P_to_E.terminal = True
    P_to_A.terminal = True
    A_to_P.terminal = True
    A_to_G.terminal = True
    # G_to_P.terminal = True
    # G_to_A.terminal = True
    A_to_S.terminal = True

    P_to_A.direction = 1.
    P_to_E.direction = 1.
    E_to_P.direction = 1.
    A_to_P.direction = 1.
    A_to_G.direction = 1.
    # G_to_P.direction = 1.
    # G_to_A.direction = 1.
    A_to_S.direction = 1.
    

    @abstractmethod
    def solution(self, t, P, t_eval_Array, max_step, dense_output=True):
        message = "Not calculated"
        method = 'Radau' # 'LSODA'# 'DOP853' #
        # t_end = np.log10(self.t_stop)
        try:
            events, next_stage, P_last = self.get_term_events()
        except BaseException as e:
            print('ERROR in termEvents:', e)
            print(self.get_term_events())
        try:
            # t_eval_Array = t_eval_Array # np.geomspace(np.log10(t), t_end,
                                        # number_of_dots, endpoint=True)
            solution = solve_ivp(self.dP_dt,
                                 (t_eval_Array[0], t_eval_Array[-1]),
                                 P, method=method,
                                 t_eval=t_eval_Array, events=events,
                                 dense_output=dense_output, max_step=max_step)
            print(solution.message)
            message = solution.message
        
        except BaseException as e:
            print('THERE IS AN ERROR in model.Simulation().solution')
            print("t_eval: ", t_eval_Array[0], t_eval_Array[-1])
            print(self.get_term_events())
            print(solution.message)
            return message, e, None
        q = -1
        
        if solution.t_events:
            for (i, t_ev) in enumerate(solution.t_events):
                if len(t_ev):
                    q = i
                    break
            if q == -1:
                return message, solution, None
            else:
                if next_stage[q] == None:
                    return message, solution, None
                
                if (P_last[q] and P_last[q] is not int and
                    events[q].terminal == 1):
                    solution.y[0][-1] = P_last[q](solution.t[-1])
                    
                if (self.Georotators and
                    str(next_stage[q]) == "<class 'model.Accretor'>" and
                    events[q].terminal == 1):
                    if self.R_m(solution.t[-1]) > self.R_G(solution.t[-1]):
                        next_stage[q] = Georotator
                        if P_last[q]:
                            solution.y[0][-1] = P_last[q](solution.t[-1])

                return message, solution, next_stage[q](self.B0, self.v0,
                                                        self.n0, self.field,
                                                        self.case, self.Omega,
                                                        self.p_out_0)
        return message, solution, None


class Ejector(Simulation):
    """ Stage 'a' in Davies & Pringle 1981 """
    def __init__(self, B, v, n, field, case='A', Omega=None, p_out=0.):
        super().__init__(B=B, v=v, n=n, field=field, case=case, Omega=Omega, p_out=p_out)

    def dP_dt(self, t, P):
        dP_dt = 4 * self.k * pi**2 * self.mu(t)**2 / (c**3 * self.I * P)
        return dP_dt
    
    def get_term_events(self):
        if self.calcPeriod:
            Parr = [self.P_EP]
        else:
            Parr = [0]
        return [self.E_to_P], [Propeller], Parr
    
    def title(self):
        return 'E'


class Propeller(Simulation):
    """ Propeller in Shakura 1975 """
    def __init__(self, B, v, n, field, case='A', Omega=None, p_out=0.):
        super().__init__(B=B, v=v, n=n, field=field, case=case, Omega=Omega, p_out=p_out)
    
    def get_term_events(self):
        if self.calcPeriod:
            Parr = [self.P_PA, self.P_PE]
        else:
            Parr = [0, 0]
            
        return ([self.P_to_A, self.P_to_E],
                [Accretor, Ejector], Parr)

    def K_P(self, t, P):
        R_m = self.R_m_cur(t, P)
        R_m = min(R_m, self.R_l(P))
        """ If disk """
        if self.Omega:
            if (self.GM*R_m)**0.5 <= self.eta*self.Omega*self.R_G(t)**2:
                R_m = self.R_A(t)
        
        if self.case == "A" or self.case == "A1":
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
            print("There are only 5 propeller cases: A1, A, B, C and D, no", self.case)
        return K_P

    def dP_dt(self, t, P):
        dP_dt = P**2 / (2 * pi * self.I) * self.K_P(t, P)
        return dP_dt
    
    def title(self):
        return 'P'


class Accretor(Simulation):
    """ Accretor in article_PASA """
    def __init__(self, B, v, n, field, case='A', Omega=None, p_out=0.):
        super().__init__(B=B, v=v, n=n, field=field, case=case, Omega=Omega, p_out=p_out)
   
    def get_term_events(self):
        if self.calcPeriod:
            Parr = self.P_AP
        else:
            Parr = 0
        if self.settle:
            return ([self.A_to_P, self.A_to_G, self.A_to_S],
                    [Propeller, Georotator, Settle], [Parr, 0, 0])
        else:
            return ([self.A_to_P, self.A_to_G],
                    [Propeller, Georotator], [Parr, 0])
   
    def dP_dt(self, t, P):
        R_m  = self.R_A(t)
        if self.settle:
            """ Settling Accretor """
            omega = 2 * np.pi / P
            K_SA = 8 * self.M_dot(t) * R_m**2 * omega
            dP_dt = P**2 / (2 * np.pi * self.I) * K_SA
        else:
            """ Lipunov Accretor """
            K_sd = self.mu(t)**2 / self.R_c(P)**3 * self.k_t
            if self.Omega:
                K_su = self.M_dot(t) * self.eta * self.Omega * self.R_G(t)**2
                R_m = max(R_m, self.R_NS)
                K_su = min(K_su, self.M_dot(t) * (self.GM * R_m)**0.5)
            else:
                K_su = 0
            dP_dt = P**2 / (2 * pi * self.I) * (K_sd - K_su)

        return dP_dt
    
    def title(self):
        return 'A'

class Settle(Simulation):
    def __init__(self, B, v, n, field, case='A', Omega=None, p_out=0.):
        super().__init__(B=B, v=v, n=n, field=field, case=case, Omega=Omega, p_out=p_out)
    
    def get_term_events(self):
        return ([self.A_to_P, self.A_to_G],
                [Propeller, Georotator], [0, 0])
    
    def solution(self, t, P, t_eval_Array, max_step, dense_output=True):
        message = "settling accretor"
        try:
            events, next_stage, P_last = self.get_term_events()
        except BaseException as e:
            print('ERROR in termEvents:', e)
            print(self.get_term_events())
        
        # P = self.P_turb(t_eval_Array)
        # ev = np.zeros(len(events))
        # for q in range(len(events)):
        #     ev[q] = events[q](t_eval_Array, P)


        solution = t_eval_Array, self.P_turb(t_eval_Array)
        return message, solution, "settle" # next_stage[q](self.B, self.v, self.n,
                          #             self.field, self.t_start,
                           #            self.t_stop, self.case)
    
    def title(self):
        return 'SA'

class Georotator(Simulation):
    """ Georotator """
    def __init__(self, B, v, n, field, case='A', Omega=None, p_out=0.):
        super().__init__(B=B, v=v, n=n, field=field, case=case, Omega=Omega, p_out=p_out)
   
    def get_term_events(self):
        return [], [Accretor, Propeller], [0, 0] #[self.G_to_A, self.G_to_P], [Accretor, Propeller], [0, 0]

    def K_sd(self, t, P):
        """ Decelerating moment of inertia on Accretor stage """ #!!!
        """ Магнитосфера R_m_geo теперь!!!!! """

        return 0. # 1 / 3 * self.mu(t)**2 / self.R_c(P)**3
    
    def dP_dt(self, t, P):
        """ Period change over time on Accretor stage """
        dP_dt = P**2 * (self.K_sd(t, P)) / (2 * pi * self.I)
        return dP_dt
    
    def solution(self, t, P, t_eval_Array, max_step, dense_output=True):
        message = "Georotator"
        try:
            events, next_stage, P_last = self.get_term_events()
        except BaseException as e:
            print('ERROR in Georotator termEvents:', e)
            print(self.get_term_events())
        
        t_eval_Array = t_eval_Array[1:-1]
        
        R_c = np.zeros(len(t_eval_Array)) + self.R_c(P)
        if self.case == "A1":
            R_m = np.array(self.R_m_A1(t_eval_Array,
                                       P + np.zeros(len(t_eval_Array))))
        else:
            R_m = np.array(self.R_m_geo(t_eval_Array))
        R_G = np.array(self.R_G(t_eval_Array))

        """ Initial difference """
        delta_R_c = abs(R_c[0] - R_m[0])
        # if delta_R_c / R_m[0]> 0.1:
        #     delta_R_c = 0
        # delta_R_G = abs(R_G[0] - R_m[0])
        # if delta_R_G / R_m[0]> 0.1:
        #     delta_R_G = 0
            
        delta_R_G = 0
        # delta_R_c = 0
        
        
        R_c[delta_R_c+R_c<R_m] = 0 # zeros[R_c<R_m]
        zeros1 = np.where(R_c==0)[0]
        if len(zeros1) > 0:
            idx1 = zeros1[0]
            print('idx1 =',idx1)
        else:
            idx1 = None

        R_G[R_G>R_m-delta_R_G] = 0
        zeros2 = np.where(R_G==0)[0]
        if len(zeros2) > 0:
            idx2 = zeros2[0]
            print('idx2 =', idx2)
            print('delta_R_c =', delta_R_c)
        else:
            idx2 = None
        
        if (idx1 is None) and (idx2 is None):
            message = "Georotator ends"
        else:
            if idx1 is None:
                idx = idx2
                idx1 = np.inf
            elif idx2 is None:
                idx = idx1
                idx2 = np.inf
            else:
                idx = min(idx1, idx2)
            if not idx:
                print(idx, "in Georotator")
                idx = 1#len(t_eval_Array) // 10
            t_eval_Array = t_eval_Array[0:idx+1]
            P = np.zeros(len(t_eval_Array)) + P
            if idx1 < idx2:
                stage = Propeller
            else:
                stage = Accretor
            solution = t_eval_Array, P
            
            return message, solution, stage(self.B, self.v, self.n,
                                                    self.field, self.case,
                                                    self.Omega)

        return message, solution, None
    
    def title(self):
        return 'G'