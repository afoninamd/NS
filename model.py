#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:31:32 2023

@author: AMD
"""

from astropy import constants as const
from dataclasses import dataclass
import numpy as np
from math import pi
from numpy import exp
from scipy.integrate import solve_ivp
from abc import abstractmethod
from constants import c, G, M_sun, m_p, year, birth_age, galaxy_age, I, R_NS

class Simulation():
    """ Contains common functions and initial parameters """
    def __init__(self, B, v, n, field, t_start=birth_age, t_stop=galaxy_age,
                 case="A"):
        self.case = case
        self.B0 = B
        self.v0 = v # it is c_s in DP
        self.n0 = n
        self.GM = G * 1.4 * M_sun
        self.k = 2.
        self.R_NS = R_NS
        self.I = I
        self.field = field
        self.t_start = t_start
        self.t_stop = t_stop
        self.R_t = 2e20
        self.v_t = 10e5
        self.k_t = 1 / 3 # Accretor coefficient
        self.t_start = t_start
        self.t_stop = t_stop

    def B(self, t):
        if isinstance(self.B0, float) or isinstance(self.B0, int):
            return self.B0
        else:
            return self.B0(t)

    def v(self, t):
        if isinstance(self.v0, float) or isinstance(self.v0, int):
            return self.v0
        else:
            return self.v0(t)

    def n(self, t):
        if isinstance(self.n0, float) or isinstance(self.n0, int):
            return self.n0
        else:
            return self.n0(t)

    def rho(self, t):
        rho = self.n(t) * m_p
        return rho

    def R_G(self, t):
        R_G = 2 * self.GM / self.v(t)**2
        return R_G

    def M_dot(self, t):
        M_dot = pi * self.R_G(t)**2 * self.rho(t) * self.v(t)
        return M_dot

    def first_stage(self, t, P):
        if P <= self.P_EP(t):# and P < self.P_PE(t):
            stage = Ejector
        elif P > self.P_PA(t):
            stage = Accretor
        else:
            stage = Propeller
        return stage(self.B, self.v, self.n, self.field, self.t_start,
                     self.t_stop, self.case)

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
        R_Sh = (self.k * self.mu(t)**2 / self.M_dot(t) * self.GM**2 * omega**4 / self.v(t)**5 / c**4)**0.5
        return R_Sh

    def R_c(self, P):
        R_c = (self.GM * P**2 / (4 * pi**2))**(1/3)
        return R_c

    def R_A(self, t):
        R_A = (self.mu(t)**2 / (8 * self.M_dot(t) * (2 * self.GM)**0.5))**(2/7)
        return R_A

    def R_m(self, t):
        R_m = self.R_A(t)**(7/9) * self.R_G(t)**(2/9)
        return R_m
    
    def R_m_geo(self, t):
        """ 23.III в Липунове """
        R_m = (2 * self.mu(t)**2 / self.M_dot(t) / self.v(t)**2
               * self.GM**2 / self.v(t)**3)**(1/6)
        return R_m

    def P_EP(self, t):
        """ R_Sh = R_G """
        P1 = 2 * pi / c * (self.k * self.mu(t)**2 / (4 * self.M_dot(t) * self.v(t)))**0.25
        """ R_Sh = R_l """
        P2 = 2 * pi / c * (self.k * self.mu(t)**2 / self.v(t)**5 * self.GM**2 / self.M_dot(t))**(1/6) # is it True? Nobody knows
        return min(P1, P2)

    def P_PE(self, t):
        """ R_m = R_l """
        P3 = 2 * pi / c * self.R_m(t)
        return P3
    
    def P_PA(self, t):
        """ R_A = R_c """
        # P_PA = pi * self.mu(t)**(6/7) / (2**0.5 * self.M_dot(t)**(3/7) * self.GM**(5/7))
        """ R_m = R_c """
        P_PA = 2 * pi / (self.GM / self.R_m(t)**3)**0.5
        return P_PA

    # def P_PA_to_P_EP(self, t):
    #     PPP = self.mu(t)**(1/6) * c / (2**(1/3) * self.k**0.25 * self.M_dot(t)**(1/12) * self.GM**(1/3) * self.v(t)**(5/12))
    #     return PPP
    
    def t_P(self, case):
        t = birth_age
        if case == "A":
            t_P = self.I / (self.M_dot(t) * self.R_m(t)**2) * np.log(self.P_PA(t) / self.P_EP(t))
        elif case == "B":
            t_P = 2 * pi * self.I / (self.M_dot(t) * (2 * self.GM * self.R_m(t))**0.5) * (1 / self.P_EP(t) - 1 / self.P_PA(t))
        else:
            print("There are only two Propeller cases: A and B")
            t_P = 0
        return t_P

    def P_min(self, case, t_E=None):
        """ Min P for NS to be Accretor withing galaxy lifetime """
        t = birth_age
        if not t_E:
            t_E = galaxy_age - self.t_P(case)
        # P_min = 2 * pi * (2 * self.k * self.mu(t)**2 * t_E / self.I / c**3 + self.P_EP(t) / (4 * pi**2))**0.5
        P_min = (self.P_EP(t)**2 - 8 * pi**2 * self.k * self.mu(t) / self.I * self.mu(t) / c**2 * t_E / c)**0.5
        return P_min

    def t_E(self, P_0):
        t = 5.
        t_E = self.I / (8*pi**2*self.k*self.mu(t)**2) * c**3 * (self.P_EP(t)**2 - P_0**2)
        return t_E

    # def t_E_max(self):
    #     return self.t_E(0)
        # t = birth_age
        # # wrong t_E_max = pi * self.I / self.mu(t) / (4 * self.k**0.5) * c / (self.M_dot(t) * self.v(t))**0.5
        # t_E_max = self.I * self.P_EP(t)**2 / (8*pi**2*self.k*self.mu(t)**2) * c**3
        # return t_E_max

    def P_turb(self, t):
        v_t = 1e5 * (self.R_G(t) / 1e18)**(1/3) #LipunovPopov1995
        v_t = self.v_t #!!! УТОЧНИТЬ
        P_turb = 3.9e8 * (self.v(t)/1e7)**(43/9)*(self.n(t))**(-2/3)*(self.mu(t)/1e30)**(2/3)*(self.I/1e45)**(1/3)*(self.GM / (G * 1.4 * M_sun))**(-26/9)
        return P_turb

    def j(self, t):
        j_t = self.v_t * self.R_G(t)**(4/3) / self.R_t**(1/3)
        j_K = (self.GM * self.R_m(t))**0.5
        j = min(j_K, j_t)
        return j
    
    def P_cr(self, t):
        """ from Ikhsanov 2001a """
        P_cr = 2 * pi * self.mu(t) / (self.GM * self.M_dot(t) * self.j(t))**0.5 * self.k_t**0.5
        return P_cr

    def E_to_P(self, t, P):
        return P - self.P_EP(t)
    
    def P_to_E(self, t, P):
        return self.R_m(t) - self.R_l(P)
    
    def P_to_A(self, t, P):
        return self.R_c(P) - self.R_m(t)
    
    def P_to_G(self, t, P):
        return self.R_c(P) - self.R_G(t)

    def A_to_G(self, t, P):
        return self.R_m(t) - self.R_G(t)
        
    def A_to_P(self, t, P):
        return -self.P_to_A(t, P)

    def G_to_P(self, t, P):
        return -self.P_to_G(t, P)
    
    def G_to_A(self, t, P):
        return -self.A_to_G(t, P)
    
    def A_to_eq(self, t, P):
        return P - self.P_turb(t)
    
    E_to_P.terminal = True
    P_to_E.terminal = True
    P_to_A.terminal = True
    P_to_G.terminal = True
    A_to_P.terminal = True
    A_to_G.terminal = True
    G_to_P.terminal = True
    G_to_A.terminal = True

    P_to_A.direction = 1.
    P_to_E.direction = 1.
    E_to_P.direction = 1.
    P_to_G.direction = 1.
    A_to_P.direction = 1.
    A_to_G.direction = 1.
    G_to_P.direction = 1.
    G_to_A.direction = 1.

    @abstractmethod
    def solution(self, t, P, t_eval_Array, max_step, dense_output=True):
        method = 'Radau' # 'LSODA'# 'DOP853' #
        # t_end = np.log10(self.t_stop)
        try:
            events, next_stage, P_last = self.get_term_events()
        except BaseException as e:
            print('ERROR in termEvents:', e)
            print(self.get_term_events())
        try:
            t_eval_Array = t_eval_Array # np.geomspace(np.log10(t), t_end,
                                        # number_of_dots, endpoint=True)
            solution = solve_ivp(self.dP_dt,
                                 (t_eval_Array[0], t_eval_Array[-1]),
                                 P, method=method,
                                 t_eval=t_eval_Array, events=events,
                                 dense_output=dense_output, max_step=max_step)
            """ If the stage has < 10 steps """
            # if events == [self.P_to_A, self.P_to_G, self.P_to_E]: # so it is Propeller
            #     print('Propeller')
            #     number_of_dots = 100
            #     t_end = solution.t[-1]
            #     if len(solution.t) > 2:
            #         t_end = solution.t[-1] * (1 + 1 / number_of_dots)
            #     if solution.status == 1:
            #         status = 0
            #         while status == 0:
            #             t_eval_Array = np.geomspace(t, t_end,
            #                                         number_of_dots, endpoint=True)
            #             solution = solve_ivp(self.dP_dt,
            #                                   (t_eval_Array[0], t_eval_Array[-1]),
            #                                   np.array(P), method=method,
            #                                   t_eval=t_eval_Array, events=events,
            #                                   dense_output=True)
            #             t_end = t_end * (1 + 10 / number_of_dots)
            #             status = solution.status
                        
        except BaseException as e:
            print('ERROR in model.Simulation().solution')
            print(self.get_term_events())
            print(solution)
            return e, None
        q = -1
        
        if solution.t_events:
            for (i, t_ev) in enumerate(solution.t_events):
                if len(t_ev):
                    q = i
                    break
            if q == -1:
                return solution, None
            else:
                if next_stage[q] == None:
                    return solution, None
                
                if P_last[q] and P_last[q] is not int:
                    solution.y[0][-1] = P_last[q](solution.t[-1])
                    
                return solution, next_stage[q](self.B, self.v, self.n,
                                               self.field, self.t_start,
                                               self.t_stop, self.case)
        return solution, None


@dataclass
class Ejector(Simulation):
    """ Stage 'a' in Davies & Pringle 1981 """
    def __init__(self, B, v, n, field, t_start, t_stop, case):
        super().__init__(B=B, v=v, n=n, field=field, t_start=t_start, 
                         t_stop=t_stop, case=case)

    def dP_dt(self, t, P):
        dP_dt = 4 * self.k * pi**2 * self.mu(t)**2 / (c**3 * self.I * P)
        return dP_dt
    
    def get_term_events(self):
        return [self.E_to_P], [Propeller], [self.P_EP]


@dataclass
class Propeller(Simulation):
    """ Propeller in Shakura 1975 """
    def __init__(self, B, v, n, field, t_start, t_stop, case):
        super().__init__(B=B, v=v, n=n, field=field, t_start=t_start, 
                         t_stop=t_stop, case=case)

    def get_term_events(self):
        return [self.P_to_A, self.P_to_G, self.P_to_E], [Accretor, Georotator, Ejector], [self.P_PA, 0, self.P_PE]

    def K_P(self, t, P):
        R_m = self.R_m(t)
        if R_m > self.R_G(t):
            R_m = self.R_m_geo(t)
        R_m = min(R_m, self.R_l(P))
        # R_m = self.R_m(t) 
        if self.case == "A":
            K_P = self.M_dot(t) * 2 * pi / P * R_m**2
        elif self.case == "B":
            K_P = self.M_dot(t) * (2 * self.GM * R_m)**0.5
        else:
            print("There are only two propeller cases: A and B")
        return K_P

    def dP_dt(self, t, P):
        dP_dt = P**2 / (2 * pi * self.I) * self.K_P(t, P)
        return dP_dt


@dataclass
class Accretor(Simulation):
    """ Accretor in article_PASA """
    def __init__(self, B, v, n, field, t_start, t_stop, case):
        super().__init__(B=B, v=v, n=n, field=field, t_start=t_start, 
                         t_stop=t_stop, case=case)
   
    def get_term_events(self):
        return [self.A_to_P, self.A_to_G, self.A_to_eq], [Propeller, Georotator, None], [self.P_PA, 0, self.P_cr]
   
    def dP_dt(self, t, P):
        dP_dt = P**2 / (2 * pi * self.I) * self.mu(t)**2 / self.R_c(P)**3 * self.k_t
        return dP_dt


@dataclass
class Georotator(Simulation):
    """ Georotator """
    def __init__(self, B, v, n, field, t_start, t_stop, case):
        super().__init__(B=B, v=v, n=n, field=field, t_start=t_start, 
                         t_stop=t_stop, case=case)
   
    def get_term_events(self):
        return [self.G_to_A, self.G_to_P], [Accretor, Propeller], [0, 0]

    def K_sd(self, t, P):
        """ Decelerating moment of inertia on Accretor stage """ #!!!
        """ Магнитосфера R_m_geo теперь!!!!! """

        return 0#1 / 3 * self.mu(t)**2 / self.R_c(P)**3
    
    def dP_dt(self, t, P):
        """ Period change over time on Accretor stage """
        dP_dt = P**2 * (self.K_sd(t, P)) / (2 * pi * self.I)
        return dP_dt