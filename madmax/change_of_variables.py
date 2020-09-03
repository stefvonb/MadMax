# -*- coding: utf-8 -*- 

import math, random
import numpy as np
from madmax.objects import FourMomentum, Event, Particle, ThreeVector
from madmax.utils import remove_complex
from . import FP_TOLERANCE, PhysicsError
from copy import deepcopy

def set_mass(m, p4):
    if m < 0:
        try:
            return p4.mass()
        except PhysicsError:
            return 0.0
    else:
        return m

def valid_energy(E):
    if E < 0.0:
        return False
    elif isinstance(E, complex):
        return False
    elif math.isnan(E):
        return False
    else:
        return True

def class_A():
    """

    """
    pass

def class_B(event_list, particle_keys, sqrt_s12, m1=0, m2=-1):
    """
    Gets the missing momentum for the following topology:
            
     x1 ╲     s_12 ╭── p2  (visible)
         █████┄┄┄┄┄┤          
     x2 ╱          ╰── p1  (missing)    
                   
    m2 can be specified, if -1 it is calculated
    """
    solutions = []
    if not isinstance(event_list, list):
        if isinstance(event_list, Event):
            event_list = [event_list]
        else:
            raise TypeError("'event_list' should be a list of Event objects")
    
    for event in event_list:
        p2 = event[particle_keys[1]]
        p_branches = FourMomentum(0.0, 0.0, 0.0, 0.0)
        for k in event.particles:
            if event.is_visible(k):
                p_branches += event[k]
        
        if particle_keys[0] not in event.particles:
            event.add_particle(particle_keys[0], Particle(0, 1, FourMomentum()))
        
        p1x = -1*p_branches.x
        p1y = -1*p_branches.y

        m2 = set_mass(m2, p2)

        s12 = sqrt_s12**2

        R12 = (s12 - m1**2 - m2**2)/2.0

        # Solve polynomial equation for p1z (Ax^2 + Bx + C)
        A = 1 - p2.z**2/p2.E**2
        B = -2.0*p2.z/p2.E**2*(R12 + p1x*p2.x + p1y*p2.y)
        C = m1**2 + p1x**2 + p1y**2 - 1.0/p2.E**2*(R12**2 + p1x**2*p2.x**2 + p1y**2*p2.y**2 + \
            2*(R12*p1x*p2.x + R12*p1y*p2.y + p1x*p2.x*p1y*p2.y))
        p1z_sols = np.roots([A, B, C])
        E1_sols = (m1**2 + p1x**2 + p1y**2 + p1z_sols**2)**0.5

        E1_sols = remove_complex(E1_sols)

        for E1, p1z in zip(E1_sols, p1z_sols):
            if not valid_energy(E1):
                continue
            p1_sol = FourMomentum(E1, p1x, p1y, p1z)
            if abs(p1_sol.m2 - m1**2) > FP_TOLERANCE:
                continue
            x1 = (E1 + p1z + p_branches.E + p_branches.z)/event.sqrt_s
            x2 = (E1 - p1z + p_branches.E - p_branches.z)/event.sqrt_s
            if x1 < 0.0 or x2 < 0.0 or x1 > 1.0 or x2 > 1.0:
                continue
            output_event = deepcopy(event)
            output_event.set_momentum(particle_keys[0], p1_sol)
            output_event.x1 = x1
            output_event.x2 = x2
            solutions.append(output_event)

    return solutions

def class_C():
    pass

def class_D(event_list, particle_keys, sqrt_s134, sqrt_s256, 
    sqrt_s13, sqrt_s25, m1=0, m2=0, m3=-1, m4=-1, m5=-1, m6=-1):
    """
    Gets the two missing momenta for the following topology:
                   ╭────────── missing (p1)
     x1 ╲    s_13 ╱╰────────── visible (p3)
         ╲       ╱──────────── visible (p4)
          ╲     ╱ s_134
           █████
          ╱     ╲ s_256
         ╱       ╲──────────── visible (p6)
     x2 ╱    s_25 ╲╭────────── visible (p5)
                   ╰────────── missing (p2)
    m3, ..., m6 can be specified, if -1 they are calculated
    """
    solutions = []
    if not isinstance(event_list, list):
        if isinstance(event_list, Event):
            event_list = [event_list]
        else:
            raise TypeError("'event_list' should be a list of Event objects")
    
    for event in event_list:
        p3 = event[particle_keys[2]]
        p4 = event[particle_keys[3]]
        p5 = event[particle_keys[4]]
        p6 = event[particle_keys[5]]
        p_branches = FourMomentum(0.0, 0.0, 0.0, 0.0)
        for k in event.particles:
            if event.is_visible(k):
                p_branches += event[k]

        [event.add_particle(key, Particle(0, 1, FourMomentum())) for key in particle_keys[:2] \
            if not key in event.particles]

        m3 = set_mass(m3, p3)
        m4 = set_mass(m4, p4)
        m5 = set_mass(m5, p5)
        m6 = set_mass(m6, p6)
        
        s13 = sqrt_s13**2
        s25 = sqrt_s25**2
        s134 = sqrt_s134**2
        s256 = sqrt_s256**2

        R13 = (s13 - m3**2 - m1**2)/2.0
        R25 = (s25 - m5**2 - m2**2)/2.0
        R134 = (s134 - m1**2 - m3**2 - m4**2 - 2*(p3*p4))/2.0
        R256 = (s256 - m2**2 - m5**2 - m6**2 - 2*(p5*p6))/2.0
        R14 = R134 - R13
        R26 = R256 - R25

        # First define coefficients for vector equations:
        # p1 = a1 + b1*E1 + c1*E2 / p2 = a2 + b2*E2 + c2*E1
        # Numbers from SymPy
        PF = 1/(p3.x*p4.z*p5.y*p6.z - p3.x*p4.z*p5.z*p6.y - \
            p3.y*p4.z*p5.x*p6.z + p3.y*p4.z*p5.z*p6.x - \
            p3.z*p4.x*p5.y*p6.z + p3.z*p4.x*p5.z*p6.y + \
            p3.z*p4.y*p5.x*p6.z - p3.z*p4.y*p5.z*p6.x)
        a1 = PF*np.array(
            [-(R13*p4.z*p5.y*p6.z - R13*p4.z*p5.z*p6.y - R14*p3.z*p5.y*p6.z + \
            R14*p3.z*p5.z*p6.y + R25*p3.y*p4.z*p6.z - R25*p3.z*p4.y*p6.z - \
            R26*p3.y*p4.z*p5.z + R26*p3.z*p4.y*p5.z - p3.y*p4.z*p5.x*p6.z*p_branches.x - \
            p3.y*p4.z*p5.y*p6.z*p_branches.y + p3.y*p4.z*p5.z*p6.x*p_branches.x + \
            p3.y*p4.z*p5.z*p6.y*p_branches.y + p3.z*p4.y*p5.x*p6.z*p_branches.x + \
            p3.z*p4.y*p5.y*p6.z*p_branches.y - p3.z*p4.y*p5.z*p6.x*p_branches.x - \
            p3.z*p4.y*p5.z*p6.y*p_branches.y),
            (R13*p4.z*p5.x*p6.z - R13*p4.z*p5.z*p6.x - R14*p3.z*p5.x*p6.z + \
            R14*p3.z*p5.z*p6.x + R25*p3.x*p4.z*p6.z - R25*p3.z*p4.x*p6.z - \
            R26*p3.x*p4.z*p5.z + R26*p3.z*p4.x*p5.z - p3.x*p4.z*p5.x*p6.z*p_branches.x - \
            p3.x*p4.z*p5.y*p6.z*p_branches.y + p3.x*p4.z*p5.z*p6.x*p_branches.x + \
            p3.x*p4.z*p5.z*p6.y*p_branches.y + p3.z*p4.x*p5.x*p6.z*p_branches.x + \
            p3.z*p4.x*p5.y*p6.z*p_branches.y - p3.z*p4.x*p5.z*p6.x*p_branches.x - \
            p3.z*p4.x*p5.z*p6.y*p_branches.y),
            (R13*p4.x*p5.y*p6.z - R13*p4.x*p5.z*p6.y - R13*p4.y*p5.x*p6.z + R13*p4.y*p5.z*p6.x - \
            R14*p3.x*p5.y*p6.z + R14*p3.x*p5.z*p6.y + R14*p3.y*p5.x*p6.z - R14*p3.y*p5.z*p6.x - \
            R25*p3.x*p4.y*p6.z + R25*p3.y*p4.x*p6.z + R26*p3.x*p4.y*p5.z - R26*p3.y*p4.x*p5.z + \
            p3.x*p4.y*p5.x*p6.z*p_branches.x + p3.x*p4.y*p5.y*p6.z*p_branches.y - \
            p3.x*p4.y*p5.z*p6.x*p_branches.x - p3.x*p4.y*p5.z*p6.y*p_branches.y - \
            p3.y*p4.x*p5.x*p6.z*p_branches.x - p3.y*p4.x*p5.y*p6.z*p_branches.y + \
            p3.y*p4.x*p5.z*p6.x*p_branches.x + p3.y*p4.x*p5.z*p6.y*p_branches.y)],
            dtype=np.float
        )
        b1 = PF*np.array(
            [-(-p3.E*p4.z*p5.y*p6.z + p3.E*p4.z*p5.z*p6.y + p3.z*p4.E*p5.y*p6.z - p3.z*p4.E*p5.z*p6.y),
            -p3.E*p4.z*p5.x*p6.z + p3.E*p4.z*p5.z*p6.x + p3.z*p4.E*p5.x*p6.z - p3.z*p4.E*p5.z*p6.x,
            (-p3.E*p4.x*p5.y*p6.z + p3.E*p4.x*p5.z*p6.y + p3.E*p4.y*p5.x*p6.z - p3.E*p4.y*p5.z*p6.x + \
            p3.x*p4.E*p5.y*p6.z - p3.x*p4.E*p5.z*p6.y - p3.y*p4.E*p5.x*p6.z + p3.y*p4.E*p5.z*p6.x)],
            dtype=np.float
        )
        c1 = PF*np.array(
            [-(-p3.y*p4.z*p5.E*p6.z + p3.y*p4.z*p5.z*p6.E + p3.z*p4.y*p5.E*p6.z - p3.z*p4.y*p5.z*p6.E),
            -p3.x*p4.z*p5.E*p6.z + p3.x*p4.z*p5.z*p6.E + p3.z*p4.x*p5.E*p6.z - p3.z*p4.x*p5.z*p6.E,
            (p3.x*p4.y*p5.E*p6.z - p3.x*p4.y*p5.z*p6.E - p3.y*p4.x*p5.E*p6.z + p3.y*p4.x*p5.z*p6.E)],
            dtype=np.float
        )
        a2 = PF*np.array(
            [R13*p4.z*p5.y*p6.z - R13*p4.z*p5.z*p6.y - R14*p3.z*p5.y*p6.z + R14*p3.z*p5.z*p6.y + \
            R25*p3.y*p4.z*p6.z - R25*p3.z*p4.y*p6.z - R26*p3.y*p4.z*p5.z + R26*p3.z*p4.y*p5.z - \
            p3.x*p4.z*p5.y*p6.z*p_branches.x + p3.x*p4.z*p5.z*p6.y*p_branches.x - \
            p3.y*p4.z*p5.y*p6.z*p_branches.y + p3.y*p4.z*p5.z*p6.y*p_branches.y + \
            p3.z*p4.x*p5.y*p6.z*p_branches.x - p3.z*p4.x*p5.z*p6.y*p_branches.x + \
            p3.z*p4.y*p5.y*p6.z*p_branches.y - p3.z*p4.y*p5.z*p6.y*p_branches.y,
            -(R13*p4.z*p5.x*p6.z - R13*p4.z*p5.z*p6.x - R14*p3.z*p5.x*p6.z + R14*p3.z*p5.z*p6.x + \
            R25*p3.x*p4.z*p6.z - R25*p3.z*p4.x*p6.z - R26*p3.x*p4.z*p5.z + R26*p3.z*p4.x*p5.z - \
            p3.x*p4.z*p5.x*p6.z*p_branches.x + p3.x*p4.z*p5.z*p6.x*p_branches.x - \
            p3.y*p4.z*p5.x*p6.z*p_branches.y + p3.y*p4.z*p5.z*p6.x*p_branches.y + \
            p3.z*p4.x*p5.x*p6.z*p_branches.x - p3.z*p4.x*p5.z*p6.x*p_branches.x + \
            p3.z*p4.y*p5.x*p6.z*p_branches.y - p3.z*p4.y*p5.z*p6.x*p_branches.y),
            R13*p4.z*p5.x*p6.y - R13*p4.z*p5.y*p6.x - R14*p3.z*p5.x*p6.y + R14*p3.z*p5.y*p6.x + \
            R25*p3.x*p4.z*p6.y - R25*p3.y*p4.z*p6.x - R25*p3.z*p4.x*p6.y + R25*p3.z*p4.y*p6.x - \
            R26*p3.x*p4.z*p5.y + R26*p3.y*p4.z*p5.x + R26*p3.z*p4.x*p5.y - R26*p3.z*p4.y*p5.x - \
            p3.x*p4.z*p5.x*p6.y*p_branches.x + p3.x*p4.z*p5.y*p6.x*p_branches.x - \
            p3.y*p4.z*p5.x*p6.y*p_branches.y + p3.y*p4.z*p5.y*p6.x*p_branches.y + \
            p3.z*p4.x*p5.x*p6.y*p_branches.x - p3.z*p4.x*p5.y*p6.x*p_branches.x + \
            p3.z*p4.y*p5.x*p6.y*p_branches.y - p3.z*p4.y*p5.y*p6.x*p_branches.y],
            dtype=np.float
        )
        b2 = PF*np.array(
            [-p3.E*p4.z*p5.y*p6.z + p3.E*p4.z*p5.z*p6.y + p3.z*p4.E*p5.y*p6.z - p3.z*p4.E*p5.z*p6.y,
            -(-p3.E*p4.z*p5.x*p6.z + p3.E*p4.z*p5.z*p6.x + p3.z*p4.E*p5.x*p6.z - p3.z*p4.E*p5.z*p6.x),
            -p3.E*p4.z*p5.x*p6.y + p3.E*p4.z*p5.y*p6.x + p3.z*p4.E*p5.x*p6.y - p3.z*p4.E*p5.y*p6.x],
            dtype=np.float
        )
        c2 = PF*np.array(
            [-p3.y*p4.z*p5.E*p6.z + p3.y*p4.z*p5.z*p6.E + p3.z*p4.y*p5.E*p6.z - p3.z*p4.y*p5.z*p6.E,
            -(-p3.x*p4.z*p5.E*p6.z + p3.x*p4.z*p5.z*p6.E + p3.z*p4.x*p5.E*p6.z - p3.z*p4.x*p5.z*p6.E),
            -p3.x*p4.z*p5.E*p6.y + p3.x*p4.z*p5.y*p6.E + p3.y*p4.z*p5.E*p6.x - p3.y*p4.z*p5.x*p6.E + \
            p3.z*p4.x*p5.E*p6.y - p3.z*p4.x*p5.y*p6.E - p3.z*p4.y*p5.E*p6.x + p3.z*p4.y*p5.x*p6.E],
            dtype=np.float
        )

        # Enforcing the mass-shell equation for p1:
        # E1^2 = m1^2 + p1·p1
        # 0 = A*E1^2 + (B0 + B1*E2)*E1 + C0 + C1*E2 + C2*E2^2
        A = np.dot(b1, b1) - 1
        B0 = 2*np.dot(a1, b1)
        B1 = 2*np.dot(b1, c1)
        C0 = np.dot(a1, a1) + m1**2
        C1 = 2*np.dot(a1, c1)
        C2 = np.dot(c1, c1)
        # 0 = D*E^2 + (F0 + F1*E1)*E2 + G0 + G1*E1 + G2*E1^2
        D = np.dot(c2, c2) - 1
        F0 = 2*np.dot(a2, c2)
        F1 = 2*np.dot(b2, c2)
        G0 = np.dot(a2, a2) + m2**2
        G1 = 2*np.dot(a2, b2)
        G2 = np.dot(b2, b2)

        E2_sols = np.roots(np.nan_to_num([(-A**2*D**2 + A*(B1*D*F1 + 2*C2*D*G2 - C2*F1**2) + \
            G2*(-B1**2*D + B1*C2*F1 - C2**2*G2))/A**2,
            (-2*A**2*D*F0 + A*(B0*D*F1 + B1*D*G1 + B1*F0*F1 + 2*C1*D*G2 - C1*F1**2 + \
            2*C2*F0*G2 - 2*C2*F1*G1) + G2*(-2*B0*B1*D + B0*C2*F1 - B1**2*F0 + B1*C1*F1 + \
            B1*C2*G1 - 2*C1*C2*G2))/A**2,
            (-A**2*(2*D*G0 + F0**2) + A*(B0*D*G1 + B0*F0*F1 + B1*F0*G1 + B1*F1*G0 + \
            2*C0*D*G2 - C0*F1**2 + 2*C1*F0*G2 - 2*C1*F1*G1 + 2*C2*G0*G2 - C2*G1**2) + \
            G2*(-B0**2*D - 2*B0*B1*F0 + B0*C1*F1 + B0*C2*G1 - B1**2*G0 + B1*C0*F1 + B1*C1*G1 - \
            2*C0*C2*G2 - C1**2*G2))/A**2,
            (-2*A**2*F0*G0 + A*(B0*F0*G1 + B0*F1*G0 + B1*G0*G1 + 2*C0*F0*G2 - 2*C0*F1*G1 + \
            2*C1*G0*G2 - C1*G1**2) + G2*(-B0**2*F0 - 2*B0*B1*G0 + B0*C0*F1 + B0*C1*G1 + B1*C0*G1 - 2*C0*C1*G2))/A**2,
            (-A**2*G0**2 + A*(B0*G0*G1 + 2*C0*G0*G2 - C0*G1**2) + G2*(-B0**2*G0 + B0*C0*G1 - C0**2*G2))/A**2]))
        E2_sols = remove_complex(E2_sols)

        # For each E2, there will be 2 p1's, one will have the wrong mass
        for E2_sol in E2_sols:
            if not valid_energy(E2_sol):
                continue
            for pm in [-1, 1]:
                E1_sol = (-B0 - B1*E2_sol + pm*(((B0 + B1*E2_sol)**2 - 4*A*(C0 + C1*E2_sol + C2*E2_sol**2))**0.5))/(2*A)
                p1_sol = FourMomentum(E1_sol, *(a1 + E1_sol*b1 + E2_sol*c1))
                if abs(p1_sol.m2 - m1**2) > FP_TOLERANCE or not valid_energy(E1_sol):
                    continue
                p2_sol = FourMomentum(E2_sol, *(a2 + E1_sol*b2 + E2_sol*c2))
                if abs(p2_sol.m2 - m2**2) > FP_TOLERANCE:
                    continue
                x1 = (p1_sol.E + p1_sol.z + p_branches.E + p_branches.z + p2_sol.E + p2_sol.z)/event.sqrt_s
                x2 = (p1_sol.E - p1_sol.z + p_branches.E - p_branches.z + p2_sol.E - p2_sol.z)/event.sqrt_s
                if x1 < 0.0 or x2 < 0.0 or x1 > 1.0 or x2 > 1.0:
                    continue
                output_event = deepcopy(event)
                output_event.set_momentum(particle_keys[0], p1_sol)
                output_event.set_momentum(particle_keys[1], p2_sol)
                output_event.x1 = x1
                output_event.x2 = x2
                solutions.append(output_event)

    return solutions

def class_E(event_list, particle_keys, sqrt_s13, sqrt_s24, sqrt_s_hat, rapidity, 
             m1=0, m2=0, m3=-1, m4=-1):
    """
    Gets the two missing momenta for the following topology:
                     ╭─────── missing (p1)
    x1 ╲       s_13 ╱╰─────── visible (p3)
        ╲          ╱
         ███┄┄┄┄███
        ╱   s_hat  ╲ 
    x2 ╱       s_24 ╲╭─────── missing (p2)
                     ╰─────── visible (p4)
    x1 and x2 can be set manually for debugging
    m1, ..., m4 can be specified, if -1 they are calculated
    """
    solutions = []
    if not isinstance(event_list, list):
        if isinstance(event_list, Event):
            event_list = [event_list]
        else:
            raise TypeError("'event_list' should be a list of Event objects")
    
    for event in event_list:
        # Set up the kinematic variables
        s_13, s_24 = sqrt_s13**2, sqrt_s24**2

        p3 = event[particle_keys[2]]
        p4 = event[particle_keys[3]]
        p_branches = FourMomentum(0.0, 0.0, 0.0, 0.0)
        for k in event.particles:
            if event.is_visible(k):
                p_branches += event[k]

        [event.add_particle(key, Particle(0, 1, FourMomentum())) for key in particle_keys[:2] \
            if not key in event.particles]

        x1 = math.exp(rapidity)*sqrt_s_hat/event.sqrt_s
        x2 = math.exp(-rapidity)*sqrt_s_hat/event.sqrt_s

        m3 = set_mass(m3, p3)
        m4 = set_mass(m4, p4)
        
        p_i1 = FourMomentum(x1*event.sqrt_s/2, 0.0, 0.0, x1*event.sqrt_s/2)
        p_i2 = FourMomentum(x2*event.sqrt_s/2, 0.0, 0.0, -x2*event.sqrt_s/2)
        p_vis = p_i1 + p_i2 - p_branches

        Rvis = (m2**2 - m1**2 + p_vis*p_vis)/2
        R3 = p_vis*p3 - (s_13 - m1**2 - m3**2)/2
        R4 = (s_24 - m2**2 - m4**2)/2

        # Solved equations in terms of E2 from SymPy
        # p = E2*v_s + v_i <- vector equation
        PF = 1/(p3.x*p4.y*p_vis.z - p3.x*p4.z*p_vis.y - p3.y*p4.x*p_vis.z + \
            p3.y*p4.z*p_vis.x + p3.z*p4.x*p_vis.y - p3.z*p4.y*p_vis.x)
        v_i = PF*np.array([-(R3*p4.y*p_vis.z - R3*p4.z*p_vis.y - R4*p3.y*p_vis.z + \
                        R4*p3.z*p_vis.y + Rvis*p3.y*p4.z - Rvis*p3.z*p4.y),
                        (R3*p4.x*p_vis.z - R3*p4.z*p_vis.x - R4*p3.x*p_vis.z + \
                        R4*p3.z*p_vis.x + Rvis*p3.x*p4.z - Rvis*p3.z*p4.x),
                        -(R3*p4.x*p_vis.y - R3*p4.y*p_vis.x - R4*p3.x*p_vis.y + \
                        R4*p3.y*p_vis.x + Rvis*p3.x*p4.y - Rvis*p3.y*p4.x)], 
                    dtype=np.float)
        v_s = PF*np.array([-(-p3.E*p4.y*p_vis.z + p3.E*p4.z*p_vis.y + p3.y*p4.E*p_vis.z - \
                        p3.y*p4.z*p_vis.E - p3.z*p4.E*p_vis.y + p3.z*p4.y*p_vis.E),
                        (-p3.E*p4.x*p_vis.z + p3.E*p4.z*p_vis.x + p3.x*p4.E*p_vis.z - \
                        p3.x*p4.z*p_vis.E - p3.z*p4.E*p_vis.x + p3.z*p4.x*p_vis.E),
                        -(-p3.E*p4.x*p_vis.y + p3.E*p4.y*p_vis.x + p3.x*p4.E*p_vis.y - \
                        p3.x*p4.y*p_vis.E - p3.y*p4.E*p_vis.x + p3.y*p4.x*p_vis.E)],
                    dtype=np.float)

        # Now the solutions can be taken from E2^2 = m2^2 + p2·p2
        # Using the symbols above, we have A*E2^2 + B*E2 + C = 0
        A = 1 - np.dot(v_s, v_s)
        B = -2*(np.dot(v_s, v_i))
        C = -(m2**2 + np.dot(v_i, v_i))

        E2_sols = np.roots(np.nan_to_num([A, B, C]))
        E2_sols = remove_complex(E2_sols)
        
        p2_sols = [FourMomentum(E2_sol, *(E2_sol*v_s + v_i)) for E2_sol in E2_sols]
        p1_sols = [p_vis - p2_sol for p2_sol in p2_sols]

        solutions = []
        for p1_sol, p2_sol in zip(p1_sols, p2_sols):
            if valid_energy(p1_sol.E) and valid_energy(p2_sol.E):
                output_event = deepcopy(event)
                output_event.set_momentum(particle_keys[0], p1_sol)
                output_event.set_momentum(particle_keys[1], p2_sol)
                output_event.x1 = x1
                output_event.x2 = x2
                solutions.append(output_event)

    return solutions

def class_F():
    pass

def block_A():
    pass

def block_B():
    pass

def block_C():
    pass

def block_D(event_list, particle_keys, sqrt_s12, theta_1=None, phi_1=None,
    m1=0, m2=-1):
    """
    s_12  ╭──  p2  (visible)
    ┄┄┄┄┄┄┤
          ╰── |p1| (missing)
    """
    solutions = []
    if not isinstance(event_list, list):
        if isinstance(event_list, Event):
            event_list = [event_list]
        else:
            raise TypeError("'event_list' should be a list of Event objects")
    
    for event in event_list:
        p2 = event[particle_keys[1]]
        if particle_keys[0] not in event.particles:
            event.add_particle(particle_keys[0], Particle(0, 1, FourMomentum()))
        p1 = event[particle_keys[0]]

        m2 = set_mass(m2, p2)

        s12 = sqrt_s12**2
        R12 = (s12 - m1**2 - m2**2)/2.0
        
        # Make unit vectors to get cos(angle)
        theta_1, phi_1 = p1.theta, p1.phi
        theta_2, phi_2 = p2.theta, p2.phi
        u1 = ThreeVector.from_mag_and_angles(1.0, theta_1, phi_1)
        u2 = ThreeVector.from_mag_and_angles(1.0, theta_2, phi_2)
        cos_angle_difference = u1*u2

        # Now solve 2nd order polynomial
        denominator = 1/(p2.three_vector().mag2()*cos_angle_difference**2)
        E1_sols = np.roots(np.nan_to_num([1 - p2.E**2*denominator,
                            2*p2.E*R12*denominator,
                            -m1**2 - R12**2*denominator]))
        E1_sols = remove_complex(E1_sols)

        for E1_sol in E1_sols:
            if not valid_energy(E1_sol):
                continue
            p1_mag = (E1_sol**2 - m1**2)**0.5
            if isinstance(p1_mag, complex):
                continue
            output_event = deepcopy(event)
            output_event[particle_keys[0]].four_momentum = \
                FourMomentum.from_mag_and_angles(E1_sol, p1_mag, theta_1, phi_1)
            solutions.append(output_event)

    return solutions

def block_E():
    pass
