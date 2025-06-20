#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 1 01:43:30 2022

@author: 6689530
"""

import math 
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt

## Initial values


a_blackhole = 0.15       # semi-major axis for black hole binary in mpc

e_blackhole = 0.0       # eccentricity of black hole binary

G = 1                   #Universal gravitational constant#

# Claculate the unit of mass using megaparsec per year and G = 1
Mass_unit = (3.086 * (10**13))**3/((6.67*10**(-11))* ((24*3600*365)**2))

# convert mass of blackholes to specified units 
blackhole_M1 = (7 * 1.98855 * (10**36))/Mass_unit
blackhole_M2 = (7 * 1.98855 * (10**33))/Mass_unit

# Calculate unit of velocity in megaparsec per year
velocity_unit = (24*3600*365)/(3.08568*(10**13))
                               
# Calculate speed of light in megaparsec per year
c = (2.997)*(10**8) * (velocity_unit)

# total mass of two binary black holes
M_blackhole_binary = blackhole_M1 + blackhole_M2 

# substitution for mass used in Post Newtonian correction
nu = (blackhole_M1 * blackhole_M2)/(M_blackhole_binary**2)

# Calculate the theoretical advance of pericentre per orbit due to General Relativity
dphi_GR = (6 * math.pi * blackhole_M1 * G)/(a_blackhole * (c**2) * (1 - e_blackhole**2))

print(f'The unit of mass using megaparsec per year is {Mass_unit}')
print(f'The unit of velocity in megaparsec per year is {velocity_unit}')
print(f'The speed of light in megaparsec per year is {c}')
print(f'The theoretical advance of pericentre per orbit due to General Relativity is {dphi_GR}')

def create_initial_values_to_file(a, e, M, file_name):
    ''' Function that calculates initial position and velocity from the values of 
    semi-major axis and eccentricity of the two particles. The motion of the two particles
    is modeled as the Keplerian two body problem inn the centre of mass reference frame.
    The initial values are saved to a file with user specified file name
    
    Innput parameters:
        a is the semi-major axis in the two body problem
        e is the eccentricity of the orbit
        M is the total mass of the two objects'''
        
    #r_p = a * (1 - e)   #pericentre for system
    r_a_x = a * (1 + e) #apocentre
    #v_p = math.sqrt((G * M/a)*((1 + e)/(1 - e)))   #pericentre velocity
    v_a_y = math.sqrt((G * M/a)*((1 - e)/(1 + e))) #apocentre velocity

    r_a = [r_a_x, 0, 0]
    v_a = [0, v_a_y, 0]

    with open(file_name, 'w') as file:
        file.write('X' + '\t' + 'Y' + '\t' + 'Z' + '\t' + 
            'velocity X' '\t' + 'velocity Y' + '\t' + 'velocity Z' + '\n')
        
        file.write(str(r_a[0]) + '\t' + str(r_a[1]) + '\t' + str(r_a[2]) + '\t' + 
                   str(v_a[0]) + '\t' + str(v_a[1]) + '\t' + str(v_a[2]))
    file.close()    
    
    return np.asarray(r_a), np.asarray(v_a)



def calculate_orbital_period(r_a, v_a, M, a):
    '''This function calculates the time period for the point mass to complete one orbit
    The orbital period is calculated in units of years
    
    Input parameters:
        r_a is the initial position at the apocentre
        v_a is the initial velocity at the apocentre
        M is the total mass of the two point masses'''
    
    r_magnitude = np.linalg.norm(r_a)
        
    orbital_period = math.sqrt((4 * math.pi**2 * np.linalg.norm(a)**3) / (G * M))
    
    return orbital_period
                


def calculate_a(r_a, v_a, r_magnitude, M, p):
    '''This function calculates the acceleration vector at each position in the orbit 
    for the two body problem. According to the user input p(see description in input parameters)
    The Post Newtonian correction is implemented
    
    Input parameters:
        r_a is the initial position at the apocentre
        v_a is the initial velocity at the apocentre
        M is the total mass of the point masses
        p is the order of PN correction
        p = 0 means without the effects of general relativity
        p = 2 means PN1 correction
        p = 3 means PN2 correction
        p = 4 means PN2.5 correction
        '''
    if p == 0:
        
       return (- (G * M) * r_a) / (r_magnitude**3)

    else:
               
        v_norm = np.linalg.norm(v_a)
        r_norm = np.linalg.norm(r_a)
        
        r_dot = (np.dot(r_a, v_a)) / r_norm
        
        n = r_a / r_norm
        
        ###### Post Newtonian Correction Equations ##########
        A_0 = 1
        A_1 = ((-3/2) * r_dot**2 * nu) + ((1 + (3 * nu)) * v_norm**2) - (2 * (2 + nu) * (G * M / r_norm))
        
        A_2 = ((15/8) * r_dot**4 * (1 - 3*nu)) + (3 * r_dot**2 * nu * v_norm**2 * (2 * nu - (3/2))) 
        + (nu * v_norm**4 * (3 - 4 * nu)) 
        + ((G * M / r_norm) * ((-2 * r_dot**2 * (1 + nu**2)) - (25 * r_dot**2 * nu) - ((13/2) * nu * v_norm**2))) 
        + (((G**2 * M**2)/ r_norm**2) * (9 + (87/4) * nu)) 
                        
        A_3 = (-8/5) * (G * M / r_norm * nu * r_dot) * (17 / 3 * G * M / r_norm) + (3 * v_norm**2)
        
        B_0 = 0
        B_1 = -2 * (2 - nu) * r_dot
        B_2 = ((3 * r_dot**3 * nu) * ((3/2) + nu)) - (r_dot * nu * v_norm**2 * ((15/2) + (2 * nu))) + ((G * M * r_dot / r_norm) * (2 + (41 * nu / 2) + (4 * nu**2)))
        B_3 = (8 * G * M * nu)/(5 * r_norm) * ((3 * G * M / r_norm) + v_norm**2)
         
        L_A = [A_1, A_2, A_3]
        L_B = [B_1, B_2, B_3]
        
        c_value = 150
        c_pow = [-2, -4, -5]
        
        A = A_0 + sum(L_A[i] * c_value**c_pow[i] for i in range(0, p - 1))
        B = B_0 + sum(L_B[i] * c_value**c_pow[i] for i in range(0, p - 1))
        
        if p == 'PN1':
            A = A_0 + (L_A[0] * c_value ** (-2))
            B = B_0 + (L_B[0] * c_value ** (-2))
        elif p == 'PN2':
            A = A_0 + (L_A[1] * c_value ** (-4))
            B = B_0 + (L_B[1] * c_value ** (-4))
        elif p == 'PN2.5':
            A = A_0 + (L_A[2] * c_value ** (-5))
            B = B_0 + (L_B[1] * c_value ** (-5))
        elif p == 'PN1 and PN2':
        
        
            
            
        
        # Calculate acceleration using Post Newtonian Correction
        dvdt = (- G * M / (r_norm**2)) * ((A * n) + (B * v_a))
        
        return dvdt
    
    
    
def calculate_eccentricity(r_a, v_a, r_magnitude, M):

    # calculate angular momentum perpendicular to the orbit 
    # of the two body problem
    h = np.cross(r_a, v_a)
    eccentricity = (np.cross(v_a, h) / G * M) - (r_a / r_magnitude)
    
    return eccentricity



def calculate_a_semi_major(r_a, v_a, M):
    
    # at every point in the orbit
    # the semi-major axis is given by
    return 1 / ((2 / np.linalg.norm(r_a)) - (np.linalg.norm(v_a)**2 / (G * M)))



def variable_leapfrog_algorithm(r_a, v_a, orbital_period, M, M1, M2, eta, p, color):
    
    t = 0
    
    r_magnitude = np.linalg.norm(r_a)
    
    a = calculate_a(r_a, v_a, r_magnitude, M, p)
            
    eccentricity = calculate_eccentricity(r_a, v_a, r_magnitude, M)
    
    a_semi_major = calculate_a_semi_major(r_a, v_a, M)
            
    initial_eccentricity = eccentricity
        
    with open('binary.circ.orb', 'w') as file:
        
        file.write('Eccentricity' + '\t' + 'Semi-major axis' + '\t' + 'Theoretical semi-major axis' + '\n')  
        file.write('INITIAL: ' + str(eccentricity) + '\t' + str(a_semi_major) + '\n')
        
        while t <= orbital_period:
            
            dt = eta * math.sqrt(r_magnitude**3 / (G * M))
            
            dt_half = dt / 2
                        
            t += dt    
            
            # v'_i+1
            v_temp = v_a + a * dt_half
       
      
            # r_i+1
            r_a = r_a + v_temp * dt
           
            
            r_magnitude = np.linalg.norm(r_a)
            
            
            # a_i+1
            a = calculate_a(r_a, v_a, r_magnitude, M, p)
           
            
            # v_i+1
            v_a = v_temp + a * dt_half
    
            eccentricity = calculate_eccentricity(r_a, v_a, r_magnitude, M)
            
            # Calculate the semi-major axis at each point on orbit
            # and save to file
            a_semi_major = calculate_a_semi_major(r_a, v_a, M)            
            
            # Peter's equation for semi-major axis to calculate
            # theoretical evolution of semi-major axis
            beta = G**3 * M1 * M2 * M / c**5
            a_p = (np.linalg.norm(a_semi_major)**4 - (256 * beta * t / 5))**(1 / 4)
            
            file.write(str(eccentricity) + '\t' + str(a_semi_major) + '\t' + str(a_p) + '\n')
            
            plt.figure(1)
            plt.plot(r_a[0], r_a[1], color, markersize=1)
            
            # plot numerical evolution of semi-major axis w.r.t. time
            plt.figure(2)
            plt.plot(t, np.linalg.norm(a_semi_major), 'ro', markersize=3)
            plt.plot(t, a_p, 'bo', markersize=3)
            
    
    file.close()    
    
    plt.figure(1)
    plt.xlabel("Position X")
    plt.ylabel("Position Y")
        
    plt.figure(2)
    plt.xlabel("T")
    plt.ylabel("Semi-major axis")
    plt.legend(['Semi-major axis ', 'Theoretical semi-major axis'])
    
    # Calculate the precession numerically
    final_eccentricity = eccentricity
    
    phi_0 = math.atan(initial_eccentricity[1]/initial_eccentricity[0])
    phi_1 = math.atan(final_eccentricity[1]/final_eccentricity[0])

    dphi_numerical = phi_1 - phi_0
    
    
    print("--------------------------------------------------------------")  
    print("--------------------------------------------------------------")  
    print("--------------------------------------------------------------")  
        
    print("INIT EC " + str(initial_eccentricity))
    print("FINAL EC " + str(final_eccentricity))
    print(f'Numerical advance in pericentre of the orbit is {dphi_numerical}')
    
    # Calculate time taken by relative orbit to precess by 2*pi radians
    t_p = (dphi_numerical * 4.84814E-6)/(360 * 1.57117E-28)
    print(f'Time taken by relative orbit to precess by 360 degrees is {t_p}')
    
    
def plot_variable_leapfrog_algorithm(r_a, v_a, M, M1, M2, a, eta, p, color):

    orbital_period = calculate_orbital_period(r_a, v_a, M, a)
    
    variable_leapfrog_algorithm(r_a, v_a, orbital_period, M, M1, M2, eta, p, color)
    
    
    
    
# Execute the variable leapfrog algorithm
r_a_blackhole, v_a_blackhole = create_initial_values_to_file(a_blackhole, e_blackhole, M_blackhole_binary, 'binary.circ.init')

# Plot Orbit of eccentric black hole binary without PN correction
# Plot Orbit of eccentric black hole binary with PN correction
# Display in same plot

eta = 0.1 #10**(-6)

#plot_variable_leapfrog_algorithm(r_a_blackhole, v_a_blackhole, M_blackhole_binary, blackhole_M1, blackhole_M2, a_blackhole, eta, 0, 'bo')
#plot_variable_leapfrog_algorithm(r_a_blackhole, v_a_blackhole, M_blackhole_binary, blackhole_M1, blackhole_M2, a_blackhole, eta, 1, 'go')
#plot_variable_leapfrog_algorithm(r_a_blackhole, v_a_blackhole, M_blackhole_binary, blackhole_M1, blackhole_M2, a_blackhole, eta, 2, 'yo')
#plot_variable_leapfrog_algorithm(r_a_blackhole, v_a_blackhole, M_blackhole_binary, blackhole_M1, blackhole_M2, a_blackhole, eta, 3, 'mo')
#plot_variable_leapfrog_algorithm(r_a_blackhole, v_a_blackhole, M_blackhole_binary, blackhole_M1, blackhole_M2, a_blackhole, eta, 4, 'ko')

# Plot the orbit for a time period of 400 years
orbital_period = 400
variable_leapfrog_algorithm(r_a_blackhole, v_a_blackhole, orbital_period, M_blackhole_binary, blackhole_M1, blackhole_M2, eta, 4, 'ko')