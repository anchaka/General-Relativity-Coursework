#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 1 01:43:30 2022

@author: 6689530
"""

import math 
import numpy as np
from matplotlib import pyplot as plt

############################# Initial values #################################

a_blackhole = 0.15       # semi-major axis for black hole binary in milliparsec

e_blackhole = 0.85       # eccentricity of black hole binary

G = 1                    # Universal gravitational constant

# Claculate the unit of mass using milliparsec per year and G = 1
Mass_unit = (3.086 * (10**13))**3/((6.67*10**(-11))* ((24*3600*365)**2))

# convert mass of blackholes to specified units 
blackhole_M1 = (7 * 1.98855 * (10**36))/Mass_unit
blackhole_M2 = (7 * 1.98855 * (10**33))/Mass_unit

# Calculate unit of velocity in milliparsec per year
velocity_unit = (24*3600*365)/(3.08568*(10**13))
                               
# Calculate speed of light in milliparsec per year
c = (2.997)*(10**8) * (velocity_unit)

# mass of two binary black holes
M_blackhole_binary = blackhole_M1 + blackhole_M2 

##############################################################################

# Calculate the theoretical advance of pericentre per orbit due to General Relativity
dphi_GR = (6 * math.pi * blackhole_M1 * G)/(a_blackhole * (c**2) * (1 - e_blackhole**2))

print("Constants in milliparsec and years with G = 1".center(24, "-"))
print()
print(f'The unit of mass using milliparsec per year is {Mass_unit}')
print()
print(f'The unit of velocity in milliparsec per year is {velocity_unit}')
print()
print(f'The speed of light in milliparsec per year is {c}')
print()
print(f'The theoretical advance of pericentre per orbit due to General Relativity is {dphi_GR}')

def create_initial_values_to_file(a, e, M, file_name):
    ''' Function that calculates initial position and velocity from the values of 
    semi-major axis and eccentricity of the two black holes. The motion of the two black holes
    is modeled as the Keplerian two body problem in the centre of mass reference frame.
    The initial values are saved to a file with user specified file name
    
    Innput parameters:
        a is the semi-major axis in the two body problem
        e is the eccentricity of the orbit
        M is the total mass of the two black hole'''
        
    #r_p = a * (1 - e)   #pericentre for system
    r_a_x = a * (1 + e)  #apocentre
    #v_p = math.sqrt((G * M/a)*((1 + e)/(1 - e)))   #pericentre velocity
    v_a_y = math.sqrt((G * M/a)*((1 - e)/(1 + e)))  #apocentre velocity

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
    
        
    orbital_period = math.sqrt((4 * math.pi**2 * np.linalg.norm(a)**3) / (G * M))
    
    return orbital_period
                


def calculate_a(r_a, v_a, r_norm, M, p1, p2, p2_5):
    '''This function calculates the acceleration vector at each position in the orbit 
    for the two body problem. According to the user input p(see description in input parameters)
    The Post Newtonian correction is implemented
    
    Input parameters:
        r_a is the initial position at the apocentre
        v_a is the initial velocity at the apocentre
        M is the total mass of the point masses
        p is the order of PN correction
        P can be 'no PN','PN1', 'PN2', 'PN2.5', OR 'PN1 + PN2', 'PN1 + PN2.5',
        'PN2 + PN2.5'

    '''
    
    if not p1 and not p2 and not p2_5:
       return (- (G * M) * r_a) / (r_norm**3)


    v_norm = np.linalg.norm(v_a)
                       
    r_dot = (np.dot(r_a, v_a)) / r_norm
    
    n = r_a / r_norm
    
    # substitution for mass used in Post Newtonian correction
    nu = (blackhole_M1 * blackhole_M2)/(M_blackhole_binary**2)
    
    ################################## Post Newtonian Correction Equations #############################################################################################
    #0
    A = 1
    B = 0
    
    c_value = 150
    
    # Select only the requested PN correction terms based on p (user input)
    if p1:
        A_1 = ((-3/2) * r_dot**2 * nu) + ((1 + (3 * nu)) * v_norm**2) - (2 * (2 + nu) * (G * M / r_norm))
        B_1 = -2 * (2 - nu) * r_dot
        
        c_exp = c_value**-2
        
        A += A_1 * c_exp
        B += B_1 * c_exp
    
    
    if p2:
        A_2 = ((15/8) * r_dot**4 * (1 - 3*nu)) + (3 * r_dot**2 * nu * v_norm**2 * (2 * nu - (3/2))) 
        + (nu * v_norm**4 * (3 - 4 * nu)) 
        + ((G * M / r_norm) * ((-2 * r_dot**2 * (1 + nu**2)) - (25 * r_dot**2 * nu) - ((13/2) * nu * v_norm**2))) 
        + (((G**2 * M**2)/ r_norm**2) * (9 + (87/4) * nu)) 
        
        B_2 = ((3 * r_dot**3 * nu) * ((3/2) + nu)) - (r_dot * nu * v_norm**2 * ((15/2) + (2 * nu))) + ((G * M * r_dot / r_norm) * (2 + (41 * nu / 2) + (4 * nu**2)))
       
        c_exp = c_value**-4
        
        A += A_2 * c_exp
        B += B_2 * c_exp 
       
    if p2_5:          
        A_2_5 = (-8/5) * (G * M * nu * r_dot/ r_norm ) * ((17 / 3) * G * M / r_norm) + (3 * v_norm**2)
        B_2_5 = (8 * G * M * nu)/(5 * r_norm) * ((3 * G * M / r_norm) + v_norm**2)
        
        c_exp = c_value**-5
        
        A += A_2_5 * c_exp
        B += B_2_5 * c_exp 
            
 
    # Calculate acceleration using Post Newtonian Correction       
    dvdt = (- G * M / (r_norm**2)) * ((A * n) + (B * v_a))
    
    return dvdt


        
def calculate_eccentricity(r_a, v_a, r_norm, M):

    # calculate angular momentum perpendicular to the orbit 
    # of the two body problem
    h = np.cross(r_a, v_a)
    eccentricity = (np.cross(v_a, h) / G * M) - (r_a / r_norm)
    
    return eccentricity



def calculate_a_semi_major(r_a, v_a, M):
    
    # at every point in the orbit
    # the semi-major axis is given by
    return 1 / ((2 / np.linalg.norm(r_a)) - (np.linalg.norm(v_a)**2 / (G * M)))


def get_p_label(p1, p2, p2_5):
    
    if not p1 and not p2 and not p2_5:
        return "No PN"
        
    p_label = ""
    
    if p1:
        p_label = "PN1"
        
    if p2:
        p_label = append_p_label(p_label, "PN2")
        
    if p2_5:
        p_label = append_p_label(p_label, "PN2.5")
        
    return p_label



def append_p_label(partial_label, addition):
    
    if partial_label != "":
        partial_label += " + "
         
    return partial_label + addition
    


def plot_variable_leapfrog_algorithm(r_a, v_a, time_period, M, M1, M2, eta, p1, p2, p2_5, shouldPlot360DegreePrecession, 
                                     plotPosition, plotEccentricity, plotSemiMajor, color):
    '''
    

    Parameters
    ----------
    r_a : position of COM at apogee.
    v_a : velocity of COM at apogee.
    time_period : how long should the alogorithm run.
    M : combined mass of the binary system.
    M1 : mass of first black hole.
    M2 : mass of second black hole.
    eta : small numerical factor related to the variable time step as follows
    dt = eta * sqrt(r_norm**3/ (G * M)), also called the accuracy parameter.
    color : distinct colours for plotting the orbital position using respective PN correction on acceleration.

    Returns
    -------
    A file documenting the variation of eccentricity and semi-major axis w.r.t. time
    Also outputs the following plots:
        orbital position in the x-y plane
        eccentricity w.r.t. time
        semi-major axis w.r.t. time

    '''
    
    t = 0
    
    r_norm = np.linalg.norm(r_a)
    
    a = calculate_a(r_a, v_a, r_norm, M, p1, p2, p2_5)
            
    eccentricity = calculate_eccentricity(r_a, v_a, r_norm, M)
    
    a_semi_major = calculate_a_semi_major(r_a, v_a, M)
            
    initial_eccentricity = eccentricity
        
    with open('binary.ecc.orb', 'w') as file:
        
        file.write('Eccentricity' + '\t' + 'Semi-major axis' + '\t' +  '\n')  
        file.write('INITIAL: ' + str(eccentricity) + '\t' + str(a_semi_major) + '\n')
        
        while t <= time_period:
            
            dt = eta * math.sqrt(r_norm**3 / (G * M))
            
            dt_half = dt / 2
                        
            t += dt    
            
            # v'_i+1
            v_temp = v_a + a * dt_half
       
            # r_i+1
            r_a = r_a + v_temp * dt
           
            r_norm = np.linalg.norm(r_a)
            
            # a_i+1
            a = calculate_a(r_a, v_a, r_norm, M, p1, p2, p2_5)
                       
            # v_i+1
            v_a = v_temp + a * dt_half
    
                        
            if plotPosition:
                plt.figure(1)
                plt.plot(r_a[0], r_a[1], color, markersize=0.3)    
            
            if plotEccentricity:  
                eccentricity = calculate_eccentricity(r_a, v_a, r_norm, M)
                
                plt.figure(2)
                plt.plot(t, np.linalg.norm(eccentricity), 'go', markersize=0.3)  
            
            if plotSemiMajor:
                # Calculate the semi-major axis at each point on orbit
                a_semi_major = calculate_a_semi_major(r_a, v_a, M)     
            
                plt.figure(3)
                plt.plot(t, np.linalg.norm(a_semi_major), 'ro', markersize=0.3)
     
    file.close()
    
    p_label = get_p_label(p1, p2, p2_5)
    
    # Plot the orbital position
    if plotPosition:
        plt.figure(1)
        plt.title(f'Orbital Position of COM of the Binary Blackhole System with {p_label} correction for a time period {time_period}')
        plt.xlabel("COM X coordinate (milliparsec)")
        plt.ylabel("COM Y coordinate (milliparsec)")
    
    # plot numerical evolution of eccentricity  w.r.t. time
    if plotEccentricity:
        plt.figure(2)
        plt.title(f'Eccentricity Evolution of COM of the Binary blackhole System with {p_label} for a time period {time_period}')
        plt.xlabel("Time in (years)")
        plt.ylabel("eccentricity of binary blackholes")
    
    # plot numerical evolution of semi-major axis w.r.t. time
    if plotSemiMajor:
        plt.figure(3)
        plt.title(f'Semi-Major axis evolution of COM of the Binary Blackhole System with {p_label} correction for a time period {time_period}')
        plt.xlabel("Time (years)")
        plt.ylabel("semi-major axis (milliparsec)")
    
    plt.show()
    

    if not plotEccentricity:  
        eccentricity = calculate_eccentricity(r_a, v_a, r_norm, M)
    
    
    # Calculate the precession numerically
    final_eccentricity = eccentricity
    
    phi_0 = math.atan(initial_eccentricity[1]/initial_eccentricity[0])
    phi_1 = math.atan(final_eccentricity[1]/final_eccentricity[0])

    dphi_numerical = abs(phi_1 - phi_0)
    
    
    print("--------------------------------------------------------------")  
    print("--------------------------------------------------------------")  
    print("--------------------------------------------------------------")  
        
    print("INIT EC " + str(initial_eccentricity))
    print()
    print("FINAL EC " + str(final_eccentricity))
    print()
    print(f'Numerical precession of the orbit is {dphi_numerical}')
    print()
    print('The percentage error between the numerical precession and the theoretical precession is\n')
    print(f'{abs((dphi_numerical - dphi_GR))/dphi_GR * 100}')
    
    # Calculate time taken by relative orbit to precess by 2*pi radians
    t_p = (2 * math.pi) * (calculate_orbital_period(r_a_blackhole, v_a_blackhole, M_blackhole_binary, a_blackhole))/ (dphi_numerical) if dphi_numerical != 0 else 0
    
    print(f'Time taken by relative orbit to precess by 360 degrees is {t_p}')
    
    # plot the 360 degrees precession with no PN and then with, PN1 and PN2
    # using function recursion
    if shouldPlot360DegreePrecession:
         plot_variable_leapfrog_algorithm(r_a, v_a, t_p, M, M1, M2, eta, p1, p2, p2_5, False, plotPosition, plotEccentricity, plotSemiMajor, color)
    
###########################################################################################################################################
    
# Write initial values to file
r_a_blackhole, v_a_blackhole = create_initial_values_to_file(a_blackhole, e_blackhole, M_blackhole_binary, 'binary.ecc.init')

eta = 0.05

time_period = calculate_orbital_period(r_a_blackhole, v_a_blackhole, M_blackhole_binary, a_blackhole)

# Plot Orbit of eccentric black hole binary without PN correction for 1 orbital period
plot_variable_leapfrog_algorithm(r_a_blackhole, v_a_blackhole, time_period, M_blackhole_binary, blackhole_M1, blackhole_M2, eta, False, False, False, True, True, False, False, 'bo')

# Plot Orbit of eccentric black hole binary with PN1 and PN2 correction for 1 orbital period
plot_variable_leapfrog_algorithm(r_a_blackhole, v_a_blackhole, time_period, M_blackhole_binary, blackhole_M1, blackhole_M2, eta, True, True, False, True, True, False, False, 'mo')

# Display in same plot

# Plot the orbit for a time period of 400 years with PN2.5 correction
time_period = 400
plot_variable_leapfrog_algorithm(r_a_blackhole, v_a_blackhole, time_period, M_blackhole_binary, blackhole_M1, blackhole_M2, eta, False, False, True, False, True, True, True, 'ko')