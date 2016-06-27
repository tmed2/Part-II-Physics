# -*- coding: utf-8 -*-
"""
Written by: Tim Duthie; tmed2; Churchill.

For numerical convenience, we have defined: the graviational field strength, 
g = 10 ms^-2; the pendulum length, l = 10m; and the pendulum bob mass, m = 1kg.
A light bar, point mass bob model has been implemented.
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


def LinSolution(t, angle_0, omega_0, a = 1):
    """
    Returns the exact numerical values for the simplest linear case.
    angle_0 and omega_0 are defined at t[0].
    """
    
    if angle_0 == 0:
        A = omega_0/np.sqrt(a)
        phi = np.pi/2 - np.sqrt(a)*t[0]
    else:
        A = angle_0
        phi = -(np.sqrt(a)*t[0])
    
    angle = A*np.cos(np.sqrt(a)*t + phi)
    omega = -A*np.sqrt(a)*np.sin(np.sqrt(a)*t + phi)    
    
    return np.array([t, angle, omega])


def NonLinPendulum(y, t, a, q, F, drivAngFreq):
    """
    A System of ODEs describing a damped, forced, pendulum. a = g/l.
    Returns an array of the system of 1st order ODEs
    """
    
    angle = y[0]
    omega = y[1]
    return [omega, -a * np.sin(angle) - q*omega + F*np.sin(drivAngFreq * t)]


def ModuloCorrection(angle):
    """
    Corrects the input angle to the range -pi to pi
    """
    
    
    newAngle = np.array([])
    for i in angle:
        if abs(i) <= np.pi:
            newAngle = np.append(newAngle, i)
            
        else:
            sign = i/np.abs(i)        
            halfRotations = int((np.abs(i)) // np.pi)
            remainder = ((np.abs(i)) % np.pi)
            if (halfRotations % 2) == 0:
                newAngle = np.append(newAngle, sign*remainder)
            else:
                newAngle = np.append(newAngle, sign*(remainder - np.pi))
            
    return newAngle
            
            

def Response(time, angle_0, omega_0, pendParas = (1,0,0,0)):
    """
    Solves the system defined by NonLinPendulum. 'angle_0' and 'omega_0' are 
    defined at t[0]. The 'pendParas' tuple = (a, q, F, drivAngFreq). Returns
    an array of arrays of t, angle, and omega respectively
    """
    
    initCondition = [angle_0, omega_0]
    solution = integrate.odeint(NonLinPendulum, y0 = initCondition, t = time, args = pendParas)
    
    return np.array([time, solution[:,0], solution[:,1]])


def EnergyTracker(response, m = 1, g = 10, l = 10):
    """
    Returns an array of the energies for the pendulum, based on a response 
    returned by the above function
    """
    
    time = response[0]
    #Zero point defined at angle = 0
    gravPotEnergy = m*g*l*(1 - np.cos(response[1]))
    kineticEnergy = 0.5*m*l*l*response[2]*response[2]
    totEnergy = gravPotEnergy + kineticEnergy
    
    return np.array([time, gravPotEnergy, kineticEnergy, totEnergy])


def EstimatePeriod(response):
    """Estimates the time period from the roots of the angle"""
    #is a bit shoddy, requires long time periods to produce consistent results
    
    
    roots = np.array([])
    for i in range(len(response[1])):
        try:
            if response[1][i] == 0:
                roots = np.append(roots, response[0][i])
                
            #tests for sign change
            elif response[1][i] * response[1][i+1] < 0:
                roots = np.append(roots, response[0][i])
                
            else:
                pass
            
        except IndexError:
            pass
    
    #from root(N) = t_0 + N*T/2, and sum of series in N. NB a divsion by N is
    #implicit in the mean
    roots = 2 * (roots - roots[0])
    period = 2 * np.mean(roots)/(len(roots) + 1)
    
    #could add error calculation in future
    return period


t1 = np.linspace(0, 10000*(2*np.pi), 500000)
lin = LinSolution(t1, 0.01, 0)
nonLin = Response(t1, 0.01, 0)
energy = EnergyTracker(nonLin)

#only valid for small angles, period not generally 2pi
plt.figure(1)
plt.loglog(energy[0]/(2*np.pi), energy[3])
plt.xlabel("Number of Periods", fontsize = "18")
plt.ylabel("Total Energy/ J", fontsize = "18")
plt.title("A Plot of the Pendulum Energy as a \nFunction of the Number of Oscillations", fontsize = "22")

plt.figure(2)
plt.semilogx(t1/(2*np.pi), lin[1] - nonLin[1])
plt.xlabel("Number of Periods", fontsize = "18")
plt.ylabel("Linear - Nonlinear Angle/ rad", fontsize = "18")
plt.title("The Difference Between the Linear and Nonlinear Responses \nas a Function of the Number of Oscillations", fontsize = "22")

t2 = np.linspace(0, 500*(2*np.pi), 50000)
nonLinPiby2 = Response(t2, 0.5*np.pi, 0)

print("The Period of the simplest nonlinear case, started from rest at pi/2, is:")
print(EstimatePeriod(nonLinPiby2))


#Estimates the period as a function of various starting ampltiudes
amplitudes = np.array([])
periods = np.array([])
for i in range(19):
    amp = ((i+1)/20)*np.pi
    amplitudes = np.append(amplitudes, amp)
    response_i = Response(t2, amp, 0)
    periods = np.append(periods, EstimatePeriod(response_i))

plt.figure(3)
plt.plot(amplitudes, periods)
plt.xlabel("Amplitude / rad", fontsize = "18")
plt.ylabel("Period / s", fontsize = "18")
plt.title("The Period of Nonlinear Oscillations as a Function of Amplitude", fontsize = "22")



#calculates and plots the response for unforced, damped cases
t3 = np.linspace(0, 10*(2*np.pi), 5000)
q1 = Response(t3, 0.2, 0, pendParas = (1,1,0,0))
q5 = Response(t3, 0.2, 0, pendParas = (1,5,0,0))
q10 = Response(t3, 0.2, 0, pendParas = (1,10,0,0))

plt.figure(4)
plt.plot(q1[0], q1[1], label = "q = 1")
plt.plot(q5[0], q5[1], label = "q = 5")
plt.plot(q10[0], q10[1], label = "q = 10")
plt.xlabel("Time / s", fontsize = "18")
plt.ylabel("Angle / rad", fontsize = "18")
plt.title("Resonse for Various Levels of Damping", fontsize = "22")
plt.legend(loc = "best")


t4 = np.linspace(0, 10*(2*np.pi), 10000)
#defines the driving frequency
OMEGA = 2/3

f1 = Response(t4, 0.2, 0, pendParas = (1, 0.5, 0.5, OMEGA))
f2 = Response(t4, 0.2, 0, pendParas = (1, 0.5, 1.2, OMEGA))
f3 = Response(t4, 0.2, 0, pendParas = (1, 0.5, 1.44, OMEGA))
f4 = Response(t4, 0.2, 0, pendParas = (1, 0.5, 1.465, OMEGA))

plt.figure(5)
plt.plot(f1[0], ModuloCorrection(f1[1]), label = "F = 0.5")
plt.plot(f2[0], ModuloCorrection(f2[1]), label = "F = 1.2")
plt.xlabel("Time / s", fontsize = "18")
plt.ylabel("Angle / rad", fontsize = "18")
plt.title("Response for Differing Magnitudes of Driving Force", fontsize = "22")
plt.legend(loc = "best")


plt.figure(6)
plt.plot(f3[0], ModuloCorrection(f3[1]), label = "F = 1.44")
plt.plot(f4[0], ModuloCorrection(f4[1]), label = "F = 1.465")
plt.xlabel("Time / s", fontsize = "18")
plt.ylabel("Angle / rad", fontsize = "18")
plt.title("Response for Differing Magnitudes of Driving Force", fontsize = "22")
plt.legend(loc = "best")


t5 = np.linspace(0, 1000*(2*np.pi), 100000)
f5 = Response(t5, 0.2, 0, pendParas = (1, 0.5, 1.2, OMEGA))
f6 = Response(t5, 0.20001, 0, pendParas = (1, 0.5, 1.2, OMEGA))

anglef5 = ModuloCorrection(f5[1])
anglef6 = ModuloCorrection(f6[1])

plt.figure(7)
plt.semilogx(t5, (anglef6 - anglef5))
plt.xlabel("Time / s", fontsize = "18")
plt.ylabel("Angle Difference / rad", fontsize = "18")
plt.title("Difference Between Oscillators with 10^-5 Difference in Initial Angle", fontsize = "22")