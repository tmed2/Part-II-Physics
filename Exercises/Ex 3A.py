# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#Defines the distances in SI
#Wavelength, lambda in manual
waveL = 500.0 * (10 ** -9)
#Slit width, d
slitW = 100.0 * (10 ** -6)
#Distance to screen, D
screenD = 1.0
#Total apperture extent, L
appL = 5.0 * (10 ** -3)



def SlitAperture(Npoints, d = slitW, L = appL):
    
    
    x = np.linspace(-L/2, L/2, Npoints)
    #creates the samples of the aperture function
    app = np.piecewise(x, [abs(x) > d/2, abs(x) <=d/2], [0,1])
    
    #defines the sample spacing of the aperture
    sampSpacing = x[1] -x[0]
    
    return x, app, sampSpacing, Npoints


def ExactSlitPattern(Npoints, L = appL, d = slitW, lamda = waveL, D = screenD):
    
        #Had to experiment tyo find resonable y range for the exact case
        y = np.linspace(-10*L, 10*L, Npoints)
        
        #NB the pi is impicitly included in the definition of np.sinc
        u = (2/(lamda*D))*y
        modPsi = np.sinc(0.5*d*u)
        intensity = modPsi ** 2
        
        return y, intensity


def SinePhase(x, m = 8, s = (100 * (10 ** -6))):
    """Defines the complex phase factor"""
    
    phase = (m/2)*np.sin(((2*np.pi)/s) * x)
    phaseFactor = complex(0, phase)
    return np.exp(phaseFactor)


def SineAperture(Npoints, d = (2 * (10 ** -2)), L = appL):
    
    x = np.linspace(-L/2, L/2, Npoints)
    #defines the sample spacing of the aperture
    sampSpacing = x[1] -x[0]
    
    app = np.array([])
    for i in x:
        if abs(i) > d:
            app = np.append(app, 0)
        else:
            app = np.append(app, SinePhase(i))
    
    return x, app, sampSpacing, Npoints    


def Fraunhofer(app, spacing, N, lamda = waveL, D = screenD):
    """
    Finds the Fraunhofer diffraction pattern of a 1-D apperture, which
    is sampled using the array 'app', with 'spacing' between samples.
    """
    u = np.fft.fftfreq(N, d=spacing)
    u = np.fft.fftshift(u)
    
    y = lamda*D*u
    psi = np.fft.fft(app)
    psi = np.fft.fftshift(psi)
    
    intensity = ((np.abs(psi)) ** 2)
    intensity /= max(intensity)
    return np.array([u, y, intensity])


#Computes and Plots the data for "Core Task 1"
A = SlitAperture(1000)
pattA = Fraunhofer(A[1], A[2], A[3])

plt.figure(1)
plt.plot(1000*pattA[1], pattA[2], label = "FFT")
plt.plot(1000*ExactSlitPattern(300)[0], ExactSlitPattern(300)[1], marker = "x", linestyle = "", label = "Exact Solution")
plt.ylabel("Relative Intensity", fontsize = "18")
plt.xlabel("Distance from Axis on Screen / mm", fontsize = "18")
plt.title("Fraunhofer Diffraction Through a Single Slit", fontsize = "24")
plt.legend(loc = "best")


#Computes and Plots the data for "Core Task 2"
B = SineAperture(1000)
pattB = Fraunhofer(B[1], B[2], B[3], D= 10)

plt.figure(2)
plt.plot(1000*pattB[1], pattB[2])
plt.ylabel("Relative Intensity", fontsize = "18")
plt.xlabel("Distance from Axis on Screen / mm", fontsize = "18")
plt.title("Fraunhofer Diffraction Through a Sinusoidal Phase Modulator", fontsize = "24")


#"Supplemental Task 1" thigs here

def SlitAppertureFresnel(Npoints, D = (5 * (10 ** -3)), lamda = waveL, d = slitW, L = appL):
    x = np.linspace(-L/2, L/2, Npoints)
    #creates the samples of the aperture function
    app = np.piecewise(x, [abs(x) > d/2, abs(x) <=d/2], [0,1])
    
    arg = complex(0,1)/(waveL*D)
    fresnelMultiplier = np.exp( arg * (x ** 2))
    
    app = app*fresnelMultiplier
    #defines the sample spacing of the aperture
    sampSpacing = x[1] - x[0]
    
    return x, app, sampSpacing, Npoints

#with the appropriate modification, the Fraunhofer result can be used for the Fresnel Cases    
C = SlitAppertureFresnel(1000)
pattC = Fraunhofer(C[1], C[2], C[3], D = (5 * (10 ** -3)))
plt.figure(3)
plt.plot(1000*pattC[1], pattC[2])
plt.ylabel("Relative Intensity", fontsize = "18")
plt.xlabel("Distance from Axis on Screen / mm", fontsize = "18")
plt.title("Fresnel Diffraction Through a Single Slit", fontsize = "24")

def SineApertureFresnel(Npoints, d = (2 * (10 ** -2)), L = appL, D = 0.5):
    
    x = np.linspace(-L/2, L/2, Npoints)
    #defines the sample spacing of the aperture
    sampSpacing = x[1] -x[0]
    
    app = np.array([])
    for i in x:
        if abs(i) > d:
            app = np.append(app, 0)
        else:
            arg = complex(0,1)/(waveL*D)
            fresnelMultiplier = np.exp( arg * (i ** 2))
            app = np.append(app, fresnelMultiplier*SinePhase(i))
    
    return x, app, sampSpacing, Npoints
    
E = SineApertureFresnel(1000)
pattE = Fraunhofer(E[1], E[2], E[3], D = 0.5)
plt.figure(4)
plt.plot(1000*pattE[1], pattE[2])
plt.ylabel("Relative Intensity", fontsize = "18")
plt.xlabel("Distance from Axis on Screen / mm", fontsize = "18")
plt.title("Fresnel Diffraction Through a Sinusoidal Phase Modulator", fontsize = "24")
