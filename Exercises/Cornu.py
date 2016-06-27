# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#defines the real and imaginary integrands respectively
def Cintegrand(x, a = 0.5):
    return np.cos(a * (np.pi)*(x ** 2))


def Sintegrand(x, a = 0.5):
    return np.sin(a * (np.pi)*(x ** 2))


def CornuPlot(Lim, Npoints):
    """
    Plots the Cornu spiral using scipy quad integration using 'Npoints' points 
    up to 'Lim' for the integration parameter
    """
    
    uVect = np.linspace(0, Lim, num = Npoints)
    realPart = np.array([])
    imagPart = np.array([])
    
    for i in uVect:
        re = sp.integrate.quad(Cintegrand, 0, i)[0]
        im = sp.integrate.quad(Sintegrand, 0, i)[0]
        realPart = np.append(realPart, re)
        imagPart = np.append(imagPart, im)
    
    """
    the spiral is antisymmetric (sub -x = z for the negative limits) so we
    don't have to evaluate the other half. The value at the origin is repeated
    """
    uVect = np.insert(uVect, 0, -uVect[::-1])
    realPart = np.insert(realPart, 0, -realPart[::-1])
    imagPart = np.insert(imagPart, 0, -imagPart[::-1])
        
    
    plt.plot(realPart, imagPart)
    plt.xlabel("Real Part", fontsize = "18")
    plt.ylabel("Imaginary Part", fontsize = "18")
    plt.title("A Plot of the Cornu Spiral in the Complex Plane", fontsize = "22")
    
    return None

#runs the program to get the plot. Experimenting with the function showed that
#10 and 1000 seem to produce a decent spiral
CornuPlot(10, 1000)
