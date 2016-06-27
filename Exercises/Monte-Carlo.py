# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def MCintegrate(N):
    """Monte-Carlo Integration using 'N' samples for the given function"""
    
    fbar = 0
    V = (np.pi/8) ** 8
    
    #generates points to estimate the mean
    for i in range(N):
        #uniform since the mean must be unbias across the domain of f
        x = np.random.uniform(0, np.pi/8, 8)
        f = np.sin(np.sum(x))
        fbar += f
    
    #normalises the sum
    fbar /= N
    
    #uses eqn 1 to estimate the integral
    integral = fbar * V
    
    return 1000000 * integral


def BestEst(nt, bigN):
    """Returns mean value and error from 'nt' repeats of 'bigN' points"""
    
    estimates = np.array([])
    for i in range(nt):
        est = MCintegrate(bigN)
        estimates = np.append(estimates, est)
    
    meanI = np.mean(estimates)
    errI = np.std(estimates)
    
    return np.array([meanI, errI])


def ErrorPlot(mag):
    """
    Plots the error in the integral as a function of N over 'mag' orders of 
    magnitude, starting from 10
    """
    
    nVar = []
    #logarithimic spacing reduces unneeded computations
    for i in range(int(mag)):
        #added half values for better plot and regression
        nVar.append(int(0.5 * 10 ** (i + 1)))
        nVar.append(int(10 ** (i+1)))
        
    
    #calculates the integrals and their errors
    errVar = np.array([])
    for i in nVar:
        valueAndError = BestEst(25, i)
        errVar = np.append(errVar, valueAndError[1])
    
    #uses a SciPy regression analysis to obtain the gradient
    regData = stats.linregress(np.log(nVar), np.log(errVar))
    
    #displays the value for the maximal order of magnitude, and the gradient
    print("The Value for N = 10 ^", int(mag), " is:")
    print(valueAndError[0], "pm", valueAndError[1])
    print("The gradient, in logspace, is:")
    print(regData[0], "pm", regData[4])    
    
    #plots the graph on a logarthimic scale
    plt.loglog(nVar, errVar)
    plt.xlabel("Number of samples, N", fontsize = "18")
    plt.ylabel("Error in the Integral", fontsize = "18")
    plt.title("The Error in a Monte-Carlo Integration as a Function of the Number of Samples", fontsize = "22")
    
#Runs program for the required oders of magnitude
ErrorPlot(7)
