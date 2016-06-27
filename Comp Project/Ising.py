# -*- coding: utf-8 -*-
#
#Part II physics computational project 2015-2016
#"The Ising Model of a Ferromagnet"
#
#Python: 3.4.3
#NumPy: 1.9.2
#SciPy: 0.15.1
#
#Only the class is defined, what you do with it is down to you

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import datetime


#Important/frequently used constants are defined globally
KB = constants.value("Boltzmann constant")
ONSANGER_T = 2/np.log(1 + np.sqrt(2))

###############################################################################
#SECTION 0
#The model is implemented in the class definition, and a few functions
#are defined for various uses


class SpinLattice(object):
    """2D Ising model of a simple magnetic system.
    
    Uses a model lattice of simple (up or down) spins to model the magnetic 
    using a behaviour nearest neighbour interaction model. Periodic boundary 
    conditions have been implemented.
    """
    
    def __init__(self, height, width, temperature, applied_field,
                 magnetic_moment, NN_interaction_energy, initial_spins = 0,
                 history = 30):
        """Constructs the lattice variables.
        
        Notes:
            height and width must both be even, an error is raised otherwise.
            All quantities are defined in SI.
        
        args:
            height (int): number of rows in the lattice.
            
            width (int): number of columns in the lattice.
            
            temperature (float): the usual physical quantity. Must be > 0.
            
            applied_field (float): the (H) magnetic field.
            
            magnetic_moment (float): the magnetic moment of an individual
                                     spin site.
                                     
            NN_interaction_energy (float): Interaction energy of nearest 
                                           neighbour spins.
                                           
            initial_spins: The initial lattice will be random if anything other 
                           than -1 or 1. In that case all the spins are all
                           -1 or 1 respectively.
                           
            history (int): number of most recent values for stats, e.g. energy,
                           used in calculations, e.g. of heat capacity.
        """
        
        
        if (height%2 == 0) and (width%2 == 0):    
            self.shape = (height, width)
        elif (height%2 == 1) or (width%2 == 1):
            raise ValueError("'height' and 'width' must be even")
        else:
            raise TypeError("'height' and 'width' must be (even) integers")

        
        #these constants have their usual meanings
        self.N = self.shape[0]*self.shape[1]
        self.T = temperature
        self.beta = 1/(temperature*KB)
        self.H = applied_field
        self.mu = magnetic_moment
        self.J = NN_interaction_energy
        
        #creates an initial lattice
        if initial_spins == 1:
            self.lattice = np.ones(self.shape, dtype = np.int8)
        elif initial_spins == -1:
            self.lattice = -1*np.ones(self.shape, dtype = np.int8)
        else:
            lattice = 2*np.random.randint(2, size = self.shape) - 1
            self.lattice = lattice.astype(np.int8)
        
        #these variable have their usual meanings
        self.moment = 0
        self.interaction_energies = 0
        self.lattice_energy = 0
        self.heat_capacity = 0
        
        #creates arrays to hold the ".s" most recent values
        self.s = history        
        self.energy_history = np.array([])
        self.moment_history = np.array([])
        self.heat_capacity_history = np.array([])
        
        #calls the update method to calculate some of the quantities
        self.update_all()
        
        #"time_step" is incremented when height*width points lattice points
        #have been tested
        self.time_step = 0


    def time_increment(self):
        self.time_step += 1
        return None


    def update_summed_moment(self):
        """Calculates and updates the total moment"""
        self.moment = self.mu * np.sum(self.lattice)
        return None


    def update_interaction_energy(self):
        """Calculates and updates the array of interaction energies
        
        Permuting the lattice to find the nearest neighbours is quick, and
        implicitly incorporates the periodic boundary conditions.
        """
        
        lefts = np.roll(self.lattice, 1, 1)
        rights = np.roll(self.lattice, -1, 1)
        tops = np.roll(self.lattice, 1, 0)
        bottoms = np.roll(self.lattice, -1, 0)
        I = lefts + rights + tops + bottoms
        I *= self.lattice
        
        self.interaction_energies = -self.J * I
        return None


    def update_energy(self):
        """Calculates and updates the energy of the lattice"""
        
        hamiltonian = np.sum(self.interaction_energies)
        hamiltonian -= self.H * self.moment
        
        self.lattice_energy = hamiltonian
        return None


    def update_heat_capacity(self):
        """Calculates the heat capacity 
        
        Uses the fluctuation-dissipation theorem on ".energy_history", only
        when the energy_history is full. The lattice must be in equilibrium for
        all of ".energy_history" for this to be accurate.
        """
        
        if len(self.energy_history) == self.s:
            sigma_E = np.std(self.energy_history)
            c = KB*self.beta*self.beta
            c *= sigma_E*sigma_E
            self.heat_capacity = c
        else:
            pass
        return None
        
    
    def update_E_history(self):
        """Takes the current energy and adds it to the history"""
                
        if len(self.energy_history) < self.s:
            self.energy_history = np.append(self.energy_history,
                                            self.lattice_energy)
        else:
            self.energy_history = np.append(self.energy_history,
                                            self.lattice_energy)
            self.energy_history = np.delete(self.energy_history, 0)
        return None
    
    
    def update_m_history(self):
        """Takes the current moment and adds it to the history"""
        
        if len(self.moment_history) < self.s:
            self.moment_history = np.append(self.moment_history,
                                            self.moment)
        else:
            self.moment_history = np.append(self.moment_history,
                                            self.moment)
            self.moment_history = np.delete(self.moment_history, 0)
        return None
        
        
    def update_c_history(self):
        """Takes the current heat capacity and adds it to the history"""
                
        if len(self.heat_capacity_history) < self.s:
            self.heat_capacity_history = np.append(self.heat_capacity_history, 
                                                   self.heat_capacity)
        else:
            self.heat_capacity_history = np.append(self.heat_capacity_history, 
                                                   self.heat_capacity)
            self.heat_capacity_history = np.delete(self.heat_capacity_history, 
                                                   0)
        return None
        
        
    def update_all(self):
        """Calls the "update" methods in such an order as to ensure that 
        each are calculated correctly.
        """
        self.update_summed_moment()
        self.update_m_history()
        
        self.update_interaction_energy()
        self.update_energy()
        self.update_E_history()
        
        self.update_heat_capacity()
        self.update_c_history()
        return None
        

    def spin_flip(self, row, column):
        """Tests a given spin for flipping, and flips accordingly
        
        Implementation of the Metropolis–Hastings algorithm (for a point).
        
        The ".interaction_energies" attribute must be correct/updated before 
        this method is called.
        """
        
        delta_E = -2*self.interaction_energies[row, column]
        delta_E += 2 * self.mu * self.H * self.lattice[row, column]
        
        if delta_E <= 0:
            #if flipping releases energy, then it occurs
            self.lattice[row][column] *= -1
        else:
            #if flipping requires energy, then we simulate a thermal
            #fluctuation (which provides the energy) using a Monte-Carlo method
            p = np.random.rand()
            boltzman_p = np.exp(-delta_E*self.beta)

            if p <= boltzman_p:
                self.lattice[row][column] *= -1
            else:
                pass
        
        return None


    def lattice_flip_checkerboard(self):
        """Tests and flips two sets of alternate lattice positions.
        
        This is a simple example of Checkerboard Decomposition; by splitting
        the lattice into 'black' and 'white' squares, we can loop over all of
        one 'colour' without having to update the interaction energies. 
        
        The 'colour' of a particular spin is found using modular arithmetic.
        
        NB the dimensions of the lattice must be even for this to work.   
        """
        
        #ensures that the interaction energies are accurate, in most cases
        #this call may be redundant
        self.update_interaction_energy()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if i%2 == j%2:
                    self.spin_flip(i, j)
                else:
                    pass
        
        #The interaction energies must be updated now, as the nearest neighbour
        #spin states may have changed
        self.update_interaction_energy()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if i%2 != j%2:
                    self.spin_flip(i, j)
                else:
                    pass
        
        #updates the rest of the system variables
        self.update_all()         
        self.time_increment()

        return None
    

    def advance_time(self, steps, data = False, timing = False):
        """Calls the ".lattice_flip_checkerboard()" method "steps" times.
        
        If "data" is True, then the energy and mean magnetic moment are 
        calculated and returned as functions of time.
        If "timing" is true, then the elapsed time will be printed every 10% 
        (ish) of the steps.
        """
        
        if timing:
            start = datetime.datetime.now()
        
        if data:
            self.update_all()
            t = np.array([self.time_step])
            energy_t = np.array([self.lattice_energy])
            
            mean_moment = (self.moment/(self.N))
            m_t = np.array([mean_moment])

            for i in range(steps):
                self.lattice_flip_checkerboard()
                
                t = np.append(t, self.time_step)
                energy_t = np.append(energy_t, self.lattice_energy)
        
                mean_moment = self.moment/(self.N)
                m_t = np.append(m_t, mean_moment)
                
                if timing and ((i + 1)%(steps//10) == 0):
                    print(i+1)
                    now = datetime.datetime.now()
                    print("elapsed time - ", now - start)

            return np.array([t, energy_t, m_t])
            
        else:
            for i in range(steps):
                self.lattice_flip_checkerboard()
                
                if timing and ((i + 1)%(steps//10) == 0):
                    print(i+1)
                    now = datetime.datetime.now()
                    print("elapsed time - ", now - start)
                    
            return None


def binary_plot(title, lattice):
    """Plots a given lattice as a binary image"""
    plt.figure()
    plt.imshow(lattice, cmap='Greys',  interpolation='none')
    plt.savefig("%s.pdf" % title, bbox_inches="tight")
    plt.close()
    return None


def estimate_Tc(temperatures, heat_caps):
    """Estimates the critical temperature by considering heat capacity
    
    For the 2D models exact solution, it can be shown that a singularity occurs
    in the heat capacity at the critical temperature. By entering (ordered)
    arrays of "heat_caps" as a function of "temperatures", Tc can be estimated
    from the maximal heat capacity.
    """
    
    Tcrit = temperatures[np.argmax(heat_caps)]
    return Tcrit


def estimate_Tc2(temperatures, magnetisation, threshold = 0.5):
    """Estimates the critical temperature
    
    When the (dimensionless mean) magnetisation falls bellow threshold,
    the corresponding temperature is an estimate for the critical 
    temperature. "magnetisation" and "temperatures" must be mutually ordered 
    arrays. This function is often wildly inaccurate.
    """
            
    for i in range(len(magnetisation)):
        if magnetisation[i] <= threshold:
            return temperatures[i]
        else:
            pass
    
    print("No critical temperature was found; try changing the threshold")
    return None
        

def mag_func(temperatures, J = KB):
    """Defines the magnetisation function (per site) for use before Tc"""
    
    beta = np.power((temperatures*KB), -1)
    m = np.sinh(2*beta*J)
    m = np.power(m, -4)
    m = 1 - m
    m = np.power(m, (1/8))
    return m

def exact_magnetisation(temperatures, J = KB):
    """Returns the magnetisation (per site) as a function of temperature
    
    Defined for SI, however the interaction energy "J" defaults to KB. So
    Kelvin is equivalent to the T*J/KB dimensionless temperature.
    """
    
    m_of_T = np.piecewise(temperatures, [temperatures < ONSANGER_T, 
                                         temperatures >= ONSANGER_T], 
                                         [mag_func, 0], (J,))
    return m_of_T
