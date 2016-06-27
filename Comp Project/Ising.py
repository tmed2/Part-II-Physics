# -*- coding: utf-8 -*-
#
#Part II physics computational project 2015-2016
#"The Ising Model of a Ferromagnet"
#
#Python: 3.4.3
#NumPy: 1.9.2
#SciPy: 0.15.1


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
    

###############################################################################
#The following sections contain specific examples which use the SpinLattice 
#class. In general, we have set J = interaction energy = KB (ie the numerical 
#value), such that the temperature in Kelvin is equivalent to units of J/KB.
#For similar reasons the magnetic moment has been set to 1 so that the 
#calculated moments are normalised.
print("This script will perform simulations of the 2D Ising Model.")
print("The plots generated are saved to disk on creation.")
print("Some important results are printed in the console. \n")
###############################################################################
#SECTION 1
#toy calculations are performed for performance analysis.

print("Performance analysis of a 32 by 32 lattice for 10000 steps is starting")
test = SpinLattice(32, 32, 1, 0, 1, KB)
test.advance_time(10000, False, True)
print("End \n")

print("Performance analysis of a 50 by 50 lattice for 10000 steps is starting")
test = SpinLattice(50, 50, 1, 0, 1, KB)
test.advance_time(10000, False, True)
print("End \n")


###############################################################################
#SECTION 2
#In this section, the energy, mean magnetisation, and heat capacity are
#investigated as a function of temperature. The initial lattices are 
#essentially pre magnetised, this is to avoid equilibrium convergence issues 
#if large straight edge domains form below the critical temperature
#
#The equilibrium values are estimated from the means of the relevant histories
#and their errors from the standard errors in the means (sdt/sqrt(n))

temperatures = np.array([])
temperatures = np.append(temperatures, np.linspace(0.1, 2.1, 11))
#More points are generated close to the critical point, as the curves change
#rapidly here
temperatures = np.append(temperatures, np.linspace(2.2, 2.3, 11))
temperatures = np.append(temperatures, np.linspace(2.4, 4.4, 11))

energy_T = np.array([])
E_errs = np.array([])
heatcap_T = np.array([])
C_errs = np.array([])
meanmag_T = np.array([])
m_errs = np.array([])


print("The statistics, as functions of temperature, are being calculated:")
print("This took about 4mins on a single 3.2GHz core (100% load)")
print("Start", datetime.datetime.now())

for T in temperatures:
    #longer history for accuracy
    latt = SpinLattice(32, 32, T, 0, 1, KB, 1, 100)
    latt.advance_time(1000, False, False)
    
    Enorm = np.mean(latt.energy_history)/(KB*latt.N)
    E_err = np.std(latt.energy_history)/(np.sqrt(latt.s)*KB*latt.N)
    
    Cnorm = np.mean(latt.heat_capacity_history)/(KB*latt.N)
    C_err = np.std(latt.heat_capacity_history)/(np.sqrt(latt.s)*KB*latt.N)
    
    mnorm = np.abs(np.mean(latt.moment_history)/latt.N)
    m_err = np.std(latt.moment_history)/(np.sqrt(latt.s)*latt.N)
    
    
    energy_T = np.append(energy_T, Enorm)
    E_errs = np.append(E_errs, E_err)
    
    heatcap_T = np.append(heatcap_T, Cnorm)
    C_errs = np.append(C_errs, C_err)
    
    meanmag_T = np.append(meanmag_T, mnorm)
    m_errs = np.append(m_errs, m_err)
    
    print(T, "K is Done")

#provides two estimate of the critical temperature
critical_T = estimate_Tc(temperatures, heatcap_T)
critical_T2 = estimate_Tc2(temperatures, meanmag_T)
print("Critical Temperature Estimate (heat capacity)", critical_T)
print("Critical Temperature Estimate (magnetic moment)", critical_T2)
print("End", datetime.datetime.now(), "\n")



plt.figure()
plt.plot(temperatures, energy_T, color = "b", marker="x")
plt.xlabel("Dimensionless Temperature", fontsize=18)
plt.ylabel("Dimensionless Energy per Site", fontsize=18)
plt.title("Dimensionless Energy per Site as a Function \n of Temperature", 
          fontsize=22)
plt.errorbar(temperatures, energy_T, yerr = E_errs, ecolor='black', marker='', 
             linestyle = "None")
plt.savefig("Energy v Temp.pdf", bbox_inches='tight')
plt.close()


plt.figure()
plt.plot(temperatures, heatcap_T, color = "b", marker="x")
plt.xlabel("Dimensionless Temperature", fontsize=18)
plt.ylabel("Dimensionless Heat Capacity per Site", fontsize=18)
plt.title("Specific Heat Capacity as a Function \n of Temperature", 
          fontsize=22)
plt.errorbar(temperatures, heatcap_T, yerr = C_errs, marker='', 
             linestyle = "None")
plt.savefig("Heat Capacity v Temp.pdf", bbox_inches='tight')
plt.close()


plt.figure()
plt.plot(temperatures, meanmag_T, linestyle = "None", color = "k", marker="x")
plt.plot(temperatures, exact_magnetisation(temperatures))
plt.xlabel("Dimensionless Temperature", fontsize=18)
plt.ylabel("Dimensionless Moment Per Site", fontsize=18)
plt.ylim([0, 1.05])
plt.title("Dimensionless Magnetic Moment as a Function \n of Temperature",
          fontsize=22)
plt.errorbar(temperatures, meanmag_T, yerr = m_errs, ecolor='k', fmt='', 
             marker='', linestyle = "None")
plt.savefig("Mean Magnetic Moment v Temp.pdf",  bbox_inches='tight')
plt.close()
###############################################################################


###############################################################################
#SECTION 3
#A few binary images of a pre magnetised lattice being "heated" are collected

temps = np.array([1, 2, 2.2, 2.3, 3, 5])

print("Making some images")
print("This took about 10mins on a single 3.2GHz core (100% load)")
print("Start", datetime.datetime.now())

latt = SpinLattice(100, 100, 1, 0, 1, KB, 1)
for T in temps:
    latt.T = T
    latt.beta = 1/(T*KB)
    latt.advance_time(1000)
    
    #odd name to avoid problems with decimal points in filenames
    name = "Binary Lattice " + str(int(1000*T)) + "mK"
    binary_plot(name, latt.lattice)
    
print("End", datetime.datetime.now(), "\n")

###############################################################################


###############################################################################
#SECTION 4
#The normalised magnetic moment is considered as a function of
#time for a few different temperatures. A random spin initial lattice is used.

print("The moments, as functions of time, are being calculated")
print("This took about 4 mins on a 3.2GHz core (100% load)")
print("Start", datetime.datetime.now())

latt1    = SpinLattice(32, 32, 1, 0, 1, KB, 0)
latt2_2 = SpinLattice(32, 32, 2.2, 0, 1, KB, 0)
latt2_4 = SpinLattice(32, 32, 2.4, 0, 1, KB, 0)
latt3    = SpinLattice(32, 32, 3, 0, 1, KB, 0)

dat1     = latt1.advance_time(10000, True)
dat2_2   = latt2_2.advance_time(10000, True)
dat2_4   = latt2_4.advance_time(10000, True)
dat3     = latt3.advance_time(10000, True)

print("End", datetime.datetime.now(), "\n")

plt.figure(4)
plt.semilogx(dat1[0], dat1[2], label = "1")
plt.semilogx(dat2_2[0], dat2_2[2], label = "2.2")
plt.semilogx(dat2_4[0], dat2_4[2], label = "2.4")
plt.semilogx(dat3[0], dat3[2], label = "3")
plt.xlabel("Time Step", fontsize=18)
plt.ylabel("Dimensionless Moment Per Site", fontsize=18)
plt.title("Magnetic Moment as a Function of Time", fontsize=22)
plt.legend(loc = "best")
plt.savefig("Magnetic Moment_t_T.pdf", bbox_inches='tight')
plt.close()

###############################################################################


###############################################################################
#SECTION 5
#An applied magnetic field is examined here
#Let us define the normalised magnetic field H_n = H*mu/(ONSAGER_T*KB)
#Unlike in (most) previous sections, the changing magnetic field is examined on
#the same lattice

temps = np.array([1, 2, 2.4, 3, 5, 10])
H_n_up = np.linspace(-2, 2, 81)
H_n_down = H_n_up[::-1]
H_n_down = np.delete(H_n_down, 0)

#we have defined mu=1 so it will not appear here
H_fields_up = ONSANGER_T*KB*H_n_up
H_fields_down = ONSANGER_T*KB*H_n_down

print("A magnetic field is being applied to the lattice")
print("This took about 19mins on a single 3.2GHz core (100% load)")
print("Start", datetime.datetime.now())

for T in temps:
    
    moments_up = np.array([])
    m_up_errs = np.array([])
    moments_down = np.array([])
    m_down_errs = np.array([])
    
    latt = SpinLattice(32, 32, T, 0, 1, KB, 0, 100)
    for H_field in H_fields_up:
        latt.H = H_field
        #Only a short history is required for accurate moments, so few steps
        #are required
        latt.advance_time(250)
        
        moments_up = np.append(moments_up, latt.moment/latt.N)
        m_up_err = np.std(latt.moment_history)/(np.sqrt(latt.s)*latt.N)
        m_up_errs = np.append(m_up_errs, m_up_err)
    
    for H_field in H_fields_down:
        latt.H = H_field
        
        latt.advance_time(250)
        
        moments_down = np.append(moments_down, latt.moment/latt.N)
        m_down_err = np.std(latt.moment_history)/(np.sqrt(latt.s)*latt.N)
        m_down_errs = np.append(m_down_errs, m_down_err)
        
    plt.figure()
    plt.plot(H_n_up, moments_up, color = "b", marker="x")
    plt.errorbar(H_n_up, moments_up, yerr = m_up_errs, ecolor='b', fmt='', 
                 marker='', linestyle = "None")
                 
    plt.plot(H_n_down, moments_down, color = "r", marker="x")
    plt.errorbar(H_n_down, moments_down, yerr = m_down_errs, ecolor='r', 
                 fmt='', 
                 marker='', linestyle = "None")
    plt.xlabel("Dimensionless Applied Field", fontsize=24)
    plt.ylabel("Dimensionless Magnetic \n Moment Per Site", fontsize=24)
    plt.ylim([-1.05, 1.05])
    
    name = "Applied field " + str(int(1000*T)) + "mK"
    plt.savefig("%s.pdf" % name, bbox_inches='tight')
    plt.close()
    
    print(T, "K is Done")
        
print("End", datetime.datetime.now())
print("All the calculations are now complete")
###############################################################################