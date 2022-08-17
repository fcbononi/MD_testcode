#!/usr/bin/env python3


""" 
Molecular Dynamics Code 
"""

import numpy as np
#import re
import os
import matplotlib.pyplot as plt

"""
Defining global variables
"""
ko = 138.935456 #kJ. molˆ–1.nmˆ2.eˆ–2
r = 0.282 #nm      .
eps_Na = 0.011590 #.kJ/mol
eps_Cl = 0.4184 #kJ./mol
sig_Na = 0.332840 #.nm
sig_Cl = 0.440104 #nm
#Combination rule
sg = 1/2*(sig_Na + sig_Cl)
eps = np.sqrt(eps_Cl*eps_Na)
coords = []
m_Na=23
m_Cl=35.45
kb = 0.0083144621  #kJ/mol/K

 
 
#XYZ writer
def write_xyz(coords,step, output='trajectory.xyz'):
    n_atoms = len(coords)
    generic_string = 'xyz file'
    atom_str = atoms_to_names(coords)
    xyz = np.hstack((np.transpose(atom_str), coords[:,1:]*10))
    xyz_file = open('%s'%output,'a')
    xyz_file.write('%i \n'%n_atoms)
    xyz_file.write('frame {} {}\n'.format(step,generic_string))
    for i in range(n_atoms):
        atom = str(xyz[i][0])
        x = float(xyz[i][1])
        y = float(xyz[i][2])
        z = float(xyz[i][3])
        xyz_file.write("%s\t % .5f\t % .5f\t % .5f\n"%(atom,x,y,z))
    return 



#convert numbers to atom names (Na,Cl)
def atoms_to_names(coords):
    atom_str = []
    for atom in coords:
        if atom[0] == 1:
            atom_str.append('Na')
        if atom[0] == -1:
            atom_str.append('Cl')
    return [atom_str]
            
def l1_norm(tuples):
    norm = np.sum(tuples[1:], axis =1)
    return norm

def even(number):
    if number % 2 == 0:
        outcome='even' 
    else:
        outcome='odd'
    return outcome

def r(coord_1, coord_2):
    r = coords[1:,np.newaxis,:] - coords[np.newaxis,1:,:]
    return r
        
#create cube of size n
def cube(n,lattice_parameter):

    r = lattice_parameter
    length = range(0,n)
    atoms = np.zeros((n**3,1))
    cube = np.meshgrid(length,length,length)
    cube = np.reshape(cube,(3,n**3)).transpose()
    cube = np.concatenate((atoms,cube), axis=1)
    norm = l1_norm(cube)
    count=1
    if even(cube[0][0])=='even':
        cube[0][0]= 1

    else:
        cube[0][0]=-1 
    for number in norm:
        if even(number)=='even':
            cube[count][0]=1
        else:
            cube[count][0]=-1 
        count+=1
   
    cube[:,1:] *= r

    return cube

#print (cube(4, 0.282))

output = 'NaCl.xyz'
if os.path.isfile(output):
        os.remove(output)
write_xyz(cube(4, 0.282),0, output='NaCl.xyz')


def coords_to_mass(coords):
    mass = np.zeros(len(coords))
    count = 0
    for atom in coords[:,0]:
        if atom==1:
            mass[count]=m_Na
        if atom==-1:
            mass[count]=m_Cl 
        count+=1
    return mass

def sigma_6(coords):
    sigma = np.zeros(len(coords))
    count = 0
    for atom in coords[:,0]:
        if atom == 1:
            sigma[count]=sig_Na
        if atom == -1:
            sigma[count]=sig_Cl
        count += 1        
    sigma_mat = 1/2*(sigma[:,np.newaxis]+sigma[np.newaxis,:])
    sigma2_mat3 = np.square(np.square(sigma_mat))*np.square(sigma_mat)
    return sigma2_mat3

def epsilon(coords):
    epsilon = np.zeros(len(coords))
    count = 0
    for atom in coords[:,0]:
        if atom == 1:
            epsilon[count]=eps_Na
        if atom == -1:
            epsilon[count]=eps_Cl
        count += 1        
    epsilon2_mat = epsilon[:,np.newaxis]*epsilon[np.newaxis,:]
    return np.sqrt(epsilon2_mat)

def coulomb_potential(coords, ko=138.935456):
    c_diff = coords[:,np.newaxis,1:] - coords[np.newaxis,:,1:]
    r2_mat = np.sum(c_diff**2, axis=-1) #r^2
    np.fill_diagonal(r2_mat, 1.0) # prevent 1/0 in the next step
    r_mat = np.sqrt(r2_mat)
    q_mat = coords[:,np.newaxis,0]*coords[np.newaxis,:,0]
    e_mat = ko*q_mat/r_mat
    np.fill_diagonal(e_mat, 0.0)
    return np.sum(e_mat)/2

def coulomb_force(coords, ko=138.935456):
   c_diff = coords[:,np.newaxis,1:] - coords[np.newaxis,:,1:]
   r2_mat = np.sum(c_diff**2, axis=-1)
   r_mat = np.sqrt(r2_mat)
   r_mat3 = r2_mat*r_mat
   np.fill_diagonal(r_mat, 1.0) # prevent 1/0 in the next step
   q_mat = coords[:,np.newaxis,0]*coords[np.newaxis,:,0]
   f_coul_mat = ko*2*q_mat/r_mat#3
   np.fill_diagonal(f_coul_mat,0)
   return np.einsum('ijk,ij->ik', c_diff, f_coul_mat)

def LJ_force(coords):
   c_diff = coords[:,np.newaxis,1:] - coords[np.newaxis,:,1:]
   r2_mat = np.sum(c_diff**2, axis=-1)
   s6 = sigma_6(coords)
   r2_mat2 = np.square(r2_mat)
   np.fill_diagonal(r2_mat2, 1.0) # prevent 1/0 in the next step
   r2_matn4 = 1.0 / np.square(r2_mat2)
   r2_matn7 = np.square(r2_matn4) * r2_mat
   f_lj_mat = (-12.0*r2_matn7 * s6 + 6.0*r2_matn4) * 4 * epsilon(coords) * s6
   return -np.einsum('ijk,ij->ik', c_diff, f_lj_mat)

def LJ_potential(coords):
    c_diff = coords[:,np.newaxis,1:] - coords[np.newaxis,:,1:]
    r2_mat = np.sum(c_diff**2, axis=-1) #r^2
    s6 = sigma_6(coords)
    r2_mat2 = np.square(r2_mat) #r^4
    np.fill_diagonal(r2_mat2, 1.0) # prevent 1/0 in the next step
    r2_matn3 = r2_mat / np.square(r2_mat2) #r^-6
    r2_matn6 = np.square(r2_matn3) #r^-12
    e_mat = ((r2_matn6 * s6 - r2_matn3) * 4 * epsilon(coords) * s6)/2
    return np.sum(e_mat)
   
print ("Total Energy for the crystal: ", LJ_potential(cube(4, 0.282))+coulomb_potential(cube(4, 0.282)), "kJ/mol")

        
def sd_min(coords, n_steps = 1000, stepsize = 0.001):
    for step in range(n_steps):
        F = LJ_force(coords)+coulomb_force(coords)
        F_norm = np.einsum('ij,i->ij',F,1/np.sqrt(np.sum(np.square(F),axis=-1)))
        coords[:,1:] += stepsize*F_norm
    return coords

def pv_x (mass, T=1200, kb=0.0083144621):
    vx = np.arange(-0.5, 0.5, 0.001)
    kT = kb*T
    P_x = np.sqrt(mass/(2*np.pi*kT))*np.exp(-mass*(vx**2)/(2*kT))
    return P_x

def pv_x_1 (vel, mass, T=1200, kb=0.0083144621):
    kT = kb*T
    P_x = np.sqrt(mass/(2*np.pi*kT))*np.exp(-mass*(vel**2)/(2*kT))
    return P_x


def random (mass, T=1200, kb=0.0083144621):
    kT = kb*T
    #max value for the maxwell-boltzmann distribution
    MBmax = np.sqrt(mass/(2*np.pi*kT))
    #rand = 0
    while True: 
        #random number between -0.5 and 0.5
        vel = np.random.random() - 0.5 
        #random number between 0 and the maximum value of the distribution
        pp_x = np.random.random() * MBmax
        #find the value of the distribution at that point
        pv = pv_x_1(vel, mass, T)
        if (pp_x < pv):
            break
    return vel
        #rand += 1
        
def init_vel(coords, T):
    mass = coords_to_mass(coords)
    velocity = np.zeros_like(coords[:,1:])
    for i in range (len(velocity)):
        for j in range (3):
            velocity[i,j] = random(mass[i], T)
    return velocity
    
def dv_verlet(timestep, old_force, new_force, mass):
    F_over_m = np.einsum('ij,i->ij',(new_force+old_force),1/mass)
    dv = timestep*F_over_m/2
    return dv

def dx(Force, mass, velocity, dt=0.001):
    F_over_m = np.einsum('ij,i->ij',Force,1/mass)
    dx = velocity*dt + 0.5*F_over_m*dt**2
    dx = np.array(dx)
    return dx    

def Kinetic_Energy(velocity, mass):
    speed_squared = np.sum(np.square(velocity),axis=-1)
    KE = 0.5*np.sum(mass*speed_squared)
    return np.sum(KE)
    
def verlet(coords,vel,timestep, n_steps, freq, output='trajectory.xyz'):
    if os.path.isfile(output):
        os.remove(output)
    Step = []
    Energy = []
    step = 0
    old_force = 0
    mass = coords_to_mass(coords)
    for step in range(n_steps+1):
        new_force=LJ_force(coords)
        coords[:,1:] += dx(new_force, mass, vel, timestep)
        vel += dv_verlet(timestep, old_force, new_force, mass)
        KE = Kinetic_Energy(vel, mass)
        PE = LJ_potential(coords)
        TotE = KE + PE
        old_force=new_force
        if freq!=0:
            if step%freq==0:
                write_xyz(coords,step,output='trajectory.xyz')
        Step.append(step)
        Energy.append(TotE)
    return Energy, Step


coords_5c = cube(4, 0.282)
coords_5c = sd_min(coords_5c)
vel_5c = init_vel(coords_5c, 1200)
Energy_5c, steps_5c = verlet(coords_5c, vel_5c, 0.005,2000, 0)
Energy_diff = Energy_5c[-1]-Energy_5c[-2]
#Checking for energy differences to be smaller than 1 kJ/mol
if Energy_diff <= 1:
    print('IT IS CONSERVED!!I REPEAT: ENERGY IS CONSERVED!!!')


def verlet_with_andersen(coords,vel,timestep, n_steps, temp, thermostat_freq, output_freq, output='trajectory.xyz'):
    """
    Function for a constant temperature simulation where the velocities are reset every 1ps
    """
    if os.path.isfile(output):
        os.remove(output)
    Rg = []
    Step = []
    Energy = []
    step = 0
    old_force = 0
    mass = coords_to_mass(coords)
    for step in range(n_steps+1):
        if step%thermostat_freq==0:
            vel = init_vel(coords, temp)
        new_force=LJ_force(coords)
        coords[:,1:] += dx(new_force, mass, vel, timestep)
        vel += dv_verlet(timestep, old_force, new_force, mass)
        KE = Kinetic_Energy(vel, mass)# (mass/2)*flat
        PE = LJ_potential(coords)
        TotE = KE + PE
        old_force=new_force
        if output_freq!=0:
            if step%output_freq==0:
                write_xyz(coords,step,output='trajectory.xyz')
        Rg.append(rg(coords))
        Step.append(step)
        Energy.append(TotE)
    return Energy, Step, Rg

def rg (coords):
    r = np.sqrt(np.sum(coords[:,1:]**2, axis=-1))
    return np.std(r)

def thermo_average(Energy,variable,T):
    e = np.exp(-Energy/(kb*T))
    expect = np.sum(e*variable)/np.sum(e)
    return expect

"""
Calculating the radius of gyration
"""

coords_5d = coords_5c
vel_5d = init_vel(coords_5d, 1200)
Energy_5d, steps_5d, Rg_5d = verlet_with_andersen(coords_5d, vel_5d, 0.005,100000, 1200, 200, 200)
#plt.plot(np.array(steps_5d)*0.005, Rg_5d, 'r', label= '1200K')

Rg_data_5d = Rg_5d[20000:]
#chose a random value as post-equilibration data based on a few runs
mean_5d = np.sum(Rg_data_5d)/(len(Rg_data_5d))
Rg_average_5d = thermo_average(np.array(Energy_5d)[20000:],np.array(Rg_5d)[20000:],1200)
Rg_variance_5d = thermo_average(np.array(Energy_5d)[20000:],np.array(Rg_5d)[20000:]**2,1200)
Rg_fluctuations_5d = np.sqrt(Rg_variance_5d - Rg_average_5d**2)
print ('Mean Rg for 1200K = ', mean_5d, 'Fluctuation:', Rg_fluctuations_5d)



Energy_5d, steps_5d, Rg_5d = verlet_with_andersen(coords_5d, vel_5d, 0.005,100000, 1400, 200, 200)

coords_5f = coords_5c
vel_5f = init_vel(coords_5f, 1400)
Energy_5f, steps_5f, Rg_5f = verlet_with_andersen(coords_5f, vel_5f, 0.005,100000, 1400, 200, 200)
#plt.plot(np.array(steps_5f)*0.005, Rg_5d, 'b', label= '1400K')


plt.plot(np.array(steps_5d)*0.005, Rg_5d, 'r', label= '1200K')
plt.plot(np.array(steps_5f)*0.005, Rg_5f, 'b', label= '1400K')
plt.ylabel('Rg (nm)')
plt.xlabel('Simulation Time (ps)')
plt.title('Radius of gyration vs Time for different temperatures')
plt.legend(loc=2)


Rg_data_5f = Rg_5f[20000:]
#chose a random value as post-equilibration data based on a few runs
mean_5f = np.sum(Rg_data_5f)/(len(Rg_data_5f))
Rg_average_5f = thermo_average(np.array(Energy_5f)[20000:],np.array(Rg_5f)[20000:],1400)
Rg_variance_5f = thermo_average(np.array(Energy_5f)[20000:],np.array(Rg_5f)[20000:]**2,1400)
Rg_fluctuations_5f = np.sqrt(Rg_variance_5f - Rg_average_5f**2)
print ('Mean Rg for 1400K = ', mean_5f, 'Fluctuation: ', Rg_fluctuations_5f)

Energy_5g, steps_5g, Rg_5g = verlet_with_andersen(coords_5f, vel_5f, 0.005,100000, 1300, 200, 200)
