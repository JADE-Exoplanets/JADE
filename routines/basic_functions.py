#--------------------------------------------------------------------------------------------
#--- Basic functions used by the JADE code.
#--------------------------------------------------------------------------------------------
#--- Author: Mara Attia
#--------------------------------------------------------------------------------------------

import numpy as np
import os
from scipy import interpolate
from astropy import units as u
from astropy.units import cds
from astropy import constants as const


#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
# Units to be used: G = 1 ; [length] = AU ; [time] = year

cds.enable()
custom_mass = u.def_unit('custom_mass', u.AU**3/(const.G*u.yr**2))
custom_mass2 = u.def_unit('custom_mass2', u.Rjup**3/(const.G*u.min**2))
K6 = u.def_unit('K6', 1e6*u.K)
orbital_units = [u.mol, u.rad, u.cd, u.AU, u.yr, u.A, custom_mass, u.K]
atmo_units = [u.mol, u.rad, u.cd, u.Rjup, u.min, u.A, custom_mass2, K6]


#----------------------------------------------------------------------------------------------------------------------------------------
# Conversion units

Rjup2orb = ((1*u.Rjup).decompose(bases=orbital_units)).value
Rearth2orb = ((1*u.Rearth).decompose(bases=orbital_units)).value
Rcgs2orb = ((1*u.cm).decompose(bases=orbital_units)).value
Patm2si = ((1*custom_mass2/(u.Rjup*u.min**2)).decompose(bases=u.si.bases)).value
Patm2cgs = ((1*custom_mass2/(u.Rjup*u.min**2)).decompose(bases=u.cgs.bases)).value
Patm2orb = ((1*custom_mass2/(u.Rjup*u.min**2)).decompose(bases=orbital_units)).value
Pbar2atm = ((1*u.bar).decompose(bases=atmo_units)).value
matm2orb = ((1*custom_mass2).decompose(bases=orbital_units)).value
mcgs2orb = ((1*u.g).decompose(bases=orbital_units)).value
morb2earth = 1/((1*u.Mearth).decompose(bases=orbital_units)).value
mjup2orb = ((1.*u.Mjup).decompose(bases=orbital_units)).value
rhosi2atm = ((1*u.kg/u.m**3).decompose(bases=atmo_units)).value
rhocgs2atm = ((1*u.g/u.cm**3).decompose(bases=atmo_units)).value
rhocgs2orb = ((1*u.g/u.cm**3).decompose(bases=orbital_units)).value
opcgs2atm = ((1*u.cm**2/u.g).decompose(bases=atmo_units)).value
opcgs2orb = ((1*u.cm**2/u.g).decompose(bases=orbital_units)).value
Tatm2cgs = ((1*K6).decompose(bases=u.cgs.bases)).value
Lcgs2orb = ((1*u.erg/u.s).decompose(bases=orbital_units)).value 
Lsun2orb = ((1*u.Lsun).decompose(bases=orbital_units)).value


#----------------------------------------------------------------------------------------------------------------------------------------
# Technical functions for JADE

def heaviside2(x):
    if x<0.:return 0.
    if x>1.:return 1.
    return x

def heaviside3(x, y):
    if x < y: return y
    return x

def clean_string(string):
    l = []
    while string.find('-', 1) != -1:
        idx = string.find('-', 1)
        substring = string[:idx]
        if len(substring) > 0:
            l.append(substring)
            string = string[idx:]
    l.append(string)
    return l


#----------------------------------------------------------------------------------------------------------------------------------------
# Returns indices of closest value in an array

def closest(array, value):   
    array = np.asarray(array)
    idx = np.argwhere(np.abs(array - value) == np.min(np.abs(array - value))).flatten()
    return idx


#----------------------------------------------------------------------------------------------------------------------------------------
# Converts a string into a boolean
# Input: 's' (string)
# Output: the converted input (bool)

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
        raise ValueError('The string to convert must be "True" or "False".')
        

#----------------------------------------------------------------------------------------------------------------------------------------    
# Converts a mass from the simulation mass units to physical units
# Input: 'm' (scalar) the mass to be converted, 'unit' (string) the unit in which the mass should be expressed
# Output: the converted mass (scalar) 
        
def convert_mass(m, unit, bases='orbital'):    
    if unit not in ['Mjup', 'Msun']: raise ValueError('Invalid unit.')  
    if bases not in ['orbital', 'atmo']: raise ValueError('Invalid base.')
    units = {'Mjup':u.Mjup, 'Msun':u.Msun}    
    if bases == 'orbital':
        return (m*custom_mass/units[unit]).decompose().value
    else:
        return (m*custom_mass2/units[unit]).decompose().value


#----------------------------------------------------------------------------------------------------------------------------------------    
# Converts a distance from the simulation distance units to physical units
# Input: 'd' (scalar) the mass to be converted, 'unit' (string) the unit in which the mass should be expressed
# Output: the converted distance (scalar) 
        
def convert_distance(d, unit, bases='orbital'):    
    if unit not in ['Rjup', 'Rsun']: raise ValueError('Invalid unit.')    
    if bases not in ['orbital', 'atmo']: raise ValueError('Invalid base.')
    units = {'Rjup':u.Rjup, 'Rsun':u.Rsun}    
    if bases == 'orbital':
        return (d*u.AU/units[unit]).decompose().value
    else:
        return (d*u.Rjup/units[unit]).decompose().value


#----------------------------------------------------------------------------------------------------------------------------------------    
# Converts a pressure from the simulation pressure units to physical units
# Input: 'P' (scalar) the pressure to be converted, 'unit' (string) the unit in which the mass should be expressed
# Output: the converted pressure (scalar) 
        
def convert_pressure(P, unit, bases='orbital'):    
    if unit not in ['bar', 'Pa']: raise ValueError('Invalid unit.')    
    if bases not in ['orbital', 'atmo']: raise ValueError('Invalid base.')
    units = {'bar':u.bar, 'Pa':u.Pa}    
    if bases == 'orbital':
        return (P*custom_mass/(u.AU*u.yr**2*units[unit])).decompose().value
    else:
        return (P*custom_mass2/(u.Rjup*u.min**2*units[unit])).decompose().value


#----------------------------------------------------------------------------------------------------------------------------------------    
# Changes low values in a vector into zero
# Input: 'y' (list) array to be rounded
# Output: 'yround' (list) transformed array
    
def round_vector(y):
    imax = np.argmax([np.abs(yy) for yy in y])
    yround = []
    for i in range(len(y)):
        if i != imax and np.abs(y[i]) < 1e-10*np.abs(y[imax]):
            yround.append(0)
        else:
            yround.append(y[i])
    return yround


#----------------------------------------------------------------------------------------------------------------------------------------    
# Returns the minimum of a function within certain bounds
# Input: 'f' (callable) function to be minimized, 'bounds' (2-iterable) minimization bounds
# Output: value for which f is minimized
    
def minimize_manual(f, bounds, num=10, rec=1):
    x = np.linspace(bounds[0], bounds[1], num=num)
    y = [f(xx) for xx in x]
    idx = np.argmin(y)
    x_min = x[idx]
    for i in range(rec - 1):
        y_copy = y.copy()
        y_copy[idx] = 1e60
        idx2 = np.argmin(y_copy)
        x_min2 = x[idx2]
        bounds2 = [np.min((x_min, x_min2)), np.max((x_min, x_min2))]
        x2 = np.linspace(bounds2[0], bounds2[1], num=num+2)[1:-1]
        y2 = [f(xx) for xx in x2]
        x = np.concatenate((x, x2))
        y = np.concatenate((y, y2))
        idx = np.argmin(y)
        x_min = x[idx]
    return x_min


#----------------------------------------------------------------------------------------------------------------------------------------    
# Returns a conveniently formatted time
# Input: 't' (float) duration to be formatted, in seconds
# Output: formatted time (string) in the form 'd-hh:mm:ss' or ms if t < 1

def format_time(t):
    if t < 1:
        ms = t*1000.
        if ms < 1: 
            f = '< 1 ms'
        else:
            f = str(int(ms)) + ' ms'
    else:
        t = int(round(t))
        s = t%60
        m = t//60
        h = m//60
        m = m%60
        d = h//24
        h = h%24
        f = '{:02d}:{:02d}:{:02d}'.format(h, m, s)
        if d > 0: f = str(d) + '-' + f
    return f


#----------------------------------------------------------------------------------------------------------------------------------------    
# Creates the required directories of 'path' so as to have the full 'root' + 'path' directory
# Input: 'root', 'path' (strings)

def create_dir(root, path):
    root = root.rstrip('/') + '/'
    path = (path.lstrip('/')).rstrip('/')
    dirs = path.split('/')
    for d in dirs:
        try:
            os.mkdir(root + d)
        except FileExistsError:
            pass
        root = root + d + '/'


#----------------------------------------------------------------------------------------------------------------------------------------    
# Returns the index of the last file 'name_3DIGITS.npz' inside the 'path' directory
# Input: 'path'(string), 'name' (string)
# Output: index of last input (integer), 0 if no output

def last_out(path, name):
    f = os.listdir(path)
    f = [ff[:-4] for ff in f if ff.startswith(name) and ff.endswith('.npz')]
    i = 1
    while name + '_{:03d}'.format(i) in f and i < 1000:
        i += 1
    i = i - 1 if i < 1000 else 0
    return i


#----------------------------------------------------------------------------------------------------------------------------------------    
# Removes 'name_3DIGITS.npz' files inside the 'path' directory
# Input: 'path', 'name' (strings)

def flush_out(path, name):
    f = os.listdir(path)
    f = [ff for ff in f if len(ff) == len(name) + 8 and ff.startswith(name) and ff.endswith('.npz')]
    for ff in f:
        try:
            _ = int(ff[len(name) + 1:len(name) + 4])
            os.remove(path.rstrip('/') + '/' + ff)
        except TypeError:
            pass


#----------------------------------------------------------------------------------------------------------------------------------------
# Function that interpolates 1D datapoints.
# 
# Input:
# +++ xth (numpy array): 1D values for which the data has to be interpolated.
# +++ xdata (numpy array): 1D original abscissa data points. Must be sorted in ascending order.
# +++ ydata (numpy array): 1D original ordinate data points.
# 
# Output:
# +++ yth (numpy array): interpolated 1D values at xth.

def interpol(xth, xdata, ydata, kind='nearest'):
    if len(xdata) == 1:
        yth = np.ones(len(xth))*ydata
    else:
        finterpol = interpolate.interp1d(xdata, ydata, kind=kind, bounds_error=False, fill_value=(ydata[0], ydata[-1]))     
        yth = finterpol(np.asarray(xth))
    return yth