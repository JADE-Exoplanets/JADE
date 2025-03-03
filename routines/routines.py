#--------------------------------------------------------------------------------------------
#--- Main backend file for the JADE code.
#--------------------------------------------------------------------------------------------
#--- Author: Mara Attia
#--------------------------------------------------------------------------------------------

import numpy as np
import time as tm
import os
from bisect import bisect_left
from astropy import units as u
from astropy.units import cds
from astropy import constants as const
from scipy.integrate import solve_ivp, odeint, ode
from scipy.special import expn
from scipy.optimize import brute, minimize_scalar
from jitcode import jitcode, UnsuccessfulIntegration
from jitcode import y as y_jit
from jitcxde_common import conditional
from grid_search import mara_search
import symengine
import warnings

from chgcoord import ell2cart
from basic_functions import str_to_bool, format_time, create_dir, last_out, flush_out, convert_mass, convert_distance, minimize_manual, round_vector, heaviside3, \
                            Lcgs2orb, Rjup2orb, Rearth2orb, Rcgs2orb, Patm2si, Patm2cgs, Patm2orb, Pbar2atm, rhosi2atm, rhocgs2atm, opcgs2atm, opcgs2orb, Tatm2cgs, \
                            matm2orb, mcgs2orb, morb2earth, rhocgs2orb, Lsun2orb
from interpolation import Saumon, Opacities, Gamma, StarLum, PlanetLum, PlanetRad


# Ignore useless warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


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
#----------------------------------------------------------------------------------------------------------------------------------------
# Physical constants

c = (const.c.decompose(bases=orbital_units)).value
sigmaB = (const.sigma_sb.decompose(bases=orbital_units)).value
Gcgs = (const.G.decompose(bases=u.cgs.bases)).value



#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
# Number of cores for multiprocessing

if 'NUM_THREADSPROCESSES' in os.environ:
        ncpu = os.environ['NUM_THREADSPROCESSES']
        ncpu = int(ncpu)
else:
        ncpu = None



#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
# Abstract class modeling the simulation

class JADE_Simulation:
    
#----------------------------------------------------------------------------------------------------------------------------------------
# Constructor
    
    def __init__(self, global_path, input_path, verbose=False):
        
        # Setting the global path
        self.global_path = global_path
        
        # Reading the settings from the input text file
        self.read_settings(global_path + input_path, verbose=verbose)

        # Full initialization
        if not self.lazy_init:

            # Creating the output directory
            create_dir(global_path + 'saved_data/', self.out)
            
            # Checking any existing simulation
            self.last_out = last_out(self.global_path + 'saved_data/' + self.out + '/', self.name)
            if self.last_out == 0: self.reuse = False
            
            # Loading last state
            if self.reuse:
                npzfile = np.load('{}saved_data/{}/{}_{:03d}.npz'.format(self.global_path, self.out, self.name, self.last_out), allow_pickle=True)

                self.t = [npzfile['t'][-1]]
                self.h1 = [npzfile['h1'][-1]]
                self.h2 = [npzfile['h2'][-1]]
                self.h3 = [npzfile['h3'][-1]]
                self.e1 = [npzfile['e1'][-1]]
                self.e2 = [npzfile['e2'][-1]]
                self.e3 = [npzfile['e3'][-1]]
                self.Os1 = [npzfile['Os1'][-1]]
                self.Os2 = [npzfile['Os2'][-1]]
                self.Os3 = [npzfile['Os3'][-1]]
                self.Op1 = [npzfile['Op1'][-1]]
                self.Op2 = [npzfile['Op2'][-1]]
                self.Op3 = [npzfile['Op3'][-1]]
                self.e = [npzfile['e'][-1]]
                self.a = [npzfile['a'][-1]]
                self.Mp = [npzfile['Mp'][-1]]
                self.Rp = [npzfile['Rp'][-1]]
                self.t_atmo = [npzfile['t_atmo'][-1]]

                self.Rcore_eval = False
                
                if verbose: 
                    if self.parallel: print('  + Parallel computing on', str(self.parallel), 'cores activated.')
                    print('  + Previously saved data was loaded. Current time: ' + format(self.t[-1], '-.5') + ' years.')
                
                # Calculating the planetary luminosity
                if self.atmo:
                    self.Lp = self.planet_luminosity(self.t[-1])

                # Manual atmospheric timescales
                if not self.atmo_eval:
                    self.atmo_eval_manual = [True if t < self.t[-1] else False for t in self.t_atmo_manual]

            # OR initializing a new one
            else:
                self.last_out = 0
                flush_out(self.global_path + 'saved_data/' + self.out + '/', self.name)

                self.h1 = []
                self.h2 = []
                self.h3 = []
                self.e1 = []
                self.e2 = []
                self.e3 = []
                self.Os1 = []
                self.Os2 = []
                self.Os3 = []
                self.Op1 = []
                self.Op2 = []
                self.Op3 = []
                self.e = []
                self.a = []
                self.Mp = []
                self.Rp = []
                self.t = []
                self.t_atmo = []
                
                Y0 = self.set_initial_vector(verbose=verbose)
                self.h1.append(Y0[0])
                self.h2.append(Y0[1])
                self.h3.append(Y0[2])
                self.e1.append(Y0[3])
                self.e2.append(Y0[4])
                self.e3.append(Y0[5])
                self.Os1.append(Y0[6])
                self.Os2.append(Y0[7])
                self.Os3.append(Y0[8])
                self.Op1.append(Y0[9])
                self.Op2.append(Y0[10])
                self.Op3.append(Y0[11])
                self.e.append(Y0[12])
                self.a.append(Y0[13])
                self.Mp.append(Y0[14])

                self.t.append(self.t_init)
                self.t_atmo.append(self.t_init)
                
                # Calculating the initial planetary radius
                if self.init_Rp is None:
                    
                    # Calculating the initial planetary luminosity
                    self.Lp = self.planet_luminosity(self.t_init)
                    
                    #self.Rp.append(420.)
                    self.Rp.append(self.retrieve_radius(self.t_init, self.init_Mp, self.init_orb_pl['a'], self.init_orb_pl['e'], \
                                                        force_comp=False, verbose=verbose))
                else:
                    self.Rp.append(self.init_Rp) 
                
                if verbose: 
                    if self.parallel: print('  + Parallel computing on', str(self.parallel), 'cores activated.')
                    print('  + The simulation was successfully initialised.')

                # Checking if the planet crashed into the star
                if self.roche and self.a[-1]*(1. - self.e[-1]) < 2*self.Rp[-1]*(self.Ms/self.Mp[-1])**(1./3):
                    print('  + Planet tidally disrupted at t = {:.2e} years.'.format(self.t[-1]))
                    self.simul = False
            
            # Progression variables
            self.prog_print = 0
            self.prog_save = 0
            self.t0 = self.t[-1]
            self.tau_atmo = self.atmo_timescale(-1) if self.atmo else 1e100
            self.tau_dyn = np.min((self.kozai_timescale(-1), self.tidal_timescale(-1), 1e7))
            self.tau_dyn = np.max((self.tau_dyn, 1e5))
            self.Rcore_eval = False
            
            # Printing useful timescales
            if verbose:
                if self.dyn and self.perturber:
                    print('       . Dynamical timescale: ' + format(self.tau_dyn, '-.3') + ' years.')
                if self.atmo:
                    print('       . Atmospheric timescale: ' + format(self.tau_atmo, '-.3') + ' years.')

            # Initializing symbols for integrator
            self.eps_sym = symengine.Symbol('eps_sym')
            self.RXUV_sym = symengine.Symbol('RXUV_sym')
            self.LXUV_sym = symengine.Symbol('LXUV_sym')
            self.Rp_sym = symengine.Symbol('Rp_sym')

        # Lazy initialization
        else:

            pass
        
        
        
        

#----------------------------------------------------------------------------------------------------------------------------------------
# Reads the settings from a text file
# Input: 'path' (string) path to the text file
        
    def read_settings(self, path, verbose=False):
        
        # Reading the settings from the input file
        settings = read_input_file(path)
        
        #------------------------------------------------------
        # Storing the settings
        
        self.name = settings['name']
        if verbose: print('  + Simulation name:', self.name)
        self.out = (settings['output_dir'].rstrip('/')).lstrip('/')
        self.freq_save = int(settings['output_freq'])
        if self.freq_save <= 0 or self.freq_save > 100: raise ValueError('Invalid save frequency.')
        if settings['output_npts'] != '':
            self.npts_save = int(settings['output_npts'])
            if self.npts_save <=0: raise ValueError('Invalid `output_npts`.')
            self.npts_save_npz = int(self.npts_save*self.freq_save/100)
        else:
            self.npts_save = None
            self.npts_save_npz = None
        self.age = ((settings['age']*1e6*u.yr).decompose(bases=orbital_units)).value
        self.t_init = ((settings['t_init']*1e6*u.yr).decompose(bases=orbital_units)).value
        self.dyn = settings['dyn']
        self.orderdyn = settings['orderdyn']
        self.atmo = settings['atmo']
        if settings['atmo_grid_path'] != '':
            self.atmo_grid = True
            self.atmo_grid_path = self.global_path + 'atmo_grids/' + settings['atmo_grid_path']
        else:
            self.atmo_grid = False
            self.atmo_grid_path = None
        self.atmo_acc = settings['atmo_acc']
        self.evap = settings['evap']
        self.parallel = int(settings['parallel']) if ncpu is None else ncpu
        self.perturber = settings['perturber']
        self.tides = settings['tides']
        self.relat = settings['relat']
        try:
            self.roche = settings['roche']
        except KeyError:
            self.roche = True
        try:
            self.lazy_init = settings['lazy_init']
        except KeyError:
            self.lazy_init = False
        self.reuse = settings['reuse']
        self.simul = settings['simul']
        
        # Host's parameters
        self.Ms = ((settings['Ms']*u.Msun).decompose(bases=orbital_units)).value
        self.Rs = ((settings['Rs']*u.Rsun).decompose(bases=orbital_units)).value
        self.ks = settings['ks']
        self.Qs = settings['Qs']
        self.alphas = settings['alphas']
        self.init_Os = ((settings['spins']/u.yr).decompose(bases=orbital_units)).value
        
        # Inner planet's parameters
        self.Mcore = ((settings['Mcore']*u.Mjup).decompose(bases=orbital_units)).value
        self.init_Mp = ((settings['Mpl']*u.Mjup).decompose(bases=orbital_units)).value
        if self.init_Mp < self.Mcore: raise ValueError('The total mass cannot be less than the core''s mass.')
        if settings['Rpl'] != '':
            self.init_Rp = ((settings['Rpl']*u.Rjup).decompose(bases=orbital_units)).value
        else:
            self.init_Rp = None
        self.kp = settings['kpl']
        self.Qp = settings['Qpl']
        self.alphap = settings['alphapl']
        self.init_Op = ((settings['spinpl']/u.yr).decompose(bases=orbital_units)).value
        
        # Atmospheric parameters
        if settings['t_atmo'] == '':
            self.atmo_eval = True
            self.atmo_eval_manual = [True]
        else:
            self.atmo_eval = False
            self.t_atmo_manual = [((t*1e6*u.yr).decompose(bases=orbital_units)).value for t in settings['t_atmo']]
            self.atmo_eval_manual = [False for t in self.t_atmo_manual]
            self.atmo_eval_manual[0] = True
        if settings['stellar_lum'] not in ['analytic', 'tabular']: 
            raise ValueError('Invalid stellar luminosity mode.')
        else:
            self.lum_mode = settings['stellar_lum']
        if self.lum_mode == 'analytic':
            self.init_Lbol = ((settings['Lbol']*u.erg/u.s).decompose(bases=orbital_units)).value
            if settings['LX_Lbol_sat'] != '':
                self.LX_Lbol_sat = settings['LX_Lbol_sat']
            else:
                self.LX_Lbol_sat = 10**(np.mean([-4.28, -4.24, -3.67, -3.71, -3.36, -3.35, -3.14]))
            if settings['tau_X_bol_sat'] != '':
                self.tau_X_bol_sat = settings['tau_X_bol_sat']
            else:
                self.tau_X_bol_sat = 10**(np.mean([7.87, 8.35, 7.84, 8.03, 7.9, 8.28, 8.21]))
            if settings['alpha_X_bol'] != '':
                self.alpha_X_bol = settings['alpha_X_bol']
            else:
                self.alpha_X_bol = np.mean([1.22, 1.24, 1.13, 1.28, 1.4, 1.09, 1.18])
        else:
            self.lum_path = self.global_path + 'luminosities/' + settings['stellar_lum_path']
            if verbose: print('  + Interpolating stellar luminosities...')
            stellar_tables = StarLum(self.lum_path)
            self.Lbol_tabular = stellar_tables.L_bol
            if verbose: print('       . Bolometric luminosities: ok.')
            self.LXUV_tabular = stellar_tables.L_XUV
            if verbose: print('       . XUV luminosities: ok.')
        self.YHe = settings['YHe']
        try:
            self.Zmet = settings['Zmet']
        except KeyError:
            self.Zmet = 0.
        try:
            self.fmantle = settings['fmantle']
        except KeyError:
            self.fmantle = 2./3
        
        # Interpolated atmospheric tables
        if self.init_Rp is None:
            if verbose: print('  + Interpolating atmospheric tables...')
            saumon_tables = Saumon(self.YHe, self.global_path, self.out, self.name, self.reuse, kind='Rbf')
            self.rho = saumon_tables.rho
            if verbose: print('       . Density tables: ok.')
            self.adg = saumon_tables.adg
            if verbose: print('       . Adiabatic gradient tables: ok.')
            opacity_tables = Opacities(self.YHe, self.Zmet, self.global_path)
            self.kappa = opacity_tables.kappa
            if verbose: 
                Y, Z = opacity_tables.get_composition()
                print('       . Rosseland-mean opacity tables: ok. Used composition: Y = {:.5f}, Z = {:.5f}.'.format(Y, Z))
            gamma_tables = Gamma()
            self.gamma = gamma_tables.gamma
            if verbose: print('       . Visible to thermal opacity ratio tables: ok.')
            planetary_tables = PlanetLum()
            self.a0_tabular = planetary_tables.a0
            self.b1_tabular = planetary_tables.b1
            self.b2_tabular = planetary_tables.b2
            self.c1_tabular = planetary_tables.c1
            self.c2_tabular = planetary_tables.c2
            if verbose: print('       . Planetary luminosity coefficients: ok.')
            if self.atmo_grid:
                planet_rad = PlanetRad(self.atmo_grid_path)
                self.radius_fit = planet_rad.interps
                if verbose: print('       . Planetary radius coefficients: ok.')

        # Perturber's parameters
        self.Mpert = ((settings['Mpert']*u.Mjup).decompose(bases=orbital_units)).value
        if not self.perturber: self.Mpert = 0.
        
        # Initial inner planet's orbital elements
        self.init_orb_pl = {}
        self.init_orb_pl['a'] = ((settings['planet_sma']*u.AU).decompose(bases=orbital_units)).value
        self.init_orb_pl['e'] = settings['planet_ecc']
        self.init_orb_pl['i'] = settings['planet_incl']
        self.init_orb_pl['l'] = settings['planet_lambd']
        self.init_orb_pl['o'] = settings['planet_omega']
        self.init_orb_pl['O'] = settings['planet_Omega']
        
        # Perturber's orbital elements
        self.orb_pert = {}
        self.orb_pert['a'] = ((settings['pert_sma']*u.AU).decompose(bases=orbital_units)).value
        self.orb_pert['e'] = settings['pert_ecc']
        self.orb_pert['i'] = settings['pert_incl']
        self.orb_pert['l'] = settings['pert_lambd']
        self.orb_pert['o'] = settings['pert_omega']
        self.orb_pert['O'] = settings['pert_Omega']
        
        # Pertuber's orbital vectors
        orb_pert = [self.orb_pert['a'], self.orb_pert['l'], \
                    self.orb_pert['e']*np.cos(self.orb_pert['o']*np.pi/180), self.orb_pert['e']*np.sin(self.orb_pert['o']*np.pi/180), \
                    np.sin(self.orb_pert['i']*np.pi/360)*np.cos(self.orb_pert['O']*np.pi/180), \
                    np.sin(self.orb_pert['i']*np.pi/360)*np.sin(self.orb_pert['O']*np.pi/180)]
        cmu_pert = self.Ms + self.init_Mp + self.Mpert
        R, R_dot = ell2cart(orb_pert, cmu_pert)
        self.H = np.cross(R, R_dot)
        self.E = np.cross(R_dot, self.H)/cmu_pert - R/np.linalg.norm(R)
        self.Q = np.cross(self.H, self.E)
        self.H1 = self.H/np.linalg.norm(self.H)
        self.E1 = self.E/np.linalg.norm(self.E)
        self.Q1 = self.Q/np.linalg.norm(self.Q)
        
        
#----------------------------------------------------------------------------------------------------------------------------------------
# Returns the initial evolving vector
# Output: 13-list [h1, h2, h3, e1, e2, e3, Os1, Os2, Os3, Op1, Op2, Op3, Mp]
        
    def set_initial_vector(self, verbose=False):
        
        # Converting the orbital parameters to cartesian coordinates
        orb = [self.init_orb_pl['a'], self.init_orb_pl['l'], \
               self.init_orb_pl['e']*np.cos(self.init_orb_pl['o']*np.pi/180), \
               self.init_orb_pl['e']*np.sin(self.init_orb_pl['o']*np.pi/180), \
               np.sin(self.init_orb_pl['i']*np.pi/360)*np.cos(self.init_orb_pl['O']*np.pi/180), \
               np.sin(self.init_orb_pl['i']*np.pi/360)*np.sin(self.init_orb_pl['O']*np.pi/180)]
        cmu = self.Ms + self.init_Mp
        r, r_dot = ell2cart(orb, cmu)
        
        # Calculating the orbital vectors
        h = np.cross(r, r_dot)
        e = round_vector((np.cross(r_dot, h)/cmu - r/np.linalg.norm(r))/self.init_orb_pl['e'])
        h = round_vector(h/np.linalg.norm(h))
        
        # Setting the spin vectors
        Os = round_vector(self.set_initial_spins(h))
        Op = round_vector(self.set_initial_spinp(h))
        
        # Setting the initial mass of the inner planet
        Mp = [self.init_Mp]          
        
        # Concatenating
        Y = np.concatenate((h, e, Os, Op, [self.init_orb_pl['e']], [self.init_orb_pl['a']], Mp))
        
        return Y
    

#----------------------------------------------------------------------------------------------------------------------------------------
# Returns the initial stellar spin vector (parallel to h)
# Input: h (3-list)
# Output: Os (3-list)

    def set_initial_spins(self, h):
        
        Os1 = h
        Os = [self.init_Os*O for O in Os1]
        return Os
    

#----------------------------------------------------------------------------------------------------------------------------------------
# Returns the initial planetary spin vector (parallel to h)
# Input: h (3-list)
# Output: Op (3-list)

    def set_initial_spinp(self, h):
        
        Op1 = h
        Op = [self.init_Op*O for O in Op1]
        return Op
        
        
#----------------------------------------------------------------------------------------------------------------------------------------
# Prints the state of the system at the last step

    def print_last_state(self):
        
        # Check-quantities
        e = self.e[-1]
        a = self.a[-1]
        i = np.arccos(self.h3[-1])*180/np.pi
        if self.perturber: i_mut = np.arccos(self.H1[0]*self.h1[-1] + self.H1[1]*self.h2[-1] + self.H1[2]*self.h3[-1])*180./np.pi
        RXUV = self.RXUV(self.t[-1], a, self.Mp[-1], self.Rp[-1])/self.Rp[-1]
        eps = self.eps_evap(self.Mp[-1], self.Rp[-1])
        
        print('  + State of the system at t = ' + format(float(self.t[-1]), '-.5') + ' years...')
        print('')
        print('       . Evolving vector:')
        print('       ... h1 = [' + format(self.h1[-1], '-.3') + ' ; ' + format(self.h2[-1], '-.3') + ' ; ' + format(self.h3[-1], '-.3') + ']')
        print('       ... e1 = [' + format(self.e1[-1], '-.3') + ' ; ' + format(self.e2[-1], '-.3') + ' ; ' + format(self.e3[-1], '-.3') + ']')
        print('       ... Omega_s = [' + format(self.Os1[-1], '-.3') + ' ; ' + format(self.Os2[-1], '-.3') + ' ; ' + format(self.Os3[-1], '-.3') + ']')
        print('       ... Omega_pl = [' + format(self.Op1[-1], '-.3') + ' ; ' + format(self.Op2[-1], '-.3') + ' ; ' + format(self.Op3[-1], '-.3') + ']')
        print('       ... M_pl = ' + format(convert_mass(self.Mp[-1], 'Mjup'), '-.3') + ' M_jup')
        print('       ... R_pl = ' + format(convert_distance(self.Rp[-1], 'Rjup'), '-.3') + ' R_jup')
        print('')
        print('       . Inner parameters:')
        print('       ... Eccentricity: ' + format(e, '-.3'))
        print('       ... Semi-major axis: ' + format(a, '-.3') + ' AU')
        print('       ... Inclination: ' + format(i, '-.3') + '°', end='')
        if self.perturber: 
            print(' (mutual inclination: ' + format(i_mut, '-.3') + '°)')
        else:
            print('')
        if self.atmo: 
            print('       ... XUV radius: ' + format(RXUV, '-.3') + ' R_pl')
            print('       ... Evaporation efficiency: ' + format(eps, '-.3'))
        print('')
        print('       . Constant quantities:')
        print('       ... M_s = ' + format(convert_mass(self.Ms, 'Msun'), '-.3') + ' M_sun')
        print('       ... R_s = ' + format(convert_distance(self.Rs, 'Rsun'), '-.3') + ' R_sun')
        if self.perturber: print('       ... M_pert = ' + format(convert_mass(self.Mpert, 'Mjup'), '-.3') + ' M_jup')
        


#----------------------------------------------------------------------------------------------------------------------------------------
# Returns the Kozai timescale at step i
# Input: 'i' (integer) 
# Output: tau_Kozai
        
    def kozai_timescale(self, i):
        if not self.dyn or not self.perturber: return 1e100
        Mp = self.Mp[i]
        a = self.a[i]
        P = 2*np.pi/np.sqrt((self.Ms + Mp)/a**3)
        Ppert = 2*np.pi/np.sqrt((self.Ms + Mp + self.Mpert)/self.orb_pert['a']**3)
        tau = (2*Ppert**2/(3*np.pi*P))*((self.Ms + Mp + self.Mpert)/self.Mpert)*(1 - self.orb_pert['e']**2)**(3/2)
        return tau


#----------------------------------------------------------------------------------------------------------------------------------------
# Returns the tidal timescale at step i
# Input: 'i' (integer) 
# Output: tau_tide
        
    def tidal_timescale(self, i):
        if not self.dyn or not self.tides: return 1e100
        Mp = self.Mp[i]
        Rp = self.Rp[i]
        kp = self.kp/self.Qp
        ks = self.ks/self.Qs
        a = self.a[i]
        e = self.e[i]
        n = np.sqrt((self.Ms + Mp)/a**3)
        def gTD(m, m2, r, k):
            return k*r**5/(m*(1. + m/m2))
        fact1 = 3*n**2/(8*a**2*(1. - e**2)**5)
        fact2 = n*(9./4)*(64. + 5*e**2*(48. + 24*e**2 + e**4))/(1. - e**2)**(3./2)
        tau = 1./(fact1*fact2*(gTD(self.Ms, Mp, self.Rs, ks) + gTD(Mp, self.Ms, Rp, kp)))
        return tau
    
    
#----------------------------------------------------------------------------------------------------------------------------------------
# Returns the photo-evaporation timescale at step i
# Input: 'i' (integer) 
# Output: tau_atmo
        
    def atmo_timescale(self, i):
        
        if not self.atmo: return 1e100

        if not self.atmo_eval:
            try:
                idx = self.atmo_eval_manual.index(False)
            except ValueError:
                return 1e100   
            self.atmo_eval_manual[idx] = True
            return self.t_atmo_manual[idx] - self.t_atmo_manual[idx - 1]             

        tau_max = 1e7
        tau_min = 1e5
        tau_kozai = self.kozai_timescale(i)/10.

        if self.evap:                
            t = self.t[i]
            if i != -1: 
                i_atm = bisect_left(self.t_atmo, t)
                if i_atm == len(self.t_atmo): i_atm = -1
            else:
                i_atm = i
            eta = 1
            Mp = self.Mp[i]
            Menv = Mp - self.Mcore
            Rp = self.Rp[i_atm]
            if Menv == 0.: 
                if not self.Rcore_eval:
                    self.Rcore_eval = True
                    return 1.
                else:
                    return 1e100
            e = self.e[i]
            a = self.a[i]
            ksi = (a/Rp)*(1 + .5*e**2)*(Mp/(3*self.Ms))**(1/3)
            Ktide = 1 - 3/(2*ksi) + 1/(2*ksi**3)
            tau = eta*Menv*Mp*Ktide*np.sqrt(1 - e**2)*a**2/(self.eps_evap(Mp, Rp)*np.pi*self.L_XUV(t, a)*Rp*self.RXUV(t, a, Mp, Rp)**2)
        else:        
            tau = 1e100

        tau = np.min((tau, tau_kozai, tau_max))
        tau = np.max((tau, tau_min))
        
        return tau
    
        
#----------------------------------------------------------------------------------------------------------------------------------------
# Returns the time derivative of the evolving vector 
# Input: 't' (scalar) time variable, 'Y' (15-list) evolving vector 
# Output: time derivative of 'Y' (15-list)
        
    def evolving_function(self, t, Y, verbose=False):
        
        # Storing the evolving quantities
        h1, h2, h3, e1, e2, e3, Os1, Os2, Os3, Op1, Op2, Op3, e, a, Mp = Y
        if Mp < self.Mcore: Mp = self.Mcore
        
        # Orbital properties
        n = np.sqrt((self.Ms + Mp)/a**3)
        h = np.sqrt((self.Ms + Mp)*a*(1 - e**2))
        
        # Evolution of the radius
        Rp = self.Rp[-1]
            
        # Moments of inertia
        Is = self.alphas*self.Ms*self.Rs**2
        Ip = self.alphap*Mp*Rp**2
            
        # Unitary vectors
        e_unit = np.sqrt(e1**2 + e2**2 + e3**2)
        h_unit = np.sqrt(h1**2 + h2**2 + h3**2)
        e1 /= e_unit
        e2 /= e_unit
        e3 /= e_unit
        h1 /= h_unit
        h2 /= h_unit
        h3 /= h_unit
        q1 = h2*e3 - h3*e2
        q2 = h3*e1 - h1*e3
        q3 = h1*e2 - h2*e1         
        
        if self.dyn:
            
            # Perturber's projected orbital vectors
            He = self.H1[0]*e1 + self.H1[1]*e2 + self.H1[2]*e3
            Hq = self.H1[0]*q1 + self.H1[1]*q2 + self.H1[2]*q3
            Hh = self.H1[0]*h1 + self.H1[1]*h2 + self.H1[2]*h3
            
            # Quadrupolar disturbing force
            dpert_factor = 3*self.Mpert*a**2 / (4*self.orb_pert['a']**3*(1 - self.orb_pert['e']**2)**(3./2))
            dpert_ee = -(1 - e**2)*Hq*Hh
            dpert_qq = (1 + 4*e**2)*He*Hh
            dpert_hh = - 5*e**2*He*Hq
            dpert1 = dpert_factor * (dpert_ee*e1 + dpert_qq*q1 + dpert_hh*h1)
            dpert2 = dpert_factor * (dpert_ee*e2 + dpert_qq*q2 + dpert_hh*h2)
            dpert3 = dpert_factor * (dpert_ee*e3 + dpert_qq*q3 + dpert_hh*h3)
            dpert_h = dpert_factor*dpert_hh
            
            gpert_factor= 3*n*self.Mpert*e*np.sqrt(1 - e**2)*a**3 / \
                          (4*(self.Ms + Mp)*self.orb_pert['a']**3*(1 - self.orb_pert['e']**2)**(3/2))
            gpert_ee = 5*He*Hq
            gpert_qq = -3 + 5*Hq**2 + 4*Hh**2
            gpert_hh = Hq*Hh
            gpert1 = gpert_factor * (gpert_ee*e1 + gpert_qq*q1 + gpert_hh*h1)
            gpert2 = gpert_factor * (gpert_ee*e2 + gpert_qq*q2 + gpert_hh*h2)
            gpert3 = gpert_factor * (gpert_ee*e3 + gpert_qq*q3 + gpert_hh*h3)
            gpert_e = gpert_factor*gpert_ee
                     
            # Octupolar disturbing force
            if self.orderdyn > 2:
                Qe = self.Q1[0]*e1 + self.Q1[1]*e2 + self.Q1[2]*e3
                Qq = self.Q1[0]*q1 + self.Q1[1]*q2 + self.Q1[2]*q3
                Qh = self.Q1[0]*h1 + self.Q1[1]*h2 + self.Q1[2]*h3
                Ee = self.E1[0]*e1 + self.E1[1]*e2 + self.E1[2]*e3
                Eq = self.E1[0]*q1 + self.E1[1]*q2 + self.E1[2]*q3
                Eh = self.E1[0]*h1 + self.E1[1]*h2 + self.E1[2]*h3
                
                dpert_factor = 15*self.Mpert*(self.Ms - Mp)*a**3*e*self.orb_pert['e'] / (64*(self.Ms + Mp)*self.orb_pert['a']**4* \
                                                                                    (1 - self.orb_pert['e']**2)**(5/2))
                dpert_ee = 10*(1 - e**2)*(Qe*(Eh*Qq + Eq*Qh) + Ee*(3*Eq*Eh + Qq*Qh))
                dpert_qq = -(Eh*(-1 + 8*e**2 + (45 + 60*e**2)*Hh**2 - 5*(2 + 5*e**2)*Qq**2) + 5*(2 + 5*e**2)*Eq*(3*Hq*Hh + Qq*Qh))
                dpert_hh = -10*(1 + 6*e**2)*Eh*Qq*Qh + Eq*(-11 + 15*Hh**2 + 10*Qh**2 + 3*e**2*(-29 + 35*Hq**2 + 30*Hh**2 + 20*Qh**2))
                dpert1 += dpert_factor * (dpert_ee*e1 + dpert_qq*q1 + dpert_hh*h1)
                dpert2 += dpert_factor * (dpert_ee*e2 + dpert_qq*q2 + dpert_hh*h2)
                dpert3 += dpert_factor * (dpert_ee*e3 + dpert_qq*q3 + dpert_hh*h3)
                dpert_h += dpert_factor*dpert_hh
                
                gpert_factor = -(15*n*self.Mpert*(self.Ms - Mp)*a**4*np.sqrt(1 - e**2)*self.orb_pert['e'] / \
                          (64*(self.Ms + Mp)**2*self.orb_pert['a']**4*(1 - self.orb_pert['e']**2)**(5/2)))
                gpert_ee = -10*(1 + 6*e**2)*Eh*Qq*Qh + Eq*(-11 + 15*Hh**2 + 10*Qh**2 + 3*e**2*(-29 + 35*Hq**2 + 30*Hh**2 + 20*Qh**2))
                gpert_qq = -((-10 + 30*e**2)*Eh*Qe*Qh + Ee*(-11 + 15*Hh**2 + 10*Qh**2 + 3*e**2*(-17 + 35*Hq**2 + 20*Hh**2 - 10*Qh**2)))
                gpert_hh = -10*e**2*(Qe*(Eh*Qq + Eq*Qh) + Ee*(3*Eq*Eh + Qq*Qh))                            
                gpert1 += gpert_factor * (gpert_ee*e1 + gpert_qq*q1 + gpert_hh*h1)
                gpert2 += gpert_factor * (gpert_ee*e2 + gpert_qq*q2 + gpert_hh*h2)
                gpert3 += gpert_factor * (gpert_ee*e3 + gpert_qq*q3 + gpert_hh*h3)
                gpert_e += gpert_factor*gpert_ee
                 
                # Hexadecapolar disturbing force
                if self.orderdyn > 3:
                    
                    dpert_factor = 45*self.Mpert*(self.Ms**3 + Mp**3)*a**4 / (256*(self.Ms + Mp)**3*self.orb_pert['a']**5* \
                                                                         (1 - self.orb_pert['e']**2)**(7/2))
                    dpert_ee = (1-e**2)*(98*e**2*self.orb_pert['e']**2*Hq**2*Qq*Qh+Hh*Hq*(e**2*(self.orb_pert['e']**2*(210*Hh**2+\
                               98*Qq**2+84*Qh**2-221)+84*Hh**2-78)+self.orb_pert['e']**2*(35*Hh**2+14*Qh**2-17)+14*Hh**2-6)+\
                               2*self.orb_pert['e']**2*Qq*Qh*((42*e**2+7)*Hh**2-13*e**2-1)+49*e**2*(5*self.orb_pert['e']**2+2)*Hh*Hq**3)
                    dpert_qq = -(He*(98*e**2*(2*e**2+1)*self.orb_pert['e']**2*Hh*Qq**2+98*e**2*(2*e**2+1)*self.orb_pert['e']**2*Hq*Qq*Qh+\
                               49*e**2*(2*e**2+1)*(5*self.orb_pert['e']**2+2)*Hq**2*Hh+14*(22*e**4+19*e**2+1)*self.orb_pert['e']**2*Hh*Qh**2+\
                               7*(8*e**4+12*e**2+1)*(5*self.orb_pert['e']**2+2)*Hh**3-Hh*(76*e**4+86*e**2+(346*e**4+309*e**2+17)*\
                               self.orb_pert['e']**2+6))-2*self.orb_pert['e']**2*Qe*Qh*(-20*e**4-2*e**2+7*(6*e**4-5*e**2-1)*Hh**2+1))
                    dpert_hh = -7*e**2*(2*self.orb_pert['e']**2*Qe*((7-7*e**2)*Hq*Hh*Qh+(e**2+2)*Qq)+He*(Hq*(e**2*(self.orb_pert['e']**2*(70*Hh**2+\
                               84*Qq**2+42*Qh**2-95)+28*Hh**2-22)+7*(5*self.orb_pert['e']**2+2)*Hh**2-self.orb_pert['e']**2-2)+\
                               14*(2*e**2+1)*self.orb_pert['e']**2*Hh*Qq*Qh+21*e**2*(5*self.orb_pert['e']**2+2)*Hq**3))
                    dpert1 += dpert_factor * (dpert_ee*e1 + dpert_qq*q1 + dpert_hh*h1)
                    dpert2 += dpert_factor * (dpert_ee*e2 + dpert_qq*q2 + dpert_hh*h2)
                    dpert3 += dpert_factor * (dpert_ee*e3 + dpert_qq*q3 + dpert_hh*h3)
                    dpert_h += dpert_factor*dpert_hh
                    
                    gpert_factor = 45*n*self.Mpert*(self.Ms**3 + Mp**3)*a**5*e*np.sqrt(1 - e**2) / \
                        (256*(self.Ms + Mp)**4*self.orb_pert['a']**5*(1 - self.orb_pert['e']**2)**(7/2))
                    gpert_ee = 7*(2*self.orb_pert['e']**2*Qe*((7-7*e**2)*Hq*Hh*Qh+(e**2+2)*Qq)+He*(Hq*(e**2*(self.orb_pert['e']**2*\
                               (70*Hh**2+84*Qq**2+42*Qh**2-95)+28*Hh**2-22)+7*(5*self.orb_pert['e']**2+2)*Hh**2-self.orb_pert['e']**2-2)+\
                               14*(2*e**2+1)*self.orb_pert['e']**2*Hh*Qq*Qh+21*e**2*(5*self.orb_pert['e']**2+2)*Hq**3))
                    gpert_qq = -(588*e**2*self.orb_pert['e']**2*Hq**2*Qq**2+196*(4*e**2+1)*self.orb_pert['e']**2*Hh*Hq*Qq*Qh+56*(4*e**2+3)*\
                               self.orb_pert['e']**2*Hh**2*Qh**2+147*e**2*(5*self.orb_pert['e']**2+2)*Hq**4+49*(4*e**2+1)*\
                               (5*self.orb_pert['e']**2+2)*Hh**2*Hq**2-7*Hq**2*(2*e**2*(53*self.orb_pert['e']**2+22)+self.orb_pert['e']**2+2)+\
                               14*(4*e**2+3)*(5*self.orb_pert['e']**2+2)*Hh**4-Hh**2*(4*e**2*(75*self.orb_pert['e']**2+38)+\
                               211*self.orb_pert['e']**2+86)+28*(e**2+1)*self.orb_pert['e']**2*Qq**2+4*(20*e**2+1)*self.orb_pert['e']**2*Qh**2+\
                               15*e**2*self.orb_pert['e']**2+46*e**2-self.orb_pert['e']**2+10)
                    gpert_hh = -(98*e**2*self.orb_pert['e']**2*Hq**2*Qq*Qh+Hh*Hq*(e**2*(self.orb_pert['e']**2*(210*Hh**2+98*Qq**2+84*Qh**2-221)+\
                               84*Hh**2-78)+self.orb_pert['e']**2*(35*Hh**2+14*Qh**2-17)+14*Hh**2-6)+2*self.orb_pert['e']**2*Qq*Qh*\
                               ((42*e**2+7)*Hh**2-13*e**2-1)+49*e**2*(5*self.orb_pert['e']**2+2)*Hh*Hq**3)
                    gpert1 += gpert_factor * (gpert_ee*e1 + gpert_qq*q1 + gpert_hh*h1)
                    gpert2 += gpert_factor * (gpert_ee*e2 + gpert_qq*q2 + gpert_hh*h2)
                    gpert3 += gpert_factor * (gpert_ee*e3 + gpert_qq*q3 + gpert_hh*h3)
                    gpert_e += gpert_factor*gpert_ee
                     
            if self.tides:
                
                # Projected spin vectors
                Ose = Os1*e1 + Os2*e2 + Os3*e3
                Osq = Os1*q1 + Os2*q2 + Os3*q3
                Osh = Os1*h1 + Os2*h2 + Os3*h3
                Ope = Op1*e1 + Op2*e2 + Op3*e3
                Opq = Op1*q1 + Op2*q2 + Op3*q3
                Oph = Op1*h1 + Op2*h2 + Op3*h3
                
                # Spin distorsion of the star
                dsds_factor = self.ks*(1 + Mp/self.Ms)*self.Rs**5 / (a**3*(1 - e**2)**(3/2))
                dsds_ee = -Osq*Osh
                dsds_qq = Ose*Osh
                dsds_hh = 0.
                dsds1 = dsds_factor * (dsds_ee*e1 + dsds_qq*q1 + dsds_hh*h1)
                dsds2 = dsds_factor * (dsds_ee*e2 + dsds_qq*q2 + dsds_hh*h2)
                dsds3 = dsds_factor * (dsds_ee*e3 + dsds_qq*q3 + dsds_hh*h3)
                dsds_h = 0.
                
                gsds_factor = n*self.ks*self.Rs**5*e / (self.Ms*a**2*(1 - e**2)**2)
                gsds_ee = 0.
                gsds_qq = (15/8)*((8 + 12*e**2 + e**4)/(1 - e**2)**3)*(Mp/a**3) - (1/2)*(Ose**2 + Osq**2 - 2*Osh**2)
                gsds_hh = Osq*Osh
                gsds1 = gsds_factor * (gsds_ee*e1 + gsds_qq*q1 + gsds_hh*h1)
                gsds2 = gsds_factor * (gsds_ee*e2 + gsds_qq*q2 + gsds_hh*h2)
                gsds3 = gsds_factor * (gsds_ee*e3 + gsds_qq*q3 + gsds_hh*h3)
                gsds_e = 0.

                   
                # Spin distorsion of the planet
                dsdp_factor = self.kp*(1 + self.Ms/Mp)*Rp**5 / (a**3*(1 - e**2)**(3/2))
                dsdp_ee = -Opq*Oph
                dsdp_qq = Ope*Oph
                dsdp_hh = 0.
                dsdp1 = dsdp_factor * (dsdp_ee*e1 + dsdp_qq*q1 + dsdp_hh*h1)
                dsdp2 = dsdp_factor * (dsdp_ee*e2 + dsdp_qq*q2 + dsdp_hh*h2)
                dsdp3 = dsdp_factor * (dsdp_ee*e3 + dsdp_qq*q3 + dsdp_hh*h3)
                dsdp_h = 0.
                
                gsdp_factor = n*self.kp*Rp**5*e / (Mp*a**2*(1 - e**2)**2)
                gsdp_ee = 0.
                gsdp_qq = (15/8)*((8 + 12*e**2 + e**4)/(1 - e**2)**3)*(self.Ms/a**3) - (1/2)*(Ope**2 + Opq**2 - 2*Oph**2)
                gsdp_hh = Opq*Oph
                gsdp1 = gsdp_factor * (gsdp_ee*e1 + gsdp_qq*q1 + gsdp_hh*h1)
                gsdp2 = gsdp_factor * (gsdp_ee*e2 + gsdp_qq*q2 + gsdp_hh*h2)
                gsdp3 = gsdp_factor * (gsdp_ee*e3 + gsdp_qq*q3 + gsdp_hh*h3)
                gsdp_e = 0.

                   
                # Tidal damping of the star
                dtds_factor = 3*n*self.ks*Mp*self.Rs**5 / (8*self.Qs*self.Ms*a**3*(1 - e**2)**(9/2))
                dtds_ee = (8 + 12*e**2 + e**4)*Ose
                dtds_qq = (8 + 36*e**2 + 5*e**4)*Osq
                dtds_hh = -(((16 + 5*e**2*(24 + 18*e**2 + e**4))/(1 - e**2)**(3/2))*n - 2*(8 + 3*e**2*(8 + e**2))*Osh)
                dtds1 = dtds_factor * (dtds_ee*e1 + dtds_qq*q1 + dtds_hh*h1)
                dtds2 = dtds_factor * (dtds_ee*e2 + dtds_qq*q2 + dtds_hh*h2)
                dtds3 = dtds_factor * (dtds_ee*e3 + dtds_qq*q3 + dtds_hh*h3)
                dtds_h = dtds_factor*dtds_hh
                
                gtds_factor = 3*n**2*self.ks*e*self.Rs**5 / (8*self.Qs*self.Ms*(1 + self.Ms/Mp)*a**2*(1 - e**2)**5)
                gtds_ee = (-9/4)*((64 + 5*e**2*(48 + 24*e**2 + e**4))/(1 - e**2)**(3/2))*n + 11*(8 + 12*e**2 + e**4)*Osh
                gtds_qq = 0.
                gtds_hh = -(8 + 12*e**2 + e**4)*Ose
                gtds1 = gtds_factor * (gtds_ee*e1 + gtds_qq*q1 + gtds_hh*h1)
                gtds2 = gtds_factor * (gtds_ee*e2 + gtds_qq*q2 + gtds_hh*h2)
                gtds3 = gtds_factor * (gtds_ee*e3 + gtds_qq*q3 + gtds_hh*h3)
                gtds_e = gtds_factor*gtds_ee
                   
                
                # Tidal damping of the planet
                dtdp_factor = 3*n*self.kp*self.Ms*Rp**5 / (8*self.Qp*Mp*a**3*(1 - e**2)**(9/2))
                dtdp_ee = (8 + 12*e**2 + e**4)*Ope
                dtdp_qq = (8 + 36*e**2 + 5*e**4)*Opq
                dtdp_hh = -(((16 + 5*e**2*(24 + 18*e**2 + e**4))/(1 - e**2)**(3/2))*n - 2*(8 + 3*e**2*(8 + e**2))*Oph)
                dtdp1 = dtdp_factor * (dtdp_ee*e1 + dtdp_qq*q1 + dtdp_hh*h1)
                dtdp2 = dtdp_factor * (dtdp_ee*e2 + dtdp_qq*q2 + dtdp_hh*h2)
                dtdp3 = dtdp_factor * (dtdp_ee*e3 + dtdp_qq*q3 + dtdp_hh*h3)
                dtdp_h = dtdp_factor*dtdp_hh
                
                gtdp_factor = 3*n**2*self.kp*e*Rp**5 / (8*self.Qp*Mp*(1 + Mp/self.Ms)*a**2*(1 - e**2)**5)
                gtdp_ee = (-9/4)*((64 + 5*e**2*(48 + 24*e**2 + e**4))/(1 - e**2)**(3/2))*n + 11*(8 + 12*e**2 + e**4)*Oph
                gtdp_qq = 0.
                gtdp_hh = -(8 + 12*e**2 + e**4)*Ope
                gtdp1 = gtdp_factor * (gtdp_ee*e1 + gtdp_qq*q1 + gtdp_hh*h1)
                gtdp2 = gtdp_factor * (gtdp_ee*e2 + gtdp_qq*q2 + gtdp_hh*h2)
                gtdp3 = gtdp_factor * (gtdp_ee*e3 + gtdp_qq*q3 + gtdp_hh*h3)
                gtdp_e = gtdp_factor*gtdp_ee
                
            else:
                dsds1, dsds2, dsds3, gsds1, gsds2, gsds3, dsdp1, dsdp2, dsdp3, gsdp1, gsdp2, gsdp3, \
                dtds1, dtds2, dtds3, gtds1, gtds2, gtds3, dtdp1, dtdp2, dtdp3, gtdp1, gtdp2, gtdp3, \
                dsds_h, gsds_e, dsdp_h, gsdp_e, dtds_h, gtds_e, dtdp_h, gtdp_e = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                                                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                   
            if self.relat:
                grel_factor = 3*a**2*n**3*e / (c**2*(1 - e**2))
                grel1 = grel_factor*q1
                grel2 = grel_factor*q2
                grel3 = grel_factor*q3
                grel_e = 0.
            else:
                grel1, grel2, grel3, grel_e = 0, 0, 0, 0
                
            # Derivative of non-unitary vectors
            hdot1 = dpert1 + dsds1 + dsdp1 + dtds1 + dtdp1
            hdot2 = dpert2 + dsds2 + dsdp2 + dtds2 + dtdp2
            hdot3 = dpert3 + dsds3 + dsdp3 + dtds3 + dtdp3
            
            edot1 = gpert1 + gsds1 + gsdp1 + gtds1 + gtdp1 + grel1
            edot2 = gpert2 + gsds2 + gsdp2 + gtds2 + gtdp2 + grel2
            edot3 = gpert3 + gsds3 + gsdp3 + gtds3 + gtdp3 + grel3
            
            # Derivative of the evolving vector
            edot = gpert_e + gsds_e + gsdp_e + gtds_e + gtdp_e + grel_e
            
            e1dot1 = (edot1 - edot*e1)/e
            e1dot2 = (edot2 - edot*e2)/e
            e1dot3 = (edot3 - edot*e3)/e
            
            hdot = dpert_h + dsds_h + dsdp_h + dtds_h + dtdp_h
            
            h1dot1 = (hdot1 - hdot*h1)/h
            h1dot2 = (hdot2 - hdot*h2)/h
            h1dot3 = (hdot3 - hdot*h3)/h
            
            adot = 2*a*((hdot/h) + e*edot/(1 - e**2))
            
            if Is != 0:
                Osdot1 = (-self.Ms*Mp / (Is*(self.Ms + Mp))) * (dsds1 + dtds1)
                Osdot2 = (-self.Ms*Mp / (Is*(self.Ms + Mp))) * (dsds2 + dtds2)
                Osdot3 = (-self.Ms*Mp / (Is*(self.Ms + Mp))) * (dsds3 + dtds3)
            else:
                Osdot1, Osdot2, Osdot3 = 0, 0, 0
            if Ip != 0:
                Opdot1 = (-self.Ms*Mp / (Ip*(self.Ms + Mp))) * (dsdp1 + dtdp1)
                Opdot2 = (-self.Ms*Mp / (Ip*(self.Ms + Mp))) * (dsdp2 + dtdp2)
                Opdot3 = (-self.Ms*Mp / (Ip*(self.Ms + Mp))) * (dsdp3 + dtdp3)
            else:
                Opdot1, Opdot2, Opdot3 = 0, 0, 0
                
        else:
            h1dot1, h1dot2, h1dot3, e1dot1, e1dot2, e1dot3, Osdot1, Osdot2, Osdot3, Opdot1, Opdot2, Opdot3, \
            adot, edot = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            
        if self.atmo and self.evap and Mp > self.Mcore:
            eps = self.eps_evap(Mp, Rp)
            LXUV = self.L_XUV(t, a)
            RXUV = self.RXUV(t, a, Mp, Rp)
            ksi = (a/Rp)*(1 + .5*e**2)*(Mp/(3*self.Ms))**(1/3)
            Ktide = 1 - 3/(2*ksi) + 1/(2*ksi**3)
            Mpdot = - eps * LXUV*Rp*RXUV**2/(4*Mp*Ktide*np.sqrt(1 - e**2)*a**2)
        else:
            Mpdot = 0.
            
        # Concatenating the results
        Ydot = np.array([h1dot1, h1dot2, h1dot3, e1dot1, e1dot2, e1dot3, Osdot1, Osdot2, Osdot3, Opdot1, Opdot2, Opdot3, edot, adot, Mpdot])
        
        return Ydot


#----------------------------------------------------------------------------------------------------------------------------------------
# Returns the time derivative of the evolving vector as a generator
        
    def evolving_function_generator(self):
        
        # Storing the evolving quantities
        h1, h2, h3, e1, e2, e3, Os1, Os2, Os3, Op1, Op2, Op3, e, a, Mp = y_jit(0), y_jit(1), y_jit(2), y_jit(3), y_jit(4), \
            y_jit(5), y_jit(6), y_jit(7), y_jit(8), y_jit(9), y_jit(10), y_jit(11), y_jit(12), y_jit(13), y_jit(14)
        
        # Orbital properties
        n = symengine.sqrt((self.Ms + Mp)/a**3)
        h = symengine.sqrt((self.Ms + Mp)*a*(1 - e**2))
        
        # Evolution of the radius
        Rp = self.Rp_sym
            
        # Moments of inertia
        Is = self.alphas*self.Ms*self.Rs**2
        Ip = self.alphap*Mp*Rp**2
            
        # Unitary vectors
        e_unit = symengine.sqrt(e1**2 + e2**2 + e3**2)
        h_unit = symengine.sqrt(h1**2 + h2**2 + h3**2)
        e1 /= e_unit
        e2 /= e_unit
        e3 /= e_unit
        h1 /= h_unit
        h2 /= h_unit
        h3 /= h_unit
        q1 = h2*e3 - h3*e2
        q2 = h3*e1 - h1*e3
        q3 = h1*e2 - h2*e1  

        # Cancelling ei and qi if the orbit becomes circular
        #SMALLE = 1e-3  
        #e1 = conditional(e, SMALLE, 0., e1) 
        #e2 = conditional(e, SMALLE, 0., e2)
        #e3 = conditional(e, SMALLE, 0., e3) 
        #q1 = conditional(e, SMALLE, 0., q1) 
        #q2 = conditional(e, SMALLE, 0., q2)
        #q3 = conditional(e, SMALLE, 0., q3)        
        
        if self.dyn:
            
            # Perturber's projected orbital vectors
            He = self.H1[0]*e1 + self.H1[1]*e2 + self.H1[2]*e3
            Hq = self.H1[0]*q1 + self.H1[1]*q2 + self.H1[2]*q3
            Hh = self.H1[0]*h1 + self.H1[1]*h2 + self.H1[2]*h3
            
            # Quadrupolar disturbing force
            dpert_factor = 3*self.Mpert*a**2 / (4*self.orb_pert['a']**3*(1 - self.orb_pert['e']**2)**(3./2))
            dpert_ee = -(1 - e**2)*Hq*Hh
            dpert_qq = (1 + 4*e**2)*He*Hh
            dpert_hh = - 5*e**2*He*Hq
            dpert1 = dpert_factor * (dpert_ee*e1 + dpert_qq*q1 + dpert_hh*h1)
            dpert2 = dpert_factor * (dpert_ee*e2 + dpert_qq*q2 + dpert_hh*h2)
            dpert3 = dpert_factor * (dpert_ee*e3 + dpert_qq*q3 + dpert_hh*h3)
            dpert_h = dpert_factor*dpert_hh
            
            gpert_factor= 3*n*self.Mpert*e*symengine.sqrt(1 - e**2)*a**3 / \
                          (4*(self.Ms + Mp)*self.orb_pert['a']**3*(1 - self.orb_pert['e']**2)**(3/2))
            gpert_ee = 5*He*Hq
            gpert_qq = -3 + 5*Hq**2 + 4*Hh**2
            gpert_hh = Hq*Hh
            gpert1 = gpert_factor * (gpert_ee*e1 + gpert_qq*q1 + gpert_hh*h1)
            gpert2 = gpert_factor * (gpert_ee*e2 + gpert_qq*q2 + gpert_hh*h2)
            gpert3 = gpert_factor * (gpert_ee*e3 + gpert_qq*q3 + gpert_hh*h3)
            gpert_e = gpert_factor*gpert_ee
                     
            # Octupolar disturbing force
            if self.orderdyn > 2:
                Qe = self.Q1[0]*e1 + self.Q1[1]*e2 + self.Q1[2]*e3
                Qq = self.Q1[0]*q1 + self.Q1[1]*q2 + self.Q1[2]*q3
                Qh = self.Q1[0]*h1 + self.Q1[1]*h2 + self.Q1[2]*h3
                Ee = self.E1[0]*e1 + self.E1[1]*e2 + self.E1[2]*e3
                Eq = self.E1[0]*q1 + self.E1[1]*q2 + self.E1[2]*q3
                Eh = self.E1[0]*h1 + self.E1[1]*h2 + self.E1[2]*h3
                
                dpert_factor = 15*self.Mpert*(self.Ms - Mp)*a**3*e*self.orb_pert['e'] / (64*(self.Ms + Mp)*self.orb_pert['a']**4* \
                                                                                    (1 - self.orb_pert['e']**2)**(5/2))
                dpert_ee = 10*(1 - e**2)*(Qe*(Eh*Qq + Eq*Qh) + Ee*(3*Eq*Eh + Qq*Qh))
                dpert_qq = -(Eh*(-1 + 8*e**2 + (45 + 60*e**2)*Hh**2 - 5*(2 + 5*e**2)*Qq**2) + 5*(2 + 5*e**2)*Eq*(3*Hq*Hh + Qq*Qh))
                dpert_hh = -10*(1 + 6*e**2)*Eh*Qq*Qh + Eq*(-11 + 15*Hh**2 + 10*Qh**2 + 3*e**2*(-29 + 35*Hq**2 + 30*Hh**2 + 20*Qh**2))
                dpert1 += dpert_factor * (dpert_ee*e1 + dpert_qq*q1 + dpert_hh*h1)
                dpert2 += dpert_factor * (dpert_ee*e2 + dpert_qq*q2 + dpert_hh*h2)
                dpert3 += dpert_factor * (dpert_ee*e3 + dpert_qq*q3 + dpert_hh*h3)
                dpert_h += dpert_factor*dpert_hh
                
                gpert_factor = -(15*n*self.Mpert*(self.Ms - Mp)*a**4*symengine.sqrt(1 - e**2)*self.orb_pert['e'] / \
                          (64*(self.Ms + Mp)**2*self.orb_pert['a']**4*(1 - self.orb_pert['e']**2)**(5/2)))
                gpert_ee = -10*(1 + 6*e**2)*Eh*Qq*Qh + Eq*(-11 + 15*Hh**2 + 10*Qh**2 + 3*e**2*(-29 + 35*Hq**2 + 30*Hh**2 + 20*Qh**2))
                gpert_qq = -((-10 + 30*e**2)*Eh*Qe*Qh + Ee*(-11 + 15*Hh**2 + 10*Qh**2 + 3*e**2*(-17 + 35*Hq**2 + 20*Hh**2 - 10*Qh**2)))
                gpert_hh = -10*e**2*(Qe*(Eh*Qq + Eq*Qh) + Ee*(3*Eq*Eh + Qq*Qh))                            
                gpert1 += gpert_factor * (gpert_ee*e1 + gpert_qq*q1 + gpert_hh*h1)
                gpert2 += gpert_factor * (gpert_ee*e2 + gpert_qq*q2 + gpert_hh*h2)
                gpert3 += gpert_factor * (gpert_ee*e3 + gpert_qq*q3 + gpert_hh*h3)
                gpert_e += gpert_factor*gpert_ee
                 
                # Hexadecapolar disturbing force
                if self.orderdyn > 3:
                    
                    dpert_factor = 45*self.Mpert*(self.Ms**3 + Mp**3)*a**4 / (256*(self.Ms + Mp)**3*self.orb_pert['a']**5* \
                                                                         (1 - self.orb_pert['e']**2)**(7/2))
                    dpert_ee = (1-e**2)*(98*e**2*self.orb_pert['e']**2*Hq**2*Qq*Qh+Hh*Hq*(e**2*(self.orb_pert['e']**2*(210*Hh**2+\
                               98*Qq**2+84*Qh**2-221)+84*Hh**2-78)+self.orb_pert['e']**2*(35*Hh**2+14*Qh**2-17)+14*Hh**2-6)+\
                               2*self.orb_pert['e']**2*Qq*Qh*((42*e**2+7)*Hh**2-13*e**2-1)+49*e**2*(5*self.orb_pert['e']**2+2)*Hh*Hq**3)
                    dpert_qq = -(He*(98*e**2*(2*e**2+1)*self.orb_pert['e']**2*Hh*Qq**2+98*e**2*(2*e**2+1)*self.orb_pert['e']**2*Hq*Qq*Qh+\
                               49*e**2*(2*e**2+1)*(5*self.orb_pert['e']**2+2)*Hq**2*Hh+14*(22*e**4+19*e**2+1)*self.orb_pert['e']**2*Hh*Qh**2+\
                               7*(8*e**4+12*e**2+1)*(5*self.orb_pert['e']**2+2)*Hh**3-Hh*(76*e**4+86*e**2+(346*e**4+309*e**2+17)*\
                               self.orb_pert['e']**2+6))-2*self.orb_pert['e']**2*Qe*Qh*(-20*e**4-2*e**2+7*(6*e**4-5*e**2-1)*Hh**2+1))
                    dpert_hh = -7*e**2*(2*self.orb_pert['e']**2*Qe*((7-7*e**2)*Hq*Hh*Qh+(e**2+2)*Qq)+He*(Hq*(e**2*(self.orb_pert['e']**2*(70*Hh**2+\
                               84*Qq**2+42*Qh**2-95)+28*Hh**2-22)+7*(5*self.orb_pert['e']**2+2)*Hh**2-self.orb_pert['e']**2-2)+\
                               14*(2*e**2+1)*self.orb_pert['e']**2*Hh*Qq*Qh+21*e**2*(5*self.orb_pert['e']**2+2)*Hq**3))
                    dpert1 += dpert_factor * (dpert_ee*e1 + dpert_qq*q1 + dpert_hh*h1)
                    dpert2 += dpert_factor * (dpert_ee*e2 + dpert_qq*q2 + dpert_hh*h2)
                    dpert3 += dpert_factor * (dpert_ee*e3 + dpert_qq*q3 + dpert_hh*h3)
                    dpert_h += dpert_factor*dpert_hh
                    
                    gpert_factor = 45*n*self.Mpert*(self.Ms**3 + Mp**3)*a**5*e*symengine.sqrt(1 - e**2) / \
                        (256*(self.Ms + Mp)**4*self.orb_pert['a']**5*(1 - self.orb_pert['e']**2)**(7/2))
                    gpert_ee = 7*(2*self.orb_pert['e']**2*Qe*((7-7*e**2)*Hq*Hh*Qh+(e**2+2)*Qq)+He*(Hq*(e**2*(self.orb_pert['e']**2*\
                               (70*Hh**2+84*Qq**2+42*Qh**2-95)+28*Hh**2-22)+7*(5*self.orb_pert['e']**2+2)*Hh**2-self.orb_pert['e']**2-2)+\
                               14*(2*e**2+1)*self.orb_pert['e']**2*Hh*Qq*Qh+21*e**2*(5*self.orb_pert['e']**2+2)*Hq**3))
                    gpert_qq = -(588*e**2*self.orb_pert['e']**2*Hq**2*Qq**2+196*(4*e**2+1)*self.orb_pert['e']**2*Hh*Hq*Qq*Qh+56*(4*e**2+3)*\
                               self.orb_pert['e']**2*Hh**2*Qh**2+147*e**2*(5*self.orb_pert['e']**2+2)*Hq**4+49*(4*e**2+1)*\
                               (5*self.orb_pert['e']**2+2)*Hh**2*Hq**2-7*Hq**2*(2*e**2*(53*self.orb_pert['e']**2+22)+self.orb_pert['e']**2+2)+\
                               14*(4*e**2+3)*(5*self.orb_pert['e']**2+2)*Hh**4-Hh**2*(4*e**2*(75*self.orb_pert['e']**2+38)+\
                               211*self.orb_pert['e']**2+86)+28*(e**2+1)*self.orb_pert['e']**2*Qq**2+4*(20*e**2+1)*self.orb_pert['e']**2*Qh**2+\
                               15*e**2*self.orb_pert['e']**2+46*e**2-self.orb_pert['e']**2+10)
                    gpert_hh = -(98*e**2*self.orb_pert['e']**2*Hq**2*Qq*Qh+Hh*Hq*(e**2*(self.orb_pert['e']**2*(210*Hh**2+98*Qq**2+84*Qh**2-221)+\
                               84*Hh**2-78)+self.orb_pert['e']**2*(35*Hh**2+14*Qh**2-17)+14*Hh**2-6)+2*self.orb_pert['e']**2*Qq*Qh*\
                               ((42*e**2+7)*Hh**2-13*e**2-1)+49*e**2*(5*self.orb_pert['e']**2+2)*Hh*Hq**3)
                    gpert1 += gpert_factor * (gpert_ee*e1 + gpert_qq*q1 + gpert_hh*h1)
                    gpert2 += gpert_factor * (gpert_ee*e2 + gpert_qq*q2 + gpert_hh*h2)
                    gpert3 += gpert_factor * (gpert_ee*e3 + gpert_qq*q3 + gpert_hh*h3)
                    gpert_e += gpert_factor*gpert_ee
                     
            if self.tides:
                
                # Projected spin vectors
                Ose = Os1*e1 + Os2*e2 + Os3*e3
                Osq = Os1*q1 + Os2*q2 + Os3*q3
                Osh = Os1*h1 + Os2*h2 + Os3*h3
                Ope = Op1*e1 + Op2*e2 + Op3*e3
                Opq = Op1*q1 + Op2*q2 + Op3*q3
                Oph = Op1*h1 + Op2*h2 + Op3*h3
                
                # Spin distorsion of the star
                dsds_factor = self.ks*(1 + Mp/self.Ms)*self.Rs**5 / (a**3*(1 - e**2)**(3/2))
                dsds_ee = -Osq*Osh
                dsds_qq = Ose*Osh
                dsds_hh = 0.
                dsds1 = dsds_factor * (dsds_ee*e1 + dsds_qq*q1 + dsds_hh*h1)
                dsds2 = dsds_factor * (dsds_ee*e2 + dsds_qq*q2 + dsds_hh*h2)
                dsds3 = dsds_factor * (dsds_ee*e3 + dsds_qq*q3 + dsds_hh*h3)
                dsds_h = 0.
                
                gsds_factor = n*self.ks*self.Rs**5*e / (self.Ms*a**2*(1 - e**2)**2)
                gsds_ee = 0.
                gsds_qq = (15/8)*((8 + 12*e**2 + e**4)/(1 - e**2)**3)*(Mp/a**3) - (1/2)*(Ose**2 + Osq**2 - 2*Osh**2)
                gsds_hh = Osq*Osh
                gsds1 = gsds_factor * (gsds_ee*e1 + gsds_qq*q1 + gsds_hh*h1)
                gsds2 = gsds_factor * (gsds_ee*e2 + gsds_qq*q2 + gsds_hh*h2)
                gsds3 = gsds_factor * (gsds_ee*e3 + gsds_qq*q3 + gsds_hh*h3)
                gsds_e = 0.

                   
                # Spin distorsion of the planet
                dsdp_factor = self.kp*(1 + self.Ms/Mp)*Rp**5 / (a**3*(1 - e**2)**(3/2))
                dsdp_ee = -Opq*Oph
                dsdp_qq = Ope*Oph
                dsdp_hh = 0.
                dsdp1 = dsdp_factor * (dsdp_ee*e1 + dsdp_qq*q1 + dsdp_hh*h1)
                dsdp2 = dsdp_factor * (dsdp_ee*e2 + dsdp_qq*q2 + dsdp_hh*h2)
                dsdp3 = dsdp_factor * (dsdp_ee*e3 + dsdp_qq*q3 + dsdp_hh*h3)
                dsdp_h = 0.
                
                gsdp_factor = n*self.kp*Rp**5*e / (Mp*a**2*(1 - e**2)**2)
                gsdp_ee = 0.
                gsdp_qq = (15/8)*((8 + 12*e**2 + e**4)/(1 - e**2)**3)*(self.Ms/a**3) - (1/2)*(Ope**2 + Opq**2 - 2*Oph**2)
                gsdp_hh = Opq*Oph
                gsdp1 = gsdp_factor * (gsdp_ee*e1 + gsdp_qq*q1 + gsdp_hh*h1)
                gsdp2 = gsdp_factor * (gsdp_ee*e2 + gsdp_qq*q2 + gsdp_hh*h2)
                gsdp3 = gsdp_factor * (gsdp_ee*e3 + gsdp_qq*q3 + gsdp_hh*h3)
                gsdp_e = 0.

                   
                # Tidal damping of the star
                dtds_factor = 3*n*self.ks*Mp*self.Rs**5 / (8*self.Qs*self.Ms*a**3*(1 - e**2)**(9/2))
                dtds_ee = (8 + 12*e**2 + e**4)*Ose
                dtds_qq = (8 + 36*e**2 + 5*e**4)*Osq
                dtds_hh = -(((16 + 5*e**2*(24 + 18*e**2 + e**4))/(1 - e**2)**(3/2))*n - 2*(8 + 3*e**2*(8 + e**2))*Osh)
                dtds1 = dtds_factor * (dtds_ee*e1 + dtds_qq*q1 + dtds_hh*h1)
                dtds2 = dtds_factor * (dtds_ee*e2 + dtds_qq*q2 + dtds_hh*h2)
                dtds3 = dtds_factor * (dtds_ee*e3 + dtds_qq*q3 + dtds_hh*h3)
                dtds_h = dtds_factor*dtds_hh
                
                gtds_factor = 3*n**2*self.ks*e*self.Rs**5 / (8*self.Qs*self.Ms*(1 + self.Ms/Mp)*a**2*(1 - e**2)**5)
                gtds_ee = (-9/4)*((64 + 5*e**2*(48 + 24*e**2 + e**4))/(1 - e**2)**(3/2))*n + 11*(8 + 12*e**2 + e**4)*Osh
                gtds_qq = 0.
                gtds_hh = -(8 + 12*e**2 + e**4)*Ose
                gtds1 = gtds_factor * (gtds_ee*e1 + gtds_qq*q1 + gtds_hh*h1)
                gtds2 = gtds_factor * (gtds_ee*e2 + gtds_qq*q2 + gtds_hh*h2)
                gtds3 = gtds_factor * (gtds_ee*e3 + gtds_qq*q3 + gtds_hh*h3)
                gtds_e = gtds_factor*gtds_ee
                   
                
                # Tidal damping of the planet
                dtdp_factor = 3*n*self.kp*self.Ms*Rp**5 / (8*self.Qp*Mp*a**3*(1 - e**2)**(9/2))
                dtdp_ee = (8 + 12*e**2 + e**4)*Ope
                dtdp_qq = (8 + 36*e**2 + 5*e**4)*Opq
                dtdp_hh = -(((16 + 5*e**2*(24 + 18*e**2 + e**4))/(1 - e**2)**(3/2))*n - 2*(8 + 3*e**2*(8 + e**2))*Oph)
                dtdp1 = dtdp_factor * (dtdp_ee*e1 + dtdp_qq*q1 + dtdp_hh*h1)
                dtdp2 = dtdp_factor * (dtdp_ee*e2 + dtdp_qq*q2 + dtdp_hh*h2)
                dtdp3 = dtdp_factor * (dtdp_ee*e3 + dtdp_qq*q3 + dtdp_hh*h3)
                dtdp_h = dtdp_factor*dtdp_hh
                
                gtdp_factor = 3*n**2*self.kp*e*Rp**5 / (8*self.Qp*Mp*(1 + Mp/self.Ms)*a**2*(1 - e**2)**5)
                gtdp_ee = (-9/4)*((64 + 5*e**2*(48 + 24*e**2 + e**4))/(1 - e**2)**(3/2))*n + 11*(8 + 12*e**2 + e**4)*Oph
                gtdp_qq = 0.
                gtdp_hh = -(8 + 12*e**2 + e**4)*Ope
                gtdp1 = gtdp_factor * (gtdp_ee*e1 + gtdp_qq*q1 + gtdp_hh*h1)
                gtdp2 = gtdp_factor * (gtdp_ee*e2 + gtdp_qq*q2 + gtdp_hh*h2)
                gtdp3 = gtdp_factor * (gtdp_ee*e3 + gtdp_qq*q3 + gtdp_hh*h3)
                gtdp_e = gtdp_factor*gtdp_ee
                
            else:
                dsds1, dsds2, dsds3, gsds1, gsds2, gsds3, dsdp1, dsdp2, dsdp3, gsdp1, gsdp2, gsdp3, \
                dtds1, dtds2, dtds3, gtds1, gtds2, gtds3, dtdp1, dtdp2, dtdp3, gtdp1, gtdp2, gtdp3, \
                dsds_h, gsds_e, dsdp_h, gsdp_e, dtds_h, gtds_e, dtdp_h, gtdp_e = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                                                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                   
            if self.relat:
                grel_factor = 3*a**2*n**3*e / (c**2*(1 - e**2))
                grel1 = grel_factor*q1
                grel2 = grel_factor*q2
                grel3 = grel_factor*q3
                grel_e = 0.
            else:
                grel1, grel2, grel3, grel_e = 0, 0, 0, 0
                
            # Derivative of non-unitary vectors
            hdot1 = dpert1 + dsds1 + dsdp1 + dtds1 + dtdp1
            hdot2 = dpert2 + dsds2 + dsdp2 + dtds2 + dtdp2
            hdot3 = dpert3 + dsds3 + dsdp3 + dtds3 + dtdp3
            
            edot1 = gpert1 + gsds1 + gsdp1 + gtds1 + gtdp1 + grel1
            edot2 = gpert2 + gsds2 + gsdp2 + gtds2 + gtdp2 + grel2
            edot3 = gpert3 + gsds3 + gsdp3 + gtds3 + gtdp3 + grel3
            
            # Derivative of the evolving vector
            edot = gpert_e + gsds_e + gsdp_e + gtds_e + gtdp_e + grel_e
            
            e1dot1 = (edot1 - edot*e1)/e
            e1dot2 = (edot2 - edot*e2)/e
            e1dot3 = (edot3 - edot*e3)/e
            
            hdot = dpert_h + dsds_h + dsdp_h + dtds_h + dtdp_h
            
            h1dot1 = (hdot1 - hdot*h1)/h
            h1dot2 = (hdot2 - hdot*h2)/h
            h1dot3 = (hdot3 - hdot*h3)/h
            
            adot = 2*a*((hdot/h) + e*edot/(1 - e**2))
            
            if Is != 0:
                Osdot1 = (-self.Ms*Mp / (Is*(self.Ms + Mp))) * (dsds1 + dtds1)
                Osdot2 = (-self.Ms*Mp / (Is*(self.Ms + Mp))) * (dsds2 + dtds2)
                Osdot3 = (-self.Ms*Mp / (Is*(self.Ms + Mp))) * (dsds3 + dtds3)
            else:
                Osdot1, Osdot2, Osdot3 = 0, 0, 0
            if Ip != 0:
                Opdot1 = (-self.Ms*Mp / (Ip*(self.Ms + Mp))) * (dsdp1 + dtdp1)
                Opdot2 = (-self.Ms*Mp / (Ip*(self.Ms + Mp))) * (dsdp2 + dtdp2)
                Opdot3 = (-self.Ms*Mp / (Ip*(self.Ms + Mp))) * (dsdp3 + dtdp3)
            else:
                Opdot1, Opdot2, Opdot3 = 0, 0, 0
                
        else:
            h1dot1, h1dot2, h1dot3, e1dot1, e1dot2, e1dot3, Osdot1, Osdot2, Osdot3, Opdot1, Opdot2, Opdot3, \
            adot, edot = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            
        if self.atmo and self.evap:
            ksi = (a/Rp)*(1 + .5*e**2)*(Mp/(3*self.Ms))**(1/3)
            Ktide = 1 - 3/(2*ksi) + 1/(2*ksi**3)
            Mpdot = - self.eps_sym * self.LXUV_sym*Rp*self.RXUV_sym**2/(4*Mp*Ktide*symengine.sqrt(1 - e**2)*a**2)
            Mpdot = conditional(Mp, self.Mcore, 0., Mpdot)
            Mpdot = conditional(Mpdot, 0., Mpdot, 0.)
        else:
            Mpdot = 0.
            
        # Yielding the results
        yield h1dot1
        yield h1dot2
        yield h1dot3
        yield e1dot1
        yield e1dot2
        yield e1dot3
        yield Osdot1
        yield Osdot2
        yield Osdot3
        yield Opdot1
        yield Opdot2
        yield Opdot3
        yield edot
        yield adot
        yield Mpdot

    
#----------------------------------------------------------------------------------------------------------------------------------------
# Runs a simulation from the last state to a certain time tf
# Input: 'tf' (scalar) end-time of the simulation in years
        
    def evolve(self, tf, integrator, verbose=False):
        
        if verbose: print('  + Simulation of the system over ' + format(self.age, '-.3') + ' years...')
        
        # Initial conditions
        t0 = self.t0
        if t0 > tf: raise ValueError('Invalid time.')
        Y0 = np.array([self.h1[-1], self.h2[-1], self.h3[-1], self.e1[-1], self.e2[-1], self.e3[-1], \
                       self.Os1[-1], self.Os2[-1], self.Os3[-1], self.Op1[-1], self.Op2[-1], self.Op3[-1], \
                       self.e[-1], self.a[-1], self.Mp[-1]])
        t1 = t0
        Y = Y0
            
        # Simulating the system's evolution
        atol = 6*[1e-6] + 6*[1.] + [1e-6, 1e-6, 1e-6*self.Mcore]
        #atol = 6*[1e-2] + 6*[1.] + [1e-2, 1e-4, 1e-2*self.Mcore]
        rtol = len(Y0)*[1e-7]
        #rtol = 6*[1e-4] + 6*[1e-3] + [1e-4, 1e-4, 1e-4]
        #rtol = 1e-6

        if verbose: start_time = tm.time()
        if integrator == 'IRK':
            method = 'Radau'
        elif integrator == 'LSODA':
            method = 'LSODA'
        elif integrator == 'ERK':
            method = 'DOP853'
        elif integrator == 'odeint':
            pass
        elif integrator == 'jit':
            ODE = jitcode(self.evolving_function_generator, \
                          control_pars=[self.Rp_sym, self.LXUV_sym, self.eps_sym, self.RXUV_sym], verbose=False)
            Rp = self.Rp[-1]
            if self.atmo:
                eps_temp = self.eps_evap(self.Mp[-1], self.Rp[-1])
                LXUV_temp = self.L_XUV(t1, self.a[-1])
                RXUV_temp = self.RXUV(t1, self.a[-1], self.Mp[-1], self.Rp[-1])
            else:
                eps_temp, LXUV_temp, RXUV_temp = 0, 0, 0
            #ODE.set_integrator('DOP853', atol=atol, rtol=rtol)
            ODE.set_integrator('dopri5', atol=atol, rtol=rtol)
            ODE.set_parameters(Rp, LXUV_temp, eps_temp, RXUV_temp)
            ODE.set_initial_value(Y, t1)
        else: 
            raise ValueError('Invalid integrator.')
        while t1 < tf and self.simul:
            #t2 = t1 + (tf - t0)/100

            t2 = np.min((t1 + self.tau_dyn, t1 + self.tau_atmo, tf))

            if integrator not in ['odeint', 'jit']:
                solution = solve_ivp(lambda t, y: self.evolving_function(t, y, verbose=verbose), (t1, t2), Y, atol=atol, rtol=rtol, method=method)
                solution_t = solution.t
                solution_y = solution.y
            elif integrator == 'odeint':
                solution_t = np.linspace(t1, t2, num=1000)
                solution = odeint(lambda y, t: self.evolving_function(t, y, verbose=verbose), Y, solution_t)
                solution_y = solution.transpose()
            else:
                solution_t = np.linspace(t1, t2, num=100)[1:]
                solution = []
                for t in solution_t:
                    try:
                        solution.append(ODE.integrate(t))
                    except UnsuccessfulIntegration:
                        print('>>> t =', t)
                        print('>>> Y =', ODE.y_dict)
                        print('>>> Y_prime =', self.evolving_function(t, Y))
                        raise SystemExit('Dynamical integration failed. Step size probably becomes too small.')
                    
                    # Evolution of the radius
                    #print(t*1e-9, self.t_atmo[-1]*1e-9, self.tau_atmo*1e-9)
                    if t > self.t_atmo[-1] + self.tau_atmo and self.init_Rp is None: #and solution[-1][14] > self.Mcore:
                        self.Lp = self.planet_luminosity(t)
                        Y = solution[-1]
                        e, a, Mp = Y[12], Y[13], Y[14]
                        Rp = self.retrieve_radius(t, Mp, a, e, Rp_guess=self.Rp[-1], verbose=False)
                        self.t_atmo.append(t)
                        self.Rp.append(Rp)
                        eps_temp = self.eps_evap(Mp, Rp)
                        LXUV_temp = self.L_XUV(t, a)
                        RXUV_temp = self.RXUV(t, a, Mp, Rp)
                        ODE.set_parameters(Rp, LXUV_temp, eps_temp, RXUV_temp)
                        self.tau_atmo = self.atmo_timescale(-1)
                solution_y = np.asarray(solution).transpose()

            self.tau_dyn = np.min((self.kozai_timescale(-1), self.tidal_timescale(-1), 1e7))
            self.tau_dyn = np.max((self.tau_dyn, 1e5))
            
            # Storing the solutions
            self.t += list(solution_t[1:])
            self.h1 += list(solution_y[0][1:])
            self.h2 += list(solution_y[1][1:])
            self.h3 += list(solution_y[2][1:])
            self.e1 += list(solution_y[3][1:])
            self.e2 += list(solution_y[4][1:])
            self.e3 += list(solution_y[5][1:])
            self.Os1 += list(solution_y[6][1:])
            self.Os2 += list(solution_y[7][1:])
            self.Os3 += list(solution_y[8][1:])
            self.Op1 += list(solution_y[9][1:])
            self.Op2 += list(solution_y[10][1:])
            self.Op3 += list(solution_y[11][1:])
            self.e += list(solution_y[12][1:])
            self.a += list(solution_y[13][1:])
            self.Mp += [(heaviside3(Mp, self.Mcore)) for Mp in solution_y[14][1:]]
            
            t1 = t2
            Y = np.array([self.h1[-1], self.h2[-1], self.h3[-1], self.e1[-1], self.e2[-1], self.e3[-1], \
            self.Os1[-1], self.Os2[-1], self.Os3[-1], self.Op1[-1], self.Op2[-1], self.Op3[-1], \
            self.e[-1], self.a[-1], self.Mp[-1]])
                
            # Progression percentage
            prog = (t1 - self.t0)*100/(self.age - self.t0)

            # Printing the progression percentage each 10%
            if verbose:
                for i in range(self.prog_print + 10, int(prog) + 1, 10):
                    print('      ', '.'*(i//10), i, '% done...')
                self.prog_print = int(prog) - int(prog)%10
            
            # Saving
            if prog >= self.prog_save + self.freq_save:
                self.save_out(check_last=True)
                self.prog_save = int(prog) - int(prog)%self.freq_save

            # Checking if the planet crashed into the star
            if self.roche and self.a[-1]*(1. - self.e[-1]) < 2*self.Rp[-1]*(self.Ms/self.Mp[-1])**(1./3):
                print('  + Planet tidally disrupted at t = {:.2e} years.'.format(self.t[-1]))
                self.simul = False
        
        # Verbose
        if verbose:
            elapsed_time = tm.time() - start_time
            print('  + Simulation successfully performed. Elapsed time:', format_time(elapsed_time))
            
        
        # Saving
        self.save_out(check_last=True)
        if verbose: print('  + Saving the results successfully done.')
        
        
#----------------------------------------------------------------------------------------------------------------------------------------
# Returns the XUV radius of the inner planet
# Input: 't' (scalar) time, 'a' (scalar) semi-major axis, 'Mp' (scalar) mass of the planet, 'Rp' (scalar) radius of the planet
# Output: XUV radius (scalar)
    
    def RXUV(self, t, a, Mp, Rp):
        Mp_Rp = (((Mp/Rp)*const.G*(custom_mass/u.AU)).decompose(bases=u.cgs.bases)).value
        FXUV = self.L_XUV(t, a)/(4*np.pi*a**2)
        FXUV = ((FXUV*(custom_mass/u.yr**3)).decompose(bases=u.cgs.bases)).value
        ratio = - 0.185*np.log10(Mp_Rp) + 0.021*np.log10(FXUV) + 2.42
        ratio = np.max((0, ratio))
        RXUV = Rp*10**ratio
        return RXUV


#----------------------------------------------------------------------------------------------------------------------------------------
# Returns the evaporation efficiency of the inner planet
# Input: 'Mp' (scalar) mass of the planet, 'Rp' (scalar) radius of the planet
# Output: evaporation efficiency (scalar)
    
    def eps_evap(self, Mp, Rp):
        Mp_Rp = (((Mp/Rp)*const.G*(custom_mass/u.AU)).decompose(bases=u.cgs.bases)).value
        phi = np.log10(Mp_Rp)
        if phi > 13.11:
            eps = 10**(-0.98 - 7.29*(phi - 13.11))
        else:
            if phi < 12: phi = 12
            eps = 10**(-0.5 - 0.44*(phi - 12.))
        return eps
    
    
#----------------------------------------------------------------------------------------------------------------------------------------
# Returns the radius of the planet, as defined by tau = 2/3 (tau being the optical depth)
# Input: 'Mp' (scalar) mass of the planet, 'sma' (scalar) semi-major axis
# Output: radius (scalar)
        
    def retrieve_radius(self, t, Mp, sma, ecc, Rp_guess=None, force_comp=False, thresh_error=0.01, n_iter_max=20, verbose=False):

        min_method = 'parallel' if self.parallel else 'scipy'
        
        if Mp/self.Mcore - 1 < .02 and len(self.Rp) > 0: return self.Rp[-1]
        if Mp < self.Mcore*.9999999: raise ValueError('The mass of the planet must be higher than the mass of its core.')
        if verbose: start_time = tm.time()

        if self.atmo_grid and not force_comp:
            Mp_earth = Mp*morb2earth
            Teq = self.temp_eq(t, sma, ecc)
            try: 
                Rp_for_Tint = self.Rp[-1]
            except (IndexError, AttributeError):
                rho_guess = 1.64*rhocgs2orb
                Rp_for_Tint = (Mp/rho_guess)**(1./3)
            Tint = self.temp_int(Rp_for_Tint)
            coeffs_fit = np.array([interp((Teq, Tint)) for interp in self.radius_fit])
            Rp = analytic_radius(Mp_earth, coeffs_fit)
            return Rp
        
        #Rp_guess = None

        # Initial guess of Rp
        if Rp_guess is None:
            # Using the density of Neptune
            '''
            rho_guess = 1.64*rhocgs2orb
            Rp_guess = (Mp/rho_guess)**(1/3)
            bounds = (.01*Rp_guess, 10.*Rp_guess)
            '''

            # Using densities as bounds
            '''
            rho1 = 1e-8*rhocgs2orb
            Rp1 = (Mp/((4*np.pi/3)*rho1))**(1/3)
            rho2 = 20*rhocgs2orb
            Rp2 = (Mp/((4*np.pi/3)*rho2))**(1/3)
            bounds = (Rp2, Rp1)
            '''

            # Using 1 - 50 R_Earth as bounds
            bounds = ((1/11.209)*Rjup2orb, (50/11.209)*Rjup2orb)
        else:
            try:
                Rp_guess = float(Rp_guess)
                factor = 0.1
                bounds = (Rp_guess*(1 - factor), Rp_guess*(1 + factor))
            except TypeError:
                bounds = (Rp_guess[0], Rp_guess[1])

        # We try to find the radius of the planet by converging towards it
        if min_method == 'scipy':
            Rp = brute(lambda Rp: self.atmospheric_structure(t, Rp[0], Mp, sma, ecc), \
                       [bounds])[0]
        elif min_method == 'manual':
            Rp = minimize_manual(lambda Rp: self.atmospheric_structure(t, Rp, Mp, sma, ecc), bounds, \
                                 num=10, rec=1)
        elif min_method == 'parallel':
            GLOB_DATA['JADE'] = self
            GLOB_DATA['t'] = t
            GLOB_DATA['Mp'] = Mp
            GLOB_DATA['sma'] = sma
            GLOB_DATA['ecc'] = ecc
            GLOB_DATA['acc'] = self.atmo_acc
            Rp, epsilon = mara_search(atmospheric_structure_glob, bounds[0], bounds[1], 
                                      self.parallel, self.parallel, thresh_error, n_iter_max, verbose=False)
            if Rp == -1:
                if self.atmo_acc:
                    GLOB_DATA['acc'] = False
                    npt = self.parallel if self.parallel > 6 else 7
                    Rp, epsilon = mara_search(atmospheric_structure_glob, bounds[0], bounds[1], npt, self.parallel, verbose=False)
                    if Rp == -1:
                        if not self.lazy_init and len(self.Rp) > 0: self.save_out(check_last=True)
                        raise SystemExit('Radius retrieval did not converge.')
                else:
                    if not self.lazy_init and len(self.Rp) > 0: self.save_out(check_last=True)
                    raise SystemExit('Radius retrieval did not converge.')
        else: raise ValueError('Invalid minimisation method.')
        if verbose: elapsed_time = tm.time() - start_time
        
        # Verbose
        if verbose:
            print('  + Atmospheric structure: ')
            print('       . Rp = ' + format(Rp/Rjup2orb, '-.3') + ' Rjup') 
            print('       . Relative error = ' + format(epsilon, '-.3')) 
            print('       . Elapsed time:', format_time(elapsed_time))
            
        return Rp   
    


#----------------------------------------------------------------------------------------------------------------------------------------
# Computes the structure of the gaseous envelope using a certain radius for the planet
# Input: 'Rp' (scalar) radius of the planet, 'Mp' (scalar) mass of the planet, 'sma' (scalar) semi-major axis, 'plot' (bool)
# Output: mass at r = 0 divided by the total mass Mp (scalar)
        
    def atmospheric_structure(self, t, Rp, Mp, sma, ecc, acc=False, verbose=False, plot=False):
        
        if verbose: print('  + Atmospheric structure for the radius', format(convert_distance(Rp, 'Rjup'), '-.15'), 'Rjup ...')       
        integrator = 'dopri5'
        
        if Mp > self.Mcore:

            # ZONE A -------------------------------------------------------------------------------------------

            # Initial values at the top of Zone A, tau=2/3
            tau = np.log10(2./3)
            P = self.P_surf(t, sma, ecc, Mp, Rp, 10**tau)
            Y = np.log10(np.array([Mp/matm2orb, P, Rp/Rjup2orb]))

            if plot:
                tau_plot = [2./3]
                m_plot = [Mp/matm2orb]
                r_plot = [Rp/Rjup2orb]
                P_plot = [P]
            
            # Transition between Zone A and Zone B
            tau_end = np.log10(100./(np.sqrt(3.)*self.gamma(self.temp_eq(t, sma, ecc))))
            if tau_end < tau: return 1e50

            # Integration of Zone A

            if acc: dtau = (tau_end - tau)/100.

            intg_A = ode(self.zone_A).set_integrator(integrator)
            #if plot:  intg_A = ode(lambda Y, m: self.zone_A(Y, tau, verbose=True)).set_integrator(integrator)
            intg_A.set_f_params(t, sma, ecc)
            intg_A.set_initial_value(Y, tau)

            if plot:
                def solout_A(tau_sol, Y_sol):
                    tau_p = 10**tau_sol
                    m_p, P_p, r_p = 10**Y_sol
                    tau_plot.append(tau_p)
                    m_plot.append(m_p)
                    r_plot.append(r_p)
                    P_plot.append(P_p)
                intg_A.set_solout(solout_A)
            
            if acc:
                while intg_A.successful() and intg_A.t < tau_end:
                    intg_A.integrate(intg_A.t + dtau)
            else:
                intg_A.integrate(tau_end)

            if intg_A.successful():
                tau = intg_A.t
                m, P, r = intg_A.y
                if verbose: print('       . Zone A successfully integrated.')
            else:
                if verbose: print('>>> Error in Zone A integration:', intg_A.get_return_code())
                if plot: raise SystemExit('>>> Error in Zone A integration:', intg_A.get_return_code())
                return 1e50
            
            
            # ZONE B -------------------------------------------------------------------------------------------

            if plot: T_plot = [self.temp_SG(t, sma, ecc, r_plot[i]*Rjup2orb, tau_plot[i])/Tatm2cgs for i in range(len(tau_plot))]

            # Initial values at the top of Zone B
            T = np.log10(self.temp_SG(t, sma, ecc, (10**r)*Rjup2orb, 10**tau)/Tatm2cgs) 
            Y = np.array([r, P, T])
            
            # Transition between Zone B and the rocky mantle
            m_end = np.log10(self.Mcore/matm2orb)
            if m_end > m: 
                if verbose: print('>>> Already exceeded rocky mantle.')
                if plot: raise SystemExit('>>> Already exceeded rocky mantle.')
                return 1e50
            
            # Integration of Zone B

            if acc: dm = (m_end - m)/100.

            intg_B = ode(self.zone_B).set_integrator(integrator)
            #if plot:  intg_B = ode(lambda Y, m: self.zone_B(Y, m, verbose=True)).set_integrator(integrator)
            intg_B.set_initial_value(Y, m)

            if plot:
                def solout_B(m_sol, Y_sol):
                    m_p = 10**m_sol
                    r_p, P_p, T_p = 10**Y_sol
                    m_plot.append(m_p)
                    r_plot.append(r_p)
                    P_plot.append(P_p)
                    T_plot.append(T_p)
                intg_B.set_solout(solout_B)

            if acc:
                while intg_B.successful() and intg_B.t > m_end:
                    intg_B.integrate(intg_B.t + dm)
            else:
                intg_B.integrate(m_end)

            if intg_B.successful():
                m = intg_B.t
                r, P, _ = intg_B.y
                if verbose: print('       . Zone B successfully integrated.')
            else:
                if verbose: print('>>> Error in Zone B integration:', intg_B.get_return_code())
                if plot: raise SystemExit('>>> Error in Zone B integration:', intg_B.get_return_code())
                return 1e50

        
        else:

            m = np.log10(self.Mcore/matm2orb)
            r = np.log10(Rp/Rjup2orb)
            P = np.log10(1e3/Patm2cgs)
            if plot:
                tau_plot = []
                T_plot = []
                m_plot = [Mp/matm2orb]
                r_plot = [Rp/Rjup2orb]
                P_plot = [10**P]

        # ROCKY MANTLE ----------------------------------------------------------------------------------------
        
        # Initial values at the top of the rocky mantle       
        Y = np.array([r, P])
        
        # End condition for the integration of the rocky mantle
        m_end = np.log10((1. - self.fmantle)*self.Mcore/matm2orb)
        if m_end > m: 
            if verbose: print('>>> Already exceeded iron core.')
            if plot: raise SystemExit('>>> Already exceeded iron core.')
            return 1e50
        
        # Integration of the rocky mantle

        if acc: dm = (m_end - m)/100.

        intg_C = ode(self.mantle).set_integrator(integrator)
        intg_C.set_initial_value(Y, m)

        if plot:
            def solout_C(m_sol, Y_sol):
                m_p = 10**m_sol
                r_p, P_p = 10**Y_sol
                m_plot.append(m_p)
                r_plot.append(r_p)
                P_plot.append(P_p)
            intg_C.set_solout(solout_C)

        if acc:
            while intg_C.successful() and intg_C.t > m_end:
                intg_C.integrate(intg_C.t + dm)
        else:
            intg_C.integrate(m_end)

        if intg_C.successful():
            m = intg_C.t
            r, P = intg_C.y
            if verbose: print('       . Rocky mantle successfully integrated.')
        else:
            if verbose: print('>>> Error in rocky mantle integration:', intg_C.get_return_code())
            if plot: raise SystemExit('>>> Error in rocky mantle integration:', intg_C.get_return_code())
            return 1e50


        # IRON CORE ----------------------------------------------------------------------------------------
        
        # Initial values at the top of the iron core        
        Y = np.array([m, P])
        
        # End condition for the integration of the iron core
        r_end = np.log10(Rp/Rjup2orb) - 4.
        if r_end > r: 
            if verbose: print('>>> Already exceeded center of the planet.')
            if plot: raise SystemExit('>>> Already exceeded center of the planet.')
            return 1e50
        
        # Integration of the iron core

        if acc: dr = (r_end - r)/100.

        intg_D = ode(self.core).set_integrator(integrator)
        intg_D.set_initial_value(Y, r)

        if plot:
            def solout_D(r_sol, Y_sol):
                r_p = 10**r_sol
                m_p, P_p = 10**Y_sol
                m_plot.append(m_p)
                r_plot.append(r_p)
                P_plot.append(P_p)
            intg_D.set_solout(solout_D)

        if acc:
            while intg_D.successful() and intg_D.t > r_end:
                intg_D.integrate(intg_D.t + dr)
        else:
            intg_D.integrate(r_end)

        if intg_D.successful():
            m, _ = intg_D.y
            if verbose: print('       . Iron core successfully integrated.')
        else:
            if verbose: print('>>> Error in iron core integration:', intg_D.get_return_code())
            if plot: raise SystemExit('>>> Error in iron core integration:', intg_D.get_return_code())
            return 1e50

        if plot: return tau_plot, m_plot, r_plot, T_plot, P_plot


        # RESULT -------------------------------------------------------------------------------------------
            
        error = (10**m) / (Mp/matm2orb)
        if verbose: print('       . Relative error:', format(error, '-.5'))
        
        return error
    

#---------------------------------------------------------------------------------------------------------------------------------------- 
# Returns the derivatives of m, P, tau with respect to tau in the Zone A
# Input: 't' (time), 'tau' (scalar), 'Y' = [m, P, r], 'sma' (scalar)
# Output: 'Y_prime'
        
    def zone_A(self, tau, Y, t, sma, ecc, verbose=False):
        
        # Storing the parameters
        m, P, r = np.power(10, Y)
        tau = 10**tau
        T = self.temp_SG(t, sma, ecc, r*Rjup2orb, tau)
        
        # Deducing the density from the EOS
        P_cgs = P*Patm2cgs
        rho_cgs = self.rho(T, P_cgs)
        rho = rho_cgs*rhocgs2atm
        
        # Deducing the opacity from the tables
        kappa = self.kappa(T, rho_cgs)*opcgs2atm
                
        # Derivatives of m, r, P
        m_prime = -4*np.pi*tau*r**2/(kappa*m)
        P_prime = m*tau/(kappa*P*r**2)
        r_prime = -tau/(kappa*r*rho)
        
        Y_prime = np.array([m_prime, P_prime, r_prime])

        if verbose:
            print('Zone A')
            print('P={:.2e}bar, T={:.0f}K, rho={:.2e}g/cm3, kappa={:.2e}cm2/g'.format(P_cgs*1e-6, T, rho_cgs, kappa/opcgs2atm))
            print('---')
        
        return Y_prime
    
    
#---------------------------------------------------------------------------------------------------------------------------------------- 
# Returns the derivatives of r, P, T with respect to m in the Zone B
# Input: 'm' (scalar), 'Y' = [r, P, T]
# Output: 'Y_prime'
        
    def zone_B(self, m, Y, verbose=False):
        
        # Storing the parameters
        r, P, T = np.power(10, Y)
        m = 10**m
        
        # Deducing the density from the EOS
        P_cgs = P*Patm2cgs
        T_cgs = T*Tatm2cgs
        rho_cgs = self.rho(T_cgs, P_cgs)
        rho = rho_cgs*rhocgs2atm
        
        # Deducing the opacity from the tables
        kappa = self.kappa(T_cgs, rho_cgs)*opcgs2orb
        
        # Deducing the adiabatic gradient from the EOS
        nabla_conv = self.adg(T_cgs, P_cgs)

        
        # Calculating the convective gradient
        P_orb = P*Patm2orb
        m_orb = m*matm2orb
        nabla_rad = 3*kappa*self.Lp*P_orb/(64*np.pi*sigmaB*m_orb*T_cgs**4)
        
        # Derivatives of m, r, T
        r_prime = m/(4*np.pi*rho*r**3)
        P_prime = -m**2/(4*np.pi*P*r**4)
        T_prime = P_prime*np.min([nabla_rad, nabla_conv])

        if verbose: 
            print('Zone B')
            print('P={:.2e}bar, T={:.0f}, rho={:.2e}, nabla_rad={:.3f}, nabla_conv={:.3f}'.format(P_cgs*1e-6, T_cgs, rho_cgs, nabla_rad, nabla_conv))
            print('k={:.2e}cm2/g, Lp={:.2e}erg/s'.format(kappa/opcgs2orb, self.Lp/Lcgs2orb))
            print('----')
        
        Y_prime = np.array([r_prime, P_prime, T_prime])
        
        return Y_prime
    
    
#---------------------------------------------------------------------------------------------------------------------------------------- 
# Returns the derivatives of r, P with respect to m in the rocky mantle
# Input: 'm' (scalar), 'Y' = [r, P]
# Output: 'Y_prime'
        
    def mantle(self, m, Y):
        
        # Storing the parameters
        r, P = np.power(10, Y)
        m = 10**m
        
        # Deducing the density from the EOS
        P_si = P*Patm2si
        rho_si = self.rho_core(P_si, 'rock')
        rho = rho_si*rhosi2atm
        
        # Derivatives of r, P
        r_prime = m/(4*np.pi*rho*r**3)
        P_prime = -m**2/(4*np.pi*P*r**4)
        
        Y_prime = np.array([r_prime, P_prime])
        
        return Y_prime


#---------------------------------------------------------------------------------------------------------------------------------------- 
# Returns the derivatives of m, P with respect to r in the iron core
# Input: 'r' (scalar), 'Y' = [m, P]
# Output: 'Y_prime'
        
    def core(self, r, Y):
        
        # Storing the parameters
        m, P = np.power(10, Y)
        r = 10**r
        
        # Deducing the density from the EOS
        P_si = P*Patm2si
        rho_si = self.rho_core(P_si, 'iron')
        rho = rho_si*rhosi2atm
        
        # Derivatives of m, P
        m_prime = 4*np.pi*r**3*rho/m
        P_prime = -m*rho/(r*P)
        
        Y_prime = np.array([m_prime, P_prime])
        
        return Y_prime
    
    
#----------------------------------------------------------------------------------------------------------------------------------------    
# Returns the surface pression of the atmosphere
# Input: 't' (scalar) time, 'sma' (scalar) semi-major axis, 'ecc' (scalar) eccentricity, 'Mp' (scalar) mass, 'Rp' (scalar) radius, 'tau' (scalar) optical depth
# Output: 'P' (scalar)
        
    def P_surf(self, t, sma, ecc, Mp, Rp, tau, verbose=False):
        
        #return 1e5/Patm2cgs
        T = self.temp_SG(t, sma, ecc, Rp, tau)
        g = Gcgs*(Mp/mcgs2orb)/(Rp/Rcgs2orb)**2

        def k1(P):
            return g*tau/P

        def k2(P):
            rho = self.rho(T, P)
            return self.kappa(T, rho)

        def error(P):
            return np.abs(1. - k1(P)/k2(P))

        res = minimize_scalar(lambda logP: error(10**logP), bounds=(1, 11))
        P_res = 10**res.x
        P = np.max([P_res, 1e4])/Patm2cgs
        P = P_res/Patm2cgs

        if verbose:
            print('T={:.0f}K, g={:.2f}cm/s2, Pres={:.2e}bar, Psurf={:.2e}bar'.format(T, g, P_res*1e-6, P*Patm2cgs*1e-6))
            print('k1={:.2e}cm2/g, k2={:.2e}cm2/g'.format(k1(P_res), k2(P_res)))
            print('---')

        return P


#----------------------------------------------------------------------------------------------------------------------------------------    
# Returns the temperature inside the Zone A of the gaseous envelope at a certain optical depth tau
# Input: 't' (scalar) time, 'sma' (scalar) semi-major axis, 'r' (scalar) radius, 'tau' (scalar) optical depth
# Output: 'T' (scalar)
        
    def temp_SG(self, t, sma, ecc, r, tau):
        
        # Intrinsic temperature that characterizes the heat flux from the planet's interior
        Tint = self.temp_int(r)
        
        # Equilibrium temperature obtained by averaging the stellar radiation over the planet surface
        Teq = self.temp_eq(t, sma, ecc)
        
        # Ratio of the visible opacity to the thermal opacity
        gamma = self.gamma(Teq)
        
        # Exponential integral used in the calculation
        E2 = expn(2, gamma*tau)
        
        # Calculation of the temperature
        T = ((0.75*Tint**4.)*(2./3 + tau) + (0.75*Teq**4.)*(2./3 + (2./(3.*gamma))*(1.+(gamma*tau/2.-1.)*np.exp(-gamma*tau)) \
              + (2*gamma/3.)*E2*(1.-0.5*tau**2.)))**(1./4)

        return T
    
    
#----------------------------------------------------------------------------------------------------------------------------------------    
# Equilibrium temperature of a planet
# Input: 't' (scalar) time, 'sma' (scalar) semi-major axis, 'ecc' (scalar) eccentricity
# Output: 'Teq' (scalar)
        
    def temp_eq(self, t, sma, ecc):
        #return 1000.
        #Ts = (self.L_bol(t)/(4*np.pi*sigmaB*self.Rs**2))**(1/4)
        #ecc = 0.
        Teq = (self.L_bol(t)/(16*np.pi*sigmaB*np.sqrt(1 - ecc**2)*sma**2))**(1./4)
        return Teq
    
    
#----------------------------------------------------------------------------------------------------------------------------------------    
# Intrinsic temperature of a planet a certain radius
# Input: 'r' (scalar) radius
# Output: 'Tint' (scalar)
        
    def temp_int(self, r):
        if r != 0.:
            Tint = (self.Lp/(4*np.pi*sigmaB*r**2))**(1./4.)
        else:
            Tint = 0.
        return Tint
    
    
#----------------------------------------------------------------------------------------------------------------------------------------    
# Density in the core
# Input: 'P' (scalar) pressure in SI, 'mode' (string) 'iron' or 'rock'
# Output: 'rho' (scalar) density in SI
    
    def rho_core(self, P, mode):
        rho0 = {'iron':8300., 'rock':4100.}
        cc = {'iron':0.00349, 'rock':0.00161}
        nn = {'iron':0.528, 'rock':0.541}
        rho = rho0[mode] + cc[mode]*P**nn[mode]
        return rho
    
    
#----------------------------------------------------------------------------------------------------------------------------------------
# Planetary luminosity coming from radiogenic heating of the core
# Input: 't' (scalar) time
# Output: 'L_radio' (scalar)
        
    def L_radio(self, t):
        
        QK = ((3.723e-7*u.erg/(u.g*u.s)).decompose(bases=orbital_units)).value
        QU = ((2.8998e-8*u.erg/(u.g*u.s)).decompose(bases=orbital_units)).value
        QTh = ((1.441e-8*u.erg/(u.g*u.s)).decompose(bases=orbital_units)).value
        nK = 0.543e-9
        nU = 0.155e-9
        nTh = 0.0495e-9
        Q = QK*np.exp(-nK*t) + QU*np.exp(-nU*t) + QTh*np.exp(-nTh*t)
        
        f_mantle = self.fmantle
        f_rocky = 1
        
        L_radio = Q*f_mantle*f_rocky*self.Mcore
        
        return L_radio
    
    
#----------------------------------------------------------------------------------------------------------------------------------------
# Planetary luminosity coming from tidal dissipation in the planet
# Output: 'L_tide' (scalar)
        
    def L_tide(self):
        return 0
        Mp, Rp, a, e = self.Mp[-1], self.Rp[-1], self.a[-1], self.e[-1]
        e1, e2, e3, h1, h2, h3, Op1, Op2, Op3 = self.e1[-1], self.e2[-1], self.e3[-1], \
            self.h1[-1], self.h2[-1], self.h3[-1], self.Op1[-1], self.Op2[-1], self.Op3[-1]
        q1 = h2*e3 - h3*e2
        q2 = h3*e1 - h1*e3
        q3 = h1*e2 - h2*e1 
        Ope = Op1*e1 + Op2*e2 + Op3*e3
        Opq = Op1*q1 + Op2*q2 + Op3*q3
        Oph = Op1*h1 + Op2*h2 + Op3*h3
        mu = self.Ms*Mp/(self.Ms + Mp)
        n = np.sqrt((self.Ms + Mp)/a**3)
        e = np.min((e, 0.1))
        h1 = (1 + (3/2)*e**2 + (1/8)*e**4)/(1 - e**2)**(9/2)
        h2 = (1 + (9/2)*e**2 + (5/8)*e**4)/(1 - e**2)**(9/2)
        h3 = (1 + 3*e**2 + (3/8)*e**4)/(1 - e**2)**(9/2)
        h4 = (1 + (15/2)*e**2 + (45/8)*e**4 + (5/16)*e**6)/(1 - e**2)**6
        h5 = (1 + (31/2)*e**2 + (255/8)*e**4 + (185/16)*e**6 + (25/64)*e**8)/(1 - e**2)**(15/2)
        L_tide = mu*a**2*n*(self.Ms/Mp)*(Rp/a)**5*(6*self.kp/self.Qp) * \
                 ((1/2)*(Ope**2*h1 + Opq**2*h2) + Oph**2*h3 - 2*n*Oph*h4 + n**2*h5)
        return L_tide
    

#----------------------------------------------------------------------------------------------------------------------------------------
# Planetary luminosity coming from the cooling of the core, plot purposes
# Input: 't' (scalar) time, 'Menv' (scalar) envelope mass
# Output: 'L_core' (scalar)
        
    def planet_luminosity_plot(self, t, Menv, verbose=False):
        Menv_red = Menv*morb2earth
        Mcore_red = self.Mcore*morb2earth
        a0, b1, b2, c1, c2 = self.a0_tabular(t), self.b1_tabular(t), self.b2_tabular(t),self.c1_tabular(t), self.c2_tabular(t)
        L_core = a0 + b1*Mcore_red + b2*Mcore_red**2 + c1*Menv_red + c2*Menv_red**2
        if verbose: print('       . L_pl =', format(L_core, '-.3'), 'L_jup')
        return L_core
    

#----------------------------------------------------------------------------------------------------------------------------------------
# Planetary luminosity coming from the cooling of the core
# Input: 't' (scalar) time
# Output: 'L_core' (scalar)
        
    def planet_luminosity(self, t, verbose=False):
        #return self.Mp[-1]*((10**(-10.5)*u.W/u.kg).decompose(bases=orbital_units)).value
        Ljup = 8.67e-10
        Menv = self.Mp[-1] - self.Mcore
        Menv_red = Menv*morb2earth
        Mcore_red = self.Mcore*morb2earth
        a0, b1, b2, c1, c2 = self.a0_tabular(t), self.b1_tabular(t), self.b2_tabular(t),self.c1_tabular(t), self.c2_tabular(t)
        L_core = (a0 + b1*Mcore_red + b2*Mcore_red**2 + c1*Menv_red + c2*Menv_red**2)*Ljup*Lsun2orb
        if verbose: print('       . L_pl =', format(L_core/(Ljup*Lsun2orb), '-.3'), 'L_jup')
        #print('>>>>', Menv_red, Mcore_red, a0, b1, b2, c1, c2, L_core)
        return L_core
    

#----------------------------------------------------------------------------------------------------------------------------------------
# Planetary total luminosity
# See L_radio, L_evap and L_tide for input
# Output: total luminosity (scalar)
        
    def planet_luminosity_OLD(self, t, verbose=False):
        L_radio = self.L_radio(t)
        L_tide = self.L_tide()
        L_core = self.L_core(t)
        L = L_radio + L_tide + L_core
        if verbose:
            print('       . L_radio =', format(L_radio, '-.3'), '; L_tide =', format(L_tide, '-.3'), \
                  '; L_core =', format(L_core, '-.3'))
        return L
    
    
#----------------------------------------------------------------------------------------------------------------------------------------
# Bolometric luminosity of the star
# Input: 't' (scalar) time
# Output: 'L_bol' (scalar)
        
    def L_bol(self, t):
        if self.lum_mode == 'analytic':
            L_bol = self.init_Lbol
        else:
            L_bol = self.Lbol_tabular(t)*Lcgs2orb
        return L_bol
    
    
#----------------------------------------------------------------------------------------------------------------------------------------
# XUV luminosity of the star
# Input: 't' (scalar) time, 'a' (scalar) semi-major axis
# Output: 'L_XUV' (scalar)
        
    def L_XUV(self, t, a):
        
        if self.lum_mode == 'analytic':
            
            # Bolometric luminosity
            L_bol = self.L_bol(t)
            
            # X-ray luminosity
            ratio_sat = self.LX_Lbol_sat
            tau_sat = self.tau_X_bol_sat
            alpha = self.alpha_X_bol
            if t <= tau_sat:
                ratio = ratio_sat
            else:
                ratio = ratio_sat*(t/tau_sat)**(-alpha)
            L_X = ratio*L_bol
            
            # EUV luminosity
            gamma = -0.45
            alpha = 650
            F_X = L_X/(4*np.pi*a**2)
            F_X = ((F_X*(custom_mass/u.yr**3)).decompose(bases=u.cgs.bases)).value
            ratio = alpha*(F_X**gamma)
            L_EUV = ratio*L_X
            L_XUV = L_X + L_EUV
            
        else:
            
            L_XUV = self.LXUV_tabular(t)
            L_XUV *= Lcgs2orb            
        
        return L_XUV


#----------------------------------------------------------------------------------------------------------------------------------------
# Saves the current state to an output file
# Input: 'flush' (boolean) reset the parameters to their last value, 'check_last' (boolean) check lastly saved output to avoid redundancy

    def save_out(self, flush=True, check_last=False):
        save = True
        if self.npts_save_npz is None:
            saved_dict = {'t':self.t, 't_atmo':self.t_atmo, 'h1':self.h1, 'h2':self.h2, 'h3':self.h3, \
                          'e1':self.e1, 'e2':self.e2, 'e3':self.e3, 'Os1':self.Os1, 'Os2':self.Os2, 'Os3':self.Os3, \
                          'Op1':self.Op1, 'Op2':self.Op2, 'Op3':self.Op3, 'e':self.e, 'a':self.a, 'Mp':self.Mp, 'Rp':self.Rp}
        else:
            saved_t = np.linspace(self.t[0], self.t[-1], self.npts_save_npz)
            saved_dict = {
                't':saved_t,
                't_atmo':saved_t,
                'h1':interpol(saved_t, self.t, self.h1),
                'h2':interpol(saved_t, self.t, self.h2),
                'h3':interpol(saved_t, self.t, self.h3),
                'e1':interpol(saved_t, self.t, self.e1),
                'e2':interpol(saved_t, self.t, self.e2),
                'e3':interpol(saved_t, self.t, self.e3),
                'Os1':interpol(saved_t, self.t, self.Os1),
                'Os2':interpol(saved_t, self.t, self.Os2),
                'Os3':interpol(saved_t, self.t, self.Os3),
                'Op1':interpol(saved_t, self.t, self.Op1),
                'Op2':interpol(saved_t, self.t, self.Op2),
                'Op3':interpol(saved_t, self.t, self.Op3),
                'e':interpol(saved_t, self.t, self.e),
                'a':interpol(saved_t, self.t, self.a),
                'Mp':interpol(saved_t, self.t, self.Mp),
                'Rp':interpol(saved_t, self.t_atmo, self.Rp)
            }
        if self.last_out == 0: check_last = False
        if check_last:
            last_dict = np.load('{}saved_data/{}/{}_{:03d}.npz'.format(self.global_path, self.out, self.name, self.last_out), allow_pickle=True)
            save = False
            for t in saved_dict['t']:
                if t not in last_dict['t']: 
                    save = True
                    break
        if save: 
            self.last_out += 1
            np.savez( '{}saved_data/{}/{}_{:03d}.npz'.format(self.global_path, self.out, self.name, self.last_out), **saved_dict)
        if flush and save:
            self.t = [self.t[-1]]
            self.h1 = [self.h1[-1]]
            self.h2 = [self.h2[-1]]
            self.h3 = [self.h3[-1]]
            self.e1 = [self.e1[-1]]
            self.e2 = [self.e2[-1]]
            self.e3 = [self.e3[-1]]
            self.Os1 = [self.Os1[-1]]
            self.Os2 = [self.Os2[-1]]
            self.Os3 = [self.Os3[-1]]
            self.Op1 = [self.Op1[-1]]
            self.Op2 = [self.Op2[-1]]
            self.Op3 = [self.Op3[-1]]
            self.e = [self.e[-1]]
            self.a = [self.a[-1]]
            self.Mp = [self.Mp[-1]]
            self.Rp = [self.Rp[-1]]
            self.t_atmo = [self.t_atmo[-1]]

    
#----------------------------------------------------------------------------------------------------------------------------------------
# Dummy function for multiprocessing purposes

GLOB_DATA = {
    'JADE': None,
    't': None,
    'Mp': None,
    'sma': None,
    'ecc': None,
    'acc': None
}

def atmospheric_structure_glob(Rp):
    return GLOB_DATA['JADE'].atmospheric_structure(GLOB_DATA['t'], Rp, GLOB_DATA['Mp'], GLOB_DATA['sma'], GLOB_DATA['ecc'], acc=GLOB_DATA['acc'])


#----------------------------------------------------------------------------------------------------------------------------------------
# Function to read input files

def read_input_file(path):

    # Initializing the dictionnary
    settings = {}
    
    #------------------------------------------------------
    # Reading settings file line by line
    
    for line in open(path, 'r'):
        temp = line.rstrip().split()
        
        # Ignoring empty lines
        if len(temp) < 1:
            continue
        
        # Ignoring comments
        if '#' in temp[0]:
            continue
        
        # Reading key and value from the line, which is formatted as 'key = value'
        key = temp[0]
        if len(temp) > 2:
            value = temp[2]
            if len(temp) > 3:
                value = temp[2:]
        else:
            value = ''
        
        # Casting the value into the right type
        if value in ['True', 'False']:
            value = str_to_bool(value)
        elif len(temp) > 3:
            value = [float(eval(val)) for val in value]
        else:
            try:
                value = float(eval(value))
            except:
                pass
        
        # Adding the key value pair to the dictionnary
        settings[key] = value

    return settings


#----------------------------------------------------------------------------------------------------------------------------------------
# Analytic functions for the radius, from an atmospheric grid exploration. To be adapted with the investigated planet.
# The example shown below is for GJ436b, see Attia et al. (2025).

def c_function(a):
    power, interc = -1.0000199845566071, -1.0000166831011739
    logc = np.log(a)*power + interc
    c = np.exp(logc)
    return c

def d_function(a):
    d = -0.9999953510919015*a + 1.9998974861128505
    return d

def analytic_radius(Mp, coeffs):
    if Mp <= 19.35:
        return 2.36*Rearth2orb
    a, b = coeffs
    c = c_function(a)
    d = d_function(a)
    logRp = a*np.log(Mp/b)**c + d
    Rp = np.exp(logRp)*Rearth2orb
    return Rp

