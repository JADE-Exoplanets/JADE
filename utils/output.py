#--------------------------------------------------------------------------------------------
#--- Util for handling JADE outputs.
#--------------------------------------------------------------------------------------------
#--- Import the 'JADE_output' class elsewhere and refer to its documentation for usage. 
#--------------------------------------------------------------------------------------------
#--- Author: Mara Attia
#--------------------------------------------------------------------------------------------

import os
import warnings
import numpy as np
import time as tm
from scipy import interpolate
from scipy.special import expn
from bisect import bisect_right
from astropy import units as u
from astropy.units import cds
from astropy import constants as const
from zipfile import BadZipFile


#--------------------------------------------------------------
# Units

cds.enable()
custom_mass = u.def_unit('custom_mass', u.AU**3/(const.G*u.yr**2))
units = [u.mol, u.rad, u.cd, u.AU, u.yr, u.A, custom_mass, u.K]
dorb2jup = 1./((1*u.Rjup).decompose(bases=units)).value
morb2jup = 1./((1.*u.Mjup).decompose(bases=units)).value
dorb2sun = 1./((1*u.Rsun).decompose(bases=units)).value
morb2sun = 1./((1*u.Msun).decompose(bases=units)).value
djup2cgs = ((1.*u.Rjup).decompose(bases=u.cgs.bases)).value
mjup2cgs = ((1.*u.Mjup).decompose(bases=u.cgs.bases)).value
dorb2cgs = dorb2jup*djup2cgs
torb2cgs = ((1*u.yr).decompose(bases=u.cgs.bases)).value
Lorb2cgs = 1./((1*u.erg/u.s).decompose(bases=units)).value
Gcgs = ((const.G).decompose(bases=u.cgs.bases)).value


#--------------------------------------------------------------
# Ignore useless warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


#--------------------------------------------------------------

class JADE_output:
	
	'''
	Class allowing to conveniently use a JADE output.
	Useful for post-processing purposes.

	Class attributes that are automatically generated:
	+++ t_dyn (numpy array): dynamical time (yr).
	+++ h1, h2, h3 (numpy arrays): components of the unitary h vector. Same shape as t_dyn.
	+++ e1, e2, e3 (numpy arrays): components of the unitary e vector. Same shape as t_dyn.
	+++ Os1, Os2, Os3 (numpy arrays): components of the stellar spin axis (rad/yr). Same shape as t_dyn.
	+++ Op1, Op2, Op3 (numpy arrays): components of the planetary spin axis (rad/yr). Same shape as t_dyn.
	+++ e (numpy array): eccentricity. Same shape as t_dyn.
	+++ a (numpy array): separation (AU). Same shape as t_dyn.
	+++ Mp (numpy array): planetary mass (Mjup). Same shape as t_dyn.
	+++ t_atm (numpy array): atmospheric time (yr).
	+++ Rp (numpy array): planetary radius (Rjup). Same shape as t_atm.
	+++ Ms (float): stellar mass (Msun).
	+++ Rs (float): stellar radius (Rsun).
	+++ ks (float): stellar apsidal constant.
	+++ Qs (float): stellar tidal dissipation parameter.
	+++ alphas (float): stellar moment of inertia parameter.
	+++ Mc (float): planetary core mass (Mjup).
	+++ fm (float): fraction of rocky mantle in the core.
	+++ Y (float): fraction of helium in the atmosphere.
	+++ kp (float): planetary apsidal constant.
	+++ Qp (float): planetary tidal dissipation parameter.
	+++ alphap (float): planetary moment of inertia parameter.
	+++ Mpert (float): perturber's mass (Mjup).
	+++ apert (float): perturber's separation (AU).
	+++ epert (float): perturber's eccentricity.
	+++ ipert (float): perturber's inclination w.r.t. l.o.s. (degrees).
	+++ lpert (float): perturber's mean longitude (degrees).
	+++ opert (float): perturber's longitude of periastron (degrees).
	+++ Opert (float): perturber's longitude of node (degrees).

	Use set_time() and generate() to construct additional parameters.

	A minimal example:
	+++ from output import JADE_output
	+++ import matplotlib.pyplot as plt
	+++ simu = JADE_output('input_file.txt')
	+++ simu.set_time()
	+++ simu.generate('P')
	+++ plt.figure()
	+++ plt.plot(simu.t, simu.P)
	+++ plt.show()
	'''

	def __init__(self, txt, npz=None, reduced=None, sort=False, verbose=False):

		'''
		Constructor.

		Input:
		+++ txt (string): path to the input file (absolute or relative to input/).
		+++ npz (string or sequence of strings): if defined, path to the npz folder or list of npz files paths (absolute or relative to saved_data/).
									        	 Overrides the output path inside the input file.
		+++ reduced (integer): if defined, reduces the simulation to a certain number of points.
		+++ sort (boolean): enforce sorting (for badly-named npz files). Default: False.
		+++ verbose (boolean): verbosity. Default: False.
		'''
		
		self.__gpath = os.path.dirname(os.path.realpath(__file__)) + '/../'
		if txt.startswith('/'):
			txt_path = txt
		else:
			txt_path = '{}input/{}'.format(self.__gpath, txt)
		self.__ipath = txt_path
		self.__verbose = verbose

		if self.__verbose: 
			ti = tm.time()
			print('+ Reading input file... ', end='')

		input_dict = read_input(txt_path)
		self.name = input_dict['name']
		self.__out = input_dict['output_dir']
		self.__age = float(eval(input_dict['age']))*1e6
		self.Ms = float(eval(input_dict['Ms']))
		self.Rs = float(eval(input_dict['Rs']))
		self.ks = float(eval(input_dict['ks']))
		self.Qs = float(eval(input_dict['Qs']))
		self.alphas = float(eval(input_dict['alphas']))
		self.Mc = float(eval(input_dict['Mcore']))
		self.kp = float(eval(input_dict['kpl']))
		self.Qp = float(eval(input_dict['Qpl']))
		self.alphap = float(eval(input_dict['alphapl']))
		try:
			self.fm = float(eval(input_dict['fmantle']))
		except KeyError:
			self.fm = 2./3
		self.Y = float(eval(input_dict['YHe']))
		self.Mpert = float(eval(input_dict['Mpert']))
		self.apert = float(eval(input_dict['pert_sma']))
		self.epert = float(eval(input_dict['pert_ecc']))
		self.ipert = float(eval(input_dict['pert_incl']))
		self.lpert = float(eval(input_dict['pert_lambd']))
		self.opert = float(eval(input_dict['pert_omega']))
		self.Opert = float(eval(input_dict['pert_Omega']))
		self.__lummode = input_dict['stellar_lum']
		if self.__lummode == 'tabular':
			self.__lumpath = input_dict['stellar_lum_path']
		else:
			self.__Lbol = float(eval(input_dict['Lbol']))
			if input_dict['LX_Lbol_sat'] != '':
				self.__LX_Lbol_sat = float(eval(input_dict['LX_Lbol_sat']))
			else:
				self.__LX_Lbol_sat = 10**(np.mean([-4.28, -4.24, -3.67, -3.71, -3.36, -3.35, -3.14]))
			if input_dict['tau_X_bol_sat'] != '':
				self.__tau_X_bol_sat = float(eval(input_dict['tau_X_bol_sat']))
			else:
				self.__tau_X_bol_sat = 10**(np.mean([7.87, 8.35, 7.84, 8.03, 7.9, 8.28, 8.21]))
			if input_dict['alpha_X_bol'] != '':
				self.__alpha_X_bol = float(eval(input_dict['alpha_X_bol']))
			else:
				self.__alpha_X_bol = np.mean([1.22, 1.24, 1.13, 1.28, 1.4, 1.09, 1.18])

		if self.__verbose: print('ok.')

		if npz is None:
			out_path = '{}saved_data/{}/'.format(self.__gpath, self.__out)
			npz_path = os.listdir(out_path)
			npz_path = [out_path + f for f in npz_path if len(f) == len(self.name) + 8 and f.startswith(self.name) and f.endswith('.npz')]
			for f in npz_path:
				try:
					_ = int(f[len(out_path) + len(self.name) + 1:len(out_path) + len(self.name) + 4])
				except TypeError:
					npz_path.remove(f)
		else:
			if type(npz) is str: 
				if os.path.isdir(npz):
					npz = '{}/'.format(npz.rstrip('/'))
					_npz = os.listdir(npz)
					_npz = [npz + f for f in _npz if len(f) == len(self.name) + 8 and f.startswith(self.name) and f.endswith('.npz')]
					for f in _npz:
						try:
							_ = int(f[len(npz) + len(self.name) + 1:len(npz) + len(self.name) + 4])
						except TypeError:
							_npz.remove(f)
					npz = list(_npz)
				else:
					npz = [npz]
			out_path = '{}saved_data/'.format(self.__gpath)
			npz_path = []
			for f in npz:
				if f.startswith('/'):
					npz_path.append(f)
				else:
					npz_path.append('{}saved_data/{}'.format(self.__gpath, f))
		if len(npz_path) == 0: raise FileNotFoundError('No npz file found.')

		npz_path = sorted(npz_path)

		self.t_dyn = []
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
		self.t_atm = []

		if reduced is not None:

			numt = 0

			for i, f in enumerate(npz_path):
				try:
					npz_file = np.load(f, allow_pickle=True)
					numt += len(npz_file['t'])
					if i == 0: mint = npz_file['t'][0]
					if i == len(npz_path) - 1: maxt = npz_file['t'][-1]
				except BadZipFile:
					raise SystemExit('Corrupted zip file: {}'.format(f))

			if numt > reduced:
				self.t = np.linspace(mint, maxt, reduced)
				self.t_dyn = self.t.copy()
				self.t_atm = self.t.copy()
			else:
				reduced = None

		if reduced is None:

			for i, f in enumerate(npz_path):
				try:
					npz_file = np.load(f, allow_pickle=True)

					self.t_dyn += list(npz_file['t'])
					self.h1 += list(npz_file['h1'])
					self.h2 += list(npz_file['h2'])
					self.h3 += list(npz_file['h3'])
					self.e1 += list(npz_file['e1'])
					self.e2 += list(npz_file['e2'])
					self.e3 += list(npz_file['e3'])
					self.Os1 += list(npz_file['Os1'])
					self.Os2 += list(npz_file['Os2'])
					self.Os3 += list(npz_file['Os3'])
					self.Op1 += list(npz_file['Op1'])
					self.Op2 += list(npz_file['Op2'])
					self.Op3 += list(npz_file['Op3'])
					self.e += list(npz_file['e'])
					self.a += list(npz_file['a'])
					self.Mp += list(npz_file['Mp'])
					self.Rp += list(npz_file['Rp'])
					self.t_atm += list(npz_file['t_atmo'])

					if self.__verbose: print('+ Loading npz files... {:d}/{:d}'.format(i + 1, len(npz_path)), end='\r')

				except BadZipFile:
					raise SystemExit('Corrupted zip file: {}'.format(f))

			if self.__verbose: print('+ Loading npz files... {:d}/{:d}'.format(len(npz_path), len(npz_path)))

			if sort:
				if self.__verbose: print('+ Sorting data... ', end='')
				self.h1 = np.asarray([x for _, x in sorted(zip(self.t_dyn, self.h1))])
				self.h2 = np.asarray([x for _, x in sorted(zip(self.t_dyn, self.h2))])
				self.h3 = np.asarray([x for _, x in sorted(zip(self.t_dyn, self.h3))])
				self.e1 = np.asarray([x for _, x in sorted(zip(self.t_dyn, self.e1))])
				self.e2 = np.asarray([x for _, x in sorted(zip(self.t_dyn, self.e2))])
				self.e3 = np.asarray([x for _, x in sorted(zip(self.t_dyn, self.e3))])
				self.Os1 = np.asarray([x for _, x in sorted(zip(self.t_dyn, self.Os1))])
				self.Os2 = np.asarray([x for _, x in sorted(zip(self.t_dyn, self.Os2))])
				self.Os3 = np.asarray([x for _, x in sorted(zip(self.t_dyn, self.Os3))])
				self.Op1 = np.asarray([x for _, x in sorted(zip(self.t_dyn, self.Op1))])
				self.Op2 = np.asarray([x for _, x in sorted(zip(self.t_dyn, self.Op2))])
				self.Op3 = np.asarray([x for _, x in sorted(zip(self.t_dyn, self.Op3))])
				self.e = np.asarray([x for _, x in sorted(zip(self.t_dyn, self.e))])
				self.a = np.asarray([x for _, x in sorted(zip(self.t_dyn, self.a))])
				self.Mp = np.asarray([x for _, x in sorted(zip(self.t_dyn, self.Mp))])
				self.Rp = np.asarray([x for _, x in sorted(zip(self.t_atm, self.Rp))])
				self.t_dyn = np.asarray(list(sorted(self.t_dyn)))
				self.t_atm = np.asarray(list(sorted(self.t_atm)))
				if self.__verbose: print('ok.')
			else:
				self.h1 = np.asarray(self.h1)
				self.h2 = np.asarray(self.h2)
				self.h3 = np.asarray(self.h3)
				self.e1 = np.asarray(self.e1)
				self.e2 = np.asarray(self.e2)
				self.e3 = np.asarray(self.e3)
				self.Os1 = np.asarray(self.Os1)
				self.Os2 = np.asarray(self.Os2)
				self.Os3 = np.asarray(self.Os3)
				self.Op1 = np.asarray(self.Op1)
				self.Op2 = np.asarray(self.Op2)
				self.Op3 = np.asarray(self.Op3)
				self.e = np.asarray(self.e)
				self.a = np.asarray(self.a)
				self.Mp = np.asarray(self.Mp)
				self.Rp = np.asarray(self.Rp)

		else:

			i_max = 0

			for i, f in enumerate(npz_path):

				if self.__verbose: print('+ Reducing npz files... {:d}/{:d}'.format(i + 1, len(npz_path)), end='\r')

				npz_file = np.load(f, allow_pickle=True)
				x_data = npz_file['t']
				i_min = i_max
				i_max = bisect_right(self.t, x_data[-1])
				x_theo = self.t[i_min:i_max]
				y_data = npz_file['h1']
				self.h1 += list(interpol(x_theo, x_data, y_data))
				y_data = npz_file['h2']
				self.h2 += list(interpol(x_theo, x_data, y_data))
				y_data = npz_file['h3']
				self.h3 += list(interpol(x_theo, x_data, y_data))
				y_data = npz_file['e1']
				self.e1 += list(interpol(x_theo, x_data, y_data))
				y_data = npz_file['e2']
				self.e2 += list(interpol(x_theo, x_data, y_data))
				y_data = npz_file['e3']
				self.e3 += list(interpol(x_theo, x_data, y_data))
				y_data = npz_file['Os1']
				self.Os1 += list(interpol(x_theo, x_data, y_data))
				y_data = npz_file['Os2']
				self.Os2 += list(interpol(x_theo, x_data, y_data))
				y_data = npz_file['Os3']
				self.Os3 += list(interpol(x_theo, x_data, y_data))
				y_data = npz_file['Op1']
				self.Op1 += list(interpol(x_theo, x_data, y_data))
				y_data = npz_file['Op2']
				self.Op2 += list(interpol(x_theo, x_data, y_data))
				y_data = npz_file['Op3']
				self.Op3 += list(interpol(x_theo, x_data, y_data))
				y_data = npz_file['e']
				self.e += list(interpol(x_theo, x_data, y_data))
				y_data = npz_file['a']
				self.a += list(interpol(x_theo, x_data, y_data))
				y_data = npz_file['Mp']
				self.Mp += list(interpol(x_theo, x_data, y_data))
				x_data = npz_file['t_atmo']
				y_data = npz_file['Rp']
				self.Rp += list(interpol(x_theo, x_data, y_data))

			if self.__verbose: print('+ Reducing npz files... {:d}/{:d}'.format(len(npz_path), len(npz_path)))

			self.h1 = np.asarray(self.h1)
			self.h2 = np.asarray(self.h2)
			self.h3 = np.asarray(self.h3)
			self.e1 = np.asarray(self.e1)
			self.e2 = np.asarray(self.e2)
			self.e3 = np.asarray(self.e3)
			self.Os1 = np.asarray(self.Os1)
			self.Os2 = np.asarray(self.Os2)
			self.Os3 = np.asarray(self.Os3)
			self.Op1 = np.asarray(self.Op1)
			self.Op2 = np.asarray(self.Op2)
			self.Op3 = np.asarray(self.Op3)
			self.e = np.asarray(self.e)
			self.a = np.asarray(self.a)
			self.Mp = np.asarray(self.Mp)
			self.Rp = np.asarray(self.Rp)

		self.Mp *= morb2jup
		self.Rp *= dorb2jup 
		self.finish = (self.t_dyn[-1] >= self.__age)

		if self.__verbose:
			tf = tm.time()
			dt = tf - ti
			if dt < 1:
				print('+ Elapsed time: {:.2f} ms.'.format(1000*dt))
			else:
				print('+ Elapsed time: {:.2f} s.'.format(dt))
			print('----------------------')

	
	def set_time(self, ti=None, tf=None, npoint=None, points=None):

		'''
		Sets the times at which all parameters are defined.

		Input:
		+++ ti (float): start of the time range (yr). If not provided, takes t_dyn[0].
		+++ tf (float): end of the time range (yr). If not provided, takes t_dyn[-1]
		+++ npoint (integer): number of points within the time range. If not provided, takes len(t_dyn).
		+++ points (sequence): user-defined time points (yr). If defined, overrides ti, tf and npoint. 

		Creates a new class attribute:
		+++ t (numpy array): time variable for all parameters (yr).
		'''

		if self.__verbose:
			print('----------------------')
			print('+ Setting a uniform time variable...', end='')
			tii = tm.time()

		if points is None:
			if ti is None: ti = self.t_dyn[0]
			if tf is None: tf = self.t_dyn[-1]
			if npoint is None: npoint = len(self.t_dyn)	
			self.t = np.linspace(ti, tf, npoint)
		else:
			self.t = np.asarray(points)

		self.h1 = interpol(self.t, self.t_dyn, self.h1)
		self.h2 = interpol(self.t, self.t_dyn, self.h2)
		self.h3 = interpol(self.t, self.t_dyn, self.h3)
		self.e1 = interpol(self.t, self.t_dyn, self.e1)
		self.e2 = interpol(self.t, self.t_dyn, self.e2)
		self.e3 = interpol(self.t, self.t_dyn, self.e3)
		self.Os1 = interpol(self.t, self.t_dyn, self.Os1)
		self.Os2 = interpol(self.t, self.t_dyn, self.Os2)
		self.Os3 = interpol(self.t, self.t_dyn, self.Os3)
		self.Op1 = interpol(self.t, self.t_dyn, self.Op1)
		self.Op2 = interpol(self.t, self.t_dyn, self.Op2)
		self.Op3 = interpol(self.t, self.t_dyn, self.Op3)
		self.e = interpol(self.t, self.t_dyn, self.e)
		self.a = interpol(self.t, self.t_dyn, self.a)
		self.Mp = interpol(self.t, self.t_dyn, self.Mp)
		self.Rp = interpol(self.t, self.t_atm, self.Rp)

		if self.__verbose:
			print(' ok.')
			tff = tm.time()
			dt = (tff - tii)*1000
			if dt < 1000:
				print('+ Elapsed time: ' + '{:.2f}'.format(dt) + ' ms.')
			else:
				print('+ Elapsed time: ' + '{:.2f}'.format(1e-3*dt) + ' s.')
			print('----------------------')

	
	def generate(self, *args):

		'''
		Generates various parameters of the simulation as class attributes.
		For each argument, the generated attribute has the same name as the argument.
		Must be used after set_time.

		Input:
		+++ *args (strings): parameters to be generated. Allowed arguments are:
		------ 'P': inner's orbital period (yr).
		------ 'p': inner's periastron distance (AU).
		------ 'q': inner's apoastron distance (AU).
		------ 'o': inner's argument of periastron (degrees).
		------ 'ip': inner's inclination w.r.t. l.o.s. (degrees).
		------ 'psi': inner's 3D spin-orbit angle (degrees).
		------ 'lambd': inner's sky-projected spin-orbit angle (degrees).
		------ 'Op': inner's spin (yr-1).
		------ 'Menv': inner's envelope mass (Mjup).
		------ 'eps': photo-evaporation efficiency.
		------ 'RXUV': inner's XUV radius (Rjup).
		------ 'Tsurf': inner's surface temperature (K).
		------ 'Teq': inner's equilibrium temperature (K).
		------ 'Tint': inner's internal temperature (K)
		------ 'Lp': inner's intrinsic luminosity (erg/s).
		------ 'massloss': inner's mass-loss rate (g/s).
		------ 'rho': inner's mean density (g/cm3).
		------ 'tau_kozai': Kozai-Lidov timescale (yr).
		------ 'tau_tide': inner's tidal timescale (yr).
		------ 'tau_evap': photo-evaporation timescale (yr).
		------ 'imut': inner-outer mutual inclination (degrees).
		------ 'Os': stellar spin (yr-1).
		------ 'istar': stellar spin inclination w.r.t. l.o.s. (degrees).
		------ 'Lbol': stellar bolometric luminosity (erg/s).
		------ 'LXUV': stellar XUV luminosity (erg/s).
		------ 'Jerr': angular momentum relative error.
		------ 'logrpert': log(apert**3/Mpert)
		'''

		for arg in args: 
			if arg not in ['P', 'p', 'q', 'o', 'ip', 'psi', 'lambd', 'Op', 'Menv', 'eps', 'RXUV', 'Tsurf', 'Lp', 'Teq', 'Tint',
						   'massloss', 'rho', 'tau_kozai', 'tau_tide', 'tau_evap', 'imut', 'Os', 'istar', 'Lbol', 'LXUV', 'Jerr', 'logrpert']:
				raise SystemExit('Invalid argument(s).')

		if self.__verbose:
			print('----------------------')
			print('+ Generating ' + str(len(args)) + ' parameter(s)... ', end='')
			ti = tm.time()

		if 'P' in args or 'tau_kozai' in args:
			self.P = 2*np.pi/np.sqrt((self.Ms/morb2sun + self.Mp/morb2jup)/self.a**3)

		if 'p' in args: 
			self.p = self.a*(1. - self.e)

		if 'q' in args: 
			self.q = self.a*(1. + self.e)

		if 'o' in args:
			coso = clean_cos(self.e2*self.h1 - self.e1*self.h2)
			self.o = np.arccos(coso)*180./np.pi

		if 'ip' in args or 'lambd' in args or 'psi' in args or 'imut' in args:
			h = np.sqrt(self.h1**2 + self.h2**2 + self.h3**2)

		if 'ip' in args or 'lambd' in args:
			cosip = clean_cos(self.h3/h)
			self.ip = np.arccos(cosip)*180./np.pi

		if 'Os' in args or 'psi' in args or 'istar' in args or 'lambd' in args:
			self.Os = np.sqrt(self.Os1**2 + self.Os2**2 + self.Os3**2)

		if 'psi' in args or 'lambd' in args: 
			cosp = clean_cos((self.h1*self.Os1 + self.h2*self.Os2 + self.h3*self.Os3)/(h*self.Os))
			self.psi = np.arccos(cosp)*180./np.pi

		if 'istar' in args or 'lambd' in args:
			cosis = clean_cos(self.Os3/self.Os)
			self.istar = np.arccos(cosis)

		if 'lambd' in args:
			if (np.sin(self.istar) == 0.).any(): print('/!\\ Warning: bad values for lambda (sin istar = 0).')
			if (np.sin(self.ip) == 0.).any(): print('/!\\ Warning: bad values for lambda (sin ip = 0).')
			cosl = clean_cos((np.cos(self.psi) - np.cos(self.istar)*np.cos(self.ip))/(np.sin(self.istar)*np.sin(self.ip)))
			self.lambd = np.arccos(cosl)*180./np.pi

		if 'Op' in args:
			self.Op = np.sqrt(self.Op1**2 + self.Op2**2 + self.Op3**2)

		if 'Menv' in args or 'Tsurf' in args or 'Tint' in args or 'Lp' in args or 'tau_evap' in args:
			self.Menv = self.Mp - self.Mc

		if 'eps' in args or 'tau_evap' in args:
			self.eps = np.zeros(len(self.t))
			for i in range(len(self.t)):
				phi = np.log10(((const.G).decompose(bases=u.cgs.bases)).value*self.Mp[i]*mjup2cgs/(self.Rp[i]*djup2cgs))
				if phi > 13.11:
					self.eps[i] = 10.**(-.98 - 7.29*(phi - 13.11))
				else:
					self.eps[i] = 10.**(-.5  - 0.44*(phi - 12.  ))

		if 'Lbol' in args or 'LXUV' in args or 'RXUV' in args or 'Tsurf' in args or 'Teq' in args or 'tau_evap' in args:
			if self.__lummode == 'tabular':
				t_tab = []
				Lbol_tab = []
				LXUV_tab = []
				lumpath = self.__gpath + 'luminosities/' + self.__lumpath
				for line in open(lumpath, 'r'):
					temp = line.rstrip().split()
					if temp[0] == '#':
						continue
					elif len(temp) == 9:
						_t = float(eval(temp[0]))
						_Lbol = float(eval(temp[5]))
						_LXUV = float(eval(temp[7]))
						t_tab.append(_t)
						Lbol_tab.append(_Lbol)
						LXUV_tab.append(_LXUV)
					else:
						continue
				t_tab = np.asarray(t_tab)
				Lbol_tab = np.asarray(Lbol_tab)
				LXUV_tab = np.asarray(LXUV_tab)
				self.Lbol = interpol(self.t, t_tab, Lbol_tab)			
				self.LXUV = interpol(self.t, t_tab, LXUV_tab)
			else:
				def LXUV(t, a):
					if t <= self.__tau_X_bol_sat:
						ratio = self.__LX_Lbol_sat
					else:
						ratio = self.__LX_Lbol_sat*(t/self.__tau_X_bol_sat)**(-self.__alpha_X_bol)
					LX = ratio*self.__Lbol			
					gamma = -0.45
					alpha = 650
					FX = LX/(4*np.pi*(a*dorb2cgs)**2)
					ratio = alpha*(FX**gamma)
					LEUV = ratio*LX
					_LXUV = LX + LEUV		   		
					return _LXUV
				self.Lbol = np.ones(len(self.t))*self.__Lbol
				self.LXUV = np.asarray([LXUV(self.t[i], self.a[i]) for i in range(len(self.t))]) 

		if 'RXUV' in args or 'tau_evap' in args:
			self.RXUV = np.zeros(len(self.t))
			for i in range(len(self.t)):
				phi = np.log10(((const.G).decompose(bases=u.cgs.bases)).value*self.Mp[i]*mjup2cgs/(self.Rp[i]*djup2cgs))
				FXUV = np.log10(self.LXUV[i]/(4*np.pi*self.a[i]*dorb2cgs))
				ratio = 10**np.max((0., -0.185*phi + 0.021*FXUV + 2.42))
				self.RXUV[i] = self.Rp[i]*ratio

		if 'Lp' in args or 'Tsurf' in args or 'Tint' in args:
			Ljup = 8.67e-10*3.9e33
			Mearth = 1./317.8
			t_tab  = np.array([0.1e9, 0.3e9, 0.5e9, 0.8e9, 1e9, 2e9, 3e9, 4e9, 5e9, 6e9, 7e9, 8e9, 9e9, 10e9])
			a0_tab = np.array([0.00252906, 0.00121324, 0.000707416, 0.000423376, 0.000352187, 0.000175775, 0.00011412, 8.81462e-5,
							   6.91819e-5, 5.49615e-5, 4.5032e-5, 3.80363e-5, 3.30102e-5, 2.92937e-5])
			b1_tab = np.array([-0.002002380, -0.000533601, -0.000394131, -0.000187283, -0.000141480, -4.07832e-5, -2.09944e-5, -2.07673e-5,
							   -1.90159e-5, -1.68620e-5, -1.51951e-5, -1.40113e-5, -1.31146e-5, -1.24023e-5])
			b2_tab = np.array([0.001044080, 0.000360703, 0.000212475, 0.000125872, 9.94382e-5, 4.58530e-5, 2.91169e-5, 2.12932e-5,
							   1.62128e-5, 1.29045e-5, 1.05948e-5, 8.93639e-6, 7.69121e-6, 6.73922e-6])
			c1_tab = np.array([.05864850, .02141140, .01381380, .00887292, .00718831, .00357941, .00232693, .00171412, .00134355, .00109019,
							   .00091005, .00077687, .000675243, .000595191])
			c2_tab = np.array([0.000967878, 0.000368533, 0.000189456, 0.000117141, 9.20563e-5, 5.52851e-5, 4.00546e-5, 2.90984e-5, 2.30387e-5,
							   1.96163e-5, 1.70934e-5, 1.50107e-5, 1.32482-5, 1.17809e-5])
			a0 = interpol(self.t, t_tab, a0_tab)
			b1 = interpol(self.t, t_tab, b1_tab)
			b2 = interpol(self.t, t_tab, b2_tab)
			c1 = interpol(self.t, t_tab, c1_tab)
			c2 = interpol(self.t, t_tab, c2_tab)
			self.Lp = Ljup*(a0 + b1*self.Mc/Mearth + b2*(self.Mc/Mearth)**2 + c1*self.Menv/Mearth + c2*(self.Menv/Mearth)**2)
			
		if 'Tint' in args or 'Tsurf' in args:
			sb = (const.sigma_sb.decompose(bases=u.cgs.bases)).value
			Ti4 = self.Lp/(4*np.pi*sb*(self.Rp*djup2cgs)**2)
			self.Tint = Ti4**.25

		if 'Tsurf' in args or 'Teq' in args:
			sb = (const.sigma_sb.decompose(bases=u.cgs.bases)).value
			Te4 = self.Lbol/(16*np.pi*sb*np.sqrt(1. - self.e**2)*(self.a*dorb2cgs)**2)
			self.Teq = Te4**.25

		if 'Tsurf' in args:
			tau = 2./3
			T_tab = [260., 388., 584., 861., 1267., 1460., 1577., 1730., 1870., 2015., 2255., 2777.]
			g_tab = [0.005, 0.008, 0.027, 0.07, 0.18, 0.19, 0.18, 0.185, 0.2, 0.22, 0.31, 0.55]
			g = interpol(Te4**.25, T_tab, g_tab)
			T4 = (3.*Ti4/4)*(2./3 + tau) + (3.*Te4/4)*(2./3 + (2./(3*g))*(1. + (g*tau/2. - 1.)*np.exp(-g*tau)) + (2.*g/3)*(1. - tau**2/2)*expn(2, g*tau))
			self.Tsurf = T4**.25

		if 'massloss' in args:
			massloss = [np.abs(self.Mp[i + 1] - self.Mp[i])/(self.t[i + 1] - self.t[i]) for i in range(len(self.t) - 1)]
			massloss.append(massloss[-1])
			massloss = np.asarray(massloss)
			massloss *= mjup2cgs/torb2cgs
			self.massloss = massloss

		if 'rho' in args:
			self.rho = self.Mp*mjup2cgs/((4./3)*np.pi*(self.Rp*djup2cgs)**3)

		if 'tau_kozai' in args:
			Ppert = 2*np.pi/np.sqrt((self.Ms/morb2sun + self.Mp/morb2jup + self.Mpert/morb2jup)/self.apert**3)
			self.tau_kozai = (2*Ppert**2/(3*np.pi*self.P))*((self.Ms/morb2sun + self.Mp/morb2jup + self.Mpert/morb2jup)/(self.Mpert/morb2jup))*(1. - self.epert**2)**(3/2)

		if 'tau_tide' in args:
			self.tau_tide = (4./63)*(self.a**6.5/(self.Ms/morb2sun)**1.5)*self.Qp*(self.Mp/morb2jup)/(self.Rp/dorb2jup)**5

		if 'tau_evap' in args:
			ksi = (self.a/(self.Rp/dorb2jup))*(1 + .5*self.e**2)*(self.Mp*morb2sun/(3*self.Ms*morb2jup))**(1/3)
			Ktide = 1 - 3/(2*ksi) + 1/(2*ksi**3)
			eta = 1.
			self.tau_evap = eta*(self.Menv*self.Mp/morb2jup**2)*Ktide*np.sqrt(1. - self.e**2)*self.a**2/(self.eps*np.pi*(self.LXUV/Lorb2cgs)*(self.Rp*self.RXUV**2/dorb2jup**3))

		if 'imut' in args or 'Jerr' in args:
			orb_pert = [self.apert, self.lpert, 
						self.epert*np.cos(self.opert*np.pi/180), self.epert*np.sin(self.opert*np.pi/180), 
						np.sin(self.ipert*np.pi/360)*np.cos(self.Opert*np.pi/180), 
						np.sin(self.ipert*np.pi/360)*np.sin(self.Opert*np.pi/180)]
			cmu_pert = self.Ms/morb2sun + self.Mp[0]/morb2jup + self.Mpert/morb2jup
			R, R_dot = ell2cart(orb_pert, cmu_pert)
			self.__H1, self.__H2, self.__H3 = np.cross(R, R_dot)

		if 'imut' in args:
			H = np.sqrt(self.__H1**2 + self.__H2**2 + self.__H3**2)
			cosim = clean_cos((self.h1*self.__H1 + self.h2*self.__H2 + self.h3*self.__H3)/(h*H))
			self.imut = np.arccos(cosim)*180./np.pi

		if 'Jerr' in args:
			h = np.sqrt((self.Ms/morb2sun + self.Mp/morb2jup)*self.a*(1. - self.e**2))
			Is = self.alphas*(self.Ms/morb2sun)*(self.Rs/dorb2sun)**2
			Ip = self.alphap*(self.Mp/morb2jup)*(self.Rp/dorb2jup)**2
			J1 = ((self.Ms/morb2sun)*(self.Mp/morb2jup)/(self.Ms/morb2sun + self.Mp/morb2jup))*h*self.h1 + \
				 ((self.Ms/morb2sun + self.Mp/morb2jup)*(self.Mpert/morb2jup)/(self.Ms/morb2sun + self.Mp/morb2jup + self.Mpert/morb2jup))*self.__H1 + \
				 Is*self.Os1 + Ip*self.Op1
			J2 = ((self.Ms/morb2sun)*(self.Mp/morb2jup)/(self.Ms/morb2sun + self.Mp/morb2jup))*h*self.h2 + \
				 ((self.Ms/morb2sun + self.Mp/morb2jup)*(self.Mpert/morb2jup)/(self.Ms/morb2sun + self.Mp/morb2jup + self.Mpert/morb2jup))*self.__H2 + \
				 Is*self.Os2 + Ip*self.Op2
			J3 = ((self.Ms/morb2sun)*(self.Mp/morb2jup)/(self.Ms/morb2sun + self.Mp/morb2jup))*h*self.h3 + \
				 ((self.Ms/morb2sun + self.Mp/morb2jup)*(self.Mpert/morb2jup)/(self.Ms/morb2sun + self.Mp/morb2jup + self.Mpert/morb2jup))*self.__H3 + \
				 Is*self.Os3 + Ip*self.Op3
			J = np.sqrt(J1**2 + J2**2 + J3**2)
			Jrel = [np.abs(J[i + 1] - J[i])/J[i] for i in range(len(self.t) - 1)]
			Jrel.append(Jrel[-1])
			self.Jerr = np.asarray(Jrel)

		if 'logrpert' in args:
			self.logrpert = np.log(self.apert**3/self.Mpert)

		if self.__verbose:
			print(' ok.')
			tf = tm.time()
			dt = (tf - ti)*1000
			if dt < 1000:
				print('+ Elapsed time: ' + '{:.2f}'.format(dt) + ' ms.')
			else:
				print('+ Elapsed time: ' + '{:.2f}'.format(1e-3*dt) + ' s.')
			print('----------------------')


	def atmo_profiles(self, t=None):

		'''
		Returns the atmospheric profiles at a given time.

		Input:
		+++ t (float): requested time (yr). If not provided, uses the last atmospheric time.

		Output:
		+++ tau_prof (numpy array): optical depth profile (only in Zone A).
		+++ m_prof (numpy array): mass (Mjup) profile.
		+++ r_prof (numpy array): radius (Rjup) profile.
		+++ T_prof (numpy array): temperature (K) profile (only in Zone A and Zone B).
		+++ P_prof (numpy array): pressure (bar) profile.
		'''

		if self.__verbose:
			ti = tm.time()
			print('----------------------')
			print('+ Initializing atmospheric profiles util...', end='')

		import shutil
		import tempfile
		import sys
		sys.path.insert(1, self.__gpath + 'routines/')
		from routines import JADE_Simulation, read_input_file
		from basic_functions import matm2orb, Tatm2cgs, Patm2cgs
		from interpolation import Opacities

		lines = []
		for line in open(self.__ipath, 'r'):
			if line.startswith('lazy_init'):
				line = 'lazy_init = True'
			line += '\n'
			lines.append(line)

		temp = tempfile.NamedTemporaryFile(mode='wt', dir=self.__gpath)
		try:
			with open(temp.name, 'w') as _temp:
				for line in lines:
					_temp.write(line)
			ipath = temp.name.split('/')[-1]
			sim = JADE_Simulation(self.__gpath, ipath, verbose=False)
		finally:
			temp.close()

		if t is not None:
			if 't' in self.__get_attributes():
				idx = (np.abs(np.asarray(self.t) - t)).argmin()
				idx_atm, idx_dyn = idx, idx
			else:
				idx_atm = (np.abs(np.asarray(self.t_atm) - t)).argmin()
				idx_dyn = (np.abs(np.asarray(self.t_dyn) - t)).argmin()
		else:
			idx_atm = -1
			idx_dyn = (np.abs(np.asarray(self.t_dyn) - self.t_atm[idx_atm])).argmin()

		t = self.t_atm[idx_atm]

		if self.__verbose:
			print(' ok.')
			print('+ Generating atmospheric profiles at t = {:.2e} yr...'.format(t))

		sim.Mp = np.array([self.Mp[idx_dyn]/morb2jup])
		sim.Rp = np.array([self.Rp[idx_atm]/dorb2jup])
		sim.a = np.array([self.a[idx_dyn]])
		sim.e = np.array([self.e[idx_dyn]])
		sim.Lp = sim.planet_luminosity(t)
		opacity_tables = Opacities(sim.YHe, sim.Zmet, self.__gpath)
		sim.kappa = opacity_tables.kappa

		tau_prof, m_prof, r_prof, T_prof, P_prof = sim.atmospheric_structure(t, sim.Rp[0], sim.Mp[0], sim.a[0], sim.e[0], plot=True, verbose=self.__verbose)

		tau_prof = np.asarray(tau_prof)
		m_prof = np.asarray(m_prof)*matm2orb*morb2jup
		r_prof = np.asarray(r_prof)
		T_prof = np.asarray(T_prof)*Tatm2cgs
		P_prof = np.asarray(P_prof)*Patm2cgs*1e-6

		if self.__verbose:
			tf = tm.time()
			dt = (tf - ti)*1000
			if dt < 1000:
				print('+ Elapsed time: ' + '{:.2f}'.format(dt) + ' ms.')
			else:
				print('+ Elapsed time: ' + '{:.2f}'.format(1e-3*dt) + ' s.')
			print('----------------------')

		return tau_prof, m_prof, r_prof, T_prof, P_prof



	def __get_attributes(self):

		'''
		Returns the constructed attributes of the class as a list.
		'''

		return [i for i in self.__dict__.keys() if i[:1] != '_']





def read_input(path):

	'''
	Function that reads a JADE input text file and returns a dictionary with the various fields.

	Input: 
	+++ path (string): path to input text file.

	Output: 
	+++ input_dict (dictionary): all input fields as strings.
	'''

	input_dict = {}
	
	for line in open(path, 'r'):
		temp = line.rstrip().split()

		if len(temp) < 1:
			continue

		if '#' in temp[0]:
			continue

		key = temp[0]
		if len(temp) > 2:
			value = temp[2]
			if len(temp) > 3:
				value = list(temp[2:])
		else:
			value = ''

		input_dict[key] = value

	return input_dict


def interpol(xth, xdata, ydata, kind='nearest'):

	'''
	Function that interpolates 1D datapoints.

	Input:
	+++ xth (numpy array): 1D values for which the data has to be interpolated.
	+++ xdata (numpy array): 1D original abscissa data points. Must be sorted in ascending order.
	+++ ydata (numpy array): 1D original ordinate data points.

	Output:
	+++ yth (numpy array): interpolated 1D values at xth.
	'''

	if len(xdata) == 1:
		yth = np.ones(len(xth))*ydata

	else:
		finterpol = interpolate.interp1d(xdata, ydata, kind=kind, bounds_error=False, fill_value=(ydata[0], ydata[-1]))		
		yth = finterpol(np.asarray(xth))
	
	return yth


def clean_cos(x, tol=1e-5):

	'''
	Function that avoids unnecessary NaNs when dealing with trigonometric functions.

	Input:
	+++ x (numpy array): argument of a arcsin/arccos.
	
	Output:
	+++ y (numpy array): cleaned x.
	'''

	c1 = (np.abs(x - 1) < tol) & (x > 1)
	c2 = (np.abs(x + 1) < tol) & (x < -1)
	x[c1] = 1.
	x[c2] = -1.
	return x


def ell2cart(ell, cmu):

	'''
	Function that converts elliptical to Cartesian coordinates.
	See routines/chgcoord.py for documentation.
	'''

	x = np.empty(3)
	xp = np.empty(3)
	tx1 = np.empty(2)
	tx1t = np.empty(2)
	rot = np.empty((3,2))

	a=ell[0]
	l=ell[1]
	k=ell[2]
	h=ell[3]
	q=ell[4]
	p=ell[5]
	na=np.sqrt(cmu/a)
	phi=np.sqrt(1.0-k**2-h**2)
	ki =np.sqrt(1.0-q**2-p**2)

	rot[0,0]=1.0-2.0*p**2
	rot[0,1]=2.0*p*q
	rot[1,0]=2.0*p*q
	rot[1,1]=1.0-2.0*q**2
	rot[2,0]=-2.0*p*ki
	rot[2,1]= 2.0*q*ki

	f=keplkh2(l,k,h)
	sf	=np.sin(f)
	cf	=np.cos(f)
	umrsa = k*cf+h*sf
	psilmf   = (-k*sf+h*cf)/(1.0+phi)
	psiumrsa =		umrsa/(1.0+phi)
	na2sr	= na/(1.0-umrsa)

	tx1[0] =a*(cf- psilmf*h-k)
	tx1[1] =a*(sf+ psilmf*k-h)
	tx1t[0]=na2sr*(-sf+psiumrsa*h)
	tx1t[1]=na2sr*( cf-psiumrsa*k)

	x[0]  =rot[0,0]*tx1[0]  + rot[0,1]*tx1[1]
	x[1]  =rot[1,0]*tx1[0]  + rot[1,1]*tx1[1]
	x[2]  =rot[2,0]*tx1[0]  + rot[2,1]*tx1[1]
	xp[0] =rot[0,0]*tx1t[0] + rot[0,1]*tx1t[1]
	xp[1] =rot[1,0]*tx1t[0] + rot[1,1]*tx1t[1]
	xp[2] =rot[2,0]*tx1t[0] + rot[2,1]*tx1t[1]

	return(x,xp)


def keplkh2(l, k, h):

	'''
	Auxiliary function for ell2cart.
	'''

	eps = 2*2.26e-16
	imax = 20
	a=l
	ca=np.cos(a)
	sa=np.sin(a)
	se=k*sa-h*ca
	ce=k*ca+h*sa
	fa=a-se-l
	f1a=1.0-ce
	f2a=se/2.0
	f3a=ce/6.0
	d1=-fa/f1a
	d2=-fa/(f1a-d1*f2a)
	d3 =-fa/(f1a+d2*(f2a+d2*f3a))
	a=a+d3
	ca=np.cos(a)
	sa=np.sin(a)
	se=k*sa-h*ca
	ce=k*ca+h*sa
	fa=a-se-l
	f1a=1.0-ce
	f2a=se/2.0
	f3a=ce/6.0
	f4a=-se/24.0
	f5a=-ce/120.0
	d1=-fa/f1a
	d2=-fa/(f1a-d1*f2a)
	d3=-fa/(f1a+d2*(f2a+d2*f3a))
	d4=-fa/(f1a+d3*(f2a+d3*(f3a+d3*f4a)))
	d5=-fa/( f1a+d4*(f2a+d4*(f3a+d4*(f4a+d4*f5a))))
	a=a+d5
	i=0
	while True:
		i=i+1
		ca=np.cos(a)
		sa=np.sin(a)
		se=k*sa-h*ca
		fa=a-se-l
		ce=k*ca+h*sa
		f1a=1.0-ce
		d1=-fa/f1a
		if (np.abs(d1)/max(1.0,np.abs(a)) > eps):
			if (i > imax):
			  return(a)
			a=a+d1
		else:	
			return(a)
