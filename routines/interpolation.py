#--------------------------------------------------------------------------------------------
#--- Common file for dealing with all interpolated quantities in JADE simulations.
#--------------------------------------------------------------------------------------------
#--- Author: Mara Attia
#--------------------------------------------------------------------------------------------

import numpy as np
import os
from scipy import interpolate
from bisect import bisect_left
from functools import partial

from basic_functions import heaviside2, closest, clean_string, create_dir

 
#Atomic mass of Hydrogen and Helium in grams
mH = 1.007975*1.660531e-24
mHe = 4.002602*1.660531e-24

#Boltzmann constant in CGS
k_boltz_cgs=1.380649e-16


# EOS: Saumon et al. 1995, Broeg 2009
# This class approximates H/He EOS for fast computations
#----------------------------------------------------------------

class SaumonApprox:

	def __init__(self, path, n_Y=11):
		"""
		Initialize interpolator for irregular H-He EOS grids
		
		Parameters:
		-----------
		path : str
			Path to EOS data files
		n_Y : int
			Number of Y points to pre-compute
		"""
		# Store paths
		self.h_path = path + 'EOS/h_tab_i.dat'
		self.he_path = path + 'EOS/he_tab_i.dat'
		
		# Create Y grid
		self.Y_values = np.linspace(0, 1, n_Y)
		
		# Initialize interpolators dictionary
		# We'll create them lazily to save memory
		self.interp_cache = {}
		
	def read_eos_file(self, filepath):
		"""Read EOS file returning T, P and values for each T block"""
		T_blocks = []
		P_blocks = []
		rho_blocks = []
		adg_blocks = []
		
		current_P = []
		current_rho = []
		current_adg = []
		
		with open(filepath, 'r') as f:
			for line in f:
				values = line.strip().split()
				if len(values) == 2:
					# New temperature block
					if current_P:
						P_blocks.append(np.array(current_P))
						rho_blocks.append(np.array(current_rho))
						adg_blocks.append(np.array(current_adg))
						current_P = []
						current_rho = []
						current_adg = []
					T_blocks.append(float(values[0]))
				else:
					current_P.append(float(values[0]))
					current_rho.append(float(values[3]))
					current_adg.append(float(values[10]))
			
			# Don't forget last block
			if current_P:
				P_blocks.append(np.array(current_P))
				rho_blocks.append(np.array(current_rho))
				adg_blocks.append(np.array(current_adg))
		
		return np.array(T_blocks), P_blocks, rho_blocks, adg_blocks
	
	def create_interpolator_for_Y(self, Y):
		"""Create 2D interpolators for a specific Y value"""
		# Read pure component data
		T_H, P_H, rho_H, adg_H = self.read_eos_file(self.h_path)
		T_He, P_He, rho_He, adg_He = self.read_eos_file(self.he_path)
		
		# Check temperatures match
		if not np.array_equal(T_H, T_He):
			raise ValueError("Temperature grids don't match between H and He files")
		
		# Calculate mixture properties for each T block
		T_points = []
		P_points = []
		rho_points = []
		adg_points = []
		
		for i in range(len(T_H)):
			# Get values for this temperature
			P_h = P_H[i]
			P_he = P_He[i]
			
			# Find common P values (they should be the same)
			if not np.array_equal(P_h, P_he):
				raise ValueError(f"Pressure grids don't match at T={T_H[i]}")
			
			# Calculate mixture properties
			rho_h = 10**rho_H[i]
			rho_he = 10**rho_He[i]
			rho_mix = ((1. - Y)/rho_h + Y/rho_he)**-1
			adg_mix = (1. - Y)*adg_H[i] + Y*adg_He[i]
			
			# Store points
			n_points = len(P_h)
			T_points.extend([T_H[i]] * n_points)
			P_points.extend(P_h)
			rho_points.extend(np.log10(rho_mix))
			adg_points.extend(adg_mix)
		
		# Create interpolators
		# Note: T and P values are already in log10 scale from the EOS files
		T_log_arr = np.array(T_points)
		P_log_arr = np.array(P_points)
		
		# Add tiny jitter to coordinates to avoid duplicate points
		# This prevents IndexError in NearestNDInterpolator
		coords = np.column_stack([T_log_arr, P_log_arr])
		unique_coords, indices = np.unique(coords, axis=0, return_index=True)
		
		if len(unique_coords) < len(coords):
			# If there are duplicate points, use only unique ones
			unique_rho = np.array(rho_points)[indices]
			unique_adg = np.array(adg_points)[indices]
			
			# Create primary interpolators with unique points
			rho_interp_linear = interpolate.LinearNDInterpolator(
				unique_coords, 
				unique_rho
			)
			
			adg_interp_linear = interpolate.LinearNDInterpolator(
				unique_coords, 
				unique_adg
			)
			
			# Create nearest-neighbor interpolators with unique points
			rho_interp_nearest = interpolate.NearestNDInterpolator(
				unique_coords, 
				unique_rho
			)
			
			adg_interp_nearest = interpolate.NearestNDInterpolator(
				unique_coords, 
				unique_adg
			)
		else:
			# No duplicates, proceed normally
			rho_interp_linear = interpolate.LinearNDInterpolator(
				coords, 
				rho_points
			)
			
			adg_interp_linear = interpolate.LinearNDInterpolator(
				coords, 
				adg_points
			)
			
			rho_interp_nearest = interpolate.NearestNDInterpolator(
				coords, 
				rho_points
			)
			
			adg_interp_nearest = interpolate.NearestNDInterpolator(
				coords, 
				adg_points
			)
		
		# Create wrapper functions that use linear interp with nearest extrapolation
		def rho_interp(T_log, P_log):
			result = rho_interp_linear(T_log, P_log)
			if np.isnan(result):
				result = rho_interp_nearest(T_log, P_log)
			return result
			
		def adg_interp(T_log, P_log):
			result = adg_interp_linear(T_log, P_log)
			if np.isnan(result):
				result = adg_interp_nearest(T_log, P_log)
			return result
		
		return rho_interp, adg_interp, T_log_arr, P_log_arr, rho_points, adg_points
	
	def get_eos_functions(self, Y):
		"""Get EOS functions for a specific Y value, with caching"""
		# Round Y to nearest pre-computed value
		Y_idx = np.abs(self.Y_values - Y).argmin()
		Y_key = self.Y_values[Y_idx]
		
		# Check if we have cached interpolators
		if Y_key not in self.interp_cache:
			# Create and cache interpolators
			self.interp_cache[Y_key] = self.create_interpolator_for_Y(Y_key)
			
			# Clear cache if it's too large
			if len(self.interp_cache) > 3:  # Keep only 3 Y values in memory
				# Remove least recently used (first key that's not Y_key)
				for k in list(self.interp_cache.keys()):
					if k != Y_key:
						del self.interp_cache[k]
						break
		
		rho_interp, adg_interp, T_log_arr, P_log_arr, rho_points, adg_points = self.interp_cache[Y_key]
		
		def rho(T, P):
			# T and P should be in normal scale, we convert to log10
			T_log = np.log10(T)
			P_log = np.log10(P)
			
			try:
				result = rho_interp(T_log, P_log)
				return 10**float(result)
			except IndexError:
				# Fallback to the nearest point in case of errors
				# Find closest point to (T_log, P_log) in the coordinates
				coords = np.column_stack([T_log_arr, P_log_arr])
				dists = np.sum((coords - np.array([T_log, P_log]))**2, axis=1)
				nearest_idx = np.argmin(dists)
				return 10**float(rho_points[nearest_idx])
		
		def adg(T, P):
			# T and P should be in normal scale, we convert to log10
			T_log = np.log10(T)
			P_log = np.log10(P)
			
			try:
				result = adg_interp(T_log, P_log)
				return float(result)
			except IndexError:
				# Fallback to the nearest point in case of errors
				coords = np.column_stack([T_log_arr, P_log_arr])
				dists = np.sum((coords - np.array([T_log, P_log]))**2, axis=1)
				nearest_idx = np.argmin(dists)
				return float(adg_points[nearest_idx])
		
		return rho, adg

	
# EOS: Saumon et al. 1995, Broeg 2009
# This class is used to generate the EOS file
#----------------------------------------------------------------
class Saumon:

	class SaumonBase:
		
		def __init__(self, path):
			self.kind = 'Rbf'
			self.T = []
			self.P = []
			self.X2 = []
			self.X = []
			self.rho = []
			self.S = []
			self.ST = []
			self.SP = []
			self.adg = []
			self.X2_func = None
			self.X_func = None
			self.rho_func = None
			self.S_func = None
			self.ST_func = None
			self.SP_func = None
			self.adg_func = None
			self.read_dat(path)
			self.generate_func()
		
		def read_dat(self, path):
			for line in open(path, 'r'):
				temp = line.rstrip().split()
				if len(temp) == 2:
					temperature = float(temp[0])
				else:
					self.T.append(temperature)
					self.P.append(float(temp[0]))
					self.X2.append(float(temp[1]))
					self.X.append(float(temp[2]))
					self.rho.append(float(temp[3]))
					self.S.append(float(temp[4]))
					self.ST.append(float(temp[8]))
					self.SP.append(float(temp[9]))
					self.adg.append(float(temp[10]))
					
		def generate_func(self):
			if self.kind == 'Rbf':
				self.X2_func = interpolate.Rbf(self.T, self.P, self.X2)
				self.X_func = interpolate.Rbf(self.T, self.P, self.X)
				self.rho_func = interpolate.Rbf(self.T, self.P, self.rho)
				self.S_func = interpolate.Rbf(self.T, self.P, self.S)
				self.ST_func = interpolate.Rbf(self.T, self.P, self.ST)
				self.SP_func = interpolate.Rbf(self.T, self.P, self.SP)
				self.adg_func = interpolate.Rbf(self.T, self.P, self.adg)
			elif self.kind == 'linear':
				self.X2_func = interpolate.interp2d(self.T, self.P, self.X2)
				self.X_func = interpolate.interp2d(self.T, self.P, self.X)
				self.rho_func = interpolate.interp2d(self.T, self.P, self.rho)
				self.S_func = interpolate.interp2d(self.T, self.P, self.S)
				self.ST_func = interpolate.interp2d(self.T, self.P, self.ST)
				self.SP_func = interpolate.interp2d(self.T, self.P, self.SP)
				self.adg_func = interpolate.interp2d(self.T, self.P, self.adg)
			else:
				raise SystemExit('Invalid interpolator.')
			
	
	def __init__(self, Y, path, outdir, name, reuse, kind='Rbf'):
		self.hydrogen = self.SaumonBase(path + 'EOS/h_tab_i.dat')
		self.helium = self.SaumonBase(path + 'EOS/he_tab_i.dat')
		self.Y = Y
		self.path = path
		self.outdir = outdir
		self.name = name
		self.reuse = reuse
		self.create_EOS()
		if kind == 'Rbf':
			self.mixture = self.SaumonBase(path + 'EOS/' + outdir + '/' + name + '.dat')
			self.rho = self.rho_Rbf
			self.adg = self.adg_Rbf
		elif kind == 'Bilinear':
			eos = EOS(path + 'EOS/' + outdir + name + '.dat')
			self.rho = lambda T,P: 10**eos.rho(np.log10(T), np.log10(P))
			self.adg = lambda T,P: 10**eos.adg(np.log10(T), np.log10(P))
		else:
			raise SystemExit('Invalid interpolation type.')
		
	
	def _rho(self, T, P):
		T = np.log10(T)
		P = np.log10(P)
		rho_H = 10.**self.hydrogen.rho_func(T, P)
		rho_He = 10.**self.helium.rho_func(T, P)
		if np.isnan(rho_H): rho_H = 10**np.max(self.hydrogen.rho)
		if np.isnan(rho_He): rho_He = 10**np.max(self.helium.rho)
		rho = ((1. - self.Y)/rho_H + self.Y/rho_He)**-1.
		return float(rho)
	
	def _S(self, T, P):
		Smix = self._Smix(T, P)
		T = np.log10(T)
		P = np.log10(P)
		S_H = 10**self.hydrogen.S_func(T, P)
		S_He = 10**self.helium.S_func(T, P)
		if np.isnan(S_H): S_H = 10**np.max(self.hydrogen.S)
		if np.isnan(S_He): S_He = 10**np.max(self.helium.S)
		S = (1 - self.Y)*S_H + self.Y*S_He + Smix
		return S
	
	def _Smix(self, T, P):
		T = np.log10(T)
		P = np.log10(P)
		X_H = self.hydrogen.X_func(T, P)
		X_H2 = self.hydrogen.X2_func(T, P)
		X_He = self.helium.X2_func(T, P)
		X_Heplus = self.helium.X_func(T, P)
		if np.isnan(X_H): X_H = np.max(self.hydrogen.X)
		if np.isnan(X_He): X_He = np.max(self.helium.X)
		if np.isnan(X_H2): X_H2 = 10**np.max(self.hydrogen.X2)
		if np.isnan(X_Heplus): X_Heplus = 10**np.max(self.helium.X2)
		Xe_H = heaviside2(.5*(1. - X_H - X_H2))
		Xe_He = heaviside2((2. - 2*X_He - X_Heplus)/3.)
		ksi_H = .5*(1. + X_H + 3* X_H2)
		ksi_He = (1. + 2*X_He + X_Heplus)/3.
		alpha = (1. - self.Y)/(mH*ksi_H)
		beta = (mH/mHe)*(self.Y/(1. - self.Y))
		gamma = ksi_H/ksi_He
		if Xe_H != 0.:
			delta = (Xe_He/Xe_H)*beta*gamma
		else:
			delta = 0.
		if delta == 0.:
			invdelta = 0.
		else:
			invdelta = 1./delta
		Smix = k_boltz_cgs*(alpha*np.log(1. + beta*gamma) + alpha*beta*gamma*np.log(1. + 1./(beta*gamma)) - alpha*Xe_H*np.log(1. + delta) - alpha*beta*gamma*Xe_He*np.log(1. + invdelta))
		return Smix 
	
	def _STmix(self, T, P):
		h = 1e-10
		STmix = (np.log10(self._Smix(T + h, P)) - np.log10(self._Smix(T, P)))/h
		return STmix
	
	def _SPmix(self, T, P):
		h = 1e-10
		SPmix = (np.log10(self._Smix(T, P + h)) - np.log10(self._Smix(T, P)))/h
		return SPmix
	
	def _adg(self, T, P):
		if self.Y == 0.:
			T = np.log10(T)
			P = np.log10(P)
			adg = self.hydrogen.adg_func(T, P)
			if np.isnan(adg): adg = np.max(self.hydrogen.adg)
			return float(adg)
		if self.Y == 1.:
			T = np.log10(T)
			P = np.log10(P)
			adg = self.helium.adg_func(T, P)
			if np.isnan(adg): adg = np.max(self.helium.adg)
			return float(adg)
		S = self._S(T, P)
		Smix = self._Smix(T, P)
		STmix = self._STmix(T, P)
		SPmix = self._SPmix(T, P)
		T = np.log10(T)
		P = np.log10(P)
		S_H = 10**self.hydrogen.S_func(T, P)
		S_He = 10**self.helium.S_func(T, P)
		if np.isnan(S_H): S_H = 10**np.max(self.hydrogen.S)
		if np.isnan(S_He): S_He = 10**np.max(self.helium.S)
		ST_H = self.hydrogen.ST_func(T, P)
		ST_He = self.helium.ST_func(T, P)
		if np.isnan(ST_H): ST_H = np.max(self.hydrogen.ST)
		if np.isnan(ST_He): ST_He = np.max(self.helium.ST)
		SP_H = self.hydrogen.SP_func(T, P)
		SP_He = self.helium.SP_func(T, P)
		if np.isnan(SP_H): SP_H = np.max(self.hydrogen.SP)
		if np.isnan(SP_He): SP_He = np.max(self.helium.SP)
		ST = (1. - self.Y)*(S_H/S)*ST_H + self.Y*(S_He/S)*ST_He + (Smix/S)*STmix
		SP = (1. - self.Y)*(S_H/S)*SP_H + self.Y*(S_He/S)*SP_He + (Smix/S)*SPmix
		adg = -(SP/ST)
		return float(adg)

	def rho_Rbf(self, T, P):
		T = np.log10(T)
		P = np.log10(P)
		rho = 10**self.mixture.rho_func(T, P)
		if np.isnan(rho): rho = 10**np.max(self.mixture.rho)
		return float(rho)

	def adg_Rbf(self, T, P):
		T = np.log10(T)
		P = np.log10(P)
		adg = self.mixture.adg_func(T, P)
		if np.isnan(adg): adg = np.max(self.mixture.adg)
		return float(adg)
	
	def create_EOS(self):
		fname = self.path + 'EOS/' + self.outdir + '/' + self.name + '.dat'
		if os.path.isfile(fname) and self.reuse: return
		create_dir(self.path + 'EOS/', self.outdir)
		try:
			os.mkdir(self.path +  'EOS/' + self.outdir)
		except FileExistsError:
			pass
		f = open(fname, 'w')
		for line in open(self.path + 'EOS/h_tab_i.dat', 'r'):
			temp = line.rstrip().split()
			if len(temp) == 2:
				T = float(temp[0])
				N = int(temp[1])
				f.write(format(T, '-.3') + '  ' + str(N) + '\n')
			else:
				P = float(temp[0])
				rho = np.log10(self._rho(10**T, 10**P))
				adg = self._adg(10**T, 10**P)
				f.write('  ' + format(P, '-.3') + '  0.  0.  ' + format(rho, '-.5') + '  0.  0.  0.  0.  0.  0.  ' + format(adg, '-.5') + '\n')
		f.close()


# This class is used to interpolate the EOS
#----------------------------------------------------------------
class EOS:

	def __init__(self, path):
		self.read_dat(path)
		self.rho = self._rho2d
		self.adg = self._adg2d

	def read_dat(self, path):
		T = []
		P = []
		rho = []
		adg = []
		for line in open(path, 'r'):
			tmp = line.rstrip().split()
			if len(tmp) == 2: 
				T.append(float(tmp[0]))
				N = int(tmp[1])
				_P = np.zeros(N)
				_rho = np.zeros(N)
				_adg = np.zeros(N)
				idx = 0
			else:
				_P[idx] = float(tmp[0])
				_rho[idx] = float(tmp[3])
				_adg[idx] = float(tmp[10])
				idx += 1
				if idx == N: 
					P.append(_P)
					rho.append(_rho)
					adg.append(_adg)
		self._T = np.asarray(T)
		self._P = np.asarray(P)
		self._rho = np.asarray(rho)
		self._adg = np.asarray(adg)

	def _rho1d(self, idxT, P):
		idxP = bisect_left(self._P[idxT], P)
		if idxP == len(self._P[idxT]): 
			rho2 = self._rho[idxT][-1]
			rho1 = self._rho[idxT][-2]
			P2 = self._P[idxT][-1]
			P1 = self._P[idxT][-2] 
			return rho2 + (rho2-rho1) * (P-P2) / (P2-P1)
		if idxP == 0: idxP += 1
		rho2 = self._rho[idxT][idxP]
		rho1 = self._rho[idxT][idxP - 1]
		P2 = self._P[idxT][idxP]
		P1 = self._P[idxT][idxP - 1]
		return rho1 + (rho2-rho1) * (P-P1) / (P2-P1)

	def _adg1d(self, idxT, P):
		idxP = bisect_left(self._P[idxT], P)
		if idxP == len(self._P[idxT]): 
			adg2 = self._adg[idxT][-1]
			adg1 = self._adg[idxT][-2]
			P2 = self._P[idxT][-1]
			P1 = self._P[idxT][-2] 
			return adg2 + (adg2-adg1) * (P-P2) / (P2-P1)
		if idxP == 0: idxP += 1
		adg2 = self._adg[idxT][idxP]
		adg1 = self._adg[idxT][idxP - 1]
		P2 = self._P[idxT][idxP]
		P1 = self._P[idxT][idxP - 1]
		return adg1 + (adg2-adg1) * (P-P1) / (P2-P1)

	def _rho2d(self, T, P):
		idxT = bisect_left(self._T, T)
		if idxT == len(self._T): 
			rho2 = self._rho1d(-1, P)
			rho1 = self._rho1d(-2, P)
			T2 = self._T[-1]
			T1 = self._T[-2]
			return rho2 + (rho2-rho1) * (T-T2) / (T2-T1)
		if idxT == 0: idxT += 1
		rho2 = self._rho1d(idxT, P)
		rho1 = self._rho1d(idxT - 1, P)
		T2 = self._T[idxT]
		T1 = self._T[idxT - 1]
		return rho1 + (rho2-rho1) * (T-T1) / (T2-T1)

	def _adg2d(self, T, P):
		idxT = bisect_left(self._T, T)
		if idxT == len(self._T): 
			adg2 = self._adg1d(-1, P)
			adg1 = self._adg1d(-2, P)
			T2 = self._T[-1]
			T1 = self._T[-2]
			return adg2 + (adg2-adg1) * (T-T2) / (T2-T1)
		if idxT == 0: idxT += 1
		adg2 = self._adg1d(idxT, P)
		adg1 = self._adg1d(idxT - 1, P)
		T2 = self._T[idxT]
		T1 = self._T[idxT - 1]
		return adg1 + (adg2-adg1) * (T-T1) / (T2-T1)

	
	
# Rosseland-mean Opacities: Ferguson et al. 2005
# Rbf interpolator
#----------------------------------------------------------------
'''
class Opacities:
	
	# Constructor, must give Helium ratio Y, metal ratio Z, and global path
	#-------------------------------------------	
	def __init__(self, Y, Z, path):
		
		self.X = 1. - Y - Z
		self.Z = Z
		self.op_prop = []
		self.best_file = None
		self.abundance = 'asplund21'
		path += 'opacities/{}/'.format(self.abundance)

		self.T = []
		self.rho = []
		self.k = []
		self.kappa_func = None

		self.read_files(path)
		self.find_file()
		self.fill_tables()
		self.generate_func()
		
	def read_files(self, path):
		op_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
		for file in op_files:
			if self.abundance in ['caffau11', 'asplund21']:
				temp = file.split('.')
			else:
				temp = ('{}.{}'.format(file[0], file[1:])).split('.')
			X = temp[1]
			Z = temp[2]
			if X == '10':
				X = 1.
			else:
				X = float('0.' + X)
			if Z == '10':
				Z = 1.
			else:
				Z = float('0.' + Z)
			prop = {'path':path + file, 'X':X, 'Z':Z}
			self.op_prop.append(prop)
		self.op_prop = np.asarray(self.op_prop)
				
	def find_file(self):
		iX = closest([prop['X'] for prop in self.op_prop], self.X)
		iZ = closest([prop['Z'] for prop in self.op_prop[iX]], self.Z)
		i = iX[iZ][0]
		self.best_file = i
		
	def fill_tables(self):
		path = self.op_prop[self.best_file]['path']
		for line in open(path, 'r'):
				temp = line.rstrip().split()
				if len(temp) == 0:
					continue
				elif temp[1] in ['et', 'R', '&', '8.2021;']:
					continue
				elif temp[1] == 'T':
					R_tab = [float(temp[i]) for i in range(2, len(temp))]
				else:
					T = float(temp.pop(0))
					i = 0
					while i < len(temp):
						R = R_tab[i]
						rho = R + 3.*T - 18.
						try:
							float(temp[i])
						except ValueError:
							temp2 = temp.pop(i)
							k = reversed(clean_string(temp2))
							for ki in k:
								temp.insert(i, ki)
						kappa = float(temp[i])
						self.T.append(T)
						self.rho.append(rho)
						self.k.append(kappa)
						i += 1
						
	def generate_func(self):
		self.kappa_func = interpolate.Rbf(self.T, self.rho, self.k)
		
	def kappa(self, T, rho):
		logT = np.log10(T)
		logrho = np.log10(rho)
		kappa = 10**self.kappa_func(logT, logrho)
		return float(kappa)

	def get_composition(self):
		X = self.op_prop[self.best_file]['X']
		Z = self.op_prop[self.best_file]['Z']
		return 1. - X - Z, Z
'''

# Rosseland-mean Opacities: Ferguson et al. 2005
# Rectangular Bivariate Spline interpolator
#----------------------------------------------------------------
class Opacities:
	
	# Constructor, must give Helium ratio Y, metal ratio Z, and global path
	#-------------------------------------------	
	def __init__(self, Y, Z, path):
		
		self.X = 1. - Y - Z
		self.Z = Z
		self.op_prop = []
		self.best_file = None
		self.abundance = 'asplund21'
		path += 'opacities/{}/'.format(self.abundance)

		self.T = []
		self.R = []
		self.k = []
		self.kappa_func = None

		self.read_files(path)
		self.find_file()
		self.fill_tables()
		self.generate_func()
		
	def read_files(self, path):
		op_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
		for file in op_files:
			if self.abundance in ['caffau11', 'asplund21']:
				temp = file.split('.')
			else:
				temp = ('{}.{}'.format(file[0], file[1:])).split('.')
			X = temp[1]
			Z = temp[2]
			if X == '10':
				X = 1.
			else:
				X = float('0.' + X)
			if Z == '10':
				Z = 1.
			else:
				Z = float('0.' + Z)
			prop = {'path':path + file, 'X':X, 'Z':Z}
			self.op_prop.append(prop)
		self.op_prop = np.asarray(self.op_prop)
				
	def find_file(self):
		iX = closest([prop['X'] for prop in self.op_prop], self.X)
		iZ = closest([prop['Z'] for prop in self.op_prop[iX]], self.Z)
		i = iX[iZ][0]
		self.best_file = i
		
	def fill_tables(self):
		path = self.op_prop[self.best_file]['path']
		T = []
		kappa = []
		for line in open(path, 'r'):
			tmp = line.rstrip().split()
			if len(tmp) == 0 or tmp[1] in ['et', 'R', '&', '8.2021;']:
				continue
			elif tmp[1] == 'T':
				_R = np.array([float(tmp[i]) for i in range(2, len(tmp))])
			else:
				T.append(float(tmp.pop(0)))
				_kappa = np.zeros(len(_R))
				idx = 0
				while idx < len(tmp):
					try:
						float(tmp[idx])
					except ValueError:
						tmp2 = tmp.pop(idx)
						k = reversed(clean_string(tmp2))
						for ki in k: tmp.insert(idx, ki)
					_kappa[idx] = float(tmp[idx])
					idx += 1
				kappa.append(_kappa)
		_T = np.asarray(T)[::-1]
		_R = _R
		_kappa = np.asarray(kappa)[::-1]
		self.T = _T.copy()
		self.R = _R.copy()
		self.k = _kappa.copy()
						
	def generate_func(self):
		self.kappa_func = interpolate.RectBivariateSpline(self.T, self.R, self.k).ev
		
	def kappa(self, T, rho):
		logT = np.log10(T)
		logrho = np.log10(rho)
		logR = logrho - 3*(logT - 6.)
		kappa = 10**(self.kappa_func(logT, logR).item())
		return float(kappa)

	def get_composition(self):
		X = self.op_prop[self.best_file]['X']
		Z = self.op_prop[self.best_file]['Z']
		return 1. - X - Z, Z
	
	
# Ratio of the visible opacity to the thermal opacity: Jin et al. 2014
#----------------------------------------------------------------
class Gamma: 
	
	def __init__(self):
		self.T = [260., 388., 584., 861., 1267., 1460., 1577., 1730., 1870., 2015., 2255., 2777.]
		self.g = [0.005, 0.008, 0.027, 0.07, 0.18, 0.19, 0.18, 0.185, 0.2, 0.22, 0.31, 0.55]
		self.gamma_func = interpolate.interp1d(self.T, self.g, kind='linear')
	 
	def gamma(self, T):
		if T < self.T[0]:
			return self.g[0]
		elif T > self.T[-1]:
			return self.g[-1]
		else:
			return self.gamma_func(T)
		

# Stellar luminosities: Jackson et al. 2012
#----------------------------------------------------------------
class StarLum:
	
	def __init__(self, path):
		self.t = []
		self.Lbol = []
		self.LXUV = []
		self.read_file(path)
		self.Lbol_func = interpolate.interp1d(self.t, self.Lbol)
		self.LXUV_func = interpolate.interp1d(self.t, self.LXUV)
		
	def read_file(self, path):
		for line in open(path, 'r'):
			temp = line.rstrip().split()
			if temp[0] == '#':
				continue
			elif len(temp) == 3:
				t = float(temp[0])
				Lbol = float(temp[1])
				LXUV = float(temp[2])
				self.t.append(t)
				self.Lbol.append(Lbol)
				self.LXUV.append(LXUV)
			else:
				continue
	
	def L_bol(self, t):
		if t < self.t[0]: 
			return self.Lbol[0]
		elif t > self.t[-1]:
			return self.Lbol[-1]
		else:
			return self.Lbol_func(t)
		
	def L_XUV(self, t):
		if t < self.t[0]: 
			return self.LXUV[0]
		elif t > self.t[-1]:
			return self.LXUV[-1]
		else:
			return self.LXUV_func(t)
		
		
# Planetary luminosity coefficients: Mordasini et al. 2020
#----------------------------------------------------------------
class PlanetLum: 
	
	def __init__(self):
		self.t = [0.1e9, 0.3e9, 0.5e9, 0.8e9, 1e9, 2e9, 3e9, 4e9, 5e9, 6e9, 7e9, 8e9, 9e9, 10e9]
		self.a0_tab = [0.00252906, 0.00121324, 0.000707416, 0.000423376, 0.000352187, 0.000175775, 0.00011412, 8.81462e-5, \
				   6.91819e-5, 5.49615e-5, 4.5032e-5, 3.80363e-5, 3.30102e-5, 2.92937e-5]
		self.b1_tab = [-0.002002380, -0.000533601, -0.000394131, -0.000187283, -0.000141480, -4.07832e-5, -2.09944e-5, -2.07673e-5, \
				   -1.90159e-5, -1.68620e-5, -1.51951e-5, -1.40113e-5, -1.31146e-5, -1.24023e-5]
		self.b2_tab = [0.001044080, 0.000360703, 0.000212475, 0.000125872, 9.94382e-5, 4.58530e-5, 2.91169e-5, 2.12932e-5, \
				   1.62128e-5, 1.29045e-5, 1.05948e-5, 8.93639e-6, 7.69121e-6, 6.73922e-6]
		self.c1_tab = [.05864850, .02141140, .01381380, .00887292, .00718831, .00357941, .00232693, .00171412, .00134355, .00109019, \
				   .00091005, .00077687, .000675243, .000595191]
		self.c2_tab = [0.000967878, 0.000368533, 0.000189456, 0.000117141, 9.20563e-5, 5.52851e-5, 4.00546e-5, 2.90984e-5, 2.30387e-5, \
				   1.96163e-5, 1.70934e-5, 1.50107e-5, 1.32482e-5, 1.17809e-5]
		self.a0_func = interpolate.interp1d(self.t, self.a0_tab, kind='linear')
		self.b1_func = interpolate.interp1d(self.t, self.b1_tab, kind='linear')
		self.b2_func = interpolate.interp1d(self.t, self.b2_tab, kind='linear')
		self.c1_func = interpolate.interp1d(self.t, self.c1_tab, kind='linear')
		self.c2_func = interpolate.interp1d(self.t, self.c2_tab, kind='linear')
	 
	def a0(self, t):
		if t < self.t[0]:
			return self.a0_tab[0]
		elif t > self.t[-1]:
			return self.a0_tab[-1]
		else:
			return self.a0_func(t)
		
	def b1(self, t):
		if t < self.t[0]:
			return self.b1_tab[0]
		elif t > self.t[-1]:
			return self.b1_tab[-1]
		else:
			return self.b1_func(t)
		
	def b2(self, t):
		if t < self.t[0]:
			return self.b2_tab[0]
		elif t > self.t[-1]:
			return self.b2_tab[-1]
		else:
			return self.b2_func(t)
		
	def c1(self, t):
		if t < self.t[0]:
			return self.c1_tab[0]
		elif t > self.t[-1]:
			return self.c1_tab[-1]
		else:
			return self.c1_func(t)
		
	def c2(self, t):
		if t < self.t[0]:
			return self.c2_tab[0]
		elif t > self.t[-1]:
			return self.c2_tab[-1]
		else:
			return self.c2_func(t)


# Fit coefficients for the atmospheric grid
#----------------------------------------------------------------
class PlanetRad:

	def __init__(self, path):
		npz  = np.load(path, allow_pickle=True)
		Teq  = npz['Teq']
		Tint = npz['Tint']
		fit  = npz['fit']
		interps = []
		for i in range(len(fit) - 1):
			coeff = fit[i]
			interp = interpolate.RegularGridInterpolator((Teq, Tint), coeff)
			interps.append(interp)
		self.interps = interps