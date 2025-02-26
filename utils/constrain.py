#--------------------------------------------------------------------------------------------
#--- Util for constraining JADE simulations with observational data.
#--------------------------------------------------------------------------------------------
#--- Author: Mara Attia
#--------------------------------------------------------------------------------------------

import os
import mcint
import numpy as np
import time as tm
import scipy.interpolate as sip
import scipy.integrate as sig
from pathos.pools import ProcessPool
from output import JADE_output


#--------------------------------------------------------------
# Constant lists.

# +++ PARAM: constrainable parameters. See output.py documentation for units.
#            NB: add '_init' at the end of a parameter to constrain its initial value.
# +++ DISTRIB: allowed probability distributions. See VarPar for details.

PARAM = ['t', 'a', 'e', 'Mp', 'Rp', 'P', 'p', 'q', 'o', 'ip', 'psi', 'lambd', 'Op', 'Menv', 'Tsurf', 
		 'Lp', 'massloss', 'rho', 'imut', 'Os', 'istar', 'Lbol', 'LXUV',
		 'Ms', 'Rs', 'Qs', 'Mc', 'Qp', 'Mpert', 'apert', 'epert', 'logrpert']

PARAM += ['{}_init'.format(p) for p in PARAM[1:]]

DISTRIB = ['uniform', 'gaussian', 'sgaussian', 'custom']

#--------------------------------------------------------------


class JADE_constrain:

	'''
	Class allowing to conveniently constrain JADE simulations with data.
	Can generate a probability grid according to simulation parameters, as well as posterior distributions.

	How to use:
	+++ Create a class object.
	+++ Add simulations to be constrained using add_sim (single simulation) or add_folder (folder of simulations).
	+++ Add parameters to be constrained along with their priors using add_param. Time must always be added.
	+++ Add data points along with their distributions using add_data.
	+++ Generate probability grid with respect to varying parameters using compute_grid.
	+++ Generate posterior distributions using compte_pdf.
	'''

	def __init__(self, verbose=False, nworker=None):

		'''
		Constructor.

		Input:
		+++ verbose (boolean): verbosity. Default: False.
		+++ nworker (integer): number of workers for multiprocessing. Default: None (no multiprocessing).
		'''

		self.__verbose = verbose
		self.__nworker = nworker 
		self.__gpath = os.path.dirname(os.path.realpath(__file__)) + '/../'
		self.__param = []
		self.__sim = []
		self.__data = []
		self.vector = None
		self.grid = None
		self.grid_sim = None
		self.interp = None
		self.pdf = None
		if verbose: 
			print('+ Beginning exploration.')
			if nworker is not None: print('+ Multiprocessing on {:d} cores activated.'.format(nworker))


	def set_nworker(self, nworker):

		'''
		Sets the number of workers for multiprocessing.

		Input:
		+++ nworker (integer): number of workers. Set to 0 to deactivate multiprocessing.
		'''

		n = nworker if nworker > 0 else None
		self.__nworker = n
		if verbose:
			if n is not None: 
				print('+ Multiprocessing on {:d} cores activated.'.format(n))
			else:
				print('+ Multiprocessing deactivated.')


	def add_param(self, param, prior, priorp1, priorp2=None, priorp3=None):

		'''
		Adds a parameter to be constrained.

		Input:
		+++ param (string): parameter to be constrained. Must be one of PARAM.
		+++ prior (string): prior for the parameter. Must be one of DISTRIB.
		+++ priorp1 (float): prior's first parameter. See documentation of VarPar for more details.
		+++ priorp2 (float): prior's second parameter. See documentation of VarPar for more details.
		+++ priorp3 (float): prior's third parameter (optional). See documentation of VarPar for more details.
		'''

		if param not in PARAM: raise SystemExit('Invalid parameter.')
		if prior not in DISTRIB: raise SystemExit('Invalid prior.')
		if priorp3 is None and prior == 'sgaussian': raise SystemExit('A third parameter is needed for a skewed Gaussian distribution.')
		if priorp2 is None and prior != 'custom': raise SystemExit('A second parameter is needed for this distribution.')
		if param in [p.n for p in self.__param]:
			if self.__verbose: print('+ Parameter ' + param + ' already exists.')
			return
		p = VarPar(param, prior, priorp1, priorp2, priorp3)
		self.__param.append(p)
		if self.__verbose: print('+ Successfully added ' + param + ' to the parameters.')



	def add_sim(self, txt, npz=None, npoint=None, only_finish=True, verbose=None):

		'''
		Adds a single simulation to be constrained.

		Input:
		+++ txt (string): path to the input file (absolute or relative to input/).
		+++ npz (sequence of strings): if defined, path to the npz files (absolute or relative to saved_data/).
									   Overrides the output path inside the input file.
		+++ npoint (integer): number of points in the simulation. Optional.
		+++ only_finish (boolean): add the simulation only if it finished. Optional.
		+++ verbose (boolean): verbosity. If not defined, uses the verbosity parameter of the class object.
		'''

		if verbose is None:
			verbose = self.__verbose
		try:
			sim = JADE_output(txt, npz=npz)
		except FileNotFoundError:
			return None
		if only_finish:
			add = sim.finish
		else:
			add = True
		if add:
			to_generate = [p.replace('_init', '') for p in list(dict.fromkeys([p.n for p in self.__param] + [p.n for p in self.__data])) \
						   if p.replace('_init', '') not in ['t', 'a', 'e', 'Rp', 'Mp', 'Ms', 'Rs', 'Qs', 'Mc', 'Qp', 'Mpert', 'apert', 'epert']]
			sim = generate_param(sim, *to_generate, npoint=npoint) if npoint is not None else generate_param(sim, *to_generate, points=sim.t_dyn)
			self.__sim.append(sim)
			if verbose: 
				try:
					i = txt.rindex('/') + 1
				except ValueError:
					i = 0
				print('+ Successfully added ' + txt[i:] + ' to the simulations.')
			return sim


	def add_folder(self, sim_folder, npoint=None, only_finish=True):

		'''
		Adds a folder of simulations to be constrained.

		Input:
		+++ sim_folder (string): name of the folder (absolute or relative to input/).
		+++ npoint (integer): number of points in each simulation. Optional.
		+++ only_finish (boolean): add a simulation only if it finished. Optional.
		'''

		if self.__verbose: n_old = len(self.__sim)
		sim_folder = '{}/'.format(sim_folder.rstrip('/'))
		if sim_folder.startswith('/'):
			real_path = sim_folder
		else:
			real_path = '{}input/{}'.format(self.__gpath, sim_folder)
		_, _, names = next(os.walk(real_path))
		names = sorted([name for name in names if name[-4:] == '.txt'])
		if self.__nworker is not None:
			pool = ProcessPool(nodes=self.__nworker)
			pool.restart()
			f = lambda name: self.add_sim('{}/{}'.format(real_path, name), npoint=npoint, only_finish=only_finish)
			self.__sim += pool.map(f, names)
			pool.close()
			self.__sim = [sim for sim in self.__sim if sim is not None]
		else:
			for name in names: 
				self.add_sim('{}/{}'.format(real_path, name), npoint=npoint, only_finish=only_finish)
		if self.__verbose: print('+ Successfully added {:d} simulation(s).'.format(len(self.__sim) - n_old))


	def add_data(self, param, distrib, distribp1, distribp2=None, distribp3=None):

		'''
		Adds a data point to constrain the simulations.

		Input:
		+++ param (string): nature of the data point. Must be one of PARAM. Cannot end with '_init'.
		+++ distrib (string): distribution of the parameter. Must be one of DISTRIB.
		+++ distribp1 (float): distribution's first parameter. See documentation of VarPar for more details.
		+++ distribp2 (float): distribution's second parameter. See documentation of VarPar for more details.
		+++ distribp3 (float): distribution's third parameter (optional). See documentation of VarPar for more details.
		'''

		if param not in PARAM: raise SystemExit('Invalid parameter.')
		if param.endswith('_init'): raise SystemExit('A data point cannot end with _init.')
		if distrib not in DISTRIB: raise SystemExit('Invalid distribution.')
		if distribp3 is None and distrib == 'sgaussian': raise SystemExit('A third parameter is needed for a skewed Gaussian distribution.')
		if distribp2 is None and distrib != 'custom': raise SystemExit('A second parameter is needed for this distribution.')
		if param in [p.n for p in self.__data]:
			if self.__verbose: print('+ Data point ' + param + ' already exists.')
			return
		p = VarPar(param, distrib, distribp1, distribp2, distribp3)
		self.__data.append(p)
		if self.__verbose: print('+ Successfully added ' + param + ' to the data points.')


	def remove_data(self, param):
		
		'''
		Removes a data point from the constraints.

		Input:
		+++ param (string): name of the parameter.
		'''

		if param in [p.n for p in self.__data]:
			i = [p.n for p in self.__data].index(param)
			self.__data.pop(i)
			if self.__verbose: print('+ Successfully removed ' + param + ' from the data points.')
		else:
			if self.__verbose: print('+ Data point ' + param + ' does not exist.')


	def compute_grid(self):

		'''
		Generates log-probability grid according to the varying parameters of the simulations and the data points.

		Creates new class attributes:
		+++ grid (2D numpy array): grid points. Each element is a vector containing simulation parameters and the associated log-probabilities.
		+++ vector (numpy array): name of the parameters/probabilities stored in the aforementioned vector.
		+++ names (numpy array): name of the simulations corresponding to each point.
		'''

		if 't' not in [p.n for p in self.__param]: raise SystemExit('t must be in the parameters.')
		if len(self.__sim) < 1: raise SystemExit('No simulation found to compute the grid.')

		if self.__verbose: 
			ti = tm.time()
			t0 = ti

		to_store = list(dict.fromkeys([p.n for p in self.__param] + [p.n for p in self.__data]))

		npoints = int(np.sum([len(sim.t) for sim in self.__sim]))
		nvector = len(to_store) + len(self.__param) + len(self.__data) + 1
		maxlen = np.max([len(sim.name) for sim in self.__sim])
		self.names = np.empty(npoints, dtype='<U{:d}'.format(maxlen))
		self.vector = to_store + ['log(Prior_' + p.n + ')' for p in self.__param] + ['log(Likelihood_' + p.n + ')' for p in self.__data] + ['log(P)']
		self.vector = np.asarray(self.vector)
		self.grid = np.zeros((npoints, nvector))

		ip = 0
		for sim in self.__sim:
			for i in range(len(sim.t)):
				iv = 0
				logPtot = 0.
				for p in to_store:
					try:
						ii = 0 if p.endswith('_init') else i
						self.grid[ip, iv] = getattr(sim, p.replace('_init', ''))[ii]
					except (TypeError, IndexError):
						self.grid[ip, iv] = getattr(sim, p.replace('_init', ''))
					iv += 1
				for p in self.__param + self.__data:
					try:
						ii = 0 if p.n.endswith('_init') else i
						vmodel = getattr(sim, p.n.replace('_init', ''))[ii]
					except (TypeError, IndexError):
						vmodel = getattr(sim, p.n.replace('_init', ''))
					logP = p.logP(vmodel)
					self.grid[ip, iv] = logP
					logPtot += logP
					iv += 1
				self.grid[ip, iv] = logPtot
				self.names[ip] = sim.name
				ip += 1
				if self.__verbose: 
					t1 = tm.time()
					if t1 - t0 > 5:
						print('+ Computing probability grid... {:.1f}%'.format(100*ip/npoints), end='\r')
						t0 = t1
		
		if self.__verbose:
			print('+ Computing probability grid...  100%')
			print('+ Number of points in grid: ' + str(npoints))
			print('+ Coordinates for each point: ' + str(nvector))
			print('  ', end='')
			print(*self.vector, sep=' ; ')
			tf = tm.time()
			dt = tf - ti
			if dt < 1:
				print('+ Elapsed time: {:.2f} ms.'.format(1000*dt))
			else:
				print('+ Elapsed time: {:.2f} s.'.format(dt))


	def compute_grid_sim(self, kind='int'):

		'''
		Generates log-probability grid according to the varying parameters of the simulations and the data points, binned by simulations.

		Input:
		+++ kind (string): 'int' to bin by integrating the log-probabilities, or 'max' to bin by taking their maximum.

		Creates new class attributes:
		+++ grid_sim (2D numpy array): grid points. Each element is a vector containing simulation parameters and the associated log-probability.
		+++ vector_sim (numpy array): name of the parameters/probability stored in the aforementioned vector.
		'''

		if kind not in ['int', 'max']: raise SystemExit('Invalid kind.')

		if self.grid is None: self.compute_grid()

		if self.__verbose: 
			ti = tm.time()
			t0 = ti

		to_store = [p.n for p in self.__param]
		to_store.remove('t')
		nvector = len(to_store) + 1
		npoints = len(self.__sim)
		maxlen = np.max([len(sim) for sim in self.names])
		self.names_sim = np.empty(npoints, dtype='<U{:d}'.format(maxlen))
		self.vector_sim = to_store + ['log(P)']
		self.grid_sim = np.zeros((npoints, nvector))

		for i, sim in enumerate(np.unique(self.names)):
			for j, p in enumerate(to_store):
				k = int(np.where(self.vector == p)[0])
				self.grid_sim[i, j] = self.grid[self.names == sim][0, k]
			k = int(np.where(self.vector == 't')[0])
			times = self.grid[self.names == sim][:, k]
			probs = self.grid[self.names == sim][:, -1]
			logPmax = np.max(probs)
			if logPmax == -np.inf:
				logP = -np.inf
			else:
				if kind == 'int':
					logP = logPmax + np.log(np.sum(np.diff(times)*np.exp(probs[1:] - logPmax)))
				else:
					logP = logPmax
			self.grid_sim[i, -1] = logP
			self.names_sim[i] = sim
			if self.__verbose: 
					t1 = tm.time()
					if t1 - t0 > 5:
						print('+ Computing probability grid binned by simulations... {:.1f}%'.format(100*i/npoints), end='\r')
						t0 = t1

		if self.__verbose:
			print('+ Computing probability grid binned by simulations...  100%')
			print('+ Number of simulations in grid: ' + str(npoints))
			print('+ Coordinates for each point: ' + str(nvector))
			print('  ', end='')
			print(*self.vector_sim, sep=' ; ')
			tf = tm.time()
			dt = tf - ti
			if dt < 1:
				print('+ Elapsed time: {:.2f} ms.'.format(1000*dt))
			else:
				print('+ Elapsed time: {:.2f} s.'.format(dt))

	
	def get_best_sim(self):

		'''
		Gets the simulation corresponding to the maximum log-probability.

		Output:
		+++ name (string): name of the best simulation.
		'''

		i = np.argmax(self.grid[:, -1])
		name = self.names[i]
		return name


	def get_logP_temporal(self, name):

		'''
		Gets the log-probabilities associated with a simulation, and the corresponding times.

		Input:
		+++ name (string): name of the desired simulation.

		Output:
		+++ times (numpy array): corresponding times.
		+++ logPs (numpy array): log-probabilities.
		'''

		if self.grid is None: raise SystemExit('The probability grid has to be computed beforehand.')

		subgrid = self.grid[self.names == name]
		times = subgrid[:, list(self.vector).index('t')]
		logPs = subgrid[:, -1]

		logPs = np.asarray([x for _, x in sorted(zip(times, logPs))])
		times = np.asarray(sorted(times))

		return times, logPs

	

	def generate_interp(self, kind='nearest', **kwargs):

		'''
		Generates an interpolator for the log-probability.

		Input:
		+++ kind (string): interpolation kind (optional). Default: nearest.
		+++ kwargs: any additional key-word arguments will be passed to the interpolator.

		Output:
		+++ interp (function): function(*parameters) that returns log(P) for specific values of the parameters. Is also a class attribute.
		'''

		if self.__verbose:
			ti = tm.time()
			print('+ Interpolating probability grid...', end='')

		self.interp = self.__generate_interp(kind=kind, **kwargs)

		if self.__verbose:
			print(' ok.')
			tf = tm.time()
			dt = (tf - ti)*1000
			if dt < 1000:
				print('+ Elapsed time: ' + '{:.2f}'.format(dt) + ' ms.')
			else:
				print('+ Elapsed time: ' + '{:.2f}'.format(1e-3*dt) + ' s.')
			print('+ Interpolator signature: function(', end='')
			print(*[p.n for p in self.__param], sep=', ', end='')
			print(')')

		return self.interp



	def generate_interp_sim(self, kind='nearest', **kwargs):

		'''
		Generates an interpolator for the log-probability binned by simulations.

		Input:
		+++ kind (string): interpolation kind (optional). Default: nearest.
		+++ kwargs: any additional key-word arguments will be passed to the interpolator.

		Output:
		+++ interp_sim (function): function(*parameters) that returns log(P) for specific values of the parameters. Is also a class attribute.
		'''

		if self.__verbose:
			ti = tm.time()
			print('+ Interpolating probability grid...', end='')

		self.interp_sim = self.__generate_interp_sim(kind=kind, **kwargs)

		if self.__verbose:
			print(' ok.')
			tf = tm.time()
			dt = (tf - ti)*1000
			if dt < 1000:
				print('+ Elapsed time: ' + '{:.2f}'.format(dt) + ' ms.')
			else:
				print('+ Elapsed time: ' + '{:.2f}'.format(1e-3*dt) + ' s.')
			print('+ Interpolator signature: function(', end='')
			print(*[p.n for p in self.__param if p.n != 't'], sep=', ', end='')
			print(')')

		return self.interp_sim



	def compute_pdf(self, param, xi=None, xf=None, npoint=50, points=None, nmc=10000):

		'''
		Computes the PDF of a specified parameter using a Monte-Carlo integrator.

		Input:
		+++ param (string): parameter for which the PDF is computed.
		+++ xi (float): start of the PDF range (optional). Default: minimal value of the parameter.
		+++ xf (float): end of the PDF range (optional). Default: maximal value of the parameter.
		+++ npoint (integer): number of points within the range (optional). Default: 50.
		+++ points (sequence): user-defined points for which the PDF is computed (optional). Overrides xi, xf, and npoint.
		+++ nmc (integer): number of evaluations for the employed Monte-Carlo integrator (optional). Default: 10000.

		Output:
		+++ pdf (2D array): the computed PDF. 
		'''

		if param not in [p.n for p in self.__param]: raise SystemExit('Invalid parameter.')

		if self.grid is None: raise SystemExit('Must define the probability grid first.')

		if self.interp is None: self.generate_interp()
		if self.pdf is None: self.pdf = {}

		sampler = self.__mc_sampler(param)
		domain = self.__mc_domain(param)

		if points is None:
			if xi is None: xi = self.__min(param)
			if xf is None: xf = self.__max(param)
			x = np.linspace(xi, xf, num=npoint)
		else:
			x = np.asarray(points)
		y = []

		if self.__verbose: 
			ti = tm.time()
			print('+ Constructing PDF for ' + param + '...', end='')

		for _x in x:
			integrand = self.__mc_integrand(param, _x)
			_y, _ = mcint.integrate(integrand, sampler(), measure=domain, n=nmc)
			y.append(_y)
		
		if self.__verbose: 
			print(' ok.')
			print('+ Normalizing PDF...', end='')
			
		y = self.__normalize(x, np.asarray(y))

		if self.__verbose:
			print(' ok.')
			tf = tm.time()
			dt = (tf - ti)*1000
			if dt < 1000:
				print('+ Elapsed time: ' + '{:.2f}'.format(dt) + ' ms.')
			else:
				print('+ Elapsed time: ' + '{:.2f}'.format(1e-3*dt) + ' s.')

		pdf = np.vstack((x, y)).T
		self.pdf[param] = pdf

		return pdf


	def get_params(self):

		'''
		Returns the parameters of the exploration.

		Output:
		+++ params (sequence): list of the parameters.
		'''

		return self.__param


	def __generate_interp(self, kind, **kwargs):

		if kind not in ['nearest', 'linear', 'rbf']: raise SystemExit('Invalid interpolator.')
		indices = [list(self.vector).index(p.n) for p in self.__param]
		if len(indices) > 1:
			points = self.grid[:, indices]
		else:
			points = self.grid[:, indices[0]]
		mins = np.min(points, axis=0)
		maxs = np.max(points, axis=0)
		points = (points - mins)/(maxs - mins)
		values = self.grid[:, -1]
		if len(indices) > 1:
			if kind == 'nearest':
				interpolator = sip.NearestNDInterpolator(points, values, **kwargs)
			elif kind == 'linear':
				interpolator = sip.LinearNDInterpolator(points, values, **kwargs)
			else:
				_interpolator = sip.RBFInterpolator(points, values, **kwargs)
				interpolator = lambda *args: _interpolator([args])[0]
		else:
			if kind == 'rbf': raise SystemExit('RBF cannot be chosen for 1D interpolation.')
			interpolator = sip.interp1d(points, values, kind=kind, **kwargs)
		function = lambda *args: interpolator(*((np.array(args) - mins)/(maxs - mins)))
		return function


	def __generate_interp_sim(self, kind, **kwargs):

		if kind not in ['nearest', 'linear', 'rbf']: raise SystemExit('Invalid interpolator.')
		indices = [list(self.vector_sim).index(p.n) for p in self.__param if p.n != 't']
		if len(indices) > 1:
			points = self.grid_sim[:, indices]
		else:
			points = self.grid_sim[:, indices[0]]
		mins = np.min(points, axis=0)
		maxs = np.max(points, axis=0)
		points = (points - mins)/(maxs - mins)
		values = self.grid_sim[:, -1]
		if len(indices) > 1:
			if kind == 'nearest':
				interpolator = sip.NearestNDInterpolator(points, values, **kwargs)
			elif kind == 'linear':
				interpolator = sip.LinearNDInterpolator(points, values, **kwargs)
			else:
				_interpolator = sip.RBFInterpolator(points, values, **kwargs)
				interpolator = lambda *args: _interpolator([args])[0]
		else:
			if kind == 'rbf': raise SystemExit('RBF cannot be chosen for 1D interpolation.')
			interpolator = sip.interp1d(points, values, kind=kind, **kwargs)
		function = lambda *args: interpolator(*((np.array(args) - mins)/(maxs - mins)))
		return function


	def __mc_sampler(self, param):

		var = [p.n for p in self.__param if p.n != param]

		def sampler():
			while True:
				v = (np.random.uniform(self.__min(_v), self.__max(_v)) for _v in var)
				yield v
		
		return sampler


	def __mc_domain(self, param):

		var = [p.n for p in self.__param if p.n != param]
		d = 1.
		for v in var: d *= self.__max(v) - self.__min(v)
		return d


	def __mc_integrand(self, param, x):

		i = [p.n for p in self.__param].index(param)
		if i == 0:
			f = lambda var: np.exp(self.interp(x, *var))
		elif i == len([p.n for p in self.__param]) - 1:
			f = lambda var: np.exp(self.interp(*var, x))
		else:
			f = lambda var: np.exp(self.interp(*var[:i], x, *var[i:]))
		return f


	def __normalize(self, x, y):

		f = sip.interp1d(x, y, kind='nearest')
		itg, _ = sig.quad(f, x[0], x[-1])/(x[-1] - x[0])
		if itg == 0.: raise SystemExit('Null PDF. Please verify compute_pdf arguments.')
		y_norm = y/itg
		return y_norm


	def __min(self, param):

		if param == 't':
			i = self.vector.index(param)
			v = np.min(self.grid[:, i])
		else:
			try:
				vall = [getattr(sim, param)[0] for sim in self.__sim]
			except TypeError:
				vall = [getattr(sim, param) for sim in self.__sim]
			v = np.min(vall)
		return v


	def __max(self, param):

		if param == 't':
			i = self.vector.index(param)
			v = np.max(self.grid[:, i])
		else:
			try:
				vall = [getattr(sim, param)[0] for sim in self.__sim]
			except TypeError:
				vall = [getattr(sim, param) for sim in self.__sim]
			v = np.max(vall)
		return v



class VarPar:

	'''
	Class representing a varying parameter within a probability distribution.
	Possible distributions and their parameters are the following.

	Uniform (uniform):
	+++ p1: lower bound.
	+++ p2: upper bound.

	Gaussian (gaussian):
	+++ p1: mean.
	+++ p2: standard deviation.

	Skewed Gaussian (sgaussian):
	+++ p1: mean.
	+++ p2: below-mean standard deviation.
	+++ p3: above-mean standard deviation.

	User defined (custom):
	+++ p1: callable that returns the log probability.
	'''

	def __init__(self, name, distrib, p1, p2, p3):

		'''
		Constructor.

		Input:
		+++ name (string): name of the parameter.
		+++ distrib (string): distribution of the parameter. Must be one of DISTRIB.
		+++ p1 (float): distribution's first parameter.
		+++ p2 (float): distribution's second parameter.
		+++ p3 (float): distribution's third parameter.
		'''

		self.n = name
		self.d = distrib
		self.a = p1
		self.b = p2
		self.c = p3


	def logP(self, value):

		'''
		Computes the distance between a single value and the probability distribution.

		Input:
		+++ value (float): value to which the distance is computed.

		Output
		+++ distance (float): distance between the distribution and the value.
		'''

		if self.d == 'uniform':
			if self.a <= value <= self.b: 
				return -np.log(self.b - self.a)
			else:
				return -np.inf

		if self.d == 'gaussian':
			return -0.5*((value - self.a)/self.b)**2.

		if self.d == 'sgaussian':
			sigma = self.c if value > self.a else self.b
			return -0.5*((value - self.a)/sigma)**2.

		if self.d == 'custom':
			return self.a(value)



def generate_param(sim, *args, ti=None, tf=None, npoint=None, points=None):

	'''
	Constructs parameters within a JADE_output object.

	Input:
	+++ sim (JADE_output): simulation for which the parameters must be constructed.
	+++ *args (strings): parameters to be constructed.
	+++ ti (float): start of the time range (yr). If not provided, takes the first time value.
	+++ tf (float): end of the time range (yr). If not provided, takes the last time value.
	+++ npoint (integer): number of points within the time range. Optional.
	+++ points (sequence): user-defined time points (yr). If defined, overrides ti, tf and npoint.

	Output:
	+++ sim (JADE_output): simulation with constructed parameters.
	'''

	sim.set_time(ti=ti, tf=tf, npoint=npoint, points=points)
	sim.generate(*args)
	return sim

