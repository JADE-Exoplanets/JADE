#--------------------------------------------------------------------------------------------
#--- Routine allowing to constrain the internal structure of a planet.
#--------------------------------------------------------------------------------------------
#--- Run this file from the terminal with the following syntax:
#---		python compo_retrieval.py path/to/input_file.txt
#--------------------------------------------------------------------------------------------
#--- IMPORTANT: path/to/input_file.txt needs to be relative to input/
#--- All your input files should hence be in input/
#--- Note: you can create subfolders there
#--------------------------------------------------------------------------------------------
#--- Settings have to be configured in this file, in the 'USER INPUT' fields.
#--- You should save the output in saved_data/ for consistency.
#--------------------------------------------------------------------------------------------
#--- Author: Mara Attia
#--------------------------------------------------------------------------------------------

import sys
import os
import emcee
import corner
import arviz
import numpy as np
import matplotlib.pyplot as plt
from time import time
from pathos.pools import ProcessPool
from matplotlib.ticker import AutoLocator, AutoMinorLocator

from routines import JADE_Simulation, read_input_file
from basic_functions import format_time, mjup2orb
from interpolation import SaumonApprox, Opacities

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# USER INPUT
#-------------------------------------------------------------------------------------

# Radius of the planet, in R_Earth
Rp_med = 3.8

# +1/-1 sigma uncertainty on the radius of the planet, in R_Earth
Rp_sig_up = 0.2
Rp_sig_dn = 0.1

# Bounds of the atmosphere mass fraction
HBOUNDS = [0., 0.2]

# Bounds of the silicate mantle mass fraction
SBOUNDS = [0., 1.]

# Bounds of the atmospheric helium mass fraction
YBOUNDS = [0, 0.4]

# Number of workers for the MCMC
nw = 16

# Number of samples for the MCMC
ns = 3000

# Number of cores for multiprocessing
nc = 2

# Path for the MCMC data file
mcmc_path = '../saved_data/examples/example_rs/mcmc.h5'

# Path for the best fit properties
best_path = '../saved_data/examples/example_rs/fit_results.log'

# Path to where the chains plot should be saved, or 'on' for on screen display, or 'no' for no plotting
chain_path = 'no'

# Path to where the corner plot should be saved, or 'on' for on screen display, or 'no' for no plotting
corner_path = 'no'

# Resume the MCMC from file (if exists) or run the MCMC from scratch 
reuse = True

# Clean chains from non-physical configurations
clean = True

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

NDIM = 3

def main(path):

	print('--------------------------------------------------------------------------------')
	print('--------------------------------------------------------------------------------')
	print('')
	print('                    ░▒▓█▓▒░░▒▓██████▓▒░░▒▓███████▓▒░░▒▓████████▓▒░              ')
	print('                    ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░                     ')
	print('                    ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░                     ')
	print('                    ░▒▓█▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░                ')
	print('             ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░                     ')
	print('             ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░                     ')
	print('              ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░░▒▓████████▓▒░              ')
	print('')
	print('--------------------------------------------------------------------------------')
	print('-----------------Joining Atmosphere and Dynamics for Exoplanets-----------------')
	print('--------------------------------------------------------------------------------')
	print('-----------------------Attia et al. (2021, A&A 647, A40)------------------------')
	print('--------------------------------------------------------------------------------')
	print('--------------------------------------------------------------------------------')

	ti = time()

	global_path = os.path.dirname(os.path.realpath(__file__)) + '/../'
	path = 'input/' + path

	print('Composition retrieval mode activated. Input file: {}'.format(path))
	print('---')
	print('Initialization... ')

	sim = JADE_Simulation(global_path, path, verbose=False)
	sim.Rp = np.array([])
	sim.Mp = np.array([sim.init_Mp])
	sim.a = np.array([sim.init_orb_pl['a']])
	sim.e = np.array([sim.init_orb_pl['e']])
	sim.Lp = sim.planet_luminosity(sim.age)
	saumon_tables = SaumonApprox(global_path, 1001)

	print('---')

	def logP(x):

	# x[0] = H/He mass fraction
	# x[1] = silicate mantle mass fraction
	# x[2] = Y of the atmosphere

		if x[0] < HBOUNDS[0] or x[0] > HBOUNDS[1] or x[1] < SBOUNDS[0] or x[1] > SBOUNDS[1] or x[2] < YBOUNDS[0] or x[2] > YBOUNDS[1]:
			return -np.inf, -np.inf

		if x[0] + x[1] > 1.: return -np.inf, -np.inf

		sim.Mcore = (1. - x[0])*sim.init_Mp
		sim.fmantle = x[1]*sim.init_Mp/sim.Mcore
		sim.YHe = x[2]
		sim.rho, sim.adg = saumon_tables.get_eos_functions(sim.YHe)
		opacity_tables = Opacities(sim.YHe, sim.Zmet, global_path)
		sim.kappa = opacity_tables.kappa
		try:
			Rp_guess = [Rp_med/(3*23454.8), Rp_med*3/23454.8]
			Rp = sim.retrieve_radius(sim.age, sim.Mp[0], sim.a[0], sim.e[0], Rp_guess=Rp_guess, n_iter_max=12)
		except SystemExit:
			return -np.inf, -np.inf
		Rp *= 23454.8

		Rp_sig = Rp_sig_up if Rp > Rp_med else Rp_sig_dn
		logprob = -0.5*((Rp - Rp_med)/Rp_sig)**2.
		if np.isnan(logprob): return -np.inf, -np.inf

		return logprob, Rp


	#p0 = np.random.uniform([FBOUNDS[0], SBOUNDS[0], ZBOUNDS[0]], [FBOUNDS[1], SBOUNDS[1], ZBOUNDS[1]], (nw, NDIM))
	p0 = np.zeros((nw, NDIM))
	i = 0
	while i < nw:
		_HHe = np.random.uniform(HBOUNDS[0], HBOUNDS[1])
		_Si = np.random.uniform(SBOUNDS[0], np.min((1 - _HHe, SBOUNDS[1])))
		_Y = np.random.uniform(YBOUNDS[0], YBOUNDS[1])
		p0[i, 0] = _HHe
		p0[i, 1] = _Si
		p0[i, 2] = _Y
		i += 1

	run = False
	backend = emcee.backends.HDFBackend(mcmc_path)

	if (not reuse) or (not os.path.exists(mcmc_path)):
		backend.reset(nw, NDIM)
		run = True
		ni = ns
		print('Applying MCMC...')
	else:
		print('Retrieving MCMC...')
		if backend.iteration < ns:
			run = True
			ni = ns - backend.iteration
			print(f'Continuing MCMC for {ni} more iterations...')
		else:
			print(f'MCMC already has {backend.iteration} samples (requested: {ns}).')

	if run:
		with ProcessPool(nodes=nc) as pool:
			pool.restart()
			sampler = emcee.EnsembleSampler(nw, NDIM, logP, pool=pool, backend=backend)
			if reuse and os.path.exists(mcmc_path):
				p0 = backend.get_last_sample()
			sampler.run_mcmc(p0, ni, progress=True)

	print('---')
	nb = ns//2
	flat_samples = sampler.get_chain(discard=nb, flat=True)
	print(f'Applying burn-in phase (discarding first {nb} iterations)... Shape after burn-in/flattening: {flat_samples.shape}.')

	clean_samples = np.zeros([flat_samples.shape[0], flat_samples.shape[1] + 2])
	clean_samples[:, :2] = flat_samples[:, :2]
	clean_samples[:, 2] = 1. - (flat_samples[:, 0] + flat_samples[:, 1])
	clean_samples[:, 3] = flat_samples[:, 2]
	clean_samples[:, 4] = sampler.get_blobs(discard=nb, flat=True)

	if clean:
		clean_samples = clean_samples[(clean_samples[:, 2] >= 0.) & (clean_samples[:, 4] > -np.inf)]
		print(f'Cleaning nonphysical configurations... Shape after cleaning/reformatting: {clean_samples.shape}.')
	else:
		clean_samples[clean_samples[:, 4] == -np.inf] = 0.

	print('---')
	print('Computing fit results values...')

	medians = np.median(clean_samples, axis=0)
	hdis = np.array([arviz.hdi(clean_samples[:, i], hdi_prob=.682) for i in range(NDIM + 2)])
	names = ['H/He mass fraction', 'Silicate mass fraction', 'Fe mass fraction', 'Atmospheric He mass fraction', 'Planet radius']
	with open(best_path, 'w') as f:
		for i, name in enumerate(names):
			f.write('{}:\n'.format(name))
			f.write('median = {:.5f} ; -1s HDI = {:.5f} ; +1s HDI = {:.5f}\n'.format(medians[i], hdis[i, 0], hdis[i, 1]))
			f.write('value = {:.5f} -{:.5f} +{:.5f}\n'.format(medians[i], medians[i] - hdis[i, 0], hdis[i, 1] - medians[i]))
			f.write('-------------------------\n')

	if (chain_path != 'no') and (corner_path != 'no'):
		print('---')
		print('Plotting...')

	if chain_path != 'no':

		fig, axes = plt.subplots(NDIM, figsize=(10, 1.5*NDIM), sharex=True, dpi=300)

		samples = sampler.get_chain()

		labels = [r'$f_{\rm H/He}$', r'$f_{\rm Si}$', '$Y$']

		for i in range(NDIM):
			ax = axes[i]
			y = samples[:, :, i]
			ax.plot(y, 'k', alpha=0.3)
			ax.set_xlim(0, len(samples))
			ax.set_ylabel(labels[i])
			ax.tick_params(direction='in', top=True, right=True, which='both', rotation=0.)
			ax.tick_params(length=6, which='major')
			ax.tick_params(length=3, which='minor')
			ax.xaxis.set_major_locator(AutoLocator())
			ax.xaxis.set_minor_locator(AutoMinorLocator())
			ax.yaxis.set_major_locator(AutoLocator())
			ax.yaxis.set_minor_locator(AutoMinorLocator())

		axes[-1].set_xlabel('Step Number')

		plt.tight_layout()
		if chain_path == 'on':
			plt.show()
		else:
			plt.savefig(chain_path)

	if corner_path != 'no':

		fig, _ = plt.subplots(NDIM + 2, NDIM + 2, figsize=(10, 10), dpi=300)
		fig = corner.corner(clean_samples, quantiles=(0.159, 0.5, 0.841), fig=fig, plot_datapoints=False, verbose=False)
		ax = np.array(fig.axes).reshape((NDIM + 2, NDIM + 2))

		ax[-1][-1].vlines(Rp_med, 0, 1, transform=ax[-1][-1].get_xaxis_transform(), colors='orange')
		ax[-1][-1].vlines([Rp_med - Rp_sig_dn, Rp_med + Rp_sig_up], 0, 1, transform=ax[-1][-1].get_xaxis_transform(), 
						  colors='orange', linestyles='dotted')

		for i in range(NDIM + 2):
			for j in range(i + 1):
				ax[i][j].tick_params(direction='in', top=True, right=True, which='both', rotation=0.)
				ax[i][j].tick_params(length=6, which='major')
				ax[i][j].tick_params(length=3, which='minor')
				ax[i][j].xaxis.set_major_locator(AutoLocator())
				ax[i][j].xaxis.set_minor_locator(AutoMinorLocator())
				if i != j:
					ax[i][j].yaxis.set_major_locator(AutoLocator())
					ax[i][j].yaxis.set_minor_locator(AutoMinorLocator())

		labels = [r'$f_{\rm H/He}$', r'$f_{\rm Si}$', r'$f_{\rm Fe}$', r'$Y$', r'$R_{\rm p}$ [$R_\oplus$]']
		for i in range(NDIM + 1):
			ax[-1, i].set_xlabel(labels[i])
			ax[i + 1, 0].set_ylabel(labels[i + 1])

		plt.tight_layout()
		if corner_path == 'on':
			plt.show()
		else:
			plt.savefig(corner_path)

	tf = time()
	et = tf - ti

	print('--------------------------------------------------------------------------------')
	print('--------------------------------------------------------------------------------')
	print('End of the simulation. Elapsed time: {}'.format(format_time(et)))
	print('--------------------------------------------------------------------------------')
	print('--------------------------------------------------------------------------------')


if __name__ == '__main__':
	if len(sys.argv) != 2: 
		raise SystemExit('Invalid number of arguments. Syntax: python compo_retrieval.py input.txt')
	path = sys.argv[1]
	main(path)
