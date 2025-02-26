#--------------------------------------------------------------------------------------------
#--- Routine to check whether a group of JADE simulations is finished or not.
#--------------------------------------------------------------------------------------------
#--- Run this file from the terminal with the following syntax:
#---		python finished.py [-v] args
#--------------------------------------------------------------------------------------------
#--- args: folders (absolute or relative to input/) where the input files are
#--- -v: verbosity (optional)
#--------------------------------------------------------------------------------------------
#--- Author: Mara Attia
#--------------------------------------------------------------------------------------------

import os
import argparse
from time import time
from output import JADE_output, read_input

def sim_finished(txt, npz=None):
	sim = JADE_output(txt, npz, verbose=False)
	finish = sim.finish
	return finish

def get_npz(txt):
	gpath = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/../') + '/'
	if not txt.startswith('/'):
		txt = '{}input/{}'.format(gpath, txt)
	inp = read_input(txt)
	n = inp['name']
	o = '{}/'.format(inp['output_dir'].rstrip('/'))
	opath = '{}saved_data/{}'.format(gpath, o)
	try:
		npath = os.listdir(opath)
	except FileNotFoundError:
		return None
	npath = sorted([opath + f for f in npath if len(f) == len(n) + 8 and f.startswith(n) and f.endswith('.npz')])
	for f in npath:
		try:
			_ = int(f[len(opath) + len(n) + 1:len(opath) + len(n) + 4])
		except TypeError:
			npath.remove(f)
	if len(npath) > 0:
		npz = npath[-1]
	else:
		npz = None
	return npz

def finished(*args, verbose=False):
	if verbose: t1 = time()
	gpath = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/../') + '/'
	f = {}
	for arg in args:
		if type(arg) in [list, tuple]:
			sims = arg.copy()
		elif type(arg) is str:
			sim_folder = '{}/'.format(arg.rstrip('/'))
			if sim_folder.startswith('/'):
				real_path = sim_folder
			else:
				real_path = '{}input/{}'.format(gpath, sim_folder)
			_, _, names = next(os.walk(real_path))
			sims = sorted([real_path + name for name in names if name.endswith('.txt')])
		else:
			raise SystemExit('Invalid input.')
		for sim in sims: 
			if sim.startswith('{}input/'.format(gpath)):
				name = sim[len('{}input/'.format(gpath)):]
			else:
				name = sim
			npz = get_npz(sim)
			if npz is not None:
				ff = sim_finished(sim, npz=npz)
			else:
				ff = False
			f[name] = ff
			if verbose: print('+ {} finished: {}'.format(name, f[name]))
	if verbose:
		t2 = time()
		dt = t2 - t1
		if dt < 1: 
			print('+ Elapsed time: {:.2f} ms.'.format(dt*1000))
		else:
			print('+ Elapsed time: {:.2f} s.'.format(dt))
	return f

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('args', nargs='*')
	parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
	inp = vars(parser.parse_args())
	args = inp['args']
	verbose = inp['verbose']
	finished(*args, verbose=verbose)