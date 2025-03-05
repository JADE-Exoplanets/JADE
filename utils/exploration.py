#--------------------------------------------------------------------------------------------
#--- Util for generating many JADE inputs for a parameter space exploration.
#--------------------------------------------------------------------------------------------
#--- Run this file from a terminal:
#---		python exploration.py 
#--------------------------------------------------------------------------------------------
#--- Settings have to be configured in this file, refer to the documentation below.
#--------------------------------------------------------------------------------------------
#--- Author: Mara Attia
#--------------------------------------------------------------------------------------------

'''
HOW TO USE:

+ Put the values of the invariant parameters inside 'constant_param' as strings. 
  Leave the values of the variable parameters as '' (or whatever you want, they will be overrided).

+ Put the variable parameters in 'variable_param'.
  'variable_param' must be a list of dictionaries [dict1, dict2, ...]
  Each dictionary should look like 'constant_param', but with parameters you want to explore.
  The generated input files will be the Cartesian product of dict1, dict2...

MINIMAL EXAMPLE:

variable_param = [
                  {'Ms':['1', '2', '3'], 'Rs':['1', '2', '3']},
                  {'Mpl':['0.5', '0.6']}
                 ]

will generate 6 input files corresponding to the following combinations:
Ms = 1, Rs = 1, Mpl = 0.5
Ms = 1, Rs = 1, Mpl = 0.6
Ms = 2, Rs = 2, Mpl = 0.5
Ms = 2, Rs = 2, Mpl = 0.6
Ms = 3, Rs = 3, Mpl = 0.5
Ms = 3, Rs = 3, Mpl = 0.6
'''

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

import os
import itertools
import numpy as np
import time as tm
from finished import finished

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

# Print verbose?
verbose = True

# Create bash files?
# In order to have consistent bash files, this parameter has to remain the same if you append new input files to an existing exploration.
create_bash = False

# Make a jobarray?
create_jobarray = False

# Launch previous simulations that are unfinished
launch_unfinished = False

# Options for the bash files: number of CPUs per simulation, partition name, required time (None if not included in the bash files)
ncpu = 1
partition = ''
reqtime = ''

# Input path (relative to input/) where the input folder will be stored (can be nested folders) 
input_tree = ''

# Name of the exploration. 
# Input files will be stored in 'input/input_tree/name/' as 'name_000001.txt', 'name_000002.txt', etc...
# The output_dir will be set to 'input_tree/name/name_000001', 'input_tree/name/name_000002', etc...
# Bash files will be stored in 'scripts/input_tree/name/' as 'name_000001.bash', 'name_000002.bash', etc...
name = ''

# Invariant parameters in the exploration. Do not change the keys!
constant_param = {
	
	'output_freq' :      '',
	'output_npts' :      '',
	'age' :              '',
	't_init' :           '',

	'dyn' :              '',
	'orderdyn' :         '',
	'perturber' :        '',
	'tides' :            '',
	'relat' :            '',
	'roche' :            '',

	'atmo' :             '',
	'atmo_acc' :         '',
	'atmo_grid_path' :   '',
	'evap' :             '',
	't_atmo' :           '',
	'parallel' :         '',

	'lazy_init' :        '',
	'reuse' :            '',
	'simul' :            '',

	'Ms' :               '',
	'Rs' :               '',
	'ks' :               '',
	'Qs' :               '',
	'alphas' :           '',
	'spins' :            '',
	          
	'Mcore' :            '',
	'Mpl' :              '',
	'Rpl' :              '',
	'kpl' :              '',
	'Qpl' :              '',
	'alphapl' :          '',
	'spinpl' :           '',
	          
	'stellar_lum' :      '',
	'stellar_lum_path' : '',
	'Lbol' :             '',
	'LX_Lbol_sat' :      '',
	'tau_X_bol_sat' :    '',
	'alpha_X_bol' :      '',

	'YHe' :              '',
	'Zmet' :             '',
	'fmantle' :          '',
	          
	'Mpert' :            '',

	'planet_sma' :       '',
	'planet_ecc' :       '',
	'planet_incl' :      '',
	'planet_lambd' :     '',
	'planet_omega' :     '',
	'planet_Omega' :     '',
	          
	'pert_sma' :         '',
	'pert_ecc' :         '',
	'pert_incl' :        '',
	'pert_lambd' :       '',
	'pert_omega' :       '',
	'pert_Omega' :       '',

}

# Parameters to explore.
variable_param = []			

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

def explore(c_param, v_param, idx=0):
	'''Creates the desired input/bash files'''
	sims = create_list(v_param)
	for sim in tqdm(sims):
		idx += 1
		fname = '{}_{:06d}'.format(name, idx)
		itree = input_tree.rstrip('/')
		if itree != '': itree += '/'
		to_file = {'name' : fname, 'output_dir': '{}{}/{}'.format(itree, name, fname)}
		for k in c_param: to_file[k] = c_param[k]
		for p in sim:
			k, v = p
			to_file[k] = v
		if check_exists(to_file, input_path, idx0):
			idx -= 1
		else:
			write_file(to_file, '{}{}.txt'.format(input_path, fname))
			if create_bash: write_bash('{}routines/main.py'.format(global_path), '{}/{}.txt'.format(name, fname), '{}{}.bash'.format(bash_path, fname))
	return idx

def explore2(i_path, c_param, v_param, idx=0):
	'''Optimized version of explore'''
	g = []
	i = 0
	for line in open('{}{}.info'.format(i_path, name), 'r'):
		i += 1
		if i < 4:
			continue
		else:
			if line == '\n':
				break
		_, prop = line.split('.txt: ')
		prop = prop.split(' ')
		prop.pop()
		keys = [p.split('=')[0] for p in prop]
		vals = [p.split('=')[1] for p in prop]
		g.append(vals)
	l = create_list(v_param)
	n = [ll[0] for ll in l[0]]
	m = []
	idxs = []
	for nn in n:
		idxs.append(keys.index(nn))
	for ll in l:
		_mm = [lll[1] for lll in ll]
		zipped_pairs = zip(idxs, _mm)
		mm = [x for _, x in sorted(zipped_pairs)]
		m.append(mm)
	for i, mm in enumerate(tqdm(m)):
		idx += 1
		fname = '{}_{:06d}'.format(name, idx)
		itree = input_tree.rstrip('/')
		if itree != '': itree += '/'
		to_file = {'name' : fname, 'output_dir': '{}{}/{}'.format(itree, name, fname)}
		for k in c_param: to_file[k] = c_param[k]
		for p in l[i]:
			k, v = p
			to_file[k] = v
		if mm in g:
			idx -= 1
		else:
			write_file(to_file, '{}{}.txt'.format(input_path, fname))
			if create_bash: write_bash('{}routines/main.py'.format(global_path), '{}/{}.txt'.format(name, fname), '{}{}.bash'.format(bash_path, fname))
	return idx

def create_list(v_param):
	'''Creates the desired list of simulations'''
	l = []
	for d in v_param:
		if len(set(len(x) for x in d.values())) > 1:
			raise SystemExit('For each dictionary inside "variable_param", all values should have the same length.')
		ll = []
		n = len_dict(d)
		ll = [[[p, d[p][i]] for p in d] for i in range(n)]
		l.append(ll)
	p = list(itertools.product(*l))
	r = [[pppp for ppp in pp for pppp in ppp] for pp in p]
	return r

def len_dict(d):
	'''Returns the length of the values of a dictionary'''
	return len(d[list(d)[0]])

def check_exists(param, path, idx):
	'''# Function that checks whether an input file with the content of 'param' already exists.
	# Input:
	#...'param': dictionary to check
	#...'path': path to the input folder
	#...'idx': first index of the current exploration
	# Output: True if such an input file exists, False otherwise'''
	if idx < 1: return False
	files = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		files.extend(filenames)
		break
	files = [f for f in files if f.split('.')[-1] == 'txt']
	files = [f for f in files if int((f.split('.')[0]).split('_')[-1]) <= idx]
	for file in files:
		fparam = read_file(path + file)
		fparam.pop('name')
		fparam.pop('output_dir')
		test = np.all([fparam[k] == param[k] for k in fparam])
		if test: return True
	return False

def write_file(param, path):
	'''# Function that writes the content of 'param' into a file.
	# Input:
	#...'param': dictionary containing all the necessary stuff to generate a JADE input
	#...'path': absolute path to the file to which the content is written'''
	f = open(path, 'w')
	for k in param.keys(): f.write(k + ' = ' + param[k] + '\n')
	f.close()

def write_bash(mpath, ipath, path):
	'''# Function that writes a bash script to launch a JADE simulation with a specified input file.
	# Input:
	#...'mpath': path to main.py
	#...'ipath': path to the input file
	#...'path': path to the bash file to which the content is written'''
	fname = (ipath.split('/')[-1]).split('.')[0]
	f = open(path, 'w')
	f.write('#! /bin/bash\n')
	f.write('#SBATCH --job-name=' + fname + '\n')
	f.write('#SBATCH --output=' + fname + '.out\n')
	f.write('#SBATCH --error=' + fname + '.err\n')
	f.write('#SBATCH --hint=nomultithread\n')
	if ncpu is not None: f.write('#SBATCH --cpus-per-task={:d}\n'.format(ncpu))
	if partition is not None: f.write('#SBATCH --partition={}\n'.format(partition))
	if reqtime is not None: f.write('#SBATCH --time={}\n'.format(reqtime))
	f.write('export NUM_THREADSPROCESSES=${SLURM_CPUS_PER_TASK}\n')
	f.write('python -u ' + mpath + ' ' + ipath)
	f.close()

def write_masterbash(name, sims, path):
	'''# Function that writes a bash script to launch all the bash scripts of the exploration.
	# Input:
	#...'name': name of the exploration
	#...'sims': simulations to be included
	#...'path': path to the bash file to which the content is written'''
	f = open(path + name + '.bash', 'w')
	f.write('#! /bin/bash\n')
	f.write('#SBATCH --partition={}\n'.format(partition))
	f.write('#SBATCH --time=1:00:00\n')
	f.write('declare -a StringArray=({})\n'.format(l2s(sims)))
	f.write('for script in "${StringArray[@]}"; do\n')
	f.write('    sbatch $script\n')
	f.write('done')
	f.close()

def write_jobarray(mpath, ipath, name, path, idx_start, idx_end):
	'''
	# Function that writes a bash script to launch all the bash scripts of the exploration.
	# Input:
	#...'mpath': path to main.py
	#...'ipath': path to the input directory
	#...'path': path to the bash file to which the content is written
	'''
	f = open(path, 'w')
	f.write('#!/bin/bash\n')
	f.write('#SBATCH --job-name={}_{}\n'.format(name, idx_end))
	f.write('#SBATCH --output={}_%05a.out\n'.format(name))
	f.write('#SBATCH --error={}_%05a.err\n'.format(name))
	f.write('#SBATCH --mem-per-cpu=10000\n')
	f.write('#SBATCH --hint=nomultithread\n')
	f.write('#SBATCH --cpus-per-task={}\n'.format(ncpu))
	f.write('#SBATCH --partition={}\n'.format(partition))
	f.write('#SBATCH --time={}\n'.format(reqtime))
	#No more than 1000 files per directory. This actually makes 1000 .out and 1000 .err
	f.write('#SBATCH --array={}-{}\n'.format(idx_start, idx_end))
	f.write('#SBATCH --hint=nomultithread\n')
	f.write('printf -v NUM %.5d $SLURM_ARRAY_TASK_ID\n')
	f.write('export NUM_THREADSPROCESSES=${SLURM_CPUS_PER_TASK}\n')
	f.write('srun python -u %s %s/%s_${NUM}.txt'%(mpath,name,name))
	f.close()

def write_info(path, name):
	'''# Function that writes an info file for the exploration.
	# Input:
	#...'path': path to the folder where the input files are. 
	#...'name': name of the info file (the file name will be name.info)'''
	f = open(path + name + '.info', 'w')
	files = [inp for inp in os.listdir(path) if inp.split('.')[-1] == 'txt']
	files.sort()
	c_param, v_keys = retrieve_info(path)
	f.write('### Info file for exploration: ' + name + ' ###\n\n')
	f.write('. Variable parameters\n')
	for inp in files:
		inp_param = read_file(path + inp)
		f.write(inp + ': ')
		for p in v_keys: f.write(p + '=' + inp_param[p] + ' ')
		f.write('\n')
	f.write('\n. Constant parameters\n')
	for p in v_keys: c_param.pop(p)
	for p in c_param.keys(): f.write(p + '=' + c_param[p] + '\n')
	f.close()

def retrieve_info(path):
	'''# Function that reads all the input files from a folder and returns the constant and variable parameters.
	# Input: path to the input folder
	# Output:
	#...'c_param': dictionary containing the constant parameters (same format as 'constant_param' above)
	#...'v_keys': list of the variable parameters '''
	files = [inp for inp in os.listdir(path) if inp.split('.')[-1] == 'txt']
	params = [read_file(path + f) for f in files]
	c_param = params[0].copy()
	c_param.pop('name')
	c_param.pop('output_dir')
	v_keys = []
	for k in c_param.keys():
		for param in params:
			if param[k] != c_param[k]:
				v_keys.append(k)
				break
	return c_param, v_keys

def read_file(path):
	'''# Function that returns a dictionary out of an input file.
	# Input: path to the input file
	# Output: dictionary containing the parameters'''
	param = {}
	for line in open(path, 'r'): 
		line = line.replace(' ', '')
		line = line.replace('\n', '')
		param[line.split('=')[0]] = line.split('=')[1]
	return param

def l2s(l):
	'''#Function that converts a list [a1, a2, ...] into a string '"a1" "a2" ...'
	# Input: list to convert
	# Output: string'''
	s = ''
	for ll in l:
		s += '"{}" '.format(ll)
	if len(s) > 0: s = s[:-1]
	return s

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

if __name__ == '__main__':
	if verbose: print('+ Exploration name:', name)

	# Creating the directories
	global_path = os.path.dirname(os.path.realpath(__file__)) + '/../'
	inp_tree = input_tree.rstrip('/')
	if inp_tree != '': inp_tree += '/'
	os.makedirs(global_path +  'input/' + inp_tree + name, exist_ok=True)

	if create_bash or create_jobarray:
			os.makedirs(global_path + 'scripts/' + inp_tree + name, exist_ok=True)

	input_path = global_path + 'input/' + inp_tree + name + '/'
	bash_path = global_path + 'scripts/' + inp_tree + name + '/'
	if verbose:
		print('+ The input files will be stored in input/' + inp_tree + name + '/')
		if create_bash: print('+ The bash files will be stored in scripts/' + inp_tree + name + '/')
		if create_jobarray: print('+ The jobarray bash files will be stored in scripts/' + inp_tree + name + '/')

	# Creating the input and bash files
	if verbose: print('+ Generating files... ')
	idx0 = len([f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] == 'txt'])
	if idx0 > 0:
		idx = explore2(input_path, constant_param, variable_param, idx0)
	else:
		idx = explore(constant_param, variable_param, idx0)
	if create_bash:
		if launch_unfinished:
			f = finished(name)
			to_bash = [sim.replace('.txt', '.bash') for sim in f if not f[sim]]
		else:
			to_bash = ['{}/{}_{:06d}.bash'.format(name, name, i) for i in range(idx0 + 1, idx + 1)]
		write_masterbash(name, to_bash, global_path + 'scripts/' + inp_tree)
	if create_jobarray:
		write_jobarray('{}routines/main.py'.format(global_path), '{}{}'.format(input_path, name), name, '{}/{}_{}.sh'.format(bash_path, name, idx), idx0 + 1, idx)
	if verbose:
		print('Successfully created', str(idx - idx0), 'input file(s).')
		if create_bash: print('+ Master bash file: scripts/' + inp_tree + name + '.bash')

	# Creating the info file
	write_info(input_path, name)
	if verbose: print('+ Info file: input/' + inp_tree + name + '/' + name + '.info')