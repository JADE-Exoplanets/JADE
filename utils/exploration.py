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
partition = 'private-astro-cpu'
reqtime = '6-23'

# Name of the exploration. 
# Input files will be stored in 'input/name/' as 'name_000001.txt', 'name_000002.txt', etc...
# Bash files will be stored in 'scripts/name/' as 'name_000001.bash', 'name_000002.bash', etc...
name = 'GJ436_qd_grid_80'

# Invariant parameters in the exploration. Do not change the keys!
constant_param = {
	
	'output_freq':'10',
	'output_npts':'5000',
	'age' : '8000.',
	't_init' : '10.',
	'dyn' : 'True',
	'orderdyn' : '2',
	'perturber' : 'True',
	'tides' : 'True',
	'relat' : 'True',
	'roche' : 'True',
	'atmo' : 'True',
	'atmo_acc' : 'False',
	'atmo_grid_path' : 'GJ436_2c.npz',
	'evap' : 'True',
	't_atmo' : '',
	'parallel' : 'True',
	'lazy_init' : 'False',
	'reuse' : 'True',
	'simul' : 'True',

	'Ms' : '0.445',
	'Rs' : '0.425',
	'ks' : '0.01',
	'Qs' : '10**5',
	'alphas' : '0.08',
	'spins' : '280.',
	          
	'Mcore' : '0.068213*0.89',
	'Mpl' : '',
	'Rpl' : '',
	'kpl' : '0.25',
	'Qpl' : '10**5',
	'alphapl' : '0.25',
	'spinpl' : '',
	          
	'stellar_lum' : 'tabular',
	'stellar_lum_path' : 'GJ436_lum.txt',
	'Lbol' : '',
	'LX_Lbol_sat' : '',
	'tau_X_bol_sat' : '',
	'alpha_X_bol' : '',
	'YHe' : '0.15',
	'Zmet' : '0.00001',
	'fmantle' : '0.66',
	          
	'Mpert' : '',

	'planet_sma' : '',
	'planet_ecc' : '0.0001',
	'planet_incl' : '0.',
	'planet_lambd' : '0.',
	'planet_omega' : '90.',
	'planet_Omega' : '90.',
	          
	'pert_sma' : '20.',
	'pert_ecc' : '0.0001',
	'pert_incl' : '80.',
	'pert_lambd' : '0.',
	'pert_omega' : '90.',
	'pert_Omega' : '90.',

	'ecc' : '',
	'incl' : '',
	'sma' : '',
	'proj_obl' : '',
	'true_obl' : '',
	'Menv' : '',
	'radius' : '',
	'init_prof' : '',
	'ang_mom_rel_error' : '',
	'ang_mom_cumul_error' : '',
	'time_units' : 'Gyr',
	'atmo_profiles' : '',
	'atmo_profiles_t' : ''

}

# Parameters to explore.
variable_param = [
				  				{'Mpl':['0.06898814772727273', '0.06960151918577445', '0.07022589542020774', '0.07086157527393996', '0.07150886850961538', '0.07216809631246968', '0.07283959182174338', '0.0735237006920415', '0.07422078168662674', '0.07493120730478589', '0.07565536444557476', '0.0763936551104263', '0.0771464971473029', '0.07791432503928758', '0.07869759074074073', '0.07949676456440405', '0.08031233612311015', '0.08114481533006']},
				  				{'planet_sma':['0.1', '0.11836734693877551', '0.13673469387755102', '0.15510204081632656', '0.17346938775510207', '0.19183673469387758', '0.21020408163265308', '0.2285714285714286', '0.2469387755102041', '0.2653061224489796', '0.2836734693877551', '0.3020408163265306', '0.3204081632653062', '0.3387755102040817', '0.3571428571428572', '0.3755102040816327', '0.3938775510204082', '0.41224489795918373', '0.43061224489795924', '0.44897959183673475', '0.46734693877551026', '0.48571428571428577', '0.5040816326530613', '0.5224489795918368', '0.5408163265306123', '0.5591836734693878', '0.5775510204081633', '0.5959183673469388', '0.6142857142857143', '0.6326530612244898', '0.6510204081632653', '0.6693877551020408', '0.6877551020408164', '0.7061224489795919', '0.7244897959183674', '0.7428571428571429', '0.7612244897959184', '0.7795918367346939', '0.7979591836734694', '0.8163265306122449', '0.8346938775510204', '0.8530612244897959', '0.8714285714285714', '0.889795918367347', '0.9081632653061225', '0.926530612244898', '0.9448979591836735', '0.963265306122449', '0.9816326530612246', '1.0', '0.27448979591836736', '0.29285714285714287', '0.3112244897959184', '0.32959183673469394', '0.34795918367346945', '0.36632653061224496', '0.38469387755102047', '0.403061224489796', '0.4214285714285715', '0.439795918367347', '0.4581632653061225'],
				  				 'spinpl':['132.5', '102.9', '82.9', '68.6', '58.0', '49.9', '43.5', '38.4', '34.2', '30.7', '27.7', '25.2', '23.1', '21.3', '19.6', '18.2', '17.0', '15.8', '14.8', '13.9', '13.1', '12.4', '11.7', '11.1', '10.5', '10.0', '9.5', '9.1', '8.7', '8.3', '8.0', '7.7', '7.3', '7.1', '6.8', '6.5', '6.3', '6.1', '5.9', '5.7', '5.5', '5.3', '5.2', '5.0', '4.8', '4.7', '4.6', '4.4', '4.3', '4.2', '29.1', '26.4', '24.1', '22.2', '20.4', '18.9', '17.6', '16.4', '15.3', '14.4', '13.5']},
 				  				{'Mpert':['505.56933254371233', '429.4133089678868', '364.72898581681034', '309.7883328644275', '263.1236203067307', '223.4882086202369', '189.8232448081168', '161.22937533009286', '136.94272003204327', '116.31446522310222', '98.7935307329268', '83.91184790263904', '71.27185521358103', '60.5358775018218', '51.417105026014205', '43.67193139599925', '37.09344567127822', '31.50590477191755', '26.760038533323247', '22.729061980255604', '19.30528828869159', '16.397251951411207', '13.92726529318674', '11.82934305831107', '10.047439626189078', '8.533951762518903', '7.248446907326463', '6.156583026293399', '5.229191169398304', '4.441496227587798', '3.7724550701301918', '3.2041943811084006', '2.7215331769537348', '2.3115772491610618', '1.9633746977944186', '1.667623353421721', '1.4164222713074275', '1.2030606590745765', '1.0218385990760896', '0.8679147761052839', '0.7371771425183684', '0.6261330656106092', '0.5318160225527756', '0.45170635025965034', '0.3836639330373803', '0.3258710297721048', '0.27678371329834056', '0.23509062465845468', '0.19967794038058298', '0.16959961688203265', '55.79049714201426', '47.38654116177904', '40.24850822822241', '34.18570705692057', '29.036170989366454', '24.66233108240039', '20.947340978281076', '17.791955374952583', '15.111878705393572', '12.835513196487907', '10.9020461472083', '9.259825328089', '7.864979101072254', '6.680244396474952', '5.673971236687839', '4.819277212634483', '4.093329324978347', '3.4767340045932564', '2.953019016802169', '2.5081934085479296', '2.13037374256257', '1.809466633447349', '1.536899104671124', '1.305389563022097']}
								 ]			

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

def explore(c_param, v_param, idx=0):
	'''Creates the desired input/bash files'''
	sims = create_list(v_param)
	for sim in tqdm(sims):
		idx += 1
		fname = '{}_{:06d}'.format(name, idx)
		to_file = {'name' : fname, 'output_dir': '{}/{}'.format(name, fname)}
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
		to_file = {'name' : fname, 'output_dir': '{}/{}'.format(name, fname)}
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
	#Yann said no more than 1000 files per directory. This actually makes 1000 .out and 1000 .err
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
	try:
		os.mkdir(global_path +  'input/' + name)
	except FileExistsError:
		pass
	if create_bash or create_jobarray:
		try:
			os.mkdir(global_path + 'scripts/' + name)
		except FileExistsError:
			pass
	input_path = global_path + 'input/' + name + '/'
	bash_path = global_path + 'scripts/' + name + '/'
	if verbose:
		print('+ The input files will be stored in input/' + name + '/')
		if create_bash: print('+ The bash files will be stored in scripts/' + name + '/')
		if create_jobarray: print('+ The jobarray bash files will be stored in scripts/' + name + '/')

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
		write_masterbash(name, to_bash, global_path + 'scripts/')
	if create_jobarray:
		write_jobarray('{}routines/main.py'.format(global_path), '{}input/{}'.format(global_path, name), name, '{}/{}_{}.sh'.format(bash_path, name, idx), idx0 + 1, idx)
	if verbose:
		print('Successfully created', str(idx - idx0), 'input file(s).')
		if create_bash: print('+ Master bash file: scripts/' + name + '.bash')

	# Creating the info file
	write_info(input_path, name)
	if verbose: print('+ Info file: input/' + name + '/' + name + '.info')