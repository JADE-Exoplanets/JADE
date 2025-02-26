#--------------------------------------------------------------------------------------------
#--- Auxiliary file for 'exploration_utils.py'.
#--------------------------------------------------------------------------------------------
#--- Author: Mara Attia
#--------------------------------------------------------------------------------------------

simulation = 'GJ436'
params = {'pert_sma':3.2, 'Mpert':1}

import pandas as pd
import numpy as np

def create_df(path):
	i = 0
	for line in open(path, 'r'):
		i += 1
		if i < 4:
			continue
		else:
			if line == '\n':
				break
		simid, prop = line.split('.txt: ')
		simid = int(simid[-5:])
		prop = prop.split(' ')
		prop.pop()
		vals = [[simid] + [float(p.split('=')[1]) for p in prop]]
		if i == 4:
			cols = ['sim_id'] + [p.split('=')[0] for p in prop]
			df = pd.DataFrame(vals, columns=cols)
		else:
			row = pd.DataFrame(vals, columns=cols)
			df = pd.concat([df, row])
	return df

def find_path(sim):
	if sim.startswith('/'):
		path = sim
	elif sim.endswith('.info'):
		path = '../input/{}'.format(sim)
	else:
		path = '../input/{}/{}.info'.format(sim, sim)
	return path


if __name__ == '__main__':
	path = find_path(simulation)
	df = create_df(path)
	condition = True
	for param in params:
		condition &= (df[param] == params[param])
	selection = df[condition]

	print('+ Simulation: {}'.format(simulation))
	print('+ Looking for...')
	print('  ', end='')
	for param in params:
		print('{}={}'.format(param, params[param]), end=' ')
	print()
	print('+ Found {} input files:'.format(len(selection)), end=' ')
	print(*selection['sim_id'], sep=', ')
	print(selection)