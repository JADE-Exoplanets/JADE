#--------------------------------------------------------------------------------------------
#--- Main file for the JADE code.
#--------------------------------------------------------------------------------------------
#--- Run this file from the terminal with the following syntax:
#---		python main.py path/to/input_file.txt
#-------------------------------------------------------------------------------------------- 
#--- IMPORTANT: path/to/input_file.txt needs to be relative to input/
#--- All your input files should hence be in input/
#--- Note: you can create subfolders there.
#--------------------------------------------------------------------------------------------
#--- All output files will be created in saved_data/
#--- You should use utils/output.py to interpret them.
#--------------------------------------------------------------------------------------------
#--- Author: Mara Attia
#--------------------------------------------------------------------------------------------

import sys
import os
from routines import JADE_Simulation

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

	global_path = os.path.dirname(os.path.realpath(__file__)) + '/../'

	path = 'input/' + path 
	simu = JADE_Simulation(global_path, path, verbose=True)

	simu.print_last_state()

	print('--------------------------------------------------------------------------------')

	simu.evolve(simu.age, 'jit', verbose=True)

	print('--------------------------------------------------------------------------------')

	if simu.simul: simu.print_last_state()

	print('--------------------------------------------------------------------------------')
	print('--------------------------------------------------------------------------------')
	print('End of the simulation.')
	print('--------------------------------------------------------------------------------')
	print('--------------------------------------------------------------------------------')


if __name__ == '__main__':
	if len(sys.argv) != 2: 
		raise SystemExit('Invalid number of arguments. Syntax: python main.py input_file.txt')
	path = sys.argv[1]
	main(path)
