#--------------------------------------------------------------------------------------------
#--- Grid-search routine used to derive the planet radius within JADE simulations.
#--------------------------------------------------------------------------------------------
#--- Author: Victor Ruelle
#--------------------------------------------------------------------------------------------

import numpy as np
from multiprocessing import Pool
from basic_functions import Rjup2orb

def mara_search(function, x_min, x_max, n_points, n_workers, thresh_error=0.01, n_iter_max=20, verbose=False):

    ''' Minimizes a given scalar function using brute-force grid-search using a pool of workers.
    Grid-search will iteratively select the best point and reproduce a new grid around that point.

    # Input:
        - function : scalar funciton
        - x_min, x_max : (float) starting bounds
        - n_points : (int) number of points used in each iteration's grid
        - n_workers : (int) number of parallel processes used at each iteration

    # Output
        - x_min : (float) coordinate of the minimum found
        - y_min : (float) minimum found
    '''

    #if verbose:
        #print("Running grid search on {}. Bounds : [{},{}]".format(function.__name__,x_min,x_max))

    # 1. Define a pool of workers
    pool = Pool(processes=n_workers)    
    
    # 2. Run a loop 
    current_x_min, current_x_max = x_min, x_max
    n = n_points
    best_image = 1.
    iteration = 0
    while best_image > thresh_error:

        # 2.a Define the grid
        iteration += 1
        if best_image == 1e50:
            n *= 2
            #current_x_min, current_x_max = x_min, x_max
        else:
            n = n_points
        step = (current_x_max-current_x_min)/n

        if iteration > n_iter_max or step == 0.: 
            if verbose: print('----')
            return -1, -1

        grid_points = np.linspace(current_x_min, current_x_max, n)

        # 2.b distribute the computation
        grid_images = pool.map(function,grid_points)

        # 2.c Get the best point and define new bounds
        best_point_index = np.argmin(grid_images)
        best_point = grid_points[best_point_index]
        best_image = grid_images[best_point_index]
        if best_image < 1e50:
            current_x_min, current_x_max = best_point - step, best_point + step

        if verbose:
            print(" Iteration {:02d} | Best point: {:.15f} Rjup | Best image: {:.5e}".format(iteration, best_point/Rjup2orb, best_image))
    
    pool.close()
    if verbose: print('----')
    return best_point, best_image
    

## Testing

def test_function(x):
    ''' Minimum in 5
    '''
    return x**2 + 5




if __name__ == "__main__":   
    print('Call mara_search directly.')
