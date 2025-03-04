#--------------------------------------------------------------------------------------------
#--- Grid-search routine used to derive the planet radius within JADE simulations.
#--------------------------------------------------------------------------------------------
#--- Authors: Victor Ruelle, Mara Attia
#--------------------------------------------------------------------------------------------

import os
import sys
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool

from basic_functions import Rjup2orb

# Set more efficient start method for macOS at module import time
if sys.platform == 'darwin':
    try:
        # On Apple Silicon, stick with 'spawn' (we'll optimize it differently)
        # On Intel Macs, use 'fork' for better performance
        if 'arm' in os.uname().machine:
            mp.set_start_method('spawn')
        else:
            mp.set_start_method('fork')
    except RuntimeError:
        # Method already set, ignore
        pass

# Global variables for multiprocessing
_jade = None
_t    = None
_Mp   = None
_sma  = None
_ecc  = None
_acc  = None

def init_worker(jade, t, Mp, sma, ecc, acc):
    """Initialize worker process with needed variables"""
    global _jade, _t, _Mp, _sma, _ecc, _acc
    _jade, _t, _Mp, _sma, _ecc, _acc = jade, t, Mp, sma, ecc, acc

def worker_function(Rp):
    """Function that worker processes will call"""
    global _jade, _t, _Mp, _sma, _ecc, _acc
    return _jade.atmospheric_structure(_t, Rp, _Mp, _sma, _ecc, acc=_acc)

# Separate update function to avoid lambda pickling issues
def update_worker_globals(jade, t, Mp, sma, ecc, acc):
    """Update global variables in worker process"""
    init_worker(jade, t, Mp, sma, ecc, acc)
    return True

def mara_search(function, x_min, x_max, n_points, n_workers, 
                thresh_error=0.01, n_iter_max=20, verbose=False, **kwargs):
    ''' Minimizes a given scalar function using brute-force grid-search using a pool of workers.
    Grid-search will iteratively select the best point and reproduce a new grid around that point.

    # Input:
        - function : scalar function
        - x_min, x_max : (float) starting bounds
        - n_points : (int) number of points used in each iteration's grid
        - n_workers : (int) number of parallel processes used at each iteration

    # Output
        - x_min : (float) coordinate of the minimum found
        - y_min : (float) minimum found
    '''
    
    if verbose and hasattr(function, '__name__'):
        print(f"Running grid search on {function.__name__}. Bounds: [{x_min},{x_max}]")

    # For Apple Silicon, use a smaller number of workers to prevent resource exhaustion
    if sys.platform == 'darwin' and 'arm' in os.uname().machine:
        # Limit to 75% of available processors on Apple Silicon for stability
        n_workers = min(n_workers, max(2, int(mp.cpu_count() * 0.75)))
        
    # 1. Create a new pool for each call (safer for Apple Silicon)
    pool = None
    try:
        # Check if we have the necessary kwargs to initialize workers
        if all(k in kwargs for k in ['jade', 't', 'Mp', 'sma', 'ecc', 'acc']):
            # Create a fresh pool with initializer
            pool = Pool(processes=n_workers, initializer=init_worker,
                        initargs=(kwargs['jade'], kwargs['t'], kwargs['Mp'],
                                  kwargs['sma'], kwargs['ecc'], kwargs['acc']))
            used_function = worker_function
        else:
            # Fall back to original behavior
            pool = Pool(processes=n_workers)
            used_function = function
        
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
            else:
                n = n_points
            step = (current_x_max-current_x_min)/n

            if iteration > n_iter_max or step == 0.: 
                if verbose: print('----')
                return -1, -1

            grid_points = np.linspace(current_x_min, current_x_max, n)

            # 2.b distribute the computation with optimized chunk size
            # On Apple Silicon, use larger chunks to reduce communication overhead
            if sys.platform == 'darwin' and 'arm' in os.uname().machine:
                chunk_size = max(1, len(grid_points) // n_workers)
            else:
                chunk_size = max(1, len(grid_points) // (n_workers * 2))
                
            grid_images = pool.map(used_function, grid_points, chunksize=chunk_size)

            # 2.c Get the best point and define new bounds
            best_point_index = np.argmin(grid_images)
            best_point = grid_points[best_point_index]
            best_image = grid_images[best_point_index]
            if best_image < 1e50:
                current_x_min, current_x_max = best_point - step, best_point + step

            if verbose:
                print(f" Iteration {iteration:02d} | Best point: {best_point/Rjup2orb:.15f} Rjup | Best image: {best_image:.5e}")
    
        if verbose: print('----')
        return best_point, best_image
        
    except Exception as e:
        # If multiprocessing fails, try a simpler approach as fallback
        if verbose:
            print(f"Multiprocessing error: {e}")
            print("Falling back to sequential processing")
            
        # Manually compute in sequence as a fallback
        best_point = None
        best_image = float('inf')
        
        for Rp in np.linspace(x_min, x_max, n_points):
            if all(k in kwargs for k in ['jade', 't', 'Mp', 'sma', 'ecc', 'acc']):
                image = kwargs['jade'].atmospheric_structure(kwargs['t'], Rp, kwargs['Mp'], kwargs['sma'], kwargs['ecc'], acc=kwargs['acc'])
            else:
                image = function(Rp)
                
            if image < best_image:
                best_image = image
                best_point = Rp
                
        return best_point, best_image
        
    finally:
        # Always ensure the pool is properly closed and joined
        if pool is not None:
            try:
                pool.close()
                pool.join()
            except:
                pass

## Testing

def test_function(x):
    ''' Minimum in 5
    '''
    return x**2 + 5

if __name__ == "__main__":   
    print('Call mara_search directly.')