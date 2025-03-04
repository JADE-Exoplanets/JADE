#--------------------------------------------------------------------------------------------
#--- Grid-search routine used to derive the planet radius within JADE simulations.
#--------------------------------------------------------------------------------------------
#--- Authors: Victor Ruelle, Mara Attia
#--------------------------------------------------------------------------------------------

import sys
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool

from basic_functions import Rjup2orb

# Set more efficient start method for macOS at module import time
if sys.platform == 'darwin':
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        # Method already set, ignore
        pass

# Global variables for multiprocessing
_pool = None  # Persistent pool for reuse
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

# Get and configure pool (create once, reuse many times)
def get_pool(n_workers, jade, t, Mp, sma, ecc, acc):
    """Get or create a worker pool with updated parameters"""
    global _pool
    
    if _pool is None:
        # Create pool once and reuse it
        _pool = Pool(processes=n_workers, initializer=init_worker, 
                     initargs=(jade, t, Mp, sma, ecc, acc))
    else:
        # Try to update worker data without recreating pool
        try:
            # Apply init function to update globals in worker processes
            results = []
            for _ in range(min(10, n_workers)):
                results.append(_pool.apply_async(lambda: init_worker(jade, t, Mp, sma, ecc, acc)))
            # Wait for all updates to complete
            for r in results:
                r.get(timeout=1)
        except:
            # If updating fails, recreate the pool
            try:
                _pool.close()
                _pool.join()
            except:
                pass
            _pool = Pool(processes=n_workers, initializer=init_worker, 
                         initargs=(jade, t, Mp, sma, ecc, acc))
    
    return _pool

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
    
    global _pool
    
    if verbose and hasattr(function, '__name__'):
        print(f"Running grid search on {function.__name__}. Bounds: [{x_min},{x_max}]")

    # 1. Define a pool of workers
    try:
        # Check if we have the necessary kwargs to initialize workers
        if all(k in kwargs for k in ['jade', 't', 'Mp', 'sma', 'ecc', 'acc']):
            # Try to get or reuse a persistent pool
            try:
                pool = get_pool(n_workers, kwargs['jade'], kwargs['t'], kwargs['Mp'],
                               kwargs['sma'], kwargs['ecc'], kwargs['acc'])
                should_close_pool = False  # We'll keep the pool for future use
                used_function = worker_function
            except Exception as e:
                # If pool reuse fails, create a new one-time pool
                pool = Pool(processes=n_workers, initializer=init_worker,
                            initargs=(kwargs['jade'], kwargs['t'], kwargs['Mp'],
                                      kwargs['sma'], kwargs['ecc'], kwargs['acc']))
                should_close_pool = True  # We'll close this temporary pool
                used_function = worker_function
        else:
            # Fall back to original behavior (for backward compatibility)
            pool = Pool(processes=n_workers)
            should_close_pool = True  # We'll close this pool
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
                if should_close_pool:
                    pool.close()
                return -1, -1

            grid_points = np.linspace(current_x_min, current_x_max, n)

            # 2.b distribute the computation with optimized chunk size
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
        
    finally:
        # Close the pool if it was created just for this call
        if 'pool' in locals() and 'should_close_pool' in locals() and should_close_pool:
            try:
                pool.close()
            except:
                pass
    
    if verbose: print('----')
    return best_point, best_image

# Register cleanup function to be called at exit
import atexit
def cleanup():
    global _pool
    if _pool is not None:
        try:
            _pool.close()
            _pool.join()
        except:
            pass
atexit.register(cleanup)

## Testing

def test_function(x):
    ''' Minimum in 5
    '''
    return x**2 + 5

if __name__ == "__main__":   
    print('Call mara_search directly.')