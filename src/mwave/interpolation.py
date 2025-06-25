from re import L
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import warnings

warnings.warn("the interpolation module is deprecated", DeprecationWarning, stacklevel=2)

def convergend(fnc, x, method='cubic', atol=1e-3, rtol=1e-3, atolfrac=1, rtolfrac=1, nmaxdiv=12, verbosity=0):
    r"""Determines the x grid size needed for an interpolator of the provided function to converge to the required tolerance. The user must provide an initial grid on which to compute the function.
    
    :param fnc: The function to interpolate. This function can take multiple arguments.
    :param x: A list of grids interpolate :code:`fnc`. The list must have as many elements as there are arguments of :code:`fnc`. For example, if :code:`fnc` takes three arguments a valid choice for :code:`x` might be :code:`x=[np.linspace(0,5), np.linspace(2,3), np.linspace(-1,1)]`.
    :param method: The interpolation method to use, i.e. :code:`'cubic'` or :code:`'linear'`. Interpolation is performed using the :code:`scipy.interpolate.RegularGridInterpolator` class. The method provided to :code:`converge1d` is directly passed to this class.
    :param atol: The absolute error tolerance.
    :param rtol: The relative error tolerance.
    :param atolfrac: The required fraction of points without absolute errors.
    :param rtolfrac: The required fraction of points without relative errors.
    :param nmaxdiv: The maximum number of divisions to perform before the function exits.
    :param verbosity: The verbosity of the algorithm. :code:`0` produces no printouts, :code:`1` produces small notifications, and :code:`2` provides detailed updates.
    
    :code:`convergend` works by starting with a coarse grid in x. It then creates a fine grid in x with half the spacing of the coarse grid. The coarse grid is used to interpolate points onto the fine grid. These points are compared with the actual values on the fine grid. If residuals exist that are larger than the absolute or relative error tolerance then the process loops, with the fine grid becoming the coarse grid for the next round. This continues until the tolerances are met or the max number of divisions (:code:`nmaxdiv`) is met.
    
    Note that this algorithm is very expensive! If you start with an :math:`(n_1,n_2,...,n_m)` grid you have :math:`n_1\times n_2\times ...\times n_m` points. The next subdivision increases the number of points by a factor of 2 in each dimension, which clearly increases the number of points by :math:`2^m`.
    
    Note that :code:`fnc` can return a multidimensional output.
    
    Interpolation is performed using the :code:`scipy.interpolate.RegularGridInterpolator` class. The method provided to :code:`converge1d` is directly passed to this class.
    
    Absolute errores are computed at each point via
    
    .. math::
    
        \left|y' - y\right|
        
    where :math:`y'` is the interpolated value and :math:`y` is the actual value. Relative errors are computed via
    
    .. math::
    
        \frac{\left|y' - y\right|}{\mathrm{max}(\left|y'\right|,\left|y\right|)}
        
    :code:`atolfrac` and :code:`rtolfrac` allow convergence to occur when only a fraction of points meet the specified error requirements. For example, setting :code:`atolfrac=0.99` means that 99% of the points must pass the absolute error check to satisfy the absolute error requirement. :code:`atolfrac` and :code:`rtolfrac` are set to 1 by default, meaning that no absolute or relative errors are allowed.
    
    Example usage
    
    .. code-block:: python

        import numpy as np
        from mwave.interpolation import convergend
        from matplotlib import pyplot as plt

        # Define a grid
        x = np.linspace(0,10,5)
        y = np.linspace(0,10,4)
        xdense = np.linspace(x[0],x[-1],1000)
        ydense = np.linspace(y[0],y[-1],3)

        # Define a funky function
        avec = np.linspace(0.0, 0.05, 3)
        def fnc_vector(args):
            x, y = args
            out = np.empty((len(x), len(y), len(avec)))
            for i in range(len(x)):
                for j in range(len(y)):
                    out[i,j,:] = np.sin(2*x[i]**1.6)*np.exp(-0.02*y[j]) + np.exp(x[i]*avec) + 1

            return out

        def fnc_scalar(args):
            x, y = args
            out = np.empty((len(x), len(y)))
            for i in range(len(x)):
                for j in range(len(y)):
                    out[i,j] = np.sin(2*x[i]**1.6)*np.exp(-0.02*y[j]) + np.exp(x[i]*0.02) + 1

            return out

        # Select which function we want to use
        use_vector = True

        # Use convergend
        if use_vector:
            fnc = fnc_vector
        else:
            fnc = fnc_scalar

        # Plot
        z = fnc((xdense, ydense))
        if use_vector:
            plt.plot(xdense, z[:,:,0], '-')
            plt.plot(xdense, z[:,:,1], '--')
            plt.plot(xdense, z[:,:,2], '-.')
        else:
            plt.plot(xdense, z)
        plt.title('example function')
        plt.show()

        # Use converge1d to create an accurate interpolator for the function
        xgrid, ygrid, ifnc = convergend(fnc, (x, y), verbosity=2)

        # Create a meshgrid so that we can compute using the interpolation function
        args_grid = np.meshgrid(*xgrid, indexing='ij')

        # Plot the resiudals between the interpolator and the actual function
        if use_vector:
            fig, ax = plt.subplots(nrows=3, figsize=(4,8))
            im = ax[0].imshow(fnc(xgrid)[:,:,0]-ifnc(tuple(args_grid))[:,:,0])
            plt.colorbar(im, ax=ax[0])
            im = ax[1].imshow(fnc(xgrid)[:,:,1]-ifnc(tuple(args_grid))[:,:,1])
            plt.colorbar(im, ax=ax[1])
            im = ax[2].imshow(fnc(xgrid)[:,:,2]-ifnc(tuple(args_grid))[:,:,2])
            plt.colorbar(im, ax=ax[2])
        else:
            plt.figure()
            im = plt.imshow(fnc(xgrid)[:,:]-ifnc(tuple(args_grid))[:,:])
            plt.colorbar()
        plt.tight_layout()
        plt.show()
        
    """

    # Determine dimension of space
    ndim = len(x)

    # Compute on coarse grid
    y1 = fnc(x)
    # function must take in list of ndarrays, like regulargridinterpolator, return an array-like with shape m1, m2, ..., mn, ...
    
    # Get shape of y
    outshape = np.shape(y1)[ndim:]
    outdims = len(outshape)
    
    # Check if output is a scalar value
    output_scalar = outdims == 0

    # Enter division loop 
    for i in range(nmaxdiv):

        # Compute x1 shape
        x1shape = np.array([len(x[j]) for j in range(ndim)])

        # Generate finer grid with half the spacing of the coarse grid
        x2 = []
        for j in range(ndim):
            arr = np.empty(2*x1shape[j]-1)
            arr[::2] = x[j]
            arr[1::2] = x[j][:-1] + np.diff(x[j])/2
            x2.append(arr)

        # Notify user
        if verbosity > 0:
            print(f'Computing grid on {[len(x2[j]) for j in range(ndim)]} points.')

        # Determine the shape of y2
        y2shape = 2*x1shape-1
        if not output_scalar:
            y2shape = np.append(y2shape, outshape)

        # Create tuple to slice y2 and x2, allowing us to extract y2 at the half grid values
        s = [slice(1,y2shape[j],2) for j in range(ndim)]
        sy = tuple(s + [slice(0,outshape[j],1) for j in range(len(outshape))])

        # Determine x2 sliced at the half grid values
        x2sliced = [x2[j][s[j]] for j in range(ndim)]

        # Create y2, compute at the sliced x2 values
        y2 = np.full(y2shape, np.nan, dtype=y1.dtype)
        y2[sy] = fnc(x2sliced)
    
        # Interpolate on larger grid
        rgi = RegularGridInterpolator(x, y1, method=method)
    
        # Interpolate
        igrid = np.meshgrid(*x2sliced, indexing='ij')
        y2interp = rgi(tuple(igrid))
        
        # Compute error of the interpolated points
        abserr = np.abs(y2interp - y2[sy])
        relerr = abserr/np.fmax(np.abs(y2[sy]), np.abs(y2interp))
    
        # Check if error limits are met
        abserr_met = abserr <= atol
        relerr_met = relerr <= rtol
    
        # Decide what to do
        if np.all(abserr_met) and np.all(relerr_met):
            # Notify user
            if verbosity > 0:
                print(f'Interpolating a coarse grid of {[len(x[j]) for j in range(ndim)]} points to a fine grid of {[len(x2[j]) for j in range(ndim)]} points results in no absolute or relative errors.')
                
            break
        elif np.sum(abserr_met)/abserr.size >= atolfrac and np.sum(relerr_met)/relerr.size >= rtolfrac:
            # Notify user
            if verbosity > 0:
                print(f'Interpolating a coarse grid of {[len(x[j]) for j in range(ndim)]} points to a fine grid of {[len(x2[j]) for j in range(ndim)]} points results in {abserr_met.size-np.sum(abserr_met)}/{abserr_met.size} absolute errors and {relerr_met.size-np.sum(relerr_met)}/{relerr_met.size} relative errors. This meets the specified fractional error requirements.')
                
            break
        else:
            # Notify user
            if verbosity > 1:
                print(f'Interpolating a coarse grid of {[len(x[j]) for j in range(ndim)]} points to a fine grid of {[len(x2[j]) for j in range(ndim)]} points results in {abserr_met.size-np.sum(abserr_met)}/{abserr_met.size} absolute errors and {relerr_met.size-np.sum(relerr_met)}/{relerr_met.size} relative errors.')

            # Put the previously computed values into y2
            sfine = [slice(1,y2shape[j],2) for j in range(ndim)]
            scoarse = [slice(0,y2shape[j],2) for j in range(ndim)]
            sy = tuple(scoarse + [slice(0,outshape[j],1) for j in range(len(outshape))])
            y2[sy] = y1
            
            # Determine the points that have not yet been computed
            not_computed = np.full(2*x1shape-1, True, dtype=bool)
            not_computed[tuple(scoarse)] = False
            not_computed[tuple(sfine)] = False
            
            # Determine the indices of the points where we haven't computed y2 yet
            not_computed_indices = np.where(not_computed)
            
            # Loop over each point, compute if we haven't computed yet
            for j in range(len(not_computed_indices[0])):
                
                # Determine the index of each argument for the current point
                index = [not_computed_indices[k][j] for k in range(ndim)]
                
                # Extend index to allow for an arbitrary output shape of fnc
                yindex = tuple(index + [slice(0,outshape[j],1) for j in range(len(outshape))])
                
                # Determine the arguments of the current point
                args = [[x2[k][index[k]]] for k in range(ndim)]
                
                # Compute
                y2[yindex] = fnc(args)
                
                # Mark the current point as computed
                not_computed[tuple([ind] for ind in index)] = False

            # Make the fine grid for this round the coarse grid for the next round
            x = x2
            y1 = y2

    # Warn user if reached max number of divisions.
    if i == nmaxdiv-1:
        warnings.warn('Reached the maximum number of divisions, try increasing nmaxdiv')

    # The coarse grid met our error requirements successfully! Return this grid.
    return x, y1, rgi

def converge1d(fnc, x, method='cubic', divideall=True, atol=1e-3, rtol=1e-3, atolfrac=1, rtolfrac=1, nmaxdiv=12, verbosity=0):
    r"""Determines the x grid size needed for an interpolator of the provided function to converge to the required tolerance. The function must take a single argument. The user must also provide an initial grid on which to compute the function.
    
    :param fnc: The function to interpolate. This must take a single argument.
    :param x: The initial grid on which to interpolate :code:`fnc`.
    :param method: The interpolation method to use, i.e. :code:`'cubic'` or :code:`'linear'`. Interpolation is performed using the :code:`scipy.interpolate.RegularGridInterpolator` class. The method provided to :code:`converge1d` is directly passed to this class.
    :param divideall: Sets the method by which the grid is subdivided after each round.
    :param atol: The absolute error tolerance.
    :param rtol: The relative error tolerance.
    :param atolfrac: The required fraction of points without absolute errors.
    :param rtolfrac: The required fraction of points without relative errors.
    :param nmaxdiv: The maximum number of divisions to perform before the function exits.
    :param verbosity: The verbosity of the algorithm. :code:`0` produces no printouts, :code:`1` produces small notifications, and :code:`2` provides detailed updates.
    
    :code:`converge1d` works by starting with a coarse grid in x. It then creates a fine grid in x with half the spacing of the coarse grid. The coarse grid is used to interpolate points onto the fine grid. These points are compared with the actual values on the fine grid. If residuals exist that are larger than the absolute or relative error tolerance then the process loops, with the fine grid becoming the coarse grid for the next round. This continues until the tolerances are met or the max number of divisions (:code:`nmaxdiv`) is met.
    
    A slightly different scheme can be selected by setting :code:`divideall=False`. In this scheme the coarse grid for each round is selected by subdividing points between which an error was detected. Points that did not have an error between them are not subdivided.
    
    Note that :code:`fnc` can return a multidimensional output.
    
    Interpolation is performed using the :code:`scipy.interpolate.RegularGridInterpolator` class. The method provided to :code:`converge1d` is directly passed to this class.
    
    Absolute errores are computed at each point via
    
    .. math::
    
        \left|y' - y\right|
        
    where :math:`y'` is the interpolated value and :math:`y` is the actual value. Relative errors are computed via
    
    .. math::
    
        \frac{\left|y' - y\right|}{\mathrm{max}(\left|y'\right|,\left|y\right|)}
        
    :code:`atolfrac` and :code:`rtolfrac` allow convergence to occur when only a fraction of points meet the specified error requirements. For example, setting :code:`atolfrac=0.99` means that 99% of the points must pass the absolute error check to satisfy the absolute error requirement. :code:`atolfrac` and :code:`rtolfrac` are set to 1 by default, meaning that no absolute or relative errors are allowed.
    
    Example usage
    
    .. code-block:: python

        import numpy as np
        from mwave.interpolation import converge1d
        from matplotlib import pyplot as plt

        # Define a grid
        x = np.linspace(0,10,5)
        xdense = np.linspace(x[0],x[-1],1000)

        # Define a funky function
        avec = np.linspace(0.0, 0.05, 6)
        onevec = np.ones(len(avec))
        fnc = lambda x: np.outer(np.sin(2*x**1.6)*np.exp(-0.02*x), onevec) + np.exp(np.outer(x,avec)) + 1

        # Plot
        plt.plot(xdense, fnc(xdense))
        plt.title('example function')
        plt.show()

        # Use converge1d to create an accurate interpolator for the function
        xgrid, ygrid, ifnc = converge1d(fnc, x, divideall=True, verbosity=2)

        # Plot the resiudals between the interpolator and the actual function
        fig, [ax1, ax2, ax3] = plt.subplots(nrows=3)
        ax1.plot(xgrid, fnc(xgrid)-ifnc(xgrid))
        ax1.set_ylabel('residuals')
        ax2.plot(np.diff(xgrid))
        ax2.set_ylabel('grid dx')
        ax3.plot(xgrid, ygrid, 'o')
        ax3.set_ylabel('grid points')
        plt.show()
    
    """

    # Compute on coarse grid
    y1 = fnc(x)
    
    # Get shape of y
    outshape = np.shape(y1)
    outdims = len(outshape) - 1
    
    # Create function to slice y at the specified x points and include all outputs
    yargslicer = lambda start,stop,step: tuple([slice(start, stop, step)] + [slice(None) for i in range(outdims)])

    for i in range(nmaxdiv):

        # Generate fine grid
        x2 = np.empty(2*len(x)-1)
        x2[::2] = x
        x2[1::2] = x[:-1] + np.diff(x)/2

        # Notify user
        if verbosity > 0:
            print(f'Computing grid on {len(x2)} points.')
            
        # Create slices of coarse points and new points
        slice_coarse = yargslicer(0,len(x2),2)
        slice_new = yargslicer(1,len(x2),2)
    
        # Generate fine grid, fill with coarse computations, compute fine
        y2 = np.empty((2*len(x)-1, *outshape[1:]), dtype=y1.dtype)
        y2[slice_coarse] = y1
        y2[slice_new] = fnc(x2[1::2])
    
        # Interpolate on larger grid
        rgi = RegularGridInterpolator((x,), y1, method=method)
    
        # Interpolate
        y2interp = rgi(x2)
    
        # Compute error of the interpolated points
        abserr = np.abs(y2interp[slice_new] - y2[slice_new])
        relerr = abserr/np.fmax(np.abs(y2[slice_new]), np.abs(y2interp[slice_new]))
    
        # Check if error limits are met
        abserr_met = abserr <= atol
        relerr_met = relerr <= rtol
    
        # Decide what to do
        if np.all(abserr_met) and np.all(relerr_met):
            # Notify user
            if verbosity > 0:
                print(f'Interpolating a coarse grid of {len(x)} points to a fine grid of {len(x2)} points results in no absolute or relative errors.')
                
            break
        elif np.sum(abserr_met)/abserr.size >= atolfrac and np.sum(relerr_met)/relerr.size >= rtolfrac:
            # Notify user
            if verbosity > 0:
                print(f'Interpolating a coarse grid of {len(x)} points to a fine grid of {len(x2)} points results in {abserr_met.size-np.sum(abserr_met)}/{abserr_met.size} absolute errors and {relerr_met.size-np.sum(relerr_met)}/{relerr_met.size} relative errors. This meets the specified fractional error requirements.')
                
            break
        else:
            # Notify user
            if verbosity > 1:
                print(f'Interpolating a coarse grid of {len(x)} points to a fine grid of {len(x2)} points results in {abserr_met.size-np.sum(abserr_met)}/{abserr_met.size} absolute errors and {relerr_met.size-np.sum(relerr_met)}/{relerr_met.size} relative errors.')

            # Subdivide the grid appropriately based on if divideall is True or False
            if divideall:
                # Make the fine grid for this round the coarse grid for the next round
                x = x2
                y1 = y2
            else:
                # Determine which points in x had no absolute/relative errors
                abserr_collapsed = np.all(abserr_met, axis=tuple(range(1,outdims+1)))
                relerr_collapsed = np.all(relerr_met, axis=tuple(range(1,outdims+1)))
                
                # Determine which points in x had errors
                errpnts = np.logical_not(np.logical_and(abserr_collapsed, relerr_collapsed))
                
                # Make a logical array of the points in x2 to put into x for the next round
                points_to_keep = np.full(len(x2), False, dtype=bool)
                points_to_keep[1::2] = errpnts
                points_to_keep[0::2] = True
                
                # Update x and y1 to contain the coarse points along with the points that had errors
                yslice = tuple([points_to_keep] + [slice(None) for i in range(outdims)])
                x = x2[points_to_keep]
                y1 = y2[yslice]

    # Warn if we reached the maximum number of divisions
    if i == nmaxdiv-1:
        warnings.warn('Reached the maximum number of divisions, try increasing nmaxdiv')

    # The coarse grid met our error requirements successfully! Return this grid along with the interpolator.
    return x, y1, rgi

def test_nd_interpolator(fnc, ifnc, bounds, npoints, seed=None, disp=False):
    """
    This function provides a quick way to test interpolator functions against actual functions. It works by randomly dropping points into the space specified by bounds and comparing the result of :code:`fnc` and :code:`ifnc`.
    
    :param fnc: The actual function.
    :param ifnc: The interpolating function.
    :param bounds: The bounds over which to compare the two functions.
    :param npoints: The number of points to drop into the space.
    :param seed: The seed to pass to the numpy random number generator.
    :param disp: If true basic statistics about the residuals are printed out.
    
    .. code-block:: python
    
        import numpy as np
        from mwave.interpolation import convergend, test_nd_interpolator
        from matplotlib import pyplot as plt

        # Define a grid
        x = np.linspace(0,10,10)
        y = np.linspace(0,5,5)

        # Define a funky function
        def fnc(args):
            x, y = args
            return np.sin(2*x**1.6)*np.exp(-0.02*y) + np.exp(x*0.2) + 1

        # Define a version we can pass to the interpolator
        def fnc_interp(args):
            x, y = np.meshgrid(*args, indexing='ij')
            return fnc((x, y))

        # Use converge1d to create an accurate interpolator for the function
        xgrid, ygrid, ifnc = convergend(fnc_interp, (x, y))

        # Test the interpolator by dropping 1000 points into the space
        args, y_actual, y_interp = test_nd_interpolator(fnc, ifnc, [[0,10], [0,5]], 1000, disp=True)

        # Compute residuals
        resid = y_actual - y_interp

        # Plot
        plt.scatter(args[0], args[1], c=resid)
        plt.colorbar()
        plt.show()
        
    """
    
    # Create the random number generator
    rng = np.random.default_rng(seed=seed)
    
    # Determine the dimension of the input to fnc
    ndim = len(bounds)
    
    # Generate the random arguments to pass to fnc
    args = [rng.uniform(low=bounds[i][0], high=bounds[i][1], size=npoints) for i in range(ndim)]
    
    # Pair up the arguments so that we can pass them to the interpolation function
    args_paired = np.array(args).T
    
    # Compute the actual value of the function
    y_actual = fnc(args)
    
    # Compute the interpolated values of the function
    y_interp = ifnc(args_paired)
    
    # Compute the residuals, display statistics if requested
    if disp:
        resid = y_actual-y_interp
        print('Sampled %i points.\nMaximum residual=%0.3e\nmean residual=%0.3e\nmedian residual=%0.3e' % (npoints, np.max(resid), np.median(resid), np.mean(resid)))
    
    # Return
    return args, y_actual, y_interp

class ParameterSpace():
    pass

class ParameterChunk():
    pass

import numpy as np

class Computation():
    """
    Provides a framework which generates all permutations of a set of function arguments. Also provides the ability to call the function with a specific permutation.

    Example usage:

    .. code-block:: python

        import multiprocessing as mp
        import numpy as np
        import h5py
        from tdse.precompute import Computation
        from time import sleep

        # Set the filename of the file we want to write to
        fname = 'tmp.hdf5'

        # Define the function that we are interested in precomputing
        def do_calculation(a, b, c):
            return a*b + c

        # Define a new Computation object and define the parameters we wish to precompute.
        # We can use the functionality built into Computation to easily iterate over all
        # permutations of these parameters.
        comp = Computation(do_calculation, [np.array([1, 2]), np.array([6, 7, 8]), np.array([9])])

        # We are going to parallelize our computation, so we must define a function which
        # can be called asychronously.
        def run_computation_by_idx(idx, lock):

            # Sleep to simulate some hard calculation
            sleep(1)
            
            # Perform computation
            result = comp.compute(idx)

            # Wait for lock and then save to file
            with lock:
                f = h5py.File(fname, "r+")
                f['calc'][idx] = result
                f['completed'][idx] = True
                f.close()

        # Create function that prints out the current progress
        ncompleted = 0 # Create a variable to track the number of completed runs
        def update_progress(_):
            global ncompleted
            ncompleted += 1
            print("%i/%i" % (ncompleted, comp.npoints), end=\'\\r\')

        if __name__ == '__main__':

            # Print CPU count
            print("The CPU count is %i" % mp.cpu_count())

            # Create a new HDF5 file that tracks the completed
            # computations, the result of the computations and
            # the arguments
            with h5py.File(fname, "w") as f:
                f.create_dataset("calc", (comp.npoints,), dtype=np.float64)
                f.create_dataset("completed", data=np.ones(comp.npoints)==0)
                g_args = f.create_group("args")
                g_args.attrs['function'] = comp.comp_fnc.__name__
                for i, arg in enumerate(comp.args):
                    g_args.create_dataset("arg%i" % i, data=arg)
            
            # Create multiprocessing objects
            pool = mp.Pool()
            manager = mp.Manager()
            lock = manager.Lock()

            # Set block to false for asynchronous calls, true for synchronous calls
            block = False

            # Loop over each parameter combination and call run_computation_by_idx
            for i in range(comp.npoints):
                if block:
                    pool.apply(run_computation_by_idx, (i, lock))
                else:
                    pool.apply_async(run_computation_by_idx, (i, lock), callback=update_progress)

            # Close pool and join all processes
            pool.close()
            pool.join()
    """

    def __init__(self, comp_fnc, args):
        """
        Creates a Computation object for the provided function and set of arguments.

        :param comp_fnc: The function to compute. Must accept the same number of arguments as len(args).
        :param args: A list containing lists of each value to run for each argument.
        """

        # Store the computation function and the argument list
        self.comp_fnc = comp_fnc
        self.args = args

        # Determine the number of values provided for each argument
        self.arglens = np.array([len(arg) for arg in self.args])
        
        # Determine the total number of points to test
        self.npoints = np.product(self.arglens)

    def get_number_of_points(self):
        """
        Returns the number of permutations made out of the provided arguments.
        
        :returns: The number of permutations made out of the provided arguments.
        """
        return self.npoints

    def get_arguments_at_index(self, idx):
        """
        Returns the function arguments at the provided linear index.

        :param idx: The index at which to determine the function arguments.
        :returns: A list containing each argument at the specified index.
        """
        # Unravel the linear index into the multidimensional index
        argidxs = np.unravel_index(idx, self.arglens)
        
        # Loop through each argument and grab the argument value at the multidimensional indexw
        args = [self.args[i][argidxs[i]] for i in range(len(self.args))]

        # Return the arguments
        return args
            
    def compute(self, idx):
        """
        Computes the function for the given parameter index.

        :param idx: The parameter index.
        :returns: The result of the computation at the specified index.
        """
        # Check that the given index is valid
        if idx < 0 or idx >= self.npoints:
            raise ValueError("idx is less than zero or greater than the total number of possible argument permutations.")

        # Determine the parameters at the given index, unpack them into the computation function, call the function, and return.
        return self.comp_fnc(*self.get_arguments_at_index(idx))