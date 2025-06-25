import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator as RGI

def write_bragg_precompute(fname, phi, kvec, grid, n0, nf, n_bragg, N_bloch=None):
    """Saves the precomputed dataset in an HDF5 file with the given name.

    :param fname: The name of the HDF5 file to write to.
    :param phi: The table of phi values to save.
    :param kvec: The momentum space vector.
    :param grid: A list of tuples describing the parameter grid over which phi was applied. Each tuple contains information about a grid axis over which phi was computed. The first element contains a vector of parameter values, the second element is the name associated with the parameter.
    :param n0: The initial momentum state.
    :param nf: The final momentum state.
    :param n_bragg: The Bragg order used.
    :param N_bloch: Optional. The Bloch order used in the simulation. This implies a multifrequency simulation was used.
    """

    # Check grid shape matches phi shape
    grid_shape = []
    for g in grid:
        if len(np.shape(g[0]))==1:
            grid_shape.append(len(g[0]))
        else:
            raise ValueError("Each grid axis must have dimension 1")
    if not np.array_equal(np.shape(phi)[:-1], grid_shape):
        raise ValueError('Dimensions of phi are not equal to those defined by grid_shape')

    # Write to HDF5 file
    with h5py.File(fname, 'a') as f:
        g_bragg = f.require_group(f'bragg{n_bragg}')

        if N_bloch is not None:
            g_bloch = g_bragg.require_group(f'bloch{N_bloch}')
            g_data = g_bloch.require_group(f'ni{n0}_nf{nf}')
        else:
            g_data = g_bragg.require_group(f'ni{n0}_nf{nf}')
        
        g_data.create_dataset('phi', data=phi, compression='gzip')
        g_data.create_dataset('kvec', data=kvec, compression='gzip')
        
        g_grid = g_data.require_group(f'grid')
        g_grid.attrs.create('grid_def', ','.join([g[1] for g in grid]))
        for g in grid:
            g_grid.create_dataset(g[1], data=g[0], compression='gzip')

def read_bragg_precompute(fname, n0, nf, n_bragg, N_bloch=None):
    """Reads a precomputed Bragg dataset from an HDF5 file.

    :param fname: The name of the HDF5 file to read from.
    :param n0: The initial momentum state of the Bragg process.
    :param nf: The final momentum state of the Bragg process.
    :param n_bragg: The Bragg order used.
    :param N_bloch:  Optional. The Bloch order used in the simulation. This will load a multifrequency simulation.
    :return: A tuple containing :code:`phi, kvec, grid`. For a description of :code:`grid` see :code:`write_bragg_precompute`.
    """

    # Define group path
    bloch_path = ''
    if N_bloch is not None:
        bloch_path = f'/bloch{N_bloch}'
    grp_path = f'bragg{n_bragg}{bloch_path}/ni{n0}_nf{nf}'

    # Open file, relevant group, and extract all info
    with h5py.File(fname, 'r') as f:
        if grp_path not in f:
            raise NotPrecomputedError(grp_path)
        g = f[grp_path] # get group
        grid_vars = g['grid'].attrs['grid_def'].split(',') # get grid variables
        return g['phi'][()], g['kvec'][()], [(g[f'grid/{var}'][()], var) for var in grid_vars] # return

class NotPrecomputedError(Exception):
    def __init__(self, path):
        super().__init__(f'Precompute table does not include {path}')
        
def load_fast_bragg_evaluator(fname, n_init, n_bragg, N_bloch):
    """This function loads a function that quickly evaluates Bragg pulse precompute tables for SCRBI geometries on a grid of inputs using the scipy regular grid interpolator with the cubic method enabled. This is useful for simulating a atom cloud with transverse motion.
    
    :param fname: The name of the HDF5 precompute table to load. :code:`phi` Datasets are loaded using the :code:`read_bragg_precompute` function. It is assumed :code:`phi` is computed on a grid of :code:`(omega,delta)`. If this is not the case this function will return gibberish!
    :param n_init: The initial momentum state, this is zero for most precompute tables.
    :param n_bragg: The Bragg order.
    :param N_bloch: The Bloch order (used when loading multifrequency pulses).
    :return: A function :code:`fbe` that takes parameters :code:`n0,nf,omega,delta`, which are the initial and target momentum states, the value of omega, and the value of delta. These arguments must be supplied as equal length numpy arrays."""

    # Load interpolators
    phi, bp1g_kvec, grid = read_bragg_precompute(fname, n_init, n_init + n_bragg, n_bragg)
    bp1g = RGI([grid[0][0], grid[1][0]], phi, method='cubic') # i.e. (b)ragg (p)ulse 1 (g)rid

    phi, bp2g_kvec, grid = read_bragg_precompute(fname, n_init + n_bragg, n_init, n_bragg)
    bp2g = RGI([grid[0][0], grid[1][0]], phi, method='cubic')

    phi, bp3dg_kvec, grid = read_bragg_precompute(fname, n_init-N_bloch, n_init-N_bloch-n_bragg, n_bragg, N_bloch)
    bp3dg = RGI([grid[0][0], grid[1][0]], phi, method='cubic')
    
    phi, bp3ug_kvec, grid = read_bragg_precompute(fname, n_init+N_bloch+n_bragg, n_init+N_bloch+2*n_bragg, n_bragg, N_bloch)
    bp3ug = RGI([grid[0][0], grid[1][0]], phi, method='cubic')
    
    phi, bp4dg_kvec, grid = read_bragg_precompute(fname, n_init-N_bloch-n_bragg, n_init-N_bloch, n_bragg, N_bloch)
    bp4dg = RGI([grid[0][0], grid[1][0]], phi, method='cubic')
    
    phi, bp4ug_kvec, grid = read_bragg_precompute(fname, n_init+N_bloch+2*n_bragg, n_init+N_bloch+n_bragg, n_bragg, N_bloch)
    bp4ug = RGI([grid[0][0], grid[1][0]], phi, method='cubic')

    # Generate fast Bragg evaluation function
    def fbe(n0, nf, omega, delta):
        if (n0 == n_init and nf == n_init+n_bragg) or (n0 == n_init and nf == n_init):
            nf_idx = np.argmin(np.abs(bp1g_kvec - 2*nf))
            return bp1g((omega, delta))[:,nf_idx]
        elif (n0 == n_init+n_bragg and nf == n_init) or (n0 == n_init+n_bragg and nf == n_init+n_bragg):
            nf_idx = np.argmin(np.abs(bp2g_kvec - 2*nf))
            return bp2g((omega, delta))[:,nf_idx]
        elif (n0 == n_init-N_bloch and nf == n_init-N_bloch-n_bragg) or (n0 == n_init-N_bloch and nf == n_init-N_bloch):
            nf_idx = np.argmin(np.abs(bp3dg_kvec - 2*nf))
            return bp3dg((omega, delta))[:,nf_idx]
        elif (n0 == n_init+N_bloch+n_bragg and nf == n_init+N_bloch+2*n_bragg) or (n0 == n_init+N_bloch+n_bragg and nf == n_init+N_bloch+n_bragg):
            nf_idx = np.argmin(np.abs(bp3ug_kvec - 2*nf))
            return bp3ug((omega, delta))[:,nf_idx]
        elif (n0 == n_init-N_bloch-n_bragg and nf == n_init-N_bloch-n_bragg) or (n0 == n_init-N_bloch-n_bragg and nf == n_init-N_bloch):
            nf_idx = np.argmin(np.abs(bp4dg_kvec - 2*nf))
            return bp4dg((omega, delta))[:,nf_idx]
        elif (n0 == n_init+N_bloch+2*n_bragg and nf == n_init+N_bloch+2*n_bragg) or (n0 == n_init+N_bloch+2*n_bragg and nf == n_init+N_bloch+n_bragg):
            nf_idx = np.argmin(np.abs(bp4ug_kvec - 2*nf))
            return bp4ug((omega, delta))[:,nf_idx]
        else:
            print(f'no match for n0={n0}, nf={nf}!')
            return 1

    # Return the fast Bragg evaluation function
    return fbe