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
        
def load_lookup_table(fname, kvec=None, method='cubic'):
    """Loads a single lookup table where :math:`n_0=0`, returns a function that interpolates this table using the provided :code:`method`. The lookup table must be a 2D grid over :math:`\Omega` and :math:`\delta`. If :code:`kvec` is provided then the function aligns its k-vector with the one defined by :code:`kvec`.
    
    :param fname: The filename to load the precompute table from. Internally the read is performed using :code:`read_bragg_precompute`.
    :param kvec: Optional. If provided only the values in :code:`kvec` will be returned by the interpolation function.
    :param method: The interpolation method to use. This is directly passed to :py:meth:`scipy.interpolate.RegularGridInterpolator`.
    :return: A tuple containing :code:`(kvec, interpolation_function)`, where the :code:`interpolation_function` is returned by :py:meth:`scipy.interpolate.RegularGridInterpolator`."""
    
    # Load table, check it is correct
    phi, kvec_precomputed, grid = read_bragg_precompute(fname, 0, 0, 0)
    if grid[0][1] != 'omegas' or grid[1][1] != 'deltas':
        raise ValueError('The provided lookup table must be a 2D grid over omegas and deltas')
    
    # Create interpolation function
    phi_interpolated = RGI((grid[0][0], grid[1][0]), phi, method=method)
    
    if kvec is None:
        # If no alignment is requested just return
        return kvec_precomputed, phi_interpolated
    else:
        # Attempt to align kvec and kvec_precomputed
        align_idxs = np.isin(kvec_precomputed, kvec)
        
        if np.array_equal(kvec_precomputed[align_idxs], kvec):
            # Can align kvec_precomputed and kvec, define a precompute function and return
            def precompfnc(args):
                return phi_interpolated(args)[...,align_idxs]
            return kvec_precomputed[align_idxs], precompfnc
        else:
            # Cannot perform alignment, raise error
            raise ValueError('Could not align kvec_precomputed with kvec, please use a precompute table that can be aligned with kvec.')
        
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
def load_precomputed_gbragg(single_path, multi_path=None, table_sigma=None, table_modulation_frequency=None, flip_negatives=True):
    
    # Multifrequency and precompute table logic checking
    disable_multifrequency = True
    if multi_path:
        disable_multifrequency = False
        
    check_sigma = False
    if table_sigma:
        check_sigma = True
        
    check_mod_freq = False
    if table_modulation_frequency:
        check_mod_freq = True
        
    # Throw error if table_modulation_frequency is provided but multi_path is not
    if table_modulation_frequency and not multi_path:
        raise ValueError('If table_modulation_frequency a valid multi_path must be provided as well.')

    # Load single frequency precompute table
    print('Loading single frequency Bragg precompute table, this could take a while...')
    kvec_precomp_single, fnc_interp_single = load_lookup_table(single_path)
    print('Precompute table loaded! Performing checks...')

    # Check that kvec_precomp can be flipped properly if flip_negatives is True
    if flip_negatives:
        if not np.array_equal(-kvec_precomp_single, np.flip(kvec_precomp_single)):
            raise ValueError('flip_negatives is True but the precompute table kvector cannot be flipped properly.')

    print('Checks passed!') 

    # If multifrequency table reference is passed in, load
    if not disable_multifrequency:
        # Load pre
        # compute table
        print('Loading multifrequency Bragg precompute table, this could take a while...')
        kvec_precomp_multi, fnc_interp_multi = load_lookup_table(multi_path)
        print('Precompute table loaded! Performing checks...')

        # Check that kvec_precomp can be flipped properly if flip_negatives is True
        if flip_negatives:
            if not np.array_equal(-kvec_precomp_multi, np.flip(kvec_precomp_multi)):
                raise ValueError('flip_negatives is True but the precompute table kvector cannot be flipped properly.')

        print('Checks passed!')

    # Define precompute function
    def gbragg_precomp(kvec, k0, sigma, omega, delta, delta_phase, mod_freq=None, mod_phase=0.0):
        
        if mod_phase != 0.0:
            raise ValueError('Only a modulation phase of 0.0 is supported currently.')
        
        # Check sigma
        if check_sigma and sigma != table_sigma:
            raise ValueError(f'Provided sigma is {sigma}, inconsistent with precompute sigma of {table_sigma}.')
        
        # Check modulation frequency
        if not disable_multifrequency and check_mod_freq and mod_freq and mod_freq != table_modulation_frequency:
            raise ValueError(f'Provided mod_freq is {mod_freq}, inconsistent with precompute mod_freq of {table_modulation_frequency}.')
        if disable_multifrequency and mod_freq:
            raise ValueError(f'No multifrequency precompute table was provided, mod_freq must be None.')
        
        # Select precompute table based on input params and disable_multifrequency state
        kvec_precomp = kvec_precomp_single
        fnc_interp = fnc_interp_single
        if not disable_multifrequency and mod_freq:
            kvec_precomp = kvec_precomp_multi
            fnc_interp = fnc_interp_multi
        
        # Transform delta to the frame moving with the state we are currently in
        deltas_transformed = delta - 4*k0

        # If flip_negatives is True we use the symmetry of the problem
        # If we start at n=0 then we can map delta -> -delta if we flip the output momentum states
        if flip_negatives:
            neg_idxs = deltas_transformed < 0.0
            
            phi_out = np.full((len(omega), len(kvec_precomp)), np.nan, dtype=np.complex128)
            
            # Make negative deltas positive, compute wavefunctions, flip output
            phi_out[neg_idxs, :] = np.flip(fnc_interp((omega[neg_idxs], -deltas_transformed[neg_idxs])), axis=1)
            
            # Compute positive delta wavefunctions
            phi_out[~neg_idxs, :] = fnc_interp((omega[~neg_idxs], deltas_transformed[~neg_idxs]))
                        
        else:
            # Compute Bloch Hamiltonian function
            phi_out = fnc_interp((omega, deltas_transformed))
                        
        # Apply phase
        phi_out *= np.exp(-1j*(delta_phase)*kvec_precomp/2)
        
        # Shift the output states
        Deltan = k0//2
        phi_out_filled = np.zeros_like(phi_out, dtype=np.complex128)
        
        if Deltan == 0:
            phi_out_filled[:,:] = phi_out[:,:]
        elif Deltan < 0:
            phi_out_filled[:,:Deltan] = phi_out[:,-Deltan:]
        else:
            phi_out_filled[:,Deltan:] = phi_out[:,:-Deltan]
        
        # Return
        alignment_idxs = np.isin(kvec_precomp, kvec)
        return phi_out_filled[:, alignment_idxs]
    
    # Return
    return gbragg_precomp