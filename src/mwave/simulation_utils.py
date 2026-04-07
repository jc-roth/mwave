import numpy as np
import numbers

def cloud_init(natoms, sigma_cloud, sigma_transverse_v, sigma_vertical_v, x_offset=0, y_offset=0, z_offset=0, vx_offset=0, vy_offset=0, vz_offset=0, seed=None):
    """Basic cloud initialization. For more complicated atom clouds (i.e. clouds where positions/velocities are correlated with each other), the user should write their own initialization function.
    
    :param natoms: The number of atoms to initialize.
    :param sigma_cloud: The standard deviation of atom position within the cloud. The user can either provide a single number (in which case the standard deviation is the same in all directions), or three numbers (in which case the standard deviation is defined as :code:`sigma_cloud=(sigma_x, sigma_y, sigma_z)`).
    :param sigma_transverse_v: The standard deviation of atom velocity within the cloud. The user can either provide a single number (in which case the standard deviation is same in both transverse directions), or two numbers (in which case the standard deviation is defined as :code:`sigma_transverse_v=(sigma_x, sigma_y)`).
    :param sigma_vertical_v: The standard deviation of atom velocity with the vertical direction of the cloud.
    :param x_offset: The offset of the cloud from :math:`x=0`.
    :param y_offset: The offset of the cloud from :math:`y=0`.
    :param z_offset: The offset of the cloud from :math:`z=0`.
    :param vx_offset: The offset of the cloud velocity from :math:`v_x=0`.
    :param vy_offset: The offset of the cloud velocity from :math:`v_y=0`.
    :param vz_offset: The offset of the cloud velocity from :math:`v_z=0`.
    :param seed: The seed to use when drawing the positions and velocities. If this function is called within an optimization loop the seed should remain the same across function calls!
    :returns: A tuple of :code:`(x0, y0, z0, vz, vx, vy)`, where each element of the tuple is a numpy array of length :code:`natoms`.
    """
        
    # Set seed if given
    if seed is not None:
        np.random.seed(seed)
        
    # Put sigma_cloud into tuple if just a single number was passed
    if isinstance(sigma_cloud, numbers.Number):
        sigma_cloud = (sigma_cloud, sigma_cloud, sigma_cloud)
        
    # Put sigma_transverse_v into tuple if just a single number was passed
    if isinstance(sigma_transverse_v, numbers.Number):
        sigma_transverse_v = (sigma_transverse_v, sigma_transverse_v)
        
        
    x0 = np.random.randn(natoms)*sigma_cloud[0] + x_offset
    y0 = np.random.randn(natoms)*sigma_cloud[1] + y_offset
    z0 = np.random.randn(natoms)*sigma_cloud[2] + z_offset
    
    vx = np.random.randn(natoms)*sigma_transverse_v[0] + vx_offset
    vy = np.random.randn(natoms)*sigma_transverse_v[1] + vy_offset
    vz = np.random.randn(natoms)*sigma_vertical_v + vz_offset
    
    # Return
    return x0, y0, z0, vz, vx, vy
