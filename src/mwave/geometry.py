import numpy as np

def cloud_to_scrbi_ellipse_xy(x0, y0, z0, vx, vy, vz, T, Tp, n, N, phi_c, phi_d, bragglookup, omegalookup, lplookup, deltalookup):
    """Computes the :math:`x` and :math:`y` points of a simultaneous conjugate Ramsey-Borde interferometer (SCRBI) ellipse from the provided vectors of atom positions and velocities. In addition to the positions :code:`x0,y0,z0` and velocities :code:`vx,vy,vz` the user must provide big :math:`T` as :code:`T` (in seconds), :math:`T'` as :code:`Tp` (in seconds), :math:`n_\mathrm{Bragg}` as :code:`n`, and :math:`N_\mathrm{Bloch}` as :code:`N`. The average population at each output port is then used to compute :code:`x` and :code:`y`.
    
    In order to generate a single ellipse from a single function call the user can pass a vector of common mode phases :code:`phi_c`. This common mode phase is then added to the appropriate wavefunctions before the calculation of the output port populations to produce an ellipse. The length of the returned :code:`x` and :code:`y` is given by the length of :code:`phi_c`. The user may also pass in a value for :code:`phi_d`, which sets the opening of the ellipse.
    
    The user must provide the functions :code:`bragglookup`, :code:`omegalookup`, and :code:`lplookup`. See below for more information on each of these functions.
    
    Currently the function does not account for gravity in the calculation of z positions at each beamsplitter.
    
    Currently the function does not account for Junk ports.
    
    Currently the function accounts for the phase produced from the Bragg beamsplitter process (determined by calling :code:`bragglookup`), and the local wavefront phase (determined by calling :code:`lplookup`). Unless the user implements it manually in :code:`bragglookup` or :code:`lplookup` the global laser phase (i.e. :math:`\\omega t-kz`) will be ignored. Other ignored sources of phase include the free evolution phase, gravity gradient phase, and separation phases.
    
    As it is difficult to program a completely general function for all of the scenarios that we might be interested in, it might be a good idea to open up the source code of this function and write a custom version!
    
    :param x0: Vector of initial atom x positions.
    :param y0: Vector of initial atom y positions.
    :param z0: Vector of initial atom z positions.
    :param vx: Vector of initial atom x velocities.
    :param vy: Vector of initial atom y velocities.
    :param vz: Vector of initial atom z velocities.
    :param T: Big :math:`T` in seconds.
    :param Tp: :math:`T'` in seconds.
    :param n: :math:`n_\mathrm{Bragg}`.
    :param N: :math:`N_\mathrm{Bloch}`.
    :param phi_c: Vector of common mode phases.
    :param phi_d: Vector of differential mode phases.
    :param bragglookup: Function that returns the phase difference between the provided initial and final states after undergoing a Bragg pulse. Must accept arguments :code:`ni, nf, omega, delta`, where :code:`omega` and :code:`delta` can be vectors.
    :param omegalookup: Function that returns the effective Rabi frequency at the specified location. Must accept arguments :code:`x, y, z`, where all arguments are vectors of the same length.
    :param lplookup: Function that returns the local laser phase at the specified location. Must accept arguments :code:`x, y, z`, where all arguments are vectors of the same length.
    :param deltalookup: Function that returns the detuning specified velocity. Must accept arguments :code:`v` in whatever units are used by the user to define the cloud velocity spread. The returned detuning must be in units of the recoil frequency.
    :return: The x and y values of the ellipse. These will have the same length as the provided vector :code:`phi_c`.
    
    .. code-block:: python
    
        import numpy as np
        from mwave.precompute import load_fast_bragg_evaluator
        from mwave.geometry import cloud_to_scrbi_ellipse_xy
        from alphautil.analysis import fit_ellipse_coeff
        from alphautil.ellipse import get_sci_params
        from matplotlib import pyplot as plt

        np.random.seed(13703599)

        npoints = 100000
        x0 = np.random.randn(npoints)*0.75e-3
        y0 = np.random.randn(npoints)*0.75e-3
        z0 = np.random.randn(npoints)*0.75e-3
        v0 = np.random.randn(npoints)*0.35e-3
        vx = np.random.randn(npoints)*3.5e-3*1.8
        vy = np.random.randn(npoints)*3.5e-3*1.8
        T = 100e-3
        Tp = 10e-3
        phi_c = np.linspace(0,2*np.pi)
        phi_d = np.pi/4
        Omega0 = 32
        w0 = 10e-3

        n_init = 0
        n_bragg = 5
        N_bloch = 100

        bragglookup = load_fast_bragg_evaluator('sig0.260.h5', n_init, n_bragg, N_bloch)

        def omegalookup(x, y, z):
            return Omega0*np.exp(-2*(x**2 + y**2)/(w0**2))

        # (l)ocal (p)hase (lookup)
        def lplookup(x, y, z):
            wavelen = 852e-9
            zR = np.pi*w0**2/wavelen
            kk = 2*np.pi/wavelen
            return kk*(x**2 + y**2)/(2*zR)
            
        def deltalookup(v, n_bragg):
            return 4*n_bragg + 4*(v/0.0035) # The modification to delta is 4 times the velocity defined in units of recoil velocities

        x, y = cloud_to_scrbi_ellipse_xy(x0, y0, z0, vx, vy, v0, T, Tp, n_bragg, N_bloch, np.exp(1j*phi_c), np.exp(1j*phi_d), bragglookup, omegalookup, lplookup, deltalookup)

        coeff = fit_ellipse_coeff(x, y)
        bx, by, Ax, Ay, phi_d_fit = get_sci_params(coeff)

        plt.scatter(x, y)
        plt.ylim([-1,1])
        plt.xlim([-1,1])
        plt.gca().set_aspect('equal')
        plt.show()

        print('differential phase fit error=%0.3f mRad' % ((phi_d - phi_d_fit)*1e3))
        print(Ax)
        print(Ay)
    """

    # Compute the ones array
    ones = np.ones(len(phi_c))

    # Compute position at each pulse
    x1, y1, z1 = x0 + vx*T, y0 + vy*T, z0 + vz*T
    x2, y2, z2 = x1 + vx*Tp, y1 + vy*Tp, z1 + vz*Tp
    x3, y3, z3 = x2 + vx*T, y2 + vy*T, z2 + vz*T

    # Compute velocity at each pulse (assuming no gravity)
    v0 = vz
    v1 = v0
    v2 = v1
    v3 = v2

    # Compute intensities at each pulse location
    omega0 = omegalookup(x0, y0, z0)
    omega1 = omegalookup(x1, y1, z1)
    omega2 = omegalookup(x2, y2, z2)
    omega3 = omegalookup(x3, y3, z3)

    # Lookup local phase
    phase0 = lplookup(x0, y0, z0)*n
    phase1 = lplookup(x1, y1, z1)*n
    phase2 = lplookup(x2, y2, z2)*n
    phase3 = lplookup(x3, y3, z3)*n

    # Compute detunings at each pulse location
    delta0 = deltalookup(v0, n)
    delta1 = deltalookup(v1, n)
    delta2 = deltalookup(v2, n)
    delta3 = deltalookup(v3, n)

    # Perform wavefunction calculations common to several paths
    wf0_0_0 = bragglookup(0, 0, omega0, delta0)
    wf0_0_n = bragglookup(0, n, omega0, delta0)

    wf1_n_n = bragglookup(n, n, omega1, delta1)
    wf1_0_n = bragglookup(0, n, omega1, delta1)
    wf1_n_0 = bragglookup(n, 0, omega1, delta1)
    wf1_0_0 = bragglookup(0, 0, omega1, delta1)

    wf2_npN_npN = bragglookup(n+N, n+N, omega2, delta2)
    wf2_npN_2npN = bragglookup(n+N, 2*n+N, omega2, delta2)
    wf2_mN_mN = bragglookup(-N, -N, omega2, delta2)
    wf2_mN_mnmN = bragglookup(-N, -N-n, omega2, delta2)

    # Compute wavefunctions
    portA_1 = wf0_0_0*wf1_0_n*np.exp(1j*phase1)*wf2_npN_2npN*np.exp(1j*phase2)*bragglookup(2*n+N, 2*n+N, omega3, delta3)
    portA_2 = wf0_0_n*np.exp(1j*phase0)*wf1_n_n*wf2_npN_npN*bragglookup(n+N, 2*n+N, omega3, delta3)*np.exp(1j*phase3)

    portB_1 = wf0_0_0*wf1_0_n*np.exp(1j*phase1)*wf2_npN_2npN*np.exp(1j*phase2)*bragglookup(2*n+N, n+N, omega3, delta3)*np.exp(-1j*phase3)
    portB_2 = wf0_0_n*np.exp(1j*phase0)*wf1_n_n*wf2_npN_npN*bragglookup(n+N, n+N, omega3, delta3)

    portC_1 = wf0_0_0*wf1_0_0*wf2_mN_mN*bragglookup(-N, -N, omega3, delta3)
    portC_2 = wf0_0_n*np.exp(1j*phase0)*wf1_n_0*np.exp(-1j*phase1)*wf2_mN_mnmN*np.exp(-1j*phase2)*bragglookup(-N-n, -N, omega3, delta3)*np.exp(1j*phase3)

    portD_1 = wf0_0_0*wf1_0_0*wf2_mN_mN*bragglookup(-N, -N-n, omega3, delta3)*np.exp(-1j*phase3)
    portD_2 = wf0_0_n*np.exp(1j*phase0)*wf1_n_0*np.exp(-1j*phase1)*wf2_mN_mnmN*np.exp(-1j*phase2)*bragglookup(-N-n, -N-n, omega3, delta3)

    # Interfere
    wfA = np.einsum('i,j->ij', portA_1, ones) + np.einsum('i,j->ij', portA_2, phi_c*phi_d)
    wfB = np.einsum('i,j->ij', portB_1, ones) + np.einsum('i,j->ij', portB_2, phi_c*phi_d)
    wfC = np.einsum('i,j->ij', portC_1, phi_c) + np.einsum('i,j->ij', portC_2, ones*phi_d)
    wfD = np.einsum('i,j->ij', portD_1, phi_c) + np.einsum('i,j->ij', portD_2, ones*phi_d)

    # Compute populations
    pA, pB, pC, pD = np.sum(np.abs(wfA)**2, axis=0), np.sum(np.abs(wfB)**2, axis=0), np.sum(np.abs(wfC)**2, axis=0), np.sum(np.abs(wfD)**2, axis=0)

    # Compute ellipse and return
    x, y = (pA - pB)/(pA + pB), (pC - pD)/(pC + pD)
    return x, y