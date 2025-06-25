# Imports
from numba import jit, complex128, float64
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from warnings import warn

@jit(nopython=True)
def bloch_rhs(t, phi, kvec, delta, omega, omega_args, phase, phase_args, transformed=False):
    """Evaluates the right hand side of the Schrodinger equation for the Bloch Hamiltonian. The function returns a vector, one for each state included in the Hamiltonian.

    The right hand side is defined in a general way so that a time-dependent field intensity and phase can be computed.

    .. math::

        \\text{returned vector}=i\\frac{\\Omega(t, a)}{2}\\left[e^{i(\\delta t+\\theta(t,b))}e^{i(-4k-4)t}\\lvert k\\rangle\\langle k + 2\\rvert + e^{-i(\\delta t+\\theta(t,b))}e^{i(4k-4)t}\\lvert k\\rangle\\langle k-2\\rvert\\right]\\rvert\\phi\\rangle
    
    where :math:`k` indexes momentum states spaced by two photon recoils. The time :math:`t` is evaluated at :code:`t`, and the state :math:`\\lvert\\phi\\rangle` is specified by :code:`phi`.
    
    The states :math:`k` included in the calculation are specified by :code:`kvec`. :math:`\\delta` is specified by :code:`delta`. :math:`\\Omega(t, a)` is specified by :code:`omega`, which must a function which takes arguments :code:`t` and :code:`omega_args`. :math:`\\theta(t, b)` is specified by :code:`phase`, which must a function which takes arguments :code:`t` and :code:`phase_args`.

    If :code:`transformed` is :code:`True` then the right hand side is evaluated in the following frame:

    .. math::

        \\text{returned vector}=-ik^2\\lvert k\\rangle\\langle k\\rvert\\phi\\rangle + i\\frac{\\Omega(t, a)}{2}\\left[e^{i(\\delta t+\\theta(t,b))}\\lvert k\\rangle\\langle k + 2\\rvert + e^{-i(\\delta t+\\theta(t,b))}\\lvert k\\rangle\\langle k-2\\rvert\\right]\\rvert\\phi\\rangle
    
    To solve the Bloch Hamiltonian in time the :code:`bloch_rhs` function can be integrated using :meth:`scipy.integrate.solve_ivp`.
    
    :param t: The time at which to evaluate the right hand side.
    :param phi: The value of phi at which to evaluate the right hand side.
    :param kvec: The momentum state values at which :code:`phi` is defined.
    :param delta: The value of :math:`\\delta` (the two-photon detuning).
    :param omega: The function that returns the value of the effective Rabi frequency :math:`\\Omega(t, a)` at an arbitrary time. The function must take two arguments, :code:`t` and :code:`omega_args`. The argument :code:`t` specifies the time at which to evaluate the effective Rabi frequency and the argument :code:`omega_args` can be used to pass in additional parameters.
    :param omega_args: A tuple of arguments to pass to the function defined by :code:`omega`.
    :param phase: The function that returns the phase of two photon detuning at an arbitrary time. The function must take two arguments, :code:`t` and :code:`phase_args`. The argument :code:`t` specifies the time at which to evaluate the phase and the argument :code:`phase_args` can be used to pass in additional parameters. This function can be set to a constant value if the user does not want to simulate a frequency swept process.
    :param phase_args: A tuple of arguments to pass to the function defined by :code:`phase`.
    :param transformed: See the function description above.
    :returns: A vector containing the evaluated right hand side values."""
    
    # Compute phi_p1 and phi_m1 (pluse and minus 1)
    phi_p1 = np.zeros_like(phi)
    phi_p1[:-1] = phi[1:]
    phi_m1 = np.zeros_like(phi)
    phi_m1[1:] = phi[:-1]

    # Compute Rabi frequency and phase at current time
    oval = omega(t, omega_args)
    phaseval = phase(t, phase_args)
    
    # Compute RHS of ODE
    if not transformed:
        return 1j*oval/2*(np.exp(1j*(delta*t+phaseval))*np.exp(1j*(-4*kvec-4)*t)*phi_p1 + np.exp(-1j*(delta*t+phaseval))*np.exp(1j*(4*kvec-4)*t)*phi_m1)
    
    # Compute RHS of ODE in transformed frame
    return -1j*phi*kvec**2 + 1j*oval/2*(np.exp(1j*(delta*t+phaseval))*phi_p1 + np.exp(-1j*(delta*t+phaseval))*phi_m1)

@jit(nopython=True)
def bloch_density_rhs(t, rho, nstates, hkvec, vkvec, loss_mat, delta, omega, omega_args, phase, phase_args):
    """Evaluates the right hand side of the Von Neumann evolution equation for the Bloch Hamiltonian (i.e. :math:`[H,\\rho]`) where

    .. math::

        H=-\\hbar\\sum_{k}\\left[\\frac{\\Omega_\\text{eff}(t,a)}{2}e^{i(\\delta t+\\theta(t,b))}e^{i\\omega_\\text{r}(-4k-4)}|k\\rangle\\langle k+2|+\\frac{\\Omega_\\text{eff}(t,a)^*}{2}e^{-i(\\delta t+\\theta(t,b))}e^{i\\omega_\\text{r}(4k-4)}|k\\rangle\\langle k-2|\\right]
         
    where :math:`\\hbar=1` and the sum over :math:`k` is limited to the values of :math:`k` defined by :code:`hkvec` and :code:`vkvec`.
         
    The parameter :code:`rho` is supplied as a vector (this makes it compatible with :code:`scipy.integrate.solve_ivp`). This is then converted to a matrix via :code:`np.reshape(rho, (len(kvec), len(kvec)))` internally. The matrices :code:`hkvec` and :code:`vkvec` are composed of horizontal or vertical vectors of the momentum state grid stacked togeather.
    
    The parameter :code:`loss_mat` is the loss matrix.
    
    The remaining parameters (:code:`delta`, :code:`omega`, :code:`omega_args`, :code:`phase`, :code:`phase_args`) are equivalent to those used in the :code:`bloch_rhs` function.
    
    :param t: The time at which to evaluate the right hand side.
    :param rho: The value of rho at which to evaluate the right hand side.
    :param nstates: The number of states in :code:`rho`, used to properly reshape the density matrix.
    :param hkvec: The momentum state values at which :code:`rho` is defined along the horizontal axis.
    :param vkvec: The momentum state values at which :code:`rho` is defined along the vertical axis.
    :param loss_mat: The loss matrix to use.
    :param delta: The value of :math:`\\delta` (the two-photon detuning).
    :param omega: The function that returns the value of the effective Rabi frequency :math:`\\Omega(t, a)` at an arbitrary time. The function must take two arguments, :code:`t` and :code:`omega_args`. The argument :code:`t` specifies the time at which to evaluate the effective Rabi frequency and the argument :code:`omega_args` can be used to pass in additional parameters.
    :param omega_args: A tuple of arguments to pass to the function defined by :code:`omega`.
    :param phase: The function that returns the phase of two photon detuning at an arbitrary time. The function must take two arguments, :code:`t` and :code:`phase_args`. The argument :code:`t` specifies the time at which to evaluate the phase and the argument :code:`phase_args` can be used to pass in additional parameters. This function can be set to a constant value if the user does not want to simulate a frequency swept process.
    :param phase_args: A tuple of arguments to pass to the function defined by :code:`phase`.
    :returns: A vector containing the evaluated right hand side values."""

    # Compute Rabi frequency and phase at current time
    oval = omega(t, omega_args)
    phaseval = phase(t, phase_args)
    
    # Reshape rho into matrix
    rho_mat = np.reshape(rho, (nstates, nstates))

    # Create shifted matrices
    sr = np.zeros_like(rho_mat)
    sr[1:,:] = rho_mat[:-1,:]
    
    sl = np.zeros_like(rho_mat)
    sl[:-1,:] = rho_mat[1:,:]

    su = np.zeros_like(rho_mat)
    su[:,:-1] = rho_mat[:,1:]

    sd = np.zeros_like(rho_mat)
    sd[:,1:] = rho_mat[:,:-1]

    # # Compute each term in the RHS
    term1 = 1j*oval/2*np.exp(1j*(delta*t+phaseval))*np.exp(1j*(-4*vkvec-4)*t)*sl
    term2 = 1j*oval/2*np.exp(-1j*(delta*t+phaseval))*np.exp(1j*(4*vkvec-4)*t)*sr
    term3 = -1j*oval/2*np.exp(1j*(delta*t+phaseval))*np.exp(1j*(-4*hkvec+4)*t)*sd
    term4 = -1j*oval/2*np.exp(-1j*(delta*t+phaseval))*np.exp(1j*(4*hkvec+4)*t)*su
    
    # Complete making RHS
    rho_mat_out = term1 + term2 + term3 + term4 + loss_mat*rho_mat

    # Reshape
    rho_out = np.reshape(rho_mat_out, nstates**2)

    # Return
    return rho_out

@jit(complex128[:](float64, complex128[:], float64[:], float64, float64, float64, float64))
def bloch_rhs_gaussian(t, phi, kvec, delta, omega, sigma, t0):
    """Evaluates the right hand side of the Schrodinger equation for the Bloch Hamiltonian in the case of a Gaussian pulse with constant phase. The function returns a vector, one for each state included in the Hamiltonian.
    
    The right hand side is given by

    .. math::

        \\text{returned vector}=i\\frac{\\Omega(t)}{2}\\left[e^{i\\delta t}e^{i(-4k-4)t}\\lvert k\\rangle\\langle k + 2\\rvert + e^{-i\\delta t}e^{i(4k-4)t}\\lvert k\\rangle\\langle k-2\\rvert\\right]\\rvert\\phi\\rangle
    
    where :math:`k` indexes momentum states spaced by two photon recoils. The time :math:`t` is evaluated at :code:`t`, and the state :math:`\\lvert\\phi\\rangle` is specified by :code:`phi`.
    
    The states :math:`k` included in the calculation are specified by :code:`kvec`. :math:`\\delta` is specified by :code:`delta`. :math:`\\Omega(t, a)` is specified as follows

    .. math::

        \\Omega(t)=\\Omega\\exp\\left(-\\frac{(t-t_0)^2}{2\\sigma^2}\\right)
    
    where :math:`\\Omega` is given by :code:`omega`, :math:`\\sigma` is given by :code:`sigma`, and :math:`t_0` is given by :code:`t0`.
    
    The :code:`bloch_rhs_gaussian` function can be integrated in time using :meth:`scipy.integrate.solve_ivp`.
    
    :param t: The time at which to evaluate the right hand side.
    :param phi: The value of phi at which to evaluate the right hand side.
    :param kvec: The momentum state values at which :code:`phi` is defined.
    :param delta: The value of :math:`\\delta` (the two-photon detuning).
    :param omega: The peak effective Rabi frequency.
    :param sigma: The Gaussian width of the Rabi frequency in time.
    :param t0: The center time of the Gaussian.
    :returns: A vector containing the evaluated right hand side values."""
    
    # Compute phi_p1 and phi_m1 (pluse and minus 1)
    phi_p1 = np.zeros_like(phi)
    phi_p1[:-1] = phi[1:]
    phi_m1 = np.zeros_like(phi)
    phi_m1[1:] = phi[:-1]

    # Compute Rabi frequency at the current time
    oval = omega*np.exp(-np.square(t-t0)/(2*(sigma**2)))
    
    # Compute RHS of ODE
    return 1j*oval/2*(np.exp(1j*(-4*kvec-4+delta)*t)*phi_p1 + np.exp(1j*(4*kvec-4-delta)*t)*phi_m1)

@jit(complex128[:](float64, complex128[:], float64[:], float64, float64, float64, float64, float64))
def bloch_rhs_multifreq_gaussian(t, phi, kvec, delta, omega, sigma, t0, omega_mod):
    """Evaluates the right hand side of the Schrodinger equation for the multifrequency Bloch Hamiltonian in the case of a Gaussian pulse with constant phase. The function returns a vector, one for each state included in the Hamiltonian.
    
    The right hand side is given by

    .. math::

        \\text{returned vector}=i\\frac{\\Omega(t)}{2}\\left[e^{i\\delta t}e^{i(-4k-4)t}\\lvert k\\rangle\\langle k + 2\\rvert + e^{-i\\delta t}e^{i(4k-4)t}\\lvert k\\rangle\\langle k-2\\rvert\\right]\\rvert\\phi\\rangle
    
    where :math:`k` indexes momentum states spaced by two photon recoils. The time :math:`t` is evaluated at :code:`t`, and the state :math:`\\lvert\\phi\\rangle` is specified by :code:`phi`.
    
    The states :math:`k` included in the calculation are specified by :code:`kvec`. :math:`\\delta` is specified by :code:`delta`. :math:`\\Omega(t)` is specified as follows

    .. math::

        \\Omega(t)=2\\Omega\\cos(\\omega_\\text{mod}t)\\exp\\left(-\\frac{(t-t_0)^2}{2\\sigma^2}\\right)
    
    where :math:`\\Omega` is given by :code:`omega`, :math:`\\sigma` is given by :code:`sigma`, :math:`t_0` is given by :code:`t0`, and :math:`\\omega_\\text{mod}` is given by :code:`omega_mod`.
    
    The :code:`bloch_rhs_multifreq_gaussian` function can be integrated in time using :meth:`scipy.integrate.solve_ivp`.
    
    :param t: The time at which to evaluate the right hand side.
    :param phi: The value of phi at which to evaluate the right hand side.
    :param kvec: The momentum state values at which :code:`phi` is defined.
    :param delta: The value of :math:`\\delta` (the two-photon detuning).
    :param omega: The peak effective Rabi frequency.
    :param sigma: The Gaussian width of the Rabi frequency in time.
    :param t0: The center time of the Gaussian.
    :param omega_mod: The modulation frequency.
    :returns: A vector containing the evaluated right hand side values."""
    
    # Compute phi_p1 and phi_m1 (pluse and minus 1)
    phi_p1 = np.zeros_like(phi)
    phi_p1[:-1] = phi[1:]
    phi_m1 = np.zeros_like(phi)
    phi_m1[1:] = phi[:-1]

    # Compute Rabi frequency at the current time
    oval = 2*omega*np.cos(omega_mod*t)*np.exp(-np.square(t-t0)/(2*(sigma**2)))
    
    # Compute RHS of ODE
    return 1j*oval/2*(np.exp(1j*(-4*kvec-4+delta)*t)*phi_p1 + np.exp(1j*(4*kvec-4-delta)*t)*phi_m1)

@jit(float64(float64, float64[:]))
def omega_fnc_gaussian(t, args):
    """Function defining a Gaussian pulse profile in time, i.e.

    .. math::

        \\Omega(t)=\\Omega\\exp\\left(-\\frac{(t-t_0)^2}{2\\sigma^2}\\right)
    
    where :math:`\\Omega`, :math:`\\sigma`, and :math:`t_0` are given by :code:`args[0]`, :code:`args[1]`, and :code:`args[2]`, respectively.
    
    :param t: The time at which to evalute the Gaussian.
    :param args: A tuple of three parameters defining :math:`\\Omega`, :math:`\\sigma`, and :math:`t_0`.
    :returns: The function value at the provided time."""
    omega, sigma, t0 = args
    return omega*np.exp(-np.square(t-t0)/(2*(sigma**2)))

@jit(float64(float64, float64[:]))
def multi_omega_fnc(t, args):
    """Function defining a multifrequency Gaussian pulse profile in time, i.e.

    .. math::

        \\Omega(t)=2\\Omega\\cos(\\omega_\\text{mod}t)\\exp\\left(-\\frac{(t-t_0)^2}{2\\sigma^2}\\right)
    
    where :math:`\\Omega`, :math:`\\sigma`, :math:`t_0`, and :math:`\\omega_\\text{mod}` are given by :code:`args[0]`, :code:`args[1]`, :code:`args[2]`, and :code:`args[3]`, respectively.
    
    :param t: The time at which to evalute the Gaussian.
    :param args: A tuple of four parameters defining :math:`\\Omega`, :math:`\\sigma`, :math:`t_0`, and :code:`\\omega_\\text{mod}`.
    :returns: The function value at the provided time."""
    omega, sigma, t0, mod_freq = args
    return 2*np.cos(mod_freq*t)*omega*np.exp(-np.square(t-t0)/(2*(sigma**2)))

# Define a constant phase function
@jit(float64(float64, float64[:]))
def phase_fnc_constant(t, args):
    """Function defining a constant phase as a function of time.
    
    :param t: The time at which to evalute the phase. Since the phase is constant this parameter has no effect.
    :param args: A tuple of one parameters defining the value of the constant phase.
    :returns: The phase value at the provided time."""
    phase = args[0]
    return phase

def opt_states(optfnc, arg_guess, n0_idx, nf_idx, pi2_weight=1, pi_weight=0, nontarget_weight=0):
    """DEPRECATED
    
    Optimizes a pulse for a particular set of final states. :code:`optfunc` should return an array of amplitudes. The amplitudes at :code:`n0_idx` and :code:`nf_idx` are specified to be the original and target states. Different combinations of the final arguments optimize for different ratios of these two states.
    
    >>> from mwave.integrate import gbragg, make_kvec, make_phi, opt_states, pops_vs_time
    >>> n0, nf = 0, 5
    >>> kvec, n0_idx, nf_idx = make_kvec(n0, nf)
    >>> opt = opt_states(lambda x: gbragg(kvec, make_phi(kvec, n0), 6*x[1], 4*(n0+nf), x[0], x[1]), [30, 0.188], n0_idx, nf_idx, pi2_weight=0, pi_weight=1)
    >>> sol = gbragg(kvec, make_phi(kvec, n0), 6*opt.x[1], 4*(n0+nf), opt.x[0], opt.x[1])
    >>> pops_vs_time(kvec, sol.t, sol.y.T)
    >>> sol.y[:,-1]
    """
    
    warn('opt_states is deprecated and will be removed in a future release', DeprecationWarning, stacklevel=2)

    # Raise error if both pi2_weight and pi_weight are non-zero
    if pi2_weight != 0 and pi_weight != 0:
        raise ValueError("One of pi2_weight and pi_weight should be zero.")
    
    # Normalize weights
    norm_factor = pi2_weight + pi_weight + nontarget_weight
    pi2_weight /= norm_factor
    nontarget_weight /= norm_factor
    
    def errfnc(args):

        # Compute the states with the provided arguments
        sol = optfnc(args)

        # Compute populations in final states
        pops = np.abs(sol.y[:,-1])**2
    
        # Compute population difference in target state
        pi2_penalty = np.abs(pops[n0_idx] - pops[nf_idx])
    
        # Compute the penalty from atoms being in the original and final states
        pi_penalty = pops[n0_idx] - pops[nf_idx]
    
        # Compute population in non-target states
        pops[n0_idx] = 0
        pops[nf_idx] = 0
        nontarget_penalty = np.sum(pops)
    
        # Compute error function and return
        return pi2_penalty*pi2_weight + pi_penalty*pi_weight + nontarget_penalty*nontarget_weight

    return minimize(errfnc, arg_guess, method = 'Nelder-Mead')

def make_kvec(n0, nf, npad=10):
    """Generates a vector of :math:`k`-states. Note that neighboring :math:`k`-states are spaced by 2 photon recoils.
    
    :param n0: The initial momentum state to include in the space.
    :param nf: The final momentums tate to include in the space.
    :param npad: The padding to include on each side of the initial and final momentum states.
    :returns: A vector of momentum states spaced by :math:`2\\hbar k`
    
    Example

    >>> from mwave.integrate import make_kvec
    >>> n0, nf = 0, 5
    >>> make_kvec(n0,nf)
    (array([-20., -18., -16., -14., -12., -10.,  -8.,  -6.,  -4.,  -2.,   0.,
             2.,   4.,   6.,   8.,  10.,  12.,  14.,  16.,  18.,  20.,  22.,
            24.,  26.,  28.,  30.]), 10, 15)
    
    """
    # Compute k0 and kf from n0 and nf
    k0 = 2*n0
    kf = 2*nf
    
    # Compute k-state vector
    k_min = np.min([k0, kf]) - 2*npad
    k_max = np.max([k0, kf]) + 2*npad
    kvec = np.arange(k_min, k_max+1, 2, dtype=np.float64)
    k0_idx = np.argmin(np.abs(kvec - k0))
    kf_idx = np.argmin(np.abs(kvec - kf))
    return kvec, k0_idx, kf_idx

def make_phi(kvec, n0):
    k0_idx = np.argmin(np.abs(kvec - n0*2))
    phi0 = np.zeros(len(kvec), dtype=np.complex128)
    phi0[k0_idx] = 1
    return phi0

def make_continuous_kvec(n0, nf, dk, npad=10):
    """DEPRECATED
    
    Creates a continuous momentum grid with spacing dk. Note that the usual states in the Bragg Hamiltonian are spaced by :math:`2\\hbar k`. Here it is assumed that :math:`\\hbar k=1`, so the Bragg couples states distance :math:`2` apart. For example usage see Simulating a momentum distribution."""
    
    warn('make_continuous_kvec is deprecated and will be removed in a future release', DeprecationWarning, stacklevel=2)
    
    # Compute k0 and kf from n0 and nf
    k0 = 2*n0
    kf = 2*nf
    
    # Compute k-state vector
    k_min = np.min([k0, kf]) - 1 - 2*npad
    k_max = np.max([k0, kf]) + 1 + 2*npad

    # Define grid in momentum space
    n2hk = int(2//dk) # Compute the number of points between momentum states separated by 2*hbar*k
    kvec = np.arange(k_min, k_max+1, 2/n2hk, dtype=np.float64)

    # Return
    return kvec, n2hk

def integrate_continuous_to_discrete(kvec, psif, n2hk):
    """DEPRECATED
    
    Integrates :code:`psif` over each spacing of :math:`2\\hbar k` on the momentum grid specified by :code:`kvec`. Returns the mean value of :code:`kvec` over each :math:`2\\hbar k` and then the integral of :code:`np.abs(psif)**2` over each :math:`2\\hbar k`. For example usage see Simulating a momentum distribution."""

    warn('integrate_continuous_to_discrete is deprecated and will be removed in a future release', DeprecationWarning, stacklevel=2)
    
    kvec2 = []
    psif2 = []
    i = 0
    while i + n2hk <= len(kvec):
        kk = kvec[i:i+n2hk]
        kvec2.append(np.mean(kk))
        psif2.append(trapezoid(np.abs(psif[i:i+n2hk])**2, kk))
        i+=n2hk

    return np.array(kvec2), np.array(psif2)

def pops_vs_time(kvec, t, phi, ax=None, legend=False):
    """Make docs"""

    return_ax = False

    if ax is None:
        fig, ax = plt.subplots()
        return_ax = True

    ax.plot(t,np.abs(phi)**2, label=[r"n=%0.1f$\hbar k$" % k for k in kvec])
    ax.set_ylabel('population')
    ax.set_xlabel(r'time [$1/\omega_\mathrm{r}$]')

    if legend:
        ax.legend()

    if return_ax:
        return ax

def gbragg(kvec, phi0, tfinal, delta, omega, sigma, omega_mod=None, method='DOP853', atol=1e-10, rtol=1e-10, dense=False, max_step=0.1):
    """Performs Bragg diffraction with a Gaussian profile and a constant phase. This function internally uses :meth:`mwave.integrate.bloch_rhs_gaussian` if :code:`omega_mod=None` and :meth:`mwave.integrate.bragg_rhs_multifreq_gaussian` if :code:`omega_mod` is not :code:`None`. The Gaussian center is automatically placed at :code:`tfinal/2`.
    
    :param kvec: The vector of momentum states to simulate.
    :param phi0: The initial value of phi.
    :param tfinal: The final time to integrate to.
    :param delta: The two-photon detuning.
    :param omega: The peak effective Rabi frequency.
    :param sigma: The Gaussian width of the Rabi frequency in time.
    :param omega_mod: The modulation frequency.
    :param method: The integration method to call :code:`scipy.integrate.solve_ivp` with. Defaults to :code:`'DOP853'`.
    :param atol: The absolute tolerance given to :code:`scipy.integrate.solve_ivp`.
    :param atol: The relative tolerance given to :code:`scipy.integrate.solve_ivp`.
    :param dense: If true dense output is returned (i.e. the integration result can be queried for any intermediate time).
    :param max_step: The max step size to use during the integration.
    :returns: The solution object output from :code:`scipy.integrate.solve_ivp`.
     
    >>> from mwave.integrate import make_kvec, make_phi, gbragg, pops_vs_time
    >>> n0, nf = 0, 5
    >>> sigma = 0.188
    >>> omega= 30
    >>> kvec, n0_idx, nf_idx = make_kvec(n0,nf)
    >>> sol = gbragg(kvec, make_phi(kvec, n0), 6*sigma, 4*(n0+nf), omega, sigma)
    >>> pops_vs_time(kvec, sol.t, sol.y.T)
    >>> sol.y[:,-1]
    array([ 3.42308432e-15+5.79652561e-15j,  1.16013222e-14+1.77518885e-15j,
            1.02665571e-14-1.09796741e-14j,  3.12034740e-14-4.32155254e-14j,
            4.39906564e-14-1.12239410e-13j, -4.03937279e-13+4.94920498e-14j,
           -2.24946429e-10+6.32936084e-11j,  1.72491002e-07-9.66758213e-08j,
            3.59608734e-05-9.70371487e-05j,  1.82135333e-02+1.69593062e-02j,
           -2.89021471e-01-6.78069289e-01j, -4.69228334e-02+1.65328320e-01j,
           -3.73301726e-02-7.29203499e-02j,  2.17166733e-02+5.01757478e-02j,
            8.47644900e-03-2.22739506e-02j,  5.87939231e-01-2.64231147e-01j,
           -1.86014548e-02-2.07189313e-02j, -3.36140745e-05+1.26132182e-04j,
           -2.13202778e-07+1.69602467e-07j,  3.21925038e-10-1.32610954e-10j,
            2.75292990e-13-1.35756738e-13j,  3.65468413e-15-6.73688241e-15j,
            4.95541492e-15+3.82119313e-15j, -1.97357818e-15-3.62804364e-15j,
           -2.39926148e-15+2.33858146e-15j, -1.94677429e-15+1.34722165e-15j])
    """
    
    # Compute t0 from tfinal
    t0 = tfinal/2

    # Convert passed arguments to floats
    tfinal = np.float64(tfinal)
    delta = np.float64(delta)
    omega = np.float64(omega)
    sigma = np.float64(sigma)

    # Determine if we should do multifrequency or single frequency
    if omega_mod is None:
        # Integrate and return
        return solve_ivp(lambda *x: bloch_rhs_gaussian(x[0], x[1], kvec, delta, omega, sigma, t0), [0, tfinal], phi0, method=method, atol=atol, rtol=rtol, dense_output=dense, max_step=max_step)
    else:
        # Integrate and return
        return solve_ivp(lambda *x: bloch_rhs_multifreq_gaussian(x[0], x[1], kvec, delta, omega, sigma, t0, omega_mod), [0, tfinal], phi0, method=method, atol=atol, rtol=rtol, dense_output=dense, max_step=max_step)

def bloch(kvec, phi0, tfinal, delta, omega, omega_args, phase, phase_args, t0=0, method='DOP853', atol=1e-10, rtol=1e-10, dense=False, max_step=0.1, transformed = False, Gamma_sps = None):
    """Evolves the provided wavefunction under the Bloch Hamiltonian. This function internally uses :meth:`mwave.integrate.bloch_rhs` if :code:`Gamma_sps=None` and :meth:`mwave.integrate.bloch_density_rhs` if :code:`Gamma_sps` is not :code:`None`. 
    
    :param kvec: The vector of momentum states to simulate.
    :param phi0: The initial value of phi.
    :param tfinal: The final time to integrate to.
    :param delta: The two-photon detuning.
    :param omega: The function that returns the value of the effective Rabi frequency :math:`\\Omega(t, a)` at an arbitrary time. The function must take two arguments, :code:`t` and :code:`omega_args`. The argument :code:`t` specifies the time at which to evaluate the effective Rabi frequency and the argument :code:`omega_args` can be used to pass in additional parameters.
    :param omega_args: A tuple of arguments to pass to the function defined by :code:`omega`.
    :param phase: The function that returns the phase of two photon detuning at an arbitrary time. The function must take two arguments, :code:`t` and :code:`phase_args`. The argument :code:`t` specifies the time at which to evaluate the phase and the argument :code:`phase_args` can be used to pass in additional parameters. This function can be set to a constant value if the user does not want to simulate a frequency swept process.
    :param phase_args: A tuple of arguments to pass to the function defined by :code:`phase`.
    :param t0: The initial time to start the simulation. Defaults to :code:`0`. This is useful if you wish to chain the output of one call to :code:`bragg` with another call to :code:`bragg` that occurs at a later time.
    :param method: The integration method to call :code:`scipy.integrate.solve_ivp` with. Defaults to :code:`'DOP853'`.
    :param atol: The absolute tolerance given to :code:`scipy.integrate.solve_ivp`.
    :param atol: The relative tolerance given to :code:`scipy.integrate.solve_ivp`.
    :param dense: If true dense output is returned (i.e. the integration result can be queried for any intermediate time).
    :param max_step: The max step size to use during the integration.
    :param transformed: See the description of :code:`bloch_rhs` for details.
    :param Gamma_sps: The rate of single photon scattering. Single photon scattering is only applied if this parameter is provided. When provided the :code:`bragg` function converts the provided wavefunction into a density matrix and then evolves the density matrix under the Bragg Hamiltonian with decoherence included. As such the returned solution contains the density matrix instead of a wavefunction.
    :returns: The solution object output from :code:`scipy.integrate.solve_ivp`."""
    
    # Check that if Gamma_sps is passed that transformed is false, as these are incompatible arguments
    if transformed and Gamma_sps is not None:
        raise NotImplementedError('The bragg function does not have a definition for propagating the density matrix under the transformed Bragg Hamiltonian.')

    # Convert passed arguments to floats
    tfinal = np.float64(tfinal)
    delta = np.float64(delta)
    
    if Gamma_sps is not None:
        
        # Convert phi0 to matrix
        rho = np.outer(phi0, phi0)
        rho_vec = np.reshape(rho, len(kvec)**2)
        
        # Determine number of states
        nstates = len(kvec)
        
        # Make loss matrix
        loss_mat = (np.ones((nstates,nstates), dtype=np.complex128) - np.diag(np.ones(nstates, dtype=np.complex128)))*-Gamma_sps/2
        
        # Create kvec matrices
        hkvec = np.tile(kvec, (nstates, 1))
        vkvec = hkvec.T

        # Integrate and return
        sol = solve_ivp(lambda *x: bloch_density_rhs(x[0], x[1], nstates, hkvec, vkvec, loss_mat, delta, omega, omega_args, phase, phase_args), [t0, tfinal], rho_vec, method=method, atol=atol, rtol=rtol, dense_output=dense, max_step=max_step)
        sol.y = np.reshape(sol.y,(len(kvec), len(kvec), len(sol.t)))
        return sol

    # Integrate and return
    return solve_ivp(lambda *x: bloch_rhs(x[0], x[1], kvec, delta, omega, omega_args, phase, phase_args, transformed=transformed), [t0, tfinal], phi0, method=method, atol=atol, rtol=rtol, dense_output=dense, max_step=max_step)

def kbragg(n0, nf, tfinal, delta, omega, omega_args, phase, phase_args, npad = 10, method='DOP853', atol=1e-10, rtol=1e-10, dense=False, max_step=0.1):
    """DEPRECATED"""

    warn('kbragg is deprecated and will be removed in a future release', DeprecationWarning, stacklevel=2)
    
    # Compute k0 and kf from n0 and nf
    k0 = 2*n0
    kf = 2*nf
    
    # Compute k-state vector
    k_min = np.min([k0, kf]) - 2*npad
    k_max = np.max([k0, kf]) + 2*npad
    kvec = np.arange(k_min, k_max+1, 2)
    nk = len(kvec)
    k0_idx = np.argmin(np.abs(kvec - k0))
    kf_idx = np.argmin(np.abs(kvec - kf))
    
    # Define initial state
    phi0 = np.zeros(nk, dtype=np.complex128)
    phi0[k0_idx] = 1

    # Integrate and return
    return kvec, solve_ivp(lambda *x: bloch_rhs(x[0], x[1], kvec, delta, omega, omega_args, phase, phase_args), [0, tfinal], phi0, method=method, atol=atol, rtol=rtol, dense_output=dense, max_step=max_step)

if __name__ == "__main__":
    import doctest
    doctest.testmod()