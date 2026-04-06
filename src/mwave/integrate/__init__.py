# Imports
from numba import jit, complex128, float64
import numpy as np
from matplotlib import pyplot as plt
import warnings
from ._backends import (
    _preeval_rk4_arrays,
    _pilot_dt,
    _rk4_bloch_single_kernel,
    _rk45_bloch_adaptive,
    _run_batched,
    _run_scipy,
)


class PropagateResult:
    """Result returned by :py:func:`propagate`.

    Attributes always available:

    - ``phi_final``: Final wavefunction, ``(N,)`` for single-atom or
      ``(natoms, N)`` for batch mode.
    - ``kvec``: Momentum-state grid used in the simulation.

    Attributes set only for the RK45 backends (``'numba'``, ``'cpp'``,
    ``'gpu'``, ``'metal'``):

    - ``dt``: Step size used by the integrator.
    - ``error``: Estimated integration error.

    Attribute set only for the ``'scipy'`` backend:

    - ``scipy_sol``: The full :class:`scipy.integrate.OdeResult`.  When
      ``dense=True`` was requested the interpolant is accessible via
      ``scipy_sol.sol``.
    """

    def __init__(self, phi_final, kvec, omega, omega_args, phase, phase_args,
                 dt=None, error=None, scipy_sol=None):
        self.phi_final = phi_final
        self.kvec = kvec
        self.dt = dt
        self.error = error
        self.scipy_sol = scipy_sol
        self._omega = omega
        self._omega_args = omega_args
        self._phase = phase
        self._phase_args = phase_args

    def population(self, k):
        """Return :math:`|\\langle k | \\phi \\rangle|^2` at the momentum state
        nearest to *k*.  For batch results returns an ``(natoms,)`` array."""
        idx = int(np.argmin(np.abs(self.kvec - k)))
        if np.ndim(self.phi_final) == 1:
            return float(np.abs(self.phi_final[idx]) ** 2)
        return np.abs(self.phi_final[:, idx]) ** 2

    def populations(self):
        """Return :math:`|\\phi|^2` for every momentum state."""
        return np.abs(self.phi_final) ** 2

    def plot(self):
        """Three-panel plot of populations, Rabi frequency, and phase vs time.

        Only available when the ``'scipy'`` backend was used, because the full
        time trajectory is required.  Raises :class:`ValueError` otherwise.

        :returns: The matplotlib :class:`~matplotlib.figure.Figure`.
        """
        if self.scipy_sol is None:
            raise ValueError(
                "plot() requires a scipy solution (use backend='scipy')"
            )
        sol = self.scipy_sol
        fig, (ax1, ax2, ax3) = plt.subplots(
            nrows=3, sharex=True,
            gridspec_kw={'height_ratios': [2, 1, 1]},
        )
        ax1.plot(
            sol.t, np.abs(sol.y.T) ** 2,
            label=[r"$n=%g\,\hbar k$" % k for k in self.kvec],
        )
        tt = np.linspace(sol.t[0], sol.t[-1], max(len(sol.t), 200))
        ax2.plot(tt, [self._omega(t, self._omega_args) for t in tt])
        ax3.plot(tt, [self._phase(t, self._phase_args) for t in tt])
        ax1.legend(bbox_to_anchor=(1.05, 0.95))
        ax1.set_ylabel('population')
        ax2.set_ylabel(r'$\Omega(t)$')
        ax3.set_ylabel(r'$\theta(t)$')
        ax3.set_xlabel(r'time [$1/\omega_r$]')
        plt.tight_layout()
        return fig


class ScanResult:
    """Collect :class:`PropagateResult` objects from a parameter scan.

    This is a user-level convenience class — it is *not* returned by
    :func:`propagate` itself.

    :param results: List of :class:`PropagateResult` instances.
    :param param_name: Human-readable name of the scanned parameter (used as
        the x-axis label in plots).
    :param param_values: The parameter values corresponding to each result.
    """

    def __init__(self, results, param_name, param_values):
        self.results = results
        self.param_name = param_name
        self.param_values = np.asarray(param_values)

    def get_pops(self, k0, kf):
        """Return ``(p0s, pfs)`` population arrays at states *k0* and *kf*."""
        p0s = np.array([r.population(k0) for r in self.results])
        pfs = np.array([r.population(kf) for r in self.results])
        return p0s, pfs

    def get_inversion(self, k0, kf):
        """Return the inversion ``(p0 - pf) / (p0 + pf)``."""
        p0s, pfs = self.get_pops(k0, kf)
        return (p0s - pfs) / (p0s + pfs)

    def get_interp(self, k0, kf):
        """Return cubic interpolators ``(interp_p0, interp_pf)`` over the scan
        parameter values."""
        from scipy.interpolate import RegularGridInterpolator as RGI
        p0s, pfs = self.get_pops(k0, kf)
        return (RGI((self.param_values,), p0s, method='cubic'),
                RGI((self.param_values,), pfs, method='cubic'))

    def plot_inversion(self, k0, kf):
        """Plot the inversion vs the scanned parameter.

        :returns: The matplotlib :class:`~matplotlib.figure.Figure`.
        """
        fig, ax = plt.subplots()
        ax.plot(self.param_values, self.get_inversion(k0, kf))
        ax.set_ylabel('inversion')
        ax.set_xlabel(self.param_name)
        return fig


@jit(nopython=True)
def bloch_rhs(t, phi, kvec, delta, omega, omega_args, phase, phase_args, omega_scale=1.0, transformed=False):
    """Evaluates the right hand side of the Schrodinger equation for the Bloch Hamiltonian. The function returns a vector, one for each state included in the Hamiltonian.

    The right hand side is defined in a general way so that a time-dependent field intensity and phase can be computed.

    .. math::

        \\text{returned vector}=i\\frac{\\Omega(t, a)}{2}\\left[e^{i(\\delta t+\\theta(t,b))}e^{i(-4k-4)t}\\lvert k\\rangle\\langle k + 2\\rvert + e^{-i(\\delta t+\\theta(t,b))}e^{i(4k-4)t}\\lvert k\\rangle\\langle k-2\\rvert\\right]\\rvert\\phi\\rangle
    
    where :math:`k` indexes momentum states spaced by two photon recoils. The time :math:`t` is evaluated at :code:`t`, and the state :math:`\\lvert\\phi\\rangle` is specified by :code:`phi`.
    
    The states :math:`k` included in the calculation are specified by :code:`kvec`. :math:`\\delta` is specified by :code:`delta`. :math:`\\Omega(t, a)` is specified by :code:`omega`, which must a function which takes arguments :code:`t` and :code:`omega_args`. :math:`\\theta(t, b)` is specified by :code:`phase`, which must a function which takes arguments :code:`t` and :code:`phase_args`.

    If :code:`transformed` is :code:`True` then the right hand side is evaluated in the following frame:

    .. math::

        \\text{returned vector}=-ik^2\\lvert k\\rangle\\langle k\\rvert\\phi\\rangle + i\\frac{\\Omega(t, a)}{2}\\left[e^{i(\\delta t+\\theta(t,b))}\\lvert k\\rangle\\langle k + 2\\rvert + e^{-i(\\delta t+\\theta(t,b))}\\lvert k\\rangle\\langle k-2\\rvert\\right]\\rvert\\phi\\rangle
    
    To solve the Bloch Hamiltonian in time the :py:meth:`mwave.integrate.bloch_rhs` function can be integrated using :py:meth:`scipy.integrate.solve_ivp`.
    
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
    oval = omega(t, omega_args) * omega_scale
    phaseval = phase(t, phase_args)

    # Compute RHS of ODE
    if not transformed:
        return 1j*oval/2*(np.exp(1j*(delta*t+phaseval))*np.exp(1j*(-4*kvec-4)*t)*phi_p1 + np.exp(-1j*(delta*t+phaseval))*np.exp(1j*(4*kvec-4)*t)*phi_m1)
    
    # Compute RHS of ODE in transformed frame
    return -1j*phi*kvec**2 + 1j*oval/2*(np.exp(1j*(delta*t+phaseval))*phi_p1 + np.exp(-1j*(delta*t+phaseval))*phi_m1)

@jit(nopython=True)
def bloch_density_rhs(t, rho, nstates, hkvec, vkvec, loss_mat, delta, omega, omega_args, phase, phase_args, omega_scale=1.0):
    """Evaluates the right hand side of the Von Neumann evolution equation for the Bloch Hamiltonian (i.e. :math:`[H,\\rho]`) where

    .. math::

        H=-\\hbar\\sum_{k}\\left[\\frac{\\Omega_\\text{eff}(t,a)}{2}e^{i(\\delta t+\\theta(t,b))}e^{i\\omega_\\text{r}(-4k-4)}|k\\rangle\\langle k+2|+\\frac{\\Omega_\\text{eff}(t,a)^*}{2}e^{-i(\\delta t+\\theta(t,b))}e^{i\\omega_\\text{r}(4k-4)}|k\\rangle\\langle k-2|\\right]
         
    where :math:`\\hbar=1` and the sum over :math:`k` is limited to the values of :math:`k` defined by :code:`hkvec` and :code:`vkvec`.
         
    The parameter :code:`rho` is supplied as a vector (this makes it compatible with :code:`scipy.integrate.solve_ivp`). This is then converted to a matrix via :code:`np.reshape(rho, (len(kvec), len(kvec)))` internally. The matrices :code:`hkvec` and :code:`vkvec` are composed of horizontal or vertical vectors of the momentum state grid stacked togeather.
    
    The parameter :code:`loss_mat` is the loss matrix.
    
    The remaining parameters (:code:`delta`, :code:`omega`, :code:`omega_args`, :code:`phase`, :code:`phase_args`) are equivalent to those used in the :py:meth:`mwave.integrate.bloch_rhs` function.
    
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
    oval = omega(t, omega_args) * omega_scale
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

def make_kvec(n0, nf, npad=10):
    """Generates a vector of :math:`k`-states. Note that neighboring :math:`k`-states are spaced by 2 photon recoils.
    
    :param n0: The initial momentum state to include in the space.
    :param nf: The final momentums tate to include in the space.
    :param npad: The padding to include on each side of the initial and final momentum states.
    :returns: A tuple containing a vector of momentum states spaced by :math:`2\\hbar k`, the index of :code:`n0` in the vector, and the index of :code:`nf` in the vector.
    
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
    """Creates a vector the same length as :code:`kvec` filled with all zeros aside from in the index where :code:`kvec==2*n0`, which is set to :code:`1`.
    
    :param kvec: The momentum state values at which :code:`phi` is defined.
    :param n0: The state to initialize all amplitude in."""
    k0_idx = np.argmin(np.abs(kvec - n0*2))
    phi0 = np.zeros(len(kvec), dtype=np.complex128)
    phi0[k0_idx] = 1
    return phi0

def pops_vs_time(kvec, t, phi, ax=None, legend=False):
    """
    Plots the population of each momentum state as a function of time.

    :param kvec: The momentum state vector.
    :param t: The time vector.
    :param phi: The wavefunction array with shape (len(t), len(kvec)).
    :param ax: The matplotlib axis to plot on. If None, a new figure is created.
    :param legend: If True, a legend is added to the plot.
    """
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

# ── Unified propagator ──────────────────────────────────────────────────────

def propagate(kvec, phi0, tfinal, delta, omega, omega_args, phase, phase_args,
              omegas=None, t0=0.0, backend=None,
              # scipy options
              dense=False, method='DOP853', atol=1e-10, rtol=1e-10,
              max_step=0.1, transformed=False, Gamma_sps=None,
              # RK45 options
              tol=1e-6, max_halvings=6, pilot_rtol=1e-8, cache=None):
    """Propagate a wavefunction under the atom-light coupling Hamiltonian.

    This function unifies single-atom and batched multi-atom integration.
    The backend is selected automatically based on input dimensionality,
    or can be chosen explicitly.

    The Rabi frequency seen by atom *i* is
    ``omega(t, omega_args) * omegas[i]``.  For single-atom calls ``omegas``
    defaults to ``1.0`` so that ``omega`` alone sets the Rabi frequency.

    **Backends**

    - ``'scipy'`` — adaptive ``solve_ivp`` (default for single-atom).
      Supports ``dense``, ``transformed``, and ``Gamma_sps``.
      Single-atom only.
    - ``'numba'`` — Numba RK45 with ``prange`` (default for batch).
    - ``'cpp'`` — C++ OpenMP kernel.
    - ``'gpu'`` — CUDA kernel via CuPy.
    - ``'metal'`` — Apple Silicon GPU via metalcompute.

    :param kvec: Momentum-state grid ``(N,) float64``.
    :param phi0: Initial wavefunction.  ``(N,)`` for single-atom or
        ``(natoms, N)`` for batch mode.
    :param tfinal: Final integration time.
    :param delta: Two-photon detuning — scalar for single-atom, ``(natoms,)``
        for batch mode.
    :param omega: Callable ``omega(t, omega_args) -> float``.
    :param omega_args: Extra arguments forwarded to *omega*.
    :param phase: Callable ``phase(t, phase_args) -> float``.
    :param phase_args: Extra arguments forwarded to *phase*.
    :param omegas: Per-atom Rabi frequency scale.  Scalar or ``(natoms,)``.
        Defaults to ``1.0`` for single-atom, ``np.ones(natoms)`` for batch.
    :param t0: Integration start time (default ``0.0``).
    :param backend: ``'scipy'``, ``'numba'``, ``'cpp'``, ``'gpu'``,
        ``'metal'``, or ``None`` (auto-select).
    :param dense: Request dense (interpolatable) output from scipy
        (default ``False``).  Requires ``backend='scipy'``.
    :param method: ODE method for scipy (default ``'DOP853'``).
    :param atol: Absolute tolerance for scipy (default ``1e-10``).
    :param rtol: Relative tolerance for scipy (default ``1e-10``).
    :param max_step: Maximum step size for scipy (default ``0.1``).
    :param transformed: Use the transformed frame in scipy (default ``False``).
    :param Gamma_sps: Single-photon scattering rate for density-matrix
        evolution in scipy (default ``None``).
    :param tol: Error tolerance for the RK45 backends (default ``1e-6``).
    :param max_halvings: Maximum Richardson halvings for ``'cpp'``/``'gpu'``/
        ``'metal'`` (default ``6``).
    :param pilot_rtol: Pilot RK45 tolerance for ``'cpp'``/``'gpu'``/
        ``'metal'`` (default ``1e-8``).
    :param cache: Optional ``dict`` for memoising RK45 results.
    :returns: A :class:`PropagateResult`.
    """
    # ── Detect single vs batch ────────────────────────────────────────────
    scalar_input = (np.ndim(phi0) == 1)

    if scalar_input:
        phi0 = np.asarray(phi0, dtype=np.complex128)
        delta = np.float64(delta)
        if omegas is None:
            omegas = np.float64(1.0)
        else:
            omegas = np.float64(omegas)
    else:
        phi0 = np.asarray(phi0, dtype=np.complex128)
        natoms = phi0.shape[0]
        delta = np.asarray(delta, dtype=np.float64)
        if omegas is None:
            omegas = np.ones(natoms, dtype=np.float64)
        else:
            omegas = np.asarray(omegas, dtype=np.float64)

        # Shape validation
        if delta.ndim == 0:
            delta = np.full(natoms, delta, dtype=np.float64)
        if omegas.ndim == 0:
            omegas = np.full(natoms, omegas, dtype=np.float64)
        if delta.shape[0] != natoms:
            raise ValueError(
                f"delta has length {delta.shape[0]} but phi0 has {natoms} atoms"
            )
        if omegas.shape[0] != natoms:
            raise ValueError(
                f"omegas has length {omegas.shape[0]} but phi0 has {natoms} atoms"
            )

    # ── Auto-select backend ───────────────────────────────────────────────
    if backend is None:
        backend = 'scipy' if scalar_input else 'numba'

    # ── Validation ────────────────────────────────────────────────────────
    if not scalar_input and backend == 'scipy':
        raise ValueError("backend='scipy' is only supported for single-atom "
                         "(1-D phi0). Use 'numba', 'cpp', 'gpu', or 'metal' "
                         "for batch mode.")
    if not scalar_input and dense:
        raise ValueError("dense=True is not supported for batch mode.")
    if dense and backend != 'scipy':
        raise ValueError(
            f"dense=True requires backend='scipy', got backend='{backend}'."
        )
    if backend != 'scipy' and (transformed or Gamma_sps is not None):
        raise ValueError(
            "transformed and Gamma_sps are only supported with backend='scipy'."
        )

    # ── scipy path ────────────────────────────────────────────────────────
    if backend == 'scipy':
        phi_final, sol = _run_scipy(
            kvec, phi0, t0, tfinal, delta, omega, omega_args,
            phase, phase_args, omega_scale=float(omegas),
            bloch_rhs=bloch_rhs, bloch_density_rhs=bloch_density_rhs,
            method=method, atol=atol, rtol=rtol, dense=dense,
            max_step=max_step, transformed=transformed, Gamma_sps=Gamma_sps,
        )
        return PropagateResult(
            phi_final=phi_final, kvec=kvec,
            omega=omega, omega_args=omega_args,
            phase=phase, phase_args=phase_args,
            scipy_sol=sol,
        )

    # ── RK45 backends ─────────────────────────────────────────────────────
    # Normalise to 2-D arrays for the batched kernel
    if scalar_input:
        phi0_2d = phi0[np.newaxis, :]
        delta_arr = np.atleast_1d(np.asarray(delta, dtype=np.float64))
        omegas_arr = np.atleast_1d(np.asarray(omegas, dtype=np.float64))
    else:
        phi0_2d = phi0
        delta_arr = delta
        omegas_arr = omegas

    if cache is not None:
        cache_key = (phi0_2d.tobytes(), delta_arr.tobytes(),
                     float(t0), float(tfinal), backend)
        if cache_key in cache:
            return cache[cache_key]

    # ── Python backend: adaptive RK45 ─────────────────────────────────────
    if backend == 'numba':
        phi_all, dt_used, error_est = _rk45_bloch_adaptive(
            phi0_2d, omegas_arr, delta_arr, t0, tfinal,
            omega, omega_args, phase, phase_args, kvec, tol)
        phi_out = phi_all[0] if scalar_input else phi_all
        result = PropagateResult(
            phi_final=phi_out, kvec=kvec,
            omega=omega, omega_args=omega_args,
            phase=phase, phase_args=phase_args,
            dt=dt_used, error=error_est,
        )
        if cache is not None:
            cache[cache_key] = result
        return result

    # ── cpp / gpu / metal: pilot + fixed-step Richardson ──────────────────
    delta_worst = float(np.max(np.abs(delta_arr)))
    max_omega = float(np.max(np.abs(omegas_arr)))

    def _scaled_envelope(t, args):
        return max_omega * float(omega(t, args))

    dt_pilot = _pilot_dt(t0, tfinal, delta_worst, _scaled_envelope, omega_args,
                         phase, phase_args, pilot_rtol)
    nsteps = max(int(np.ceil((tfinal - t0) / dt_pilot)), 1)

    # Coarse integration
    h = (tfinal - t0) / nsteps
    arrays = _preeval_rk4_arrays(kvec, t0, nsteps, h,
                                 omega, omega_args, phase, phase_args)
    env_full, env_half, eph_full, eph_half, \
        kinp_full, kinm_full, kinp_half, kinm_half = arrays

    phi_coarse = _run_batched(
        backend, phi0_2d, delta_arr, omegas_arr, h, t0,
        env_full, env_half, eph_full, eph_half,
        kinp_full, kinm_full, kinp_half, kinm_half)
    error_est = np.inf

    for _ in range(max_halvings + 1):
        nsteps *= 2
        h = (tfinal - t0) / nsteps
        arrays = _preeval_rk4_arrays(kvec, t0, nsteps, h,
                                     omega, omega_args, phase, phase_args)
        env_full, env_half, eph_full, eph_half, \
            kinp_full, kinm_full, kinp_half, kinm_half = arrays
        phi_fine = _run_batched(
            backend, phi0_2d, delta_arr, omegas_arr, h, t0,
            env_full, env_half, eph_full, eph_half,
            kinp_full, kinm_full, kinp_half, kinm_half)
        error_est = float(np.max(np.abs(phi_coarse - phi_fine))) / 15.0
        if error_est <= tol:
            phi_out = phi_fine[0] if scalar_input else phi_fine
            result = PropagateResult(
                phi_final=phi_out, kvec=kvec,
                omega=omega, omega_args=omega_args,
                phase=phase, phase_args=phase_args,
                dt=(tfinal - t0) / nsteps, error=error_est,
            )
            if cache is not None:
                cache[cache_key] = result
            return result
        phi_coarse = phi_fine

    warnings.warn(
        f"propagate: tol={tol} not met after {max_halvings} halvings "
        f"(error_est={error_est:.2e}); returning best result.",
        RuntimeWarning, stacklevel=2,
    )
    phi_out = phi_fine[0] if scalar_input else phi_fine
    result = PropagateResult(
        phi_final=phi_out, kvec=kvec,
        omega=omega, omega_args=omega_args,
        phase=phase, phase_args=phase_args,
        dt=(tfinal - t0) / nsteps, error=error_est,
    )
    if cache is not None:
        cache[cache_key] = result
    return result


# ── Backend benchmark ─────────────────────────────────────────────────────────

def score_backends(n0=0, nf=5, natoms=8, tol=1e-6, repeat=3):
    """Benchmark all available :func:`propagate` backends and return a ranked
    table.

    Runs a fixed Gaussian-pulse Bragg scenario (5hk, ``natoms`` atoms) through
    each backend, measures wall-clock time (best of ``repeat`` runs), and
    verifies that the result agrees with the ``'numba'`` backend to within
    ``1e-3``.  Backends whose dependencies are missing are listed as
    ``'unavailable'``.

    :param n0: Lower momentum-state order (default ``0``).
    :param nf: Upper momentum-state order (default ``5``).
    :param natoms: Number of atoms in the benchmark batch (default ``8``).
    :param tol: Integration tolerance (default ``1e-6``).
    :param repeat: Number of timed repetitions per backend; the minimum is used
        (default ``3``).
    :returns: List of result dicts sorted by ``time_s`` (``None`` last).
    """
    import time as _time

    kvec, phi0, tfinal, delta, omega_args, phase_args = _score_setup(n0, nf)
    deltas  = np.linspace(delta * 0.98, delta * 1.02, natoms)
    omegas  = np.ones(natoms)
    phi0b   = np.tile(phi0[np.newaxis, :], (natoms, 1))

    # Reference: python backend (always available)
    ref = propagate(
        kvec, phi0b, tfinal, deltas,
        omega_fnc_gaussian, omega_args,
        phase_fnc_constant, phase_args,
        omegas=omegas, tol=tol, backend='numba',
    )
    phi_ref = ref.phi_final

    results = []
    for bk in ('numba', 'cpp', 'gpu', 'metal'):
        # Warm-up: compile/JIT on first call so timing reflects steady-state.
        try:
            propagate(
                kvec, phi0b, tfinal, deltas,
                omega_fnc_gaussian, omega_args,
                phase_fnc_constant, phase_args,
                omegas=omegas, tol=tol, backend=bk,
            )
        except Exception as exc:
            results.append({'backend': bk, 'time_s': None,
                            'max_err': None, 'status': 'unavailable',
                            'error': str(exc)})
            continue

        # Timed runs
        best_t = float('inf')
        for _ in range(repeat):
            t0_ = _time.perf_counter()
            res = propagate(
                kvec, phi0b, tfinal, deltas,
                omega_fnc_gaussian, omega_args,
                phase_fnc_constant, phase_args,
                omegas=omegas, tol=tol, backend=bk,
            )
            best_t = min(best_t, _time.perf_counter() - t0_)

        max_err = float(np.max(np.abs(res.phi_final - phi_ref)))
        status  = 'ok' if max_err < 1e-3 else f'MISMATCH (err={max_err:.2e})'
        results.append({'backend': bk, 'time_s': best_t,
                        'max_err': max_err, 'status': status, 'error': None})

    # scipy (DOP853 reference) — single-atom only, compared against phi_ref[0]
    phi_ref1 = phi_ref[0]
    best_t = float('inf')
    for _ in range(repeat):
        t0_ = _time.perf_counter()
        res_scipy = propagate(
            kvec, phi0, tfinal, deltas[0],
            omega_fnc_gaussian, omega_args,
            phase_fnc_constant, phase_args,
            backend='scipy',
        )
        best_t = min(best_t, _time.perf_counter() - t0_)
    max_err = float(np.max(np.abs(res_scipy.phi_final - phi_ref1)))
    status  = 'ok' if max_err < 1e-3 else f'MISMATCH (err={max_err:.2e})'
    results.append({'backend': 'scipy(1)', 'time_s': best_t,
                    'max_err': max_err, 'status': status, 'error': None})

    # Sort: available (by time) before unavailable
    results.sort(key=lambda r: (r['time_s'] is None, r['time_s'] or 0))

    # Print table
    print(f"propagate backend benchmark  (natoms={natoms}, tol={tol}, best of {repeat})")
    print(f"  scipy uses DOP853, single atom only")
    hdr = f"  {'backend':<10}  {'time (s)':>10}  {'max_err':>10}  status"
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for r in results:
        t_str = f"{r['time_s']:.4f}" if r['time_s'] is not None else '—'
        e_str = f"{r['max_err']:.2e}" if r['max_err'] is not None else '—'
        print(f"  {r['backend']:<10}  {t_str:>10}  {e_str:>10}  {r['status']}")

    return results


def _score_setup(n0, nf):
    """Return (kvec, phi0, tfinal, delta, omega_args, phase_args) for scoring."""
    kvec, _, _ = make_kvec(n0, nf)
    phi0       = make_phi(kvec, n0)
    sigma      = 0.188
    omega_peak = 30.0
    delta      = float(4 * (n0 + nf))
    tfinal     = 6.0 * sigma
    omega_args = np.array([omega_peak, sigma, tfinal / 2.0])
    phase_args = np.array([0.0])
    return kvec, phi0, tfinal, delta, omega_args, phase_args
