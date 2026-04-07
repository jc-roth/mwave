# Imports
from numba import jit, float64
import numpy as np
from matplotlib import pyplot as plt
from ._backends import (
    _rk45_bloch_adaptive,
    _run_scipy,
)

class PropagateResult:
    """Result returned by :py:func:`propagate`.

    Attributes always available:

    - ``phi_final``: Final wavefunction, ``(N,)`` for single-atom or
      ``(natoms, N)`` for batch mode.
    - ``kvec``: Momentum-state grid used in the simulation.

    Attributes set only for the ``'numba'`` backend:

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
            return np.abs(self.phi_final[idx]) ** 2
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

def propagate(kvec, phi0, tfinal, delta, omega, omega_args, phase, phase_args, omegas=None, t0=0.0, backend=None,
              # scipy options
              dense=False, method='DOP853', atol=1e-10, rtol=1e-10, max_step=0.1, transformed=False, Gamma_sps=None,
              # RK45 options
              tol=1e-10, cache=None):
    """Evolves the provided wavefunction using the equations of motion described in :doc:`/backends`. The user can provide a single atom wavefunction (in which case the ``scipy`` backend is used), or a batch of wavefunctions to be integrated in parallel (in which case the ``numba`` backend is used). The wavefunction batching provided by the function is significantly more efficient than looping over a single atom wavefunction call multiple times.

    **Backends**

    - ``'scipy'`` — adaptive ``solve_ivp`` (default for single-atom). Supports ``dense`` output, the ``transformed`` frame, and spontaneous emission via ``Gamma_sps``. Unavailable for batch mode.
    - ``'numba'`` — Numba RK45 with ``prange`` (default for batch).

    :param kvec: The vector of momentum states to simulate ``(N,) float64``.
    :param phi0: The initial value of phi. ``(N,)`` for single-atom or ``(natoms, N)`` for batch mode.
    :param tfinal: The final time to integrate to.
    :param delta: The two-photon detuning. Scalar for single-atom, ``(natoms,)`` for batch mode.
    :param omega: Callable ``omega(t, omega_args) -> float``.
    :param omega_args: Extra arguments forwarded to ``omega``.
    :param phase: Callable ``phase(t, phase_args) -> float``.
    :param phase_args: Extra arguments forwarded to ``phase``.
    :param omegas: Per-atom Rabi frequency scale.  Scalar or ``(natoms,)``. Defaults to ``1.0`` for single-atom, ``np.ones(natoms)`` for batch.
    :param t0: Integration start time (default ``0.0``).
    :param backend: ``'scipy'``, ``'numba'``, or ``None``. If ``None``, ``scipy`` will be selected for a single wavefunction and ``numba`` will be selected for a batch of wavefunctions.
    :param dense: If true dense output is returned (i.e. the integration result can be queried for any intermediate time). Requires ``backend='scipy'``.
    :param method: ODE method for ``backend='scipy'`` (default ``'DOP853'``). Ignored for other backends.
    :param atol: Absolute tolerance for ``backend='scipy'`` (default ``1e-10``). Ignored for other backends.
    :param rtol: Relative tolerance for ``backend='scipy'`` (default ``1e-10``). Ignored for other backends.
    :param max_step: Maximum step size for ``backend='scipy'`` (default ``0.1``). Ignored for other backends.
    :param transformed: Use the transformed frame for ``backend='scipy'`` (default ``False``). Ignored for other backends.
    :param Gamma_sps: Single-photon scattering rate for density-matrix evolution for ``backend='scipy'`` (default ``None``). Ignored for other backends.
    :param tol: Error tolerance for the ``numba`` RK45 backend (default ``1e-10``).
    :param cache: Optional ``dict`` for memoising results from the ``numba`` backend. The cache key incorporates every input that affects the output (``phi0``, ``delta``, ``omegas``, ``kvec``, ``omega_args``, ``phase_args``, ``t0``, ``tfinal``, ``backend``, ``tol``) along with the ``omega`` and ``phase`` callables themselves. The callables are hashed by object identity, so to get cache hits across calls you must reuse the *same* function object. Wrapping the same underlying function in a fresh ``lambda`` on each call will produce a distinct object and miss the cache. Define the wrapper once and reuse it.
    :returns: A :class:`PropagateResult`."""

    # Validate shape of inputs
    scalar_input = np.ndim(phi0) == 1

    if scalar_input:
        phi0 = np.asarray(phi0, dtype=np.complex128)
        if np.ndim(delta) != 0:
            raise ValueError(f"delta must be a scalar for single-atom mode, got shape {np.shape(delta)}")
        delta = np.float64(delta)
        if omegas is None:
            omegas = np.float64(1.0)
        else:
            if np.ndim(omegas) != 0:
                raise ValueError(f"omegas must be a scalar for single-atom mode, got shape {np.shape(omegas)}")
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

    # Select backend
    if backend is None:
        backend = 'scipy' if scalar_input else 'numba'

    # Validate inputs match the selected backend
    if backend not in ('scipy', 'numba'):
        raise ValueError(f"unknown backend {backend!r}; choose 'scipy' or 'numba'.")
    if not scalar_input and backend == 'scipy':
        raise ValueError("backend='scipy' is only supported for single-atom (1D phi0). Use 'numba' for batch mode.")
    if not scalar_input and dense:
        raise ValueError("dense=True is not supported for batch mode.")
    if dense and backend != 'scipy':
        raise ValueError(f"dense=True requires backend='scipy', got backend='{backend}'.")
    if backend != 'scipy' and (transformed or Gamma_sps is not None):
        raise ValueError("transformed and Gamma_sps are only supported with backend='scipy'.")

    # Call scipy
    if backend == 'scipy':
        phi_final, sol = _run_scipy(
            kvec, phi0, t0, tfinal, delta, omega, omega_args,
            phase, phase_args, omega_scale=float(omegas),
            method=method, atol=atol, rtol=rtol, dense=dense,
            max_step=max_step, transformed=transformed, Gamma_sps=Gamma_sps,
        )
        return PropagateResult(
            phi_final=phi_final, kvec=kvec,
            omega=omega, omega_args=omega_args,
            phase=phase, phase_args=phase_args,
            scipy_sol=sol,
        )

    # Make inputs 2D arrays, as at this point we are not using the scipy backend
    if scalar_input:
        phi0_2d = phi0[np.newaxis, :]
        delta_arr = np.atleast_1d(np.asarray(delta, dtype=np.float64))
        omegas_arr = np.atleast_1d(np.asarray(omegas, dtype=np.float64))
    else:
        phi0_2d = phi0
        delta_arr = delta
        omegas_arr = omegas

    if cache is not None:
        cache_key = (
            phi0_2d.tobytes(), delta_arr.tobytes(), omegas_arr.tobytes(),
            np.asarray(kvec).tobytes(),
            omega, np.asarray(omega_args).tobytes(),
            phase, np.asarray(phase_args).tobytes(),
            float(t0), float(tfinal), backend, float(tol),
        )
        if cache_key in cache:
            return cache[cache_key]

    # ── numba backend: adaptive RK45 ──────────────────────────────────────
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

def score_backends(n0=0, nf=5, natoms=10000, tol=1e-10, repeat=3):
    """Benchmark :func:`propagate` in single-atom and batch modes.

    Two timed runs are performed on a fixed Gaussian-pulse Bragg scenario:

    1. **Single-atom run** — both ``scipy`` and ``numba`` are timed, and the
       numba result is compared against the scipy reference.
    2. **Batch run** — only ``numba`` is timed; ``scipy`` does not support
       batch input so there is no reference to compare against.

    :param n0: Lower momentum-state order (default ``0``).
    :param nf: Upper momentum-state order (default ``5``).
    :param natoms: Number of atoms in the batch run (default ``1000``).
    :param tol: Integration tolerance (default ``1e-10``).
    :param repeat: Number of timed repetitions per backend; the minimum is
        used (default ``3``).
    :returns: A dict with timings (and the single-atom error) for each run.
    """
    import time as _time

    kvec, phi0, tfinal, delta, omega_args, phase_args = _score_setup(n0, nf)

    def best_of(backend, phi0_in, delta_in, omegas_in):
        # Warm-up so JIT / scipy startup costs don't pollute the timing.
        propagate(kvec, phi0_in, tfinal, delta_in,
                  omega_fnc_gaussian, omega_args,
                  phase_fnc_constant, phase_args,
                  omegas=omegas_in, tol=tol, backend=backend)
        best = float('inf')
        last = None
        for _ in range(repeat):
            t0_ = _time.perf_counter()
            last = propagate(kvec, phi0_in, tfinal, delta_in,
                             omega_fnc_gaussian, omega_args,
                             phase_fnc_constant, phase_args,
                             omegas=omegas_in, tol=tol, backend=backend)
            best = min(best, _time.perf_counter() - t0_)
        return last, best

    # ── Single atom: time both, compare numba to scipy ────────────────────
    print(f"\nsingle atom (n0={n0}, nf={nf}, tol={tol}, best of {repeat})")
    res_scipy, t_scipy = best_of('scipy', phi0, delta, None)
    res_numba, t_numba = best_of('numba', phi0, delta, None)
    err = float(np.max(np.abs(res_numba.phi_final - res_scipy.phi_final)))
    print(f"  scipy   {t_scipy:8.4f} s")
    print(f"  numba   {t_numba:8.4f} s    err vs scipy: {err:.2e}")

    # ── Batch: time numba only (scipy can't do batch, no reference) ───────
    deltas = np.linspace(delta * 0.98, delta * 1.02, natoms)
    omegas = np.ones(natoms)
    phi0b  = np.tile(phi0[np.newaxis, :], (natoms, 1))

    _, t_batch = best_of('numba', phi0b, deltas, omegas)
    print(f"\nbatch (natoms={natoms}, tol={tol}, best of {repeat})")
    print(f"  numba   {t_batch:8.4f} s")

    return {
        'single_atom': {'scipy': t_scipy, 'numba': t_numba, 'err': err},
        'batch': {'numba': t_batch},
    }


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
