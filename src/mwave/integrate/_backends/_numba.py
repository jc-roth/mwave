"""Numba backend for :py:func:`mwave.integrate.propagate`.

Contains:
- Dormand-Prince RK45 step kernel             (_rk45_dp_step)
- Adaptive RK45 driver                        (_rk45_bloch_adaptive)
"""

import numpy as np
import numba as nb
from numba import njit, prange
import warnings


# ── Adaptive RK45 (Dormand-Prince) ────────────────────────────────────────────
#
# Replaces the pilot + fixed-step Richardson loop when backend='numba'.
# A single pass with embedded error control avoids the pilot entirely and
# takes very few steps in regions where the coupling is small (e.g. Gaussian
# pulse tails), while automatically refining near the pulse peak.
#
# Stage structure (c = [0, 1/5, 3/10, 4/5, 8/9, 1]):
#   k1 = f(t,           phi)
#   k2 = f(t + h/5,     phi + h*a21*k1)
#   k3 = f(t + 3h/10,   phi + h*(a31*k1 + a32*k2))
#   k4 = f(t + 4h/5,    phi + h*(a41*k1 + a42*k2 + a43*k3))
#   k5 = f(t + 8h/9,    phi + h*(a51*k1 + ... + a54*k4))
#   k6 = f(t + h,       phi + h*(a61*k1 + ... + a65*k5))
#   phi5 = phi + h*(b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)  [5th order]
#   k7 = f(t + h,       phi5)                                 [for error only]
#   err  = h*(e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6 + e7*k7)
#
# om, eph, kinp, kinm are pre-evaluated at the 6 unique stage times and passed
# in from Python so that omega_envelope and phase callables are never invoked
# inside Numba (supporting both JIT and plain-Python callables).

@njit(cache=True, parallel=True)
def _rk45_dp_step(phi_in, phi_out, phi_err,
                  omegas, deltas, t, h,
                  om, eph, kinp, kinm):
    """Dormand-Prince RK45 step for all atoms in parallel.

    :param phi_in:   ``(natoms, N) complex128`` — state at start of step.
    :param phi_out:  ``(natoms, N) complex128`` — 5th-order solution (output).
    :param phi_err:  ``(natoms, N) complex128`` — error estimate ``phi5-phi4`` (output).
    :param omegas:   ``(natoms,) float64`` — per-atom Rabi scale.
    :param deltas:   ``(natoms,) float64`` — per-atom detuning.
    :param t:        Current time (float64).
    :param h:        Step size (float64).
    :param om:       ``(6,) float64`` — ``omega_envelope`` at the 6 stage times.
    :param eph:      ``(6,) complex128`` — ``exp(i*phase)`` at the 6 stage times.
    :param kinp:     ``(6, N) complex128`` — ``exp(i*(-4k-4)*t_stage)``.
    :param kinm:     ``(6, N) complex128`` — ``exp(i*(4k-4)*t_stage)``.
    """
    # ── Dormand-Prince coefficients ───────────────────────────────────────────
    a21 = 1.0 / 5.0
    a31 = 3.0 / 40.0;      a32 = 9.0 / 40.0
    a41 = 44.0 / 45.0;     a42 = -56.0 / 15.0;      a43 = 32.0 / 9.0
    a51 = 19372.0 / 6561.0; a52 = -25360.0 / 2187.0
    a53 = 64448.0 / 6561.0; a54 = -212.0 / 729.0
    a61 = 9017.0 / 3168.0;  a62 = -355.0 / 33.0
    a63 = 46732.0 / 5247.0; a64 = 49.0 / 176.0;     a65 = -5103.0 / 18656.0
    # 5th-order weights (b2 = b7 = 0)
    b1 = 35.0 / 384.0;  b3 = 500.0 / 1113.0; b4 = 125.0 / 192.0
    b5 = -2187.0 / 6784.0; b6 = 11.0 / 84.0
    # Error weights e = b_5th - b_4th (e2 = 0)
    e1 = 71.0 / 57600.0;    e3 = -71.0 / 16695.0;  e4 = 71.0 / 1920.0
    e5 = -17253.0 / 339200.0; e6 = 22.0 / 525.0;   e7 = -1.0 / 40.0
    # c-values for stage times (index → offset as fraction of h)
    _c2 = 1.0 / 5.0; _c3 = 3.0 / 10.0; _c4 = 4.0 / 5.0
    _c5 = 8.0 / 9.0; _c6 = 1.0

    natoms = phi_in.shape[0]
    N      = phi_in.shape[1]
    Nm1    = N - 1

    for i in prange(natoms):
        delta_i = deltas[i]
        omega_i = omegas[i]
        phi     = phi_in[i].copy()

        k1  = np.empty(N, nb.complex128)
        k2  = np.empty(N, nb.complex128)
        k3  = np.empty(N, nb.complex128)
        k4  = np.empty(N, nb.complex128)
        k5  = np.empty(N, nb.complex128)
        k6  = np.empty(N, nb.complex128)
        k7  = np.empty(N, nb.complex128)
        tmp = np.empty(N, nb.complex128)

        # Detuning phase at each of the 6 unique stage times
        ep0 = np.exp(1j * delta_i * t)
        ep1 = ep0
        ep2 = ep0 * np.exp(1j * delta_i * h * _c2)
        ep3 = ep0 * np.exp(1j * delta_i * h * _c3)
        ep4 = ep0 * np.exp(1j * delta_i * h * _c4)
        ep5 = ep0 * np.exp(1j * delta_i * h * _c5)
        ep6 = ep0 * np.exp(1j * delta_i * h * _c6)

        # Combined coupling scalar c_plus[s] = i*omega_i*om[s]/2 * ep[s]*eph[s]
        # c_minus = -conj(c_plus)
        c1p = 1j * omega_i * om[0] * 0.5 * (ep1 * eph[0])
        c2p = 1j * omega_i * om[1] * 0.5 * (ep2 * eph[1])
        c3p = 1j * omega_i * om[2] * 0.5 * (ep3 * eph[2])
        c4p = 1j * omega_i * om[3] * 0.5 * (ep4 * eph[3])
        c5p = 1j * omega_i * om[4] * 0.5 * (ep5 * eph[4])
        c6p = 1j * omega_i * om[5] * 0.5 * (ep6 * eph[5])

        # ── k1: RHS at (t, phi) ───────────────────────────────────────────────
        k1[0] = c1p * kinp[0, 0] * phi[1]
        for n in range(1, Nm1):
            k1[n] = c1p * kinp[0, n] * phi[n+1] - c1p.conjugate() * kinm[0, n] * phi[n-1]
        k1[Nm1] = -c1p.conjugate() * kinm[0, Nm1] * phi[Nm1-1]

        # ── k2: RHS at (t + h/5, phi + h*a21*k1) ─────────────────────────────
        for n in range(N):
            tmp[n] = phi[n] + h * a21 * k1[n]
        k2[0] = c2p * kinp[1, 0] * tmp[1]
        for n in range(1, Nm1):
            k2[n] = c2p * kinp[1, n] * tmp[n+1] - c2p.conjugate() * kinm[1, n] * tmp[n-1]
        k2[Nm1] = -c2p.conjugate() * kinm[1, Nm1] * tmp[Nm1-1]

        # ── k3: RHS at (t + 3h/10, phi + h*(a31*k1 + a32*k2)) ───────────────
        for n in range(N):
            tmp[n] = phi[n] + h * (a31 * k1[n] + a32 * k2[n])
        k3[0] = c3p * kinp[2, 0] * tmp[1]
        for n in range(1, Nm1):
            k3[n] = c3p * kinp[2, n] * tmp[n+1] - c3p.conjugate() * kinm[2, n] * tmp[n-1]
        k3[Nm1] = -c3p.conjugate() * kinm[2, Nm1] * tmp[Nm1-1]

        # ── k4: RHS at (t + 4h/5, phi + h*(a41*k1 + a42*k2 + a43*k3)) ──────
        for n in range(N):
            tmp[n] = phi[n] + h * (a41 * k1[n] + a42 * k2[n] + a43 * k3[n])
        k4[0] = c4p * kinp[3, 0] * tmp[1]
        for n in range(1, Nm1):
            k4[n] = c4p * kinp[3, n] * tmp[n+1] - c4p.conjugate() * kinm[3, n] * tmp[n-1]
        k4[Nm1] = -c4p.conjugate() * kinm[3, Nm1] * tmp[Nm1-1]

        # ── k5: RHS at (t + 8h/9, phi + h*(a51*k1 + ... + a54*k4)) ─────────
        for n in range(N):
            tmp[n] = phi[n] + h * (a51*k1[n] + a52*k2[n] + a53*k3[n] + a54*k4[n])
        k5[0] = c5p * kinp[4, 0] * tmp[1]
        for n in range(1, Nm1):
            k5[n] = c5p * kinp[4, n] * tmp[n+1] - c5p.conjugate() * kinm[4, n] * tmp[n-1]
        k5[Nm1] = -c5p.conjugate() * kinm[4, Nm1] * tmp[Nm1-1]

        # ── k6: RHS at (t + h, phi + h*(a61*k1 + ... + a65*k5)) ─────────────
        for n in range(N):
            tmp[n] = phi[n] + h * (a61*k1[n] + a62*k2[n] + a63*k3[n]
                                    + a64*k4[n] + a65*k5[n])
        k6[0] = c6p * kinp[5, 0] * tmp[1]
        for n in range(1, Nm1):
            k6[n] = c6p * kinp[5, n] * tmp[n+1] - c6p.conjugate() * kinm[5, n] * tmp[n-1]
        k6[Nm1] = -c6p.conjugate() * kinm[5, Nm1] * tmp[Nm1-1]

        # ── 5th-order solution ────────────────────────────────────────────────
        for n in range(N):
            phi_out[i, n] = phi[n] + h * (b1*k1[n] + b3*k3[n] + b4*k4[n]
                                           + b5*k5[n] + b6*k6[n])

        # ── k7: RHS at (t + h, phi5) — same coupling scalars as k6 ──────────
        k7[0] = c6p * kinp[5, 0] * phi_out[i, 1]
        for n in range(1, Nm1):
            k7[n] = (c6p * kinp[5, n] * phi_out[i, n+1]
                     - c6p.conjugate() * kinm[5, n] * phi_out[i, n-1])
        k7[Nm1] = -c6p.conjugate() * kinm[5, Nm1] * phi_out[i, Nm1-1]

        # ── Error estimate ────────────────────────────────────────────────────
        for n in range(N):
            phi_err[i, n] = h * (e1*k1[n] + e3*k3[n] + e4*k4[n]
                                  + e5*k5[n] + e6*k6[n] + e7*k7[n])


def _rk45_bloch_adaptive(phi0, omegas, deltas, t0, tfinal,
                          omega_envelope, omega_args, phase, phase_args,
                          kvec, tol, max_steps=200_000):
    """Adaptive Dormand-Prince RK45 integration of the batched Bloch equation.

    Replaces the pilot + fixed-step Richardson loop for the ``'numba'``
    backend.  Step size is controlled so that the local error satisfies
    ``max|phi_err| <= tol * h / (tfinal - t0)``, which bounds the accumulated
    global error to ``tol``.

    :param phi0:    ``(natoms, N) complex128`` — initial state.
    :param omegas:  ``(natoms,) float64`` — per-atom Rabi scale.
    :param deltas:  ``(natoms,) float64`` — per-atom detuning.
    :param t0:      Start time.
    :param tfinal:  End time.
    :param omega_envelope: Callable ``(t, omega_args) -> float``.
    :param omega_args: Extra arguments for ``omega_envelope``.
    :param phase:   Callable ``(t, phase_args) -> float``.
    :param phase_args: Extra arguments for ``phase``.
    :param kvec:    ``(N,) float64`` — momentum-state grid.
    :param tol:     Maximum allowable global error.
    :param max_steps: Hard cap on the number of step *attempts* (default
        ``200_000``).  This is a runaway-loop safety net, not a target — a
        typical Bragg-pulse run takes ~50–500 steps.  If the cap is reached
        the integrator returns the partial result with a ``RuntimeWarning``.
    :returns: ``(phi_final, h_last, err_last)`` where ``h_last`` is the last
        accepted step size and ``err_last`` is the last normalised error.
    """
    T      = tfinal - t0
    base_p = -4.0 * kvec - 4.0
    base_m =  4.0 * kvec - 4.0

    # Butcher c-values for the 6 unique stage times
    _c = np.array([0.0, 1.0/5, 3.0/10, 4.0/5, 8.0/9, 1.0])

    phi     = phi0.copy()
    phi_out = np.empty_like(phi)
    phi_err = np.empty_like(phi)

    # Initial step: start at T/20, let the controller adapt
    h       = T / 20.0
    t       = t0
    h_last  = h
    err_last = 0.0

    SAFETY     = 0.9
    MIN_FACTOR = 0.1
    MAX_FACTOR = 10.0
    H_MIN      = T * 1e-12

    for _step in range(max_steps):
        if t >= tfinal - 1e-12 * T:
            break

        h = min(h, tfinal - t)

        # ── Pre-evaluate stage data at 6 time points ──────────────────────────
        t_stages = t + _c * h

        om  = np.array([float(omega_envelope(tc, omega_args)) for tc in t_stages])
        ph  = np.array([float(phase(tc, phase_args))          for tc in t_stages])
        eph = np.exp(1j * ph).astype(np.complex128)

        kinp = np.ascontiguousarray(np.exp(1j * np.outer(t_stages, base_p)))
        kinm = np.ascontiguousarray(np.exp(1j * np.outer(t_stages, base_m)))

        # ── RK45 step for all atoms ───────────────────────────────────────────
        _rk45_dp_step(phi, phi_out, phi_err, omegas, deltas,
                      t, h, om, eph, kinp, kinm)

        # ── Error norm and acceptance ─────────────────────────────────────────
        err_abs  = float(np.max(np.abs(phi_err)))
        tol_step = tol * h / T          # local budget that guarantees global <= tol

        if err_abs <= tol_step or h <= H_MIN:
            # Accept
            t      += h
            phi[:]  = phi_out
            h_last  = h
            err_last = err_abs

        # ── Step-size update (5th-order controller) ───────────────────────────
        if err_abs > 0.0:
            factor = SAFETY * (tol_step / err_abs) ** (1.0 / 5.0)
            factor = max(MIN_FACTOR, min(MAX_FACTOR, factor))
        else:
            factor = MAX_FACTOR
        h = max(H_MIN, h * factor)

    else:
        warnings.warn(
            "propagate (numba): step limit reached before tfinal; "
            "result may not satisfy tol.",
            RuntimeWarning, stacklevel=3,
        )

    return phi, h_last, err_last
