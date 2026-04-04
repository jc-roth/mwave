"""Python/Numba backend for bloch_rk4.

Contains:
- Fixed-step RK4 single-atom kernel          (_rk4_bloch_single_kernel)
- Fixed-step RK4 batched kernel (prange)      (_rk4_bloch_batched_kernel)
- Warm-up helpers                             (_bloch_python_warmup, _ensure_bloch_python)
- Dormand-Prince RK45 step kernel             (_rk45_dp_step)
- Adaptive RK45 driver                        (_rk45_bloch_adaptive)
"""

import numpy as np
import numba as nb
from numba import njit, prange
import warnings


# ── Fixed-step RK4 ────────────────────────────────────────────────────────────

@njit(cache=True)
def _rk4_bloch_single_kernel(phi, omega_i, h,
                               env_full, env_half,
                               eph_full, eph_half,
                               kinp_full, kinm_full,
                               kinp_half, kinm_half,
                               delta, delta_phase_init):
    """Fixed-step RK4 kernel for the Bloch Hamiltonian, single atom.

    Adapted from ``_rk4_mega_par`` in ``rk4_flat_v3``, simplified to one atom.
    The detuning phase ``exp(i*delta*t)`` is accumulated incrementally in
    float64 for precision (matching the population-candidate pattern).  The
    arbitrary pulse phase ``phase(t)`` is supplied pre-evaluated via
    ``eph_full`` / ``eph_half`` and multiplied in at each RK4 stage.

    :param phi: ``(N,) complex128`` — wavefunction, modified **in-place**.
    :param omega_i: Per-atom Rabi-frequency scale (set to ``1.0`` when the
        full envelope is already encoded in ``env_full``/``env_half``).
    :param h: Step size.
    :param env_full: ``(nsteps+1,) float64`` — ``omega(t_j)``.
    :param env_half: ``(nsteps,) float64`` — ``omega(t_j + h/2)``.
    :param eph_full: ``(nsteps+1,) complex128`` — ``exp(i*phase(t_j))``.
    :param eph_half: ``(nsteps,) complex128`` — ``exp(i*phase(t_j + h/2))``.
    :param kinp_full: ``(nsteps+1, N) complex128`` — ``exp(i*(-4k-4)*t_j)``.
    :param kinm_full: ``(nsteps+1, N) complex128`` — ``exp(i*(4k-4)*t_j)``.
    :param kinp_half: ``(nsteps, N) complex128`` — ``exp(i*(-4k-4)*(t_j+h/2))``.
    :param kinm_half: ``(nsteps, N) complex128`` — ``exp(i*(4k-4)*(t_j+h/2))``.
    :param delta: Two-photon detuning.
    :param delta_phase_init: Initial detuning phase ``= delta * t0``.
    """
    N      = phi.shape[0]
    Nm1    = N - 1
    nsteps = env_half.shape[0]
    h2     = 0.5 * h
    h6     = h / 6.0

    ep         = np.exp(1j * delta_phase_init)
    coup_inc   = np.exp(1j * delta * h)
    coup_inc_h = np.exp(1j * delta * h * 0.5)

    k1  = np.empty(N, nb.complex128)
    k2  = np.empty(N, nb.complex128)
    k3  = np.empty(N, nb.complex128)
    k4  = np.empty(N, nb.complex128)
    tmp = np.empty(N, nb.complex128)

    for j in range(nsteps):
        ep1h = ep * coup_inc_h
        ep1  = ep * coup_inc

        # Combined phase: exp(i*(delta*t_j + phase(t_j))) etc.
        # c_plus  = i*A*exp(+i*(delta*t+phase))
        # c_minus = i*A*exp(-i*(delta*t+phase)) = -conj(c_plus)
        c0p = 1j * omega_i * env_full[j]   * 0.5 * (ep   * eph_full[j])
        c0m = -c0p.conjugate()
        c1p = 1j * omega_i * env_half[j]   * 0.5 * (ep1h * eph_half[j])
        c1m = -c1p.conjugate()
        c2p = 1j * omega_i * env_full[j+1] * 0.5 * (ep1  * eph_full[j+1])
        c2m = -c2p.conjugate()

        kp0 = kinp_full[j];   km0 = kinm_full[j]
        kph = kinp_half[j];   kmh = kinm_half[j]
        kp1 = kinp_full[j+1]; km1 = kinm_full[j+1]

        # k1: RHS at (t_j, phi)
        k1[0] = c0p * kp0[0] * phi[1]
        for n in range(1, Nm1):
            k1[n] = c0p * kp0[n] * phi[n+1] + c0m * km0[n] * phi[n-1]
        k1[Nm1] = c0m * km0[Nm1] * phi[Nm1-1]

        for n in range(N):
            tmp[n] = phi[n] + h2 * k1[n]

        # k2: RHS at (t_j + h/2, tmp)
        k2[0] = c1p * kph[0] * tmp[1]
        for n in range(1, Nm1):
            k2[n] = c1p * kph[n] * tmp[n+1] + c1m * kmh[n] * tmp[n-1]
        k2[Nm1] = c1m * kmh[Nm1] * tmp[Nm1-1]

        for n in range(N):
            tmp[n] = phi[n] + h2 * k2[n]

        # k3: same coupling as k2
        k3[0] = c1p * kph[0] * tmp[1]
        for n in range(1, Nm1):
            k3[n] = c1p * kph[n] * tmp[n+1] + c1m * kmh[n] * tmp[n-1]
        k3[Nm1] = c1m * kmh[Nm1] * tmp[Nm1-1]

        for n in range(N):
            tmp[n] = phi[n] + h * k3[n]

        # k4: RHS at (t_j + h, tmp) + phi update
        k4[0] = c2p * kp1[0] * tmp[1]
        for n in range(1, Nm1):
            k4[n] = c2p * kp1[n] * tmp[n+1] + c2m * km1[n] * tmp[n-1]
        k4[Nm1] = c2m * km1[Nm1] * tmp[Nm1-1]

        for n in range(N):
            phi[n] += h6 * (k1[n] + 2.0 * k2[n] + 2.0 * k3[n] + k4[n])

        ep = ep1


@njit(cache=True, fastmath=True, parallel=True)
def _rk4_bloch_batched_kernel(phi_all, omegas, h,
                               env_full, env_half,
                               eph_full, eph_half,
                               kinp_full, kinm_full,
                               kinp_half, kinm_half,
                               deltas, delta_phase_inits):
    """Batched fixed-step RK4 over ``natoms`` atoms using Numba ``prange``.

    Each atom is independent; the inner body mirrors
    :py:func:`_rk4_bloch_single_kernel` with ``prange`` added over atoms.
    ``eph_full``/``eph_half`` carry ``exp(i*phase(t_j))`` (shared across atoms);
    the incremental ``ep`` accumulator tracks ``exp(i*delta_i*t_j)`` only.

    :param phi_all: ``(natoms, N) complex128`` — modified in-place.
    :param omegas: ``(natoms,) float64`` — per-atom Rabi frequency scale.
    :param h: Step size.
    :param env_full: ``(nsteps+1,) float64`` — ``omega_envelope(t_j)``.
    :param env_half: ``(nsteps,)   float64`` — ``omega_envelope(t_j + h/2)``.
    :param eph_full: ``(nsteps+1,) complex128`` — ``exp(i*phase(t_j))``.
    :param eph_half: ``(nsteps,)   complex128`` — ``exp(i*phase(t_j + h/2))``.
    :param kinp_full: ``(nsteps+1, N) complex128``.
    :param kinm_full: ``(nsteps+1, N) complex128``.
    :param kinp_half: ``(nsteps,   N) complex128``.
    :param kinm_half: ``(nsteps,   N) complex128``.
    :param deltas: ``(natoms,) float64`` — per-atom detuning.
    :param delta_phase_inits: ``(natoms,) float64`` — ``delta_i * t0``.
    """
    natoms = phi_all.shape[0]
    N      = phi_all.shape[1]
    Nm1    = N - 1
    nsteps = env_half.shape[0]
    h2     = 0.5 * h
    h6     = h / 6.0

    for i in prange(natoms):
        omega_i    = omegas[i]
        ep         = np.exp(1j * delta_phase_inits[i])
        coup_inc   = np.exp(1j * deltas[i] * h)
        coup_inc_h = np.exp(1j * deltas[i] * h * 0.5)

        phi = phi_all[i].copy()
        k1  = np.empty(N, nb.complex128)
        k2  = np.empty(N, nb.complex128)
        k3  = np.empty(N, nb.complex128)
        k4  = np.empty(N, nb.complex128)
        tmp = np.empty(N, nb.complex128)

        for j in range(nsteps):
            ep1h = ep * coup_inc_h
            ep1  = ep * coup_inc

            c0p = 1j * omega_i * env_full[j]   * 0.5 * (ep   * eph_full[j])
            c0m = -c0p.conjugate()
            c1p = 1j * omega_i * env_half[j]   * 0.5 * (ep1h * eph_half[j])
            c1m = -c1p.conjugate()
            c2p = 1j * omega_i * env_full[j+1] * 0.5 * (ep1  * eph_full[j+1])
            c2m = -c2p.conjugate()

            kp0 = kinp_full[j];   km0 = kinm_full[j]
            kph = kinp_half[j];   kmh = kinm_half[j]
            kp1 = kinp_full[j+1]; km1 = kinm_full[j+1]

            k1[0] = c0p * kp0[0] * phi[1]
            for n in range(1, Nm1):
                k1[n] = c0p * kp0[n] * phi[n+1] + c0m * km0[n] * phi[n-1]
            k1[Nm1] = c0m * km0[Nm1] * phi[Nm1-1]
            for n in range(N):
                tmp[n] = phi[n] + h2 * k1[n]

            k2[0] = c1p * kph[0] * tmp[1]
            for n in range(1, Nm1):
                k2[n] = c1p * kph[n] * tmp[n+1] + c1m * kmh[n] * tmp[n-1]
            k2[Nm1] = c1m * kmh[Nm1] * tmp[Nm1-1]
            for n in range(N):
                tmp[n] = phi[n] + h2 * k2[n]

            k3[0] = c1p * kph[0] * tmp[1]
            for n in range(1, Nm1):
                k3[n] = c1p * kph[n] * tmp[n+1] + c1m * kmh[n] * tmp[n-1]
            k3[Nm1] = c1m * kmh[Nm1] * tmp[Nm1-1]
            for n in range(N):
                tmp[n] = phi[n] + h * k3[n]

            k4[0] = c2p * kp1[0] * tmp[1]
            for n in range(1, Nm1):
                k4[n] = c2p * kp1[n] * tmp[n+1] + c2m * km1[n] * tmp[n-1]
            k4[Nm1] = c2m * km1[Nm1] * tmp[Nm1-1]

            for n in range(N):
                phi[n] += h6 * (k1[n] + 2.0 * k2[n] + 2.0 * k3[n] + k4[n])

            ep = ep1

        phi_all[i] = phi


def _bloch_python_warmup():
    na, N_, ns = 2, 3, 2
    phi_w  = np.zeros((na, N_), dtype=np.complex128)
    phi_w[:, 1] = 1.0
    om_w   = np.ones(na)
    dl_w   = np.zeros(na)
    dp_w   = np.zeros(na)
    h_w    = 0.01
    ef_w   = np.ones(ns + 1)
    eh_w   = np.ones(ns)
    eph_fw = np.ones(ns + 1, dtype=np.complex128)
    eph_hw = np.ones(ns,     dtype=np.complex128)
    bp_w   = np.array([-4.0, 0.0, 4.0])
    bm_w   = np.array([ 4.0, 0.0, -4.0])
    tg_w   = np.arange(ns + 1, dtype=np.float64) * h_w
    th_w   = (np.arange(ns, dtype=np.float64) + 0.5) * h_w
    kpf_w  = np.exp(1j * np.outer(tg_w, bp_w))
    kmf_w  = np.exp(1j * np.outer(tg_w, bm_w))
    kph_w  = np.exp(1j * np.outer(th_w, bp_w))
    kmh_w  = np.exp(1j * np.outer(th_w, bm_w))
    _rk4_bloch_batched_kernel(phi_w, om_w, h_w, ef_w, eh_w, eph_fw, eph_hw,
                               kpf_w, kmf_w, kph_w, kmh_w, dl_w, dp_w)


_bloch_python_warmed = False


def _ensure_bloch_python():
    global _bloch_python_warmed
    if not _bloch_python_warmed:
        _bloch_python_warmup()
        _bloch_python_warmed = True


# ── Adaptive RK45 (Dormand-Prince) ────────────────────────────────────────────
#
# Replaces the pilot + fixed-step Richardson loop when backend='python'.
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
                          kvec, tol):
    """Adaptive Dormand-Prince RK45 integration of the batched Bloch equation.

    Replaces the pilot + fixed-step Richardson loop for the ``'python'``
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

    for _step in range(200000):
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
            "bloch_rk4 adaptive: step limit reached before tfinal; "
            "result may not satisfy tol.",
            RuntimeWarning, stacklevel=3,
        )

    return phi, h_last, err_last
