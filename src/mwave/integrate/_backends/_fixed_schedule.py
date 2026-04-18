"""Fixed-schedule DP5 backend.

Complements the adaptive ``_numba`` backend: callers first build a step
schedule on a worst-case pilot trajectory (:func:`pilot_schedule`), pre-bake
the per-stage :math:`\\Omega(t)` and :math:`k`-coupling arrays
(:func:`precompute_step_arrays`), and then run :func:`fused_dp5` once per
batch with every atom sharing the same schedule.

Trading adaptivity for a fixed schedule lets the inner loop skip callable
dispatch and reuse the per-stage arrays across a whole batch of atoms — the
optimisation that made population candidate ``t0250003`` land.
"""

import numpy as np
import numba as nb
from numba import njit, prange

from ._numba import _rk45_dp_step


BUTCHER_C = np.array([0.0, 1.0/5, 3.0/10, 4.0/5, 8.0/9, 1.0])
_EPH_ONES = np.ones(6, dtype=np.complex128)


@njit(cache=True, parallel=True, fastmath=True)
def fused_dp5(phi_in, phi_out_final, omegas, deltas,
              step_hs, om_all, kinp_all, kinm_all):
    """Run a fused Dormand-Prince RK45 propagation on a precomputed schedule.

    All atoms in the batch share ``step_hs`` / ``om_all`` / ``kinp_all`` /
    ``kinm_all`` (produced by :func:`precompute_step_arrays`); only the
    per-atom ``omegas`` and ``deltas`` vary. The six DP5 stages are inlined
    so there is no per-step callable dispatch.

    :param phi_in: ``(natoms, N) complex128`` initial wavefunctions.
    :param phi_out_final: ``(natoms, N) complex128`` output buffer; filled on return.
    :param omegas: ``(natoms,) float64`` per-atom peak Rabi scale.
    :param deltas: ``(natoms,) float64`` per-atom two-photon detuning.
    :param step_hs: ``(nsteps,) float64`` step sizes from :func:`pilot_schedule`.
    :param om_all: ``(nsteps, 6) float64`` Ω(t) evaluated at each Butcher stage.
    :param kinp_all: ``(nsteps, 6, N) complex128`` upper-coupling phase factors.
    :param kinm_all: ``(nsteps, 6, N) complex128`` lower-coupling phase factors.
    """
    a21 = 1.0 / 5.0
    a31 = 3.0 / 40.0;      a32 = 9.0 / 40.0
    a41 = 44.0 / 45.0;     a42 = -56.0 / 15.0;      a43 = 32.0 / 9.0
    a51 = 19372.0 / 6561.0; a52 = -25360.0 / 2187.0
    a53 = 64448.0 / 6561.0; a54 = -212.0 / 729.0
    a61 = 9017.0 / 3168.0;  a62 = -355.0 / 33.0
    a63 = 46732.0 / 5247.0; a64 = 49.0 / 176.0;     a65 = -5103.0 / 18656.0
    b1 = 35.0 / 384.0;  b3 = 500.0 / 1113.0; b4 = 125.0 / 192.0
    b5 = -2187.0 / 6784.0; b6 = 11.0 / 84.0

    natoms = phi_in.shape[0]
    N      = phi_in.shape[1]
    Nm1    = N - 1
    nsteps = len(step_hs)

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
        tmp = np.empty(N, nb.complex128)
        t = 0.0

        for si in range(nsteps):
            h = step_hs[si]
            om = om_all[si]
            kinp = kinp_all[si]
            kinm = kinm_all[si]

            ep0 = np.exp(1j * delta_i * t)
            ep2 = ep0 * np.exp(1j * delta_i * h * (1.0/5.0))
            ep3 = ep0 * np.exp(1j * delta_i * h * (3.0/10.0))
            ep4 = ep0 * np.exp(1j * delta_i * h * (4.0/5.0))
            ep5 = ep0 * np.exp(1j * delta_i * h * (8.0/9.0))
            ep6 = ep0 * np.exp(1j * delta_i * h * 1.0)

            c1p = 1j * omega_i * om[0] * 0.5 * ep0
            c2p = 1j * omega_i * om[1] * 0.5 * ep2
            c3p = 1j * omega_i * om[2] * 0.5 * ep3
            c4p = 1j * omega_i * om[3] * 0.5 * ep4
            c5p = 1j * omega_i * om[4] * 0.5 * ep5
            c6p = 1j * omega_i * om[5] * 0.5 * ep6

            k1[0] = c1p * kinp[0, 0] * phi[1]
            for n in range(1, Nm1):
                k1[n] = c1p * kinp[0, n] * phi[n+1] - c1p.conjugate() * kinm[0, n] * phi[n-1]
            k1[Nm1] = -c1p.conjugate() * kinm[0, Nm1] * phi[Nm1-1]

            for n in range(N):
                tmp[n] = phi[n] + h * a21 * k1[n]
            k2[0] = c2p * kinp[1, 0] * tmp[1]
            for n in range(1, Nm1):
                k2[n] = c2p * kinp[1, n] * tmp[n+1] - c2p.conjugate() * kinm[1, n] * tmp[n-1]
            k2[Nm1] = -c2p.conjugate() * kinm[1, Nm1] * tmp[Nm1-1]

            for n in range(N):
                tmp[n] = phi[n] + h * (a31 * k1[n] + a32 * k2[n])
            k3[0] = c3p * kinp[2, 0] * tmp[1]
            for n in range(1, Nm1):
                k3[n] = c3p * kinp[2, n] * tmp[n+1] - c3p.conjugate() * kinm[2, n] * tmp[n-1]
            k3[Nm1] = -c3p.conjugate() * kinm[2, Nm1] * tmp[Nm1-1]

            for n in range(N):
                tmp[n] = phi[n] + h * (a41 * k1[n] + a42 * k2[n] + a43 * k3[n])
            k4[0] = c4p * kinp[3, 0] * tmp[1]
            for n in range(1, Nm1):
                k4[n] = c4p * kinp[3, n] * tmp[n+1] - c4p.conjugate() * kinm[3, n] * tmp[n-1]
            k4[Nm1] = -c4p.conjugate() * kinm[3, Nm1] * tmp[Nm1-1]

            for n in range(N):
                tmp[n] = phi[n] + h * (a51*k1[n] + a52*k2[n] + a53*k3[n] + a54*k4[n])
            k5[0] = c5p * kinp[4, 0] * tmp[1]
            for n in range(1, Nm1):
                k5[n] = c5p * kinp[4, n] * tmp[n+1] - c5p.conjugate() * kinm[4, n] * tmp[n-1]
            k5[Nm1] = -c5p.conjugate() * kinm[4, Nm1] * tmp[Nm1-1]

            for n in range(N):
                tmp[n] = phi[n] + h * (a61*k1[n] + a62*k2[n] + a63*k3[n]
                                        + a64*k4[n] + a65*k5[n])
            k6[0] = c6p * kinp[5, 0] * tmp[1]
            for n in range(1, Nm1):
                k6[n] = c6p * kinp[5, n] * tmp[n+1] - c6p.conjugate() * kinm[5, n] * tmp[n-1]
            k6[Nm1] = -c6p.conjugate() * kinm[5, Nm1] * tmp[Nm1-1]

            for n in range(N):
                phi[n] = phi[n] + h * (b1*k1[n] + b3*k3[n] + b4*k4[n]
                                        + b5*k5[n] + b6*k6[n])
            t += h

        for n in range(N):
            phi_out_final[i, n] = phi[n]


def pilot_schedule(kvec_use, phi0_pilot, omegas_pilot, deltas_pilot,
                   t0, tfinal, om_vec_fnc, tol):
    """Build an adaptive step schedule on a worst-case pilot trajectory.

    Runs a standard adaptive RK45 loop (via the single-step
    :func:`_rk45_dp_step` primitive) on a small set of pilot atoms and
    returns the accepted step sizes. The schedule is later replayed
    verbatim by :func:`fused_dp5` for the full batch.

    :param kvec_use: ``(N,) float64`` momentum grid.
    :param phi0_pilot: ``(n_pilot, N) complex128`` pilot initial wavefunctions.
    :param omegas_pilot: ``(n_pilot,) float64`` pilot Rabi scales.
    :param deltas_pilot: ``(n_pilot,) float64`` pilot detunings.
    :param t0: Start time.
    :param tfinal: End time.
    :param om_vec_fnc: Callable ``t_stages -> Ω(t_stages)`` that vectorizes
        over the 6 Butcher stage times of a single step.
    :param tol: Error tolerance driving step acceptance.
    :returns: ``(nsteps,) float64`` array of accepted step sizes.
    """
    T = tfinal - t0
    base_p = -4.0 * kvec_use - 4.0
    base_m =  4.0 * kvec_use - 4.0
    phi = phi0_pilot.copy()
    phi_out = np.empty_like(phi)
    natoms = phi.shape[0]
    err_sq = np.empty(natoms, dtype=np.float64)
    h, t = T / 20.0, t0
    H_MIN = T * 1e-12
    SAFETY, MIN_FACTOR, MAX_FACTOR = 0.9, 0.1, 10.0
    step_sizes = []
    for _step in range(200_000):
        if t >= tfinal - 1e-12 * T:
            break
        h = min(h, tfinal - t)
        t_stages = t + BUTCHER_C * h
        om = om_vec_fnc(t_stages)
        kinp = np.exp(1j * np.outer(t_stages, base_p))
        kinm = np.exp(1j * np.outer(t_stages, base_m))
        _rk45_dp_step(phi, phi_out, err_sq, omegas_pilot, deltas_pilot,
                      t, h, om, _EPH_ONES, kinp, kinm)
        err_abs = float(np.sqrt(np.max(err_sq)))
        tol_step = tol * h / T
        if err_abs <= tol_step or h <= H_MIN:
            step_sizes.append(h)
            t += h
            phi, phi_out = phi_out, phi
        if err_abs > 0.0:
            factor = SAFETY * (tol_step / err_abs) ** (1.0 / 5.0)
            factor = max(MIN_FACTOR, min(MAX_FACTOR, factor))
        else:
            factor = MAX_FACTOR
        h = max(H_MIN, h * factor)
    return np.array(step_sizes, dtype=np.float64)


def precompute_step_arrays(kvec_use, step_schedule, om_vec_fnc):
    """Pre-bake per-stage Ω and coupling-phase arrays for :func:`fused_dp5`.

    :param kvec_use: ``(N,) float64`` momentum grid.
    :param step_schedule: ``(nsteps,) float64`` step sizes.
    :param om_vec_fnc: Callable ``t -> Ω(t)`` that vectorises over times.
    :returns: ``(om_all, kinp_all, kinm_all)`` — shapes ``(nsteps, 6)`` and
        ``(nsteps, 6, N)`` — ready to pass to :func:`fused_dp5`.
    """
    base_p = -4.0 * kvec_use - 4.0
    base_m = 4.0 * kvec_use - 4.0
    nsteps = len(step_schedule)
    N = len(kvec_use)
    t_cum = np.concatenate(([0.0], np.cumsum(step_schedule[:-1])))
    all_t_stages = t_cum[:, None] + BUTCHER_C[None, :] * step_schedule[:, None]
    t_flat = all_t_stages.ravel()
    om_all = om_vec_fnc(t_flat).reshape(nsteps, 6)
    kinp_all = np.ascontiguousarray(np.exp(1j * np.outer(t_flat, base_p)).reshape(nsteps, 6, N))
    kinm_all = np.ascontiguousarray(np.exp(1j * np.outer(t_flat, base_m)).reshape(nsteps, 6, N))
    return om_all, kinp_all, kinm_all
