"""Scipy (``solve_ivp``) backend for :py:func:`mwave.integrate.propagate`.

Provides adaptive ODE integration via :func:`scipy.integrate.solve_ivp`.
Single-atom only.  Supports dense output, the transformed frame, and
density-matrix evolution with single-photon scattering.

This module also defines the right-hand-side functions used by
``solve_ivp``: :py:func:`_wf_rhs` for wavefunction evolution and
:py:func:`_rho_rhs` for density-matrix evolution. See the
``Integration backends`` page in the documentation for the equations
being integrated.
"""

import numpy as np
from numba import jit
from scipy.integrate import solve_ivp


@jit(nopython=True)
def _wf_rhs(t, phi, kvec, delta, omega, omega_args, phase, phase_args,
            omega_scale=1.0, transformed=False):
    """Schrodinger-equation RHS for the Bragg-Bloch Hamiltonian.

    Returns :math:`d\\phi/dt` evaluated at time *t* for the wavefunction
    *phi* defined on the momentum grid *kvec*.
    """
    # Compute phi_p1 and phi_m1 (plus and minus 1)
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
def _rho_rhs(t, rho, nstates, hkvec, vkvec, loss_mat, delta, omega, omega_args,
             phase, phase_args, omega_scale=1.0):
    """Von Neumann RHS for the Bragg-Bloch Hamiltonian with single-photon
    scattering loss.

    Returns :math:`d\\rho/dt` for the density matrix *rho* (passed as a
    flattened vector for compatibility with ``solve_ivp``). See the
    documentation for the explicit form of the equation.
    """
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

    # Compute each term in the RHS
    term1 = 1j*oval/2*np.exp(1j*(delta*t+phaseval))*np.exp(1j*(-4*vkvec-4)*t)*sl
    term2 = 1j*oval/2*np.exp(-1j*(delta*t+phaseval))*np.exp(1j*(4*vkvec-4)*t)*sr
    term3 = -1j*oval/2*np.exp(1j*(delta*t+phaseval))*np.exp(1j*(-4*hkvec+4)*t)*sd
    term4 = -1j*oval/2*np.exp(-1j*(delta*t+phaseval))*np.exp(1j*(4*hkvec+4)*t)*su

    # Complete making RHS
    rho_mat_out = term1 + term2 + term3 + term4 + loss_mat*rho_mat

    # Reshape and return
    return np.reshape(rho_mat_out, nstates**2)


def _run_scipy(kvec, phi0, t0, tfinal, delta, omega, omega_args,
               phase, phase_args, omega_scale,
               method='DOP853', atol=1e-10, rtol=1e-10, dense=False,
               max_step=0.1, transformed=False, Gamma_sps=None):
    """Integrate via scipy ``solve_ivp``.

    :param kvec: Momentum-state grid ``(N,) float64``.
    :param phi0: Initial wavefunction ``(N,) complex128``.
    :param t0: Integration start time.
    :param tfinal: Integration end time.
    :param delta: The two-photon detuning.
    :param omega: Callable ``omega(t, omega_args) -> float``.
    :param omega_args: Extra arguments for ``omega``.
    :param phase: Callable ``phase(t, phase_args) -> float``.
    :param phase_args: Extra arguments for ``phase``.
    :param omega_scale: Multiplicative scale applied to the Rabi frequency.
    :param method: ODE method (default ``'DOP853'``).
    :param atol: Absolute tolerance.
    :param rtol: Relative tolerance.
    :param dense: Request dense (interpolatable) output.
    :param max_step: Maximum step size.
    :param transformed: Use the transformed frame.
    :param Gamma_sps: Single-photon scattering rate (``None`` to disable).
    :returns: ``(phi_final, sol)`` where *phi_final* is the final state and ``sol`` is the full :class:`~scipy.integrate.OdeResult`.
    """
    tfinal_f = np.float64(tfinal)
    delta_f = np.float64(delta)

    if transformed and Gamma_sps is not None:
        raise NotImplementedError('propagate does not support density-matrix evolution in the transformed frame.')

    if Gamma_sps is not None:
        rho = np.outer(phi0, phi0)
        rho_vec = np.reshape(rho, len(kvec) ** 2)
        nstates = len(kvec)
        loss_mat = (np.ones((nstates, nstates), dtype=np.complex128)-np.diag(np.ones(nstates, dtype=np.complex128)))*-Gamma_sps/2
        hkvec = np.tile(kvec, (nstates, 1))
        vkvec = hkvec.T

        sol = solve_ivp(
            lambda *x: _rho_rhs(
                x[0], x[1], nstates, hkvec, vkvec, loss_mat, delta_f,
                omega, omega_args, phase, phase_args, omega_scale),
            [t0, tfinal_f], rho_vec,
            method=method, atol=atol, rtol=rtol,
            dense_output=dense, max_step=max_step,
        )
        sol.y = np.reshape(sol.y, (len(kvec), len(kvec), len(sol.t)))
        phi_final = sol.y[:, :, -1]
    else:
        sol = solve_ivp(
            lambda *x: _wf_rhs(
                x[0], x[1], kvec, delta_f, omega, omega_args,
                phase, phase_args, omega_scale=omega_scale,
                transformed=transformed),
            [t0, tfinal_f], phi0,
            method=method, atol=atol, rtol=rtol,
            dense_output=dense, max_step=max_step,
        )
        phi_final = sol.y[:, -1]

    return phi_final, sol
