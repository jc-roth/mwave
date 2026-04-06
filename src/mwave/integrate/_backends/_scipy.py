"""Scipy (``solve_ivp``) backend for :py:func:`mwave.integrate.propagate`.

Provides adaptive ODE integration via :func:`scipy.integrate.solve_ivp`.
Single-atom only.  Supports dense output, the transformed frame, and
density-matrix evolution with single-photon scattering.
"""

import numpy as np
from scipy.integrate import solve_ivp


def _run_scipy(kvec, phi0, t0, tfinal, delta, omega, omega_args,
               phase, phase_args, omega_scale, bloch_rhs, bloch_density_rhs,
               method='DOP853', atol=1e-10, rtol=1e-10, dense=False,
               max_step=0.1, transformed=False, Gamma_sps=None):
    """Integrate via scipy ``solve_ivp``.

    :param kvec: Momentum-state grid ``(N,) float64``.
    :param phi0: Initial wavefunction ``(N,) complex128``.
    :param t0: Start time.
    :param tfinal: Final time.
    :param delta: Two-photon detuning (scalar).
    :param omega: Callable ``omega(t, omega_args) -> float``.
    :param omega_args: Extra arguments for *omega*.
    :param phase: Callable ``phase(t, phase_args) -> float``.
    :param phase_args: Extra arguments for *phase*.
    :param omega_scale: Multiplicative scale applied to the Rabi frequency.
    :param bloch_rhs: The ``bloch_rhs`` function from the integrate module.
    :param bloch_density_rhs: The ``bloch_density_rhs`` function.
    :param method: ODE method (default ``'DOP853'``).
    :param atol: Absolute tolerance.
    :param rtol: Relative tolerance.
    :param dense: Request dense (interpolatable) output.
    :param max_step: Maximum step size.
    :param transformed: Use the transformed frame.
    :param Gamma_sps: Single-photon scattering rate (``None`` to disable).
    :returns: ``(phi_final, sol)`` where *phi_final* is the final state and
        *sol* is the full :class:`~scipy.integrate.OdeResult`.
    """
    tfinal_f = np.float64(tfinal)
    delta_f = np.float64(delta)

    if transformed and Gamma_sps is not None:
        raise NotImplementedError(
            'propagate does not support density-matrix evolution in the '
            'transformed frame.'
        )

    if Gamma_sps is not None:
        rho = np.outer(phi0, phi0)
        rho_vec = np.reshape(rho, len(kvec) ** 2)
        nstates = len(kvec)
        loss_mat = (np.ones((nstates, nstates), dtype=np.complex128)
                    - np.diag(np.ones(nstates, dtype=np.complex128))) \
                   * -Gamma_sps / 2
        hkvec = np.tile(kvec, (nstates, 1))
        vkvec = hkvec.T

        sol = solve_ivp(
            lambda *x: bloch_density_rhs(
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
            lambda *x: bloch_rhs(
                x[0], x[1], kvec, delta_f, omega, omega_args,
                phase, phase_args, omega_scale=omega_scale,
                transformed=transformed),
            [t0, tfinal_f], phi0,
            method=method, atol=atol, rtol=rtol,
            dense_output=dense, max_step=max_step,
        )
        phi_final = sol.y[:, -1]

    return phi_final, sol
