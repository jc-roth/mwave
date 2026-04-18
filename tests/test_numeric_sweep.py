"""
Parity test for ``NumericBraggInterferometer.compile_numeric_sweep``.

Builds a small 3-pulse (single, multi, single) geometry, runs it through both
the fast sweep path and an equivalent hand-wired ``compile(...)`` path that
uses ``mwave.integrate.propagate`` to compute the pulse amplitudes, and
verifies the per-port population curves match.
"""

import numpy as np

from mwave import integrate as intgr
from mwave.numeric import NumericBraggInterferometer
from mwave.utils.cloud import cloud_init


def _build_ifr(nbragg, T):
    k_extent = 2 * nbragg + 6
    ifr = NumericBraggInterferometer(kmin=-k_extent, kmax=k_extent, distance=3)
    ifr.split(nbragg)
    ifr.propagate(T)
    ifr.split([nbragg, -nbragg])
    ifr.propagate(T)
    ifr.split(nbragg)
    return ifr


def test_compile_numeric_sweep_matches_compile():
    # ── geometry + pulse params ───────────────────────────────────────────
    nbragg = 3
    T = 5.0
    sigma = 0.188
    tau = 3.0
    w0 = 1e5
    tfinal = 2 * tau * sigma
    mod_freq = 8 * nbragg - np.pi / (4 * nbragg * T)

    ifr = _build_ifr(nbragg, T)
    kvec = ifr.kvec
    nstates = len(kvec)

    # ── cloud ─────────────────────────────────────────────────────────────
    natoms = 1000
    x0, y0, z0, vz, vx, vy = cloud_init(
        natoms, 5e2, 0.02, 0.02, 0.02, seed=12345,
    )
    omega0 = 30.0
    cloud = dict(x0=x0, y0=y0, z0=z0, vx=vx, vy=vy, vz=vz)
    cphases = np.linspace(0.0, 2 * np.pi, 9)

    # ── fast path ─────────────────────────────────────────────────────────
    t_center = tau * sigma
    s2 = sigma ** 2
    w0_sq = w0 ** 2
    def envelope(t):
        return np.exp(-0.5 * (t - t_center) ** 2 / s2)
    def multi_env(t):
        return 2.0 * np.cos(mod_freq * t) * envelope(t)
    def beam_profile(x, y, z):
        return omega0 * np.exp(-2.0 * (x ** 2 + y ** 2) / w0_sq)
    sweep = ifr.compile_numeric_sweep(
        envelope=envelope, tfinal=tfinal, beam_profile=beam_profile,
        multi_envelope=multi_env, internal_tol=1e-10,
    )
    fast_pops = sweep(cloud, cphases)

    # ── reference path: precompute phi per unique (op_idx, k_init) ────────
    phi_raw = {}
    for op_i, (op_type, op_args) in enumerate(ifr.operations):
        if op_type != 'split':
            continue
        is_multi = isinstance(op_args[0], (list, tuple, np.ndarray))

        seen = set()
        for leaf in ifr.current_level:
            anc = leaf.get_ancestry()
            seen.add((int(anc[op_i].k), float(anc[op_i].t)))

        for k_init, t_pulse in seen:
            k0_idx = int(np.argmin(np.abs(kvec)))
            phi0 = np.zeros((natoms, nstates), dtype=np.complex128)
            phi0[:, k0_idx] = 1.0
            deltas = 4 * nbragg + 4 * vz - 4 * k_init
            omega_per_atom = omega0 * np.exp(
                -2 * ((x0 + vx * t_pulse) ** 2 + (y0 + vy * t_pulse) ** 2) / w0 ** 2
            )
            if is_multi:
                om_fn = intgr.multi_omega_fnc
                om_args = np.array([1.0, sigma, tfinal / 2, mod_freq])
            else:
                om_fn = intgr.omega_fnc_gaussian
                om_args = np.array([1.0, sigma, tfinal / 2])
            res = intgr.propagate(
                kvec, phi0, tfinal, deltas,
                om_fn, om_args,
                intgr.phase_fnc_constant, np.array([0.0]),
                omegas=omega_per_atom,
                backend='numba', tol=1e-10,
            )
            phi_raw[(op_i, k_init)] = res.phi_final

    # phase-op detection (first multifreq split) — matches the fast path
    phase_op_idx = next(
        i for i, (op, args) in enumerate(ifr.operations)
        if op == 'split' and isinstance(args[0], (list, tuple, np.ndarray))
    )
    deltas_all = 4 * nbragg + 4 * vz

    # ── build compile() inputs ────────────────────────────────────────────
    def _make_split_func(op_i):
        def split_func(ones, cphase, k_init, k_final, klattice, t, x):
            Deltan = int(k_init) // 2
            j_final = int(np.argmin(np.abs(kvec - k_final)))
            j_raw = j_final - Deltan
            if j_raw < 0 or j_raw >= nstates:
                return np.zeros(natoms, dtype=np.complex128)
            bs = phi_raw[(op_i, int(k_init))][:, j_raw]
            kv = float(kvec[j_raw])
            amp = bs * np.exp(-1j * deltas_all * t * kv / 2)
            if op_i == phase_op_idx:
                h = int(round(-kv / 2))
                amp = amp * np.exp(1j * h * cphase)
            return amp
        return split_func

    def propagate_func(ones, cphase, t, k):
        return np.exp(-1j * t * k * k) * np.ones(natoms, dtype=np.complex128)

    split_funcs = [
        _make_split_func(i)
        for i, (op, _) in enumerate(ifr.operations) if op == 'split'
    ]

    output_momentums = [4 * nbragg, 2 * nbragg, 0, -2 * nbragg]
    calc_pops = ifr.compile(
        split_funcs=split_funcs,
        propagate_func=propagate_func,
        output_momentums=output_momentums,
        func_pop_init=lambda *a: np.zeros(natoms, dtype=np.float64),
        func_wf_init=lambda *a: np.ones(natoms, dtype=np.complex128),
        func_wf2_init=lambda *a: np.zeros(natoms, dtype=np.complex128),
        x_tolerance=1e-11,
    )

    # ── evaluate reference populations ────────────────────────────────────
    ref_pops = []
    for m in output_momentums:
        totals = np.empty(len(cphases), dtype=np.float64)
        for ci, cp in enumerate(cphases):
            comm = [np.ones(natoms, dtype=np.complex128), float(cp)]
            totals[ci] = calc_pops(m, comm).sum()
        ref_pops.append(totals)

    # ── compare ───────────────────────────────────────────────────────────
    for p, (fast, ref) in enumerate(zip(fast_pops, ref_pops)):
        assert np.allclose(fast, ref, rtol=1e-6, atol=1e-6), (
            f"port {output_momentums[p]}: fast={fast}, ref={ref}, "
            f"max|diff|={np.max(np.abs(fast - ref)):.3e}"
        )
