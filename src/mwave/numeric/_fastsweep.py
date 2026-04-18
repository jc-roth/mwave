"""Fast batched phase-sweep evaluator for :class:`NumericBraggInterferometer`.

Compiles a single batched DP5 propagation over every unique ``(op_idx, k_init)``
pulse configuration in the interferometer tree, walks the ancestry once to
build a Fourier decomposition of port populations vs. sweep phase, and then
evaluates that decomposition at arbitrary ``cphases`` essentially for free.
"""

import numpy as np

from ..integrate._backends import (
    fused_dp5 as _monolithic_dp5,
    pilot_schedule as _pilot_run,
    precompute_step_arrays as _precompute_step_arrays,
)


# ── Pulse combo extraction ───────────────────────────────────────────────

def _get_pulse_combos(ifr):
    seen, combos = set(), []
    for leaf in ifr.current_level:
        nodes = leaf.get_ancestry()
        for i, (op_type, _) in enumerate(ifr.operations):
            if op_type == 'split':
                n = nodes[i]
                if (i, n.k) not in seen:
                    seen.add((i, n.k))
                    combos.append((i, n.k, n.t, n.x))
    return sorted(combos)


def _is_multifreq(op_idx, operations):
    klattice = operations[op_idx][1][0]
    return isinstance(klattice, (list, tuple, np.ndarray))


# ── Sweep object ─────────────────────────────────────────────────────────

class _NumericSweep:
    """Callable produced by :meth:`NumericBraggInterferometer.compile_numeric_sweep`.

    Invoking the instance runs the batched DP5 propagation over the cloud,
    builds the Fourier decomposition across the configured output momenta,
    and evaluates it at the requested ``cphases`` grid.
    """

    def __init__(self, ifr, envelope, tfinal, beam_profile, multi_envelope,
                 output_momentums, tol, internal_tol):
        self._ifr = ifr
        self._envelope = envelope
        self._multi_envelope = multi_envelope
        self._tfinal = float(tfinal)
        self._beam_profile = beam_profile
        self._tol = float(tol)
        self._internal_tol = float(internal_tol)

        nbragg = ifr.operations[0][1][0]
        self._nbragg = nbragg

        has_multi = any(
            op == 'split' and isinstance(args[0], (list, tuple, np.ndarray))
            for op, args in ifr.operations
        )
        if has_multi and multi_envelope is None:
            raise ValueError(
                "interferometer geometry contains multi-frequency splits but "
                "no `multi_envelope` was provided."
            )

        if output_momentums is None:
            output_momentums = (4 * nbragg, 2 * nbragg, 0, -2 * nbragg)
        self._output_momentums = tuple(output_momentums)

        self._sched_cache = {}
        self._array_cache = {}
        self._coeff_cache = {}

    # ── internal helpers ──────────────────────────────────────────────

    def _compute_schedule(self, kvec_use, combo_keys, cloud, om_vec_fnc, omega_worst):
        nstates = len(kvec_use)
        k0_idx = int(np.argmin(np.abs(kvec_use)))
        n = len(combo_keys)
        vz_arr = cloud['vz']
        vz_pos = float(np.max(vz_arr))
        vz_neg = float(np.min(vz_arr))

        phi0 = np.zeros((n, nstates), dtype=np.complex128)
        omegas = np.full(n, omega_worst)
        deltas = np.empty(n)
        for row, (op_idx, k_init) in enumerate(combo_keys):
            phi0[row, k0_idx] = 1.0
            d_pos = 4 * self._nbragg + 4 * vz_pos - 4 * k_init
            d_neg = 4 * self._nbragg + 4 * vz_neg - 4 * k_init
            deltas[row] = d_pos if abs(d_pos) >= abs(d_neg) else d_neg
        return _pilot_run(kvec_use, phi0, omegas, deltas, 0.0, self._tfinal,
                          om_vec_fnc, self._internal_tol)

    def _propagate_group(self, cloud, group, om_vec_fnc):
        ifr = self._ifr
        x0, y0, z0 = cloud['x0'], cloud['y0'], cloud['z0']
        vx, vy, vz = cloud['vx'], cloud['vy'], cloud['vz']
        kvec = ifr.kvec
        natoms = len(vz)
        nstates = len(kvec)
        k0_idx = int(np.argmin(np.abs(kvec)))
        n = len(group)

        phi0_b = np.zeros((n * natoms, nstates), dtype=np.complex128)
        omegas_b = np.empty(n * natoms)
        deltas_b = np.empty(n * natoms)
        for row, (op_idx, k_init, t_pulse, _) in enumerate(group):
            sl = slice(row * natoms, (row + 1) * natoms)
            phi0_b[sl, k0_idx] = 1.0
            omegas_b[sl] = self._beam_profile(
                x0 + vx * t_pulse, y0 + vy * t_pulse, z0 + vz * t_pulse,
            )
            deltas_b[sl] = (4 * self._nbragg + 4 * vz) - 4 * k_init

        omega_worst = float(np.max(np.abs(omegas_b)))

        combo_keys = tuple((op_idx, int(k_init)) for op_idx, k_init, _, _ in group)
        is_multi = _is_multifreq(group[0][0], ifr.operations)
        om_type = 'multi' if is_multi else 'single'
        sched_key = (om_type, combo_keys, nstates)
        sched = self._sched_cache.get(sched_key)
        if sched is None:
            sched = self._compute_schedule(kvec, combo_keys, cloud, om_vec_fnc, omega_worst)
            self._sched_cache[sched_key] = sched

        arr_key = (om_type, id(sched), nstates)
        arrs = self._array_cache.get(arr_key)
        if arrs is None:
            arrs = _precompute_step_arrays(kvec, sched, om_vec_fnc)
            self._array_cache[arr_key] = arrs
        om_all, kinp_all, kinm_all = arrs

        phi_out = np.empty_like(phi0_b)
        _monolithic_dp5(phi0_b, phi_out, omegas_b, deltas_b,
                        sched, om_all, kinp_all, kinm_all)

        result = {}
        for row, (op_idx, k_init, _, _) in enumerate(group):
            sl = slice(row * natoms, (row + 1) * natoms)
            result[op_idx, k_init] = phi_out[sl]
        return result

    def _precompute_pulses(self, cloud):
        ifr = self._ifr
        combos = _get_pulse_combos(ifr)
        operations = ifr.operations
        phi_raw = {}

        single = [c for c in combos if not _is_multifreq(c[0], operations)]
        multi  = [c for c in combos if     _is_multifreq(c[0], operations)]

        if single:
            phi_raw.update(self._propagate_group(cloud, single, self._envelope))
        if multi:
            phi_raw.update(self._propagate_group(cloud, multi, self._multi_envelope))
        return phi_raw

    def _compute_fourier_dense(self, phi_raw, cloud, injected_dphase):
        ifr = self._ifr
        kvec = ifr.kvec
        nstates = len(kvec)
        operations = ifr.operations
        natoms = len(cloud['vz'])
        vz = cloud['vz']
        deltas = 4 * self._nbragg + 4 * vz
        nbragg = self._nbragg

        phase_op_idx = None
        for i, (op, args) in enumerate(operations):
            if op == 'split' and isinstance(args[0], (list, tuple, np.ndarray)):
                phase_op_idx = i
                break

        output_momentums = self._output_momentums
        nodedict = ifr.get_nodes(x_tolerance=1e-11)

        port_data = {}
        for m in output_momentums:
            if m not in nodedict:
                port_data[m] = (0.0, {})
                continue
            base = 0.0
            fourier = {}

            for pos, nodes in nodedict[m].items():
                harmonics_at_pos = {}
                for node in nodes:
                    ancestry = node.get_ancestry()
                    amp = np.ones(natoms, dtype=np.complex128)
                    h = 0

                    for op_i, (op_type, op_args) in enumerate(operations):
                        initial = ancestry[op_i]
                        final = ancestry[op_i + 1]

                        if op_type == 'propagate':
                            t_prop = op_args[0]
                            k = float(initial.k)
                            amp *= np.exp(-1j * t_prop * k * k)

                        elif op_type == 'split':
                            k_init = initial.k
                            k_final = final.k
                            t_pulse = float(initial.t)

                            Deltan = int(k_init) // 2
                            j_final = int(np.argmin(np.abs(kvec - k_final)))
                            j_raw = j_final - Deltan

                            if j_raw < 0 or j_raw >= nstates:
                                amp[:] = 0.0
                                break

                            bs = phi_raw[op_i, k_init][:, j_raw]
                            kv = float(kvec[j_raw])
                            delta_phase = deltas * t_pulse
                            amp *= bs * np.exp(-1j * delta_phase * kv / 2)

                            if phase_op_idx is not None and op_i == phase_op_idx:
                                h = int(round(-kv / 2))
                                if int(k_init) == 0 and int(k_final) == int(-2 * nbragg):
                                    amp *= np.exp(-1j * injected_dphase * kv / 2)

                    if h not in harmonics_at_pos:
                        harmonics_at_pos[h] = np.zeros(natoms, dtype=np.complex128)
                    harmonics_at_pos[h] += amp

                for h_val, amp_h in harmonics_at_pos.items():
                    base += float(np.sum(np.abs(amp_h) ** 2))

                hlist = sorted(harmonics_at_pos.keys())
                for i in range(len(hlist)):
                    for j in range(i + 1, len(hlist)):
                        dh = hlist[j] - hlist[i]
                        coeff = np.dot(harmonics_at_pos[hlist[i]].conj(),
                                       harmonics_at_pos[hlist[j]])
                        if dh not in fourier:
                            fourier[dh] = 0.0j
                        fourier[dh] += coeff
            port_data[m] = (base, fourier)

        all_dh = set()
        for m in output_momentums:
            all_dh.update(port_data[m][1].keys())

        if not all_dh:
            harmonics_arr = np.empty(0, dtype=np.float64)
            coeffs_real = np.empty((len(output_momentums), 0), dtype=np.float64)
            coeffs_imag = np.empty((len(output_momentums), 0), dtype=np.float64)
        else:
            harmonics_arr = np.array(sorted(all_dh), dtype=np.float64)
            dh_to_idx = {dh: i for i, dh in enumerate(sorted(all_dh))}
            n_h = len(harmonics_arr)
            n_p = len(output_momentums)
            coeffs_real = np.zeros((n_p, n_h), dtype=np.float64)
            coeffs_imag = np.zeros((n_p, n_h), dtype=np.float64)
            for pi, m in enumerate(output_momentums):
                for dh, c in port_data[m][1].items():
                    idx = dh_to_idx[dh]
                    coeffs_real[pi, idx] = c.real
                    coeffs_imag[pi, idx] = c.imag

        bases_arr = np.array([port_data[m][0] for m in output_momentums], dtype=np.float64)
        return bases_arr, harmonics_arr, coeffs_real, coeffs_imag

    def _get_sweep_coeffs(self, cloud, injected_dphase):
        key = (id(cloud['vz']), injected_dphase)
        cached = self._coeff_cache.get(key)
        if cached is not None:
            return cached
        self._coeff_cache.clear()
        phi_raw = self._precompute_pulses(cloud)
        coeffs = self._compute_fourier_dense(phi_raw, cloud, injected_dphase)
        self._coeff_cache[key] = coeffs
        return coeffs

    # ── public call ────────────────────────────────────────────────────

    def __call__(self, cloud, cphases, injected_dphase=0.0):
        """Evaluate port populations over the phase sweep.

        :param cloud: Dict with keys ``x0, y0, z0, vx, vy, vz``.
        :param cphases: 1D array of sweep phases.
        :param injected_dphase: Extra phase applied to the k=0 → k=-2*nbragg branch.
        :returns: Tuple of 1D arrays, one per entry in ``output_momentums``.
        """
        bases, harmonics, coeffs_real, coeffs_imag = self._get_sweep_coeffs(
            cloud, injected_dphase)

        cphases = np.asarray(cphases, dtype=np.float64)
        if harmonics.size == 0:
            out = np.broadcast_to(bases[:, None], (bases.size, cphases.size))
        else:
            angles = harmonics[:, None] * cphases[None, :]
            out = bases[:, None] + 2.0 * (
                coeffs_real @ np.cos(angles) - coeffs_imag @ np.sin(angles)
            )
        return tuple(out[p] for p in range(out.shape[0]))
