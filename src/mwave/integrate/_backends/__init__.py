"""Backend implementations for :py:func:`mwave.integrate.propagate`.

- ``'scipy'``  — adaptive ``solve_ivp`` (single-atom only, supports dense output)
- ``'numba'``  — Numba ``@njit(parallel=True)`` with ``prange``  (no external deps)
"""

from ._scipy import _run_scipy
from ._numba import (
    _rk45_dp_step,
    _rk45_bloch_adaptive,
)
