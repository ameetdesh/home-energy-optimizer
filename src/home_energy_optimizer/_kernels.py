"""Optional compiled backward recursions.

`wasm/build/dp_kernels.pyx` is a C translation of the inner loops in
`dp_battery` and `dp_thermal`, cross-compiled to a Pyodide wasm wheel so the
browser build is not bound by Pyodide's numpy. It is entirely optional: when
the module is absent the Python paths run unchanged, and they remain the
reference implementation either way.

These are plain functions rather than methods on a module object because the
browser build flattens the package into one namespace, where `_kernels.battery`
has no module to resolve against.

The dispatch is deliberately conservative. The kernels implement the DEFAULT
configuration only, so anything exotic - SoC gates, per-slot charge floors,
soc goals, the charger deadband - stays in Python rather than being duplicated
in C where it would have to be kept in step by hand.
"""

from __future__ import annotations

try:  # pragma: no cover - presence depends on the build
    import dp_kernels as _k
except ImportError:  # pragma: no cover
    _k = None

HAVE_KERNELS = _k is not None


def kernel_battery(*args) -> bool:
    if _k is None:
        return False
    _k.battery_backward(*args)
    return True


def kernel_water_heater(*args) -> bool:
    if _k is None:
        return False
    _k.water_heater_backward(*args)
    return True


def kernel_hvac(*args) -> bool:
    if _k is None:
        return False
    _k.hvac_backward(*args)
    return True
