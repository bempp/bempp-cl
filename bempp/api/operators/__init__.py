"""Basic operator imports and definitions."""

import collections as _collections
import numpy as _np

from . import boundary
from . import potential
from . import far_field


OperatorDescriptor = _collections.namedtuple(
    "OperatorDescriptor", "identifier options compute_kernel"
)

MultitraceOperatorDescriptor = _collections.namedtuple(
    "MultitraceOperatorDescriptor",
    "identifier options multitrace_kernel singular_contribution",
)


def _add_wavenumber(options, wavenumber, identifier="WAVENUMBER"):
    """Add a real/complex wavenumber to an options array."""

    if 'kernel_parameters' not in options:
        options['kernel_parameters'] = []
    if _np.iscomplexobj(wavenumber):
        options[identifier + "_REAL"] = None
        options[identifier + "_COMPLEX"] = None
        options['kernel_parameters'].append(1.0 * _np.real(wavenumber))
        options['kernel_parameters'].append(1.0 * _np.imag(wavenumber))
    else:
        options[identifier + "_REAL"] = None
        options['kernel_parameters'].append(1.0 * _np.real(wavenumber))
