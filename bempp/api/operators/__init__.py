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


def _add_wavenumber(options, wavenumber):
    """Add a real/complex wavenumber to an options array."""

    if 'kernel_parameters' not in options:
        options['kernel_parameters'] = []
    if 'source' not in options:
        options['source'] = dict()
    if _np.iscomplexobj(wavenumber):
        options["source"]["WAVENUMBER_REAL"] = None
        options["source"]["WAVENUMBER_COMPLEX"] = None
        options['kernel_parameters'].append(1.0 * _np.real(wavenumber))
        options['kernel_parameters'].append(1.0 * _np.imag(wavenumber))
    else:
        options["source"]["WAVENUMBER_REAL"] = None
        options['kernel_parameters'].append(1.0 * _np.real(wavenumber))
        options['kernel_parameters'].append(1.0 * _np.imag(wavenumber))

    return options

def _add_source_option(options, name, value=None):
    """Add a source parameter."""

    if 'source' not in options:
        options['source'] = dict()

    options['source'][name] = value

    return options

