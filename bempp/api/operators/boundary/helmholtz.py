"""Interfaces to Helmholtz operators."""
import numpy as _np

from bempp.api.operators.boundary import common as _common


def single_layer(
    domain,
    range_,
    dual_to_range,
    wavenumber,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Helmholtz single-layer boundary operator."""
    from bempp.api.operators import _add_wavenumber
    from .modified_helmholtz import single_layer as _modified_single_layer

    if _np.real(wavenumber) == 0:
        return _modified_single_layer(
            domain,
            range_,
            dual_to_range,
            _np.imag(wavenumber),
            parameters,
            assembler,
            device_interface,
            precision,
        )

    options = {"KERNEL_FUNCTION": "helmholtz_single_layer", "COMPLEX_KERNEL": None}

    _add_wavenumber(options, wavenumber)

    return _common.create_operator(
        "helmholtz_single_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        options,
        "default_scalar",
        device_interface,
        precision,
    )


def double_layer(
    domain,
    range_,
    dual_to_range,
    wavenumber,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Helmholtz double-layer boundary operator."""
    from bempp.api.operators import _add_wavenumber
    from .modified_helmholtz import double_layer as _modified_double_layer

    if _np.real(wavenumber) == 0:
        return _modified_double_layer(
            domain,
            range_,
            dual_to_range,
            _np.imag(wavenumber),
            parameters,
            assembler,
            device_interface,
            precision,
        )

    options = {"KERNEL_FUNCTION": "helmholtz_double_layer", "COMPLEX_KERNEL": None}

    _add_wavenumber(options, wavenumber)

    return _common.create_operator(
        "helmholtz_double_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        options,
        "default_scalar",
        device_interface,
        precision,
    )


def adjoint_double_layer(
    domain,
    range_,
    dual_to_range,
    wavenumber,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Helmholtz adj. double-layer boundary operator."""
    from bempp.api.operators import _add_wavenumber
    from .modified_helmholtz import (
        adjoint_double_layer as _modified_adjoint_double_layer,
    )

    if _np.real(wavenumber) == 0:
        return _modified_adjoint_double_layer(
            domain,
            range_,
            dual_to_range,
            _np.imag(wavenumber),
            parameters,
            assembler,
            device_interface,
            precision,
        )

    options = {
        "KERNEL_FUNCTION": "helmholtz_adjoint_double_layer",
        "COMPLEX_KERNEL": None,
    }

    _add_wavenumber(options, wavenumber)

    return _common.create_operator(
        "helmholtz_adjoint_double_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        options,
        "default_scalar",
        device_interface,
        precision,
    )


def hypersingular(
    domain,
    range_,
    dual_to_range,
    wavenumber,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Helmholtz hypersingular boundary operator."""
    from bempp.api.operators import _add_wavenumber
    from .modified_helmholtz import hypersingular as _hypersingular

    if _np.real(wavenumber) == 0:
        return _hypersingular(
            domain,
            range_,
            dual_to_range,
            _np.imag(wavenumber),
            parameters,
            assembler,
            device_interface,
            precision,
        )

    options = {"KERNEL_FUNCTION": "helmholtz_single_layer", "COMPLEX_KERNEL": None}

    _add_wavenumber(options, wavenumber)

    return _common.create_operator(
        "helmholtz_hypersingular_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        options,
        "helmholtz_hypersingular",
        device_interface,
        precision,
    )


def multitrace_operator(
    grid,
    wavenumber,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Helmholtz multitrace operator."""
    from bempp.api.space import function_space
    from bempp.api.operators import _add_wavenumber

    if assembler != "multitrace_evaluator":
        raise ValueError("Only multitrace evaluator supported.")

    domain = function_space(grid, "P", 1)
    range_ = domain
    dual_to_range = domain

    slp = single_layer(
        domain,
        range_,
        dual_to_range,
        wavenumber,
        parameters,
        "only_singular_part",
        device_interface,
        precision,
    )

    dlp = double_layer(
        domain,
        range_,
        dual_to_range,
        wavenumber,
        parameters,
        "only_singular_part",
        device_interface,
        precision,
    )

    adlp = adjoint_double_layer(
        domain,
        range_,
        dual_to_range,
        wavenumber,
        parameters,
        "only_singular_part",
        device_interface,
        precision,
    )

    hyp = hypersingular(
        domain,
        range_,
        dual_to_range,
        wavenumber,
        parameters,
        "only_singular_part",
        device_interface,
        precision,
    )

    options = {"COMPLEX_KERNEL": None}

    _add_wavenumber(options, wavenumber)

    singular_contribution = _np.array(
        [[-dlp, slp], [hyp, adlp]], dtype=_np.object
    )
    return _common.create_multitrace_operator(
        "helmholtz_multitrace",
        [domain, domain],
        [range_, range_],
        [dual_to_range, dual_to_range],
        parameters,
        assembler,
        options,
        "helmholtz_multitrace",
        singular_contribution,
        device_interface,
        precision,
    )

def transmission_operator(
    grid,
    wavenumber,
    rho_rel,
    refractive_index,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Helmholtz transmission operator."""
    from bempp.api.space import function_space
    from bempp.api.operators import _add_wavenumber

    if assembler != "multitrace_evaluator":
        raise ValueError("Only multitrace evaluator supported.")

    wavenumber_int = wavenumber * refractive_index

    domain = function_space(grid, "P", 1)
    range_ = domain
    dual_to_range = domain

    slp = single_layer(
        domain,
        range_,
        dual_to_range,
        wavenumber,
        parameters,
        "only_singular_part",
        device_interface,
        precision,
    )

    slp_int = single_layer(
        domain,
        range_,
        dual_to_range,
        wavenumber_int,
        parameters,
        "only_singular_part",
        device_interface,
        precision,
    )

    dlp = double_layer(
        domain,
        range_,
        dual_to_range,
        wavenumber,
        parameters,
        "only_singular_part",
        device_interface,
        precision,
    )

    dlp_int = double_layer(
        domain,
        range_,
        dual_to_range,
        wavenumber_int,
        parameters,
        "only_singular_part",
        device_interface,
        precision,
    )

    adlp = adjoint_double_layer(
        domain,
        range_,
        dual_to_range,
        wavenumber,
        parameters,
        "only_singular_part",
        device_interface,
        precision,
    )

    adlp_int = adjoint_double_layer(
        domain,
        range_,
        dual_to_range,
        wavenumber_int,
        parameters,
        "only_singular_part",
        device_interface,
        precision,
    )

    hyp = hypersingular(
        domain,
        range_,
        dual_to_range,
        wavenumber,
        parameters,
        "only_singular_part",
        device_interface,
        precision,
    )

    hyp_int = hypersingular(
        domain,
        range_,
        dual_to_range,
        wavenumber_int,
        parameters,
        "only_singular_part",
        device_interface,
        precision,
    )

    options = {"COMPLEX_KERNEL": None, "TRANSMISSION": None}

    _add_wavenumber(options, wavenumber)
    _add_wavenumber(options, rho_rel, "RHO_REL")
    _add_wavenumber(options, wavenumber_int, "WAVENUMBER_INT")

    singular_contribution = _np.array(
        [[-dlp - dlp_int, slp + rho_rel * slp_int], [hyp + 1. / rho_rel * hyp_int, adlp + adlp_int]], dtype=_np.object
    )

    return _common.create_multitrace_operator(
        "helmholtz_transmission",
        [domain, domain],
        [range_, range_],
        [dual_to_range, dual_to_range],
        parameters,
        assembler,
        options,
        "helmholtz_multitrace",
        singular_contribution,
        device_interface,
        precision,
    )
