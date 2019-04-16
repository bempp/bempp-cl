"""Interfaces to Maxwell operators."""
import numpy as _np

from bempp.api.operators.boundary import common as _common


def electric_field(
    domain,
    range_,
    dual_to_range,
    wavenumber,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the electric field boundary operator."""
    from bempp.api.operators import _add_wavenumber

    if not (domain.identifier == "rwg0" and dual_to_range.identifier == "snc0"):
        raise ValueError(
            "Operator only defined for domain = 'rwg' and 'dual_to_range = 'snc"
        )

    options = {"KERNEL_FUNCTION": "helmholtz_single_layer", "COMPLEX_KERNEL": None}

    _add_wavenumber(options, wavenumber)

    return _common.create_operator(
        "maxwell_electric_field_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        options,
        "maxwell_electric_field",
        device_interface,
        precision,
    )


def magnetic_field(
    domain,
    range_,
    dual_to_range,
    wavenumber,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the electric field boundary operator."""
    from bempp.api.operators import _add_wavenumber

    if not (domain.identifier == "rwg0" and dual_to_range.identifier == "snc0"):
        raise ValueError(
            "Operator only defined for domain = 'rwg' and 'dual_to_range = 'snc"
        )

    options = {"KERNEL_FUNCTION": "helmholtz_gradient", "COMPLEX_KERNEL": None}

    _add_wavenumber(options, wavenumber)

    return _common.create_operator(
        "maxwell_magnetic_field_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        options,
        "maxwell_magnetic_field",
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
    """Assemble the Maxwell multitrace operator."""
    from bempp.api.space import function_space
    from bempp.api.operators import _add_wavenumber

    if assembler != "multitrace_evaluator":
        raise ValueError("Only multitrace evaluator supported.")

    domain = function_space(grid, "RWG", 0)
    range_ = domain
    dual_to_range = function_space(grid, "SNC", 0)
    magnetic = magnetic_field(
        domain,
        range_,
        dual_to_range,
        wavenumber,
        parameters,
        "only_singular_part",
        device_interface,
        precision,
    )
    electric = electric_field(
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
        [[magnetic, electric], [-electric, magnetic]], dtype=_np.object
    )
    return _common.create_multitrace_operator(
        "maxwell_multitrace",
        [domain, domain],
        [range_, range_],
        [dual_to_range, dual_to_range],
        parameters,
        assembler,
        options,
        "maxwell_multitrace",
        singular_contribution,
        device_interface,
        precision,
    )


def transmission_operator(
    grid,
    wavenumber,
    eps_rel,
    mu_rel,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Maxwell transmission operator."""
    from bempp.api.space import function_space
    from bempp.api.operators import _add_wavenumber

    if assembler != "multitrace_evaluator":
        raise ValueError("Only multitrace evaluator supported.")

    domain = function_space(grid, "RWG", 0)
    range_ = domain
    dual_to_range = function_space(grid, "SNC", 0)

    sqrt_eps_rel = _np.sqrt(eps_rel)
    sqrt_mu_rel = _np.sqrt(mu_rel)

    wavenumber_int = wavenumber * sqrt_eps_rel * sqrt_mu_rel

    magnetic_ext = magnetic_field(
        domain,
        range_,
        dual_to_range,
        wavenumber,
        parameters,
        "only_singular_part",
        device_interface,
        precision,
    )

    magnetic_int = magnetic_field(
        domain,
        range_,
        dual_to_range,
        wavenumber_int,
        parameters,
        "only_singular_part",
        device_interface,
        precision,
    )

    electric_ext = electric_field(
        domain,
        range_,
        dual_to_range,
        wavenumber,
        parameters,
        "only_singular_part",
        device_interface,
        precision,
    )

    electric_int = electric_field(
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
    _add_wavenumber(options, sqrt_mu_rel, "SQRT_MU_REL")
    _add_wavenumber(options, sqrt_eps_rel, "SQRT_EPS_REL")
    _add_wavenumber(options, wavenumber_int, "WAVENUMBER_INT")
    _add_wavenumber(options, sqrt_mu_rel / sqrt_eps_rel, "SQRT_MU_EPS_RATIO")
    _add_wavenumber(options, sqrt_eps_rel / sqrt_mu_rel, "SQRT_EPS_MU_RATIO")

    fac = sqrt_mu_rel / sqrt_eps_rel

    singular_contribution = _np.array(
        [
            [magnetic_ext + magnetic_int, electric_ext + fac * electric_int],
            [(-1.0 / fac) * electric_int - electric_ext, magnetic_int + magnetic_ext],
        ],
        dtype=_np.object,
    )
    return _common.create_multitrace_operator(
        "maxwell_multitrace",
        [domain, domain],
        [range_, range_],
        [dual_to_range, dual_to_range],
        parameters,
        assembler,
        options,
        "maxwell_multitrace",
        singular_contribution,
        device_interface,
        precision,
    )
