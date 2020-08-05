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

    return _common.create_operator(
        "helmholtz_single_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [_np.real(wavenumber), _np.imag(wavenumber)],
        "helmholtz_single_layer",
        "default_scalar",
        device_interface,
        precision,
        True,
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

    return _common.create_operator(
        "helmholtz_double_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [_np.real(wavenumber), _np.imag(wavenumber)],
        "helmholtz_double_layer",
        "default_scalar",
        device_interface,
        precision,
        True,
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

    return _common.create_operator(
        "helmholtz_adjoint_double_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [_np.real(wavenumber), _np.imag(wavenumber)],
        "helmholtz_adjoint_double_layer",
        "default_scalar",
        device_interface,
        precision,
        True,
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

    return _common.create_operator(
        "helmholtz_hypersingular_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [_np.real(wavenumber), _np.imag(wavenumber)],
        "helmholtz_single_layer",
        "helmholtz_hypersingular",
        device_interface,
        precision,
        True,
    )


def multitrace_operator(
    grid,
    wavenumber,
    target=None,
    space_type="p1",
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """
    Simplified version of multitrace operator assembly.

    Parameters
    ----------
    grid : Grid
        Bempp grid object.
    wavenumber : complex
        A real or complex wavenumber
    target : Grid
        The grid for the range spaces. If target is None then
        target is set to the input grid (that is the domain
        grid).
    space_type : string
        Currently only "p1" is supported, which means
        that the operator is discretised with all P1 basis
        functions.
    parameters : Parameters
        An optional parameters object.
    assembler : string
        The assembler type.
    device_interface : DeviceInterface
        The device interface object to be used.
    precision : string
        Either "single" or "double" for single or
        double precision mode.

    Output
    ------
    The Helmholtz multitrace operator of the form
    [[-dlp, slp], [hyp, adj_dlp]], where
    dlp : double layer boundary operator
    slp : single layer boundary operator
    hyp : hypersingular boundary operator
    adj_dlp : adjoint double layer boundary operator.

    """
    import bempp.api
    from bempp.api.assembly.blocked_operator import BlockedOperator
    space = bempp.api.function_space(grid, "P", 1)

    if target is not None:
        target_space = bempp.api.function_space(target, "P", 1)
    else:
        target_space = space

    slp = single_layer(
        space,
        target_space,
        target_space,
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    dlp = double_layer(
        space,
        target_space,
        target_space,
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    hyp = hypersingular(
        space,
        target_space,
        target_space,
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    adj_dlp = adjoint_double_layer(
        space,
        target_space,
        target_space,
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    blocked = BlockedOperator(2, 2)

    blocked[0, 0] = -dlp
    blocked[0, 1] = slp
    blocked[1, 0] = hyp
    blocked[1, 1] = adj_dlp

    return blocked


# def multitrace_operator(
# grid,
# wavenumber,
# segments=None,
# parameters=None,
# swapped_normals=None,
# assembler="dense_evaluator",
# device_interface=None,
# precision=None,
# ):
# """Assemble the Helmholtz multitrace operator."""
# from bempp.api.space import function_space
# from bempp.api.operators import _add_wavenumber
# from bempp.api.assembly.blocked_operator import GeneralizedBlockedOperator

# domain = function_space(
# grid,
# "P",
# 1,
# segments=segments,
# include_boundary_dofs=True,
# swapped_normals=swapped_normals,
# )
# range_ = domain
# dual_to_range = domain

# slp = single_layer(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# dlp = double_layer(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# adlp = adjoint_double_layer(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# hyp = hypersingular(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# options = {"COMPLEX_KERNEL": None}

# _add_wavenumber(options, wavenumber)

# return GeneralizedBlockedOperator([[-dlp, slp], [hyp, adlp]])


# def transmission_operator(
# grid,
# wavenumber,
# rho_rel,
# refractive_index,
# segments=None,
# parameters=None,
# swapped_normals=None,
# assembler="dense_evaluator",
# device_interface=None,
# precision=None,
# ):
# """Assemble the Helmholtz transmission operator."""
# from bempp.api.space import function_space
# from bempp.api.operators import _add_wavenumber
# from bempp.api.assembly.blocked_operator import GeneralizedBlockedOperator


# wavenumber_int = wavenumber * refractive_index

# domain = function_space(
# grid,
# "P",
# 1,
# segments=segments,
# swapped_normals=swapped_normals,
# include_boundary_dofs=True,
# )
# range_ = domain
# dual_to_range = domain

# slp = single_layer(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# slp_int = single_layer(
# domain,
# range_,
# dual_to_range,
# wavenumber_int,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# dlp = double_layer(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# dlp_int = double_layer(
# domain,
# range_,
# dual_to_range,
# wavenumber_int,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# adlp = adjoint_double_layer(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# adlp_int = adjoint_double_layer(
# domain,
# range_,
# dual_to_range,
# wavenumber_int,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# hyp = hypersingular(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# hyp_int = hypersingular(
# domain,
# range_,
# dual_to_range,
# wavenumber_int,
# parameters,
# assembler,
# device_interface,
# precision,
# )

# options = {"COMPLEX_KERNEL": None, "TRANSMISSION": None}

# _add_wavenumber(options, wavenumber)
# _add_wavenumber(options, rho_rel, "RHO_REL")
# _add_wavenumber(options, wavenumber_int, "WAVENUMBER_INT")

# return GeneralizedBlockedOperator(
# [
# [-dlp - dlp_int, slp + rho_rel * slp_int],
# [hyp + 1.0 / rho_rel * hyp_int, adlp + adlp_int],
# ]
# )
