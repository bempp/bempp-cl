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
    target=None,
    space_type="electric_dual",
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
        One of "all_rwg", "all_dual", "electric_dual" (default),
        "magnetic_dual". These lead to the following
        choices of space, range, and dual_to_range:
        default - (RWG, RWG), (RWG, RWG), (SNC, SNC)
        all_dual - (BC, BC), (BC, BC), (RBC, RBC)
        electric_dual - (RWG, BC), (RWG, BC), (RBC, SNC)
        magnetic_dual - (BC, RWG), (BC, RWG), (SNC, RBC)
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
    The Maxwell multitrace operator of the form
    [[M, E], [-E, M]], where M represens the magnetic
    and E the electric field boundary operators in 
    the respective spaces defined through space_type.
    Note that the operators in the first and second
    row have different discretisations depending on
    the type of spaces used.

    """
    import bempp.api

    if space_type == "all_rwg":
        rwg = bempp.api.function_space(grid, "RWG", 0)
        snc = bempp.api.function_space(grid, "SNC", 0)

        if target is not None:
            rwg_target = bempp.api.function_space(target, "RWG", 0)
            snc_target = bempp.api.function_space(target, "SNC", 0)
        else:
            rwg_target = rwg
            snc_target = snc

        domain = [rwg, rwg]
        range_ = [rwg_target, rwg_target]
        dual_to_range = [snc_target, snc_target]
    elif space_type == "all_dual":
        bc = bempp.api.function_space(grid, "BC", 0)
        rbc = bempp.api.function_space(grid, "RBC", 0)

        if target is not None:
            bc_target = bempp.api.function_space(target, "BC", 0)
            rbc_target = bempp.api.function_space(target, "RBC", 0)
        else:
            bc_target = bc 
            rbc_target = rbc

        domain = [bc, bc]
        range_ = [bc_target, bc_target]
        dual_to_range = [rbc_target, rbc_target]

    elif space_type == "electric_dual":
        rwg = bempp.api.function_space(grid, "RWG", 0)
        snc = bempp.api.function_space(grid, "SNC", 0)
        bc = bempp.api.function_space(grid, "BC", 0)
        rbc = bempp.api.function_space(grid, "RBC", 0)

        if target is not None:
            rwg_target = bempp.api.function_space(target, "RWG", 0)
            snc_target = bempp.api.function_space(target, "SNC", 0)
            bc_target = bempp.api.function_space(target, "BC", 0)
            rbc_target = bempp.api.function_space(target, "RBC", 0)
        else:
            rwg_target = rwg
            snc_target = snc
            bc_target = bc
            rbc_target = rbc


        domain = [rwg, bc]
        range_ = [rwg_target, bc_target]
        dual_to_range = [rbc_target, snc_target]

    elif space_type == "magnetic_dual":
        rwg = bempp.api.function_space(grid, "RWG", 0)
        snc = bempp.api.function_space(grid, "SNC", 0)
        bc = bempp.api.function_space(grid, "BC", 0)
        rbc = bempp.api.function_space(grid, "RBC", 0)

        if target is not None:
            rwg_target = bempp.api.function_space(target, "RWG", 0)
            snc_target = bempp.api.function_space(target, "SNC", 0)
            bc_target = bempp.api.function_space(target, "BC", 0)
            rbc_target = bempp.api.function_space(target, "RBC", 0)
        else:
            rwg_target = rwg
            snc_target = snc
            bc_target = bc
            rbc_target = rbc

        domain = [bc, rwg]
        range_ = [bc_target, rwg_target]
        dual_to_range = [snc_target, rbc_target]

    else:
        raise ValueError(
            "space_type must be one of 'all_rwg', 'all_dual', 'electric_dual', 'magnetic_dual'"
        )
    return _multitrace_operator_impl(
        domain,
        range_,
        dual_to_range,
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

def _multitrace_operator_impl(
    domain,
    range_,
    dual_to_range,
    wavenumber,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):

    from bempp.api.assembly.blocked_operator import BlockedOperator

    magnetic1 = magnetic_field(
        domain[0],
        range_[0],
        dual_to_range[0],
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    magnetic2 = magnetic_field(
        domain[1],
        range_[1],
        dual_to_range[1],
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    electric1 = electric_field(
        domain[1],
        range_[0],
        dual_to_range[0],
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    electric2 = electric_field(
        domain[0],
        range_[1],
        dual_to_range[1],
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    blocked = BlockedOperator(2, 2)
    blocked[0, 0] = magnetic1
    blocked[0, 1] = electric1
    blocked[1, 0] = -electric2
    blocked[1, 1] = magnetic2

    return blocked


# def multitrace_operator(
# grid,
# wavenumber,
# segments=None,
# parameters=None,
# swapped_normals=None,
# assembler="multitrace_evaluator",
# device_interface=None,
# precision=None,
# ):
# """Assemble the Maxwell multitrace operator."""
# from bempp.api.space import function_space
# from bempp.api.operators import _add_wavenumber

# if assembler != "multitrace_evaluator":
# raise ValueError("Only multitrace evaluator supported.")

# domain = function_space(
# grid,
# "RWG",
# 0,
# segments=segments,
# swapped_normals=swapped_normals,
# include_boundary_dofs=True,
# )
# range_ = domain
# dual_to_range = function_space(
# grid,
# "SNC",
# 0,
# segments=segments,
# swapped_normals=swapped_normals,
# include_boundary_dofs=True,
# )
# magnetic = magnetic_field(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# "only_singular_part",
# device_interface,
# precision,
# )
# electric = electric_field(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# "only_singular_part",
# device_interface,
# precision,
# )

# options = {"COMPLEX_KERNEL": None}

# _add_wavenumber(options, wavenumber)

# singular_contribution = _np.array(
# [[magnetic, electric], [-electric, magnetic]], dtype=_np.object
# )
# return _common.create_multitrace_operator(
# "maxwell_multitrace",
# [domain, domain],
# [range_, range_],
# [dual_to_range, dual_to_range],
# parameters,
# assembler,
# options,
# "maxwell_multitrace",
# singular_contribution,
# device_interface,
# precision,
# )


# def transmission_operator(
# grid,
# wavenumber,
# eps_rel,
# mu_rel,
# segments=None,
# parameters=None,
# swapped_normals=None,
# assembler="multitrace_evaluator",
# device_interface=None,
# precision=None,
# ):
# """Assemble the Maxwell transmission operator."""
# from bempp.api.space import function_space
# from bempp.api.operators import _add_wavenumber

# if assembler != "multitrace_evaluator":
# raise ValueError("Only multitrace evaluator supported.")

# domain = function_space(
# grid,
# "RWG",
# 0,
# segments=segments,
# swapped_normals=swapped_normals,
# include_boundary_dofs=True,
# )
# range_ = domain
# dual_to_range = function_space(
# grid,
# "SNC",
# 0,
# segments=segments,
# swapped_normals=swapped_normals,
# include_boundary_dofs=True,
# )

# sqrt_eps_rel = _np.sqrt(eps_rel)
# sqrt_mu_rel = _np.sqrt(mu_rel)

# wavenumber_int = wavenumber * sqrt_eps_rel * sqrt_mu_rel

# magnetic_ext = magnetic_field(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# "only_singular_part",
# device_interface,
# precision,
# )

# magnetic_int = magnetic_field(
# domain,
# range_,
# dual_to_range,
# wavenumber_int,
# parameters,
# "only_singular_part",
# device_interface,
# precision,
# )

# electric_ext = electric_field(
# domain,
# range_,
# dual_to_range,
# wavenumber,
# parameters,
# "only_singular_part",
# device_interface,
# precision,
# )

# electric_int = electric_field(
# domain,
# range_,
# dual_to_range,
# wavenumber_int,
# parameters,
# "only_singular_part",
# device_interface,
# precision,
# )

# options = {"COMPLEX_KERNEL": None, "TRANSMISSION": None}

# _add_wavenumber(options, wavenumber)
# _add_wavenumber(options, sqrt_mu_rel, "SQRT_MU_REL")
# _add_wavenumber(options, sqrt_eps_rel, "SQRT_EPS_REL")
# _add_wavenumber(options, wavenumber_int, "WAVENUMBER_INT")
# _add_wavenumber(options, sqrt_mu_rel / sqrt_eps_rel, "SQRT_MU_EPS_RATIO")
# _add_wavenumber(options, sqrt_eps_rel / sqrt_mu_rel, "SQRT_EPS_MU_RATIO")

# fac = sqrt_mu_rel / sqrt_eps_rel

# singular_contribution = _np.array(
# [
# [magnetic_ext + magnetic_int, electric_ext + fac * electric_int],
# [(-1.0 / fac) * electric_int - electric_ext, magnetic_int + magnetic_ext],
# ],
# dtype=_np.object,
# )
# return _common.create_multitrace_operator(
# "maxwell_transmission",
# [domain, domain],
# [range_, range_],
# [dual_to_range, dual_to_range],
# parameters,
# assembler,
# options,
# "maxwell_multitrace",
# singular_contribution,
# device_interface,
# precision,
# )
