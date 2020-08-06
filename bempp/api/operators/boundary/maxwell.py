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

    return _common.create_operator(
        "maxwell_electric_field_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [_np.real(wavenumber), _np.imag(wavenumber)],
        "helmholtz_single_layer",
        "maxwell_electric_field",
        device_interface,
        precision,
        True,
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
    """Assemble the magnetic field boundary operator."""

    return _common.create_operator(
        "maxwell_magnetic_field_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [_np.real(wavenumber), _np.imag(wavenumber)],
        "helmholtz_single_layer",
        "maxwell_magnetic_field",
        device_interface,
        precision,
        True,
    )


# def magnetic_field(
#     domain,
#     range_,
#     dual_to_range,
#     wavenumber,
#     parameters=None,
#     assembler="default_nonlocal",
#     device_interface=None,
#     precision=None,
# ):
#     """Assemble the electric field boundary operator."""
#     from bempp.api.operators import _add_wavenumber

#     if not (domain.identifier == "rwg0" and dual_to_range.identifier == "snc0"):
#         raise ValueError(
#             "Operator only defined for domain = 'rwg' and 'dual_to_range = 'snc"
#         )

#     options = {"KERNEL_FUNCTION": "helmholtz_gradient", "COMPLEX_KERNEL": None}

#     _add_wavenumber(options, wavenumber)

#     return _common.create_operator(
#         "maxwell_magnetic_field_boundary",
#         domain,
#         range_,
#         dual_to_range,
#         parameters,
#         assembler,
#         options,
#         "maxwell_magnetic_field",
#         device_interface,
#         precision,
#     )

# def multitrace_operator(
#     grid,
#     wavenumber,
#     epsilon_r=1,
#     mu_r=1,
#     target=None,
#     space_type="magnetic_dual",
#     parameters=None,
#     assembler="default_nonlocal",
#     device_interface=None,
#     precision=None,
# ):
#     """
#     Simplified version of multitrace operator assembly.

#     Parameters
#     ----------
#     grid : Grid
#         Bempp grid object.
#     wavenumber : complex
#         A real or complex wavenumber
#     epsilon_r : float
#         Relative permittivity with respect to vacuum.
#     mu_r : float
#         Relative permeability with respect to vacuum.
#     target : Grid
#         The grid for the range spaces. If target is None then
#         target is set to the input grid (that is the domain
#         grid).
#     space_type : string
#         One of "all_rwg", "all_bc", "magnetic_dual" (default),
#         "electric_dual". These lead to the following
#         choices of space, range, and dual_to_range:
#         default - (RWG, RWG), (BC, BC), (SNC, SNC)
#         all_dual - (BC, BC), (RWG, RWG), (RBC, RBC)
#         magnetic_dual - (RWG, BC), (RWG, BC), (RBC, SNC)
#         electric_dual - (BC, RWG), (BC, RWG), (SNC, RBC)
#     parameters : Parameters
#         An optional parameters object.
#     assembler : string
#         The assembler type.
#     device_interface : DeviceInterface
#         The device interface object to be used.
#     precision : string
#         Either "single" or "double" for single or
#         double precision mode.

#     Output
#     ------
#     The Maxwell multitrace operator of the form
#     [[M, E], [-E, M]], where M represens the magnetic
#     and E the electric field boundary operators in
#     the respective spaces defined through space_type.
#     Note that the operators in the first and second
#     row have different discretisations depending on
#     the type of spaces used.

#     """
#     import bempp.api

#     if space_type == "all_rwg":
#         rwg = bempp.api.function_space(grid, "RWG", 0)
#         bc = bempp.api.function_space(grid, "BC", 0)
#         snc = bempp.api.function_space(grid, "SNC", 0)

#         if target is not None:
#             bc_target = bempp.api.function_space(target, "BC", 0)
#             snc_target = bempp.api.function_space(target, "SNC", 0)
#         else:
#             bc_target = bc
#             snc_target = snc

#         domain = [rwg, rwg]
#         range_ = [bc_target, bc_target]
#         dual_to_range = [snc_target, snc_target]
#     elif space_type == "all_bc":
#         bc = bempp.api.function_space(grid, "BC", 0)
#         rwg = bempp.api.function_space(grid, "RWG", 0)
#         rbc = bempp.api.function_space(grid, "RBC", 0)

#         if target is not None:
#             rwg_target = bempp.api.function_space(target, "RWG", 0)
#             rbc_target = bempp.api.function_space(target, "RBC", 0)
#         else:
#             rwg_target = rwg
#             rbc_target = rbc

#         domain = [bc, bc]
#         range_ = [rwg_target, rwg_target]
#         dual_to_range = [rbc_target, rbc_target]

#     elif space_type == "electric_dual":
#         rwg = bempp.api.function_space(grid, "RWG", 0)
#         snc = bempp.api.function_space(grid, "SNC", 0)
#         bc = bempp.api.function_space(grid, "BC", 0)
#         rbc = bempp.api.function_space(grid, "RBC", 0)

#         if target is not None:
#             rwg_target = bempp.api.function_space(target, "RWG", 0)
#             snc_target = bempp.api.function_space(target, "SNC", 0)
#             bc_target = bempp.api.function_space(target, "BC", 0)
#             rbc_target = bempp.api.function_space(target, "RBC", 0)
#         else:
#             rwg_target = rwg
#             snc_target = snc
#             bc_target = bc
#             rbc_target = rbc


#         domain = [rwg, bc]
#         range_ = [rwg_target, bc_target]
#         dual_to_range = [rbc_target, snc_target]

#     elif space_type == "magnetic_dual":
#         rwg = bempp.api.function_space(grid, "RWG", 0)
#         snc = bempp.api.function_space(grid, "SNC", 0)
#         bc = bempp.api.function_space(grid, "BC", 0)
#         rbc = bempp.api.function_space(grid, "RBC", 0)

#         if target is not None:
#             rwg_target = bempp.api.function_space(target, "RWG", 0)
#             snc_target = bempp.api.function_space(target, "SNC", 0)
#             bc_target = bempp.api.function_space(target, "BC", 0)
#             rbc_target = bempp.api.function_space(target, "RBC", 0)
#         else:
#             rwg_target = rwg
#             snc_target = snc
#             bc_target = bc
#             rbc_target = rbc

#         domain = [bc, rwg]
#         range_ = [bc_target, rwg_target]
#         dual_to_range = [snc_target, rbc_target]

#     else:
#         raise ValueError(
#             "space_type must be one of 'all_rwg', 'all_dual', 'electric_dual', 'magnetic_dual'"
#         )
#     return _multitrace_operator_impl(
#         domain,
#         range_,
#         dual_to_range,
#         wavenumber,
#         epsilon_r=epsilon_r,
#         mu_r=mu_r,
#         parameters=parameters,
#         assembler=assembler,
#         device_interface=device_interface,
#         precision=precision,
#     )

# def _multitrace_operator_impl(
#     domain,
#     range_,
#     dual_to_range,
#     wavenumber,
#     epsilon_r=1,
#     mu_r=1,
#     parameters=None,
#     assembler="default_nonlocal",
#     device_interface=None,
#     precision=None,
# ):

#     from bempp.api.assembly.blocked_operator import BlockedOperator

#     rho = _np.sqrt(epsilon_r / mu_r)

#     magnetic1 = magnetic_field(
#         domain[0],
#         range_[0],
#         dual_to_range[0],
#         wavenumber,
#         parameters=parameters,
#         assembler=assembler,
#         device_interface=device_interface,
#         precision=precision,
#     )

#     magnetic2 = magnetic_field(
#         domain[1],
#         range_[1],
#         dual_to_range[1],
#         wavenumber,
#         parameters=parameters,
#         assembler=assembler,
#         device_interface=device_interface,
#         precision=precision,
#     )

#     electric1 = electric_field(
#         domain[1],
#         range_[0],
#         dual_to_range[0],
#         wavenumber,
#         parameters=parameters,
#         assembler=assembler,
#         device_interface=device_interface,
#         precision=precision,
#     )

#     electric2 = electric_field(
#         domain[0],
#         range_[1],
#         dual_to_range[1],
#         wavenumber,
#         parameters=parameters,
#         assembler=assembler,
#         device_interface=device_interface,
#         precision=precision,
#     )

#     blocked = BlockedOperator(2, 2)
#     blocked[0, 0] = magnetic1
#     blocked[0, 1] = (1. / rho) * electric1
#     blocked[1, 0] = -rho * electric2
#     blocked[1, 1] = magnetic2

#     return blocked
