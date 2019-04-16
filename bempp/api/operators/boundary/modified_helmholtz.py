"""Interfaces to modified Helmholtz operators."""
from bempp.api.operators.boundary import common as _common


def single_layer(
    domain,
    range_,
    dual_to_range,
    omega,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the mod. Helmholtz single-layer boundary operator."""
    return _common.create_operator(
        "modified_helmholtz_real_single_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        {
            "KERNEL_FUNCTION": "modified_helmholtz_real_single_layer",
            "OMEGA": 1.0 * omega,
        },
        "default_scalar",
        device_interface,
        precision,
    )  # Ensure that variable is float type


def double_layer(
    domain,
    range_,
    dual_to_range,
    omega,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the mod. Helmholtz double-layer boundary operator."""
    return _common.create_operator(
        "modified_helmholtz_real_double_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        {
            "KERNEL_FUNCTION": "modified_helmholtz_real_double_layer",
            "OMEGA": 1.0 * omega,
        },
        "default_scalar",
        device_interface,
        precision,
    )  # Ensure that variable is float type


def adjoint_double_layer(
    domain,
    range_,
    dual_to_range,
    omega,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the mod. Helmholtz adj. double-layer boundary op."""
    return _common.create_operator(
        "modified_helmholtz_real_adjoint_double_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        {
            "KERNEL_FUNCTION": "modified_helmholtz_real_adjoint_double_layer",
            "OMEGA": 1.0 * omega,
        },
        "default_scalar",
        device_interface,
        precision,
    )  # Ensure that variable is float type


def hypersingular(
    domain,
    range_,
    dual_to_range,
    omega,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the mod. Helmholtz hypersingular boundary op."""
    return _common.create_operator(
        "modified_helmholtz_real_hypersingular_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        {
            "KERNEL_FUNCTION": "modified_helmholtz_real_single_layer",
            "OMEGA": 1.0 * omega,
        },
        "helmholtz_hypersingular",
        device_interface,
        precision,
    )  # Ensure that variable is float type
