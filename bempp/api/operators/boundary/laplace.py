"""Interfaces to Laplace operators."""
from bempp.api.operators.boundary import common as _common


def single_layer(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Laplace single-layer boundary operator."""
    return _common.create_operator(
        "laplace_single_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [],
        "laplace_single_layer",
        "default_scalar",
        device_interface,
        precision,
        False,
    )


def double_layer(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Laplace double-layer boundary operator."""
    return _common.create_operator(
        "laplace_double_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [],
        "laplace_double_layer",
        "default_scalar",
        device_interface,
        precision,
        False,
    )


def adjoint_double_layer(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Laplace adjoint double-layer boundary operator."""
    return _common.create_operator(
        "laplace_adjoint_double_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [],
        "laplace_adjoint_double_layer",
        "default_scalar",
        device_interface,
        precision,
        False,
    )


def hypersingular(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Laplace hypersingular boundary operator."""
    return _common.create_operator(
        "laplace_hypersingular_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        {"KERNEL_FUNCTION": "laplace_single_layer"},
        "laplace_hypersingular",
        device_interface,
        precision,
    )
