"""Unit tests for modified Helmholtz operators."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers", "small_sphere")


def test_helmholtz_slp_p1(default_parameters, helpers, precision, device_interface):
    """Test Helmholtz slp with p1 basis functions."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import single_layer

    grid = helpers.load_grid("sphere_h_01")

    default_parameters.quadrature.singular = 8
    default_parameters.quadrature.regular = 8

    space = function_space(grid, "P", 1)

    discrete_op = single_layer(
        space,
        space,
        space,
        2.5,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npz_data(
        "helmholtz_single_layer_sphere_h_01_w_2_5_p1_cont"
    )["arr_0"]
    _np.testing.assert_allclose(discrete_op.A, expected, rtol=1e-5)


def test_helmholtz_dlp_p1(default_parameters, helpers, precision, device_interface):
    """Test Helmholtz dlp with p1 basis functions."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import double_layer

    grid = helpers.load_grid("sphere_h_01")

    default_parameters.quadrature.singular = 8
    default_parameters.quadrature.regular = 8

    space = function_space(grid, "P", 1)

    discrete_op = double_layer(
        space,
        space,
        space,
        2.5,
        assembler="dense",
        device_interface=device_interface,
        precision=precision,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npz_data("helmholtz_double_layer_w_2_5")["arr_0"]
    _np.testing.assert_allclose(discrete_op.A, expected, rtol=5e-5)


def test_helmholtz_adjoint_dlp_p1(
    default_parameters, helpers, device_interface, precision
):
    """Test Helmholtz adj. dlp with p1 basis functions."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import adjoint_double_layer

    grid = helpers.load_grid("sphere_h_01")

    default_parameters.quadrature.singular = 8
    default_parameters.quadrature.regular = 8

    space = function_space(grid, "P", 1)

    discrete_op = adjoint_double_layer(
        space,
        space,
        space,
        2.5,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    expected = helpers.load_npz_data("helmholtz_adjoint_double_layer_w_2_5")["arr_0"]
    _np.testing.assert_allclose(discrete_op.A, expected, rtol=5e-5)


def test_helmholtz_hypersingular(
    default_parameters, helpers, device_interface, precision
):
    """Test Helmholtz hypersingular with p1 basis functions."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import hypersingular

    grid = helpers.load_grid("sphere_h_01")

    default_parameters.quadrature.singular = 8
    default_parameters.quadrature.regular = 8

    space = function_space(grid, "P", 1)

    if precision == "double":
        rtol = 5e-5
    elif precision == "single":
        rtol = 5e-3
    else:
        raise ValueError("precision must be 'single' or 'double'")

    discrete_op = hypersingular(
        space,
        space,
        space,
        2.5,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    expected = helpers.load_npz_data("helmholtz_hypersingular")["arr_0"]
    _np.testing.assert_allclose(discrete_op.A, expected, rtol=rtol)
