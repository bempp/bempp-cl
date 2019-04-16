"""Unit tests for the dense assembler."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers", "small_sphere")


def test_laplace_single_layer_p0(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Laplace slp with p0 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.laplace import single_layer

    grid = helpers.load_grid("sphere_h_01")

    space = function_space(grid, "DP", 0)

    default_parameters.quadrature.singular = 8
    default_parameters.quadrature.regular = 8

    discrete_op = single_layer(
        space,
        space,
        space,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    expected = helpers.load_npy_data("laplace_single_layer_sphere_h_01")
    _np.testing.assert_allclose(discrete_op.A, expected, rtol=1e-5)


def test_laplace_single_layer_p1_disc(
    default_parameters, small_sphere, helpers, precision, device_interface
):
    """Test dense assembler for the Laplace slp with disc. p1 basis."""
    from bempp.api.operators.boundary.laplace import single_layer
    from bempp.api import function_space

    space = function_space(small_sphere, "DP", 1)

    discrete_op = single_layer(
        space,
        space,
        space,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npy_data("laplace_small_sphere_p1_disc")
    _np.testing.assert_allclose(discrete_op.A, expected, rtol=1e-3)


def test_laplace_single_layer_p1_p0(
    default_parameters, small_sphere, helpers, precision, device_interface
):
    """Test dense assembler for the slp with disc. p1/p0 basis."""
    from bempp.api.operators.boundary.laplace import single_layer
    from bempp.api import function_space

    space0 = function_space(small_sphere, "DP", 0)
    space1 = function_space(small_sphere, "DP", 1)

    discrete_op = single_layer(
        space0,
        space1,
        space1,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npy_data("laplace_small_sphere_p1_p0_disc")
    _np.testing.assert_allclose(discrete_op.A, expected, rtol=1e-3)


def test_laplace_single_layer_p0_p1(
    default_parameters, small_sphere, helpers, precision, device_interface
):
    """Test dense assembler for the slp with disc. p0/p1 basis."""
    from bempp.api.operators.boundary.laplace import single_layer
    from bempp.api import function_space

    space0 = function_space(small_sphere, "DP", 0)
    space1 = function_space(small_sphere, "DP", 1)

    discrete_op = single_layer(
        space1,
        space1,
        space0,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npy_data("laplace_small_sphere_p0_p1_disc")
    _np.testing.assert_allclose(discrete_op.A, expected, rtol=1e-3)


def test_laplace_single_layer_p1_cont(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Laplace slp with p1 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.laplace import single_layer

    grid = helpers.load_grid("sphere_h_01")

    space = function_space(grid, "P", 1)

    default_parameters.quadrature.singular = 8
    default_parameters.quadrature.regular = 8

    discrete_op = single_layer(
        space,
        space,
        space,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npy_data("laplace_single_layer_sphere_h_01_p1_cont")
    _np.testing.assert_allclose(discrete_op.A, expected, rtol=1e-5)


def test_laplace_double_layer_p1_cont(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Laplace dlp with p1 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.laplace import double_layer

    grid = helpers.load_grid("sphere_h_01")

    space = function_space(grid, "P", 1)

    default_parameters.quadrature.singular = 8
    default_parameters.quadrature.regular = 8

    discrete_op = double_layer(
        space,
        space,
        space,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npz_data("laplace_double_layer_sphere_h_01_p1_cont")[
        "arr_0"
    ]
    _np.testing.assert_allclose(discrete_op.A, expected, rtol=1e-4)


def test_laplace_adjoint_double_layer_p1_cont(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Laplace adjoint dlp with p1 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.laplace import adjoint_double_layer

    grid = helpers.load_grid("sphere_h_01")

    space = function_space(grid, "P", 1)

    default_parameters.quadrature.singular = 8
    default_parameters.quadrature.regular = 8

    discrete_op = adjoint_double_layer(
        space,
        space,
        space,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npz_data(
        "laplace_adjoint_double_layer_sphere_h_01_p1_cont"
    )["arr_0"]
    _np.testing.assert_allclose(discrete_op.A, expected, rtol=1e-4)


def test_laplace_hypersingular(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Laplace hypersingular operator."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.laplace import hypersingular

    grid = helpers.load_grid("sphere_h_01")

    space = function_space(grid, "P", 1)

    default_parameters.quadrature.singular = 10
    default_parameters.quadrature.regular = 10

    discrete_op = hypersingular(
        space,
        space,
        space,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npz_data("laplace_hypersingular")["arr_0"]

    if precision == "single":
        rtol = 1e-2
    if precision == "double":
        rtol = 1e-5

    _np.testing.assert_allclose(discrete_op.A, expected, rtol=rtol)

def test_laplace_single_layer_evaluator(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Laplace slp with p0 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.laplace import single_layer

    grid = helpers.load_grid("sphere_h_01")

    space = function_space(grid, "DP", 0)

    default_parameters.quadrature.singular = 4
    default_parameters.quadrature.regular = 4

    discrete_op = single_layer(
        space,
        space,
        space,
        assembler="dense_evaluator",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

