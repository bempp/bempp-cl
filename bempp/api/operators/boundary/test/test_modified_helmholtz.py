"""Unit tests for modified Helmholtz operators."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")

OMEGA = 2.5

def test_modified_helmholtz_single_layer(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for modified Helmholtz."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.modified_helmholtz import single_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = single_layer(
        space,
        space,
        space,
        OMEGA,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    # Bempp 3 assembles modified Helmholtz as complex type, so cast to real.
    expected = _np.real(helpers.load_npy_data("modified_helmholtz_single_layer_boundary"))
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_modified_helmholtz_double_layer_p1_cont(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the modified Helmholtz dlp."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.modified_helmholtz import double_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = double_layer(
        space,
        space,
        space,
        OMEGA,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = _np.real(helpers.load_npy_data("modified_helmholtz_double_layer_boundary"))
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_modified_helmholtz_adjoint_double_layer(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Helmholtz adjoint dlp."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.modified_helmholtz import adjoint_double_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = adjoint_double_layer(
        space,
        space,
        space,
        OMEGA,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = _np.real(helpers.load_npy_data("modified_helmholtz_adj_double_layer_boundary"))
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_modified_helmholtz_hypersingular(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the modified Helmholtz hypersingular operator."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.modified_helmholtz import hypersingular

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = hypersingular(
        space,
        space,
        space,
        OMEGA,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = _np.real(helpers.load_npy_data("modified_helmholtz_hypersingular_boundary"))
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_modified_helmholtz_single_layer_evaluator(
    default_parameters, helpers, precision, device_interface
):
    """Test dense evaluator for modified Helmholtz slp."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.modified_helmholtz import single_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = single_layer(
        space,
        space,
        space,
        OMEGA,
        assembler="dense_evaluator",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    mat = single_layer(
        space,
        space,
        space,
        OMEGA,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    x = _np.random.RandomState(0).randn(space.global_dof_count)

    actual = discrete_op @ x
    expected = mat @ x

    if precision == "single":
        tol = 1e-4
    else:
        tol = 1e-12

    _np.testing.assert_allclose(actual, expected, rtol=tol)

def test_modified_helmholtz_double_layer_evaluator(
    default_parameters, helpers, precision, device_interface
):
    """Test dense evaluator for modified Helmholtz dlp."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.modified_helmholtz import double_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = double_layer(
        space,
        space,
        space,
        OMEGA,
        assembler="dense_evaluator",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    mat = double_layer(
        space,
        space,
        space,
        OMEGA,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    x = _np.random.RandomState(0).randn(space.global_dof_count)

    actual = discrete_op @ x
    expected = mat @ x

    if precision == "single":
        tol = 1e-4
    else:
        tol = 1e-12

    _np.testing.assert_allclose(actual, expected, rtol=tol)


def test_modified_helmholtz_adj_double_layer_evaluator(
    default_parameters, helpers, precision, device_interface
):
    """Test dense evaluator for modified Helmholtz adj dlp."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.modified_helmholtz import adjoint_double_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = adjoint_double_layer(
        space,
        space,
        space,
        OMEGA,
        assembler="dense_evaluator",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    mat = adjoint_double_layer(
        space,
        space,
        space,
        OMEGA,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    x = _np.random.RandomState(0).randn(space.global_dof_count)

    actual = discrete_op @ x
    expected = mat @ x

    if precision == "single":
        tol = 1e-4
    else:
        tol = 1e-12

    _np.testing.assert_allclose(actual, expected, rtol=tol)

def test_modified_helmholtz_hypersingular_evaluator(
    default_parameters, helpers, precision, device_interface
):
    """Test dense evaluator for modified Helmholtz hypersingular."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.modified_helmholtz import hypersingular

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = hypersingular(
        space,
        space,
        space,
        OMEGA,
        assembler="dense_evaluator",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    mat = hypersingular(
        space,
        space,
        space,
        OMEGA,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    x = _np.random.RandomState(0).randn(space.global_dof_count)

    actual = discrete_op @ x
    expected = mat @ x

    if precision == "single":
        tol = 1e-4
    else:
        tol = 1e-12

    _np.testing.assert_allclose(actual, expected, rtol=tol)
