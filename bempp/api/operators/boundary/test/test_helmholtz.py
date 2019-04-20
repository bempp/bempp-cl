"""Unit tests for the dense assembler."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")

WAVENUMBER = 2.5
WAVENUMBER_COMPLEX = 2.5 + 1j

def test_helmholtz_single_layer_p0(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Helmholtz slp with p0 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import single_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "DP", 0)

    discrete_op = single_layer(
        space,
        space,
        space,
        WAVENUMBER,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    expected = helpers.load_npy_data("helmholtz_single_layer_boundary_p0_p0")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_helmholtz_single_layer_p1_disc(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Helmholtz slp with disc. p1 basis."""
    from bempp.api.operators.boundary.helmholtz import single_layer
    from bempp.api import function_space

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "DP", 1)

    discrete_op = single_layer(
        space,
        space,
        space,
        WAVENUMBER,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npy_data("helmholtz_single_layer_boundary_dp1_dp1")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_helmholtz_single_layer_p1_p0(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the slp with disc. p1/p0 basis."""
    from bempp.api.operators.boundary.helmholtz import single_layer
    from bempp.api import function_space

    grid = helpers.load_grid("sphere")

    space0 = function_space(grid, "DP", 0)
    space1 = function_space(grid, "DP", 1)

    discrete_op = single_layer(
        space0,
        space1,
        space1,
        WAVENUMBER,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npy_data("helmholtz_single_layer_boundary_dp1_p0")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_helmholtz_single_layer_p0_p1(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the slp with disc. p0/p1 basis."""
    from bempp.api.operators.boundary.helmholtz import single_layer
    from bempp.api import function_space

    grid = helpers.load_grid("sphere")

    space0 = function_space(grid, "DP", 0)
    space1 = function_space(grid, "DP", 1)

    discrete_op = single_layer(
        space1,
        space1,
        space0,
        WAVENUMBER,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npy_data("helmholtz_single_layer_boundary_p0_dp1")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_helmholtz_single_layer_p1_cont(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Helmholtz slp with p1 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import single_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = single_layer(
        space,
        space,
        space,
        WAVENUMBER,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npy_data("helmholtz_single_layer_boundary_p1_p1")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_helmholtz_double_layer_p1_cont(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Helmholtz dlp with p1 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import double_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = double_layer(
        space,
        space,
        space,
        WAVENUMBER,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npy_data("helmholtz_double_layer_boundary")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_helmholtz_adjoint_double_layer_p1_cont(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Helmholtz adjoint dlp with p1 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import adjoint_double_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = adjoint_double_layer(
        space,
        space,
        space,
        WAVENUMBER,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npy_data("helmholtz_adj_double_layer_boundary")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_helmholtz_hypersingular(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Helmholtz hypersingular operator."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import hypersingular

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = hypersingular(
        space,
        space,
        space,
        WAVENUMBER,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npy_data("helmholtz_hypersingular_boundary")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )

def test_helmholtz_hypersingular_complex_wavenumber(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Helmholtz hypersingular operator with complex wavenumber."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import hypersingular

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = hypersingular(
        space,
        space,
        space,
        WAVENUMBER_COMPLEX,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npy_data("helmholtz_complex_hypersingular_boundary")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )

def test_helmholtz_single_layer_complex_wavenumber(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Helmholtz single layer operator with complex wavenumber."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import single_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = single_layer(
        space,
        space,
        space,
        WAVENUMBER_COMPLEX,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npy_data("helmholtz_complex_single_layer_boundary")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_helmholtz_double_layer_complex_wavenumber(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Helmholtz double layer operator with complex wavenumber."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import double_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = double_layer(
        space,
        space,
        space,
        WAVENUMBER_COMPLEX,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npy_data("helmholtz_complex_double_layer_boundary")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )

def test_helmholtz_single_layer_evaluator_p0_p0(
    default_parameters, helpers, precision, device_interface
):
    """Test dense evaluator for the Helmholtz slp with p0 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import single_layer

    grid = helpers.load_grid("sphere")

    space1 = function_space(grid, "DP", 0)
    space2 = function_space(grid, "DP", 0)

    discrete_op = single_layer(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense_evaluator",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    mat = single_layer(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    x = _np.random.RandomState(0).randn(space1.global_dof_count)

    actual = discrete_op @ x
    expected = mat @ x

    if precision == "single":
        tol = 1e-4
    else:
        tol = 1e-12

    _np.testing.assert_allclose(actual, expected, rtol=tol)


def test_helmholtz_single_layer_evaluator_p0_dp1(
    default_parameters, helpers, precision, device_interface
):
    """Test dense evaluator for the Helmholtz slp with p0/dp1 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import single_layer

    grid = helpers.load_grid("sphere")

    space1 = function_space(grid, "DP", 1)
    space2 = function_space(grid, "DP", 0)

    discrete_op = single_layer(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense_evaluator",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    mat = single_layer(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    x = _np.random.RandomState(0).randn(space1.global_dof_count)

    actual = discrete_op @ x
    expected = mat @ x

    if precision == "single":
        tol = 1e-4
    else:
        tol = 1e-12

    _np.testing.assert_allclose(actual, expected, rtol=tol)


def test_helmholtz_single_layer_evaluator_p0_p1(
    default_parameters, helpers, precision, device_interface
):
    """Test dense evaluator for the Helmholtz slp with p0/p1 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import single_layer

    grid = helpers.load_grid("sphere")

    space1 = function_space(grid, "P", 1)
    space2 = function_space(grid, "DP", 0)

    discrete_op = single_layer(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense_evaluator",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    mat = single_layer(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    x = _np.random.RandomState(0).randn(space1.global_dof_count)

    actual = discrete_op @ x
    expected = mat @ x

    if precision == "single":
        tol = 1e-4
    else:
        tol = 1e-12

    _np.testing.assert_allclose(actual, expected, rtol=tol)


def test_helmholtz_single_layer_evaluator_complex(
    default_parameters, helpers, precision, device_interface
):
    """Test dense evaluator with complex vector."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import single_layer

    grid = helpers.load_grid("sphere")

    space1 = function_space(grid, "DP", 0)
    space2 = function_space(grid, "DP", 0)

    discrete_op = single_layer(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense_evaluator",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    mat = single_layer(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    x = _np.random.RandomState(0).randn(
        space1.global_dof_count
    ) + 1j * _np.random.RandomState(0).randn(space1.global_dof_count)

    actual = discrete_op @ x
    expected = mat @ x

    if precision == "single":
        tol = 1e-4
    else:
        tol = 1e-12

    _np.testing.assert_allclose(actual, expected, rtol=tol)


def test_helmholtz_double_layer_evaluator(
    default_parameters, helpers, precision, device_interface
):
    """Test dense evaluator for the Helmholtz dlp with p0 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import double_layer

    grid = helpers.load_grid("sphere")

    space1 = function_space(grid, "DP", 0)
    space2 = function_space(grid, "DP", 0)

    discrete_op = double_layer(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense_evaluator",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    mat = double_layer(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    x = _np.random.RandomState(0).randn(space1.global_dof_count)

    actual = discrete_op @ x
    expected = mat @ x


def test_helmholtz_adj_double_layer_evaluator(
    default_parameters, helpers, precision, device_interface
):
    """Test dense evaluator for the Helmholtz adj dlp with p0 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import adjoint_double_layer

    grid = helpers.load_grid("sphere")

    space1 = function_space(grid, "DP", 0)
    space2 = function_space(grid, "DP", 0)

    discrete_op = adjoint_double_layer(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense_evaluator",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    mat = adjoint_double_layer(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    x = _np.random.RandomState(0).randn(space1.global_dof_count)

    actual = discrete_op @ x
    expected = mat @ x

def test_helmholtz_hypersingular_evaluator(
    default_parameters, helpers, precision, device_interface
):
    """Test dense evaluator for the Helmholtz hypersingular with p1 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import hypersingular

    grid = helpers.load_grid("sphere")

    space1 = function_space(grid, "P", 1)
    space2 = function_space(grid, "P", 1)

    discrete_op = hypersingular(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense_evaluator",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    mat = hypersingular(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).assemble()

    x = _np.random.RandomState(0).randn(space1.global_dof_count)

    actual = discrete_op @ x
    expected = mat @ x

