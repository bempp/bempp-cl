"""Unit tests for the dense assembler."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")


def test_laplace_single_layer_p0(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Laplace slp with p0 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.laplace import single_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "DP", 0)

    discrete_op = single_layer(
        space,
        space,
        space,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).weak_form()

    expected = helpers.load_npy_data("laplace_single_layer_boundary_p0_p0")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_laplace_single_layer_p1_disc(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Laplace slp with disc. p1 basis."""
    from bempp.api.operators.boundary.laplace import single_layer
    from bempp.api import function_space

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "DP", 1)

    discrete_op = single_layer(
        space,
        space,
        space,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("laplace_single_layer_boundary_dp1_dp1")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_laplace_single_layer_p1_p0(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the slp with disc. p1/p0 basis."""
    from bempp.api.operators.boundary.laplace import single_layer
    from bempp.api import function_space

    grid = helpers.load_grid("sphere")

    space0 = function_space(grid, "DP", 0)
    space1 = function_space(grid, "DP", 1)

    discrete_op = single_layer(
        space0,
        space1,
        space1,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("laplace_single_layer_boundary_dp1_p0")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_laplace_single_layer_p0_p1(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the slp with disc. p0/p1 basis."""
    from bempp.api.operators.boundary.laplace import single_layer
    from bempp.api import function_space

    grid = helpers.load_grid("sphere")

    space0 = function_space(grid, "DP", 0)
    space1 = function_space(grid, "DP", 1)

    discrete_op = single_layer(
        space1,
        space1,
        space0,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("laplace_single_layer_boundary_p0_dp1")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_laplace_single_layer_p1_cont(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Laplace slp with p1 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.laplace import single_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = single_layer(
        space,
        space,
        space,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("laplace_single_layer_boundary_p1_p1")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_laplace_double_layer_p1_cont(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Laplace dlp with p1 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.laplace import double_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = double_layer(
        space,
        space,
        space,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("laplace_double_layer_boundary")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_laplace_adjoint_double_layer_p1_cont(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Laplace adjoint dlp with p1 basis."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.laplace import adjoint_double_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = adjoint_double_layer(
        space,
        space,
        space,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("laplace_adj_double_layer_boundary")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_laplace_hypersingular(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Laplace hypersingular operator."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.laplace import hypersingular

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = hypersingular(
        space,
        space,
        space,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("laplace_hypersingular_boundary")

    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


# def test_laplace_single_layer_evaluator_p0_p0(
    # default_parameters, helpers, precision, device_interface
# ):
    # """Test dense evaluator for the Laplace slp with p0 basis."""
    # from bempp.api import function_space
    # from bempp.api.operators.boundary.laplace import single_layer

    # grid = helpers.load_grid("sphere")

    # space1 = function_space(grid, "DP", 0)
    # space2 = function_space(grid, "DP", 0)

    # discrete_op = single_layer(
        # space1,
        # space1,
        # space2,
        # assembler="dense_evaluator",
        # parameters=default_parameters,
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # mat = single_layer(
        # space1,
        # space1,
        # space2,
        # assembler="dense",
        # parameters=default_parameters,
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # x = _np.random.RandomState(0).rand(space1.global_dof_count)

    # actual = discrete_op @ x
    # expected = mat @ x

    # if precision == "single":
        # tol = 2e-4
    # else:
        # tol = 1e-12

    # _np.testing.assert_allclose(actual, expected, rtol=tol)


# def test_laplace_single_layer_evaluator_p0_dp1(
    # default_parameters, helpers, precision, device_interface
# ):
    # """Test dense evaluator for the Laplace slp with p0/dp1 basis."""
    # from bempp.api import function_space
    # from bempp.api.operators.boundary.laplace import single_layer

    # grid = helpers.load_grid("sphere")

    # space1 = function_space(grid, "DP", 1)
    # space2 = function_space(grid, "DP", 0)

    # discrete_op = single_layer(
        # space1,
        # space1,
        # space2,
        # assembler="dense_evaluator",
        # parameters=default_parameters,
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # mat = single_layer(
        # space1,
        # space1,
        # space2,
        # assembler="dense",
        # parameters=default_parameters,
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # x = _np.random.RandomState(0).rand(space1.global_dof_count)

    # actual = discrete_op @ x
    # expected = mat @ x

    # if precision == "single":
        # tol = 2e-4
    # else:
        # tol = 1e-12

    # _np.testing.assert_allclose(actual, expected, rtol=tol)


# def test_laplace_single_layer_evaluator_p0_p1(
    # default_parameters, helpers, precision, device_interface
# ):
    # """Test dense evaluator for the Laplace slp with p0/p1 basis."""
    # from bempp.api import function_space
    # from bempp.api.operators.boundary.laplace import single_layer

    # grid = helpers.load_grid("sphere")

    # space1 = function_space(grid, "P", 1)
    # space2 = function_space(grid, "DP", 0)

    # discrete_op = single_layer(
        # space1,
        # space1,
        # space2,
        # assembler="dense_evaluator",
        # parameters=default_parameters,
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # mat = single_layer(
        # space1,
        # space1,
        # space2,
        # assembler="dense",
        # parameters=default_parameters,
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # x = _np.random.RandomState(0).rand(space1.global_dof_count)

    # actual = discrete_op @ x
    # expected = mat @ x

    # if precision == "single":
        # tol = 1e-4
    # else:
        # tol = 1e-12

    # _np.testing.assert_allclose(actual, expected, rtol=tol)


# def test_laplace_single_layer_evaluator_complex(
    # default_parameters, helpers, precision, device_interface
# ):
    # """Test dense evaluator with complex vector."""
    # from bempp.api import function_space
    # from bempp.api.operators.boundary.laplace import single_layer

    # grid = helpers.load_grid("sphere")

    # space1 = function_space(grid, "DP", 0)
    # space2 = function_space(grid, "DP", 0)

    # discrete_op = single_layer(
        # space1,
        # space1,
        # space2,
        # assembler="dense_evaluator",
        # parameters=default_parameters,
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # mat = single_layer(
        # space1,
        # space1,
        # space2,
        # assembler="dense",
        # parameters=default_parameters,
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # x = _np.random.RandomState(0).rand(
        # space1.global_dof_count
    # ) + 1j * _np.random.RandomState(0).rand(space1.global_dof_count)

    # actual = discrete_op @ x
    # expected = mat @ x

    # if precision == "single":
        # tol = 2e-4
    # else:
        # tol = 1e-12

    # _np.testing.assert_allclose(actual, expected, rtol=tol)


# def test_laplace_double_layer_evaluator(
    # default_parameters, helpers, precision, device_interface
# ):
    # """Test dense evaluator for the Laplace dlp with p0 basis."""
    # from bempp.api import function_space
    # from bempp.api.operators.boundary.laplace import double_layer

    # grid = helpers.load_grid("sphere")

    # space1 = function_space(grid, "DP", 0)
    # space2 = function_space(grid, "DP", 0)

    # discrete_op = double_layer(
        # space1,
        # space1,
        # space2,
        # assembler="dense_evaluator",
        # parameters=default_parameters,
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # mat = double_layer(
        # space1,
        # space1,
        # space2,
        # assembler="dense",
        # parameters=default_parameters,
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # x = _np.random.RandomState(0).rand(space1.global_dof_count)

    # actual = discrete_op @ x
    # expected = mat @ x


# def test_laplace_adjoint_double_layer_evaluator(
    # default_parameters, helpers, precision, device_interface
# ):
    # """Test dense evaluator for the Laplace adjoint dlp with p0 basis."""
    # from bempp.api import function_space
    # from bempp.api.operators.boundary.laplace import adjoint_double_layer

    # grid = helpers.load_grid("sphere")

    # space1 = function_space(grid, "DP", 0)
    # space2 = function_space(grid, "DP", 0)

    # discrete_op = adjoint_double_layer(
        # space1,
        # space1,
        # space2,
        # assembler="dense_evaluator",
        # parameters=default_parameters,
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # mat = adjoint_double_layer(
        # space1,
        # space1,
        # space2,
        # assembler="dense",
        # parameters=default_parameters,
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # x = _np.random.RandomState(0).rand(space1.global_dof_count)

    # actual = discrete_op @ x
    # expected = mat @ x


# def test_laplace_hypersingular_evaluator(
    # default_parameters, helpers, precision, device_interface
# ):
    # """Test dense evaluator for the Laplace hypersingular with p1 basis."""
    # from bempp.api import function_space
    # from bempp.api.operators.boundary.laplace import hypersingular

    # grid = helpers.load_grid("sphere")

    # space1 = function_space(grid, "P", 1)
    # space2 = function_space(grid, "P", 1)

    # discrete_op = hypersingular(
        # space1,
        # space1,
        # space2,
        # assembler="dense_evaluator",
        # parameters=default_parameters,
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # mat = hypersingular(
        # space1,
        # space1,
        # space2,
        # assembler="dense",
        # parameters=default_parameters,
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # x = _np.random.RandomState(0).randn(space1.global_dof_count)

    # actual = discrete_op @ x
    # expected = mat @ x
