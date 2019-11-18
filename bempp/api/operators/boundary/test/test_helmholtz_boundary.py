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
    ).weak_form()

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
    ).weak_form()

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
    ).weak_form()

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
    ).weak_form()

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
    ).weak_form()

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
    ).weak_form()

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
    ).weak_form()

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
    ).weak_form()

    expected = helpers.load_npy_data("helmholtz_hypersingular_boundary")
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
    ).weak_form()

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
    ).weak_form()

    expected = helpers.load_npy_data("helmholtz_complex_double_layer_boundary")
    _np.testing.assert_allclose(
        discrete_op.A, expected, rtol=helpers.default_tolerance(precision)
    )


def test_helmholtz_adjoint_double_layer_complex_wavenumber(
    default_parameters, helpers, precision, device_interface
):
    """Test dense assembler for the Helmholtz adj double layer operator with complex wavenumber."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import adjoint_double_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = adjoint_double_layer(
        space,
        space,
        space,
        WAVENUMBER_COMPLEX,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("helmholtz_complex_adj_double_layer_boundary")
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
    ).weak_form()

    expected = helpers.load_npy_data("helmholtz_complex_hypersingular_boundary")
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
    ).weak_form()

    mat = single_layer(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).weak_form()

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
    ).weak_form()

    mat = single_layer(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).weak_form()

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
    ).weak_form()

    mat = single_layer(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).weak_form()

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
    ).weak_form()

    mat = single_layer(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).weak_form()

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
    ).weak_form()

    mat = double_layer(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).weak_form()

    x = _np.random.RandomState(0).randn(space1.global_dof_count)

    actual = discrete_op @ x
    expected = mat @ x

    if precision == "single":
        tol = 1e-4
    else:
        tol = 1e-12

    _np.testing.assert_allclose(actual, expected, rtol=tol)


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
    ).weak_form()

    mat = adjoint_double_layer(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).weak_form()

    x = _np.random.RandomState(0).randn(space1.global_dof_count)

    actual = discrete_op @ x
    expected = mat @ x

    if precision == "single":
        tol = 1e-4
    else:
        tol = 1e-12

    _np.testing.assert_allclose(actual, expected, rtol=tol)


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
    ).weak_form()

    mat = hypersingular(
        space1,
        space1,
        space2,
        WAVENUMBER,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).weak_form()

    x = _np.random.RandomState(0).randn(space1.global_dof_count)

    actual = discrete_op @ x
    expected = mat @ x

    if precision == "single":
        tol = 1e-4
    else:
        tol = 1e-12

    _np.testing.assert_allclose(actual, expected, rtol=tol)


def test_helmholtz_hypersingular_complex_evaluator(
    default_parameters, helpers, precision, device_interface
):
    """Test dense evaluator for the Helmholtz hypersingular op with complex wavenumber."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.helmholtz import hypersingular

    grid = helpers.load_grid("sphere")

    space1 = function_space(grid, "P", 1)
    space2 = function_space(grid, "P", 1)

    discrete_op = hypersingular(
        space1,
        space1,
        space2,
        WAVENUMBER_COMPLEX,
        assembler="dense_evaluator",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).weak_form()

    mat = hypersingular(
        space1,
        space1,
        space2,
        WAVENUMBER_COMPLEX,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).weak_form()

    x = _np.random.RandomState(0).randn(space1.global_dof_count)

    actual = discrete_op @ x
    expected = mat @ x

    if precision == "single":
        tol = 1e-4
    else:
        tol = 1e-12

    _np.testing.assert_allclose(actual, expected, rtol=tol)


# def test_helmholtz_multitrace_sphere(
    # default_parameters, helpers, device_interface, precision
# ):
    # """Test Maxwell magnetic field on sphere."""
    # import bempp.api
    # from bempp.api import get_precision
    # from bempp.api import function_space
    # from bempp.api.shapes import regular_sphere
    # from bempp.api.operators.boundary.helmholtz import (
        # single_layer,
        # double_layer,
        # adjoint_double_layer,
        # hypersingular,
    # )
    # from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator

    # # if precision == 'single':
    # #    pytest.skip("Test runs only in double precision mode.")

    # grid = helpers.load_grid("sphere")

    # op = bempp.api.operators.boundary.helmholtz.multitrace_operator(
        # grid,
        # 2.5,
        # parameters=default_parameters,
        # device_interface=device_interface,
        # precision=precision,
    # )

    # slp = single_layer(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # 2.5,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # dlp = double_layer(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # 2.5,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # adlp = adjoint_double_layer(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # 2.5,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # hyp = hypersingular(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # 2.5,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # expected = BlockedDiscreteOperator(_np.array([[-dlp, slp], [hyp, adlp]]))

    # rand = _np.random.RandomState(0)
    # x = rand.randn(expected.shape[1])

    # y_expected = expected @ x
    # y_actual = op.weak_form() @ x

    # _np.testing.assert_allclose(
        # y_actual, y_expected, rtol=helpers.default_tolerance(precision)
    # )


# def test_helmholtz_transmission_sphere(
    # default_parameters, helpers, device_interface, precision
# ):
    # """Test Helmholtz transmission on sphere."""
    # import bempp.api
    # from bempp.api import get_precision
    # from bempp.api import function_space
    # from bempp.api.shapes import regular_sphere
    # from bempp.api.operators.boundary.helmholtz import (
        # single_layer,
        # double_layer,
        # adjoint_double_layer,
        # hypersingular,
    # )
    # from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator

    # # if precision == 'single':
    # #    pytest.skip("Test runs only in double precision mode.")

    # grid = helpers.load_grid("sphere")

    # wavenumber = 2.5
    # refractive_index = 1.5
    # rho_rel = 1.3

    # wavenumber_int = wavenumber * refractive_index

    # op = bempp.api.operators.boundary.helmholtz.transmission_operator(
        # grid,
        # wavenumber,
        # rho_rel,
        # refractive_index,
        # parameters=default_parameters,
        # device_interface=device_interface,
        # precision=precision,
    # )

    # slp = single_layer(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # wavenumber,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # slp_int = single_layer(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # wavenumber_int,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # dlp = double_layer(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # wavenumber,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # dlp_int = double_layer(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # wavenumber_int,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # adlp = adjoint_double_layer(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # wavenumber,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # adlp_int = adjoint_double_layer(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # wavenumber_int,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # hyp = hypersingular(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # wavenumber,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # hyp_int = hypersingular(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # wavenumber_int,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # expected = BlockedDiscreteOperator(
        # _np.array(
            # [
                # [-dlp - dlp_int, slp + rho_rel * slp_int],
                # [hyp + 1.0 / rho_rel * hyp_int, adlp + adlp_int],
            # ]
        # )
    # )

    # rand = _np.random.RandomState(0)
    # x = rand.randn(expected.shape[1])

    # y_expected = expected @ x
    # y_actual = op.weak_form() @ x

    # _np.testing.assert_allclose(
        # y_actual, y_expected, rtol=helpers.default_tolerance(precision)
    # )


# def test_helmholtz_transmission_complex_sphere(
    # default_parameters, helpers, device_interface, precision
# ):
    # """Test Helmholtz transmission on sphere with cmplex wavenumber."""
    # import bempp.api
    # from bempp.api import get_precision
    # from bempp.api import function_space
    # from bempp.api.shapes import regular_sphere
    # from bempp.api.operators.boundary.helmholtz import (
        # single_layer,
        # double_layer,
        # adjoint_double_layer,
        # hypersingular,
    # )
    # from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator

    # grid = helpers.load_grid("sphere")

    # wavenumber = 2.5 + 1j
    # refractive_index = 1.5 + 0.7j
    # rho_rel = 1.3

    # wavenumber_int = wavenumber * refractive_index

    # op = bempp.api.operators.boundary.helmholtz.transmission_operator(
        # grid,
        # wavenumber,
        # rho_rel,
        # refractive_index,
        # parameters=default_parameters,
        # device_interface=device_interface,
        # precision=precision,
    # )

    # slp = single_layer(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # wavenumber,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # slp_int = single_layer(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # wavenumber_int,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # dlp = double_layer(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # wavenumber,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # dlp_int = double_layer(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # wavenumber_int,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # adlp = adjoint_double_layer(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # wavenumber,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # adlp_int = adjoint_double_layer(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # wavenumber_int,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # hyp = hypersingular(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # wavenumber,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # hyp_int = hypersingular(
        # op.domain_spaces[0],
        # op.range_spaces[0],
        # op.dual_to_range_spaces[0],
        # wavenumber_int,
        # parameters=default_parameters,
        # assembler="dense",
        # device_interface=device_interface,
        # precision=precision,
    # ).weak_form()

    # expected = BlockedDiscreteOperator(
        # _np.array(
            # [
                # [-dlp - dlp_int, slp + rho_rel * slp_int],
                # [hyp + 1.0 / rho_rel * hyp_int, adlp + adlp_int],
            # ]
        # )
    # )

    # rand = _np.random.RandomState(0)
    # x = rand.randn(expected.shape[1])

    # y_expected = expected @ x
    # y_actual = op.weak_form() @ x

    # _np.testing.assert_allclose(
        # y_actual, y_expected, rtol=helpers.default_tolerance(precision)
    # )


# def test_helmholtz_multitrace_subgrid(
    # default_parameters, helpers, device_interface, precision
# ):
    # """Test multitrace operator on subgrids of a skeleton with junctions."""
    # import bempp.api
    # from bempp.api import get_precision
    # from bempp.api import function_space
    # from bempp.api.shapes import multitrace_cube
    # from bempp.api.grid.grid import grid_from_segments

    # grid = bempp.api.shapes.multitrace_cube()

    # seglists = [[1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11]]
    # swapped_normal_lists = [{}, {6}]

    # rand = _np.random.RandomState(0)

    # for seglist, swapped_normals in zip(seglists, swapped_normal_lists):
        # new_grid = grid_from_segments(grid, seglist)

        # mult1 = bempp.api.operators.boundary.helmholtz.multitrace_operator(
            # grid, WAVENUMBER, segments=seglist, swapped_normals=swapped_normals
        # )
        # mult2 = bempp.api.operators.boundary.helmholtz.multitrace_operator(
            # new_grid, WAVENUMBER, swapped_normals=swapped_normals
        # )

        # coeffs1 = rand.rand(mult1.domain_spaces[0].global_dof_count)
        # coeffs2 = rand.rand(mult1.domain_spaces[0].global_dof_count)

        # fun1 = [
            # bempp.api.GridFunction(mult1.domain_spaces[0], coefficients=coeffs1),
            # bempp.api.GridFunction(mult1.domain_spaces[1], coefficients=coeffs2),
        # ]
        # fun2 = [
            # bempp.api.GridFunction(mult2.domain_spaces[0], coefficients=coeffs1),
            # bempp.api.GridFunction(mult2.domain_spaces[1], coefficients=coeffs2),
        # ]

        # ident1 = bempp.api.operators.boundary.sparse.multitrace_identity(mult1)
        # calderon1 = 0.5 * ident1 + mult1
        # ident2 = bempp.api.operators.boundary.sparse.multitrace_identity(mult2)
        # calderon2 = 0.5 * ident2 + mult2

        # res11 = calderon1 * fun1
        # res12 = calderon1 * res11
        # res21 = calderon2 * fun2
        # res22 = calderon2 * res21

        # rel_diff1_dirichlet = (res11[0] - res12[0]).l2_norm() / res11[0].l2_norm()
        # rel_diff1_neumann = (res11[1] - res12[1]).l2_norm() / res11[1].l2_norm()
        # rel_diff2_dirichlet = (res21[0] - res22[0]).l2_norm() / res22[0].l2_norm()
        # rel_diff2_neumann = (res21[1] - res22[1]).l2_norm() / res22[1].l2_norm()

        # _np.testing.assert_equal(res22[0].coefficients, res12[0].coefficients)
        # _np.testing.assert_equal(res22[1].coefficients, res12[1].coefficients)
        # assert rel_diff1_dirichlet < 5e-3
        # assert rel_diff1_neumann < 5e-3


# def test_helmholtz_operators_subgrid_evaluator_p1(
    # default_parameters, helpers, device_interface, precision
# ):
    # """Test standard evaluators on subgrids of a skeleton with junctions."""
    # import bempp.api
    # from bempp.api import get_precision
    # from bempp.api import function_space
    # from bempp.api.shapes import multitrace_cube
    # from bempp.api.grid.grid import grid_from_segments

    # grid = bempp.api.shapes.multitrace_cube()

    # seglists = [[1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11]]
    # swapped_normal_lists = [{}, {6}]

    # rand = _np.random.RandomState(0)

    # operators = [
        # bempp.api.operators.boundary.helmholtz.single_layer,
        # bempp.api.operators.boundary.helmholtz.double_layer,
        # bempp.api.operators.boundary.helmholtz.adjoint_double_layer,
        # bempp.api.operators.boundary.helmholtz.hypersingular,
    # ]

    # for seglist, swapped_normals in zip(seglists, swapped_normal_lists):
        # new_grid = grid_from_segments(grid, seglist)
        # space1 = bempp.api.function_space(
            # grid,
            # "P",
            # 1,
            # segments=seglist,
            # swapped_normals=swapped_normals,
            # include_boundary_dofs=True,
        # )
        # space2 = bempp.api.function_space(
            # new_grid, "P", 1, swapped_normals=swapped_normals
        # )
        # coeffs = rand.rand(space1.global_dof_count)

        # for op in operators:
            # res_actual = op(space1, space1, space1, WAVENUMBER).weak_form() @ coeffs
            # res_expected = op(space2, space2, space2, WAVENUMBER).weak_form() @ coeffs
            # _np.testing.assert_equal(res_actual, res_expected)


# def test_helmholtz_operators_subgrid_evaluator_dp1(
    # default_parameters, helpers, device_interface, precision
# ):
    # """Test standard evaluators on subgrids of a skeleton with junctions."""
    # import bempp.api
    # from bempp.api import get_precision
    # from bempp.api import function_space
    # from bempp.api.shapes import multitrace_cube
    # from bempp.api.grid.grid import grid_from_segments

    # grid = bempp.api.shapes.multitrace_cube()

    # seglists = [[1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11]]
    # swapped_normal_lists = [{}, {6}]

    # rand = _np.random.RandomState(0)

    # operators = [
        # bempp.api.operators.boundary.helmholtz.single_layer,
        # bempp.api.operators.boundary.helmholtz.double_layer,
        # bempp.api.operators.boundary.helmholtz.adjoint_double_layer,
    # ]

    # for seglist, swapped_normals in zip(seglists, swapped_normal_lists):
        # new_grid = grid_from_segments(grid, seglist)
        # space1 = bempp.api.function_space(
            # grid,
            # "DP",
            # 1,
            # segments=seglist,
            # swapped_normals=swapped_normals,
            # include_boundary_dofs=True,
        # )
        # space2 = bempp.api.function_space(
            # new_grid, "DP", 1, swapped_normals=swapped_normals
        # )
        # coeffs = rand.rand(space1.global_dof_count)

        # for op in operators:
            # res_actual = (
                # op(
                    # space1, space1, space1, WAVENUMBER, assembler="dense_evaluator"
                # ).weak_form()
                # @ coeffs
            # )
            # res_expected = (
                # op(
                    # space2, space2, space2, WAVENUMBER, assembler="dense_evaluator"
                # ).weak_form()
                # @ coeffs
            # )
            # _np.testing.assert_equal(res_actual, res_expected)


# def test_helmholtz_operators_subgrid_p1(
    # default_parameters, helpers, device_interface, precision
# ):
    # """Test standard operators on subgrids of a skeleton with junctions."""
    # import bempp.api
    # from bempp.api import get_precision
    # from bempp.api import function_space
    # from bempp.api.shapes import multitrace_cube
    # from bempp.api.grid.grid import grid_from_segments

    # grid = bempp.api.shapes.multitrace_cube()

    # seglists = [[1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11]]
    # swapped_normal_lists = [{}, {6}]

    # rand = _np.random.RandomState(0)

    # operators = [
        # bempp.api.operators.boundary.helmholtz.single_layer,
        # bempp.api.operators.boundary.helmholtz.double_layer,
        # bempp.api.operators.boundary.helmholtz.adjoint_double_layer,
        # bempp.api.operators.boundary.helmholtz.hypersingular,
    # ]

    # for seglist, swapped_normals in zip(seglists, swapped_normal_lists):
        # new_grid = grid_from_segments(grid, seglist)
        # space1 = bempp.api.function_space(
            # grid,
            # "P",
            # 1,
            # segments=seglist,
            # swapped_normals=swapped_normals,
            # include_boundary_dofs=True,
        # )
        # space2 = bempp.api.function_space(
            # new_grid, "P", 1, swapped_normals=swapped_normals
        # )

        # for op in operators:
            # res_actual = op(space1, space1, space1, WAVENUMBER).weak_form().A
            # res_expected = op(space2, space2, space2, WAVENUMBER).weak_form().A
            # _np.testing.assert_equal(res_actual, res_expected)
