"""Unit tests for Laplace potential assemblers."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers", "small_sphere")


def test_laplace_single_layer_p0(
    default_parameters, helpers, device_interface, precision
):
    """Test Laplace slp potential with p0 basis."""
    from bempp.api import function_space
    from bempp.api import GridFunction
    from bempp.api.operators.potential.laplace import single_layer

    default_parameters.quadrature.regular = 8

    grid = helpers.load_grid("sphere_h_01")
    space = function_space(grid, "DP", 0)

    npoints = 100
    theta = _np.linspace(0, 2 * _np.pi, npoints)
    points = _np.vstack(
        [_np.cos(theta), _np.sin(theta), 3 * _np.ones(npoints, dtype="float64")]
    )

    data = helpers.load_npz_data("laplace_single_layer_pot_p0")

    coefficients = data["arr_0"]
    expected = data["arr_1"]

    fun = GridFunction(space, coefficients=coefficients)

    actual = single_layer(
        space,
        points,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    ).evaluate(fun)

    _np.testing.assert_allclose(actual, expected, rtol=5e-5)


def test_laplace_single_layer_p1(
    default_parameters, helpers, device_interface, precision
):
    """Test Laplace slp potential with p1 basis."""
    from bempp.api import function_space
    from bempp.api.operators.potential.laplace import single_layer
    from bempp.api import GridFunction

    default_parameters.quadrature.regular = 8

    grid = helpers.load_grid("sphere_h_01")
    space = function_space(grid, "P", 1)

    npoints = 100
    theta = _np.linspace(0, 2 * _np.pi, npoints)
    points = _np.vstack(
        [_np.cos(theta), _np.sin(theta), 3 * _np.ones(npoints, dtype="float64")]
    )

    data = helpers.load_npz_data("laplace_single_layer_pot_p1")

    coefficients = data["arr_0"]
    expected = data["arr_1"]

    fun = GridFunction(space, coefficients=coefficients)

    actual = single_layer(
        space,
        points,
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).evaluate(fun)

    _np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_laplace_single_layer_p1_complex(
    default_parameters, helpers, device_interface, precision
):
    """Test Laplace slp potential with p1 basis and complex coeffs."""
    from bempp.api import function_space
    from bempp.api.operators.potential.laplace import single_layer
    from bempp.api import GridFunction

    default_parameters.quadrature.regular = 8

    grid = helpers.load_grid("sphere_h_01")
    space = function_space(grid, "P", 1)

    npoints = 100
    theta = _np.linspace(0, 2 * _np.pi, npoints)
    points = _np.vstack(
        [_np.cos(theta), _np.sin(theta), 3 * _np.ones(npoints, dtype="float64")]
    )

    data = helpers.load_npz_data("laplace_single_layer_pot_p1_complex")

    coefficients = data["arr_0"]
    expected = data["arr_1"]

    fun = GridFunction(space, coefficients=coefficients)

    actual = single_layer(
        space,
        points,
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).evaluate(fun)

    _np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_laplace_double_layer_p1_complex(
    default_parameters, helpers, device_interface, precision
):
    """Test Laplace dlp potential with p1 basis and complex coeffs."""
    from bempp.api import function_space
    from bempp.api.operators.potential.laplace import double_layer
    from bempp.api import GridFunction

    default_parameters.quadrature.regular = 8

    grid = helpers.load_grid("sphere_h_01")
    space = function_space(grid, "P", 1)

    npoints = 100
    theta = _np.linspace(0, 2 * _np.pi, npoints)
    points = _np.vstack(
        [_np.cos(theta), _np.sin(theta), 3 * _np.ones(npoints, dtype="float64")]
    )

    data = helpers.load_npz_data("laplace_double_layer_pot_p1_complex")

    coefficients = data["arr_0"]
    expected = data["arr_1"]

    fun = GridFunction(space, coefficients=coefficients)

    actual = double_layer(
        space,
        points,
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).evaluate(fun)

    _np.testing.assert_allclose(actual, expected, rtol=5e-6)
