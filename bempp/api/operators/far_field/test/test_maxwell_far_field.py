"""Unit tests for Maxwell far-field assemblers."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

WAVENUMBER = 2.5

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")


def test_maxwell_electric_far_field(
    default_parameters, helpers, device_interface, precision
):
    """Test Maxwell electric far field."""
    from bempp.api import function_space
    from bempp.api import GridFunction
    from bempp.api.operators.far_field.maxwell import electric_field

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "RWG", 0)

    data = helpers.load_npz_data("maxwell_electric_far_field")

    coefficients = data["vec"]
    points = data["points"]
    expected = data["result"]

    fun = GridFunction(space, coefficients=coefficients)

    actual = electric_field(
        space,
        points,
        WAVENUMBER,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    ).evaluate(fun)

    _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))

# def test_helmholtz_double_layer_potential_p1(
    # default_parameters, helpers, device_interface, precision
# ):
    # """Test Helmholtz dlp far field with p1 basis."""
    # from bempp.api import function_space
    # from bempp.api import GridFunction
    # from bempp.api.operators.far_field.helmholtz import double_layer

    # grid = helpers.load_grid("sphere")
    # space = function_space(grid, "P", 1)

    # data = helpers.load_npz_data("helmholtz_double_layer_far_field_p1")

    # coefficients = data["vec"]
    # points = data["points"]
    # expected = data["result"]

    # fun = GridFunction(space, coefficients=coefficients)

    # actual = double_layer(
        # space,
        # points,
        # WAVENUMBER,
        # parameters=default_parameters,
        # precision=precision,
        # device_interface=device_interface,
    # ).evaluate(fun)

    # _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))

# def test_helmholtz_single_layer_potential_p1_complex_coeffs(
    # default_parameters, helpers, device_interface, precision
# ):
    # """Test Helmholtz slp potential with p1 basis and complex coeffs."""
    # from bempp.api import function_space
    # from bempp.api import GridFunction
    # from bempp.api.operators.potential.helmholtz import single_layer

    # grid = helpers.load_grid("sphere")
    # space = function_space(grid, "P", 1)

    # data = helpers.load_npz_data("helmholtz_single_layer_potential_p1")

    # points = data["points"]
    # coefficients = _np.random.rand(space.global_dof_count) + 1j * _np.random.rand(space.global_dof_count)

    # fun = GridFunction(space, coefficients=coefficients)
    # fun_real = GridFunction(space, coefficients=_np.real(coefficients))
    # fun_complex = GridFunction(space, coefficients=_np.imag(coefficients))

    # op = single_layer(
        # space,
        # points,
        # WAVENUMBER,
        # parameters=default_parameters,
        # precision=precision,
        # device_interface=device_interface,
    # )

    # expected = op.evaluate(fun_real) + 1j * op.evaluate(fun_complex)
    # actual = op.evaluate(fun)

    # _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))

