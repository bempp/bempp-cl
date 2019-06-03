"""Unit tests for Maxwell potential assemblers."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")

WAVENUMBER = 2.5
WAVENUMBER_COMPLEX = 2.5 + 1j


def test_maxwell_electric_field_potential_complex(
    default_parameters, helpers, device_interface, precision
):
    """Test Maxwell efield potential with complex wavenumber."""
    from bempp.api import function_space
    from bempp.api import GridFunction
    from bempp.api.operators.potential.maxwell import electric_field

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "RWG", 0)

    data = helpers.load_npz_data("maxwell_electric_field_potential_complex")

    coefficients = data["vec"]
    points = data["points"]
    expected = data["result"]

    fun = GridFunction(space, coefficients=coefficients)

    actual = electric_field(
        space,
        points,
        WAVENUMBER_COMPLEX,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    ).evaluate(fun)

    _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))


def test_maxwell_electric_field_potential_rwg(
    default_parameters, helpers, device_interface, precision
):
    """Test Maxwell efield potential."""
    from bempp.api import function_space
    from bempp.api import GridFunction
    from bempp.api.operators.potential.maxwell import electric_field

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "RWG", 0)

    data = helpers.load_npz_data("maxwell_electric_field_potential")

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
