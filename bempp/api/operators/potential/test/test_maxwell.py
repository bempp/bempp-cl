"""Unit tests for Maxwell potential assemblers."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers", "small_sphere")


def test_maxwell_electric_field_rwg(
    default_parameters, helpers, device_interface, precision
):
    """Test Maxwell efield potential with RWG basis."""
    from bempp.api import function_space
    from bempp.api.operators.potential.maxwell import electric_field
    from bempp.api import GridFunction

    default_parameters.quadrature.regular = 8

    grid = helpers.load_grid("sphere_h_01")
    space = function_space(grid, "RWG", 0)

    npoints = 100
    theta = _np.linspace(0, 2 * _np.pi, npoints)
    points = _np.vstack(
        [_np.cos(theta), _np.sin(theta), 3 * _np.ones(npoints, dtype="float64")]
    )

    data = helpers.load_npz_data("maxwell_electric_field_potential")

    coefficients = data["arr_0"]
    expected = data["arr_1"]

    fun = GridFunction(space, coefficients=coefficients)

    actual = electric_field(
        space,
        points,
        2.5,
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).evaluate(fun)

    _np.testing.assert_allclose(actual, expected, rtol=5E-3)
