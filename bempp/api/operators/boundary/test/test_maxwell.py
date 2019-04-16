"""Unit tests for modified Helmholtz operators."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

import bempp.api

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers", "small_sphere")


def test_maxwell_electric_field_sphere(
    default_parameters, helpers, device_interface, precision
):
    """Test Maxwell electric field on sphere."""
    from bempp.api import get_precision
    from bempp.api import function_space
    from bempp.api.operators.boundary.maxwell import electric_field

    grid = helpers.load_grid("sphere_h_01")

    default_parameters.quadrature.singular = 10
    default_parameters.quadrature.regular = 10

    space1 = function_space(grid, "RWG", 0)
    space2 = function_space(grid, "SNC", 0)

    discrete_op = electric_field(
        space1,
        space1,
        space2,
        2.5,
        assembler="dense",
        device_interface=device_interface,
        precision=precision,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npz_data("maxwell_electric_field_sphere_h_01_w_2_5")[
        "arr_0"
    ]
    _np.testing.assert_allclose(discrete_op.A, expected, rtol=1e-5, atol=1e-8)


def test_maxwell_electric_field_screen(
    default_parameters, helpers, device_interface, precision
):
    """Test Maxwell electric field on sphere."""
    from bempp.api import get_precision
    from bempp.api import function_space
    from bempp.api.operators.boundary.maxwell import electric_field

    if precision == "single":
        rtol = 1e-3
    else:
        rtol = 5e-5

    grid = helpers.load_grid("structured_grid")

    default_parameters.quadrature.singular = 8
    default_parameters.quadrature.regular = 10

    space1 = function_space(grid, "RWG", 0)
    space2 = function_space(grid, "SNC", 0)

    discrete_op = electric_field(
        space1,
        space1,
        space2,
        2.5,
        assembler="dense",
        device_interface=device_interface,
        precision=precision,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npz_data("maxwell_electric_field_screen_w_2_5")["arr_0"]
    _np.testing.assert_allclose(discrete_op.A, expected, rtol=rtol)


def test_maxwell_magnetic_field_sphere(
    default_parameters, helpers, device_interface, precision
):
    """Test Maxwell magnetic field on sphere."""
    from bempp.api import get_precision
    from bempp.api import function_space
    from bempp.api.operators.boundary.maxwell import magnetic_field

    # if precision == 'single':
    #    pytest.skip("Test runs only in double precision mode.")

    if precision == "double":
        rtol = 1e-4
        atol = 1e-9
    elif precision == "single":
        rtol = 1e-3
        atol = 1e-8
    else:
        raise ValueError("precision must be one of 'single' or 'double'")

    grid = helpers.load_grid("sphere_h_01")

    default_parameters.quadrature.singular = 6
    default_parameters.quadrature.regular = 6

    space1 = function_space(grid, "RWG", 0)
    space2 = function_space(grid, "SNC", 0)

    discrete_op = magnetic_field(
        space1,
        space1,
        space2,
        2.5,
        assembler="dense",
        device_interface=device_interface,
        precision=precision,
        parameters=default_parameters,
    ).assemble()

    expected = helpers.load_npz_data("maxwell_magnetic_field")["arr_0"]
    _np.testing.assert_allclose(discrete_op.A, expected, rtol=rtol, atol=atol)


def test_maxwell_multitrace_sphere(default_parameters, device_interface, precision):
    """Test Maxwell magnetic field on sphere."""
    from bempp.api import get_precision
    from bempp.api import function_space
    from bempp.api.shapes import regular_sphere
    from bempp.api.operators.boundary.maxwell import electric_field, magnetic_field
    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator

    # if precision == 'single':
    #    pytest.skip("Test runs only in double precision mode.")

    if precision == "double":
        rtol = 1e-14
    elif precision == "single":
        rtol = 1e-6
    else:
        raise ValueError("precision must be one of 'single' or 'double'")

    grid = regular_sphere(3)

    default_parameters.quadrature.singular = 6
    default_parameters.quadrature.regular = 6

    op = bempp.api.operators.boundary.maxwell.multitrace_operator(
        grid,
        2.5,
        default_parameters,
        assembler="multitrace_evaluator",
        device_interface=device_interface,
        precision=precision,
    )

    efield = electric_field(
        op.domain_spaces[0],
        op.range_spaces[0],
        op.dual_to_range_spaces[0],
        2.5,
        parameters=default_parameters,
        assembler="dense",
        device_interface=device_interface,
        precision=precision,
    ).weak_form()

    mfield = magnetic_field(
        op.domain_spaces[0],
        op.range_spaces[0],
        op.dual_to_range_spaces[0],
        2.5,
        parameters=default_parameters,
        assembler="dense",
        device_interface=device_interface,
        precision=precision,
    ).weak_form()

    expected = BlockedDiscreteOperator(_np.array([[mfield, efield], [-efield, mfield]]))

    rand = _np.random.RandomState(0)
    x = rand.randn(expected.shape[1])

    y_expected = expected @ x
    y_actual = op.weak_form() @ x

    error = _np.linalg.norm(y_expected - y_actual) / _np.linalg.norm(y_expected)
    assert error < rtol


def test_maxwell_transmission_sphere(default_parameters, device_interface, precision):
    """Test Maxwell magnetic field on sphere."""
    from bempp.api import get_precision
    from bempp.api import function_space
    from bempp.api.shapes import regular_sphere
    from bempp.api.operators.boundary.maxwell import electric_field, magnetic_field
    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator

    # if precision == 'single':
    #    pytest.skip("Test runs only in double precision mode.")

    if precision == "double":
        rtol = 1e-14
    elif precision == "single":
        rtol = 1e-6
    else:
        raise ValueError("precision must be one of 'single' or 'double'")

    grid = regular_sphere(3)

    default_parameters.quadrature.singular = 6
    default_parameters.quadrature.regular = 6

    eps_rel = 1.3
    mu_rel = 1.5

    wavenumber = 2.5

    op = bempp.api.operators.boundary.maxwell.transmission_operator(
        grid,
        wavenumber,
        eps_rel,
        mu_rel,
        default_parameters,
        assembler="multitrace_evaluator",
        device_interface=device_interface,
        precision=precision,
    )

    domain = op.domain_spaces[0]
    range_ = op.domain_spaces[0]
    dual_to_range = op.dual_to_range_spaces[0]

    sqrt_eps_rel = _np.sqrt(eps_rel)
    sqrt_mu_rel = _np.sqrt(mu_rel)

    wavenumber_int = wavenumber * sqrt_eps_rel * sqrt_mu_rel

    magnetic_ext = magnetic_field(
        domain,
        range_,
        dual_to_range,
        wavenumber,
        default_parameters,
        "dense",
        device_interface,
        precision,
    ).weak_form()

    magnetic_int = magnetic_field(
        domain,
        range_,
        dual_to_range,
        wavenumber_int,
        default_parameters,
        "dense",
        device_interface,
        precision,
    ).weak_form()

    electric_ext = electric_field(
        domain,
        range_,
        dual_to_range,
        wavenumber,
        default_parameters,
        "dense",
        device_interface,
        precision,
    ).weak_form()

    electric_int = electric_field(
        domain,
        range_,
        dual_to_range,
        wavenumber_int,
        default_parameters,
        "dense",
        device_interface,
        precision,
    ).weak_form()

    fac = sqrt_mu_rel / sqrt_eps_rel

    expected = BlockedDiscreteOperator(
        _np.array(
            [
                [magnetic_ext + magnetic_int, electric_ext + fac * electric_int],
                [
                    (-1.0 / fac) * electric_int - electric_ext,
                    magnetic_int + magnetic_ext,
                ],
            ],
            dtype=_np.object,
        )
    )

    rand = _np.random.RandomState(0)
    x = rand.randn(expected.shape[1])

    y_expected = expected @ x
    y_actual = op.weak_form() @ x

    error = _np.linalg.norm(y_expected - y_actual) / _np.linalg.norm(y_expected)
    assert error < rtol

