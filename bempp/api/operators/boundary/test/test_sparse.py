"""Unit tests for sparse operators."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers", "small_sphere")


def test_sparse_identity_p0(
    default_parameters, helpers, small_sphere, device_interface, precision
):
    """Test singular assembler for the sparse L^2 identity with p0 basis."""
    from scipy.sparse import coo_matrix
    from bempp.api import function_space
    from bempp.api.operators.boundary.sparse import identity

    grid = small_sphere
    expected_data = helpers.load_npz_data("identity_p0_p0")
    expected = coo_matrix(
        (expected_data["data"], (expected_data["row"], expected_data["col"]))
    ).todense()

    space = function_space(grid, "DP", 0)

    actual = (
        identity(
            space,
            space,
            space,
            parameters=default_parameters,
            device_interface=device_interface,
            precision=precision,
        )
        .assemble()
        .A.todense()
    )

    _np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_sparse_identity_p1(
    default_parameters, helpers, small_sphere, device_interface, precision
):
    """Test singular assembler for the sparse L^2 identity with p1 basis."""
    from scipy.sparse import coo_matrix
    from bempp.api import function_space
    from bempp.api.operators.boundary.sparse import identity

    grid = small_sphere
    expected_data = helpers.load_npz_data("identity_p1_p1")
    expected = coo_matrix(
        (expected_data["data"], (expected_data["row"], expected_data["col"]))
    ).todense()

    space = function_space(grid, "DP", 1)

    actual = (
        identity(
            space,
            space,
            space,
            parameters=default_parameters,
            device_interface=device_interface,
            precision=precision,
        )
        .assemble()
        .A.todense()
    )

    _np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_sparse_identity_p0_p1(
    default_parameters, helpers, small_sphere, device_interface, precision
):
    """Test singular assembler for the sparse L^2 identity with p0/p1 basis."""
    from scipy.sparse import coo_matrix
    from bempp.api import function_space
    from bempp.api.operators.boundary.sparse import identity

    grid = small_sphere
    number_of_elements = grid.number_of_elements
    expected_data = helpers.load_npz_data("identity_p0_p1")
    expected = coo_matrix(
        (expected_data["data"], (expected_data["row"], expected_data["col"])),
        shape=(number_of_elements, 3 * number_of_elements),
    ).todense()

    space0 = function_space(grid, "DP", 0)
    space1 = function_space(grid, "DP", 1)

    actual = (
        identity(
            space1,
            space1,
            space0,
            parameters=default_parameters,
            device_interface=device_interface,
            precision=precision,
        )
        .assemble()
        .A.todense()
    )

    _np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_sparse_identity_p1_p0(
    default_parameters, helpers, small_sphere, device_interface, precision
):
    """Test singular assembler for the sparse L^2 identity with p1/p0 basis."""
    from scipy.sparse import coo_matrix
    from bempp.api import function_space
    from bempp.api.operators.boundary.sparse import identity

    grid = small_sphere
    number_of_elements = grid.number_of_elements
    expected_data = helpers.load_npz_data("identity_p1_p0")
    expected = coo_matrix(
        (expected_data["data"], (expected_data["row"], expected_data["col"])),
        shape=(3 * number_of_elements, number_of_elements),
    ).todense()

    space0 = function_space(grid, "DP", 1)
    space1 = function_space(grid, "DP", 0)

    actual = (
        identity(
            space1,
            space1,
            space0,
            parameters=default_parameters,
            device_interface=device_interface,
            precision=precision,
        )
        .assemble()
        .A.todense()
    )

    _np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_maxwell_identity(
    default_parameters, helpers, device_interface, precision
):
    """Test singular assembler for the sparse L^2 identity with p0 basis."""
    from scipy.sparse import coo_matrix
    from bempp.api import function_space
    from bempp.api.operators.boundary.sparse import identity

    grid = helpers.load_grid("sphere_h_01")
    expected_data = helpers.load_npz_data("maxwell_identity")
    expected = coo_matrix(
        (expected_data["data"], (expected_data["row"], expected_data["col"]))
    ).todense()

    space1 = function_space(grid, "RWG", 0)
    space2 = function_space(grid, "SNC", 0)

    actual = (
        identity(
            space1,
            space1,
            space2,
            parameters=default_parameters,
            device_interface=device_interface,
            precision=precision,
        )
        .assemble()
        .A.todense()
    )

    _np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1E-8)
