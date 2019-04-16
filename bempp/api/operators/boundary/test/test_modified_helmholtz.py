"""Unit tests for modified Helmholtz operators."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers", "small_sphere")


def test_modified_helmholtz_real_slp_p0(default_parameters, helpers):
    """Test modified Helmholtz slp with real omega."""
    from bempp.api import function_space
    from bempp.api.operators.boundary.modified_helmholtz import single_layer

    grid = helpers.load_grid("sphere_h_01")

    default_parameters.quadrature.singular = 8
    default_parameters.quadrature.regular = 8

    space = function_space(grid, "DP", 0)

    discrete_op = single_layer(
        space, space, space, 2.5, assembler="dense", parameters=default_parameters
    ).assemble()

    expected = helpers.load_npy_data(
        "modified_helmholtz_single_layer_sphere_h_01_w_2_5"
    )
    _np.testing.assert_allclose(discrete_op.A, expected, rtol=1e-5)
