
# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")

def test_rwg_barycentric(helpers, precision):
    """Test B-RWG space."""
    import bempp.api
    import math

    grid = bempp.api.shapes.regular_sphere(3)
    space = bempp.api.function_space(grid, "RWG", 0, include_boundary_dofs=True)
    space_bary = space.barycentric_representation()

    # Define a function on this space

    rand = _np.random.RandomState(0)
    coeffs = rand.randn(space.global_dof_count)

    fun = bempp.api.GridFunction(space, coefficients=coeffs)
    fun_bary = bempp.api.GridFunction(space_bary, coefficients=coeffs)

    assert math.isclose(fun.l2_norm(), fun_bary.l2_norm(),
            rel_tol=helpers.default_tolerance(precision))

