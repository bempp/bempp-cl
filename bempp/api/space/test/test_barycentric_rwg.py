
# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")

def test_rwg_barycentric_on_reference_triangle():
    """Test B-RWG space on unit triangle."""
    import bempp.api

    grid = bempp.api.shapes.reference_triangle()
    space = bempp.api.function_space(grid, "RWG", 0, include_boundary_dofs=True)
    space_bary = space.barycentric_representation
    assert space_bary.global_dof_count == 3

    # Define a function on this space

    coeffs = _np.array([0, 0, 1], dtype='float64')

    fun = bempp.api.GridFunction(space, coefficients=coeffs)
    fun_bary = bempp.api.GridFunction(space_bary, coefficients=coeffs)
    fun_bary.l2_norm()


