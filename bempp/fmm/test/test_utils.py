"""Test the FMM utility routines."""

import numpy as np
import pytest


def test_space_to_point_matrix():
    """Test map of space coefficients to quad points."""
    import bempp.api
    from bempp.api.grid.octree import Octree
    from bempp.api.fmm.utils import space_to_points_matrix

    grid = bempp.api.shapes.regular_sphere(5)
    octree = Octree(grid)
    space = bempp.api.function_space(grid, "P", 1)
    mat = space_to_points_matrix(space, octree, 3)

    v = np.ones(space.global_dof_count, "float64")

    # The sum over all matrix elements should be the size
    # of the surface of the unit sphere. (4pi)

    actual = np.sum(mat @ v)
    expected = 4 * np.pi

    np.testing.assert_approx_equal(
        actual,
        expected,
        significant=3,
        err_msg="Sum over all matrix elements should be approximately 4 pi.",
    )
