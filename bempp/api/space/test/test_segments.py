"""Unit tests for Space objects."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest


@pytest.mark.parametrize('space_info', [("P", 1), ("DUAL", 0)])
def test_segments_space_with_boundary_dofs(space_info):
    """Test barycentric space."""
    import bempp.api

    grid = bempp.api.shapes.cube(h=0.4)

    space0 = bempp.api.function_space(
        grid, space_info[0], space_info[1], segments=[1],
        include_boundary_dofs=False
    )
    fun0 = bempp.api.GridFunction(space0, coefficients=_np.ones(space0.global_dof_count))

    space1 = bempp.api.function_space(
        grid, space_info[0], space_info[1], segments=[1],
        include_boundary_dofs=True,
        truncate_at_segment_edge=True
    )
    fun1 = bempp.api.GridFunction(space1, coefficients=_np.ones(space1.global_dof_count))

    space2 = bempp.api.function_space(
        grid, space_info[0], space_info[1], segments=[1],
        include_boundary_dofs=True,
        truncate_at_segment_edge=False
    )
    fun2 = bempp.api.GridFunction(space2, coefficients=_np.ones(space2.global_dof_count))

    assert space0.global_dof_count < space1.global_dof_count
    assert space1.global_dof_count == space2.global_dof_count
    assert fun0.l2_norm() < fun1.l2_norm() < fun2.l2_norm()


@pytest.mark.parametrize('space_info', [("DP", 0), ("DP", 1)])
def test_segments_space_without_boundary_dofs(space_info, helpers, precision):
    """Test spaces on segments."""
    import bempp.api
    import math

    grid = bempp.api.shapes.cube(h=0.4)

    space0 = bempp.api.function_space(
        grid, space_info[0], space_info[1], segments=[1],
        include_boundary_dofs=False
    )
    fun0 = bempp.api.GridFunction(space0, coefficients=_np.ones(space0.global_dof_count))

    space1 = bempp.api.function_space(
        grid, space_info[0], space_info[1], segments=[1],
        include_boundary_dofs=True,
        truncate_at_segment_edge=True
    )
    fun1 = bempp.api.GridFunction(space1, coefficients=_np.ones(space1.global_dof_count))

    space2 = bempp.api.function_space(
        grid, space_info[0], space_info[1], segments=[1],
        include_boundary_dofs=True,
        truncate_at_segment_edge=False
    )
    fun2 = bempp.api.GridFunction(space2, coefficients=_np.ones(space2.global_dof_count))

    assert space0.global_dof_count == space1.global_dof_count == space2.global_dof_count
    assert math.isclose(fun0.l2_norm(), fun1.l2_norm(),
                        rel_tol=helpers.default_tolerance(precision))
    assert math.isclose(fun0.l2_norm(), fun2.l2_norm(),
                        rel_tol=helpers.default_tolerance(precision))


@pytest.mark.parametrize('space_info', [("DUAL", 1)])
def test_segments_dual1_space(space_info, helpers, precision):
    """Test spaces on segments."""
    import bempp.api
    import math

    grid = bempp.api.shapes.cube(h=0.4)

    space0 = bempp.api.function_space(
        grid, space_info[0], space_info[1], segments=[1],
        include_boundary_dofs=False,
        truncate_at_segment_edge=False
    )
    fun0 = bempp.api.GridFunction(space0, coefficients=_np.ones(space0.global_dof_count))

    space1 = bempp.api.function_space(
        grid, space_info[0], space_info[1], segments=[1],
        include_boundary_dofs=True,
        truncate_at_segment_edge=True
    )
    fun1 = bempp.api.GridFunction(space1, coefficients=_np.ones(space1.global_dof_count))

    space2 = bempp.api.function_space(
        grid, space_info[0], space_info[1], segments=[1],
        include_boundary_dofs=True,
        truncate_at_segment_edge=False
    )
    fun2 = bempp.api.GridFunction(space2, coefficients=_np.ones(space2.global_dof_count))

    assert space0.global_dof_count == space1.global_dof_count == space2.global_dof_count
    assert math.isclose(fun0.l2_norm(), fun2.l2_norm(),
                        rel_tol=helpers.default_tolerance(precision))
    assert 0 < fun1.l2_norm() < fun2.l2_norm()
