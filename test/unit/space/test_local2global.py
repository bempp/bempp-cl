"""Unit tests for Space objects."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest
import bempp.api


@pytest.mark.parametrize(
    "grid",
    [
        bempp.api.shapes.sphere(h=0.5),
        bempp.api.shapes.cube(h=0.5),
        bempp.api.shapes.shapes.screen(
            _np.array([(1, 1, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0)]), h=0.5
        ),
    ],
)
@pytest.mark.parametrize(
    "space_type",
    [
        ("DP", 0),
        ("DP", 1),
        ("P", 1),
        ("DUAL", 0),
        ("DUAL", 1),
        ("RWG", 0),
        ("SNC", 0),
        ("BC", 0),
        ("RBC", 0),
    ],
)
@pytest.mark.parametrize("include_boundary_dofs", [True, False])
@pytest.mark.parametrize("truncate_at_segment_edge", [True, False])
def test_local2global(
    grid, space_type, include_boundary_dofs, truncate_at_segment_edge
):
    """Check that data in local2global and global2local agree."""
    space = bempp.api.function_space(
        grid,
        *space_type,
        include_boundary_dofs=include_boundary_dofs,
        truncate_at_segment_edge=truncate_at_segment_edge
    )
    test_local2global = _np.full_like(space.local2global, -1)
    for i, locals in enumerate(space.global2local):
        for cell, dof in locals:
            assert space.local2global[cell][dof] == i
            test_local2global[cell][dof] = i
    assert _np.allclose(space.local2global, test_local2global)
