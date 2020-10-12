"""Unit tests for Space objects."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest
import bempp.api

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")


def test_p1_color_map():
    """Test if the color map for p1 spaces is correct."""
    grid = bempp.api.shapes.cube()

    space = bempp.api.function_space(grid, "P", 1)

    colors_unique = True

    for local_dofs in space.global2local:
        colors = [space.color_map[elem] for elem, _ in local_dofs]
        if len(colors) != len(set(colors)):
            colors_unique = False

    assert colors_unique


def test_rwg_color_map():
    """Test if the color map for RWG spaces is correct."""
    grid = bempp.api.shapes.cube()

    space = bempp.api.function_space(grid, "RWG", 0)

    colors_unique = True

    for local_dofs in space.global2local:
        colors = [space.color_map[elem] for elem, _ in local_dofs]
        if len(colors) != len(set(colors)):
            colors_unique = False

    assert colors_unique


def test_p1_open_segment():
    """Check a P1 open segment."""
    grid = bempp.api.shapes.cube()

    space = bempp.api.function_space(grid, "P", 1, segments=[1])

    dofs_empty = True
    boundary_dofs_empty = True

    for elem_index in range(grid.number_of_elements):
        if space.support[elem_index]:
            if _np.any(
                space.support[
                    grid.element_neighbors.indices[
                        grid.element_neighbors.indexptr[
                            elem_index
                        ] : grid.element_neighbors.indexptr[elem_index + 1]
                    ]
                ]
                is False
            ):
                # Element is on the boundary
                for index, vertex_index in enumerate(grid.elements[:, elem_index]):
                    neighbors = grid.vertex_neighbors.indices[
                        grid.vertex_neighbors.indexptr[
                            vertex_index
                        ] : grid.vertex_neighbors.indexptr[vertex_index + 1]
                    ]
                    if _np.any(space.support[neighbors] is False):
                        # Vertex is on the boundary
                        if space.local_multipliers[elem_index, index] != 0:
                            boundary_dofs_empty = False
        else:
            if _np.any(space.local_multipliers[elem_index] != 0):
                dofs_empty = False

    assert dofs_empty
    assert boundary_dofs_empty


def test_p1_extended_segment():
    """Check a P1 extended segment."""
    grid = bempp.api.shapes.cube()

    space = bempp.api.function_space(
        grid,
        "P",
        1,
        segments=[1],
        include_boundary_dofs=True,
        truncate_at_segment_edge=False,
    )

    eligible_index_pairs = set()

    for vertex_index in range(grid.number_of_vertices):
        neighbors = grid.vertex_neighbors.indices[
            grid.vertex_neighbors.indexptr[
                vertex_index
            ] : grid.vertex_neighbors.indexptr[vertex_index + 1]
        ]
        if 1 in grid.domain_indices[neighbors]:
            # Vertex adjacent an element with domain index 1
            for index_pair in zip(*_np.where(grid.elements == vertex_index)):
                eligible_index_pairs.add(index_pair)

    for local_index in range(3):
        for elem_index in range(grid.number_of_elements):
            if (local_index, elem_index) in eligible_index_pairs:
                assert space.local_multipliers[elem_index, local_index] == 1
            else:
                assert space.local_multipliers[elem_index, local_index] == 0


def test_p1_closed_segment():
    """Check a P1 closed segment."""
    grid = bempp.api.shapes.cube()

    space = bempp.api.function_space(
        grid, "P", 1, segments=[1], include_boundary_dofs=True
    )

    dofs_empty = True
    boundary_dofs_included = True

    for elem_index in range(grid.number_of_elements):
        if space.support[elem_index]:
            if _np.any(
                space.support[
                    grid.element_neighbors.indices[
                        grid.element_neighbors.indexptr[
                            elem_index
                        ] : grid.element_neighbors.indexptr[elem_index + 1]
                    ]
                ]
                is False
            ):
                # Element is on the boundary
                if _np.any(space.local_multipliers[elem_index] == 0):
                    boundary_dofs_included = False
        else:
            if _np.any(space.local_multipliers[elem_index] != 0):
                dofs_empty = False

    assert dofs_empty
    assert boundary_dofs_included


def test_rwg_open_segment():
    """Check an RWG open segment."""
    grid = bempp.api.shapes.cube()

    space = bempp.api.function_space(grid, "RWG", 0, segments=[1])

    dofs_empty = True
    boundary_dofs_empty = True

    for elem_index in range(grid.number_of_elements):
        if space.support[elem_index]:
            if _np.any(
                space.support[
                    grid.element_neighbors.indices[
                        grid.element_neighbors.indexptr[
                            elem_index
                        ] : grid.element_neighbors.indexptr[elem_index + 1]
                    ]
                ]
                is False
            ):
                # Element is on the boundary
                for index, edge_index in enumerate(grid.element_edges[:, elem_index]):
                    neighbors = list(grid.edge_neighbors[edge_index])
                    if _np.any(space.support[neighbors] is False):
                        # Edge is on the boundary
                        if space.local_multipliers[elem_index, index] != 0:
                            boundary_dofs_empty = False
        else:
            if _np.any(space.local_multipliers[elem_index] != 0):
                dofs_empty = False

    assert dofs_empty
    assert boundary_dofs_empty


def test_rwg_closed_segment():
    """Check an RWG closed segment."""
    grid = bempp.api.shapes.cube()

    space = bempp.api.function_space(
        grid, "RWG", 0, segments=[1], include_boundary_dofs=True
    )

    for elem_index in range(grid.number_of_elements):
        if space.support[elem_index]:
            assert _np.all(space.local_multipliers[elem_index] != 0)
        else:
            assert _np.all(space.local_multipliers[elem_index] == 0)


def test_snc_closed_segment():
    """Check an SNC closed segment."""
    grid = bempp.api.shapes.cube()

    space = bempp.api.function_space(
        grid, "SNC", 0, segments=[1], include_boundary_dofs=True
    )

    for elem_index in range(grid.number_of_elements):
        if space.support[elem_index]:
            assert _np.all(space.local_multipliers[elem_index] != 0)
        else:
            assert _np.all(space.local_multipliers[elem_index] == 0)


def test_snc_open_segment():
    """Check an SNC open segment."""
    grid = bempp.api.shapes.cube()

    space = bempp.api.function_space(grid, "SNC", 0, segments=[1])

    dofs_empty = True
    boundary_dofs_empty = True

    for elem_index in range(grid.number_of_elements):
        if space.support[elem_index]:
            if _np.any(
                space.support[
                    grid.element_neighbors.indices[
                        grid.element_neighbors.indexptr[
                            elem_index
                        ] : grid.element_neighbors.indexptr[elem_index + 1]
                    ]
                ]
                is False
            ):
                # Element is on the boundary
                for index, edge_index in enumerate(grid.element_edges[:, elem_index]):
                    neighbors = list(grid.edge_neighbors[edge_index])
                    if _np.any(space.support[neighbors] is False):
                        # Edge is on the boundary
                        if space.local_multipliers[elem_index, index] != 0:
                            boundary_dofs_empty = False
        else:
            if _np.any(space.local_multipliers[elem_index] != 0):
                dofs_empty = False

    assert dofs_empty
    assert boundary_dofs_empty


def test_dp1_closed_segment():
    """Check an DP1 closed segment."""
    grid = bempp.api.shapes.cube()

    space = bempp.api.function_space(grid, "DP", 1, segments=[1])

    for elem_index in range(grid.number_of_elements):
        if space.support[elem_index]:
            assert _np.all(space.local_multipliers[elem_index] != 0)
        else:
            assert _np.all(space.local_multipliers[elem_index] == 0)


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
def test_local2global(grid, space_type):
    """Check that data in local2global and global2local agree."""
    space = bempp.api.function_space(grid, *space_type)
    test_local2global = _np.full_like(space.local2global, -1)
    for i, locals in enumerate(space.global2local):
        for cell, dof in locals:
            assert space.local2global[cell][dof] == i
            test_local2global[cell][dof] = i
    assert _np.allclose(space.local2global, test_local2global)
