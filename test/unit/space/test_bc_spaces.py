"""Unit tests for Buffa-Christansen spaces."""

import bempp.api
import pytest
import numpy as np


@pytest.mark.parametrize("space_type", ["BC", "RBC"])
def test_segments(space_type):
    """Test creation of BC space on multitrace cube."""
    grid = bempp.api.shapes.multitrace_cube(h=1)
    bempp.api.function_space(grid, space_type, 0, segments=[1, 2, 3, 4, 5, 6])


@pytest.mark.parametrize("space_type", ["BC", "RBC"])
def test_swapped_normals(space_type):
    """Test creation of BC space on multitrace cube with swapped normals."""
    grid = bempp.api.shapes.multitrace_cube(h=1)
    bempp.api.function_space(
        grid, space_type, 0, segments=[6, 7, 8, 9, 10, 11], swapped_normals=[6]
    )


@pytest.mark.parametrize("space_type", ["BC", "RBC"])
def test_segments_6_7(space_type):
    """Test creation of BC space on multitrace cube."""
    grid = bempp.api.shapes.multitrace_cube(h=1)
    bempp.api.function_space(
        grid, space_type, 0, segments=[6, 7, 8, 9, 10, 11], swapped_normals=[6]
    )


@pytest.mark.parametrize("space_type", ["RT", "BC"])
def test_Hdiv_continuity(space_type):
    """Test that BC spaces have continuous normal components."""
    grid = bempp.api.shapes.sphere(h=0.2)
    space = bempp.api.function_space(grid, space_type, 0)
    f = bempp.api.GridFunction(space, coefficients=np.random.rand(space.global_dof_count))

    for e, (vertices, neighbors) in enumerate(zip(grid.edges.T, grid.edge_neighbors)):
        if grid.element_edges[0, neighbors[0]] == e:
            v0 = [[0, 0], [1, 0]]
        elif grid.element_edges[1, neighbors[0]] == e:
            v0 = [[0, 0], [0, 1]]
        else:
            assert grid.element_edges[2, neighbors[0]] == e
            v0 = [[1, 0], [0, 1]]
        points0 = np.array([[a + i / 5 * (b - a) for i in range(6)] for a, b in zip(*v0)])

        if grid.element_edges[0, neighbors[1]] == e:
            v1 = [[0, 0], [1, 0]]
        elif grid.element_edges[1, neighbors[1]] == e:
            v1 = [[0, 0], [0, 1]]
        else:
            assert grid.element_edges[2, neighbors[1]] == e
            v1 = [[0, 1], [0, 1]]
        points1 = np.array([[a + i / 5 * (b - a) for i in range(6)] for a, b in zip(*v1)])

        tangent = grid.vertices[:, vertices[1]] - grid.vertices[:, vertices[0]]
        tangent /= np.linalg.norm(tangent)
        normal0 = np.cross(tangent, grid.normals[neighbors[0]])
        normal1 = np.cross(tangent, grid.normals[neighbors[1]])

        assert np.isclose(np.linalg.norm(normal0), 1)
        assert np.isclose(np.linalg.norm(normal1), 1)

        values0 = f.evaluate(neighbors[0], points0)
        values1 = f.evaluate(neighbors[1], points1)

        components0 = [normal0.dot(v) for v in values0.T]
        components1 = [normal1.dot(v) for v in values1.T]

        assert np.allclose(components0, components1)


@pytest.mark.parametrize("space_type", ["NC", "RBC"])
def test_Hcurl_continuity(space_type):
    """Test that BC spaces have continuous normal components."""
    grid = bempp.api.shapes.sphere(h=0.2)
    space = bempp.api.function_space(grid, space_type, 0)
    f = bempp.api.GridFunction(space, coefficients=np.random.rand(space.global_dof_count))

    for e, (vertices, neighbors) in enumerate(zip(grid.edges.T, grid.edge_neighbors)):
        if grid.element_edges[0, neighbors[0]] == e:
            v0 = [[0, 0], [1, 0]]
        elif grid.element_edges[1, neighbors[0]] == e:
            v0 = [[0, 0], [0, 1]]
        else:
            assert grid.element_edges[2, neighbors[0]] == e
            v0 = [[1, 0], [0, 1]]
        points0 = np.array([[a + i / 5 * (b - a) for i in range(6)] for a, b in zip(*v0)])

        if grid.element_edges[0, neighbors[1]] == e:
            v1 = [[0, 0], [1, 0]]
        elif grid.element_edges[1, neighbors[1]] == e:
            v1 = [[0, 0], [0, 1]]
        else:
            assert grid.element_edges[2, neighbors[1]] == e
            v1 = [[0, 1], [0, 1]]
        points1 = np.array([[a + i / 5 * (b - a) for i in range(6)] for a, b in zip(*v1)])

        tangent = grid.vertices[:, vertices[1]] - grid.vertices[:, vertices[0]]
        tangent /= np.linalg.norm(tangent)

        values0 = f.evaluate(neighbors[0], points0)
        values1 = f.evaluate(neighbors[1], points1)

        components0 = [tangent.dot(v) for v in values0.T]
        components1 = [tangent.dot(v) for v in values1.T]

        assert np.allclose(components0, components1)
