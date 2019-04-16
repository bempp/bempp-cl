"""Unit tests for the grid class."""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest


@pytest.mark.usefixtures("two_element_grid")
@pytest.fixture
def two_element_geometries(two_element_grid):
    """Return geometries of two element grid."""

    geometries = [elem.geometry for elem in two_element_grid.entity_iterator(0)]
    return geometries


def test_grid_number_of_elements(two_element_grid):
    """Test number of elements in Grid."""
    assert two_element_grid.number_of_elements == 2


def test_number_of_vertices(two_element_grid):
    """Test the number of vertices in grid."""
    assert two_element_grid.number_of_vertices == 4


def test_number_of_edges(two_element_grid):
    """Test the number of edges."""
    assert two_element_grid.number_of_edges == 5


def test_as_array(two_element_grid):
    """Test conversion of a grid to an array."""
    expected = np.array(
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], dtype="float64"
    )

    actual = two_element_grid.as_array

    np.testing.assert_allclose(expected, actual)


def test_edge_adjacency(small_sphere):
    """Check edge connectivity information for a small sphere."""
    # pylint: disable=too-many-locals
    from bempp.api.grid.grid import get_element_to_element_matrix

    edge_adjacency = small_sphere.edge_adjacency

    e2e = get_element_to_element_matrix(
        small_sphere.vertices, small_sphere.elements
    ).tocoo()

    number_edge_adjacent_elements = np.count_nonzero(e2e.data == 2)

    assert number_edge_adjacent_elements == edge_adjacency.shape[1]

    for adjacency in edge_adjacency.T:
        test_element = adjacency[0]
        trial_element = adjacency[1]
        test_local_vertex_indices = adjacency[2:4]
        trial_local_vertex_indices = adjacency[4:]

        test_global_vertex_indices = small_sphere.elements[
            test_local_vertex_indices, test_element
        ]
        trial_global_vertex_indices = small_sphere.elements[
            trial_local_vertex_indices, trial_element
        ]

        np.testing.assert_equal(test_global_vertex_indices, trial_global_vertex_indices)


def test_vertex_adjacency(small_sphere):
    """Check vertex connectivity information for a small sphere."""
    # pylint: disable=too-many-locals
    from bempp.api.grid.grid import get_element_to_element_matrix

    vertex_adjacency = small_sphere.vertex_adjacency

    e2e = get_element_to_element_matrix(
        small_sphere.vertices, small_sphere.elements
    ).tocoo()

    number_vertex_adjacent_elements = np.count_nonzero(e2e.data == 1)

    assert number_vertex_adjacent_elements == vertex_adjacency.shape[1]

    for adjacency in vertex_adjacency.T:
        test_element = adjacency[0]
        trial_element = adjacency[1]
        test_local_vertex_index = adjacency[2]
        trial_local_vertex_index = adjacency[3]

        test_global_vertex_index = small_sphere.elements[
            test_local_vertex_index, test_element
        ]
        trial_global_vertex_index = small_sphere.elements[
            trial_local_vertex_index, trial_element
        ]

        assert test_global_vertex_index == trial_global_vertex_index


def test_element_volume(two_element_geometries):
    """Check the volume of an element."""
    for geom in two_element_geometries:
        np.testing.assert_almost_equal(geom.volume, 0.5)


def test_integration_element(two_element_geometries):
    """Check the volume of an element."""
    for geom in two_element_geometries:
        np.testing.assert_almost_equal(geom.integration_element, 1)
