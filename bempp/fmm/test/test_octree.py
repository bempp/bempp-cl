"""Test the Octree."""

import bempp.api
import numpy as np
import pytest

TREE_LEVEL = 3
LBOUND = np.array([-1., -1., -1.])
UBOUND = np.array([1., 1., 1.])

@pytest.fixture
def octree():
    """Return a simple Octree."""
    from bempp.api.fmm.octree import Octree
    lbound = LBOUND
    ubound = UBOUND

    grid = bempp.api.shapes.regular_sphere(5)

    tree = Octree(lbound, ubound, TREE_LEVEL, grid.vertices)
    return tree



def test_octree_diameter(octree):
    """Query simple octree properties."""

    actual = octree.upper_bound - octree.lower_bound
    expected = UBOUND - LBOUND
    np.testing.assert_allclose(actual, expected)


def test_assign_data(octree):
    """Test if node assignments and parents is correct."""

    index_ptr = octree.leaf_nodes_ptr
    vertices = octree.vertices


    count = 0
    for index, leaf_node in enumerate(octree.non_empty_leaf_nodes):
        for vertex_index in octree.sorted_indices[index_ptr[index]: index_ptr[index + 1]]:
            lbound, ubound = octree.node_bounds(
                    leaf_node, octree.maximum_level)
            count += 1
            assert (np.all(lbound <= vertices[:, vertex_index]) and 
                          np.all(ubound >= vertices[:, vertex_index]) and not 
                            np.all(ubound == vertices[:, vertex_index]))

    assert count == vertices.shape[1]


def test_node_parents(octree):
    """Test that parents have been correctly assigned."""

    expected = set(octree.non_empty_leaf_nodes)

    assert len(expected) == len(octree.non_empty_leaf_nodes)
    nodes_ptr = octree.non_empty_nodes_ptr

    for level_index in range(octree.maximum_level, -1, -1):
        actual = octree.non_empty_nodes_by_level[
                    nodes_ptr[level_index]:nodes_ptr[1 + level_index]
                    ]
        assert len(actual) == len(set(actual))
        difference = expected.difference(actual)
        assert len(difference) == 0
        expected = {octree.parent(node) for node in actual}
