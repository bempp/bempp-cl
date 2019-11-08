"""Common FMM routines."""
import abc as _abc
import numpy as _np
import numba as _numba


class LeafNode(object):
    """Definition of an FMM leaf node."""

    def __init__(self, identifier, source_ids, target_ids, colleagues):
        """Initialize a node."""

        self._identifier = identifier
        self._source_ids = _np.array(source_ids, dtype=_np.int64)
        self._target_ids = _np.array(target_ids, dtype=_np.int64)
        self._colleagues = _np.array(colleagues, dtype=_np.int64)

    @property
    def identifier(self):
        """Return Identifier."""
        return self._identifier

    @property
    def source_ids(self):
        """A list of source ids associated with the node."""
        return self._source_ids

    @property
    def target_ids(self):
        """A list of target ids associated with the node."""
        return self._target_ids

    @property
    def colleagues(self):
        """Return the colleagues of the node."""
        return self._colleagues


def map_space_to_points(
    nodes, space, local_points, weights, mode, return_transpose=False
):
    """Return mapper from grid coeffs to point evaluations."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import aslinearoperator

    local_space = space.localised_space
    grid = local_space.grid
    number_of_local_points = local_points.shape[1]
    nshape_funs = space.number_of_shape_functions
    number_of_vertices = number_of_local_points * grid.number_of_elements

    global_dofs = []
    node_dofs = []
    values = []

    if mode == "source":
        attr = "source_ids"
    elif mode == "target":
        attr = "target_ids"
    else:
        raise ValueError("'mode' must be one of 'source' or 'target'.")

    for key in nodes:
        vertex_ids = getattr(nodes[key], attr)
        associated_elements = set(
            [vertex // number_of_local_points for vertex in vertex_ids]
        )
        # Evaluate basis on the elements
        basis_values = {}
        for elem in associated_elements:
            # Spaces are scalar, so can use 2nd and 2rd component of eval
            basis_values[elem] = (
                local_space.evaluate(elem, local_points)[0, :, :]
                * weights
                * grid.integration_elements[elem]
            )

        # Now fill up the matrix elements.
        for vertex in vertex_ids:
            elem = vertex // number_of_local_points
            local_point_index = vertex % number_of_local_points
            global_dofs.extend(local_space.local2global[elem, :])
            node_dofs.extend(nshape_funs * [vertex])
            values.extend(basis_values[elem][:, local_point_index])

    if return_transpose:
        transform = coo_matrix(
            (values, (global_dofs, node_dofs)),
            shape=(local_space.global_dof_count, number_of_vertices),
        )

        return aslinearoperator(space.map_to_localised_space.T) @ aslinearoperator(
            transform
        )
    else:
        transform = coo_matrix(
            (values, (node_dofs, global_dofs)),
            shape=(number_of_vertices, local_space.global_dof_count),
        )
        return aslinearoperator(transform) @ aslinearoperator(
            space.map_to_localised_space
        )


def grid_to_points(grid, support_elements, local_points):
    """
    Map a grid to an array of points.

    Returns a (N, 3) point array that stores the global vertices
    associated with the local points in each triangle.
    Points are stored in consecutive order for each element 
    in the support_elements list. Hence, the returned array is of the form
    [ v_1^1, v_2^1, ..., v_M^1, v_1^2, v_2^2, ...], where
    v_i^j is the ith point in the jth element in
    the support_elements list.

    Parameters
    ----------
    grid : Grid
        A Bempp Grid object.
    support_elements : List of Integers
        A list of integers storing the indices of
        the support elements in the grid.
    local_points : np.ndarray
        (2, M) array of local coordinates.
    """
    number_of_elements = len(support_elements)
    number_of_points = local_points.shape[1]

    points = _np.empty((number_of_points * number_of_elements, 3), dtype=_np.float64)

    for index, elem in enumerate(support_elements):
        points[number_of_points * index : number_of_points * (1 + index)] = (
            grid.get_element(elem).geometry.local2global(local_points).T
        )

    return points
