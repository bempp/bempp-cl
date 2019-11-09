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


def map_space_to_points(space, local_points, weights, mode, return_transpose=False):
    """Return mapper from grid coeffs to point evaluations."""
    import bempp.api
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import aslinearoperator

    grid = space.grid
    number_of_local_points = local_points.shape[1]
    nshape_funs = space.number_of_shape_functions
    number_of_vertices = number_of_local_points * grid.number_of_elements

    data, global_indices, vertex_indices = map_space_to_points_impl(
        grid.data,
        space.localised_space.local2global,
        space.localised_space.local_multipliers,
        space.localised_space.normal_multipliers,
        space.support_elements,
        space.numba_evaluate,
        space.shapeset.evaluate,
        local_points,
        weights,
        space.number_of_shape_functions,
    )

    if return_transpose:
        transform = coo_matrix(
            (data, (global_indices, vertex_indices)),
            shape=(space.localised_space.global_dof_count, number_of_vertices),
        )

        return aslinearoperator(space.map_to_localised_space.T) @ aslinearoperator(
            transform
        )
    else:
        transform = coo_matrix(
            (data, (vertex_indices, global_indices)),
            shape=(number_of_vertices, space.localised_space.global_dof_count),
        )
        return aslinearoperator(transform) @ aslinearoperator(
            space.map_to_localised_space
        )


# def map_space_to_points(
# nodes, space, local_points, weights, mode, return_transpose=False
# ):
# """Return mapper from grid coeffs to point evaluations."""
# import bempp.api
# from scipy.sparse import coo_matrix
# from scipy.sparse.linalg import aslinearoperator

# local_space = space.localised_space
# grid = local_space.grid
# number_of_local_points = local_points.shape[1]
# nshape_funs = space.number_of_shape_functions
# number_of_vertices = number_of_local_points * grid.number_of_elements

# global_dofs = []
# node_dofs = []
# values = []

# if mode == "source":
# attr = "source_ids"
# elif mode == "target":
# attr = "target_ids"
# else:
# raise ValueError("'mode' must be one of 'source' or 'target'.")

# for key in nodes:
# vertex_ids = getattr(nodes[key], attr)
# associated_elements = set(
# [vertex // number_of_local_points for vertex in vertex_ids]
# )
# # Evaluate basis on the elements
# basis_values = {}
# for elem in associated_elements:
# # Spaces are scalar, so can use 2nd and 2rd component of eval
# basis_values[elem] = (
# local_space.evaluate(elem, local_points)[0, :, :]
# * weights
# * grid.integration_elements[elem]
# )

# # Now fill up the matrix elements.
# for vertex in vertex_ids:
# elem = vertex // number_of_local_points
# local_point_index = vertex % number_of_local_points
# global_dofs.extend(local_space.local2global[elem, :])
# node_dofs.extend(nshape_funs * [vertex])
# values.extend(basis_values[elem][:, local_point_index])

# if return_transpose:
# transform = coo_matrix(
# (values, (global_dofs, node_dofs)),
# shape=(local_space.global_dof_count, number_of_vertices),
# )

# return aslinearoperator(space.map_to_localised_space.T) @ aslinearoperator(
# transform
# )
# else:
# transform = coo_matrix(
# (values, (node_dofs, global_dofs)),
# shape=(number_of_vertices, local_space.global_dof_count),
# )
# return aslinearoperator(transform) @ aslinearoperator(
# space.map_to_localised_space
# )


@_numba.njit
def map_space_to_points_impl(
    grid_data,
    local2global,
    local_multipliers,
    normal_multipliers,
    support_elements,
    numba_evaluate,
    shape_fun,
    local_points,
    weights,
    number_of_shape_functions,
):
    """Numba accelerated computational parts for point map."""

    number_of_local_points = local_points.shape[1]
    number_of_support_elements = len(support_elements)

    nlocal = number_of_local_points * number_of_shape_functions

    data = _np.empty(nlocal * number_of_support_elements, dtype=_np.float64)
    global_indices = _np.empty(nlocal * number_of_support_elements, dtype=_np.int64)
    vertex_indices = _np.empty(nlocal * number_of_support_elements, dtype=_np.int64)

    for elem in support_elements:
        basis_values = (
            numba_evaluate(
                elem,
                shape_fun,
                local_points,
                grid_data,
                local_multipliers,
                normal_multipliers,
            )[0, :, :]
            * weights
            * grid_data.integration_elements[elem]
        )
        data[elem * nlocal : (1 + elem) * nlocal] = basis_values.ravel()
        for index in range(number_of_shape_functions):
            vertex_indices[
                elem * nlocal
                + index * number_of_local_points : elem * nlocal
                + (1 + index) * number_of_local_points
            ] = _np.arange(
                elem * number_of_local_points, (1 + elem) * number_of_local_points
            )
        global_indices[elem * nlocal : (1 + elem) * nlocal] = _np.repeat(
            local2global[elem, :], number_of_local_points
        )

    return (data, global_indices, vertex_indices)


def grid_to_points(grid, local_points):
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
    local_points : np.ndarray
        (2, M) array of local coordinates.
    """
    # number_of_elements = len(support_elements)
    # number_of_points = local_points.shape[1]

    # points = _np.empty((number_of_points * number_of_elements, 3), dtype=_np.float64)

    # for index, elem in enumerate(support_elements):
        # points[number_of_points * index : number_of_points * (1 + index)] = (
            # grid.get_element(elem).geometry.local2global(local_points).T
        # )

    return grid_to_points_impl(grid.data, local_points)

@_numba.njit
def grid_to_points_impl(
        grid_data, local_points):
    """Numba implementation for grid_to_points."""
    number_of_elements = grid_data.elements.shape[1]
    number_of_points = local_points.shape[1]

    points = _np.empty((number_of_points * number_of_elements, 3), dtype=_np.float64)

    for elem in range(number_of_elements):
        points[number_of_points * elem : number_of_points * (1 + elem), :] = (
                _np.expand_dims(grid_data.vertices[:, grid_data.elements[0, elem]], 1) +
                grid_data.jacobians[elem].dot(local_points)).T
    return points

