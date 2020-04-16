import numpy as _np
import numba as _numba

M_INV_4PI = 1.0 / (4 * _np.pi)


@_numba.jit(
    nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False
)
def laplace_kernel(target_point, source_point, kernel_parameters, dtype):
    """Evaluate the laplace kernel."""

    diff = target_point - source_point

    dist = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]
    dist = _np.sqrt(dist)

    result = _np.empty(4, dtype=dtype)
    green = dtype.type(M_INV_4PI) / dist

    result[0] = green
    result[1:4] = -diff * green / (dist * dist)

    return result


@_numba.jit(
    nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False
)
def evaluate_interactions_on_singular_elements(
    grid_data, charges, local_points, kernel_function, kernel_parameters, dtype
):
    """Evaluate charges on the singular elements."""

    nelements = grid_data.elements.shape[1]
    npoints = local_points.shape[1]
    neighbor_indices = grid_data.element_neighbor_indices
    neighbor_indexptr = grid_data.element_neighbor_indexptr

    result = _np.zeros((4, nelements), dtype=dtype)

    for element_index in _numba.prange(nelements):
        element_points = grid_data.local2global(element_index, local_points)
        nneighbors = neighbor_indexptr[1 + element_index] - neighbor_indexptr[element_index]
        for n_index in range(nneighbors):
            neighbor = neighbor_indices[neighbor_indexptr[element_index] + n_index]
            neighbor_points = grid_data.local2global(neighbor, local_points)
            for element_point_index in range(npoints):
                for neighbor_point_index in range(npoints):
                    if neighbor == element_index and element_point_index == neighbor_point_index:
                        continue
                    result[:, element_index] += (
                        kernel_function(
                            element_points[:, element_point_index],
                            neighbor_points[:, neighbor_point_index],
                            kernel_parameters,
                            dtype
                        )
                        * charges[neighbor * npoints + neighbor_point_index]
                    )
    return result.sum(axis=1)


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
