"""Definition of Maxwell spaces."""

import numpy as _np
import numba as _numba


def rwg0_function_space(
    grid, support_elements=None, segments=None, swapped_normals=None, include_boundary_dofs=False
):
    """Define a space of RWG functions of order 0"""
    from .space import SpaceBuilder, _process_segments, invert_local2global
    from bempp.api.utils.helpers import serialise_list_of_lists

    support, normal_multipliers = _process_segments(
        grid, support_elements, segments, swapped_normals
    )

    edge_neighbors, edge_neighbors_ptr = serialise_list_of_lists(grid.edge_neighbors)

    global_dof_count, support, local2global, local_multipliers = _compute_rwg0_space_data(
        support,
        edge_neighbors,
        edge_neighbors_ptr,
        grid.element_edges,
        grid.number_of_elements,
        grid.number_of_edges,
        include_boundary_dofs,
    )

    global2local = invert_local2global(local2global, local_multipliers)

    return (
        SpaceBuilder(grid)
        .set_codomain_dimension(3)
        .set_support(support)
        .set_normal_multipliers(normal_multipliers)
        .set_order(0)
        .set_is_localised(False)
        .set_shapeset("rwg0")
        .set_identifier("rwg0")
        .set_local2global(local2global)
        .set_global2local(global2local)
        .set_local_multipliers(local_multipliers)
        .set_barycentric_representation(rwg0_barycentric_function_space)
        .set_numba_surface_gradient(_numba_rwg0_evaluate)
        .build()
    )

def rwg0_barycentric_function_space(coarse_space):
    """Define a space of RWG functions of order 0 over a barycentric grid."""
    from .space import SpaceBuilder, _process_segments, invert_local2global
    from bempp.api.utils.helpers import serialise_list_of_lists
    from scipy.sparse import coo_matrix


    number_of_support_elements = coarse_space.number_of_support_elements
    bary_grid_number_of_elements = 6 * coarse_space.grid.number_of_elements

    bary_support_elements = 6 * _np.repeat(
        coarse_space.support_elements, 6
    ) + _np.tile(_np.arange(6), number_of_support_elements)

    bary_support_size = len(bary_support_elements)

    support = _np.zeros(6 * coarse_space.grid.number_of_elements, dtype=_np.bool)
    support[bary_support_elements] = True

    normal_multipliers = _np.repeat(coarse_space.normal_multipliers, 6)
    

    local_coords = _np.array(
        [[0, 0], [0.5, 0], [1, 0], [0.5, 0.5], [0, 1], [0, 0.5], [1.0 / 3, 1.0 / 3]]
    ).T

    coeffs = (
        _np.array(
            [
                [1, -1.0 / 3, 0],
                [-1.0 / 3, 1, 0],
                [0, 1.0 / 3, -1.0 / 6],
                [0, 0, 1.0 / 6],
                [0, 0, 1.0 / 6],
                [1.0 / 3, 0, -1.0 / 6],
            ]
        ),
        _np.array(
            [
                [0, 1.0 / 3, -1.0 / 6],
                [0, 0, 1.0 / 6],
                [0, 0, 1.0 / 6],
                [1.0 / 3, 0, -1.0 / 6],
                [1, -1.0 / 3, 0],
                [-1.0 / 3, 1, 0],
            ]
        ),
        _np.array(
            [
                [0, 0, 1.0 / 6],
                [1.0 / 3, 0, -1.0 / 6],
                [1, -1.0 / 3, 0],
                [-1.0 / 3, 1, 0],
                [0, 1.0 / 3, -1.0 / 6],
                [0, 0, 1.0 / 6],
            ]
        ),
    )

    coarse_dofs, bary_dofs, values = generate_rwg0_map(
        coarse_space.grid.data, coarse_space.support_elements, local_coords, coeffs
    )

    local2global = _np.zeros((bary_grid_number_of_elements, 3), dtype='uint32')
    local_multipliers = _np.zeros((bary_grid_number_of_elements, 3), dtype='uint32')

    local2global[support] = _np.arange(3 * bary_support_size).reshape(
            bary_support_size, 3)

    local_multipliers[support] = 1
    global2local = invert_local2global(local2global, local_multipliers)

    transform = coo_matrix(
        (values, (bary_dofs, coarse_dofs)),
        shape=(3 * bary_support_size, 3 * number_of_support_elements),
        dtype=_np.float64,
    ).tocsr()

    dof_transformation = transform @ coarse_space.map_to_localised_space

    return (
        SpaceBuilder(coarse_space.grid.barycentric_refinement)
        .set_codomain_dimension(3)
        .set_support(support)
        .set_normal_multipliers(normal_multipliers)
        .set_order(0)
        .set_is_localised(True)
        .set_shapeset("rwg0")
        .set_identifier("rwg0")
        .set_local2global(local2global)
        .set_global2local(global2local)
        .set_local_multipliers(local_multipliers)
        .set_dof_transformation(dof_transformation)
        .set_numba_surface_gradient(_numba_rwg0_evaluate)
        .build()
    )

def snc0_function_space(
    grid, support_elements=None, segments=None, swapped_normals=None, include_boundary_dofs=False
):
    """Define a space of SNC functions of order 0"""
    from .space import SpaceBuilder, _process_segments, invert_local2global
    from bempp.api.utils.helpers import serialise_list_of_lists

    support, normal_multipliers = _process_segments(
        grid, support_elements, segments, swapped_normals
    )

    edge_neighbors, edge_neighbors_ptr = serialise_list_of_lists(grid.edge_neighbors)

    global_dof_count, support, local2global, local_multipliers = _compute_rwg0_space_data(
        support,
        edge_neighbors,
        edge_neighbors_ptr,
        grid.element_edges,
        grid.number_of_elements,
        grid.number_of_edges,
        include_boundary_dofs,
    )

    global2local = invert_local2global(local2global, local_multipliers)

    return (
        SpaceBuilder(grid)
        .set_codomain_dimension(3)
        .set_support(support)
        .set_normal_multipliers(normal_multipliers)
        .set_order(0)
        .set_is_localised(False)
        .set_shapeset("rwg0")
        .set_identifier("snc0")
        .set_local2global(local2global)
        .set_global2local(global2local)
        .set_local_multipliers(local_multipliers)
        .set_numba_surface_gradient(_numba_snc0_evaluate)
        .build()
    )

@_numba.njit(cache=True)
def _compute_rwg0_space_data(
    support,
    edge_neighbors,
    edge_neighbors_ptr,
    element_edges,
    number_of_elements,
    number_of_edges,
    include_boundary_dofs,
):
    """Compute the local2global map for the space."""

    local2global_map = _np.zeros((number_of_elements, 3), dtype=_np.uint32)
    local_multipliers = _np.zeros((number_of_elements, 3), dtype=_np.float64)
    edge_dofs = -_np.ones(number_of_edges, dtype=_np.int32)

    support_elements = _np.flatnonzero(support)

    delete_from_support = []

    count = 0
    for element_index in support_elements:
        dofmap = -_np.ones(3, dtype=_np.int32)
        for local_index in range(3):
            edge_index = element_edges[local_index, element_index]
            current_neighbors = edge_neighbors[
                edge_neighbors_ptr[edge_index] : edge_neighbors_ptr[1 + edge_index]
            ]
            support_neighbors = [n for n in current_neighbors if support[n]]

            if len(support_neighbors) == 1:
                other = -1  # There is no other neighbor
            else:
                other = (
                    support_neighbors[1]
                    if element_index == support_neighbors[0]
                    else support_neighbors[0]
                )

            if other == -1:
                # We are at the boundary.
                if not include_boundary_dofs:
                    local_multipliers[element_index, local_index] = 0
                else:
                    local_multipliers[element_index, local_index] = 1
                    dofmap[local_index] = count
                    count += 1
            else:
                # Assign 1 or -1 depending on element index
                local_multipliers[element_index, local_index] = (
                    1 if element_index == min(support_neighbors) else -1
                )
                if edge_dofs[edge_index] == -1:
                    edge_dofs[edge_index] = count
                    count += 1
                dofmap[local_index] = edge_dofs[edge_index]

        # Check if no dof was assigned to element. In that case the element
        # needs to be deleted from the support.
        all_not_assigned = True
        for dof in dofmap:
            if dof != -1:
                all_not_assigned = False

        if all_not_assigned:
            delete_from_support.append(element_index)
            local_multipliers[element_index, :] = 0
            local2global_map[element_index, :] = 0
        else:
            # For every zero local multiplier assign an existing global dof
            # in this element. This does not change the result as zero multipliers
            # do not contribute. But it allows us not to have to distinguish between
            # existing and non existing dofs later on.
            first_nonzero = 0
            for local_index in range(3):
                if local_multipliers[element_index, local_index] != 0:
                    first_nonzero = local_index
                    break

            for local_index in range(3):
                if local_multipliers[element_index, local_index] == 0:
                    dofmap[local_index] = first_nonzero
            local2global_map[element_index, :] = dofmap

    for elem in delete_from_support:
        support[elem] = False

    return count, support, local2global_map, local_multipliers

@_numba.njit(cache=True)
def generate_rwg0_map(grid_data, support_elements, local_coords, coeffs):
    """Actually generate the sparse matrix data."""

    number_of_elements = len(support_elements)

    coarse_dofs = _np.empty(3 * 18 * number_of_elements, dtype=_np.uint32)
    bary_dofs = _np.empty(3 * 18 * number_of_elements, dtype=_np.uint32)
    values = _np.empty(3 * 18 * number_of_elements, dtype=_np.float64)

    # Iterate through the global dofs and fill up the
    # corresponding coefficients.

    count = 0

    for index, elem_index in enumerate(support_elements):
        # Compute all the local vertices
        local_vertices = grid_data.local2global(elem_index, local_coords)
        l1 = _np.linalg.norm(local_vertices[:, 6] - local_vertices[:, 4])
        l2 = _np.linalg.norm(local_vertices[:, 6] - local_vertices[:, 3])
        l3 = _np.linalg.norm(local_vertices[:, 6] - local_vertices[:, 5])
        l4 = _np.linalg.norm(local_vertices[:, 6] - local_vertices[:, 2])
        l5 = _np.linalg.norm(local_vertices[:, 6] - local_vertices[:, 1])
        l6 = _np.linalg.norm(local_vertices[:, 6] - local_vertices[:, 0])
        le1 = _np.linalg.norm(local_vertices[:, 2] - local_vertices[:, 0])
        le2 = _np.linalg.norm(local_vertices[:, 4] - local_vertices[:, 0])
        le3 = _np.linalg.norm(local_vertices[:, 4] - local_vertices[:, 2])

        outer_edges = [le1, le2, le3]

        dof_mult = _np.array(
            [
                [le1, l6, l5],
                [l4, le1, l5],
                [le3, l4, l2],
                [l1, le3, l2],
                [le2, l1, l3],
                [l6, le2, l3],
            ]
        )

        # Assign the dofs for the six barycentric elements

        bary_elements = _np.arange(6) + 6 * index
        for local_dof in range(3):
            coarse_dof = 3 * index + local_dof
            bary_coeffs = coeffs[local_dof]
            dof_coeffs = bary_coeffs * outer_edges[local_dof] / dof_mult
            coarse_dofs[count : count + 18] = coarse_dof
            bary_dofs[count : count + 18] = _np.arange(
                3 * bary_elements[0], 3 * bary_elements[0] + 18
            )
            values[count : count + 18] = dof_coeffs.ravel()
            count += 18
    return coarse_dofs, bary_dofs, values

@_numba.njit()
def _numba_rwg0_evaluate(
    element_index,
    shapeset_evaluate,
    local_coordinates,
    grid_data,
    local_multipliers,
    normal_multipliers,
):
    """Evaluate the basis on an element."""
    reference_values = shapeset_evaluate(local_coordinates)
    npoints = local_coordinates.shape[1]
    result = _np.empty((3, 3, npoints), dtype=_np.float64)

    edge_lengths = _np.empty(3, dtype=_np.float64)
    edge_lengths[0] = _np.linalg.norm(
        grid_data.vertices[:, grid_data.elements[0, element_index]]
        - grid_data.vertices[:, grid_data.elements[1, element_index]]
    )
    edge_lengths[1] = _np.linalg.norm(
        grid_data.vertices[:, grid_data.elements[2, element_index]]
        - grid_data.vertices[:, grid_data.elements[0, element_index]]
    )
    edge_lengths[2] = _np.linalg.norm(
        grid_data.vertices[:, grid_data.elements[1, element_index]]
        - grid_data.vertices[:, grid_data.elements[2, element_index]]
    )

    for index in range(3):
        result[:, index, :] = (
            local_multipliers[element_index, index]
            * edge_lengths[index]
            / grid_data.integration_elements[element_index]
            * grid_data.jacobians[element_index].dot(reference_values[:, index, :])
        )
    return result


@_numba.njit()
def _numba_snc0_evaluate(
    element_index, shapeset_evaluate, local_coordinates, grid_data, local_multipliers, normal_multipliers
):
    """Evaluate the basis on an element."""
    reference_values = shapeset_evaluate(local_coordinates)
    npoints = local_coordinates.shape[1]
    result = _np.empty((3, 3, npoints), dtype=_np.float64)
    tmp = _np.empty((3, 3, npoints), dtype=_np.float64)
    normal = grid_data.normals[element_index] * normal_multipliers[element_index]


    edge_lengths = _np.empty(3, dtype=_np.float64)
    edge_lengths[0] = _np.linalg.norm(
        grid_data.vertices[:, grid_data.elements[0, element_index]]
        - grid_data.vertices[:, grid_data.elements[1, element_index]]
    )
    edge_lengths[1] = _np.linalg.norm(
        grid_data.vertices[:, grid_data.elements[2, element_index]]
        - grid_data.vertices[:, grid_data.elements[0, element_index]]
    )
    edge_lengths[2] = _np.linalg.norm(
        grid_data.vertices[:, grid_data.elements[1, element_index]]
        - grid_data.vertices[:, grid_data.elements[2, element_index]]
    )

    for index in range(3):
        tmp[:, index, :] = (
            local_multipliers[element_index, index]
            * edge_lengths[index]
            / grid_data.integration_elements[element_index]
            * grid_data.jacobians[element_index].dot(reference_values[:, index, :])
        )

    result[0, :, :] = normal[1] * tmp[2, :, :] - normal[2] * tmp[1, :, :]
    result[1, :, :] = normal[2] * tmp[0, :, :] - normal[0] * tmp[2, :, :]
    result[2, :, :] = normal[0] * tmp[1, :, :] - normal[1] * tmp[0, :, :]

    return result

