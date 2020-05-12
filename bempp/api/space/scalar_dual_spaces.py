import numpy as _np

from .scalar_spaces import (p1_continuous_function_space, p0_discontinuous_function_space,
                            _numba_p0_surface_gradient, _numba_p1_surface_gradient)


def dual0_function_space(grid, support_elements=None, segments=None, swapped_normals=None):
    """Define a space of DP0 functions on the dual grid."""
    from .space import SpaceBuilder, invert_local2global
    from scipy.sparse import coo_matrix

    coarse_space = p1_continuous_function_space(
        grid, support_elements, segments, swapped_normals
    )

    number_of_support_elements = coarse_space.number_of_support_elements

    bary_support_elements = 6 * _np.repeat(coarse_space.support_elements, 6) + _np.tile(
        _np.arange(6), number_of_support_elements
    )

    bary_grid = grid.barycentric_refinement
    bary_support_size = len(bary_support_elements)

    support = _np.zeros(bary_grid.number_of_elements, dtype=_np.bool)
    support[bary_support_elements] = True

    local2global = _np.zeros((bary_grid.number_of_elements, 1), dtype="uint32")
    local_multipliers = _np.zeros((bary_grid.number_of_elements, 1), dtype="uint32")

    local2global[support] = _np.arange(bary_support_size).reshape(
        bary_support_size, 1
    )

    local_multipliers[support] = 1
    global2local = invert_local2global(local2global, local_multipliers)

    coarse_dofs, bary_dofs, values = generate_dual0_map(
        grid.data,
        bary_grid.data,
        coarse_space.global_dof_count,
        coarse_space.global2local
    )

    dof_transformation = coo_matrix(
        (values, (bary_dofs, coarse_dofs)),
        shape=(bary_support_size, coarse_space.global_dof_count),
        dtype=_np.float64,
    ).tocsr()

    collocation_points = _np.array([[1.0 / 3], [1.0 / 3]])

    return (
        SpaceBuilder(bary_grid)
        .set_codomain_dimension(1)
        .set_support(support)
        .set_order(0)
        .set_is_localised(True)
        .set_is_barycentric(True)
        .set_shapeset("p0_discontinuous")
        .set_identifier("p0_discontinuous")
        .set_local2global(local2global)
        .set_global2local(global2local)
        .set_local_multipliers(local_multipliers)
        .set_collocation_points(collocation_points)
        .set_dof_transformation(dof_transformation)
        .set_numba_surface_gradient(_numba_p0_surface_gradient)
        .build()
    )


def generate_dual0_map(
    coarse_grid_data,
    bary_grid_data,
    global_dof_count,
    global2local
):
    """Generate the DUAL0 map."""

    bary_dofs = []
    coarse_dofs = []

    for global_dof_index in range(global_dof_count):
        local_dofs = global2local[global_dof_index]

        for face, vertex in local_dofs:
            coarse_dofs.append(global_dof_index)
            coarse_dofs.append(global_dof_index)
            bary_dofs.append(6 * face + (2 * vertex - 1) % 6)
            bary_dofs.append(6 * face + 2 * vertex)

    nentries = len(bary_dofs)

    np_coarse_dofs = _np.zeros(nentries, dtype=_np.uint32)
    np_coarse_dofs[:] = coarse_dofs

    np_bary_dofs = _np.zeros(nentries, dtype=_np.uint32)
    np_bary_dofs[:] = bary_dofs

    np_values = _np.ones(nentries, dtype=_np.float64)

    return np_coarse_dofs, np_bary_dofs, np_values


def dual1_function_space(grid, support_elements=None, segments=None, swapped_normals=None):
    """Define a space of DP1 functions on the dual grid."""
    from .space import SpaceBuilder, invert_local2global
    from scipy.sparse import coo_matrix

    if segments is not None:
        raise NotImplementedError()
    if support_elements is not None:
        raise NotImplementedError()
    if swapped_normals is not None:
        raise NotImplementedError()

    coarse_space = p0_discontinuous_function_space(
        grid, support_elements, segments, swapped_normals
    )

    bary_grid = grid.barycentric_refinement

    number_of_support_elements = coarse_space.number_of_support_elements

    bary_support_elements = 6 * _np.repeat(coarse_space.support_elements, 6) + _np.tile(
        _np.arange(6), number_of_support_elements
    )

    bary_support_size = len(bary_support_elements)

    support = _np.zeros(bary_grid.number_of_elements, dtype=_np.bool)
    support[bary_support_elements] = True

    local2global = _np.zeros((bary_grid.number_of_elements, 3), dtype="uint32")
    local_multipliers = _np.zeros((bary_grid.number_of_elements, 3), dtype="uint32")

    local2global[support] = _np.arange(3 * bary_support_size).reshape(
        bary_support_size, 3
    )

    local_multipliers[support] = 1
    global2local = invert_local2global(local2global, local_multipliers)

    faces_by_vertex = [[] for i in grid.entity_iterator(2)]
    faces_by_edge = [[] for i in grid.entity_iterator(1)]

    for index, element in enumerate(grid.entity_iterator(0)):
        for vertex in element.sub_entity_iterator(2):
            faces_by_vertex[vertex.index].append(index)
        for edge in element.sub_entity_iterator(1):
            faces_by_edge[edge.index].append(index)

    num_entries = (2 * sum(len(i) ** 2 for i in faces_by_edge)
                   + 2 * sum(len(i) ** 2 for i in faces_by_vertex)
                   + 6 * grid.entity_count(0))

    values = _np.empty(num_entries, dtype=_np.uint32)
    bary_dofs = _np.empty(num_entries, dtype=_np.uint32)
    coarse_dofs = _np.empty(num_entries, dtype=_np.float64)

    bary_dof = 0
    entry = 0
    for index, element in enumerate(grid.entity_iterator(0)):
        for n in [1, 5, 7, 11, 13, 17]:
            bary_dofs[entry] = bary_dof + n
            coarse_dofs[entry] = index
            values[entry] = 1
            entry += 1
        for vertex, dofs in zip(element.sub_entity_iterator(2),
                                [[0, 3], [6, 9], [12, 15]]):
            for coarse_dof in faces_by_vertex[vertex.index]:
                for n in dofs:
                    bary_dofs[entry] = bary_dof + n
                    coarse_dofs[entry] = coarse_dof
                    values[entry] = 1 / len(faces_by_vertex[vertex.index])
                    entry += 1
        for edge, dofs in zip(element.sub_entity_iterator(1),
                              [[4, 8], [2, 16], [10, 14]]):
            for coarse_dof in faces_by_edge[edge.index]:
                for n in dofs:
                    bary_dofs[entry] = bary_dof + n
                    coarse_dofs[entry] = coarse_dof
                    values[entry] = 0.5
                    entry += 1
        bary_dof += 18

    assert entry == num_entries

    dof_transformation = coo_matrix(
        (values, (bary_dofs, coarse_dofs)),
        shape=(3 * bary_support_size, coarse_space.global_dof_count),
        dtype=_np.float64,
    ).tocsr()

    return (
        SpaceBuilder(bary_grid)
        .set_codomain_dimension(1)
        .set_support(support)
        .set_order(1)
        .set_is_localised(True)
        .set_is_barycentric(True)
        .set_shapeset("p1_discontinuous")
        .set_identifier("p1_discontinuous")
        .set_local2global(local2global)
        .set_global2local(global2local)
        .set_local_multipliers(local_multipliers)
        .set_dof_transformation(dof_transformation)
        .set_numba_surface_gradient(_numba_p1_surface_gradient)
        .build()
    )
