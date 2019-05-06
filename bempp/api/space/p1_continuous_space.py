"""Definition of a piecewise constant function space."""

# pylint: disable=unused-argument

import numpy as _np
import numba as _numba

from .space import (_FunctionSpace, _SpaceData,
        _process_segments)


class P1ContinuousFunctionSpace(_FunctionSpace):
    """A space of continuous piecewise linear functions."""

    def __init__(self, grid, support_elements=None, segments=None, swapped_normals=None,
            include_boundary_dofs=False, ensure_global_continuity=False):
        """Initialize with a given grid."""
        from .localised_space import LocalisedFunctionSpace
        from scipy.sparse import coo_matrix

        shapeset = "p1_discontinuous"

        number_of_elements = grid.number_of_elements

        support, normal_multipliers = _process_segments(grid, support_elements, segments, swapped_normals)
        elements_in_support = _np.flatnonzero(support)

        local2global = -_np.ones((number_of_elements, 3), dtype=_np.int)
        
        vertex_is_dof = _np.zeros(grid.number_of_vertices, dtype=_np.bool)
        delete_from_support = []

        for element_index in elements_in_support:
            for local_index in range(3):
                vertex = grid.elements[local_index, element_index]
                element_neighbors = grid.element_neighbors[element_index]
                neighbors = [elem for elem in element_neighbors if vertex in grid.elements[:, elem]]
                non_support_neighbors = [n for n in neighbors if support[n] == False]
                if include_boundary_dofs or len(non_support_neighbors) == 0:
                    # Just add dof
                    local2global[element_index, local_index] = vertex
                    vertex_is_dof[vertex] = True
                    element_has_dof = True
                if (len(non_support_neighbors) > 0 and 
                        ensure_global_continuity and include_boundary_dofs):
                    for en in non_support_neighbors:
                        other_local_index = _np.flatnonzero(
                                grid.elements[:, en] == vertex)[0]
                        local2global[en, other_local_index] = vertex

        # Now enumerate the vertices that are dof to obtain the final dofmap.

        support_final = _np.zeros(number_of_elements, dtype=_np.bool)
        local2global_final = _np.zeros((number_of_elements, 3), dtype='uint32')
        local_multipliers = _np.zeros((number_of_elements, 3), dtype='float64')

        dofs = -_np.ones(grid.number_of_elements)
        used_dofs = _np.flatnonzero(vertex_is_dof)
        global_dof_count = len(used_dofs)
        dofs[used_dofs] = _np.arange(global_dof_count)


        for element_index in range(grid.number_of_elements):
            used_indices = _np.flatnonzero(local2global[element_index] != -1)
            if len(used_indices) > 0:
                mapped_dofs = dofs[local2global[element_index, used_indices]]
                support_final[element_index] = True
                local2global_final[element_index, used_indices] = mapped_dofs
                unused_indices = _np.flatnonzero(local2global[element_index] == -1)
                local2global_final[element_index, unused_indices] = min(mapped_dofs)
                local_multipliers[element_index, used_indices] = 1


        support_size = len(_np.flatnonzero(support_final))

        codomain_dimension = 1
        order = 1
        identifier = "p1_continuous"

        localised_space = LocalisedFunctionSpace(
            grid,
            codomain_dimension,
            order,
            shapeset,
            3,
            identifier,
            support_final,
            normal_multipliers,
            self.numba_evaluate,
            None,
        )

        color_map = _color_grid(grid, support)

        map_to_localised_space = coo_matrix(
            (
                local_multipliers[support_final].ravel(),
                (_np.arange(3 * support_size), local2global_final[support_final].ravel()),
            ),
            shape=(3 * support_size, global_dof_count),
            dtype="float64",
        ).tocsr()

        space_data = _SpaceData(
            grid,
            codomain_dimension,
            global_dof_count,
            order,
            shapeset,
            local2global_final,
            local_multipliers,
            identifier,
            support,
            localised_space,
            color_map,
            map_to_localised_space,
            normal_multipliers
        )

        super().__init__(space_data)

    @property
    def numba_evaluate(self):
        """Return numba method that evaluates the basis."""
        return _numba_evaluate

    @property
    def numba_surface_gradient(self):
        """Return numba method that evaluates the surface gradient."""
        return _numba_surface_gradient

    def evaluate(self, element, local_coordinates):
        """Evaluate the basis on an element."""
        return _numba_evaluate(
            element.index,
            self.shapeset.evaluate,
            local_coordinates,
            self.grid.data,
            self.local_multipliers,
            self.normal_multipliers
        )

    def surface_gradient(self, element, local_coordinates):
        """Return the surface gradient."""
        return _numba_surface_gradient(
            element.index,
            self.shapeset.gradient,
            local_coordinates,
            self.grid.data,
            self.local_multipliers,
            self.normal_multipliers
        )


@_numba.njit
def _numba_evaluate(
    element_index, shapeset_evaluate, local_coordinates, grid_data, local_multipliers, normal_multipliers
):
    """Evaluate the basis on an element."""
    return shapeset_evaluate(local_coordinates)


@_numba.njit
def _numba_surface_gradient(
    element_index, shapeset_gradient, local_coordinates, grid_data, local_multipliers, normal_multipliers
):
    """Evaluate the surface gradient."""
    reference_values = shapeset_gradient(local_coordinates)
    result = _np.empty((1, 3, 3, local_coordinates.shape[1]), dtype=_np.float64)
    for index in range(3):
        result[0, :, index, :] = grid_data.jac_inv_trans[element_index].dot(
            reference_values[0, :, index, :]
        )
    return result


def _color_grid(grid, support):
    """
    Find and return a coloring of the grid.

    The coloring is defined so that two elements are neighbours
    if they share a common vertex. This ensures that all elements
    of the same color do not share any vertices. The coloring
    algorithm is a simple greedy algorithm.
    """
    support_elements = _np.flatnonzero(support)
    number_of_elements = len(support_elements)
    colors = number_of_elements * [-1]

    for element in grid.entity_iterator(0):
        element_index = element.index
        if not support[element.index]: continue
        neighbor_colors = [colors[e.index] for e in element.neighbors if support[e.index]]
        colors[element.index] = next(
            color for color in range(number_of_elements) if color not in neighbor_colors
        )
    return _np.array(colors, dtype="uint32")
