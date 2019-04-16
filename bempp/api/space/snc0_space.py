"""Definition of the lowest order RWG space."""

# pylint: disable=unused-argument

import numpy as _np
import numba as _numba

from .space import _FunctionSpace, _SpaceData


class Snc0FunctionSpace(_FunctionSpace):
    """A space of RWG functions."""

    def __init__(self, grid):
        """Initialize with a given grid."""
        from .rwg0_localised_space import Rwg0LocalisedFunctionSpace

        from scipy.sparse import coo_matrix

        shapeset = "rwg0"
        number_of_elements = grid.number_of_elements

        local2global_map = _np.empty((number_of_elements, 3), dtype="uint32")

        local_multipliers = _np.empty((number_of_elements, 3), dtype="float64")

        edge_dofs = -_np.ones(grid.number_of_edges, dtype="int32")

        count = 0
        for element_index in range(number_of_elements):
            dofmap = -_np.ones(3, dtype="int32")
            for local_index in range(3):
                edge_index = grid.element_edges[local_index, element_index]
                edge_neighbors = grid.edge_neighbors[edge_index]
                if len(edge_neighbors) == 1:
                    # We are at the boundary and the dof should not be used.

                    # Make sure that the grid is consistent.
                    # The neighbor of the edge is really the element.
                    assert edge_neighbors[0] == element_index
                    local_multipliers[element_index, local_index] = 0
                else:
                    # Assign 1 or -1 depending on element index
                    local_multipliers[element_index, local_index] = (
                        1 if element_index == min(edge_neighbors) else -1
                    )
                    if edge_dofs[edge_index] == -1:
                        edge_dofs[edge_index] = count
                        count += 1
                    dofmap[local_index] = edge_dofs[edge_index]
            # For every zero local multiplier assign an existing global dof
            # in this element. This does not change the result as zero multipliers
            # do not contribute. But it allows us not to have to distinguish between
            # existing and non existing dofs later on.
            arg_zeros = _np.flatnonzero(local_multipliers[element_index] == 0)
            first_nonzero = _np.min(
                _np.flatnonzero(local_multipliers[element_index] != 0)
            )
            dofmap[arg_zeros] = dofmap[first_nonzero]
            local2global_map[element_index, :] = dofmap

        global_dof_count = count

        support = _np.full(number_of_elements, True, dtype=bool)

        codomain_dimension = 3
        order = 0
        identifier = "snc0"

        localised_space = Rwg0LocalisedFunctionSpace(grid)

        map_to_localised_space = coo_matrix(
            (
                local_multipliers.ravel(),
                (_np.arange(3 * number_of_elements), local2global_map.ravel()),
            ),
            shape=(3 * number_of_elements, global_dof_count),
            dtype="float64",
        ).tocsr()

        color_map = _color_grid(grid)

        space_data = _SpaceData(
            grid,
            codomain_dimension,
            global_dof_count,
            order,
            shapeset,
            local2global_map,
            local_multipliers,
            identifier,
            support,
            localised_space,
            color_map,
            map_to_localised_space,
        )

        super().__init__(space_data)

    @property
    def numba_evaluate(self):
        """Return numba method that evaluates the basis."""
        return _numba_evaluate

    @property
    def numba_surface_gradient(self):
        """Return numba method that evaluates the surface gradient."""
        raise NotImplementedError

    def evaluate(self, element, local_coordinates):
        """Evaluate the basis on an element."""
        return _numba_evaluate(
            element.index,
            self.shapeset.evaluate,
            local_coordinates,
            self.grid.data,
            self.local_multipliers,
        )

    def surface_gradient(self, element, local_coordinates):
        """Return the surface gradient."""
        raise NotImplementedError


@_numba.njit
def _numba_evaluate(
    element_index, shapeset_evaluate, local_coordinates, grid_data, local_multipliers
):
    """Evaluate the basis on an element."""
    reference_values = shapeset_evaluate(local_coordinates)
    npoints = local_coordinates.shape[1]
    result = _np.empty((3, 3, npoints), dtype=_np.float64)
    tmp = _np.empty((3, 3, npoints), dtype=_np.float64)
    normal = grid_data.normals[element_index]


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


def _color_grid(grid):
    """
    Find and return a coloring of the grid.

    The coloring is defined so that two elements are neighbours
    if they share a common edge. This ensures that all elements
    of the same color do not share any edges The coloring
    algorithm is a simple greedy algorithm.
    """
    number_of_elements = grid.number_of_elements
    colors = number_of_elements * [-1]

    for element in grid.entity_iterator(0):
        element_index = element.index
        neighbors = []
        for local_index in range(3):
            edge_neighbors = grid.edge_neighbors[
                grid.element_edges[local_index, element_index]
            ]
            if len(edge_neighbors) > 1:
                other = (
                    edge_neighbors[1]
                    if edge_neighbors[0] == element_index
                    else edge_neighbors[0]
                )
                neighbors.append(other)
        if neighbors is None:
            # No neighbors, can have color 0
            colors[element.index] = 0
        else:
            neighbor_colors = [colors[index] for index in neighbors]
            colors[element_index] = next(
                color
                for color in range(number_of_elements)
                if color not in neighbor_colors
            )
    return _np.array(colors, dtype="uint32")
