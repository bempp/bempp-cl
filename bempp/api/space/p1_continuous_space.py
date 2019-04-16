"""Definition of a piecewise constant function space."""

# pylint: disable=unused-argument

import numpy as _np
import numba as _numba

from .space import _FunctionSpace, _SpaceData


class P1ContinuousFunctionSpace(_FunctionSpace):
    """A space of continuous piecewise linear functions."""

    def __init__(self, grid):
        """Initialize with a given grid."""

        from .p1_discontinuous_space import P1DiscontinuousFunctionSpace

        from scipy.sparse import coo_matrix

        shapeset = "p1_discontinuous"

        global_dof_count = grid.number_of_vertices
        number_of_elements = grid.number_of_elements

        local2global_map = _np.empty((number_of_elements, 3), dtype="uint32")

        for element in grid.entity_iterator(0):
            for local_index, vertex in enumerate(element.sub_entity_iterator(2)):
                local2global_map[element.index, local_index] = vertex.index

        local_multipliers = _np.ones((number_of_elements, 3), dtype="float64")

        support = _np.full(number_of_elements, True, dtype=bool)

        codomain_dimension = 1
        order = 1
        identifier = "p1_continuous"

        localised_space = P1DiscontinuousFunctionSpace(grid)

        color_map = _color_grid(grid)

        map_to_localised_space = coo_matrix(
            (
                _np.ones(3 * number_of_elements, dtype="float64"),
                (_np.arange(3 * number_of_elements), grid.elements.ravel(order="F")),
            ),
            shape=(3 * number_of_elements, global_dof_count),
            dtype="float64",
        ).tocsr()

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
        return _numba_surface_gradient

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
        return _numba_surface_gradient(
            element.index,
            self.shapeset.gradient,
            local_coordinates,
            self.grid.data,
            self.local_multipliers,
        )


@_numba.njit
def _numba_evaluate(
    element_index, shapeset_evaluate, local_coordinates, grid_data, local_multipliers
):
    """Evaluate the basis on an element."""
    return shapeset_evaluate(local_coordinates)


@_numba.njit
def _numba_surface_gradient(
    element_index, shapeset_gradient, local_coordinates, grid_data, local_multipliers
):
    """Evaluate the surface gradient."""
    reference_values = shapeset_gradient(local_coordinates)
    result = _np.empty((1, 3, 3, local_coordinates.shape[1]), dtype=_np.float64)
    for index in range(3):
        result[0, :, index, :] = grid_data.jac_inv_trans[element_index].dot(
            reference_values[0, :, index, :]
        )
    return result


def _color_grid(grid):
    """
    Find and return a coloring of the grid.

    The coloring is defined so that two elements are neighbours
    if they share a common vertex. This ensures that all elements
    of the same color do not share any vertices. The coloring
    algorithm is a simple greedy algorithm.
    """
    number_of_elements = grid.number_of_elements
    colors = number_of_elements * [-1]

    for element in grid.entity_iterator(0):
        neighbor_colors = [colors[e.index] for e in element.neighbors]
        colors[element.index] = next(
            color for color in range(number_of_elements) if color not in neighbor_colors
        )
    return _np.array(colors, dtype="uint32")
