"""Definition of a piecewise constant function space."""

# pylint: disable=unused-argument

import numpy as _np
import numba as _numba

from bempp.api.space.space import _FunctionSpace, _SpaceData


class P1DiscontinuousFunctionSpace(_FunctionSpace):
    """A space of discontinuous piecewise linear functions."""

    def __init__(self, grid):
        """Initialize with a given grid."""
        from scipy.sparse import identity

        shapeset = "p1_discontinuous"

        number_of_elements = grid.number_of_elements
        global_dof_count = 3 * number_of_elements

        local2global_map = _np.array(
            [
                [3 * index, 3 * index + 1, 3 * index + 2]
                for index in range(number_of_elements)
            ],
            dtype="uint32",
        )

        local_multipliers = _np.ones((number_of_elements, 3), dtype="float64")

        support = _np.full(grid.number_of_elements, True, dtype=bool)

        codomain_dimension = 1
        order = 1
        identifier = "p1_discontinuous"

        localised_space = self

        color_map = _np.zeros(number_of_elements, dtype="uint32")

        map_to_localised_space = identity(
            3 * grid.number_of_elements, dtype="float64", format="csr"
        )

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
