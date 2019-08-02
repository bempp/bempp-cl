"""Definition of a piecewise constant function space."""

# pylint: disable=unused-argument

import numpy as _np
import numba as _numba

from bempp.api.space.space import (_FunctionSpace, _SpaceData,
        _process_segments)


class P1DiscontinuousFunctionSpace(_FunctionSpace):
    """A space of discontinuous piecewise linear functions."""

    def __init__(self, grid, support_elements=None, segments=None, swapped_normals=None):
        """Initialize with a given grid."""
        from scipy.sparse import identity

        shapeset = "p1_discontinuous"

        support, normal_multipliers = _process_segments(grid, support_elements, segments, swapped_normals)
        elements_in_support = _np.flatnonzero(support)

        number_of_support_elements = len(elements_in_support)
        global_dof_count = 3 * number_of_support_elements

        local2global = _np.zeros((grid.number_of_elements, 3), dtype='uint32')
        local_multipliers = _np.zeros((grid.number_of_elements, 3), dtype='uint32')

        local2global[support] = _np.arange(3 * number_of_support_elements).reshape(
                number_of_support_elements, 3)

        local_multipliers[support] = 1

        codomain_dimension = 1
        order = 1
        identifier = "p1_discontinuous"

        localised_space = self
        requires_dof_transformation = False

        is_barycentric = False
        barycentric_representation = None

        space_data = _SpaceData(
            grid,
            codomain_dimension,
            order,
            shapeset,
            local2global,
            local_multipliers,
            identifier,
            support,
            localised_space,
            normal_multipliers,
            identity(global_dof_count, dtype='float64'),
            requires_dof_transformation,
            is_barycentric,
            barycentric_representation
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
            normal_multipliers
        )

    def surface_gradient(self, element, local_coordinates):
        """Return the surface gradient."""
        return _numba_surface_gradient(
            element.index,
            self.shapeset.gradient,
            local_coordinates,
            self.grid.data,
            self.local_multipliers,
            normal_multipliers
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
