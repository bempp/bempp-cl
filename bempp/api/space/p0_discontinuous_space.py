"""Definition of a piecewise constant function space."""

# pylint: disable=unused-argument

import numpy as _np
import numba as _numba

from .space import (_FunctionSpace, _SpaceData,
        _process_segments)


class P0DiscontinuousFunctionSpace(_FunctionSpace):
    """A space of piecewise constant functions."""

    def __init__(self, grid, support_elements=None, segments=None, swapped_normals=None):
        """Initialize with a given grid."""
        from scipy.sparse import identity

        shapeset = "p0_discontinuous"

        number_of_elements = grid.number_of_elements

        support, normal_multipliers = _process_segments(grid, support_elements, segments, swapped_normals)

        elements_in_support = _np.flatnonzero(support)
        support_size = len(elements_in_support)

        local2global_map = _np.zeros((number_of_elements, 1), dtype="uint32")

        local2global_map[support] = _np.expand_dims(
            _np.arange(support_size, dtype="uint32"), 1
        )

        local_multipliers = _np.zeros((number_of_elements, 1), dtype="float64")
        local_multipliers[support] = 1

        codomain_dimension = 1
        order = 0
        identifier = "p0_discontinuous"

        localised_space = self
        requires_dof_transformation = False

        is_barycentric = False
        barycentric_representation = None

        space_data = _SpaceData(
            grid,
            codomain_dimension,
            order,
            shapeset,
            local2global_map,
            local_multipliers,
            identifier,
            support,
            localised_space,
            normal_multipliers,
            identity(support_size, dtype='float64'),
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

    @property
    def collocation_points(self):
        """Return collocation points."""
        return _np.array([[1./3], [1./3]])


    def evaluate(self, element_index, local_coordinates):
        """Evaluate the basis on an element."""
        return _numba_evaluate(
            element_index,
            self.shapeset.evaluate,
            local_coordinates,
            self.grid.data,
            self.local_multipliers,
            self.normal_multipliers
        )

    def surface_gradient(self, element_index, local_coordinates):
        """Return the surface gradient."""
        return _numba_surface_gradient(
            element_index,
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
    return _np.zeros((1, 3, 1, local_coordinates.shape[1]), dtype=_np.float64)
