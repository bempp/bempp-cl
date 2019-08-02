"""Definition of a generic localised space."""

# pylint: disable=unused-argument

import numpy as _np
import numba as _numba

from .space import _FunctionSpace, _SpaceData


class LocalisedFunctionSpace(_FunctionSpace):
    """Generic definition of a localised space."""

    def __init__(self, grid, codomain_dimension, order, 
            shapeset, local_size, identifier, support, normal_multipliers, 
            numba_evaluate, numba_surface_gradient):
        """Initialize with a given grid."""

        from scipy.sparse import identity

        number_of_elements = grid.number_of_elements
        support_size = _np.count_nonzero(support)

        support_size = _np.count_nonzero(support)
        local2global_map = _np.zeros((number_of_elements, local_size), dtype="uint32")
        local2global_map[support] = _np.arange(
                local_size * support_size, dtype="uint32").reshape(
                        (support_size, 3)
        )

        local_multipliers = _np.zeros((number_of_elements, local_size), dtype="float64")
        local_multipliers[support] = 1

        global_dof_count = local_size * support_size

        localised_space = self
        requires_dof_transformation = False

        is_barycentric = False
        barycentric_representation = False

        localised_space_data = _SpaceData(
                grid,
                codomain_dimension,
                order,
                shapeset,
                local2global_map,
                local_multipliers,
                identifier + "_localised",
                support,
                self,
                normal_multipliers,
                identity(global_dof_count, dtype='float64'),
                requires_dof_transformation,
                is_barycentric,
                barycentric_representation
                )


        self._numba_evaluate = numba_evaluate
        self._numba_surface_gradient = numba_surface_gradient

        super().__init__(localised_space_data)

    @property
    def numba_evaluate(self):
        """Return numba method that evaluates the basis."""
        return self.__numba_evaluate

    @property
    def numba_surface_gradient(self):
        """Return numba method that evaluates the surface gradient."""
        if self._numba_surface_gradient is not None:
            return self._numba_surface_gradient
        else:
            raise NotImplementedError

    def evaluate(self, element, local_coordinates):
        """Evaluate the basis on an element."""
        return self._numba_evaluate(
            element.index,
            self.shapeset.evaluate,
            local_coordinates,
            self.grid.data,
            self.local_multipliers,
            self.normal_multipliers,
        )

    def surface_gradient(self, element, local_coordinates):
        """Return the surface gradient."""
        return self._numba_surface_gradient(
                element.index,
                self.shapeset.evaluate,
                local_coordinates,
                self.grid.data,
                self.local_multipliers,
                self.normal_multipliers)


