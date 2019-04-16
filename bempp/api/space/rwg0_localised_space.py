"""Definition of the localised lowest order RWG space."""

# pylint: disable=unused-argument

import numpy as _np
import numba as _numba

from .space import _FunctionSpace, _SpaceData


class Rwg0LocalisedFunctionSpace(_FunctionSpace):
    """A space of RWG functions."""

    def __init__(self, grid):
        """Initialize with a given grid."""

        from scipy.sparse import identity

        shapeset = 'rwg0'
        number_of_elements = grid.number_of_elements

        global_dof_count = 3 * number_of_elements

        local2global_map = _np.arange(3 * number_of_elements, dtype="uint32").reshape(
            (number_of_elements, 3)
        )

        local_multipliers = _np.ones((number_of_elements, 3), dtype="float64")

        support = _np.full(number_of_elements, True, dtype=bool)

        codomain_dimension = 3
        order = 0
        identifier = "rwg0_localised"

        localised_space = self

        map_to_localised_space = identity(
            3 * grid.number_of_elements, dtype="float64", format="csr"
        )

        color_map = _np.zeros(number_of_elements, dtype="uint32")

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
