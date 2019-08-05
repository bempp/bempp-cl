"""Definition of the lowest order RWG space."""

# pylint: disable=unused-argument

import numpy as _np
import numba as _numba

from .space import _FunctionSpace, _SpaceData, _process_segments


class Rwg0LocalisedFunctionSpace(_FunctionSpace):
    """A space of RWG functions."""

    def __init__(
        self,
        grid,
        support_elements=None,
        segments=None,
        swapped_normals=None,
    ):
        """Initialize with a given grid."""
        from .localised_space import LocalisedFunctionSpace

        from scipy.sparse import coo_matrix
        from scipy.sparse import identity

        shapeset = "rwg0"

        support, normal_mult = _process_segments(
            grid, support_elements, segments, swapped_normals
        )
        elements_in_support = _np.flatnonzero(support)

        number_of_support_elements = len(elements_in_support)
        global_dof_count = 3 * number_of_support_elements

        local2global = _np.zeros((grid.number_of_elements, 3), dtype='uint32')
        local_multipliers = _np.zeros((grid.number_of_elements, 3), dtype='uint32')

        local2global[support] = _np.arange(3 * number_of_support_elements).reshape(
                number_of_support_elements, 3)

        local_multipliers[support] = 1
        
        codomain_dimension = 3
        order = 0
        identifier = "rwg0-localised"

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
            normal_mult,
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
        raise NotImplementedError


@_numba.njit
def _numba_evaluate(
    element_index, shapeset_evaluate, local_coordinates, grid_data, local_multipliers, normal_multipliers
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

