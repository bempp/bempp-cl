"""Definition of the lowest order RWG space."""

# pylint: disable=unused-argument

import numpy as _np
import numba as _numba

from .space import _FunctionSpace, _SpaceData, _process_segments


class Rwg0FunctionSpace(_FunctionSpace):
    """A space of RWG functions."""

    def __init__(
        self,
        grid,
        support_elements=None,
        segments=None,
        swapped_normals=None,
        include_boundary_dofs=False,
    ):
        """Initialize with a given grid."""
        from bempp.api.space.maxwell_barycentric import Rwg0BarycentricSpace
        from .localised_space import LocalisedFunctionSpace
        from scipy.sparse import identity
        from scipy.sparse import coo_matrix
        import bempp.api

        shapeset = "rwg0"
        number_of_elements = grid.number_of_elements

        support, normal_mult = _process_segments(
            grid, support_elements, segments, swapped_normals
        )

        edge_neighbors = [pair for sublist in grid.edge_neighbors for pair in sublist]
        edge_neighbors_ptr = _np.empty(grid.number_of_edges + 1, dtype=_np.int32)
        count = 0
        for index, sublist in enumerate(grid.edge_neighbors):
            edge_neighbors_ptr[index] = count
            count += len(sublist)
        edge_neighbors_ptr[-1] = count

        global_dof_count, support, local2global_map, local_multipliers = _compute_space_data(
            support,
            edge_neighbors,
            edge_neighbors_ptr,
            grid.element_edges,
            grid.number_of_elements,
            grid.number_of_edges,
            include_boundary_dofs,
        )

        support_size = _np.count_nonzero(support)

        if support_size == 0:
            raise ValueError("The support of the function space is empty.")

        codomain_dimension = 3
        order = 0
        identifier = "rwg0"

        localised_space = LocalisedFunctionSpace(
            grid,
            codomain_dimension,
            order,
            shapeset,
            3,
            identifier,
            support,
            normal_mult,
            self.numba_evaluate,
            None,
        )

        requires_dof_transformation = False

        is_barycentric = False

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
            normal_mult,
            identity(global_dof_count, dtype="float64"),
            requires_dof_transformation,
            is_barycentric,
            None,
        )

        super().__init__(space_data)

        self._barycentric_representation = lambda: Rwg0BarycentricSpace(
            grid,
            support_elements=support_elements,
            segments=segments,
            swapped_normals=swapped_normals,
            include_boundary_dofs=include_boundary_dofs,
            coarse_space=self,
        )

    @property
    def numba_evaluate(self):
        """Return numba method that evaluates the basis."""
        return _numba_evaluate

    @property
    def numba_surface_gradient(self):
        """Return numba method that evaluates the surface gradient."""
        raise NotImplementedError


@_numba.njit(cache=True)
def _compute_space_data(
    support,
    edge_neighbors,
    edge_neighbors_ptr,
    element_edges,
    number_of_elements,
    number_of_edges,
    include_boundary_dofs,
):
    """Compute the local2global map for the space."""

    local2global_map = _np.zeros((number_of_elements, 3), dtype=_np.uint32)
    local_multipliers = _np.zeros((number_of_elements, 3), dtype=_np.float64)
    edge_dofs = -_np.ones(number_of_edges, dtype=_np.int32)

    support_elements = _np.flatnonzero(support)

    delete_from_support = []

    count = 0
    for element_index in support_elements:
        dofmap = -_np.ones(3, dtype=_np.int32)
        for local_index in range(3):
            edge_index = element_edges[local_index, element_index]
            current_neighbors = edge_neighbors[
                edge_neighbors_ptr[edge_index] : edge_neighbors_ptr[1 + edge_index]
            ]
            support_neighbors = [n for n in current_neighbors if support[n]]

            if len(support_neighbors) == 1:
                other = -1  # There is no other neighbor
            else:
                other = (
                    support_neighbors[1]
                    if element_index == support_neighbors[0]
                    else support_neighbors[0]
                )

            if other == -1:
                # We are at the boundary.
                if not include_boundary_dofs:
                    local_multipliers[element_index, local_index] = 0
                else:
                    local_multipliers[element_index, local_index] = 1
                    dofmap[local_index] = count
                    count += 1
            else:
                # Assign 1 or -1 depending on element index
                local_multipliers[element_index, local_index] = (
                    1 if element_index == min(support_neighbors) else -1
                )
                if edge_dofs[edge_index] == -1:
                    edge_dofs[edge_index] = count
                    count += 1
                dofmap[local_index] = edge_dofs[edge_index]

        # Check if no dof was assigned to element. In that case the element
        # needs to be deleted from the support.
        all_not_assigned = True
        for dof in dofmap:
            if dof != -1:
                all_not_assigned = False

        if all_not_assigned:
            delete_from_support.append(element_index)
            local_multipliers[element_index, :] = 0
            local2global_map[element_index, :] = 0
        else:
            # For every zero local multiplier assign an existing global dof
            # in this element. This does not change the result as zero multipliers
            # do not contribute. But it allows us not to have to distinguish between
            # existing and non existing dofs later on.
            first_nonzero = 0
            for local_index in range(3):
                if local_multipliers[element_index, local_index] != 0:
                    first_nonzero = local_index
                    break

            for local_index in range(3):
                if local_multipliers[element_index, local_index] == 0:
                    dofmap[local_index] = first_nonzero
            local2global_map[element_index, :] = dofmap

    for elem in delete_from_support:
        support[elem] = False

    return count, support, local2global_map, local_multipliers


@_numba.njit(cache=True)
def _numba_evaluate(
    element_index,
    shapeset_evaluate,
    local_coordinates,
    grid_data,
    local_multipliers,
    normal_multipliers,
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
