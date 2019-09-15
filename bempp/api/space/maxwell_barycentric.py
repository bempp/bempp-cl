"""
Definition of Maxwell spaces on barycentric refinements.
"""
import numba as _numba
import numpy as _np
from .rwg0_localised_space import (
    Rwg0LocalisedFunctionSpace as _Rwg0LocalisedFunctionSpace,
)


class Rwg0BarycentricSpace(_Rwg0LocalisedFunctionSpace):
    """RWG space on barycentric grid."""

    def __init__(
        self,
        grid,
        support_elements=None,
        segments=None,
        swapped_normals=None,
        include_boundary_dofs=False,
        coarse_space=None,
    ):
        """Definition of RWG spaces over barycentric refinements."""
        import bempp.api
        from bempp.api.space.rwg0_space import Rwg0FunctionSpace
        from bempp.api.space.rwg0_localised_space import Rwg0LocalisedFunctionSpace
        from numba.typed import List
        from scipy.sparse import coo_matrix

        if coarse_space is None:
            coarse_space = Rwg0FunctionSpace(
                grid, support_elements, segments, swapped_normals, include_boundary_dofs
            )

        number_of_support_elements = coarse_space.number_of_support_elements

        bary_support_elements = 6 * _np.repeat(
            coarse_space.support_elements, 6
        ) + _np.tile(_np.arange(6), number_of_support_elements)

        super().__init__(
            grid.barycentric_refinement,
            support_elements=bary_support_elements,
            swapped_normals=swapped_normals,
        )

        local_coords = _np.array(
            [[0, 0], [0.5, 0], [1, 0], [0.5, 0.5], [0, 1], [0, 0.5], [1.0 / 3, 1.0 / 3]]
        ).T

        coeffs = (
            _np.array(
                [
                    [1, -1.0 / 3, 0],
                    [-1.0 / 3, 1, 0],
                    [0, 1.0 / 3, -1.0 / 6],
                    [0, 0, 1.0 / 6],
                    [0, 0, 1.0 / 6],
                    [1.0 / 3, 0, -1.0 / 6],
                ]
            ),
            _np.array(
                [
                    [0, 1.0 / 3, -1.0 / 6],
                    [0, 0, 1.0 / 6],
                    [0, 0, 1.0 / 6],
                    [1.0 / 3, 0, -1.0 / 6],
                    [1, -1.0 / 3, 0],
                    [-1.0 / 3, 1, 0],
                ]
            ),
            _np.array(
                [
                    [0, 0, 1.0 / 6],
                    [1.0 / 3, 0, -1.0 / 6],
                    [1, -1.0 / 3, 0],
                    [-1.0 / 3, 1, 0],
                    [0, 1.0 / 3, -1.0 / 6],
                    [0, 0, 1.0 / 6],
                ]
            ),
        )

        coarse_dofs, bary_dofs, values = generate_rwg0_map(
            grid.data, coarse_space.support_elements, local_coords, coeffs
        )

        transform = coo_matrix(
            (values, (bary_dofs, coarse_dofs)),
            shape=(super().global_dof_count, 3 * number_of_support_elements),
            dtype=_np.float64,
        ).tocsr()

        self._dof_transformation = transform @ coarse_space.map_to_localised_space
        self._identifier = "rwg0"
        self._requires_dof_transformation = True
        self._is_barycentric = True
        self._barycentric_representation = lambda: self


class BCSpace(_Rwg0LocalisedFunctionSpace):
    """RWG space on barycentric grid."""

    def __init__(
        self,
        grid,
        support_elements=None,
        segments=None,
        swapped_normals=None,
        include_boundary_dofs=False,
        coarse_space=None,
    ):
        """Definition of RWG spaces over barycentric refinements."""
        import bempp.api
        from bempp.api.space.rwg0_space import Rwg0FunctionSpace
        from bempp.api.space.rwg0_localised_space import Rwg0LocalisedFunctionSpace
        from numba.typed import List
        from scipy.sparse import coo_matrix

        if coarse_space is None:
            coarse_space = Rwg0FunctionSpace(
                grid,
                support_elements,
                segments,
                swapped_normals,
                include_boundary_dofs=False,
            )

        number_of_support_elements = coarse_space.number_of_support_elements

        bary_support_elements = 6 * _np.repeat(
            coarse_space.support_elements, 6
        ) + _np.tile(_np.arange(6), number_of_support_elements)

        super().__init__(
            grid.barycentric_refinement,
            support_elements=bary_support_elements,
            swapped_normals=swapped_normals,
        )

        coarse_dofs, bary_dofs, values = generate_bc_map(
            grid.data, coarse_space.support_elements, local_coords, coeffs
        )

        transform = coo_matrix(
            (values, (bary_dofs, coarse_dofs)),
            shape=(super().global_dof_count, coarse_space.global_dof_count),
            dtype=_np.float64,
        ).tocsr()

        self._dof_transformation = transform
        self._identifier = "bc"
        self._requires_dof_transformation = True
        self._is_barycentric = True
        self._barycentric_representation = lambda: self


def generate_bc_map(
    coarse_grid_data,
    bary_grid_data,
    global_dof_count,
    global2local,
    local_multipliers,
    bary_vertex_to_edge,
    bary_local2global,
):
    """Generate the BC map."""

    def find_position(value, array):
        """
        Find first occurence of element in array.
        
        Return -1 if value not found 
        
        """
        for ind, current_value in enumerate(array):
            if value == current_value:
                return ind
        return -1


    def process_vertex(
        vertex_index,
        reference_element,
        bary_vertex_to_edge,
        bary_grid_data,
        bary_local2global,
        global_dof_index,
        coarse_dofs,
        bary_dofs,
        values,
    ):
        """Assign coefficients based on vertex index and element index."""
        # Get barycentric element index adjacent to vertex
        bary_element = 6 * reference_element

        # Find the corresponding index in elements adjacent to that vertex

        for ind, elem in enumerate(bary_vertex_to_edge[vertex_index]):
            if bary_element == elem[0]:
                break

        # Now get all the relevant edges starting to count above
        # ind

        nc = len(bary_vertex_to_edge[vertex_index])
        vertex_edges = []
        for index in range(nc):
            elem_edge_pair = bary_vertex_to_edge[vertex_index][(index + ind) % nc]
            for n in range(1, 3):
                vertex_edges.append((elem_edge_pair[0], elem_edge_pair[n]))

        # We do not want the reference edge part of this list
        vertex_edges.pop(0)

        # We now have a list of edges associated with the vertex counting from edge
        # after the reference edge onwards in anti-clockwise order. We can now
        # assign the coefficients

        sign = 1.0
        count = 1.0
        for edge in vertex_edges:
            elem_index, local_edge_index = edge[:]
            edge_length = _compute_edge_length(
                bary_grid_data, elem_index, local_edge_index
            )
            bary_dofs.append(bary_local2global[elem_index, local_id])
            coarse_dofs.append(global_dof_index)
            values.append(-sign * (2 * nc - count) / (2 * edge_length * nc))
            sign *= -1

    coarse_dofs = []
    bary_dofs = []
    values = []

    for global_dof_index in range(global_dof_count):
        local_dofs = global2local[global_dof_index]
        edge_index = coarse_grid_data.element_edges[local_dofs[0][1], local_dofs[0][0]]
        if local_multipliers[local_dofs[0][0], local_dofs[0][1]] > 0:
            lower = local_dofs[0]
            upper = local_dofs[1]
        else:
            lower = local_dofs[1]
            upper = local_dofs[0]
        vertex1, vertex2 = coarse_grid_data.edges[:, edge_index]
        # Re-order the vertices so that they appear in anti-clockwise
        # order.
        for local_index, vertex_index in enumerate(coarse_grid.elements[:, upper]):
            if vertex_index == vertex1:
                break
        if vertex2 == coarse_grid_data.elements[(local_index - 1) % 3]:
            vertex1, vertex2 = vertex2, vertex1

        process_vertex(
            vertex1,
            upper,
            bary_vertex_to_edge,
            bary_grid_data,
            bary_local2global,
            global_dof_index,
            coarse_dofs,
            bary_dofs,
            values,
        )

        process_vertex(
            vertex2,
            lower,
            bary_vertex_to_edge,
            bary_grid_data,
            bary_local2global,
            global_dof_index,
            coarse_dofs,
            bary_dofs,
            values,
        )

        # Now process the tangential rwgs close to the reference edge

        # Get the local indices of vertex1 and vertex2 in upper and lower



def _compute_edge_length(grid_data, elem_index, local_edge_index):
    """Compute the length of a given edge in a given element."""

    edge_index = grid_data.element_edges[local_edge_index, elem_index]
    vertices = grid_data.vertices[:, grid_data.edges[:, edge_index]]
    return _np.linalg.norm(vertices[:, 0] - vertices[:, 1])


@_numba.njit(cache=True)
def generate_rwg0_map(grid_data, support_elements, local_coords, coeffs):
    """Actually generate the sparse matrix data."""

    number_of_elements = len(support_elements)

    coarse_dofs = _np.empty(3 * 18 * number_of_elements, dtype=_np.uint32)
    bary_dofs = _np.empty(3 * 18 * number_of_elements, dtype=_np.uint32)
    values = _np.empty(3 * 18 * number_of_elements, dtype=_np.float64)

    # Iterate through the global dofs and fill up the
    # corresponding coefficients.

    count = 0

    for index, elem_index in enumerate(support_elements):
        # Compute all the local vertices
        local_vertices = grid_data.local2global(elem_index, local_coords)
        l1 = _np.linalg.norm(local_vertices[:, 6] - local_vertices[:, 4])
        l2 = _np.linalg.norm(local_vertices[:, 6] - local_vertices[:, 3])
        l3 = _np.linalg.norm(local_vertices[:, 6] - local_vertices[:, 5])
        l4 = _np.linalg.norm(local_vertices[:, 6] - local_vertices[:, 2])
        l5 = _np.linalg.norm(local_vertices[:, 6] - local_vertices[:, 1])
        l6 = _np.linalg.norm(local_vertices[:, 6] - local_vertices[:, 0])
        le1 = _np.linalg.norm(local_vertices[:, 2] - local_vertices[:, 0])
        le2 = _np.linalg.norm(local_vertices[:, 4] - local_vertices[:, 0])
        le3 = _np.linalg.norm(local_vertices[:, 4] - local_vertices[:, 2])

        outer_edges = [le1, le2, le3]

        dof_mult = _np.array(
            [
                [le1, l6, l5],
                [l4, le1, l5],
                [le3, l4, l2],
                [l1, le3, l2],
                [le2, l1, l3],
                [l6, le2, l3],
            ]
        )

        # Assign the dofs for the six barycentric elements

        bary_elements = _np.arange(6) + 6 * index
        for local_dof in range(3):
            coarse_dof = 3 * index + local_dof
            bary_coeffs = coeffs[local_dof]
            dof_coeffs = bary_coeffs * outer_edges[local_dof] / dof_mult
            coarse_dofs[count : count + 18] = coarse_dof
            bary_dofs[count : count + 18] = _np.arange(
                3 * bary_elements[0], 3 * bary_elements[0] + 18
            )
            values[count : count + 18] = dof_coeffs.ravel()
            count += 18
    return coarse_dofs, bary_dofs, values
