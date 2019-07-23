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
    ):
        """Definition of RWG spaces over barycentric refinements."""
        from bempp.api.space.rwg0_space import Rwg0FunctionSpace
        from bempp.api.space.rwg0_localised_space import Rwg0LocalisedFunctionSpace
        from numba.typed import List
        from scipy.sparse import coo_matrix

        @_numba.njit
        def generate_map(grid_data, support_elements, local_coords, coeffs):
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
                le2 = _np.linalg.norm(local_vertices[:, 4] - local_vertices[:, 2])
                le3 = _np.linalg.norm(local_vertices[:, 4] - local_vertices[:, 0])

                outer_edges = [le1, le2, le3]

                dof_mult = _np.array(
                    [
                        [1, l6, l5],
                        [l4, 1, l5],
                        [1, l4, l2],
                        [l1, 1, l2],
                        [1, l1, l3],
                        [l6, 1, l3],
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

        coarse_space = Rwg0FunctionSpace(
            grid, support_elements, segments, swapped_normals
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

        coarse_dofs, bary_dofs, values = generate_map(
            grid.data, coarse_space.support_elements, local_coords, coeffs
        )

        transform = coo_matrix(
            (values, (bary_dofs, coarse_dofs)),
            shape=(super().global_dof_count, 3 * number_of_support_elements),
            dtype=_np.float64,
        ).tocsr()

        self._dof_transform = transform
        self._identifier = "b-rwg0"
