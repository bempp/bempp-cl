"""Useful FMM Utility routines."""

import numpy as _np
from scipy.sparse.linalg import LinearOperator as _LinearOperator

from bempp.api.utils.timing import timeit as _timeit


def space_to_points_matrix(space, octree, quadrature_order):
    """
    Create map from space coefficients to quadrature points.
    
    Applying the returned matrix to a vector of grid function
    coefficients returns the evaluation of the function values
    on the quadrature points of each grid element multiplied by 
    the associated quadrature weights. For the grid elements it
    uses the ordering imposed by the permutation array of the
    octree.
    """
    from bempp.api.integration.triangle_gauss import rule
    from scipy.sparse import coo_matrix

    points, weights = rule(quadrature_order)
    npoints = len(weights)

    ndofs = space.global_dof_count
    nelem = space.grid.number_of_elements

    rows = []
    cols = []
    values = []

    for index, elem_index in enumerate(octree.element_permutation):
        global_dofs = space.local2global[elem_index]
        multipliers = space.local_multipliers[elem_index]
        element = space.grid.get_element(elem_index)

        element_values = (
            space.evaluate(element, points)[0]
            * weights
            * space.grid.integration_elements[elem_index]
        )

        for local_index, global_dof in enumerate(global_dofs):
            if global_dof == -1:
                continue
            rows += range(index * npoints, (1 + index) * npoints)
            cols += npoints * [global_dof]
            values += (element_values[local_index] * multipliers[local_index]).tolist()

    return coo_matrix((values, (rows, cols)), shape=(nelem * npoints, ndofs)).tocsr()
