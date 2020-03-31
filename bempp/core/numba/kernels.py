from bempp.api.utils.helpers import numba_decorate as _numba_decorate

import numba as _numba
import numpy as _np

M_INV_4PI = 1.0 / (4 * _np.pi)


def select_numba_kernels(operator_descriptor, mode="regular"):
    """Select the Numba kernels."""
    assembly_functions_singular = {
        "default_scalar": default_scalar_singular_kernel,
    }
    kernel_functions = {
        "laplace_single_layer": laplace_single_layer,
    }

    assembly_functions_regular = {}

    if mode == "regular":
        return (
            assembly_functions_regular[operator_descriptor.assembly_type],
            kernel_functions[operator_descriptor.kernel_type],
        )
    elif mode == "singular":
        return (
            assembly_functions_singular[operator_descriptor.assembly_type],
            kernel_functions[operator_descriptor.kernel_type],
        )
    else:
        raise ValueError("mode must be one of 'singular' or 'regular'")


@_numba_decorate
def get_normals(grid_data, nrepetitions, elements):
    """Get normals to be repeated n times per element."""
    output = _np.empty((3, nrepetitions * len(elements)), dtype=grid_data.normals.dtype)
    for index, element in enumerate(elements):
        for dim in range(3):
            for n in range(nrepetitions):
                output[dim, nrepetitions * index + n] = grid_data.normals[element, dim]

    return output


@_numba_decorate
def laplace_single_layer(
    test_point, trial_points, test_normal, trial_normals, kernel_parameters
):
    """Laplace single layer."""
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    output = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            output[j] += (trial_points[i, j] - test_point[i]) ** 2
    for j in range(npoints):
        output[j] = m_inv_4pi / _np.sqrt(output[j])
    return output


@_numba_decorate
def default_scalar_singular_kernel(
    grid_data,
    test_points,
    trial_points,
    quad_weights,
    test_indices,
    trial_indices,
    test_offsets,
    trial_offsets,
    weights_offsets,
    number_of_quad_points,
    test_normal_multipliers,
    trial_normal_multipliers,
    nshape_test,
    nshape_trial,
    test_shapeset,
    trial_shapeset,
    kernel_evaluator,
):
    """Singular evaluator."""

    dtype = grid_data.vertices.dtype
    nindices = len(test_indices)

    result = _np.empty(nshape_test * nshape_trial * nindices, dtype=dtype)

    return result

    # for index in _numba.prange(nindices):
    #     test_element = test_indices[index]
    #     trial_element = trial_indices[index]
    #     test_offset = test_offsets[index]
    #     trial_offset = trial_offsets[index]
    #     weights_offset = weights_offsets[index]
    #     npoints = number_of_quad_points[index]
    #     test_fun_values = test_shapeset(
    #         test_points[:, test_offset : test_offset + npoints],
    #     )
    #     trial_fun_values = trial_shapeset(
    #         test_points[:, test_offset : test_offset + npoints],
    #     )
    #     test_normal = grid_data.normals[test_element]
    #     trial_normal_elements = [trial_element]
    #     trial_normals = get_normals(grid_data, npoints, trial_normal_elements)
