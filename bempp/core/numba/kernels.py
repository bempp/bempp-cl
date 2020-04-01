from bempp.api.utils.helpers import numba_decorate as _numba_decorate

import numba as _numba
import numpy as _np

M_INV_4PI = 1.0 / (4 * _np.pi)


def select_numba_kernels(operator_descriptor, mode="regular"):
    """Select the Numba kernels."""
    assembly_functions_singular = {
        "default_scalar": default_scalar_singular_kernel,
    }
    assembly_functions_regular = {}
    kernel_functions_regular = {
        "laplace_single_layer": laplace_single_layer_regular,
    }
    kernel_functions_singular = {
        "laplace_single_layer": laplace_single_layer_singular,
    }

    if mode == "regular":
        return (
            assembly_functions_regular[operator_descriptor.assembly_type],
            kernel_functions_regular[operator_descriptor.kernel_type],
        )
    elif mode == "singular":
        return (
            assembly_functions_singular[operator_descriptor.assembly_type],
            kernel_functions_singular[operator_descriptor.kernel_type],
        )
    else:
        raise ValueError("mode must be one of 'singular' or 'regular'")


@_numba_decorate
def get_normals(grid_data, nrepetitions, elements, multipliers):
    """Get normals to be repeated n times per element."""
    output = _np.empty((3, nrepetitions * len(elements)), dtype=grid_data.normals.dtype)
    for index, element in enumerate(elements):
        for dim in range(3):
            for n in range(nrepetitions):
                output[dim, nrepetitions * index + n] = (
                    grid_data.normals[element, dim] * multipliers[element]
                )

    return output


@_numba_decorate
def laplace_single_layer_regular(
    test_point, trial_points, test_normal, trial_normals, kernel_parameters
):
    """Laplace single layer for regular kernels."""
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
def laplace_single_layer_singular(
    test_points, trial_points, test_normals, trial_normals, kernel_parameters
):
    """Laplace single layer for singular kernels."""
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    output = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            output[j] += (trial_points[i, j] - test_points[i, j]) ** 2
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
    kernel_parameters,
):
    """Singular evaluator."""

    dtype = grid_data.vertices.dtype
    nindices = len(test_indices)
    result = _np.zeros(nshape_test * nshape_trial * nindices, dtype=dtype)

    for index in _numba.prange(nindices):
        test_element = test_indices[index]
        trial_element = trial_indices[index]
        test_offset = test_offsets[index]
        trial_offset = trial_offsets[index]
        weights_offset = weights_offsets[index]
        npoints = number_of_quad_points[index]
        test_local_points = test_points[:, test_offset : test_offset + npoints]
        trial_local_points = trial_points[:, trial_offset : trial_offset + npoints]
        test_global_points = grid_data.local2global(test_element, test_local_points)
        trial_global_points = grid_data.local2global(trial_element, trial_local_points)
        test_fun_values = test_shapeset(
            test_points[:, test_offset : test_offset + npoints],
        )
        trial_fun_values = trial_shapeset(
            trial_points[:, trial_offset : trial_offset + npoints],
        )
        test_normals = get_normals(
            grid_data, npoints, [test_element], trial_normal_multipliers
        )
        trial_normals = get_normals(
            grid_data, npoints, [trial_element], trial_normal_multipliers
        )
        kernel_values = kernel_evaluator(
            test_global_points,
            trial_global_points,
            test_normals,
            trial_normals,
            kernel_parameters,
        )
        for test_fun_index in range(nshape_test):
            for trial_fun_index in range(nshape_trial):
                for point_index in range(npoints):
                    result[
                        nshape_trial * nshape_test * index
                        + test_fun_index * nshape_trial
                        + trial_fun_index
                    ] += (
                        kernel_values[point_index]
                        * quad_weights[weights_offset + point_index]
                        * test_fun_values[0, test_fun_index, point_index]
                        * trial_fun_values[0, trial_fun_index, point_index]
                    )
                result[
                    nshape_trial * nshape_test * index
                    + test_fun_index * nshape_trial
                    + trial_fun_index
                ] *= (
                    grid_data.integration_elements[test_element]
                    * grid_data.integration_elements[trial_element]
                )

    return result
