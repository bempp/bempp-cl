"""Actual implementation of Numba assemblers."""
import numpy as _np


def singular_assembler(
    device_interface,
    operator_descriptor,
    grid,
    domain,
    dual_to_range,
    test_points,
    trial_points,
    quad_weights,
    test_elements,
    trial_elements,
    test_offsets,
    trial_offsets,
    weights_offsets,
    number_of_quad_points,
    kernel_options,
    result,
):
    """Numba assembler for the singular part of integral operators."""
    from bempp.api.utils.helpers import get_type
    from bempp.core.numba_kernels import select_numba_kernels

    numba_assembly_function, numba_kernel_function = select_numba_kernels(
            operator_descriptor, mode="singular"
            )


    precision = operator_descriptor.precision
    dtype = get_type(precision).real

    numba_assembly_function(
        grid.data(precision),
        test_points,
        trial_points,
        quad_weights,
        test_elements,
        trial_elements,
        test_offsets,
        trial_offsets,
        weights_offsets,
        number_of_quad_points,
        dual_to_range.normal_multipliers,
        domain.normal_multipliers,
        dual_to_range.number_of_shape_functions,
        domain.number_of_shape_functions,
        dual_to_range.shapeset.evaluate,
        domain.shapeset.evaluate,
        numba_kernel_function,
        _np.array(kernel_options, dtype=dtype),
        result
    )


