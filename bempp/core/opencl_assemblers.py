"""Actual implementation of OpenCL assemblers."""
import numpy as _np
import pyopencl as _cl

WORKGROUP_SIZE_GALERKIN = 16


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
    """OpenCL assembler for the singular part of integral operators."""
    from bempp.api.utils.helpers import get_type
    from bempp.core.opencl_kernels import get_kernel_from_operator_descriptor
    from bempp.core.opencl_kernels import default_context, default_device

    mf = _cl.mem_flags
    ctx = default_context()
    device = default_device()

    precision = operator_descriptor.precision
    dtype = get_type(precision).real

    options = {
        "WORKGROUP_SIZE": WORKGROUP_SIZE_GALERKIN,
        "TEST": dual_to_range.shapeset.identifier,
        "TRIAL": domain.shapeset.identifier,
        "NUMBER_OF_TEST_SHAPE_FUNCTIONS": dual_to_range.number_of_shape_functions,
        "NUMBER_OF_TRIAL_SHAPE_FUNCTIONS": domain.number_of_shape_functions,
    }

    if operator_descriptor.is_complex:
        options["COMPLEX_KERNEL"] = None

    kernel = get_kernel_from_operator_descriptor(
        operator_descriptor, options, "singular"
    )

    # Initialize OpenCL Buffers

    grid_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=grid.as_array.astype(dtype)
    )
    test_normals_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dual_to_range.normal_multipliers
    )
    trial_normals_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=domain.normal_multipliers
    )
    test_points_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_points
    )
    trial_points_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=trial_points
    )
    quad_weights_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=quad_weights
    )
    test_elements_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_elements
    )
    trial_elements_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=trial_elements
    )
    test_offsets_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_offsets
    )
    trial_offsets_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=trial_offsets
    )
    weights_offsets_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weights_offsets
    )

    local_quad_points = number_of_quad_points // WORKGROUP_SIZE_GALERKIN

    local_quad_points_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=local_quad_points
    )

    result_buffer = _cl.Buffer(ctx, mf.WRITE_ONLY, size=result.nbytes)

    if not kernel_options:
        kernel_options = [0.0]

    kernel_options_array = _np.array(kernel_options, dtype=dtype)

    kernel_options_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kernel_options_array
    )

    number_of_singular_indices = len(test_elements)

    with _cl.CommandQueue(ctx, device=device) as queue:
        kernel(
            queue,
            (number_of_singular_indices,),
            (WORKGROUP_SIZE_GALERKIN,),
            grid_buffer,
            test_normals_buffer,
            trial_normals_buffer,
            test_points_buffer,
            trial_points_buffer,
            quad_weights_buffer,
            test_elements_buffer,
            trial_elements_buffer,
            test_offsets_buffer,
            trial_offsets_buffer,
            weights_offsets_buffer,
            local_quad_points_buffer,
            result_buffer,
            kernel_options_buffer,
            g_times_l=True,
        )
        _cl.enqueue_copy(queue, result, result_buffer)


def dense_assembler(
    device_interface, operator_descriptor, domain, dual_to_range, parameters, result
):
    """Assemble dense with OpenCL."""
    from bempp.api.integration.triangle_gauss import rule
    from bempp.api.utils.helpers import get_type
    from bempp.core.opencl_kernels import get_kernel_from_operator_descriptor
    from bempp.core.opencl_kernels import default_context, default_device

    mf = _cl.mem_flags
    ctx = default_context()
    device = default_device()

    precision = operator_descriptor.precision
    dtype = get_type(precision).real

    quad_points, quad_weights = rule(parameters.quadrature.regular)

    test_indices, test_color_indexptr = dual_to_range.get_elements_by_color()
    trial_indices, trial_color_indexptr = domain.get_elements_by_color()

    options = {
            "NUMBER_OF_QUAD_POINTS": len(quad_weights),
            "TEST": dual_to_range.shapeset.identifier,
            "TRIAL": domain.shapeset.identifier,
            "TRIAL_NUMBER_OF_ELEMENTS": domain.number_of_support_elements,
            "TEST_NUMBER_OF_ELEMENTS": dual_to_range.number_of_support_elements,
            "NUMBER_OF_TEST_SHAPE_FUNCTIONS": dual_to_range.number_of_shape_functions,
            "NUMBER_OF_TRIAL_SHAPE_FUNCTIONS": domain.number_of_shape_functions,
            }

    if operator_descriptor.is_complex:
        options["COMPLEX_KERNEL"] = None

    main_kernel = get_kernel_from_operator_descriptor(
        operator_descriptor, options, "regular"
    )
    remainder_kernel = get_kernel_from_operator_descriptor(
        operator_descriptor, options, "regular", force_novec=True
    )

    test_indices_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_indices)
    trial_indices_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=trial_indices)

    test_normals_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dual_to_range.normal_multipliers
    )
    trial_normals_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=domain.normal_multipliers
    )
    test_grid_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dual_to_range.grid.as_array.astype(dtype)
    )
    trial_grid_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=domain.grid.as_array.astype(dtype)

    test_elements_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_TR, hostbuf=dual_to_range.grid.elements.ravel(order='F')

    trial_elements_buffer = _cl.Buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_TR, hostbuf=domain.grid.elements.ravel(order='F')

