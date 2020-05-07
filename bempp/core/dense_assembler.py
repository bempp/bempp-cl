"""Dense Assembly of integral operators."""
import numpy as _np

import bempp.core.cl_helpers as _cl_helpers
from bempp.api.assembly import assembler as _assembler
from bempp.helpers import timeit as _timeit


class DenseAssembler(_assembler.AssemblerBase):
    """Implementation of a dense assembler for integral operators."""

    # pylint: disable=useless-super-delegation
    def __init__(self, domain, dual_to_range, parameters=None):
        """Create a dense assembler instance."""
        super().__init__(domain, dual_to_range, parameters)

    def assemble(
        self, operator_descriptor, device_interface, precision, *args, **kwargs
    ):
        """Dense assembly of the integral operator."""
        from bempp.api.assembly.discrete_boundary_operator import DenseDiscreteBoundaryOperator
        from bempp.api.utils.helpers import promote_to_double_precision
        from .dense_assembly_helpers import choose_source_name

        # Check if we can use the simple assember

        mat = None

        source_name = choose_source_name(operator_descriptor.compute_kernel)

        if self.domain.requires_dof_transformation or self.dual_to_range.requires_dof_transformation:
            raise ValueError("Spaces that require dof transformations not supported for dense assembly.")

        mat = assemble_dense(
            self.domain,
            self.dual_to_range,
            self.parameters,
            operator_descriptor,
            source_name,
            device_interface,
            precision,
        )

        if self.parameters.assembly.always_promote_to_double:
            mat = promote_to_double_precision(mat)

        return DenseDiscreteBoundaryOperator(mat)


@_timeit
def assemble_dense(
    domain,
    dual_to_range,
    parameters,
    operator_descriptor,
    source_name,
    device_interface,
    precision,
):
    """
    Really assemble the operator.

    Assembles the complete operator (near-field and far-field)
    Returns a dense matrix.
    """
    from bempp.api.integration.triangle_gauss import rule as regular_rule
    from bempp.api import log
    from bempp.core.singular_assembler import assemble_singular_part
    from bempp.core import kernel_helpers

    options = operator_descriptor.options.copy()

    order = parameters.quadrature.regular
    quad_points, quad_weights = regular_rule(order)

    if "COMPLEX_KERNEL" in options:
        complex_kernel = True
    else:
        complex_kernel = False

    use_collocation = parameters.assembly.discretization_type == 'collocation'

    buffers = _prepare_buffers(
        domain,
        dual_to_range,
        quad_points,
        quad_weights,
        complex_kernel,
        device_interface,
        precision,
        use_collocation,
    )

    options["NUMBER_OF_QUAD_POINTS"] = len(quad_weights)
    options["TEST"] = dual_to_range.shapeset.identifier
    options["TRIAL"] = domain.shapeset.identifier
    options["TRIAL_NUMBER_OF_ELEMENTS"] = domain.number_of_support_elements
    options["TEST_NUMBER_OF_ELEMENTS"] = dual_to_range.number_of_support_elements

    options["NUMBER_OF_TEST_SHAPE_FUNCTIONS"] = dual_to_range.number_of_shape_functions

    options["NUMBER_OF_TRIAL_SHAPE_FUNCTIONS"] = domain.number_of_shape_functions

    if use_collocation:
        collocation_string = '_collocation'
    else:
        collocation_string = ''

    vec_extension, vec_length = kernel_helpers.get_vectorization_information(
        device_interface, precision
    )

    log(
        "Regular kernel vector length: {0} ({1} precision)".format(
            vec_length, precision
        ), "debug"
    )

    main_source = _cl_helpers.kernel_source_from_identifier(
        source_name + collocation_string + "_regular" + vec_extension, options
    )

    remainder_source = _cl_helpers.kernel_source_from_identifier(
        source_name + collocation_string + "_regular_novec", options
    )

    main_kernel = _cl_helpers.Kernel(main_source, device_interface.context, precision)
    remainder_kernel = _cl_helpers.Kernel(
        remainder_source, device_interface.context, precision
    )

    # Now process the elements according to the given
    # coloring in test/domain space.

    test_indices, test_color_indexptr = dual_to_range.get_elements_by_color()
    trial_indices, trial_color_indexptr = domain.get_elements_by_color()

    test_indices_buffer = _cl_helpers.DeviceBuffer.from_array(
        test_indices, device_interface, dtype=_np.uint32, access_mode="read_only"
    )
    trial_indices_buffer = _cl_helpers.DeviceBuffer.from_array(
        trial_indices, device_interface, dtype=_np.uint32, access_mode="read_only"
    )

    test_normal_signs_buffer = _cl_helpers.DeviceBuffer.from_array(
        dual_to_range.normal_multipliers,
        device_interface,
        dtype=_np.int32,
        access_mode="read_only",
    )
    trial_normal_signs_buffer = _cl_helpers.DeviceBuffer.from_array(
        domain.normal_multipliers, device_interface, dtype=_np.int32, access_mode="read_only"
    )

    kernel_helpers.run_chunked_kernel(
        main_kernel,
        remainder_kernel,
        device_interface,
        vec_length,
        [
            test_indices_buffer,
            trial_indices_buffer,
            test_normal_signs_buffer,
            trial_normal_signs_buffer,
            *buffers,
        ],
        parameters,
        chunks=(test_color_indexptr, trial_color_indexptr),
    )
    # log("Regular kernel runtime [ms]: {0}".format(runtime), "timing")

    regular_result = buffers[-4].get_host_copy(device_interface)

    if domain.grid == dual_to_range.grid:
        trial_local2global = domain.local2global.ravel()
        test_local2global = dual_to_range.local2global.ravel()
        trial_multipliers = domain.local_multipliers.ravel()
        test_multipliers = dual_to_range.local_multipliers.ravel()

        singular_rows, singular_cols, singular_values = assemble_singular_part(
            domain.localised_space,
            dual_to_range.localised_space,
            parameters,
            operator_descriptor,
            source_name,
            device_interface,
            precision,
        )

        rows = test_local2global[singular_rows]
        cols = trial_local2global[singular_cols]
        values = (
            singular_values
            * trial_multipliers[singular_cols]
            * test_multipliers[singular_rows]
        )

        _np.add.at(regular_result, (rows, cols), values)

    return regular_result


def _prepare_buffers(
    domain,
    dual_to_range,
    quad_points,
    quad_weights,
    complex_kernel,
    device_interface,
    precision,
    use_collocation,
):
    """Prepare kernel buffers."""

    trial_grid = domain.grid
    test_grid = dual_to_range.grid

    shape = (dual_to_range.global_dof_count, domain.global_dof_count)

    dtype = _cl_helpers.get_type(precision).real

    if complex_kernel:
        result_type = _cl_helpers.get_type(precision).complex
    else:
        result_type = _cl_helpers.get_type(precision).real

    quad_points_buffer = _cl_helpers.DeviceBuffer.from_array(
        quad_points, device_interface, dtype=dtype, access_mode="read_only", order="F"
    )

    quad_weights_buffer = _cl_helpers.DeviceBuffer.from_array(
        quad_weights, device_interface, dtype=dtype, access_mode="read_only", order="F"
    )

    test_connectivity = _cl_helpers.DeviceBuffer.from_array(
        dual_to_range.grid.elements,
        device_interface,
        dtype=_np.uint32,
        access_mode="read_only",
        order="F",
    )

    trial_connectivity = _cl_helpers.DeviceBuffer.from_array(
        domain.grid.elements,
        device_interface,
        dtype=_np.uint32,
        access_mode="read_only",
        order="F",
    )

    test_local2global = _cl_helpers.DeviceBuffer.from_array(
        dual_to_range.local2global,
        device_interface,
        dtype=_np.uint32,
        access_mode="read_only",
        order="C",
    )

    trial_local2global = _cl_helpers.DeviceBuffer.from_array(
        domain.local2global,
        device_interface,
        dtype=_np.uint32,
        access_mode="read_only",
        order="C",
    )

    test_multipliers = _cl_helpers.DeviceBuffer.from_array(
        dual_to_range.local_multipliers,
        device_interface,
        dtype=dtype,
        access_mode="read_only",
        order="C",
    )

    trial_multipliers = _cl_helpers.DeviceBuffer.from_array(
        domain.local_multipliers,
        device_interface,
        dtype=dtype,
        access_mode="read_only",
        order="C",
    )

    if use_collocation:

        collocation_points = _cl_helpers.DeviceBuffer.from_array(
            dual_to_range.collocation_points,
            device_interface,
            dtype=dtype,
            access_mode="read_only",
            order="F"
        )

    test_grid_buffer = test_grid.push_to_device(device_interface, precision).buffer
    trial_grid_buffer = trial_grid.push_to_device(device_interface, precision).buffer

    result_buffer = _cl_helpers.DeviceBuffer(
        shape,
        result_type,
        device_interface.context,
        access_mode="read_write",
        order="C",
    )
    result_buffer.set_zero(device_interface)

    if use_collocation:

        buffers = [
            test_grid_buffer,
            trial_grid_buffer,
            test_connectivity,
            trial_connectivity,
            test_local2global,
            trial_local2global,
            test_multipliers,
            trial_multipliers,
            quad_points_buffer,
            quad_weights_buffer,
            collocation_points,
            result_buffer,
            _np.int32(dual_to_range.global_dof_count),
            _np.int32(domain.global_dof_count),
            _np.uint8(domain.grid != dual_to_range.grid),
        ]

    else:

        buffers = [
            test_grid_buffer,
            trial_grid_buffer,
            test_connectivity,
            trial_connectivity,
            test_local2global,
            trial_local2global,
            test_multipliers,
            trial_multipliers,
            quad_points_buffer,
            quad_weights_buffer,
            result_buffer,
            _np.int32(dual_to_range.global_dof_count),
            _np.int32(domain.global_dof_count),
            _np.uint8(domain.grid != dual_to_range.grid),
        ]

    return buffers
