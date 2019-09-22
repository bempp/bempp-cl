"""Assembly of sparse operators."""
import numpy as _np

import bempp.core.cl_helpers as _cl_helpers
from bempp.api.assembly import assembler as _assembler


class SparseAssembler(_assembler.AssemblerBase):
    """Implementation of an assembler for sparse boundary operators."""

    # pylint: disable=useless-super-delegation
    def __init__(self, domain, dual_to_range, parameters=None):
        """Create a sparse assembler instance."""
        super().__init__(domain, dual_to_range, parameters)

    def assemble(
        self, operator_descriptor, device_interface, precision, *args, **kwargs
    ):
        """Sparse assembly of the integral operator."""
        from bempp.api.assembly.discrete_boundary_operator import (
            SparseDiscreteBoundaryOperator,
        )
        from bempp.api.utils.helpers import promote_to_double_precision
        from bempp.api.space.space import return_compatible_representation
        from scipy.sparse import coo_matrix

        domain, dual_to_range = return_compatible_representation(
                self.domain, self.dual_to_range)

        trial_local2global = domain.local2global.ravel()
        test_local2global = dual_to_range.local2global.ravel()
        trial_multipliers = domain.local_multipliers.ravel()
        test_multipliers = dual_to_range.local_multipliers.ravel()

        rows, cols, data = assemble_sparse(
            domain.localised_space,
            dual_to_range.localised_space,
            self.parameters,
            operator_descriptor,
            device_interface,
            precision,
        )

        new_rows = test_local2global[rows]
        new_cols = trial_local2global[cols]
        new_data = data * trial_multipliers[cols] * test_multipliers[rows]

        if self.parameters.assembly.always_promote_to_double:
            new_data = promote_to_double_precision(new_data)

        nrows = dual_to_range.dof_transformation.shape[0]
        ncols = domain.dof_transformation.shape[0]

        mat = coo_matrix((new_data, (new_rows, new_cols)), shape=(nrows, ncols)).tocsr()

        if domain.requires_dof_transformation:
            mat = mat @ domain.dof_transformation

        if dual_to_range.requires_dof_transformation:
            mat = dual_to_range.dof_transformation.T @ mat

        return SparseDiscreteBoundaryOperator(mat)

def assemble_sparse(
    domain, dual_to_range, parameters, operator_descriptor, device_interface, precision
):
    """
    Really assemble the operator.

    Assembles the complete sparse operator.
    Returns a sparse matrix.

    """
    from bempp.api.integration.triangle_gauss import rule as regular_rule
    from bempp.core import kernel_helpers

    if domain.grid != dual_to_range.grid:
        raise ValueError("domain and dual_to_range must be defined on the same grid.")

    identifier = operator_descriptor.identifier
    options = operator_descriptor.options.copy()

    order = parameters.quadrature.regular
    quad_points, quad_weights = regular_rule(order)

    buffers = _prepare_buffers(
        domain, dual_to_range, quad_points, quad_weights, device_interface, precision
    )

    number_of_test_shape_functions = dual_to_range.number_of_shape_functions
    number_of_trial_shape_functions = domain.number_of_shape_functions

    options["NUMBER_OF_QUAD_POINTS"] = len(quad_weights)
    options["TEST"] = dual_to_range.shapeset.identifier
    options["TRIAL"] = domain.shapeset.identifier

    options["NUMBER_OF_TEST_SHAPE_FUNCTIONS"] = number_of_test_shape_functions

    options["NUMBER_OF_TRIAL_SHAPE_FUNCTIONS"] = number_of_trial_shape_functions

    vec_length = 1

    main_source = _cl_helpers.kernel_source_from_identifier(
        identifier + "_novec", options
    )

    remainder_source = _cl_helpers.kernel_source_from_identifier(
        identifier + "_novec", options
    )

    main_kernel = _cl_helpers.Kernel(main_source, device_interface.context, precision)
    remainder_kernel = _cl_helpers.Kernel(
        remainder_source, device_interface.context, precision
    )

    elements = _np.flatnonzero(domain.support * dual_to_range.support).astype("uint32")
    number_of_elements = len(elements)

    kernel_helpers.run_chunked_kernel(
        main_kernel,
        remainder_kernel,
        device_interface,
        vec_length,
        buffers,
        parameters,
        ([0, 1], [0, number_of_elements]),
    )

    result = buffers[-2].get_host_copy(device_interface)

    irange = _np.arange(number_of_test_shape_functions)
    jrange = _np.arange(number_of_trial_shape_functions)

    i_ind = _np.tile(
        _np.repeat(irange, number_of_trial_shape_functions), number_of_elements
    ) + _np.repeat(
        elements * number_of_test_shape_functions,
        number_of_test_shape_functions * number_of_trial_shape_functions,
    )

    j_ind = _np.tile(
        _np.tile(jrange, number_of_test_shape_functions), number_of_elements
    ) + _np.repeat(
        elements * number_of_trial_shape_functions,
        number_of_test_shape_functions * number_of_trial_shape_functions,
    )

    return (i_ind, j_ind, result)


def _prepare_buffers(
    domain, dual_to_range, quad_points, quad_weights, device_interface, precision
):
    """Prepare kernel buffers."""

    grid = domain.grid

    dtype = _cl_helpers.get_type(precision).real

    quad_points_buffer = _cl_helpers.DeviceBuffer.from_array(
        quad_points, device_interface, dtype=dtype, access_mode="read_only", order="F"
    )

    quad_weights_buffer = _cl_helpers.DeviceBuffer.from_array(
        quad_weights, device_interface, dtype=dtype, access_mode="read_only", order="F"
    )

    grid_buffer = grid.push_to_device(device_interface, precision).buffer

    elements = _np.flatnonzero(domain.support * dual_to_range.support).astype("uint32")

    elements_buffer = _cl_helpers.DeviceBuffer.from_array(
            elements, device_interface, dtype=_np.uint32, access_mode='read_only')

    number_of_elements = len(elements)

    result_buffer_size = number_of_elements * (
        domain.number_of_shape_functions * dual_to_range.number_of_shape_functions
    )

    result_buffer = _cl_helpers.DeviceBuffer(
        (result_buffer_size,),
        dtype,
        device_interface.context,
        access_mode="write_only",
        order="C",
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



    buffers = [
        grid_buffer,
        elements_buffer,
        test_normal_signs_buffer,
        trial_normal_signs_buffer,
        quad_points_buffer,
        quad_weights_buffer,
        result_buffer,
        _np.int32(grid.number_of_elements),
    ]

    return buffers
