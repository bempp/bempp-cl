"""Generic implementation of a dense potential operator."""
import numpy as _np

from bempp.core import cl_helpers as _cl_helpers
from bempp.api.utils.helpers import list_to_float as _list_to_float


class DensePotentialAssembler(object):
    """
    Implementation of a dense potential operator.
    """

    def __init__(
        self,
        space,
        operator_descriptor,
        points,
        kernel_dimension,
        is_complex,
        device_interface,
        precision,
        parameters=None,
    ):

        from bempp.api import assign_parameters
        from bempp.api import default_device, get_precision

        self._points = points
        self._space = space
        self._parameters = assign_parameters(parameters)
        self._identifier = operator_descriptor.identifier
        self._is_complex = is_complex
        self._operator_descriptor = operator_descriptor
        self._kernel_dimension = kernel_dimension

        options = operator_descriptor.options.copy()

        self._kernel_parameters = options.get('kernel_parameters', [0])
        self._source_options = options['source']

        if device_interface is None:
            self._device_interface = default_device()
        else:
            self._device_interface = device_interface

        if precision is None:
            self._precision = get_precision(self._device_interface)
        else:
            self._precision = precision

        if operator_descriptor.compute_kernel == "default_dense":
            self._compute_kernel = "evaluate_scalar_potential"
        elif operator_descriptor.compute_kernel == "maxwell_electric_field":
            self._compute_kernel = "evaluate_electric_field_potential"
        elif operator_descriptor.compute_kernel == "maxwell_magnetic_field":
            self._compute_kernel = "evaluate_magnetic_field_potential"
        elif operator_descriptor.compute_kernel == "helmholtz_scalar_far_field":
            self._compute_kernel = "evaluate_helmholtz_far_field"
        elif operator_descriptor.compute_kernel == "maxwell_electric_far_field":
            self._compute_kernel = "evaluate_maxwell_electric_far_field"
        elif operator_descriptor.compute_kernel == "maxwell_magnetic_far_field":
            self._compute_kernel = "evaluate_maxwell_magnetic_far_field"
        else:
            raise ValueError("Unknown compute kernel for potential.")

        self._run_kernel = self._init_operator()

    def update(self, kernel_parameters=None):
        """Re-assemble with updated parameters."""
        if kernel_parameters is not None:
            self._kernel_parameters_buffer.fill_buffer(
                    self._device_interface, kernel_parameters)

    @property
    def space(self):
        """Return space."""
        return self._space

    @property
    def points(self):
        """Return points."""
        return self._points

    @property
    def parameters(self):
        """Return parameters."""
        return self._parameters

    @property
    def kernel_dimension(self):
        """Return kernel dimension."""
        return self._kernel_dimension

    @property
    def operator_descriptor(self):
        """Return operator descriptor."""
        return self._operator_descriptor

    @property
    def device_interface(self):
        """Return device interface."""
        return self._device_interface

    @property
    def precision(self):
        """Return precision."""
        return self._precision

    def _init_operator(self):
        """Setup the operator."""
        from bempp.api.integration.triangle_gauss import rule as regular_rule
        from bempp.api import log
        from bempp.core import kernel_helpers

        source_options = self._source_options

        localised_space = self.space.localised_space
        grid = localised_space.grid

        order = self.parameters.quadrature.regular
        dtype = _cl_helpers.get_type(self.precision).real

        npoints = self.points.shape[1]

        source_options = self.operator_descriptor.options['source']

        if self._is_complex:
            source_options["COMPLEX_RESULT"] = None
            source_options["COMPLEX_KERNEL"] = None
            source_options["COMPLEX_COEFFICIENTS"] = None
            result_type = _cl_helpers.get_type(self.precision).complex
            coefficient_type = _cl_helpers.get_type(self.precision).complex
        else:
            result_type = _cl_helpers.get_type(self.precision).real
            coefficient_type = _cl_helpers.get_type(self.precision).real

        points_buffer = _cl_helpers.DeviceBuffer.from_array(
            self.points,
            self.device_interface,
            dtype=dtype,
            access_mode="read_only",
            order="F",
        )

        quad_points, quad_weights = regular_rule(order)

        quad_points_buffer = _cl_helpers.DeviceBuffer.from_array(
            quad_points,
            self.device_interface,
            dtype=dtype,
            access_mode="read_only",
            order="F",
        )

        quad_weights_buffer = _cl_helpers.DeviceBuffer.from_array(
            quad_weights,
            self.device_interface,
            dtype=dtype,
            access_mode="read_only",
            order="F",
        )

        coefficients_buffer = _cl_helpers.DeviceBuffer(
            self.space.map_to_full_grid.shape[0],
            coefficient_type,
            self.device_interface.context,
            access_mode="read_only",
            order="C",
        )

        normal_signs_buffer = _cl_helpers.DeviceBuffer.from_array(
            localised_space.normal_multipliers,
            self.device_interface,
            dtype=_np.int32,
            access_mode="read_only",
        )

        indices = localised_space.support_elements.astype("uint32")
        indices_buffer = _cl_helpers.DeviceBuffer.from_array(
            indices, self.device_interface, dtype=_np.uint32, access_mode="read_only"
        )

        grid_buffer = grid.push_to_device(self.device_interface, self.precision).buffer

        workgroup_size = 128

        source_options["SHAPESET"] = localised_space.shapeset.identifier
        source_options["NUMBER_OF_SHAPE_FUNCTIONS"] = localised_space.number_of_shape_functions
        source_options["KERNEL_DIMENSION"] = self.kernel_dimension
        source_options["NUMBER_OF_QUAD_POINTS"] = len(quad_weights)

        main_size, remainder_size = kernel_helpers.closest_multiple_to_number(
            localised_space.number_of_support_elements, workgroup_size
        )

        result_buffer = _cl_helpers.DeviceBuffer(
            (self.kernel_dimension * npoints,),
            result_type,
            self.device_interface.context,
            access_mode="read_write",
            order="C",
        )
        
        kernel_parameters_buffer = _cl_helpers.DeviceBuffer.from_array(
                _list_to_float(self._kernel_parameters, self._precision),
                self.device_interface,
                dtype=dtype,
                access_mode="read_only",
                order="C"
                )

        self._kernel_parameters_buffer = kernel_parameters_buffer

        if main_size > 0:

            vec_extension, vec_length = kernel_helpers.get_vectorization_information(
                self.device_interface, self.precision
            )

            source_options["WORKGROUP_SIZE"] = workgroup_size // vec_length

            sum_buffer = _cl_helpers.DeviceBuffer(
                (
                    self.kernel_dimension * npoints,
                    localised_space.number_of_support_elements // workgroup_size,
                ),
                result_type,
                self.device_interface.context,
                access_mode="read_write",
                order="C",
            )
            sum_buffer.set_zero(self.device_interface)

            main_source = _cl_helpers.kernel_source_from_identifier(
                self._compute_kernel + vec_extension, source_options
            )

            sum_source = _cl_helpers.kernel_source_from_identifier(
                "sum_for_potential_novec", source_options
            )

            main_kernel = _cl_helpers.Kernel(
                main_source, self.device_interface.context, self.precision
            )
            sum_kernel = _cl_helpers.Kernel(
                sum_source, self.device_interface.context, self.precision
            )


        if remainder_size > 0:

            source_options["WORKGROUP_SIZE"] = remainder_size

            remainder_source = _cl_helpers.kernel_source_from_identifier(
                self._compute_kernel + "_novec", source_options
            )

            remainder_kernel = _cl_helpers.Kernel(
                remainder_source, self.device_interface.context, self.precision
            )

        def run_kernel(coefficients):
            """Actually run the kernel."""
            localised_coefficients = self.space.map_to_full_grid.dot(
                self.space.dof_transformation @ coefficients
            )

            coefficients_buffer.fill_buffer(self.device_interface, localised_coefficients)
            if main_size > 0:
                sum_buffer.set_zero(self.device_interface)
                event = main_kernel.run(
                    self.device_interface,
                    (npoints, main_size // vec_length),
                    (1, workgroup_size // vec_length),
                    grid_buffer,
                    indices_buffer,
                    normal_signs_buffer,
                    points_buffer,
                    coefficients_buffer,
                    quad_points_buffer,
                    quad_weights_buffer,
                    sum_buffer,
                    kernel_parameters_buffer
                )

                event.wait()

                event = sum_kernel.run(
                    self.device_interface,
                    (self.kernel_dimension * npoints,),
                    (1,),
                    sum_buffer,
                    result_buffer,
                    _np.uint32(
                        localised_space.number_of_support_elements // workgroup_size
                    ),
                )

                event.wait()

            if remainder_size > 0:
                event = remainder_kernel.run(
                    self.device_interface,
                    (npoints, remainder_size),
                    (1, remainder_size),
                    grid_buffer,
                    indices_buffer,
                    normal_signs_buffer,
                    points_buffer,
                    coefficients_buffer,
                    quad_points_buffer,
                    quad_weights_buffer,
                    result_buffer,
                    kernel_parameters_buffer,
                    global_offset=(0, main_size),
                )

                event.wait()

            
            result = result_buffer.get_host_copy(self.device_interface).reshape(
                self.kernel_dimension, npoints, order="F"
            )

            return result

        return run_kernel

    def evaluate(self, coefficients):
        """Evaluate the potential for given coefficients."""

        if not self._is_complex and _np.iscomplexobj(coefficients):
            return self._run_kernel(_np.real(coefficients)) + self._run_kernel(
                    _np.imag(coefficients))
        else:
            return self._run_kernel(coefficients)




