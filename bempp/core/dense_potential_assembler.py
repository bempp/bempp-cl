"""Generic implementation of a dense potential operator."""
import numpy as _np

from bempp.core import cl_helpers as _cl_helpers


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
        else:
            raise ValueError("Unknown compute kernel for potential.")

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

    def evaluate(self, coefficients):
        """Evaluate the potential for given coefficients."""
        from bempp.api.integration.triangle_gauss import rule as regular_rule
        from bempp.api import log
        from bempp.core import kernel_helpers

        localised_space = self.space.localised_space
        grid = localised_space.grid
        localised_coefficients = self.space.map_to_full_grid.dot(coefficients)

        order = self.parameters.quadrature.regular
        dtype = _cl_helpers.get_type(self.precision).real

        npoints = self.points.shape[1]

        options = self.operator_descriptor.options.copy()

        if self._is_complex:
            options["COMPLEX_RESULT"] = None
            options["COMPLEX_KERNEL"] = None
            result_type = _cl_helpers.get_type(self.precision).complex
        else:
            result_type = _cl_helpers.get_type(self.precision).real

        if _np.iscomplexobj(localised_coefficients):
            coefficient_type = _cl_helpers.get_type(self.precision).complex
            result_type = _cl_helpers.get_type(self.precision).complex
            options["COMPLEX_COEFFICIENTS"] = None
            options["COMPLEX_RESULT"] = None
        else:
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

        coefficients_buffer = _cl_helpers.DeviceBuffer.from_array(
            localised_coefficients,
            self.device_interface,
            dtype=coefficient_type,
            access_mode="read_only",
            order="F",
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

        options["SHAPESET"] = localised_space.shapeset.identifier
        options["NUMBER_OF_SHAPE_FUNCTIONS"] = localised_space.number_of_shape_functions
        options["KERNEL_DIMENSION"] = self.kernel_dimension
        options["NUMBER_OF_QUAD_POINTS"] = len(quad_weights)

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

        if main_size > 0:

            vec_extension, vec_length = kernel_helpers.get_vectorization_information(
                self.device_interface, self.precision
            )

            options["WORKGROUP_SIZE"] = workgroup_size // vec_length

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
                self._compute_kernel + vec_extension, options
            )

            sum_source = _cl_helpers.kernel_source_from_identifier(
                "sum_for_potential_novec", options
            )

            main_kernel = _cl_helpers.Kernel(
                main_source, self.device_interface.context, self.precision
            )
            sum_kernel = _cl_helpers.Kernel(
                sum_source, self.device_interface.context, self.precision
            )

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
            )

            event.wait()

            event = sum_kernel.run(
                self.device_interface,
                (self.kernel_dimension * npoints,),
                (1,),
                sum_buffer,
                result_buffer,
                _np.uint32(localised_space.number_of_support_elements // workgroup_size),
            )

            event.wait()


        if remainder_size > 0:

            options["WORKGROUP_SIZE"] = remainder_size

            remainder_source = _cl_helpers.kernel_source_from_identifier(
                self._compute_kernel + "_novec", options
            )

            remainder_kernel = _cl_helpers.Kernel(
                remainder_source, self.device_interface.context, self.precision
            )
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
                global_offset=(0, main_size),
            )

            event.wait()

        result = result_buffer.get_host_copy(self.device_interface).reshape(
            self.kernel_dimension, npoints, order="F"
        )

        return result
