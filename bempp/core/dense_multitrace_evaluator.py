"""Dense Assembly of integral operators."""
import numpy as _np

import bempp.core.cl_helpers as _cl_helpers
from bempp.api.assembly import assembler as _assembler


class DenseMultitraceEvaluatorAssembler(_assembler.AssemblerBase):
    """A matvec evaluator for Calderon operators."""

    # pylint: disable=useless-super-delegation
    def __init__(self, domain, dual_to_range, parameters=None):
        """Create a dense evaluator instance."""
        super().__init__(domain, dual_to_range, parameters)
        self._singular_contribution = None
        self._buffers = None
        self._result_buffer = None
        self._input_buffer = None
        self._sum_buffer = None
        self._main_kernel = None
        self._remainder_kernel = None
        self._sum_kernel = None
        self._vec_extension = None
        self._vec_length = None
        self._device_interface = None
        self._precision = None
        self._main_size = None
        self._remainder_size = None

        self._trial_to_localised = None
        self._test_to_localised = None

        self._workgroup_size = 128

        self._dtype = None

        self._shape = (
            self.dual_to_range[0].global_dof_count
            + self.dual_to_range[1].global_dof_count,
            self.domain[0].global_dof_count + self.domain[1].global_dof_count,
        )

    @property
    def shape(self):
        """Return shape."""
        return self._shape

    @property
    def dtype(self):
        """Return dtype."""
        return self._dtype

    def assemble(
        self, operator_descriptor, device_interface, precision, *args, **kwargs
    ):
        """Prepares the evaluator and returns itself."""
        from bempp.api.assembly.discrete_boundary_operator import (
            GenericDiscreteBoundaryOperator,
        )
        from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
        from bempp.api.assembly.discrete_boundary_operator import (
            SparseDiscreteBoundaryOperator,
        )

        from bempp.api.utils.helpers import promote_to_double_precision
        from .dense_assembly_helpers import (
            choose_source_name_dense_multitrace_evaluator,
        )

        from bempp.api.integration.triangle_gauss import rule
        from bempp.core import kernel_helpers
        from bempp.api import log

        self._singular_contribution = BlockedDiscreteOperator(
            _np.array(
                [
                    [
                        operator_descriptor.singular_contribution[0, 0].weak_form(),
                        operator_descriptor.singular_contribution[0, 1].weak_form(),
                    ],
                    [
                        operator_descriptor.singular_contribution[1, 0].weak_form(),
                        operator_descriptor.singular_contribution[1, 1].weak_form(),
                    ],
                ],
                dtype=_np.object,
            )
        )

        _, quad_weights = rule(self.parameters.quadrature.regular)

        self._device_interface = device_interface
        self._precision = precision

        localised_domain = [domain.localised_space for domain in self.domain]
        localised_dual_to_range = [
            dual_to_range.localised_space for dual_to_range in self.dual_to_range
        ]

        self._trial_to_localised = BlockedDiscreteOperator(
            _np.array(
                [
                    [
                        SparseDiscreteBoundaryOperator(
                            self._domain[0].map_to_localised_space
                        ),
                        None,
                    ],
                    [
                        None,
                        SparseDiscreteBoundaryOperator(
                            self._domain[1].map_to_localised_space
                        ),
                    ],
                ],
                dtype=_np.object,
            )
        )

        self._test_to_localised = BlockedDiscreteOperator(
            _np.array(
                [
                    [
                        SparseDiscreteBoundaryOperator(
                            self._dual_to_range[0].map_to_localised_space.T
                        ),
                        None,
                    ],
                    [
                        None,
                        SparseDiscreteBoundaryOperator(
                            self._dual_to_range[1].map_to_localised_space.T
                        ),
                    ],
                ],
                dtype=_np.object,
            )
        )

        options = operator_descriptor.options.copy()

        if "COMPLEX_KERNEL" in options:
            complex_kernel = True
            options["COMPLEX_RESULT"] = None
            if (
                precision == "double"
                or self.parameters.assembly.always_promote_to_double
            ):
                self._dtype = _np.complex128
            else:
                self._dtype = _np.complex64
        else:
            complex_kernel = False
            if (
                precision == "double"
                or self.parameters.assembly.always_promote_to_double
            ):
                self._dtype = _np.float64
            else:
                self._dtype = _np.float32

        self._prepare_buffers(complex_kernel, device_interface, precision)

        options["NUMBER_OF_QUAD_POINTS"] = len(quad_weights)
        options["TRIAL0_NUMBER_OF_ELEMENTS"] = localised_domain[
            0
        ].grid.number_of_elements
        options["TRIAL1_NUMBER_OF_ELEMENTS"] = localised_domain[
            1
        ].grid.number_of_elements

        options["TEST0_NUMBER_OF_ELEMENTS"] = localised_dual_to_range[
            0
        ].grid.number_of_elements

        options["TEST1_NUMBER_OF_ELEMENTS"] = localised_dual_to_range[
            1
        ].grid.number_of_elements

        self._vec_extension, self._vec_length = kernel_helpers.get_vectorization_information(
            device_interface, precision
        )

        log(
            "Regular kernel vector length: {0} ({1} precision)".format(
                self._vec_length, precision
            )
        )

        source_name = choose_source_name_dense_multitrace_evaluator(
            operator_descriptor.multitrace_kernel
        )

        self._main_size, self._remainder_size = kernel_helpers.closest_multiple_to_number(
            self.domain[0].grid.number_of_elements, self._workgroup_size
        )

        options["WORKGROUP_SIZE"] = self._workgroup_size

        main_source = _cl_helpers.kernel_source_from_identifier(
            source_name + "_regular" + self._vec_extension, options
        )

        self._main_kernel = _cl_helpers.Kernel(
            main_source, device_interface.context, precision
        )

        sum_source = _cl_helpers.kernel_source_from_identifier(
            "sum_for_potential_novec", options
        )

        self._sum_kernel = _cl_helpers.Kernel(
            sum_source, device_interface.context, precision
        )

        if self._remainder_size > 0:
            options["WORKGROUP_SIZE"] = self._remainder_size
            remainder_source = _cl_helpers.kernel_source_from_identifier(
                source_name + "_regular_novec", options
            )
            self._remainder_kernel = _cl_helpers.Kernel(
                remainder_source, device_interface.context, precision
            )

        test_indices = _np.arange(
            localised_dual_to_range[0].grid.number_of_elements, dtype="uint32"
        )
        trial_indices = _np.arange(
            localised_domain[0].grid.number_of_elements, dtype="uint32"
        )

        test_indices_buffer = _cl_helpers.DeviceBuffer.from_array(
            test_indices, device_interface, dtype=_np.uint32, access_mode="read_only"
        )
        trial_indices_buffer = _cl_helpers.DeviceBuffer.from_array(
            trial_indices, device_interface, dtype=_np.uint32, access_mode="read_only"
        )

        self._buffers.insert(0, trial_indices_buffer)
        self._buffers.insert(0, test_indices_buffer)

        return GenericDiscreteBoundaryOperator(self)

    def _prepare_buffers(self, complex_kernel, device_interface, precision):
        """Prepare kernel buffers."""
        from bempp.api.integration.triangle_gauss import rule

        quad_points, quad_weights = rule(self.parameters.quadrature.regular)

        trial_grid = self.domain[0].localised_space.grid
        test_grid = self.dual_to_range[0].localised_space.grid

        shape = (
            self.dual_to_range[0].localised_space.global_dof_count
            + self.dual_to_range[1].localised_space.global_dof_count,
            self.domain[0].localised_space.global_dof_count
            + self.domain[1].localised_space.global_dof_count,
        )

        dtype = _cl_helpers.get_type(precision).real

        if complex_kernel:
            result_type = _cl_helpers.get_type(precision).complex
        else:
            result_type = _cl_helpers.get_type(precision).real

        quad_points_buffer = _cl_helpers.DeviceBuffer.from_array(
            quad_points,
            device_interface,
            dtype=dtype,
            access_mode="read_only",
            order="F",
        )

        quad_weights_buffer = _cl_helpers.DeviceBuffer.from_array(
            quad_weights,
            device_interface,
            dtype=dtype,
            access_mode="read_only",
            order="F",
        )

        test_connectivity = _cl_helpers.DeviceBuffer.from_array(
            self.dual_to_range[0].localised_space.grid.elements,
            device_interface,
            dtype=_np.uint32,
            access_mode="read_only",
            order="F",
        )

        trial_connectivity = _cl_helpers.DeviceBuffer.from_array(
            self.domain[0].localised_space.grid.elements,
            device_interface,
            dtype=_np.uint32,
            access_mode="read_only",
            order="F",
        )

        test_grid_buffer = test_grid.push_to_device(device_interface, precision).buffer
        trial_grid_buffer = trial_grid.push_to_device(
            device_interface, precision
        ).buffer

        input_buffer = _cl_helpers.DeviceBuffer(
            shape[1],
            result_type,
            device_interface.context,
            access_mode="read_only",
            order="C",
        )

        sum_buffer = _cl_helpers.DeviceBuffer(
            (shape[0], trial_grid.number_of_elements // self._workgroup_size),
            result_type,
            device_interface.context,
            access_mode="read_write",
            order="C",
        )

        result_buffer = _cl_helpers.DeviceBuffer(
            shape[0],
            result_type,
            device_interface.context,
            access_mode="read_write",
            order="C",
        )

        buffers = [
            test_grid_buffer,
            trial_grid_buffer,
            test_connectivity,
            trial_connectivity,
            quad_points_buffer,
            quad_weights_buffer,
            input_buffer,
            _np.uint8(
                self.domain[0].localised_space.grid
                != self.dual_to_range[0].localised_space.grid
            ),
        ]

        self._buffers = buffers
        self._result_buffer = result_buffer
        self._input_buffer = input_buffer
        self._sum_buffer = sum_buffer

    def matvec(self, x):
        """Evaluate the product with a vector."""
        from bempp.core import kernel_helpers
        from bempp.api import log

        with self._input_buffer.host_array(self._device_interface, "write") as array:
            array[:] = self._trial_to_localised @ x.flat

        self._sum_buffer.set_zero(self._device_interface)

        runtime = 0.0

        if self._main_size > 0:

            event = self._main_kernel.run(
                self._device_interface,
                (
                    self.dual_to_range[0].grid.number_of_elements,
                    self._main_size // self._vec_length,
                ),
                (1, self._workgroup_size // self._vec_length),
                *self._buffers,
                self._sum_buffer
            )

            event.wait()
            runtime += event.runtime()


            event = self._sum_kernel.run(
                self._device_interface,
                (
                    self.dual_to_range[0].localised_space.global_dof_count
                    + self.dual_to_range[1].localised_space.global_dof_count,
                ),
                (1,),
                self._sum_buffer,
                self._result_buffer,
                _np.uint32(
                    self.domain[0].grid.number_of_elements // self._workgroup_size
                ),
            )

            event.wait()
            runtime += event.runtime()


        if self._remainder_size > 0:

            event = self._remainder_kernel.run(
                self._device_interface,
                (self.dual_to_range[0].grid.number_of_elements, self._remainder_size),
                (1, self._remainder_size),
                *self._buffers,
                self._result_buffer,
                global_offset=(0, self._main_size)
            )

            event.wait()
            runtime += event.runtime()

        log("Regular kernel runtime [ms]: {0}".format(runtime))

        result = self._result_buffer.get_host_copy(self._device_interface)

        result = self._test_to_localised @ result

        if x.ndim > 1:
            return _np.expand_dims(result, 1) + self._singular_contribution @ x
        else:
            return result + self._singular_contribution @ x

