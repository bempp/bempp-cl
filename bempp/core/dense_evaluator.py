"""Dense Assembly of integral operators."""
import numpy as _np

import bempp.core.cl_helpers as _cl_helpers
from bempp.api.assembly import assembler as _assembler


class DenseEvaluatorAssembler(_assembler.AssemblerBase):
    """Matvec kernel evaluator."""

    def __init__(self, domain, dual_to_range, parameters=None):
        """Initialise the assembler."""

        super().__init__(domain, dual_to_range, parameters)

        self._dtype = None
        self._shape = (
            self.dual_to_range.global_dof_count,
            self.domain.global_dof_count,
        )
        self._singular_contribution = None

        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor()

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
        """The assembler calls the assembler instances for each device."""
        from bempp.core.singular_assembler import SingularAssembler
        from bempp.api.assembly.discrete_boundary_operator import (
            GenericDiscreteBoundaryOperator,
        )
        from bempp.api.space.space import return_compatible_representation
        from collections.abc import Iterable

        if not isinstance(device_interface, Iterable):
            self._device_group = [device_interface]
        else:
            self._device_group = device_interface

        self._actual_domain, self._actual_dual_to_range = return_compatible_representation(
            self.domain, self.dual_to_range
        )

        self._singular_contribution = (
            SingularAssembler(self.domain, self.dual_to_range, self.parameters)
            .assemble(
                operator_descriptor, self._device_group[0], precision, *args, **kwargs
            )
            .A
        )

        ndevices = len(self._device_group)
        support_elements = self._actual_domain.support_elements

        chunks = _np.array_split(support_elements, ndevices)
        self._assembler_instances = []

        for device, chunk in zip(self._device_group, chunks):
            instance = DenseEvaluatorAssemblerInstance(
                self._actual_domain,
                self._actual_dual_to_range,
                chunk,
                parameters=self.parameters,
            )
            instance.assemble(operator_descriptor, device, precision, *args, **kwargs)
            self._assembler_instances.append(instance)

        self._dtype = self._assembler_instances[0].dtype
        return GenericDiscreteBoundaryOperator(self)

    def matvec(self, x):
        """Apply operator to a vector x."""

        transformed_vec = self._actual_domain.map_to_full_grid @ (
            self._actual_domain.dof_transformation @ x.flat
        )

        result = sum(
            self._executor.map(
                lambda instance: instance.matvec(transformed_vec),
                self._assembler_instances,
            )
        )

        # result = sum(
        # [instance.matvec(transformed_vec) for instance in self._assembler_instances]
        # )

        result = self._actual_dual_to_range.dof_transformation.T @ (
            self._actual_dual_to_range.map_to_localised_space.T @ result
        )

        if x.ndim > 1:
            return _np.expand_dims(result, 1) + self._singular_contribution @ x
        else:
            return result + self._singular_contribution @ x


class DenseEvaluatorAssemblerInstance(_assembler.AssemblerBase):
    """A matvec evaluator."""

    # pylint: disable=useless-super-delegation
    def __init__(self, domain, dual_to_range, chunk=None, parameters=None):
        """Create a dense evaluator instance."""
        from bempp.api.space.space import return_compatible_representation

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

        self._workgroup_size = 128

        self._dtype = None

        self._shape = (
            self.dual_to_range.global_dof_count,
            self.domain.global_dof_count,
        )

        self._actual_domain, self._actual_dual_to_range = return_compatible_representation(
            self.domain, self.dual_to_range
        )

        if chunk is None:
            self._chunk = self._actual_domain.support_elements
        else:
            self._chunk = chunk

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

        from bempp.core.singular_assembler import SingularAssembler
        from bempp.api.utils.helpers import promote_to_double_precision
        from .dense_assembly_helpers import choose_source_name_dense_evaluator
        from bempp.api.integration.triangle_gauss import rule
        from bempp.core import kernel_helpers
        from bempp.api import log

        _, quad_weights = rule(self.parameters.quadrature.regular)

        self._device_interface = device_interface
        self._precision = precision

        localised_domain = self._actual_domain.localised_space
        localised_dual_to_range = self._actual_dual_to_range.localised_space

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

        domain_support_size = len(self._chunk)
        dual_to_range_support_size = localised_dual_to_range.number_of_support_elements

        options["NUMBER_OF_QUAD_POINTS"] = len(quad_weights)
        options["TEST"] = localised_dual_to_range.shapeset.identifier
        options["TRIAL"] = localised_domain.shapeset.identifier
        options["TRIAL_NUMBER_OF_ELEMENTS"] = domain_support_size
        options["TEST_NUMBER_OF_ELEMENTS"] = dual_to_range_support_size

        options[
            "NUMBER_OF_TEST_SHAPE_FUNCTIONS"
        ] = localised_dual_to_range.number_of_shape_functions

        options[
            "NUMBER_OF_TRIAL_SHAPE_FUNCTIONS"
        ] = localised_domain.number_of_shape_functions

        self._vec_extension, self._vec_length = kernel_helpers.get_vectorization_information(
            device_interface, precision
        )

        log(
            "Regular kernel vector length: {0} ({1} precision)".format(
                self._vec_length, precision
            )
        )

        source_name = choose_source_name_dense_evaluator(
            operator_descriptor.compute_kernel
        )

        self._main_size, self._remainder_size = kernel_helpers.closest_multiple_to_number(
            domain_support_size, self._workgroup_size
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

        test_indices = localised_dual_to_range.support_elements.astype("uint32")
        trial_indices = self._chunk.astype("uint32")

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

        trial_grid = self._actual_domain.localised_space.grid
        test_grid = self._actual_dual_to_range.localised_space.grid

        trial_nshape_fun = self._actual_domain.number_of_shape_functions

        domain_support_size = len(self._chunk)
        dual_to_range_support_size = (
            self._actual_dual_to_range.localised_space.number_of_support_elements
        )

        shape = (
            self._actual_dual_to_range.localised_space.grid_dof_count,
            self._actual_domain.localised_space.grid_dof_count,
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

        test_connectivity = test_grid.push_to_device(
            device_interface, precision
        ).elements_buffer
        trial_connectivity = trial_grid.push_to_device(
            device_interface, precision
        ).elements_buffer

        test_grid_buffer = test_grid.push_to_device(
            device_interface, precision
        ).grid_buffer
        trial_grid_buffer = trial_grid.push_to_device(
            device_interface, precision
        ).grid_buffer

        input_buffer = _cl_helpers.DeviceBuffer(
            trial_nshape_fun * trial_grid.number_of_elements,
            result_type,
            device_interface.context,
            access_mode="read_only",
            order="C",
        )

        sum_buffer = _cl_helpers.DeviceBuffer(
            (shape[0], domain_support_size // self._workgroup_size),
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

        test_normal_signs_buffer = _cl_helpers.DeviceBuffer.from_array(
            self._actual_dual_to_range.localised_space.normal_multipliers,
            device_interface,
            dtype=_np.int32,
            access_mode="read_only",
        )
        trial_normal_signs_buffer = _cl_helpers.DeviceBuffer.from_array(
            self._actual_domain.localised_space.normal_multipliers,
            device_interface,
            dtype=_np.int32,
            access_mode="read_only",
        )

        buffers = [
            test_normal_signs_buffer,
            trial_normal_signs_buffer,
            test_grid_buffer,
            trial_grid_buffer,
            test_connectivity,
            trial_connectivity,
            quad_points_buffer,
            quad_weights_buffer,
            input_buffer,
            _np.uint8(
                self._actual_domain.localised_space.grid
                != self._actual_dual_to_range.localised_space.grid
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
            array[:] = x

        self._sum_buffer.set_zero(self._device_interface)

        runtime = 0.0

        if self._main_size > 0:

            event = self._main_kernel.run(
                self._device_interface,
                (
                    self._actual_dual_to_range.localised_space.number_of_support_elements,
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
                (self._actual_dual_to_range.localised_space.grid_dof_count,),
                (1,),
                self._sum_buffer,
                self._result_buffer,
                _np.uint32(len(self._chunk) // self._workgroup_size),
            )

            event.wait()
            runtime += event.runtime()

        if self._remainder_size > 0:

            event = self._remainder_kernel.run(
                self._device_interface,
                (
                    self._actual_dual_to_range.localised_space.number_of_support_elements,
                    self._remainder_size,
                ),
                (1, self._remainder_size),
                *self._buffers,
                self._result_buffer,
                global_offset=(0, self._main_size)
            )

            event.wait()
            runtime += event.runtime()

        log("Regular kernel runtime [ms]: {0}".format(runtime), "debug")

        result = self._result_buffer.get_host_copy(self._device_interface)

        return result
