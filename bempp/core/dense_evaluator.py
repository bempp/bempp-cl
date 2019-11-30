"""Dense Assembly of integral operators."""
import numpy as _np

import bempp.core.cl_helpers as _cl_helpers
from bempp.api.assembly import assembler as _assembler

WORKGROUP_SIZE = 128

_evaluator = None


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
        self._parameters = parameters

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
        from bempp.core.cl_helpers import get_context_by_name
        from bempp.api.utils import pool
        from bempp.api.assembly.discrete_boundary_operator import (
            GenericDiscreteBoundaryOperator,
        )
        from bempp.api.space.space import return_compatible_representation
        import multiprocessing as mp


        self._ndevices = pool.number_of_workers()

        self._actual_domain, self._actual_dual_to_range = return_compatible_representation(
            self.domain, self.dual_to_range
        )

        self._singular_contribution = (
            SingularAssembler(self.domain, self.dual_to_range, self.parameters)
            .assemble(
                operator_descriptor, device_interface, precision, *args, **kwargs
            )
            .A
        )

        support_elements = self._actual_domain.support_elements
        chunks = _np.array_split(support_elements, self._ndevices)
        self._assembler_instances = []

        complex_kernel = "COMPLEX_KERNEL" in operator_descriptor.options

        if complex_kernel:
            self._dtype = _np.complex128
        else:
            self._dtype = _np.float64

        self._worker_ids = pool.execute(
            _prepare_evaluator,
            self.dual_to_range.id,
            self.domain.id,
            chunks,
            self._parameters.quadrature.regular,
            operator_descriptor,
            precision,
        )

        # self._pool = mp.get_context("spawn").Pool(
        # ndevices,
        # initializer=_init_workers,
        # initargs=[
        # self._actual_dual_to_range.grid.as_array,
        # self._actual_domain.grid.as_array,
        # self._actual_dual_to_range.grid.elements,
        # self._actual_domain.grid.elements,
        # self._actual_dual_to_range.normal_multipliers,
        # self._actual_domain.normal_multipliers,
        # self._actual_dual_to_range.support_elements,
        # self._parameters.quadrature.regular,
        # self._actual_dual_to_range.shapeset.identifier,
        # self._actual_domain.shapeset.identifier,
        # self._actual_dual_to_range.number_of_shape_functions,
        # self._actual_domain.number_of_shape_functions,
        # platform_name,
        # operator_descriptor,
        # precision,
        # self._actual_dual_to_range.grid != self._actual_domain.grid,
        # ]
        # )

        # self._pool.starmap(prepare_evaluator, zip(chunks, range(ndevices)), chunksize=1)

        return GenericDiscreteBoundaryOperator(self)

    def matvec(self, x):
        """Apply operator to a vector x."""
        from bempp.api.utils import pool

        transformed_vec = self._actual_domain.map_to_full_grid @ (
            self._actual_domain.dof_transformation @ x.flat
        )

        # for instance in self._assembler_instances:
        # instance.compute(transformed_vec)

        # result = sum([instance.get_result() for instance in self._assembler_instances])
        result = sum(pool.starmap(_worker, zip(self._worker_ids, self._ndevices * [transformed_vec])))

        result = self._actual_dual_to_range.dof_transformation.T @ (
            self._actual_dual_to_range.map_to_localised_space.T @ result
        )

        if x.ndim > 1:
            return _np.expand_dims(result, 1) + self._singular_contribution @ x
        else:
            return result + self._singular_contribution @ x


class OldDenseEvaluatorAssembler(_assembler.AssemblerBase):
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

        for instance in self._assembler_instances:
            instance.compute(transformed_vec)

        result = sum([instance.get_result() for instance in self._assembler_instances])

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

    def compute(self, x):
        """Evaluate the product with a vector."""
        from bempp.core import kernel_helpers
        from bempp.api import log

        with self._input_buffer.host_array(self._device_interface, "write") as array:
            array[:] = x

        self._sum_buffer.set_zero(self._device_interface)

        if self._main_size > 0:

            self._main_kernel.run(
                self._device_interface,
                (
                    self._actual_dual_to_range.localised_space.number_of_support_elements,
                    self._main_size // self._vec_length,
                ),
                (1, self._workgroup_size // self._vec_length),
                *self._buffers,
                self._sum_buffer,
            )

            self._sum_kernel.run(
                self._device_interface,
                (self._actual_dual_to_range.localised_space.grid_dof_count,),
                (1,),
                self._sum_buffer,
                self._result_buffer,
                _np.uint32(len(self._chunk) // self._workgroup_size),
            )

        if self._remainder_size > 0:

            self._remainder_kernel.run(
                self._device_interface,
                (
                    self._actual_dual_to_range.localised_space.number_of_support_elements,
                    self._remainder_size,
                ),
                (1, self._remainder_size),
                *self._buffers,
                self._result_buffer,
                global_offset=(0, self._main_size),
            )

    def get_result(self):
        """Return result."""

        return self._result_buffer.get_host_copy(self._device_interface)


class DenseEvaluatorMultiprocessingInstance(object):
    """Evaluator Instance for MultiProcessing."""

    def __init__(
        self,
        test_space_id,
        trial_space_id,
        chunk,
        number_of_quad_points,
        kernel_options,
        precision,
    ):
        """Instantiate the class."""
        import bempp.api
        from bempp.api.utils import pool
        from bempp.api.utils import helpers
        from bempp.api.space.space import return_compatible_representation

        test_space = pool.get_data(test_space_id)
        trial_space = pool.get_data(trial_space_id)

        actual_trial_space, actual_test_space = return_compatible_representation(
            trial_space, test_space
        )

        self._test_grid = actual_test_space.grid.as_array
        self._trial_grid = actual_trial_space.grid.as_array
        self._test_elements = actual_test_space.grid.elements
        self._trial_elements = actual_trial_space.grid.elements
        self._test_normal_signs = actual_test_space.normal_multipliers
        self._trial_normal_signs = actual_trial_space.normal_multipliers
        self._test_support_elements = actual_test_space.support_elements
        self._number_of_quad_points = number_of_quad_points
        self._test_shape_set = actual_test_space.shapeset.identifier
        self._trial_shape_set = actual_trial_space.shapeset.identifier
        self._chunk = chunk
        self._number_of_test_shape_functions = (
            actual_test_space.number_of_shape_functions
        )
        self._number_of_trial_shape_functions = (
            actual_trial_space.number_of_shape_functions
        )
        self._kernel_options = kernel_options
        self._precision = precision
        self._grids_disjoint = test_space.grid != trial_space.grid

        self._device_interface = bempp.api.default_device()

        self._id = helpers.create_unique_id()

        self.compile_kernel()

        pool.insert_data(self._id, self)

    @property
    def id(self):
        """Return id."""
        return self._id

    def compile_kernel(self):
        """Compile the kernel."""
        from bempp.api.integration.triangle_gauss import rule
        from bempp.core import cl_helpers
        from bempp.core import kernel_helpers
        from .dense_assembly_helpers import choose_source_name_dense_evaluator

        quad_points, quad_weights = rule(self._number_of_quad_points)
        domain_support_size = len(self._chunk)
        dual_to_range_support_size = len(self._test_support_elements)
        shape = (
            self._number_of_test_shape_functions * self._test_elements.shape[1],
            self._number_of_trial_shape_functions * self._trial_elements.shape[1],
        )

        complex_kernel = "COMPLEX_KERNEL" in self._kernel_options.options

        dtype = cl_helpers.get_type(self._precision).real

        if complex_kernel:
            result_type = cl_helpers.get_type(self._precision).complex
        else:
            result_type = cl_helpers.get_type(self._precision).real

        quad_points_buffer = _cl_helpers.DeviceBuffer.from_array(
            quad_points,
            self._device_interface,
            dtype=dtype,
            access_mode="read_only",
            order="F",
        )

        quad_weights_buffer = cl_helpers.DeviceBuffer.from_array(
            quad_weights,
            self._device_interface,
            dtype=dtype,
            access_mode="read_only",
            order="F",
        )

        test_connectivity = cl_helpers.DeviceBuffer.from_array(
            self._test_elements,
            self._device_interface,
            dtype="uint32",
            access_mode="read_only",
            order="F",
        )

        trial_connectivity = cl_helpers.DeviceBuffer.from_array(
            self._trial_elements,
            self._device_interface,
            dtype="uint32",
            access_mode="read_only",
            order="F",
        )

        test_grid_buffer = cl_helpers.DeviceBuffer.from_array(
            self._test_grid,
            self._device_interface,
            dtype=dtype,
            access_mode="read_only",
        )

        trial_grid_buffer = cl_helpers.DeviceBuffer.from_array(
            self._trial_grid,
            self._device_interface,
            dtype=dtype,
            access_mode="read_only",
        )

        input_buffer = cl_helpers.DeviceBuffer(
            shape[1],
            result_type,
            self._device_interface.context,
            access_mode="read_only",
            order="C",
        )

        sum_buffer = cl_helpers.DeviceBuffer(
            (shape[0], domain_support_size // WORKGROUP_SIZE),
            result_type,
            self._device_interface.context,
            access_mode="read_write",
            order="C",
        )

        result_buffer = cl_helpers.DeviceBuffer(
            shape[0],
            result_type,
            self._device_interface.context,
            access_mode="read_write",
            order="C",
        )

        test_normal_signs_buffer = cl_helpers.DeviceBuffer.from_array(
            self._test_normal_signs,
            self._device_interface,
            dtype=_np.int32,
            access_mode="read_only",
        )

        trial_normal_signs_buffer = cl_helpers.DeviceBuffer.from_array(
            self._trial_normal_signs,
            self._device_interface,
            dtype=_np.int32,
            access_mode="read_only",
        )

        test_indices = self._test_support_elements.astype("uint32")
        trial_indices = self._chunk.astype("uint32")

        test_indices_buffer = _cl_helpers.DeviceBuffer.from_array(
            test_indices,
            self._device_interface,
            dtype=_np.uint32,
            access_mode="read_only",
        )
        trial_indices_buffer = _cl_helpers.DeviceBuffer.from_array(
            trial_indices,
            self._device_interface,
            dtype=_np.uint32,
            access_mode="read_only",
        )

        self._buffers = [
            test_indices_buffer,
            trial_indices_buffer,
            test_normal_signs_buffer,
            trial_normal_signs_buffer,
            test_grid_buffer,
            trial_grid_buffer,
            test_connectivity,
            trial_connectivity,
            quad_points_buffer,
            quad_weights_buffer,
            input_buffer,
            _np.uint8(self._grids_disjoint),
        ]

        self._result_buffer = result_buffer
        self._input_buffer = input_buffer
        self._sum_buffer = sum_buffer

        options = self._kernel_options.options.copy()

        if "COMPLEX_KERNEL" in options:
            options["COMPLEX_RESULT"] = None

        options["NUMBER_OF_QUAD_POINTS"] = len(quad_weights)
        options["TEST"] = self._test_shape_set
        options["TRIAL"] = self._trial_shape_set

        options["NUMBER_OF_TEST_SHAPE_FUNCTIONS"] = self._number_of_test_shape_functions

        options[
            "NUMBER_OF_TRIAL_SHAPE_FUNCTIONS"
        ] = self._number_of_trial_shape_functions

        self._vec_extension, self._vec_length = kernel_helpers.get_vectorization_information(
            self._device_interface, self._precision
        )

        source_name = choose_source_name_dense_evaluator(
            self._kernel_options.compute_kernel
        )

        self._main_size, self._remainder_size = kernel_helpers.closest_multiple_to_number(
            domain_support_size, WORKGROUP_SIZE
        )

        options["WORKGROUP_SIZE"] = WORKGROUP_SIZE

        main_source = cl_helpers.kernel_source_from_identifier(
            source_name + "_regular" + self._vec_extension, options
        )

        self._main_kernel = cl_helpers.Kernel(
            main_source, self._device_interface.context, self._precision
        )

        sum_source = cl_helpers.kernel_source_from_identifier(
            "sum_for_potential_novec", options
        )

        self._sum_kernel = cl_helpers.Kernel(
            sum_source, self._device_interface.context, self._precision
        )

        if self._remainder_size > 0:
            options["WORKGROUP_SIZE"] = self._remainder_size
            remainder_source = _cl_helpers.kernel_source_from_identifier(
                source_name + "_regular_novec", options
            )
            self._remainder_kernel = _cl_helpers.Kernel(
                remainder_source, self._device_interface.context, self._precision
            )

    def compute(self, x):
        """Evaluate the product with a vector."""
        from bempp.core import kernel_helpers
        from bempp.api import log

        with self._input_buffer.host_array(self._device_interface, "write") as array:
            array[:] = x

        self._sum_buffer.set_zero(self._device_interface)

        if self._main_size > 0:

            self._main_kernel.run(
                self._device_interface,
                (len(self._test_support_elements), self._main_size // self._vec_length),
                (1, WORKGROUP_SIZE // self._vec_length),
                *self._buffers,
                self._sum_buffer,
            )

            self._sum_kernel.run(
                self._device_interface,
                (self._number_of_test_shape_functions * self._test_elements.shape[1],),
                (1,),
                self._sum_buffer,
                self._result_buffer,
                _np.uint32(len(self._chunk) // WORKGROUP_SIZE),
            )

        if self._remainder_size > 0:

            self._remainder_kernel.run(
                self._device_interface,
                (len(self._test_support_elements), self._remainder_size),
                (1, self._remainder_size),
                *self._buffers,
                self._result_buffer,
                global_offset=(0, self._main_size),
            )

    def get_result(self):
        """Return result."""

        return self._result_buffer.get_host_copy(self._device_interface)


def _prepare_evaluator(
    test_space_id,
    trial_space_id,
    chunks,
    number_of_quad_points,
    kernel_options,
    precision,
):
    """Initialize the worker."""
    from bempp.api.utils import pool
    from bempp.core import dense_evaluator

    chunk = chunks[pool.get_id()]

    evaluator = dense_evaluator.DenseEvaluatorMultiprocessingInstance(
        test_space_id,
        trial_space_id,
        chunk,
        number_of_quad_points,
        kernel_options,
        precision,
    )

    return evaluator.id


def _worker(obj_id, x):
    """Apply the operator."""
    from bempp.api.utils import pool

    evaluator = pool.get_data(obj_id)
    evaluator.compute(x)
    return evaluator.get_result()
