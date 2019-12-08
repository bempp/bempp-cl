"""Dense Assembly of integral operators."""
import numpy as _np

import bempp.core.cl_helpers as _cl_helpers
from bempp.api.assembly import assembler as _assembler
from bempp.helpers import timeit as _timeit

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

        if pool.is_initialised():
            self._ndevices = pool.number_of_workers()
        else:
            self._ndevices = 1

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

        complex_kernel = "COMPLEX_KERNEL" in operator_descriptor.options

        if complex_kernel:
            self._dtype = _np.complex128
        else:
            self._dtype = _np.float64

        if pool.is_initialised():
            self._worker_ids = pool.execute(
                _prepare_evaluator,
                self.dual_to_range.id,
                self.domain.id,
                chunks,
                self._parameters.quadrature.regular,
                operator_descriptor,
            )
        else:
            self._assembler_instance = DenseEvaluatorMultiprocessingInstance(
                    self.dual_to_range,
                    self.domain,
                    chunks[0],
                    self._parameters.quadrature.regular,
                    operator_descriptor,
                    precision,
                    device_interface=device_interface)


        return GenericDiscreteBoundaryOperator(self)

    @_timeit
    def matvec(self, x):
        """Apply operator to a vector x."""
        from bempp.api.utils import pool

        transformed_vec = self._actual_domain.map_to_full_grid @ (
            self._actual_domain.dof_transformation @ x.flat
        )

        if pool.is_initialised():
            result = sum(pool.starmap(_worker, zip(self._worker_ids, self._ndevices * [transformed_vec])))
        else:
            result = self._assembler_instance.compute(transformed_vec)


        result = self._actual_dual_to_range.dof_transformation.T @ (
            self._actual_dual_to_range.map_to_localised_space.T @ result
        )

        return result + self._singular_contribution @ x.flat


class DenseEvaluatorMultiprocessingInstance(object):
    """Evaluator Instance for MultiProcessing."""

    def __init__(
        self,
        test_space,
        trial_space,
        chunk,
        number_of_quad_points,
        kernel_options,
        precision=None,
        device_interface=None,
    ):
        """Instantiate the class."""
        import bempp.api
        from bempp.api.utils import pool
        from bempp.api.utils import helpers
        from bempp.api.space.space import return_compatible_representation

        self._id = helpers.create_unique_id()

        if device_interface is None or pool.is_worker():
            self._device_interface = bempp.api.default_device()
        else:
            self._device_interface = device_interface

        if precision is None:
            precision = self._device_interface.default_precision


        if pool.is_worker():
            test_space = pool.get_data(test_space)
            trial_space = pool.get_data(trial_space)
            pool.insert_data(self._id, self)


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
        self._grids_disjoint = actual_test_space.grid != actual_trial_space.grid



        self.compile_kernel()


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

        return self._result_buffer.get_host_copy(self._device_interface)

    # def get_result(self):
        # """Return result."""

        # return self._result_buffer.get_host_copy(self._device_interface)


def _prepare_evaluator(
    test_space_id,
    trial_space_id,
    chunks,
    number_of_quad_points,
    kernel_options,
):
    """Initialize the worker."""
    import bempp.api
    from bempp.api.utils import pool
    from bempp.core import dense_evaluator

    chunk = chunks[pool.get_id()]

    evaluator = dense_evaluator.DenseEvaluatorMultiprocessingInstance(
        test_space_id,
        trial_space_id,
        chunk,
        number_of_quad_points,
        kernel_options,
    )

    return evaluator.id


def _worker(obj_id, x):
    """Apply the operator."""
    from bempp.api.utils import pool

    evaluator = pool.get_data(obj_id)
    return evaluator.compute(x)
