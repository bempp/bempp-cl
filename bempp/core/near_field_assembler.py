"""Implementation of a near field assembler."""
import numpy as _np
from . import cl_helpers as _cl_helpers

_WORKGROUP_SIZE = 1


class NearFieldAssembler(object):
    """Initialize a near field assembler."""

    def __init__(
        self, fmm_interface, device_interface, precision, kernel_type="laplace"
    ):
        """Initialize with an FMM interface."""
        self._fmm_interface = fmm_interface

        self._kernel_type = "laplace"

        if kernel_type == "laplace":
            self._mode = "real"

        if precision == "double":
            self._real_type = _np.float64
            self._result_type = _np.float64 if self._mode == "real" else _np.complex128
        elif precision == "single":
            self._real_type = _np.float32
            self._result_type = _np.float32 if self._mode == "real" else _np.complex64
        else:
            raise ValueError("'precision' must be one of 'single' or 'double'.")

        self._device_interface = device_interface
        self._precision = precision
        self._buffers = None
        self._input_buffer = None
        self._result_buffer = None

        self._source_permutation = None
        self._target_permutation = None

        self._source_index_ptr = None
        self._target_index_ptr = None
        self._source_elements = None
        self._target_elements = None
        self._source_vertices = None
        self._target_vertices = None
        self._leaf_source_ids = None
        self._leaf_target_ids = None

        self._shape = (len(fmm_interface.targets), len(fmm_interface.sources))
        self._dtype = _np.float64

        self._main_kernel = None

        self._setup()

    @property
    def dtype(self):
        """Return dtype."""
        return self._dtype

    @property
    def shape(self):
        """Return shape."""
        return self._shape

    def _collect_targets_and_sources(self):
        """Collect arrays for targets and sources."""

        leaf_target_ids = []
        leaf_source_ids = []

        target_index_ptr = [0]
        source_index_ptr = [0]

        target_element_indices = None
        source_element_indices = None

        target_count = 0
        near_field_count = 0

        number_of_local_points = self._fmm_interface.local_points.shape[1]

        for key in self._fmm_interface.leaf_node_keys:
            target_node = self._fmm_interface.nodes[key]
            if len(target_node.target_ids) == 0:
                continue

            my_near_field = [
                source_id
                for colleague in target_node.colleagues
                if colleague != -1
                for source_id in self._fmm_interface.nodes[colleague].source_ids
            ]

            if len(my_near_field) == 0:
                continue

            leaf_target_ids.extend(list(target_node.target_ids))
            leaf_source_ids.extend(my_near_field)

            target_count += len(target_node.target_ids)
            near_field_count += len(my_near_field)

            target_index_ptr.append(target_count)
            source_index_ptr.append(near_field_count)

        target_element_indices = [
            target_id // number_of_local_points for target_id in leaf_target_ids
        ]
        source_element_indices = [
            source_id // number_of_local_points for source_id in leaf_source_ids
        ]

        target_elements = self._fmm_interface.target_grid.elements[
            :, target_element_indices
        ].ravel(order="F")
        source_elements = self._fmm_interface.source_grid.elements[
            :, source_element_indices
        ].ravel(order="F")
        target_vertices = self._fmm_interface.targets[leaf_target_ids].ravel()
        source_vertices = self._fmm_interface.sources[leaf_source_ids].ravel()

        return (
            _np.array(source_index_ptr, dtype=_np.int64),
            _np.array(target_index_ptr, dtype=_np.int64),
            _np.array(source_elements, dtype=_np.uint32),
            _np.array(target_elements, dtype=_np.uint32),
            _np.array(source_vertices, dtype=self._real_type),
            _np.array(target_vertices, dtype=self._real_type),
            leaf_source_ids,
            leaf_target_ids,
        )

    def _prepare_buffers(self):
        """Prepare buffers."""

        (
            self._source_index_ptr,
            self._target_index_ptr,
            self._source_elements,
            self._target_elements,
            self._source_vertices,
            self._target_vertices,
            self._leaf_source_ids,
            self._leaf_target_ids,
        ) = self._collect_targets_and_sources()

        device_interface = self._device_interface

        target_id_buffer = _cl_helpers.DeviceBuffer.from_array(
            _np.array(self._leaf_target_ids, dtype=_np.int64),
            device_interface,
            access_mode="read_only",
        )

        source_index_ptr_buffer = _cl_helpers.DeviceBuffer.from_array(
            self._source_index_ptr, device_interface, access_mode="read_only"
        )

        source_elements_buffer = _cl_helpers.DeviceBuffer.from_array(
            self._source_elements, device_interface, access_mode="read_only"
        )

        target_index_ptr_buffer = _cl_helpers.DeviceBuffer.from_array(
            self._target_index_ptr, device_interface, access_mode="read_only"
        )

        target_elements_buffer = _cl_helpers.DeviceBuffer.from_array(
            self._target_elements, device_interface, access_mode="read_only"
        )

        source_vertex_buffer = _cl_helpers.DeviceBuffer.from_array(
            self._source_vertices, device_interface, access_mode="read_only"
        )

        target_vertex_buffer = _cl_helpers.DeviceBuffer.from_array(
            self._target_vertices, device_interface, access_mode="read_only"
        )

        input_buffer = _cl_helpers.DeviceBuffer(
            len(self._leaf_source_ids),
            self._result_type,
            device_interface.context,
            access_mode="read_only",
            alloc_host_memory=True,
        )

        result_buffer = _cl_helpers.DeviceBuffer(
            len(self._leaf_target_ids),
            self._result_type,
            device_interface.context,
            access_mode="write_only",
            alloc_host_memory=True,
        )

        buffers = [
            target_id_buffer,
            source_index_ptr_buffer,
            target_index_ptr_buffer,
            source_elements_buffer,
            target_elements_buffer,
            source_vertex_buffer,
            target_vertex_buffer,
            input_buffer,
            result_buffer,
        ]

        self._buffers = buffers
        self._input_buffer = input_buffer
        self._result_buffer = result_buffer

    def _setup(self):
        """Setup the near field assembler."""
        from . import kernel_helpers
        from bempp.api import log

        self._prepare_buffers()

        vec_extension, vec_length = kernel_helpers.get_vectorization_information(
            self._device_interface, self._precision
        )

        options = {}
        options["VEC_LENGTH"] = 1
        options["WORKGROUP_SIZE"] = _WORKGROUP_SIZE
        options["MAX_NUM_TARGETS"] = _np.max(_np.diff(self._target_index_ptr))

        log(
            "Near field kernel vector length: {0} ({1} precision)".format(
                vec_length, self._precision
            )
        )

        main_source = _cl_helpers.kernel_source_from_identifier(
            self._kernel_type + "_near_field", options
        )

        self._main_kernel = _cl_helpers.Kernel(
            main_source, self._device_interface.context, self._precision
        )

    def matvec(self, vec):
        """Evaluate the near field for a given input vector."""
        from bempp.api import log

        number_of_target_blocks = len(self._target_index_ptr) - 1
        modified_vec = vec[self._leaf_source_ids]
        self._input_buffer.fill_buffer(self._device_interface, modified_vec)
        self._result_buffer.set_zero(self._device_interface)

        event = self._main_kernel.run(
            self._device_interface,
            (number_of_target_blocks,),
            (_WORKGROUP_SIZE,),
            *self._buffers,
            wait_for=None,
            g_times_l=True,
        )
        event.wait()

        log("Near field runtime [ms]: {0}".format(event.runtime()))

        return self._result_buffer.get_host_copy(self._device_interface)

    def as_linear_operator(self):
        """Return as linear operator."""
        from scipy.sparse.linalg import aslinearoperator

        return aslinearoperator(self)
