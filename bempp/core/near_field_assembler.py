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
        self._max_num_targets = None
        self._number_of_target_blocks = None

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

    def _prepare_buffers(self):
        """Prepare buffers."""

        source_index_ptr, source_ids, target_index_ptr, target_ids = process_near_field(
            self._fmm_interface.leaf_nodes
        )

        self._source_index_ptr = source_index_ptr
        self._source_ids = source_ids
        self._target_index_ptr = target_index_ptr
        self._target_ids = target_ids

        self._max_num_targets = _np.max(_np.diff(target_index_ptr))
        self._number_of_target_blocks = len(target_index_ptr) - 1

        source_elements = self._fmm_interface.domain.grid.elements
        target_elements = self._fmm_interface.dual_to_range.grid.elements

        device_interface = self._device_interface

        source_id_buffer = _cl_helpers.DeviceBuffer.from_array(
            source_ids, device_interface, access_mode="read_only"
        )

        target_id_buffer = _cl_helpers.DeviceBuffer.from_array(
            target_ids, device_interface, access_mode="read_only"
        )

        source_index_ptr_buffer = _cl_helpers.DeviceBuffer.from_array(
            source_index_ptr, device_interface, access_mode="read_only"
        )

        source_elements_buffer = _cl_helpers.DeviceBuffer.from_array(
            source_elements, device_interface, access_mode="read_only"
        )

        target_index_ptr_buffer = _cl_helpers.DeviceBuffer.from_array(
            target_index_ptr, device_interface, access_mode="read_only"
        )

        target_elements_buffer = _cl_helpers.DeviceBuffer.from_array(
            target_elements, device_interface, access_mode="read_only"
        )

        source_vertex_buffer = _cl_helpers.DeviceBuffer.from_array(
            self._fmm_interface.sources.astype(self._real_type),
            device_interface,
            access_mode="read_only",
        )

        target_vertex_buffer = _cl_helpers.DeviceBuffer.from_array(
            self._fmm_interface.targets.astype(self._real_type),
            device_interface,
            access_mode="read_only",
        )

        input_buffer = _cl_helpers.DeviceBuffer(
            len(self._fmm_interface.sources),
            self._result_type,
            device_interface.context,
            access_mode="read_only",
            alloc_host_memory=True,
        )

        result_buffer = _cl_helpers.DeviceBuffer(
            len(self._fmm_interface.targets),
            self._result_type,
            device_interface.context,
            access_mode="write_only",
            alloc_host_memory=True,
        )

        buffers = [
            source_id_buffer,
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

        if self._fmm_interface.domain.grid == self._fmm_interface.dual_to_range.grid:
            grids_disjoint = 0
        else:
            grids_disjoint = 1

        options = {}
        options["VEC_LENGTH"] = vec_length
        options["WORKGROUP_SIZE"] = _WORKGROUP_SIZE
        options["MAX_NUM_TARGETS"] = self._max_num_targets
        options["GRIDS_DISJOINT"] = grids_disjoint
        options["N_LOCAL_POINTS"] = self._fmm_interface.local_points.shape[1]

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

        self._input_buffer.fill_buffer(self._device_interface, vec)
        self._result_buffer.set_zero(self._device_interface)

        event = self._main_kernel.run(
            self._device_interface,
            (self._number_of_target_blocks,),
            (_WORKGROUP_SIZE,),
            *self._buffers,
            wait_for=None,
            g_times_l=True,
        )
        event.wait()

        log("Near field runtime [ms]: {0}".format(event.runtime()))

        return self._result_buffer.get_host_copy(self._device_interface).astype(
            "float64"
        )

    def as_linear_operator(self):
        """Return as linear operator."""
        from scipy.sparse.linalg import aslinearoperator

        return aslinearoperator(self)


def process_near_field(nodes):
    """Return a flat representation of the near field information."""

    # First get the length of the target and near-field arrays.

    number_of_near_field_sources = 0
    number_of_near_field_targets = 0

    source_index_ptr = [0]
    target_index_ptr = [0]

    targets = []
    sources = []

    for key in nodes:
        ntargets = len(nodes[key].target_ids)
        if ntargets == 0:
            continue
        my_number_of_sources = 0
        for colleague in nodes[key].colleagues:
            nsources = len(nodes[colleague].source_ids)
            if nsources == 0:
                continue
            my_number_of_sources += nsources
            sources.append(nodes[colleague].source_ids)
        # Do not process if the target has no sources in near field
        if my_number_of_sources == 0:
            continue
        number_of_near_field_targets += ntargets
        number_of_near_field_sources += my_number_of_sources
        target_index_ptr.append(number_of_near_field_targets)
        source_index_ptr.append(number_of_near_field_sources)
        targets.append(nodes[key].target_ids)

    return (
        _np.array(source_index_ptr, dtype=_np.int64),
        _np.concatenate(sources),
        _np.array(target_index_ptr, dtype=_np.int64),
        _np.concatenate(targets),
    )
